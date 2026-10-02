//! [`Vm::rebind`] carries a [`Snapshot`] into an *edited* program by name rather than by
//! position. A plain [`Vm::restore`] writes registers, frames and `ip` back verbatim, so it only
//! works on the exact program the snapshot came from — any recompile that shifts an instruction
//! or a register (constant folding `x * 0.0` away, a new `let`, a new field) makes it resume
//! mid-instruction. Rebinding instead starts from the edited program paused at its own first
//! yield (correct layout, fresh values) and overwrites every named local with the snapshot's
//! value of the same name, translating the heap on the way:
//!
//! * struct instances are matched by struct name and their fields by field name — a field the edit
//!   added keeps the value the edited program's first frame gave it, when the same local path has
//!   one there;
//! * function values (`Val::Fn`, closure bodies) are matched by item name;
//! * each paused call frame must be the same function paused on the same source text.
//!
//! Anything that can't be matched is an error, and the host falls back to replaying inputs.
//! Unnamed temporaries always come from the edited program's first frame — state that only lives
//! in a temporary across a yield (a `for` loop's bound, say) restarts.

use std::collections::{HashMap, HashSet};

use shared::BodyId;

use crate::{SnapNode, SnapVal, Snapshot, vm::Vm};

/// What [`Vm::rebind`] carried.
#[derive(Debug, Default, Clone)]
pub struct Rebind {
    /// Named locals whose values came from the snapshot.
    pub carried: usize,
    /// Locals (`fn::name`) and struct fields (`Struct.field`) the edit introduced — they keep the
    /// edited program's first-frame values.
    pub fresh: Vec<String>,
}

impl Vm {
    /// `(function name, source text)` of the call site a frame is paused at.
    fn paused_site(&self, chunk: u32, ip: usize) -> (String, String) {
        let body = BodyId::from(chunk);
        let c = &self.chunks[body];
        let loc = c.loc_at(u32::try_from(ip.saturating_sub(c.offset)).unwrap_or(0));
        let text = if loc.is_synthetic() {
            String::new()
        } else {
            self.source_text(loc.file_id)
                .and_then(|t| {
                    t.get(loc.span.start..loc.span.end)
                        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
                })
                .unwrap_or_default()
        };
        (self.chunk_name(body).unwrap_or("<anon>").to_string(), text)
    }

    /// Resume `snap` — taken on `old`'s program — in this Vm's program, matching state by name.
    /// This Vm must be paused at a frame boundary (the edited program run to its first yield);
    /// its state is replaced. On error nothing is written.
    pub fn rebind(&mut self, old: &mut Vm, snap: &Snapshot) -> Result<Rebind, String> {
        let fresh = self.snapshot().map_err(|e| e.to_string())?;
        let n = snap.frames.len();
        if fresh.frames.len() != n {
            return Err(format!(
                "the snapshot is paused {n} calls deep, the edited program first yields {} deep",
                fresh.frames.len()
            ));
        }
        for i in 0..n {
            let top = i + 1 == n;
            let a = old.paused_site(
                snap.frames[i].chunk,
                if top { snap.ip } else { snap.frames[i].ip },
            );
            let b = self.paused_site(
                fresh.frames[i].chunk,
                if top { fresh.ip } else { fresh.frames[i].ip },
            );
            if a != b {
                return Err(format!(
                    "the snapshot is paused at `{}` in {}, the edited program first yields at `{}` in {}",
                    a.1, a.0, b.1, b.0
                ));
            }
        }

        let old_structs = old.struct_names();
        let new_structs = self.struct_names();
        let mut struct_ids: HashMap<String, Option<u32>> = HashMap::new();
        for (id, name) in new_structs.iter().enumerate() {
            struct_ids
                .entry(name.clone())
                .and_modify(|e| *e = None)
                .or_insert(Some(id as u32));
        }
        let mut tx = Tx {
            snap: &snap.nodes,
            fresh: &fresh.nodes,
            out: fresh.nodes.clone(),
            memo: HashMap::new(),
            old_bodies: old
                .chunk_names
                .iter()
                .map(|(b, n)| (u32::from(*b), n.as_str()))
                .collect(),
            new_bodies: self
                .items
                .iter()
                .map(|(n, b)| (n.as_str(), u32::from(*b)))
                .collect(),
            old_structs: &old_structs,
            old_fields: &old.field_names,
            new_fields: &self.field_names,
            struct_ids: &struct_ids,
            seen_fresh: HashSet::new(),
            report: Rebind::default(),
        };
        let mut regs = fresh.regs.clone();
        for (of, nf) in snap.frames.iter().zip(&fresh.frames) {
            let oc = &old.chunks[BodyId::from(of.chunk)];
            let nb = BodyId::from(nf.chunk);
            let nc = &self.chunks[nb];
            let fname = self.chunk_name(nb).unwrap_or("<anon>");
            let mut names: Vec<_> = nc.locals.iter().collect();
            names.sort_by_key(|(_, r)| r.index());
            for (name, nr) in names {
                let at = nf.base + nr.index();
                match oc.locals.get(name) {
                    Some(or) => {
                        let v = snap.regs[of.base + or.index()].clone();
                        regs[at] = tx.val(&v, fresh.regs.get(at), name)?;
                        tx.report.carried += 1;
                    }
                    None => tx.report.fresh.push(format!("{fname}::{name}")),
                }
            }
        }
        let Tx { out, report, .. } = tx;
        let merged = Snapshot {
            ip: fresh.ip,
            ops_left: fresh.ops_left,
            regs,
            frames: fresh.frames.clone(),
            nodes: out,
            warned: snap.warned.clone(),
        };
        self.restore(&merged).map_err(|e| e.to_string())?;
        Ok(report)
    }
}

/// The heap translation: old snapshot nodes re-emitted into the edited program's vocabulary,
/// appended after the fresh snapshot's own nodes (which the hints and temporaries still name).
struct Tx<'a> {
    snap: &'a [SnapNode],
    fresh: &'a [SnapNode],
    out: Vec<SnapNode>,
    memo: HashMap<u32, u32>,
    old_bodies: HashMap<u32, &'a str>,
    new_bodies: HashMap<&'a str, u32>,
    old_structs: &'a [String],
    old_fields: &'a [Vec<String>],
    new_fields: &'a [Vec<String>],
    struct_ids: &'a HashMap<String, Option<u32>>,
    seen_fresh: HashSet<String>,
    report: Rebind,
}

impl<'a> Tx<'a> {
    fn val(&mut self, v: &SnapVal, hint: Option<&SnapVal>, path: &str) -> Result<SnapVal, String> {
        Ok(match v {
            SnapVal::Fn(b) => SnapVal::Fn(self.body(*b, path)?),
            SnapVal::Node(i) => SnapVal::Node(self.node(*i, hint, path)?),
            other => other.clone(),
        })
    }

    fn body(&self, b: u32, path: &str) -> Result<u32, String> {
        let name = self
            .old_bodies
            .get(&b)
            .ok_or_else(|| format!("`{path}` holds an unnamed function"))?;
        self.new_bodies
            .get(name)
            .copied()
            .ok_or_else(|| format!("`{path}` holds `{name}`, which the edit removed"))
    }

    fn node(&mut self, i: u32, hint: Option<&SnapVal>, path: &str) -> Result<u32, String> {
        if let Some(&j) = self.memo.get(&i) {
            return Ok(j);
        }
        let j = self.out.len() as u32;
        self.out.push(SnapNode::Array(Vec::new()));
        self.memo.insert(i, j);
        let (snap, fresh) = (self.snap, self.fresh);
        let hint = match hint {
            Some(SnapVal::Node(h)) => fresh.get(*h as usize),
            _ => None,
        };
        let node = match &snap[i as usize] {
            SnapNode::Array(items) => {
                let h = match hint {
                    Some(SnapNode::Array(h)) => Some(h),
                    _ => None,
                };
                let mut vals = Vec::with_capacity(items.len());
                for (k, v) in items.iter().enumerate() {
                    vals.push(self.val(v, h.and_then(|h| h.get(k)), &format!("{path}[{k}]"))?);
                }
                SnapNode::Array(vals)
            }
            // primitive stores hold no `SnapVal`s — nothing to translate, the
            // elements carry over as-is (typed tags ride with the contents)
            SnapNode::IntArray(items) => SnapNode::IntArray(items.clone()),
            SnapNode::FloatArray(items) => SnapNode::FloatArray(items.clone()),
            SnapNode::Dict(items) => {
                let h = match hint {
                    Some(SnapNode::Dict(h)) => Some(h),
                    _ => None,
                };
                let mut vals = Vec::with_capacity(items.len());
                for (k, v) in items {
                    let hv = h.and_then(|h| h.iter().find(|(hk, _)| hk == k).map(|(_, v)| v));
                    vals.push((k.clone(), self.val(v, hv, &format!("{path}[{k:?}]"))?));
                }
                SnapNode::Dict(vals)
            }
            SnapNode::Instance { struct_id, fields } => {
                let name = self
                    .old_structs
                    .get(*struct_id as usize)
                    .map(String::as_str)
                    .unwrap_or("?");
                let new_id = match self.struct_ids.get(name) {
                    Some(Some(id)) => *id,
                    Some(None) => {
                        return Err(format!(
                            "`{path}` is a `{name}`, a name the edit made ambiguous"
                        ));
                    }
                    None => return Err(format!("`{path}` is a `{name}`, which the edit removed")),
                };
                let old_names = self
                    .old_fields
                    .get(*struct_id as usize)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let new_names = self
                    .new_fields
                    .get(new_id as usize)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let h = match hint {
                    Some(SnapNode::Instance {
                        struct_id: hs,
                        fields: hf,
                    }) if *hs == new_id => Some(hf),
                    _ => None,
                };
                let mut vals = Vec::with_capacity(new_names.len().max(fields.len()));
                if old_names.len() != fields.len()
                    || new_names.len() != fields.len() && new_names.is_empty()
                {
                    // positional (unnamed) fields — carry only an unchanged shape
                    if old_names != new_names {
                        return Err(format!("`{path}`'s `{name}` changed shape"));
                    }
                    for (k, v) in fields.iter().enumerate() {
                        vals.push(self.val(v, h.and_then(|h| h.get(k)), &format!("{path}.{k}"))?);
                    }
                } else {
                    for (k, field) in new_names.iter().enumerate() {
                        let hv = h.and_then(|h| h.get(k));
                        match old_names.iter().position(|o| o == field) {
                            Some(p) => {
                                vals.push(self.val(&fields[p], hv, &format!("{path}.{field}"))?)
                            }
                            None => match hv {
                                Some(hv) => {
                                    let key = format!("{name}.{field}");
                                    if self.seen_fresh.insert(key.clone()) {
                                        self.report.fresh.push(key);
                                    }
                                    vals.push(hv.clone());
                                }
                                None => {
                                    return Err(format!(
                                        "`{path}` gained field `{field}`, and the edited program's first frame has no value there to start it from"
                                    ));
                                }
                            },
                        }
                    }
                }
                SnapNode::Instance {
                    struct_id: new_id,
                    fields: vals,
                }
            }
            SnapNode::Closure { function, captures } => {
                let function = self.body(*function, path)?;
                let mut vals = Vec::with_capacity(captures.len());
                for (k, v) in captures.iter().enumerate() {
                    vals.push(self.val(v, None, &format!("{path}.capture{k}"))?);
                }
                SnapNode::Closure {
                    function,
                    captures: vals,
                }
            }
        };
        self.out[j as usize] = node;
        Ok(j)
    }
}
