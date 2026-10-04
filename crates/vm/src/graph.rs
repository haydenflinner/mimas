//! A program's live values as a heap graph -- every array, dict and instance is one node with
//! a stable id, and every reference to it is an edge. [`Inspect`] is a tree (a second visit
//! collapses to [`Inspect::Cycle`]), which is right for text but loses *where* a shared or
//! cyclic reference points; a host drawing boxes and arrows needs the referent.

use std::collections::HashMap;

use shared::units::{self, Dim};

use crate::{Inspect, Val, Vm};

/// Most heap objects one graph will walk; anything further reads as [`Slot::Elided`].
pub const GRAPH_MAX_OBJS: usize = 400;
/// Most elements read per array; the middle of a longer one is counted in
/// [`Obj::Array::more`].
pub const GRAPH_MAX_ELEMS: usize = 64;
/// Of those, how many come from a long array's end -- its last cells matter as much as its
/// first (a `push`ed tail, a queue's back).
pub const GRAPH_TAIL_ELEMS: usize = 16;

/// One value position: a scalar (or opaque) leaf, or an edge to a heap object.
#[derive(Debug, Clone)]
pub enum Slot {
    Leaf(Inspect),
    /// An index into [`HeapGraph::objs`].
    Ref(u32),
    /// Past [`GRAPH_MAX_OBJS`] -- the walk stopped here.
    Elided,
}

#[derive(Debug, Clone)]
pub struct GField {
    pub name: String,
    pub slot: Slot,
    /// The field's declared unit (through `?`/`[..]`), e.g. [`units::INDEX`] for `next: index?`.
    pub dim: Option<Dim>,
}

impl GField {
    /// Declared `index`/`idx`: an int position into some array, not a quantity.
    pub fn is_index(&self) -> bool {
        self.dim == Some(units::INDEX)
    }
}

#[derive(Debug, Clone)]
pub enum Obj {
    /// `items[..gap_at]` are the first cells, then `more` unread ones, then the rest of
    /// `items` -- the array's last cells. `gap_at == items.len()` when `more == 0`.
    Array {
        items: Vec<Slot>,
        more: usize,
        gap_at: usize,
    },
    Dict(Vec<(String, Slot)>),
    Instance {
        type_name: String,
        fields: Vec<GField>,
    },
}

/// Named roots plus every heap object reachable from them, ids in first-visit order.
#[derive(Debug, Clone, Default)]
pub struct HeapGraph {
    pub roots: Vec<(String, Slot)>,
    pub objs: Vec<Obj>,
    /// The walk hit [`GRAPH_MAX_OBJS`].
    pub truncated: bool,
}

/// The name tables a graph walk labels instances with -- see [`Vm::graph_names`].
#[derive(Debug, Clone, Default)]
pub struct GraphNames {
    pub struct_names: Vec<String>,
    pub field_names: Vec<Vec<String>>,
    pub field_dims: Vec<Vec<Option<Dim>>>,
}

impl GraphNames {
    /// Walk `roots` into one graph -- shared objects across roots get one id. Must run inside
    /// the arena's `mutate` scope, like [`Val::inspect`].
    pub fn graph<'gc>(&self, roots: impl IntoIterator<Item = (String, Val<'gc>)>) -> HeapGraph {
        let mut b = Builder {
            names: self,
            ids: HashMap::new(),
            objs: Vec::new(),
            truncated: false,
        };
        let roots = roots.into_iter().map(|(n, v)| (n, b.slot(v))).collect();
        HeapGraph {
            roots,
            objs: b
                .objs
                .into_iter()
                .map(|o| o.expect("every claimed id is filled"))
                .collect(),
            truncated: b.truncated,
        }
    }
}

struct Builder<'n> {
    names: &'n GraphNames,
    ids: HashMap<*const (), u32>,
    objs: Vec<Option<Obj>>,
    truncated: bool,
}

impl Builder<'_> {
    /// `Ok(id)` for an already-seen object, `Err(Some(id))` for a freshly claimed one the
    /// caller must fill, `Err(None)` when over budget.
    fn claim(&mut self, ptr: *const ()) -> Result<u32, Option<u32>> {
        if let Some(&id) = self.ids.get(&ptr) {
            return Ok(id);
        }
        if self.objs.len() >= GRAPH_MAX_OBJS {
            self.truncated = true;
            return Err(None);
        }
        let id = self.objs.len() as u32;
        self.ids.insert(ptr, id);
        self.objs.push(None);
        Err(Some(id))
    }

    fn node<'gc>(&mut self, ptr: *const (), fill: impl FnOnce(&mut Self) -> Obj) -> Slot {
        match self.claim(ptr) {
            Ok(id) => Slot::Ref(id),
            Err(None) => Slot::Elided,
            Err(Some(id)) => {
                let obj = fill(self);
                self.objs[id as usize] = Some(obj);
                Slot::Ref(id)
            }
        }
    }

    fn array<'gc>(&mut self, vals: impl ExactSizeIterator<Item = Val<'gc>>) -> Obj {
        let n = vals.len();
        let more = n.saturating_sub(GRAPH_MAX_ELEMS);
        let gap_at = if more > 0 {
            GRAPH_MAX_ELEMS - GRAPH_TAIL_ELEMS
        } else {
            n
        };
        let items = vals
            .enumerate()
            .filter(|(i, _)| *i < gap_at || *i >= gap_at + more)
            .map(|(_, v)| self.slot(v))
            .collect();
        Obj::Array {
            items,
            more,
            gap_at,
        }
    }

    fn slot<'gc>(&mut self, v: Val<'gc>) -> Slot {
        use gc_arena::Gc;
        match v {
            Val::Array(a) => self.node(Gc::as_ptr(a.0) as *const (), |b| {
                let vals: Vec<Val<'gc>> = a.0.borrow().iter().copied().collect();
                b.array(vals.into_iter())
            }),
            Val::IntArray(a) | Val::FloatArray(a) => self.node(Gc::as_ptr(a.0) as *const (), |b| {
                let vals = a.0.borrow().to_vals();
                b.array(vals.into_iter())
            }),
            Val::Dict(d) => self.node(Gc::as_ptr(d.0) as *const (), |b| {
                let entries: Vec<(String, Val<'gc>)> =
                    d.0.borrow()
                        .iter()
                        .map(|(k, v)| (k.as_str().to_string(), *v))
                        .collect();
                Obj::Dict(entries.into_iter().map(|(k, v)| (k, b.slot(v))).collect())
            }),
            Val::Instance(inst) => self.node(Gc::as_ptr(inst.0) as *const (), |b| {
                let (sid, vals): (usize, Vec<Val<'gc>>) = {
                    let r = inst.0.borrow();
                    (r.struct_id as usize, r.fields.as_slice().to_vec())
                };
                let names = b.names;
                let type_name = names
                    .struct_names
                    .get(sid)
                    .cloned()
                    .unwrap_or_else(|| format!("@{sid}"));
                let fields = vals
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| GField {
                        name: names
                            .field_names
                            .get(sid)
                            .and_then(|n| n.get(i))
                            .cloned()
                            .unwrap_or_else(|| i.to_string()),
                        dim: names
                            .field_dims
                            .get(sid)
                            .and_then(|d| d.get(i))
                            .copied()
                            .flatten(),
                        slot: b.slot(v),
                    })
                    .collect();
                Obj::Instance { type_name, fields }
            }),
            leaf => {
                let mut seen = std::collections::HashSet::new();
                Slot::Leaf(leaf.inspect(
                    &self.names.struct_names,
                    &self.names.field_names,
                    &mut seen,
                ))
            }
        }
    }
}

impl Vm {
    /// The tables [`GraphNames::graph`] labels with -- for a host walking a value it holds
    /// inside its own `mutate` (a `call_fn_read` result, say).
    pub fn graph_names(&mut self) -> GraphNames {
        GraphNames {
            struct_names: self.struct_names(),
            field_names: self.field_names.clone(),
            field_dims: self.field_dims.clone(),
        }
    }

    /// The innermost frame's named locals, then the entry frame's globals it doesn't shadow,
    /// as one graph (aliases share ids) -- `names` filters to those bindings when given.
    /// Empty when no frame is live.
    pub fn locals_graph(&mut self, names: Option<&[&str]>) -> HeapGraph {
        let gn = self.graph_names();
        let chunks = &self.chunks;
        self.arena.mutate(|_mc, state| {
            let t = state.thread.borrow();
            let mut roots: Vec<(String, Val<'_>)> = Vec::new();
            let want = |n: &str| names.is_none_or(|ns| ns.contains(&n));
            let frames: Vec<_> = match (t.frames.first(), t.frames.last()) {
                (Some(entry), Some(top)) if entry.base != top.base => vec![top, entry],
                (Some(entry), _) => vec![entry],
                _ => Vec::new(),
            };
            for f in frames {
                let mut locals: Vec<_> = chunks[f.chunk].locals.iter().collect();
                locals.sort_by_key(|(_, reg)| reg.index());
                for (name, reg) in locals {
                    if !want(name) || roots.iter().any(|(n, _)| n == name) {
                        continue;
                    }
                    roots.push((name.clone(), t.regs[f.base + reg.index()]));
                }
            }
            gn.graph(roots)
        })
    }
}
