//! Link-by-hash instantiation (roadmap step 4): given a scoped address,
//! fetch its blob and every blob reachable through its `@dep:` edges,
//! then rewrite the canonical markers back into a compilable module.
//!
//! Blob layout: `S` → one canonical fn per blob; `G:i` → the `i`-th line
//! of the dependency cycle's joined group blob (`@scc:<i>` refs name
//! siblings inside it). Generated fns get content-derived names —
//! `x{hash…}` per blob, `_i` per group member — so the output reads
//! like a linked object file and can be compiled directly.

use std::collections::{HashMap, HashSet};

/// A decl in the linked module: its generated name and the address it
/// was instantiated from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedItem {
    pub name: String,
    pub hash: String,
}

/// The linked module: `source` compiles as ordinary Mimas, `items`
/// records which generated name each address got.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linked {
    pub source: String,
    pub items: Vec<LinkedItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// The fetch callback returned `None` for a blob key.
    MissingBlob(String),
    /// The root address, or a `@dep:`/`@scc:` marker inside a fetched
    /// blob, didn't parse.
    BadAddress(String),
    /// A group blob doesn't hold the member an address points at.
    MissingMember(String),
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::MissingBlob(key) => write!(f, "no blob for {}", short(key)),
            LinkError::BadAddress(addr) => write!(f, "bad address: {addr}"),
            LinkError::MissingMember(addr) => write!(f, "no member at {addr}"),
        }
    }
}

impl std::error::Error for LinkError {}

fn short(key: &str) -> &str {
    key.get(..10).unwrap_or(key)
}

/// Link `root` — a scoped address (`S` or `G:i`) — into a compilable
/// module. `fetch` maps a blob key (`S` or a bare group hash) to its
/// stored text; the store's scoped dir is the usual source.
///
/// Emission is depth-first post-order — dependencies before dependents,
/// root last — so `source` reads like a linked object file.
pub fn link(
    root: &str,
    mut fetch: impl FnMut(&str) -> Option<String>,
) -> Result<Linked, LinkError> {
    let mut linker = Linker {
        fetch: &mut fetch,
        names: HashMap::new(),
        used: HashSet::new(),
        emitted: HashSet::new(),
        out: Vec::new(),
        items: Vec::new(),
    };
    linker.visit(root)?;
    Ok(Linked {
        source: linker.out.join("\n"),
        items: linker.items,
    })
}

struct Linker<'a> {
    fetch: &'a mut dyn FnMut(&str) -> Option<String>,
    /// addr (`S` or `G:i`) → generated name.
    names: HashMap<String, String>,
    /// Generated names already spoken for — hash prefixes lengthen
    /// rather than collide.
    used: HashSet<String>,
    emitted: HashSet<String>,
    /// Rewritten decl sources, in emission order.
    out: Vec<String>,
    items: Vec<LinkedItem>,
}

impl Linker<'_> {
    fn visit(&mut self, addr: &str) -> Result<(), LinkError> {
        let (key, member) = parse_addr(addr)?;
        let addr = addr.to_string();
        if self.emitted.contains(&addr) {
            return Ok(());
        }
        let text = (self.fetch)(key)
            .ok_or_else(|| LinkError::MissingBlob(key.to_string()))?;
        match member {
            // A lone blob: one canonical fn, `@dep:`/`@self`/`@v` markers.
            None => {
                let name = self.name_for(&addr, key, None);
                let mut deps = Vec::new();
                let decl = self.rewrite(&text, &name, None, &mut deps)?;
                for dep in deps {
                    self.visit(&dep)?;
                }
                self.emit(&addr, &name, self.fill_names(decl));
            }
            // A group blob: member canonicals one per line. Every member
            // lands — a cycle links as a unit — with `@scc:i` resolving
            // to sibling member names.
            Some(i) => {
                let members: Vec<&str> = text.lines().collect();
                if i >= members.len() {
                    return Err(LinkError::MissingMember(addr));
                }
                let mut names = Vec::with_capacity(members.len());
                for (m, _) in members.iter().enumerate() {
                    names.push(self.name_for(&format!("{key}:{m}"), key, Some(m)));
                }
                let mut decls = Vec::with_capacity(members.len());
                let mut deps = Vec::new();
                for (m, canonical) in members.iter().enumerate() {
                    decls.push(self.rewrite(canonical, &names[m], Some(&names), &mut deps)?);
                }
                for dep in deps {
                    self.visit(&dep)?;
                }
                for (m, decl) in decls.into_iter().enumerate() {
                    let decl = self.fill_names(decl);
                    self.emit(&format!("{key}:{m}"), &names[m].clone(), decl);
                }
            }
        }
        Ok(())
    }

    /// `@addr:<addr>;` placeholders — written by the marker pass before a
    /// dep's generated name exists — become the visited names. `@` can't
    /// appear in real source and the `;` stops `G:1`/`G:12` prefix
    /// collisions, so every placeholder is a true site.
    fn fill_names(&self, mut decl: String) -> String {
        for (addr, name) in &self.names {
            let placeholder = format!("@addr:{addr};");
            if decl.contains(&placeholder) {
                decl = decl.replace(&placeholder, name);
            }
        }
        decl
    }

    fn emit(&mut self, addr: &str, name: &str, decl: String) {
        self.emitted.insert(addr.to_string());
        self.out.push(decl);
        self.items.push(LinkedItem {
            name: name.to_string(),
            hash: addr.to_string(),
        });
    }

    /// The generated name for an address: `x{hash-prefix}`, `_i` for a
    /// group member. Prefixes lengthen until unique — two blobs that
    /// share their first digits can't collide.
    fn name_for(&mut self, addr: &str, key: &str, member: Option<usize>) -> String {
        if let Some(name) = self.names.get(addr) {
            return name.clone();
        }
        let mut len = 8usize;
        let name = loop {
            let base = format!("x{}", &key[..len.min(key.len())]);
            let candidate = match member {
                Some(i) => format!("{base}_{i}"),
                None => base,
            };
            if self.used.insert(candidate.clone()) {
                break candidate;
            }
            len += 4;
            if len > 64 {
                break format!("{candidate}_{}", self.used.len());
            }
        };
        self.names.insert(addr.to_string(), name.clone());
        name
    }

    /// One canonical line → decl source. `self_name` is this member's
    /// generated name (for `@self`), `members` the sibling names of a
    /// group blob (for `@scc:i`), `deps` collects `@dep:` addresses in
    /// appearance order for the post-order walk.
    fn rewrite(
        &self,
        canonical: &str,
        self_name: &str,
        members: Option<&[String]>,
        deps: &mut Vec<String>,
    ) -> Result<String, LinkError> {
        rewrite(canonical, self_name, members, deps)
    }
}

/// `S` → `(S, None)`; `G:i` → `(G, Some(i))`.
fn parse_addr(addr: &str) -> Result<(&str, Option<usize>), LinkError> {
    match addr.split_once(':') {
        None => Ok((addr, None)),
        Some((key, i)) => {
            let i = i
                .parse::<usize>()
                .map_err(|_| LinkError::BadAddress(addr.to_string()))?;
            Ok((key, Some(i)))
        }
    }
}

/// Scanner state: which quoting/interpolation context we're inside.
/// Markers rewrite in `Normal` and `Interp`; plain `"…"` strings copy
/// verbatim; `f"…"` literals copy except inside `{…}` interpolations.
enum Mode {
    Normal,
    Str,
    FStr,
    Interp { depth: usize },
}

fn rewrite(
    canonical: &str,
    self_name: &str,
    members: Option<&[String]>,
    deps: &mut Vec<String>,
) -> Result<String, LinkError> {
    let bytes = canonical.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(canonical.len());
    let mut stack = vec![Mode::Normal];
    let mut after_fn = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let in_interp = matches!(stack.last(), Some(Mode::Interp { .. }));
        match stack.last().unwrap() {
            Mode::Str => {
                out.push(bytes[i]);
                i += 1;
                if c == '\\' && i < bytes.len() {
                    out.push(bytes[i]);
                    i += 1;
                } else if c == '"' {
                    stack.pop();
                }
            }
            Mode::FStr => {
                out.push(bytes[i]);
                i += 1;
                if c == '\\' && i < bytes.len() {
                    out.push(bytes[i]);
                    i += 1;
                } else if c == '"' {
                    stack.pop();
                } else if c == '{' {
                    stack.push(Mode::Interp { depth: 1 });
                }
            }
            _ => {
                if c == '{' {
                    if in_interp {
                        if let Some(Mode::Interp { depth }) = stack.last_mut() {
                            *depth += 1;
                        }
                    }
                    out.push(bytes[i]);
                    i += 1;
                    continue;
                }
                if c == '}' {
                    if in_interp {
                        let mut done = false;
                        if let Some(Mode::Interp { depth }) = stack.last_mut() {
                            *depth -= 1;
                            done = *depth == 0;
                        }
                        if done {
                            stack.pop();
                        }
                    }
                    out.push(bytes[i]);
                    i += 1;
                    continue;
                }
                if c == '"' {
                    stack.push(Mode::Str);
                    out.push(bytes[i]);
                    i += 1;
                    continue;
                }
                if c == '@' {
                    let (text, len) = marker(&canonical[i..], self_name, members, deps)?;
                    out.extend_from_slice(text.as_bytes());
                    i += len;
                    after_fn = false;
                    continue;
                }
                if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
                    let start = i;
                    while i < bytes.len()
                        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    {
                        i += 1;
                    }
                    let word = &canonical[start..i];
                    if word == "f" && bytes.get(i) == Some(&b'"') {
                        stack.push(Mode::FStr);
                        out.extend_from_slice(b"f\"");
                        i += 1;
                        after_fn = false;
                        continue;
                    }
                    if after_fn && word == "_" {
                        out.extend_from_slice(self_name.as_bytes());
                    } else {
                        out.extend_from_slice(word.as_bytes());
                    }
                    after_fn = word == "fn";
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    Ok(String::from_utf8(out).unwrap_or_else(|_| canonical.to_string()))
}

/// The marker at `text`'s start — `@v{n}`, `@self`, `@dep:<addr>`,
/// `@scc:<i>` — and its replacement. Returns the replacement and the
/// marker's byte length. A lone `@` that matches no marker is copied
/// verbatim (canonical text can't produce one — hand-edited blobs can).
fn marker(
    text: &str,
    self_name: &str,
    members: Option<&[String]>,
    deps: &mut Vec<String>,
) -> Result<(String, usize), LinkError> {
    let bytes = text.as_bytes();
    let ident_end = |start: usize| -> usize {
        start
            + bytes[start..]
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
                .count()
    };
    if text.starts_with("@self") {
        let end = 5;
        if bytes.get(end).is_none_or(|b| !b.is_ascii_alphanumeric() && *b != b'_') {
            return Ok((self_name.to_string(), end));
        }
    }
    if text.starts_with("@dep:") {
        let hash_start = 5;
        let hex = bytes[hash_start..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count();
        let mut end = hash_start + hex;
        if bytes.get(end) == Some(&b':') {
            let digits = bytes[end + 1..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
            end += 1 + digits;
        }
        let addr = text[hash_start..end].to_string();
        deps.push(addr.clone());
        return Ok((format!("@addr:{addr};"), end));
    }
    if text.starts_with("@scc:") {
        let digits = bytes[5..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        let end = 5 + digits;
        let i: usize = text[5..end]
            .parse()
            .map_err(|_| LinkError::BadAddress(text[..end].to_string()))?;
        let names = members.ok_or_else(|| LinkError::BadAddress(text[..end].into()))?;
        let name = names
            .get(i)
            .ok_or_else(|| LinkError::MissingMember(text[..end].into()))?;
        return Ok((name.clone(), end));
    }
    if text.starts_with("@v") {
        let end = ident_end(2);
        if end > 2 {
            return Ok((format!("v{}", &text[2..end]), end));
        }
    }
    // Not a marker — copy the `@` through.
    Ok(("@".to_string(), 1))
}
