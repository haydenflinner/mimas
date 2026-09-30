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

/// Link several roots into ONE module, keeping real names for the given
/// entries — a page's `(name, addr)` manifest, in declaration order.
/// Members bind as `fn name`, so a consumer can `use` the module where
/// it would splice the live page source: the entry names are the
/// include's exports. Dep edges between members rewrite to the member
/// names; deps outside the set (a member's own `use` graph, baked in
/// at publish time) land as `x{hash}` generated fns — pinned too.
///
/// Duplicate entry names emit verbatim: positional shadowing is a page
/// feature, so two `fn helper` decls keep the same shadowing the live
/// page had. Generated names skip every entry name.
///
/// The module is self-contained — it holds the members' exact pinned
/// content plus their full transitive dep closure — so storing the
/// `source` as a blob and fetching it later needs no further linking.
pub fn link_named(
    entries: &[(&str, &str)],
    mut fetch: impl FnMut(&str) -> Option<String>,
) -> Result<Linked, LinkError> {
    let mut linker = Linker {
        fetch: &mut fetch,
        names: entries
            .iter()
            .map(|(name, addr)| (addr.to_string(), name.to_string()))
            .collect(),
        used: entries.iter().map(|(name, _)| name.to_string()).collect(),
        emitted: HashSet::new(),
        out: Vec::new(),
        items: Vec::new(),
    };
    for (_, addr) in entries {
        linker.visit(addr)?;
    }
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
        let text = (self.fetch)(key).ok_or_else(|| LinkError::MissingBlob(key.to_string()))?;
        let members: Vec<&str> = text.lines().collect();
        if member.is_none() && members.len() == 1 {
            // A lone blob: one canonical fn, `@dep:`/`@self`/`@v` markers.
            let name = self.name_for(&addr, key, None);
            let mut deps = Vec::new();
            let decl = self.rewrite(&text, &name, None, &mut deps)?;
            for dep in deps {
                self.visit(&dep)?;
            }
            self.emit(&addr, &name, self.fill_names(decl));
            return Ok(());
        }
        // A group blob — addressed as `G:i`, or bare `G` (a multi-line
        // blob can't be anything else). Every member lands: a cycle
        // links as a unit, `@scc:i` resolving to sibling member names.
        let i = member.unwrap_or(0);
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
    // `fn` seen, name slot still open, and not yet consumed: the first
    // ident after the decl's `fn` becomes the generated name — `_` in
    // scoped canonicals, the written name in token-fallback blobs (an
    // unparseable decl publishes `canonical_source`, which keeps names).
    let mut after_fn = false;
    let mut renamed = false;
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
                    after_fn = false;
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
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
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
                    if after_fn && !renamed {
                        out.extend_from_slice(self_name.as_bytes());
                        renamed = true;
                    } else {
                        out.extend_from_slice(word.as_bytes());
                    }
                    after_fn = word == "fn";
                    continue;
                }
                if !c.is_ascii_whitespace() {
                    // `fn (`/`fn <` isn't a named decl — the name slot is
                    // only open to the ident immediately following `fn`.
                    after_fn = false;
                }
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    Ok(String::from_utf8(out).unwrap_or_else(|_| canonical.to_string()))
}

/// Every `@dep:` address a stored blob's markers name — the fetch keys a
/// [`link`] of it needs. String/f-string-aware like [`rewrite`], so a
/// marker-shaped string literal isn't reported; `@self`/`@scc:`/`@v`
/// markers aren't fetch keys and don't appear. The housekeeping sweep
/// walks these to find edges pointing at absent blobs.
pub fn dep_addrs(blob_text: &str) -> Vec<String> {
    let bytes = blob_text.as_bytes();
    let mut deps = Vec::new();
    let mut stack = vec![Mode::Normal];
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        let in_interp = matches!(stack.last(), Some(Mode::Interp { .. }));
        match stack.last().unwrap() {
            Mode::Str => {
                if c == '\\' && i + 1 < bytes.len() {
                    i += 1;
                } else if c == '"' {
                    stack.pop();
                }
            }
            Mode::FStr => {
                if c == '\\' && i + 1 < bytes.len() {
                    i += 1;
                } else if c == '"' {
                    stack.pop();
                } else if c == '{' {
                    stack.push(Mode::Interp { depth: 1 });
                }
            }
            _ => {
                if c == '{' && in_interp {
                    if let Some(Mode::Interp { depth }) = stack.last_mut() {
                        *depth += 1;
                    }
                } else if c == '}' && in_interp {
                    let mut done = false;
                    if let Some(Mode::Interp { depth }) = stack.last_mut() {
                        *depth -= 1;
                        done = *depth == 0;
                    }
                    if done {
                        stack.pop();
                    }
                } else if c == '"' {
                    stack.push(Mode::Str);
                } else if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
                    let start = i;
                    while i < bytes.len()
                        && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    {
                        i += 1;
                    }
                    if &blob_text[start..i] == "f" && bytes.get(i) == Some(&b'"') {
                        stack.push(Mode::FStr);
                        i += 1; // past the opening quote — FStr owns it
                    }
                    continue;
                } else if c == '@' && blob_text[i..].starts_with("@dep:") {
                    let mut end = i + 5;
                    while end < bytes.len() && bytes[end].is_ascii_hexdigit() {
                        end += 1;
                    }
                    if bytes.get(end) == Some(&b':') {
                        let digits = bytes[end + 1..]
                            .iter()
                            .take_while(|b| b.is_ascii_digit())
                            .count();
                        end += 1 + digits;
                    }
                    deps.push(blob_text[i + 5..end].to_string());
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    deps
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
        if bytes
            .get(end)
            .is_none_or(|b| !b.is_ascii_alphanumeric() && *b != b'_')
        {
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
        let digits = bytes[5..].iter().take_while(|b| b.is_ascii_digit()).count();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Globals, extract};

    /// blob key → blob text for every decl in `src` — the map a
    /// `blobs-scoped/` dir would hold after publishing the page.
    fn blobs_of(src: &str) -> (Globals, HashMap<String, String>) {
        let globals = Globals::for_source(src);
        let mut blobs = HashMap::new();
        for i in 0..globals.items().len() {
            blobs.insert(
                Globals::blob_key(globals.address(i)).to_string(),
                globals.blob_text(i),
            );
        }
        (globals, blobs)
    }

    fn addr_of(globals: &Globals, name: &str) -> String {
        let i = globals.items().iter().position(|i| i.name == name).unwrap();
        globals.address(i).to_string()
    }

    fn name_of<'a>(linked: &'a Linked, addr: &str) -> &'a str {
        linked
            .items
            .iter()
            .find(|i| i.hash == addr)
            .map(|i| i.name.as_str())
            .unwrap()
    }

    /// One fn, no deps: the blob round-trips to a renamed decl.
    #[test]
    fn a_leaf_links_to_itself() {
        let src = "fn leaf() -> int { 42 }\n";
        let (globals, blobs) = blobs_of(src);
        let addr = addr_of(&globals, "leaf");
        let linked = link(&addr, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 1);
        let name = &linked.items[0].name;
        assert!(name.starts_with('x'), "{name}");
        assert_eq!(linked.source, format!("fn {name} ( ) -> int {{ 42 }}"));
        assert_eq!(extract(&linked.source).len(), 1);
    }

    /// `twice` → `square` → leaf: deps land before dependents, call
    /// sites point at generated names, markers are gone.
    #[test]
    fn transitive_deps_link_post_order() {
        let src = "fn twice(n: int) -> int { square(n) + square(n) }\nfn square(n: int) -> int { n * n }\nfn main() -> int { twice(2) }\n";
        let (globals, blobs) = blobs_of(src);
        let root = addr_of(&globals, "main");
        let linked = link(&root, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 3);
        // Post-order: deps precede their dependents, root last.
        let order: Vec<&str> = linked.items.iter().map(|i| i.hash.as_str()).collect();
        let pos = |n: &str| {
            order
                .iter()
                .position(|a| *a == addr_of(&globals, n))
                .unwrap()
        };
        assert!(pos("square") < pos("twice"));
        assert!(pos("twice") < pos("main"));
        // No markers survive; every dep site names its target.
        assert!(!linked.source.contains('@'), "{}", linked.source);
        let twice_name = name_of(&linked, &addr_of(&globals, "twice"));
        let square_name = name_of(&linked, &addr_of(&globals, "square"));
        let twice_decl = linked
            .source
            .lines()
            .find(|l| l.contains(&format!("fn {twice_name}")))
            .unwrap();
        assert!(
            twice_decl.contains(&format!("{square_name} (")),
            "{twice_decl}"
        );
        // The linked module still parses to the same fn count.
        assert_eq!(extract(&linked.source).len(), 3);
    }

    /// `f` → `f`: the lone blob links without looping, and the self
    /// call names the generated fn.
    #[test]
    fn self_recursion_terminates() {
        let src = "fn f(n: int) -> int { if n <= 0 { 0 } else { f(n - 1) } }\n";
        let (globals, blobs) = blobs_of(src);
        let addr = addr_of(&globals, "f");
        assert!(!addr.contains(':'), "self-recursion is no group: {addr}");
        let linked = link(&addr, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 1);
        let name = &linked.items[0].name;
        assert!(
            linked.source.contains(&format!("{name} (")),
            "{}",
            linked.source
        );
        assert!(!linked.source.contains('@'));
    }

    /// `a` ⇄ `b` is one group blob: linking `G:0` lands both members,
    /// `@scc:` sites resolve to sibling names, and the group hash is
    /// the fetch key.
    #[test]
    fn a_cycle_links_as_one_blob() {
        let src = "fn a(n: int) -> int { b(n) }\nfn b(n: int) -> int { a(n) }\n";
        let (globals, blobs) = blobs_of(src);
        assert_eq!(blobs.len(), 1, "one group blob, {blobs:?}");
        let root = addr_of(&globals, "a");
        assert!(root.ends_with(":0"), "{root}");
        let linked = link(&root, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 2);
        let a = name_of(&linked, &addr_of(&globals, "a"));
        let b = name_of(&linked, &addr_of(&globals, "b"));
        assert!(
            linked.source.contains(&format!("fn {a}")),
            "{}",
            linked.source
        );
        assert!(
            linked.source.contains(&format!("fn {b}")),
            "{}",
            linked.source
        );
        assert!(!linked.source.contains('@'), "{}", linked.source);
        assert_eq!(extract(&linked.source).len(), 2);
    }

    /// A cycle's edges *out* of the group resolve too: `m` inside the
    /// cycle calls the plain `help` dep, which links alongside.
    #[test]
    fn a_cycles_external_deps_link() {
        let src = "fn help() -> int { 7 }\nfn a(n: int) -> int { b(n) + help() }\nfn b(n: int) -> int { a(n) }\n";
        let (globals, blobs) = blobs_of(src);
        let root = addr_of(&globals, "a");
        let linked = link(&root, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 3);
        let help = name_of(&linked, &addr_of(&globals, "help"));
        assert!(
            linked.source.contains(&format!("{help} (")),
            "{}",
            linked.source
        );
        assert!(!linked.source.contains('@'));
    }

    /// A fetch that comes back `None` names the missing key.
    #[test]
    fn a_missing_blob_is_an_error() {
        let src = "fn go() -> int { leaf() }\nfn leaf() -> int { 1 }\n";
        let (globals, blobs) = blobs_of(src);
        let mut blobs = blobs;
        let leaf = addr_of(&globals, "leaf");
        blobs.remove(&leaf);
        let root = addr_of(&globals, "go");
        match link(&root, |k| blobs.get(k).cloned()) {
            Err(LinkError::MissingBlob(key)) => assert_eq!(key, leaf),
            other => panic!("expected MissingBlob, got {other:?}"),
        }
    }

    /// `G:9` on a two-member group is an error, not a panic.
    #[test]
    fn a_member_past_the_end_is_an_error() {
        let src = "fn a(n: int) -> int { b(n) }\nfn b(n: int) -> int { a(n) }\n";
        let (globals, blobs) = blobs_of(src);
        let group = Globals::blob_key(&addr_of(&globals, "a")).to_string();
        match link(&format!("{group}:9"), |k| blobs.get(k).cloned()) {
            Err(LinkError::MissingMember(addr)) => assert_eq!(addr, format!("{group}:9")),
            other => panic!("expected MissingMember, got {other:?}"),
        }
    }

    /// Renaming the program's fns changes nothing about the linked
    /// shape — same addresses, same generated names, same source.
    #[test]
    fn a_rename_links_identically() {
        let (ga, ba) =
            blobs_of("fn go(x: int) -> int { leaf(x) }\nfn leaf(y: int) -> int { y + 1 }\n");
        let (gb, bb) =
            blobs_of("fn run(x: int) -> int { base(x) }\nfn base(y: int) -> int { y + 1 }\n");
        let a = link(&addr_of(&ga, "go"), |k| ba.get(k).cloned()).unwrap();
        let b = link(&addr_of(&gb, "run"), |k| bb.get(k).cloned()).unwrap();
        assert_eq!(a, b);
    }

    /// `@v` locals and f-string interpolation survive the rewrite —
    /// markers inside `f"…{…}…"` resolve, the literal bytes outside
    /// interpolation copy through.
    #[test]
    fn locals_and_interpolations_rewrite() {
        let src = "fn greet(name: str) -> str { f\"hi {name}!\" }\n";
        let (globals, blobs) = blobs_of(src);
        let addr = addr_of(&globals, "greet");
        let linked = link(&addr, |k| blobs.get(k).cloned()).unwrap();
        let name = &linked.items[0].name;
        assert_eq!(
            linked.source,
            format!("fn {name} ( v0 : str ) -> str {{ f\"hi {{v0}}!\" }}")
        );
    }

    /// A decl that can't scope-parse publishes its token canonical —
    /// which keeps the written name, not `_`. The link still renames it
    /// so call sites and decl agree.
    #[test]
    fn a_token_fallback_blob_still_links_by_address() {
        let leaf = "fn realname ( ) -> int { 7 }".to_string();
        let leaf_key = blake3::hash(leaf.as_bytes()).to_hex().to_string();
        let caller = format!("fn _ ( ) -> int {{ @dep:{leaf_key} () }}");
        let caller_key = blake3::hash(caller.as_bytes()).to_hex().to_string();
        let blobs: HashMap<String, String> =
            [(leaf_key.clone(), leaf), (caller_key.clone(), caller)].into();
        let linked = link(&caller_key, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 2);
        let leaf_name = name_of(&linked, &leaf_key);
        // The call site and the decl agree — `realname` is gone.
        assert!(
            linked.source.contains(&format!("fn {leaf_name}")),
            "{}",
            linked.source
        );
        assert!(
            linked.source.contains(&format!("{leaf_name} (")),
            "{}",
            linked.source
        );
        assert!(!linked.source.contains("realname"), "{}", linked.source);
    }

    /// A bare group hash — no `:i` — can only name a cycle's joined
    /// blob, so it links the whole cycle.
    #[test]
    fn a_bare_group_address_links_every_member() {
        let src = "fn a(n: int) -> int { b(n) }\nfn b(n: int) -> int { a(n) }\n";
        let (globals, blobs) = blobs_of(src);
        let group = Globals::blob_key(&addr_of(&globals, "a")).to_string();
        let linked = link(&group, |k| blobs.get(k).cloned()).unwrap();
        assert_eq!(linked.items.len(), 2);
        assert!(!linked.source.contains('@'), "{}", linked.source);
    }
}
