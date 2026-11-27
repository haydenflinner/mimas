#![allow(unused)] // temp

use crate::FxHashMap;

use parse::{Expr, ExprKind, Ident, NodeId};
use shared::{AdtId, Location, PactId};

use crate::components::{Ty, Vid};

#[derive(Debug, Clone)]
pub(crate) struct Ribs {
    inner: Vec<Rib>,
}

impl Default for Ribs {
    fn default() -> Self {
        // The import rib sits *below* the module rib: resolving walks innermost
        // out, so every name a file declares itself beats every name a `use`
        // pulls in -- a using-directive's precedence, but by stack order, with
        // nothing to check or repair at insert time.
        Self {
            inner: vec![
                Rib::new(RibKind::Import),
                Rib::new(RibKind::Module(AdtId::DANGLING)),
            ],
        }
    }
}

impl Ribs {
    pub(crate) fn push_module(&mut self, adt: AdtId) {
        self.inner.push(Rib::new(RibKind::Module(adt)));
    }

    /// AdtId of the innermost module rib. `AdtId::DANGLING` for the implicit root rib
    /// (no `module` decl) -- treat that as "no module" for visibility purposes.
    pub(crate) fn current_module(&self) -> AdtId {
        self.inner
            .iter()
            .rev()
            .find_map(|r| match r.kind {
                RibKind::Module(a) => Some(a),
                _ => None,
            })
            .unwrap_or(AdtId::DANGLING)
    }

    pub(crate) fn push_rib(&mut self, rib: Rib) {
        self.inner.push(rib);
    }

    pub(crate) fn push_import(&mut self) {
        self.inner.push(Rib::new(RibKind::Import));
    }

    pub(crate) fn push_block(&mut self) {
        self.inner.push(Rib::new(RibKind::Block));
    }

    /// The file's top-level statement scope: where `let` bindings declared at module level
    /// live. Unlike an ordinary block it stays visible across function boundaries -- a fn
    /// body reads the shared entry-frame slot rather than a dead local.
    pub(crate) fn push_script(&mut self) {
        self.inner.push(Rib::new(RibKind::Script));
    }

    pub(crate) fn push_function(&mut self) {
        self.inner.push(Rib::new(RibKind::Function));
    }

    pub(crate) fn push_closure(&mut self, node_id: NodeId) {
        self.inner.push(Rib::new(RibKind::Closure(node_id)));
    }

    pub(crate) fn push_impl(&mut self, adt_id: AdtId) {
        self.inner.push(Rib::new(RibKind::Impl(adt_id)));
    }

    pub(crate) fn pop(&mut self) -> Rib {
        self.inner.pop().expect("tried to pop the root module rib")
    }

    pub(crate) fn current_mut(&mut self) -> &mut Rib {
        self.inner.last_mut().unwrap()
    }

    pub(crate) fn import_mut(&mut self) -> &mut Rib {
        self.inner
            .iter_mut()
            .rev()
            .find(|rib| rib.kind == RibKind::Import)
            .expect("no import rib in scope")
    }

    /// The active module scope that holds hoisted items.
    pub(crate) fn module_mut(&mut self) -> &mut Rib {
        self.inner
            .iter_mut()
            .rev()
            .find(|rib| matches!(rib.kind, RibKind::Module(_)))
            .expect("no module rib in scope")
    }

    /// The active module rib.
    pub(crate) fn module(&self) -> &Rib {
        self.inner
            .iter()
            .rev()
            .find(|rib| matches!(rib.kind, RibKind::Module(_)))
            .expect("no module rib in scope")
    }

    /// Looks a name up in the active module scope only.
    pub(crate) fn module_lookup(&self, ident: &Ident) -> Option<DecId> {
        self.inner
            .iter()
            .rev()
            .find(|rib| matches!(rib.kind, RibKind::Module(_)))
            .and_then(|rib| rib.get(ident))
    }

    /// Resolves a name from the innermost scope outward. Locals stop being visible past the first
    /// function boundary, module items and imports never do.
    /// A top-level `let` from another region yields to an item of the use's
    /// own kit region: a kit's call of its own `fn scale` reaches the fn even
    /// when the page bound `let scale` before the kit's text (C9).
    /// r[impl mimas.region.own-first]
    fn yields_to_region(&self, rib: &Rib, ident: &Ident) -> bool {
        if !matches!(rib.kind, RibKind::Script) || !regions::any() {
            return false;
        }
        let here = regions::of(ident.location);
        if here.is_empty() {
            return false;
        }
        let Some(at) = rib.script_at.get(&ident.lexeme) else { return false };
        regions::of(*at) != here
            && self.inner.iter().any(|r| {
                matches!(r.kind, RibKind::Module(_)) && r.in_region(&ident.lexeme, &here).is_some()
            })
    }

    pub(crate) fn resolve(&self, ident: &Ident) -> Option<DecId> {
        let mut crossed_fn = false;
        for rib in self.inner.iter().rev() {
            if let Some(binding) = rib.get(ident)
                && (rib.kind.visible_across_functions() || !crossed_fn)
                && !self.yields_to_region(rib, ident)
            {
                return Some(binding);
            }
            if matches!(rib.kind, RibKind::Function) {
                crossed_fn = true;
            }
        }
        None
    }

    /// Like [`resolve`], but passes over every binding `skip` refuses: a `::`
    /// path's head looks past `let`s to modules and types.
    pub(crate) fn resolve_skipping(&self, ident: &Ident, skip: impl Fn(DecId) -> bool) -> Option<DecId> {
        let mut crossed_fn = false;
        for rib in self.inner.iter().rev() {
            if let Some(binding) = rib.get(ident)
                && (rib.kind.visible_across_functions() || !crossed_fn)
                && !skip(binding)
            {
                return Some(binding);
            }
            if matches!(rib.kind, RibKind::Function) {
                crossed_fn = true;
            }
        }
        None
    }

    /// Like [`resolve`], but also reports the closure boundaries crossed on the way out -- used by
    /// `Ident::solve` to record captures.
    pub(crate) fn resolve_with_closures(&self, ident: &Ident) -> Option<(DecId, Vec<NodeId>)> {
        let mut crossed_fn = false;
        let mut crossed_closures = Vec::new();
        for rib in self.inner.iter().rev() {
            if let Some(binding) = rib.get(ident)
                && (rib.kind.visible_across_functions() || !crossed_fn)
                && !self.yields_to_region(rib, ident)
            {
                return Some((binding, crossed_closures));
            }
            match rib.kind {
                RibKind::Function => crossed_fn = true,
                RibKind::Closure(id) => crossed_closures.push(id),
                _ => {}
            }
        }
        None
    }

    /// Looks a name up in the innermost rib only.
    pub(crate) fn resolve_current_only(&self, ident: &Ident) -> Option<DecId> {
        self.inner.last().and_then(|rib| rib.get(ident))
    }

    pub(crate) fn impl_target(&self) -> Option<AdtId> {
        self.inner.iter().rev().find_map(|v| {
            if let RibKind::Impl(adt) = v.kind {
                Some(adt)
            } else {
                None
            }
        })
    }

    pub(crate) fn pop_module(&mut self) -> Rib {
        let module = self.pop();
        assert!(matches!(module.kind, RibKind::Module(_)));
        module
    }

    pub(crate) fn pop_import(&mut self) -> Rib {
        let imports = self.pop();
        assert_eq!(imports.kind, RibKind::Import);
        imports
    }

    pub(crate) fn pop_block(&mut self) -> Rib {
        let block = self.pop();
        assert_eq!(block.kind, RibKind::Block);
        block
    }

    pub(crate) fn pop_script(&mut self) -> Rib {
        let script = self.pop();
        assert_eq!(script.kind, RibKind::Script);
        script
    }

    /// The rib declarations land in right now.
    pub(crate) fn current_kind(&self) -> RibKind {
        self.inner.last().map_or(RibKind::Block, |rib| rib.kind)
    }
}

/// The `//# page: NAME` regions of a spliced source (C2): each region is a
/// namespace. Set once per solve from the asts' page marks; a file with no
/// marks is one region, the page's own (`""`).
pub(crate) mod regions {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    thread_local! {
        static REGIONS: RefCell<FxHashMap<shared::FileId, Rc<Vec<(usize, String)>>>> =
            RefCell::default();
        /// Bare names two regions define, met at a use: (name, use, regions).
        static AMBIGUOUS: RefCell<Vec<(String, Location, Vec<String>)>> = RefCell::default();
    }

    /// The region of a declaration outside the program's own files: std
    /// and the prelude. It ranks after every page region.
    pub(crate) const OUTSIDE: &str = "\u{0}";

    thread_local! {
        static NAMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Installs the regions of the program's files.
    pub(crate) fn install(files: Vec<(shared::FileId, Vec<(usize, String)>)>) {
        let named = files.iter().any(|(_, m)| m.iter().any(|(_, n)| !n.is_empty()));
        NAMED.with(|n| n.set(named));
        REGIONS.with(|r| {
            let mut r = r.borrow_mut();
            r.clear();
            for (file, marks) in files {
                r.insert(file, Rc::new(marks));
            }
        });
        AMBIGUOUS.with(|a| a.borrow_mut().clear());
    }

    /// The region a location falls in: the last mark at or before it, or
    /// [`OUTSIDE`] for a file that is not the program's.
    pub(crate) fn of(at: Location) -> String {
        // a native or other synthetic decl has no place in any file
        if at.span().is_synthetic() {
            return OUTSIDE.to_string();
        }
        REGIONS.with(|r| {
            match r.borrow().get(&at.file_id()) {
                Some(marks) => {
                    let start = at.span().start();
                    marks.iter().rev().find(|(s, _)| *s <= start).map(|(_, n)| n.clone()).unwrap_or_default()
                }
                None => OUTSIDE.to_string(),
            }
        })
    }

    /// Whether any file of this solve has a named region.
    pub(crate) fn any() -> bool {
        NAMED.with(|n| n.get())
    }

    /// How a region is spelled before `::`: `man-kit` is `man_kit`.
    pub(crate) fn spelled(region: &str) -> String {
        region.replace('-', "_")
    }

    pub(crate) fn note_ambiguous(name: &str, at: Location, regions: Vec<String>) {
        AMBIGUOUS.with(|a| {
            let mut a = a.borrow_mut();
            if !a.iter().any(|(n, l, _)| n == name && *l == at) {
                a.push((name.to_string(), at, regions));
            }
        });
    }

    pub(crate) fn take_ambiguous() -> Vec<(String, Location, Vec<String>)> {
        AMBIGUOUS.with(|a| std::mem::take(&mut *a.borrow_mut()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Rib {
    pub(crate) table: FxHashMap<String, DecId>,
    /// Every declaration a module-level name has had, with where its name appears in the
    /// source. A name declared more than once (two scripts each defining `struct Shape`)
    /// resolves, at each use, to the most recent declaration that precedes it -- reading
    /// order, so an include placed after your own struct wins from there on. Across the
    /// `//# page:` regions of a spliced source the region rule picks first (`regions`).
    history: FxHashMap<String, Vec<(Location, DecId)>>,
    /// A script rib's bindings: where the latest of each name was declared.
    script_at: FxHashMap<String, Location>,
    pub(crate) kind: RibKind,
}

impl Rib {
    pub(crate) fn new(kind: RibKind) -> Self {
        Self {
            table: FxHashMap::default(),
            history: FxHashMap::default(),
            script_at: FxHashMap::default(),
            kind,
        }
    }

    pub(crate) fn insert(&mut self, ident: Ident, dec_id: DecId) {
        if matches!(self.kind, RibKind::Module(_)) {
            self.history
                .entry(ident.lexeme.clone())
                .or_default()
                .push((ident.location, dec_id));
        }
        if matches!(self.kind, RibKind::Script) {
            self.script_at.insert(ident.lexeme.clone(), ident.location);
        }
        self.table.insert(ident.lexeme, dec_id);
    }

    /// The latest declaration at or before `at` among `decls`, else the latest.
    /// The declaration a use at `at` reads, by reading order. A byte offset
    /// orders positions within one source only: a page's offset 1800 says
    /// nothing about a library file's offset 1713. So only declarations in
    /// the use's own file compete by position: the latest at or before the
    /// use, else the first after it (an item is hoisted). With none in that
    /// file, the latest declared (the file solved last) answers.
    /// r[impl mimas.decl.source-order]
    fn reading_order(decls: &[&(Location, DecId)], at: Location) -> Option<DecId> {
        let pos = at.span.start;
        let same: Vec<&(Location, DecId)> = decls
            .iter()
            .copied()
            .filter(|(loc, _)| loc.file_id() == at.file_id())
            .collect();
        if same.is_empty() {
            return decls.last().map(|(_, id)| *id);
        }
        let mut before: Option<(usize, DecId)> = None;
        let mut after: Option<(usize, DecId)> = None;
        for &&(loc, id) in &same {
            let start = loc.span.start;
            if start <= pos {
                if before.is_none_or(|(b, _)| start >= b) {
                    before = Some((start, id));
                }
            } else if after.is_none_or(|(a, _)| start < a) {
                after = Some((start, id));
            }
        }
        before.or(after).map(|(_, id)| id)
    }

    /// A declaration of `name` in the region `region` of this module rib.
    pub(crate) fn in_region(&self, name: &str, region: &str) -> Option<(Location, DecId)> {
        self.history
            .get(name)?
            .iter()
            .rev()
            .find(|(loc, _)| regions::of(*loc) == region)
            .copied()
    }

    fn get(&self, ident: &Ident) -> Option<DecId> {
        // `kit_a::rect`: a region-qualified name reaches that region's item
        // r[impl mimas.region.qualified]
        if matches!(self.kind, RibKind::Module(_))
            && let Some((region, name)) = ident.lexeme.rsplit_once("::")
            && let Some(decls) = self.history.get(name)
        {
            let picked: Vec<_> = decls
                .iter()
                .filter(|(loc, _)| regions::spelled(&regions::of(*loc)) == region)
                .collect();
            return Self::reading_order(&picked, ident.location);
        }
        // a redeclared module-level name: the latest declaration at or before this use
        if let Some(decls) = self.history.get(&ident.lexeme)
            && decls.len() > 1
        {
            let at = ident.location.span.start;
            if regions::any() {
                // r[impl mimas.region.bare]
                let mut named: Vec<String> = Vec::new();
                for (loc, _) in decls {
                    let r = regions::of(*loc);
                    if !named.contains(&r) {
                        named.push(r);
                    }
                }
                if named.len() > 1 {
                    let here = regions::of(ident.location);
                    let pick = |r: &str| -> Vec<&(Location, DecId)> {
                        decls.iter().filter(|(loc, _)| regions::of(*loc) == r).collect()
                    };
                    // the use's own region, then the page's own text
                    for r in [here.as_str(), ""] {
                        let own = pick(r);
                        if !own.is_empty() {
                            return Self::reading_order(&own, ident.location);
                        }
                    }
                    // then the one kit that declares it, then std
                    let kits: Vec<String> = named
                        .iter()
                        .filter(|r| !r.is_empty() && r.as_str() != regions::OUTSIDE)
                        .cloned()
                        .collect();
                    match kits.len() {
                        1 => return Self::reading_order(&pick(&kits[0]), ident.location),
                        0 => return Self::reading_order(&pick(regions::OUTSIDE), ident.location),
                        // two kits define it and neither is the use's: ambiguous
                        _ => regions::note_ambiguous(&ident.lexeme, ident.location, kits),
                    }
                }
            }
            let all: Vec<_> = decls.iter().collect();
            return Self::reading_order(&all, ident.location);
        }
        self.table.get(&ident.lexeme).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RibKind {
    /// The root scope. Holds hoisted items; stays visible across function boundaries.
    /// Carries the file's module `AdtId`, or `AdtId::DANGLING` for the implicit/no-file root
    /// rib -- used to attribute decs to their owning module for visibility checks.
    Module(AdtId),
    /// A file/module-level import scope. Imports are visible across function boundaries, but are
    /// not exported when a module rib is closed into an ADT. Sits below its file's module rib so
    /// the file's own declarations always outrank what a `use` pulls in.
    Import,
    /// An ordinary lexical scope: a block, loop, match arm, or closure body.
    Block,
    /// A file's top-level statement scope -- holds its `let` bindings. Visible across
    /// function boundaries (the bindings are backed by the entry frame's registers, so a fn
    /// reads/writes the same storage). Nested blocks inside it stay ordinary `Block`s.
    Script,
    /// A function body. Names bound outside it cannot be captured from within.
    Function,
    /// A closure body. Same visibility as a block, but identified so resolutions that cross it
    /// can be recorded as captures of the closure expression with this NodeId.
    Closure(NodeId),
    /// An impl body. Defines the current Adt that Self refers to
    Impl(AdtId),
}

impl RibKind {
    fn visible_across_functions(self) -> bool {
        matches!(self, Self::Module(_) | Self::Import | Self::Script)
    }
}

shared::id!(pub DecId);

#[derive(Debug, Clone)]
pub struct Dec {
    pub name: String,
    pub location: Location,
    pub kind: DecKind,
    pub vid: Vid,
    pub vis: Vis,
    pub module: AdtId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecKind {
    Local,
    /// A `let` bound in the file's top-level statement scope. Runtime-mutable like a local,
    /// but stored in the entry frame so function bodies can reach it. Never captured by
    /// closures -- they address it directly, the same way fns do.
    Global,
    Item {
        defaults: Vec<Option<parse::Literal>>,
        /// `where` predicates from the fn decl, in source order. Stored unevaluated --
        /// call sites re-derive provability against their own arguments.
        wheres: Vec<parse::Expr>,
    },
    Constant(Option<parse::Literal>),
    Variant {
        parent: AdtId,
        layout: AdtId,
    },
    Adt(AdtId),
    Pact(PactId),
    LoopVar,
}

impl DecKind {
    pub fn is_constant(&self) -> bool {
        matches!(self, DecKind::Constant(_))
    }

    pub fn const_value(&self) -> Option<&parse::Literal> {
        match self {
            DecKind::Constant(Some(lit)) => Some(lit),
            _ => None,
        }
    }

    pub fn as_adt(&self) -> Option<AdtId> {
        match self {
            DecKind::Adt(a) => Some(*a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Vis {
    #[default]
    Private,
    Public,
}
