//! Atoms: `#EmptyList`, a name that needs no declaration.
//! r[impl mimas.atom]
//!
//! Every bare atom is a variant of one built-in enum, `atom`. A variant is
//! minted the first time the program writes the name, so `#EmptyList`
//! displays, compares and matches like an enum value without an `enum`
//! anywhere. The variant's layout adt is named `#EmptyList`, which is what
//! the VM prints. Where a declared enum is expected, the same literal is that
//! enum's variant instead (`#Wed` for `Day::Wed`): the checker reads the
//! slot's type, the way a record literal does (r[impl mimas.atom.enum]).
//!
//! Nothing declares which atoms a value can hold, so the checker infers it
//! after the bodies are solved ([`check`]): the atoms a `T!` function can
//! raise, the atoms a `let` or a function return can hold, the atoms an
//! `absolve` handler's payload can be. A set is *closed* when every way into
//! it is known, and *open* (`None`) when anything is unknown: a parameter, a
//! field, a native, a raised string. A closed set checks a match two ways:
//! an arm outside the set is an error, and a missing arm is an error. An open
//! set never makes either error; it asks only for the `_` arm any
//! unenumerable scrutinee asks for (r[impl mimas.atom.set]).

use std::collections::BTreeSet;

use crate::{FxHashMap, FxHashSet, FxIndexMap};
use itertools::Itertools;
use parse::{
    Ast, AtomPayload, Expr, ExprKind, FieldKey, Function, Item, ItemKind, NodeId, Stmt,
    StmtKind, Visitor,
    components::{Pat, PatKind},
};
use shared::{Located, Location};

use crate::{
    Result, Solver,
    components::{
        Adt, AdtFlags, AdtId, DecId, DecKind, Field, Fields, StructVariant, Ty, TyExt,
        TupleVariant, Variant, Vis,
    },
    errors::{
        AtomNotInEnum, AtomOutsideSet, AtomPayloadMismatch, NonExhaustiveMatch, UnknownAtom,
    },
    traits::{Query, Solve},
};

/// What an atom's payload looks like: nothing, `n` positional values, or
/// named fields. One atom name has one shape in a program
/// (r[impl mimas.atom.payload]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Shape {
    Unit,
    Tuple(usize),
    Struct(Vec<String>),
}

impl Shape {
    pub(crate) fn of(payload: &AtomPayload) -> Shape {
        match payload {
            AtomPayload::None => Shape::Unit,
            AtomPayload::Positional(values) => Shape::Tuple(values.len()),
            AtomPayload::Named(fields) => {
                Shape::Struct(fields.iter().map(|(k, _)| k.lexeme.clone()).collect())
            }
        }
    }

    fn spell(&self, name: &str) -> String {
        match self {
            Shape::Unit => format!("#{name}"),
            Shape::Tuple(n) => format!("#{name}({})", vec!["_"; *n].join(", ")),
            Shape::Struct(keys) => format!(
                "#{name}({})",
                keys.iter().map(|k| format!("{k}: _")).join(", ")
            ),
        }
    }
}

/// The checker's atom bookkeeping, one per solver.
#[derive(Clone, Default)]
pub(crate) struct AtomState {
    /// The built-in `atom` enum, minted on first use.
    pub adt: Option<AdtId>,
    /// Each atom literal's layout: the variant emit builds an instance of.
    pub layouts: FxIndexMap<NodeId, AdtId>,
    /// Each variant's shape, the site that set it, and whether a bare literal
    /// or an arm has relied on it yet. An unrelied shape came from the
    /// pre-scan alone, and a bare literal may still reshape it: the literal
    /// the pre-scan saw first may have been one that fit a declared enum.
    shapes: FxHashMap<String, (Shape, Location, bool)>,
    /// Names a bare literal makes, with the first site.
    pub made: FxIndexMap<String, Location>,
    /// Atom arms read against the `atom` enum, with their sites.
    pub arms: Vec<(String, Location)>,
    /// Matches whose exhaustiveness waits for the inferred sets.
    pub matches: FxHashSet<NodeId>,
}

impl Solver {
    /// The built-in `atom` enum (r[impl mimas.atom.type]).
    pub(crate) fn atom_adt(&mut self) -> AdtId {
        if let Some(aid) = self.atoms.adt {
            return aid;
        }
        // not `IS_BUILTIN`: that flag marks a namespace (`int::MAX`), and an
        // atom is a value
        let aid = self.push_adt(Adt::new_enum("atom".into(), FxIndexMap::default()));
        self.atoms.adt = Some(aid);
        aid
    }

    pub(crate) fn is_atom_adt(&self, aid: AdtId) -> bool {
        self.atoms.adt == Some(aid)
    }

    /// The layout of the `atom` variant `name`, minted or reshaped as needed.
    /// `shape` is what this site claims (an arm claims none); `rely` marks a
    /// site that depends on the shape from now on.
    pub(crate) fn atom_variant(
        &mut self,
        name: &str,
        shape: Option<&Shape>,
        at: Location,
        rely: bool,
    ) -> Result<AdtId> {
        let aid = self.atom_adt();
        let existing = self.atoms.shapes.get(name).cloned();
        match (existing, shape) {
            (None, claimed) => {
                let shape = claimed.cloned().unwrap_or(Shape::Unit);
                self.install_atom(name, &shape, at);
                self.atoms
                    .shapes
                    .insert(name.into(), (shape, at, rely && claimed.is_some()));
            }
            (Some((old, first, relied)), Some(claimed)) if &old != claimed => {
                if relied {
                    Err(AtomPayloadMismatch {
                        src: self.src(at),
                        at: at.into(),
                        atom: format!("#{name}"),
                        why: format!(
                            "made as `{}` at line {}; one atom has one payload shape",
                            old.spell(name),
                            self.line_of(first)
                        ),
                    })?
                }
                self.install_atom(name, claimed, at);
                self.atoms
                    .shapes
                    .insert(name.into(), (claimed.clone(), at, rely));
            }
            (Some((old, first, relied)), claimed) => {
                let rely = relied || (rely && claimed.is_some());
                self.atoms.shapes.insert(name.into(), (old, first, rely));
            }
        }
        Ok(self.adts[aid].variants[name]
            .layout()
            .expect("an atom variant always has a layout"))
    }

    /// 1-based line of a location, for messages that name a second site.
    fn line_of(&self, at: Location) -> usize {
        let src = self.src(at);
        let text = src.inner();
        text[..at.span.start().min(text.len())]
            .bytes()
            .filter(|b| *b == b'\n')
            .count()
            + 1
    }

    fn install_atom(&mut self, name: &str, shape: &Shape, at: Location) {
        let aid = self.atom_adt();
        let mut variant = match shape {
            Shape::Unit => Variant::Tuple(TupleVariant {
                location: Some(at),
                ..Default::default()
            }),
            Shape::Tuple(n) => Variant::Tuple(TupleVariant {
                members: (0..*n).map(|_| self.vid().into()).collect(),
                location: Some(at),
                ..Default::default()
            }),
            Shape::Struct(keys) => Variant::Struct(StructVariant {
                fields: Fields(
                    keys.iter()
                        .map(|key| {
                            let ty: Ty = self.vid().into();
                            let ident = parse::Ident::new(key.clone(), at);
                            let dec =
                                self.dec_id(&ident, ty.clone(), DecKind::Local, Vis::Public);
                            (key.clone(), Field::new(ty, at, dec))
                        })
                        .collect(),
                ),
                location: Some(at),
                ..Default::default()
            }),
        };
        // the layout is named `#Name`: the VM prints that, and one layout
        // per name makes instance equality compare the name, then the
        // payload (r[impl mimas.atom.display] r[impl mimas.atom.equality])
        let qualified = format!("#{name}");
        let mut layout_adt = Adt::new_variant_layout(qualified.clone(), &variant);
        layout_adt.enum_of = Some(aid);
        let (layout, dec) = match self.adts[aid].variants.get(name) {
            Some(old) => {
                let layout = old.layout().expect("an atom variant always has a layout");
                let dec = old.dec().expect("an atom variant always has a dec");
                self.adts[layout] = layout_adt;
                (layout, dec)
            }
            None => {
                let layout = self.push_adt(layout_adt);
                let ident = parse::Ident::new(qualified, at);
                let dec = self.dec_id(
                    &ident,
                    Ty::adt(aid),
                    DecKind::Variant { parent: aid, layout },
                    Vis::Public,
                );
                (layout, dec)
            }
        };
        variant.set_layout(layout);
        variant.set_dec(dec);
        self.adts[aid].variants.insert(name.into(), variant);
    }

    /// Mint every atom variant a literal in `asts` writes, before any body is
    /// solved: an arm read before the literal that makes its atom still
    /// finds the variant's shape.
    pub(crate) fn prescan_atoms(&mut self, asts: &[&Ast]) -> Result<()> {
        #[derive(Default)]
        struct Scan {
            heads: FxHashSet<NodeId>,
            found: Vec<(String, Shape, Location)>,
        }
        impl Visitor for Scan {
            fn pat(&mut self, pat: &Pat) {
                if let PatKind::Variant(p) | PatKind::TupleVariant(p, _) | PatKind::Struct(p, _) =
                    pat.kind()
                {
                    self.heads.insert(p.id());
                }
            }
            fn expr(&mut self, expr: &Expr) {
                if let ExprKind::Atom(atom) = expr.kind()
                    && !self.heads.contains(&expr.id())
                {
                    self.found.push((
                        atom.name.lexeme.clone(),
                        Shape::of(&atom.payload),
                        expr.location(),
                    ));
                }
            }
        }
        let mut scan = Scan::default();
        for ast in asts {
            parse::walk_stmts(ast.stmts(), &mut scan);
        }
        for (name, shape, at) in scan.found {
            if !self.atoms.shapes.contains_key(&name) {
                self.atom_variant(&name, Some(&shape), at, false)?;
            }
        }
        Ok(())
    }

    /// The type an arm's `#Name` head reads against `scrut`: the declared
    /// enum's variant when the scrutinee is one, else the `atom` variant.
    pub(crate) fn atom_arm(
        &mut self,
        name: &parse::Ident,
        scrut: &Ty,
        dot: bool,
    ) -> Result<(AdtId, String)> {
        let lexeme = name.lexeme.clone();
        if let Ty::Adt(eid, _) | Ty::Identity(eid, _) = scrut.clone().normalized(self)
            && !self.is_atom_adt(eid)
            && self.adts[eid].flags.contains(AdtFlags::IS_ENUM)
        {
            let Some(variant) = self.adts[eid].variants.get(&lexeme).cloned() else {
                Err(self.not_in_enum(&lexeme, eid, name.location, dot))?
            };
            if let Some(dec) = variant.dec() {
                self.node_decs.insert(name.id, dec);
            }
            return Ok((eid, lexeme));
        }
        if dot {
            return Err(self.dot_needs_enum(&lexeme, name.location).into());
        }
        self.atom_variant(&lexeme, None, name.location, true)?;
        self.atoms.arms.push((lexeme.clone(), name.location));
        Ok((self.atom_adt(), lexeme))
    }

    fn not_in_enum(&self, name: &str, eid: AdtId, at: Location, dot: bool) -> AtomNotInEnum {
        let sigil = if dot { "." } else { "#" };
        let names: Vec<String> = self.adts[eid].variants.keys().cloned().collect();
        AtomNotInEnum {
            src: self.src(at),
            at: at.into(),
            atom: format!("{sigil}{name}"),
            enum_name: self.adts[eid].name.clone(),
            variants: names.iter().map(|n| format!("`{sigil}{n}`")).join(", "),
            help: nearest(name, names.iter().map(String::as_str))
                .map(|n| format!("did you mean `{sigil}{n}`?")),
        }
    }

    /// `.Down` with no enum expected: name the declared enums that have a
    /// variant `Down`, so the fix is one of their full names
    /// (r[impl mimas.enum.dot.no-slot]).
    pub(crate) fn dot_needs_enum(&self, name: &str, at: Location) -> crate::errors::DotVariantNeedsEnum {
        let mut owners: Vec<String> = self
            .adts
            .iter()
            .filter(|(aid, adt)| {
                adt.flags.contains(AdtFlags::IS_ENUM)
                    && !self.is_atom_adt(*aid)
                    && adt.enum_of.is_none()
                    && adt.variants.contains_key(name)
            })
            .map(|(_, adt)| adt.name.clone())
            .collect();
        owners.sort();
        owners.dedup();
        let (label, help) = match owners.as_slice() {
            [] => (
                "no enum here, and no declared enum has this variant".to_string(),
                format!("write the enum's name, `E::{name}`, or `#{name}` for an atom"),
            ),
            [one] => (
                format!("no enum is expected here; `{one}` has `{name}`"),
                format!("write `{one}::{name}`, or give the place a type: `let x: {one} = .{name}`"),
            ),
            many => (
                format!(
                    "no enum is expected here; {} have `{name}`",
                    many.iter().map(|e| format!("`{e}`")).join(", ")
                ),
                format!("write one of {}", many.iter().map(|e| format!("`{e}::{name}`")).join(", ")),
            ),
        };
        crate::errors::DotVariantNeedsEnum {
            src: self.src(at),
            at: at.into(),
            name: name.to_string(),
            label,
            help,
        }
    }
}

// r[impl mimas.atom.literal]
impl Solve for parse::Atom {
    fn solve(&self, id: NodeId, location: Location, solver: &mut Solver) -> Result<Ty> {
        let name = self.name.lexeme.clone();
        let hint = solver
            .record_hints
            .get(&id)
            .cloned()
            .map(|t| t.normalized(solver));
        let hint = match hint {
            Some(Ty::Option(inner)) => Some(*inner),
            other => other,
        };
        // r[impl mimas.atom.enum]
        if let Some(Ty::Adt(eid, _) | Ty::Identity(eid, _)) = hint
            && !solver.is_atom_adt(eid)
            && solver.adts[eid].flags.contains(AdtFlags::IS_ENUM)
        {
            return fit_enum(self, id, location, eid, solver);
        }
        // `.Down` is never a free atom (r[impl mimas.enum.dot.no-slot])
        if self.dot {
            return Err(solver.dot_needs_enum(&name, location).into());
        }

        let shape = Shape::of(&self.payload);
        let layout = solver.atom_variant(&name, Some(&shape), location, true)?;
        solver.atoms.made.entry(name.clone()).or_insert(location);
        let aid = solver.atom_adt();
        let variant = solver.adts[aid].variants[&name].clone();
        match (&self.payload, &variant) {
            (AtomPayload::None, _) => {}
            (AtomPayload::Positional(values), Variant::Tuple(tv)) => {
                for (value, member) in values.iter().zip(&tv.members) {
                    value.fulfill_ty(member.clone(), solver)?;
                }
            }
            (AtomPayload::Named(fields), Variant::Struct(sv)) => {
                for (key, value) in fields {
                    let ty = sv.fields[&key.lexeme].ty.clone();
                    value.fulfill_ty(ty, solver)?;
                }
            }
            _ => unreachable!("atom_variant checked the shape"),
        }
        solver.atoms.layouts.insert(id, layout);
        Ok(Ty::adt(aid))
    }
}

/// `#Wed` in a slot that expects `Day`: `Day::Wed`, its payload checked like
/// the variant's constructor.
fn fit_enum(
    atom: &parse::Atom,
    id: NodeId,
    location: Location,
    eid: AdtId,
    solver: &mut Solver,
) -> Result<Ty> {
    let name = atom.name.lexeme.clone();
    let Some(variant) = solver.adts[eid].variants.get(&name).cloned() else {
        Err(solver.not_in_enum(&name, eid, location, atom.dot))?
    };
    if let Some(dec) = variant.dec() {
        solver.node_decs.insert(atom.name.id, dec);
    }
    let layout = variant.layout().unwrap_or(eid);
    let mismatch = |solver: &Solver| AtomPayloadMismatch {
        src: solver.src(location),
        at: location.into(),
        atom: format!("#{name}"),
        why: format!(
            "`{}` is built as `{}`",
            solver.adts[eid].name,
            variant.ctor(&format!("#{name}"))
        ),
    };
    let ty = match (&atom.payload, &variant) {
        (AtomPayload::None, v) if v.is_empty() => solver.instantiated_adt_at(eid, location),
        (AtomPayload::Positional(values), Variant::Tuple(tv))
            if values.len() == tv.members.len() =>
        {
            let ty = solver.instantiated_adt_at(eid, location);
            let args = match &ty {
                Ty::Adt(_, args) => args.clone(),
                _ => vec![],
            };
            for (value, member) in values.iter().zip(&tv.members) {
                let member = solver.specialize(member, eid, &args);
                value.fulfill_ty(member, solver)?;
            }
            ty
        }
        (AtomPayload::Named(fields), Variant::Struct(sv)) => {
            let keyed: Vec<(FieldKey, &Expr)> = fields
                .iter()
                .map(|(k, v)| (FieldKey::Ident(k.clone()), v))
                .collect();
            crate::traits::struct_fields(solver, eid, sv, layout, &keyed, None, location)?
        }
        _ => Err(mismatch(solver))?,
    };
    solver.atoms.layouts.insert(id, layout);
    Ok(ty.normalized(solver))
}

/// The closest of `names` to `name`, when one is close enough to be a typo.
fn nearest<'a>(name: &str, names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    fn distance(a: &str, b: &str) -> usize {
        let b: Vec<char> = b.chars().collect();
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut prev = row[0];
            row[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                let cur = row[j + 1];
                row[j + 1] = (prev + usize::from(ca != *cb))
                    .min(row[j] + 1)
                    .min(row[j + 1] + 1);
                prev = cur;
            }
        }
        row[b.len()]
    }
    let limit = (name.chars().count() / 3).max(1);
    names
        .filter(|n| *n != name)
        .map(|n| (distance(&name.to_lowercase(), &n.to_lowercase()), n))
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, n)| n)
}

type Set = BTreeSet<String>;

fn spell(set: &Set) -> String {
    set.iter().map(|n| format!("`#{n}`")).join(", ")
}

fn union(a: Option<Set>, b: Option<Set>) -> Option<Set> {
    let (mut a, b) = (a?, b?);
    a.extend(b);
    Some(a)
}

/// Where the inferred sets come from: the program's fns, `let`s and
/// `absolve` handlers, indexed once.
struct Sets<'a> {
    solver: &'a Solver,
    fns: FxHashMap<DecId, &'a Function>,
    lets: FxHashMap<DecId, &'a Expr>,
    assigned: FxHashSet<DecId>,
    handlers: FxHashMap<DecId, &'a Expr>,
    matches: Vec<&'a Expr>,
    returns: FxHashMap<DecId, Option<Set>>,
    raises: FxHashMap<DecId, Option<Set>>,
    busy: FxHashSet<DecId>,
}

/// Every node of one kind under an expression, closures excluded: what a
/// function body does itself, not what a closure it builds does later.
fn own_nodes<'a>(body: &'a Expr, want: fn(&ExprKind) -> bool) -> Vec<&'a Expr> {
    // `Visitor` hands out short borrows, so collect ids and re-find by walk
    struct Ids {
        want: fn(&ExprKind) -> bool,
        hits: Vec<NodeId>,
        closed: FxHashSet<NodeId>,
    }
    impl Visitor for Ids {
        fn expr(&mut self, e: &Expr) {
            if (self.want)(e.kind()) {
                self.hits.push(e.id());
            }
            if let ExprKind::Closure(c) = e.kind() {
                struct Inner<'s>(&'s mut FxHashSet<NodeId>);
                impl Visitor for Inner<'_> {
                    fn expr(&mut self, e: &Expr) {
                        self.0.insert(e.id());
                    }
                }
                parse::walk_expr(&c.body, &mut Inner(&mut self.closed));
            }
        }
    }
    let mut ids = Ids {
        want,
        hits: vec![],
        closed: FxHashSet::default(),
    };
    parse::walk_expr(body, &mut ids);
    let keep: FxHashSet<NodeId> = ids
        .hits
        .into_iter()
        .filter(|id| !ids.closed.contains(id))
        .collect();
    let mut out = vec![];
    find_all(body, &keep, &mut out);
    out
}

/// The exprs under `root` whose ids are in `keep`, with `root`'s lifetime.
fn find_all<'a>(root: &'a Expr, keep: &FxHashSet<NodeId>, out: &mut Vec<&'a Expr>) {
    if keep.contains(&root.id()) {
        out.push(root);
    }
    for child in children(root) {
        find_all(child, keep, out);
    }
}

/// The direct sub-expressions of `e` that a value or a raise can come
/// through, statements included.
fn children(e: &Expr) -> Vec<&Expr> {
    fn stmt_exprs<'s>(s: &'s Stmt, out: &mut Vec<&'s Expr>) {
        match s.kind() {
            StmtKind::Let(l) => {
                out.push(&l.right);
                if let Some(e) = &l.else_branch {
                    out.push(e);
                }
            }
            StmtKind::Assignment(a) => {
                out.push(&a.left);
                out.push(&a.right);
            }
            StmtKind::Expr(e) => out.push(e),
            _ => {}
        }
    }
    let mut out = vec![];
    match e.kind() {
        ExprKind::Absolve(a) => out.extend([&a.left, &a.handler]),
        ExprKind::Atom(a) => out.extend(a.payload.values()),
        ExprKind::Block(b) => {
            for s in &b.body {
                stmt_exprs(s, &mut out);
            }
            out.extend(b.yielded_expr.as_ref());
        }
        ExprKind::Call(c) => {
            out.push(&c.left);
            out.extend(c.arguments.iter().map(|a| &a.value));
        }
        ExprKind::Closure(c) => out.push(&c.body),
        ExprKind::Grouping(g) => out.push(&g.inner),
        ExprKind::If(i) => {
            out.extend([&i.condition, &i.main_body]);
            out.extend(i.else_expr.as_ref());
        }
        ExprKind::Match(m) => {
            out.push(&m.identity);
            for case in &m.cases {
                out.extend(case.guard());
                out.push(case.body());
            }
        }
        ExprKind::Raise(r) => out.push(&r.value),
        ExprKind::Return(r) => out.extend(r.value.as_ref()),
        ExprKind::Propagate(p) => out.push(&p.expr),
        ExprKind::Unwrap(u) => out.push(&u.expr),
        ExprKind::Coalescence(c) => out.extend([&c.left, &c.right]),
        ExprKind::Loop(l) => out.push(&l.body),
        ExprKind::While(w) => out.extend([&w.header, &w.body]),
        ExprKind::For(f) => out.extend([&f.iterator, &f.body]),
        ExprKind::Evaluation(v) => out.extend([&v.left, &v.right]),
        ExprKind::Equality(v) => out.extend([&v.left, &v.right]),
        ExprKind::Logical(v) => out.extend([&v.left, &v.right]),
        ExprKind::Unary(u) => out.push(&u.right),
        ExprKind::Break(b) => out.extend(b.value.as_ref()),
        ExprKind::Collect(c) => out.push(&c.value),
        ExprKind::In(i) => out.extend([&i.left, &i.right]),
        ExprKind::Range(r) => out.extend([&r.start, &r.end]),
        ExprKind::FString(f) => out.extend(f.parts.iter().filter_map(|p| match p {
            parse::FStringPart::Expr(e) => Some(e),
            _ => None,
        })),
        ExprKind::Access(a) => match a {
            parse::Access::Dot { left, .. } | parse::Access::DoubleColon { left, .. } => {
                out.push(left)
            }
            parse::Access::Square { left, key, .. } => out.extend([left, key]),
            parse::Access::Identity { .. } => {}
        },
        ExprKind::Literal(l) => match l {
            parse::Literal::Array(es) | parse::Literal::Tuple(es) => out.extend(es),
            parse::Literal::Record(fields) => out.extend(fields.iter().map(|(_, v)| v)),
            parse::Literal::Struct(s) => {
                out.extend(s.fields.iter().map(|(_, v)| v));
                out.extend(s.base.as_ref());
            }
            _ => {}
        },
        ExprKind::Ident(_) | ExprKind::Continue(_) | ExprKind::Poison(_) => {}
    }
    out
}

impl<'a> Sets<'a> {
    fn new(solver: &'a Solver, asts: &[&'a Ast]) -> Self {
        let mut by_site: FxHashMap<(usize, usize), DecId> = FxHashMap::default();
        for (id, dec) in solver.decs.iter() {
            if matches!(dec.kind, DecKind::Item { .. }) {
                by_site
                    .entry((dec.location.file_id, dec.location.span.start))
                    .or_insert(id);
            }
        }
        let mut sets = Sets {
            solver,
            fns: FxHashMap::default(),
            lets: FxHashMap::default(),
            assigned: FxHashSet::default(),
            handlers: FxHashMap::default(),
            matches: vec![],
            returns: FxHashMap::default(),
            raises: FxHashMap::default(),
            busy: FxHashSet::default(),
        };
        for ast in asts {
            for stmt in ast.stmts() {
                sets.index_stmt(stmt, &by_site);
            }
        }
        sets
    }

    fn index_item(&mut self, item: &'a Item, by_site: &FxHashMap<(usize, usize), DecId>) {
        match item.kind() {
            ItemKind::Function(f) => {
                if let Some(&dec) =
                    by_site.get(&(f.name.location.file_id, f.name.location.span.start))
                {
                    self.fns.insert(dec, f);
                }
                self.index_expr(&f.body, by_site);
            }
            ItemKind::Impl(imp) => {
                for inner in &imp.items {
                    self.index_item(inner, by_site);
                }
            }
            ItemKind::Const(c) => self.index_expr(&c.right, by_site),
            ItemKind::Tests(tests) => {
                for case in &tests.cases {
                    self.index_expr(&case.expr, by_site);
                }
            }
            _ => {}
        }
    }

    fn index_stmt(&mut self, stmt: &'a Stmt, by_site: &FxHashMap<(usize, usize), DecId>) {
        match stmt.kind() {
            StmtKind::Let(l) => {
                if let PatKind::Ident(ident) = l.left.kind()
                    && l.else_branch.is_none()
                    && let Some(&dec) = self.solver.node_decs.get(&ident.id)
                {
                    self.lets.insert(dec, &l.right);
                }
                self.index_expr(&l.right, by_site);
                if let Some(e) = &l.else_branch {
                    self.index_expr(e, by_site);
                }
            }
            StmtKind::Assignment(a) => {
                if let ExprKind::Ident(ident) = a.left.kind()
                    && let Some(&dec) = self.solver.node_decs.get(&ident.id)
                {
                    self.assigned.insert(dec);
                }
                self.index_expr(&a.right, by_site);
            }
            StmtKind::Expr(e) => self.index_expr(e, by_site),
            StmtKind::Item(item) => self.index_item(item, by_site),
            _ => {}
        }
    }

    fn index_expr(&mut self, e: &'a Expr, by_site: &FxHashMap<(usize, usize), DecId>) {
        let e = self.solver.rewrites.get(&e.id()).unwrap_or(e);
        match e.kind() {
            ExprKind::Absolve(a) => {
                if let ExprKind::Closure(c) = a.handler.kind()
                    && let Some(param) = c.parameters.first()
                    && let PatKind::Ident(ident) = param.left.kind()
                    && let Some(&dec) = self.solver.node_decs.get(&ident.id)
                {
                    self.handlers.insert(dec, &a.left);
                }
            }
            ExprKind::Match(_) if self.solver.atoms.matches.contains(&e.id()) => {
                self.matches.push(e);
            }
            ExprKind::Block(b) => {
                for s in &b.body {
                    self.index_stmt(s, by_site);
                }
                if let Some(y) = &b.yielded_expr {
                    self.index_expr(y, by_site);
                }
                return;
            }
            _ => {}
        }
        for child in children(e) {
            self.index_expr(child, by_site);
        }
    }

    fn ty(&self, e: &Expr) -> Option<Ty> {
        let vid = self.solver.node_to_vid.get(&e.id())?;
        Some(Ty::Vid(*vid).normalized(self.solver))
    }

    fn diverges(&self, e: &Expr) -> bool {
        matches!(self.ty(e), Some(Ty::Never))
            || matches!(e.kind(), ExprKind::Raise(_) | ExprKind::Return(_))
    }

    fn callee(&self, callee: &Expr) -> Option<DecId> {
        match callee.kind() {
            ExprKind::Ident(i) => self.solver.node_decs.get(&i.id).copied(),
            _ => None,
        }
    }

    /// The atoms a value can be: `None` when anything about it is unknown.
    fn value(&mut self, e: &'a Expr) -> Option<Set> {
        let e = self.solver.rewrites.get(&e.id()).unwrap_or(e);
        if self.diverges(e) {
            return Some(Set::new());
        }
        match e.kind() {
            ExprKind::Atom(atom) => {
                let layout = self.solver.atoms.layouts.get(&e.id())?;
                let parent = self.solver.adts[*layout].enum_of?;
                self.solver
                    .is_atom_adt(parent)
                    .then(|| Set::from([atom.name.lexeme.clone()]))
            }
            ExprKind::Grouping(g) => self.value(&g.inner),
            ExprKind::Block(b) => self.value(b.yielded_expr.as_ref()?),
            ExprKind::If(i) => {
                let main = self.value(&i.main_body);
                union(main, self.value(i.else_expr.as_ref()?))
            }
            ExprKind::Match(m) => m
                .cases
                .iter()
                .map(|c| self.value(c.body()))
                .try_fold(Set::new(), |acc, s| union(Some(acc), s)),
            ExprKind::Ident(ident) => {
                let dec = *self.solver.node_decs.get(&ident.id)?;
                if let Some(left) = self.handlers.get(&dec).copied() {
                    return self.raised(left);
                }
                if self.assigned.contains(&dec) {
                    return None;
                }
                let init = self.lets.get(&dec).copied()?;
                self.value(init)
            }
            ExprKind::Call(c) => {
                let dec = self.callee(&c.left)?;
                self.returns_of(dec)
            }
            ExprKind::Unwrap(u) => self.value(&u.expr),
            ExprKind::Propagate(p) => self.value(&p.expr),
            ExprKind::Absolve(a) => {
                let left = self.value(&a.left);
                let ExprKind::Closure(c) = a.handler.kind() else {
                    return None;
                };
                union(left, self.value(&c.body))
            }
            _ => None,
        }
    }

    /// The atoms a `T!` value can raise.
    fn raised(&mut self, e: &'a Expr) -> Option<Set> {
        let e = self.solver.rewrites.get(&e.id()).unwrap_or(e);
        // a diverging or a plain value raises nothing
        if self.diverges(e) || !matches!(self.ty(e), Some(Ty::Result(_))) {
            return Some(Set::new());
        }
        match e.kind() {
            ExprKind::Call(c) => {
                let dec = self.callee(&c.left)?;
                self.raises_of(dec)
            }
            ExprKind::Grouping(g) => self.raised(&g.inner),
            ExprKind::Block(b) => self.raised(b.yielded_expr.as_ref()?),
            ExprKind::If(i) => {
                let main = self.raised(&i.main_body);
                union(main, self.raised(i.else_expr.as_ref()?))
            }
            ExprKind::Match(m) => m
                .cases
                .iter()
                .map(|c| self.raised(c.body()))
                .try_fold(Set::new(), |acc, s| union(Some(acc), s)),
            _ => None,
        }
    }

    fn returns_of(&mut self, dec: DecId) -> Option<Set> {
        if let Some(memo) = self.returns.get(&dec) {
            return memo.clone();
        }
        let f = *self.fns.get(&dec)?;
        if !self.busy.insert(dec) {
            return None;
        }
        let mut set = self.value(&f.body);
        for ret in own_nodes(&f.body, |k| matches!(k, ExprKind::Return(_))) {
            let ExprKind::Return(r) = ret.kind() else { unreachable!() };
            set = union(set, r.value.as_ref().and_then(|v| self.value(v)));
        }
        self.busy.remove(&dec);
        self.returns.insert(dec, set.clone());
        set
    }

    /// The atoms a function can raise: its `raise`s, the raises its `?`s pass
    /// on, and the `T!` values it hands back.
    // r[impl mimas.atom.raise]
    fn raises_of(&mut self, dec: DecId) -> Option<Set> {
        if let Some(memo) = self.raises.get(&dec) {
            return memo.clone();
        }
        let f = *self.fns.get(&dec)?;
        if !self.busy.insert(dec) {
            return None;
        }
        let mut set = Some(Set::new());
        let exits = own_nodes(&f.body, |k| {
            matches!(
                k,
                ExprKind::Raise(_) | ExprKind::Propagate(_) | ExprKind::Return(_)
            )
        });
        for exit in exits {
            let more = match exit.kind() {
                ExprKind::Raise(r) => {
                    let atomish = matches!(
                        self.ty(&r.value),
                        Some(Ty::Adt(a, _)) if self.solver.is_atom_adt(a)
                    ) || matches!(r.value.kind(), ExprKind::Ident(i)
                        if self.solver.node_decs.get(&i.id).is_some_and(|d| self.handlers.contains_key(d)));
                    if atomish { self.value(&r.value) } else { None }
                }
                ExprKind::Propagate(p) => match self.solver.propagates.get(&exit.id()) {
                    Some(crate::Propagation::Reraise) => self.raised(&p.expr),
                    Some(crate::Propagation::RaiseNull(_)) => None,
                    _ => Some(Set::new()),
                },
                ExprKind::Return(r) => match &r.value {
                    Some(v) => self.raised(v),
                    None => Some(Set::new()),
                },
                _ => unreachable!(),
            };
            set = union(set, more);
        }
        let tail = match f.body.kind() {
            ExprKind::Block(b) => b.yielded_expr.as_ref(),
            _ => Some(&f.body),
        };
        if let Some(tail) = tail {
            set = union(set, self.raised(tail));
        }
        self.busy.remove(&dec);
        self.raises.insert(dec, set.clone());
        set
    }
}

/// The atom names an arm's pattern reads, flattening or-patterns, and
/// whether the arm covers each whole (its payload patterns only bind).
fn arm_atoms(pat: &Pat, out: &mut Vec<(String, Location, bool)>) {
    let binds = |p: &Pat| matches!(p.kind(), PatKind::Ident(_));
    match pat.kind() {
        PatKind::Or(alts) => alts.iter().for_each(|a| arm_atoms(a, out)),
        PatKind::Variant(p) | PatKind::TupleVariant(p, _) | PatKind::Struct(p, _) => {
            let ExprKind::Atom(atom) = p.kind() else { return };
            let whole = match pat.kind() {
                PatKind::TupleVariant(_, subs) => subs.iter().all(binds),
                PatKind::Struct(_, fields) => fields.values().all(binds),
                _ => true,
            };
            out.push((atom.name.lexeme.clone(), p.location(), whole));
        }
        _ => {}
    }
}

/// The checks that wait for every body: an atom arm names an atom some code
/// makes, and a match over a closed set names only atoms in it and misses
/// none (r[impl mimas.atom.set] r[impl mimas.atom.match.unknown]
/// r[impl mimas.atom.match.exhaustive]).
pub(crate) fn check(solver: &mut Solver, asts: &[&Ast]) -> Result<()> {
    let Some(aid) = solver.atoms.adt else {
        return Ok(());
    };
    // a payload member no literal ever pinned is `any`
    let layouts: Vec<AdtId> = solver.adts[aid]
        .variants
        .values()
        .filter_map(Variant::layout)
        .collect();
    for id in std::iter::once(aid).chain(layouts) {
        let mut adt = solver.adts[id].clone();
        adt.normalize_members(solver);
        solver.adts[id] = adt;
    }

    for (name, at) in solver.atoms.arms.clone() {
        if !solver.atoms.made.contains_key(&name) {
            let made: Vec<&str> = solver.atoms.made.keys().map(String::as_str).collect();
            Err(UnknownAtom {
                src: solver.src(at),
                at: at.into(),
                atom: format!("#{name}"),
                help: nearest(&name, made.iter().copied())
                    .map(|n| format!("did you mean `#{n}`?")),
            })?
        }
    }

    let mut sets = Sets::new(solver, asts);
    for m in sets.matches.clone() {
        let ExprKind::Match(mm) = m.kind() else { continue };
        let set = sets.value(&mm.identity);
        let mut named = vec![];
        let mut wild = false;
        for case in &mm.cases {
            if case.guard().is_none() && matches!(case.pat().kind(), PatKind::Ident(_)) {
                wild = true;
            }
            let mut atoms = vec![];
            arm_atoms(case.pat(), &mut atoms);
            if case.guard().is_some() {
                atoms.iter_mut().for_each(|a| a.2 = false);
            }
            named.extend(atoms);
        }
        match &set {
            Some(set) => {
                for (name, at, _) in &named {
                    if !set.contains(name) {
                        Err(AtomOutsideSet {
                            src: solver.src(*at),
                            at: (*at).into(),
                            atom: format!("#{name}"),
                            set: spell(set),
                            help: nearest(name, set.iter().map(String::as_str))
                                .map(|n| format!("did you mean `#{n}`?")),
                        })?
                    }
                }
                let covered: Set = named
                    .iter()
                    .filter(|(_, _, whole)| *whole)
                    .map(|(n, _, _)| n.clone())
                    .collect();
                let missing: Set = set.difference(&covered).cloned().collect();
                if !wild && !mm.panic_terminator && !missing.is_empty() {
                    Err(NonExhaustiveMatch {
                        src: solver.src(m.location()),
                        at: m.location().into(),
                        summary: format!(
                            "missing {} {}",
                            if missing.len() == 1 { "atom" } else { "atoms" },
                            spell(&missing)
                        ),
                        help: Some(format!(
                            "this value is one of {}: add the missing arms, or an `_` arm",
                            spell(set)
                        )),
                        variant_defs: vec![],
                    })?
                }
            }
            None => {
                if !wild && !mm.panic_terminator {
                    Err(NonExhaustiveMatch {
                        src: solver.src(m.location()),
                        at: m.location().into(),
                        summary: "missing pattern `_`".into(),
                        help: Some(
                            "this value can be an atom no arm names: add an `_` arm, or a `!` panic terminator"
                                .into(),
                        ),
                        variant_defs: vec![],
                    })?
                }
            }
        }
    }
    Ok(())
}
