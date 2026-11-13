//! Dimensional analysis: `25kW * 4h` is energy, `25kW + 4h` is a mistake.
//!
//! Quantities are plain floats kept in base units, so this pass is a checker and nothing else --
//! it runs after the program has type checked, reads no types of its own, and changes nothing the
//! compiler emits. It follows each number's [`Dim`] through the program: a unit literal (`25kW`)
//! starts one, `+ - *` and `/` carry it along, and unit annotations (`fn cost(e: kWh) -> usd`,
//! `struct S { p: usd/kWh }`) pin it down across function and field boundaries.
//!
//! The rules, deliberately small:
//!
//! * A plain `float` (or `int`) is *dimensionless*. It scales anything (`2.0 * 5kW`), but is not
//!   itself a power: `let p: float = 5kW` and `takes_plain(5kW)` are errors.
//! * `+`, `-`, `%` and comparisons need both sides in the same dimension. Comparing against a
//!   literal `0` is always fine.
//! * `*` and `/` combine dimensions; `.sqrt()` halves them, `.pow(n)` scales them.
//! * Anything the checker can't follow (a native's result, an untyped closure, a container it
//!   doesn't track) is *unknown*, and unknown never causes an error. So code that doesn't use
//!   units is untouched, and units can be adopted one function at a time.

use crate::FxHashMap;

use crate::FxIndexMap;
use crate::components::TyExt;
use itertools::Itertools;
use miette::NamedSource;
use parse::NodeId;
use parse::{
    Access, Ast, Expr, ExprKind, FStringPart, FieldKey, Function, Ident, Item, ItemKind, Literal,
    Stmt, StmtKind, Struct,
    components::{Annotation, Binding, Pat, PatKind},
    expr::{EqualityOp, EvaluationOp, UnaryOp},
    lex::TyKw,
    stmt::AssignmentOp,
};
use shared::{Located, Location, Ty, units, units::Dim};

use crate::{Result, Solver, components::DecId, errors::DimensionMismatch};

/// What the checker knows about a value.
#[derive(Debug, Clone, PartialEq)]
enum D {
    /// Can't tell (or isn't a number): never an error.
    Any,
    /// A number in this dimension; [`Dim::NONE`] is a plain number.
    Q(Dim),
    /// A list whose items share one dimension.
    Arr(Box<D>),
    /// An `Interval` of numbers in this dimension: `lo`, `mid` and `hi` are all quantities of it,
    /// and arithmetic on it keeps the unit like arithmetic on a plain quantity does.
    Iv(Dim),
}

const PLAIN: D = D::Q(Dim::NONE);

impl D {
    /// The dimension of a quantity or an interval of them.
    fn dim(&self) -> Option<Dim> {
        match self {
            D::Q(d) | D::Iv(d) => Some(*d),
            _ => None,
        }
    }
}

/// An `index` and a plain number interconvert freely -- `i += 1`, `i <
/// xs.len()`, `let i: index = 0` are the whole point of writing indexes
/// down. What the dimension is for is keeping *quantities* out: `i + 3px`
/// still clashes.
fn index_plain(x: Dim, y: Dim) -> bool {
    (x == units::INDEX && y.is_none()) || (y == units::INDEX && x.is_none())
}

/// A `rad` and a plain number interconvert freely -- radians are
/// dimensionless and trig has always read a plain number as radians, so
/// `theta + p.atan2(q)` needs no `.to("rad")` unwrap. What the dimension
/// keeps out is other quantities: `theta + 3px` still clashes.
fn angle_plain(x: Dim, y: Dim) -> bool {
    (x == units::ANGLE && y.is_none()) || (y == units::ANGLE && x.is_none())
}

/// Two known dimensions that differ; `None` when either is unknown or they agree.
fn clash(a: &D, b: &D) -> Option<(Dim, Dim)> {
    match (a, b) {
        (D::Q(x) | D::Iv(x), D::Q(y) | D::Iv(y))
            if !Dim::agree(*x, *y)
                && !index_plain(x.bare(), y.bare())
                && !angle_plain(x.bare(), y.bare())
            =>
        {
            Some((*x, *y))
        }
        (D::Arr(x), D::Arr(y)) => clash(x, y),
        _ => None,
    }
}

/// The dimension two agreeing values share (unknown wins, so a doubt propagates as a doubt).
fn join(a: D, b: &D) -> D {
    match (&a, b) {
        // an untagged side adopts the other's axis: `b.x + BALL` stays along x
        (D::Q(x), D::Q(y)) if Dim::agree(*x, *y) => D::Q(Dim::absorb(*x, *y)),
        (D::Iv(x), D::Iv(y)) if Dim::agree(*x, *y) => D::Iv(Dim::absorb(*x, *y)),
        // `index` absorbs a plain operand, so `if .. { i } else { 0 }` stays an index
        (D::Q(x), D::Q(y)) if index_plain(x.bare(), y.bare()) => {
            D::Q(if x.bare().is_none() { *y } else { *x })
        }
        // `rad` dissolves into a plain operand instead -- a mixed sum is a
        // plain radian count, so `theta = theta + atan2(..)` doesn't turn a
        // plain accumulator into a quantity the next `theta * v` rejects.
        (D::Q(x), D::Q(y)) if angle_plain(x.bare(), y.bare()) => PLAIN,
        (D::Arr(x), D::Arr(y)) => D::Arr(Box::new(join((**x).clone(), y))),
        _ => D::Any,
    }
}

struct Pass<'a> {
    solver: &'a Solver,
    ast: &'a Ast,
    /// Struct declarations by name (top level; the first of a name wins).
    structs: FxHashMap<String, &'a Struct>,
    /// Free functions by name, and methods / associated functions by (type, name).
    fns: FxHashMap<String, &'a Function>,
    methods: FxHashMap<(String, String), &'a Function>,
    /// A function's declaration site, so a call resolved by the solver finds its signature.
    fns_by_site: FxHashMap<(usize, usize), &'a Function>,
    /// Declared enum names -- needed for `unit_of` to tell `enum A` from the ampere.
    enum_names: std::collections::HashSet<String>,
    /// Generic type-param names in scope (`fn f<T>`, `struct P<A, B>` fields read at a literal,
    /// an impl's target params). They name types, not units, so `unit_of` must not see them.
    ty_params: Vec<Vec<String>>,
    env: FxHashMap<DecId, D>,
    consts: FxHashMap<DecId, D>,
    /// The declared return of each function being walked.
    rets: Vec<D>,
    /// Every expr's computed dimension — surfaces to `Resolutions::node_dims`
    /// so tools can name a node's unit (`b.x + BALL` is `px`).
    dims: FxIndexMap<NodeId, Dim>,
    /// The dimension the surrounding context demanded of an expr (a call-arg
    /// slot's declared param dim) — surfaces to `Resolutions::want_dims`. An
    /// expr whose own dim is `Any` (`b.x` on a dimless `Vec2` field) can still
    /// carry the unit its slot required.
    want_dims: FxIndexMap<NodeId, Dim>,
    /// `print`/`dbg`/`warn` call site → the dim of the reported value — surfaces to
    /// `Resolutions::print_dims` so the program can ship it to the natives.
    print_dims: FxIndexMap<Location, Dim>,
    /// Each top-level echo site's computed dim (a quantity, or a list of them) —
    /// resolved to [`units::Shown`] once the walk has seen every written unit.
    echo_ds: FxIndexMap<Location, D>,
    /// Each value a failing check's message interpolates, with its
    /// dimension, so the message can say the unit (r[impl check.report.units]).
    check_ds: FxIndexMap<parse::NodeId, D>,
    /// A number written with its unit in a check (`47lb`), as written, so
    /// the message doesn't show it after a round trip through base units.
    check_texts: FxIndexMap<parse::NodeId, String>,
    /// `dec` quantities, scaled when the pass ends (r[impl mimas.dec.units]).
    dec_notes: FxIndexMap<parse::NodeId, DecNote>,
    /// The first unit the program writes for each dimension (`12lb`, a bare `lb`
    /// constant): an echo shows a quantity of that dimension in it.
    written: FxIndexMap<Dim, (String, f64)>,
    /// Unit-constant declarations (`lb` read as `1lb`) by name, so a bare unit in
    /// an expression counts as writing it.
    unit_decs: FxHashMap<DecId, String>,
    /// One slot per enclosing loop: the joined dim of the values its `collect`s
    /// hand over, so `for i in .. collect qty[i] * price[i]` is a list of usd.
    collects: Vec<Option<D>>,
    report: bool,
    errors: Vec<miette::Report>,
}

pub(crate) fn check(solver: &mut Solver, asts: &[&Ast]) -> Result<()> {
    for ast in asts {
        let mut pass = Pass {
            solver: &*solver,
            ast,
            structs: FxHashMap::default(),
            fns: FxHashMap::default(),
            methods: FxHashMap::default(),
            fns_by_site: FxHashMap::default(),
            enum_names: Default::default(),
            ty_params: Vec::new(),
            env: FxHashMap::default(),
            // unit constants (`kWh` reads as `1kWh`) carry their dimension — the
            // solver minted their decs without a `const` statement, so `bind_const`
            // never saw them; seed them here once.
            consts: solver
                .unit_consts
                .values()
                .map(|(dec, dim)| (*dec, D::Q(*dim)))
                .chain(
                    solver
                        .const_dims
                        .iter()
                        .map(|(dec, dim)| (*dec, D::Q(*dim))),
                )
                .collect(),
            rets: Vec::new(),
            dims: FxIndexMap::default(),
            want_dims: FxIndexMap::default(),
            print_dims: FxIndexMap::default(),
            echo_ds: FxIndexMap::default(),
            check_ds: FxIndexMap::default(),
            check_texts: FxIndexMap::default(),
            dec_notes: FxIndexMap::default(),
            written: FxIndexMap::default(),
            unit_decs: solver
                .unit_consts
                .iter()
                .map(|(name, (dec, _))| (*dec, name.clone()))
                .collect(),
            collects: Vec::new(),
            report: false,
            errors: Vec::new(),
        };
        pass.index(ast.stmts());
        let member_units = pass.member_units();
        // two quiet rounds settle constants that build on each other in any order, then the
        // real one reports
        for _ in 0..2 {
            pass.run(ast.stmts());
        }
        pass.report = true;
        pass.env.clear();
        pass.run(ast.stmts());
        match pass.errors.len() {
            0 => {}
            1 => return Err(pass.errors.pop().unwrap()),
            // one report per problem is the useful thing; miette's `{:?}` keeps each
            // diagnostic's snippet rendering intact inside the joined message
            _ => {
                let joined = pass.errors.iter().map(|e| format!("{e:?}")).join("\n\n");
                return Err(miette::Report::msg(joined));
            }
        }
        let dims = std::mem::take(&mut pass.dims);
        let want_dims = std::mem::take(&mut pass.want_dims);
        let print_dims = std::mem::take(&mut pass.print_dims);
        // a print site's unit as the page writes it, for a `dec` quantity,
        // which is held in that unit (r[impl mimas.dec.units])
        let print_shown: Vec<(Location, units::Shown)> = print_dims
            .iter()
            .filter_map(|(loc, d)| pass.shown(&D::Q(*d)).map(|s| (*loc, s)))
            .collect();
        let echo_units: Vec<(Location, units::Shown)> = std::mem::take(&mut pass.echo_ds)
            .into_iter()
            .filter_map(|(loc, d)| pass.shown(&d).map(|s| (loc, s)))
            .collect();
        let check_texts = std::mem::take(&mut pass.check_texts);
        let scale_of = |pass: &Pass, d: Dim| if d.is_none() { 1.0 } else { pass.shown(&D::Q(d)).map_or(1.0, |s| s.scale) };
        let dec_scales: Vec<(parse::NodeId, f64)> = std::mem::take(&mut pass.dec_notes)
            .into_iter()
            .map(|(id, note)| {
                let k = match note {
                    // the written number in the page's unit for its dimension
                    DecNote::Lit { dim, scale } => scale / scale_of(&pass, dim),
                    // a product or quotient of two page units, in the page's unit
                    // for the result (1 when the page's units agree)
                    DecNote::Fix { mul: true, l, r, out } => scale_of(&pass, l) * scale_of(&pass, r) / scale_of(&pass, out),
                    DecNote::Fix { mul: false, l, r, out } => scale_of(&pass, l) / scale_of(&pass, r) / scale_of(&pass, out),
                };
                (id, k)
            })
            .filter(|(_, k)| (k - 1.0).abs() > 1e-12)
            .collect();
        let check_units: Vec<(parse::NodeId, units::Shown)> = std::mem::take(&mut pass.check_ds)
            .into_iter()
            .filter_map(|(id, d)| pass.shown(&d).filter(|s| !s.list).map(|s| (id, s)))
            .collect();
        drop(pass); // ends the `&*solver` borrow
        for (loc, shown) in print_shown {
            solver.echo_units.entry(loc).or_insert(shown);
        }
        solver.echo_units.extend(echo_units);
        solver.check_units.extend(check_units);
        solver.check_texts.extend(check_texts);
        solver.dec_scales.extend(dec_scales);
        solver.node_dims.extend(dims);
        solver.want_dims.extend(want_dims);
        solver.print_dims.extend(print_dims);
        solver.member_units.extend(member_units);
    }
    Ok(())
}

impl<'a> Pass<'a> {
    /// How an echo of a value in `d` shows its unit: the unit written for the
    /// dimension, else a ratio or product of two written units (`usd/lb`), else
    /// the coherent unit. `None` for a value with no known dimension.
    /// r[impl mimas.echo.unit]
    fn shown(&self, d: &D) -> Option<units::Shown> {
        let (dim, list) = match d {
            D::Q(dim) => (*dim, false),
            D::Arr(inner) => match &**inner {
                D::Q(dim) => (*dim, true),
                _ => return None,
            },
            _ => return None,
        };
        if dim.is_none() || dim == units::INDEX {
            return None;
        }
        let pick = || -> (String, f64) {
            if let Some((n, s)) = self.written.get(&dim) {
                return (n.clone(), *s);
            }
            // a dimension with a plain name of its own (`usd`, `W`) reads
            // better as that than as `lb*usd/lb`
            let coherent = dim.name();
            if !coherent.contains(['/', '*', '^']) {
                return (coherent, 1.0);
            }
            // compose only from units written as one word
            let atoms: Vec<(&Dim, &(String, f64))> = self
                .written
                .iter()
                .filter(|(_, (n, _))| !n.contains(['/', '*', '^']))
                .collect();
            for (ad, (an, asc)) in atoms.iter().copied() {
                for (bd, (bn, bsc)) in atoms.iter().copied() {
                    if ad.div(*bd) == dim {
                        return (format!("{an}/{bn}"), asc / bsc);
                    }
                }
            }
            for (ad, (an, asc)) in atoms.iter().copied() {
                for (bd, (bn, bsc)) in atoms.iter().copied() {
                    if ad.mul(*bd) == dim {
                        // a unit times itself is its square: `12 px²`, not `12 px*px`
                        let name = if an == bn { format!("{an}²") } else { format!("{an}*{bn}") };
                        return (name, asc * bsc);
                    }
                }
            }
            (dim.name(), 1.0)
        };
        let (name, scale) = pick();
        Some(units::Shown { name, scale, list })
    }

    // ---- indexing

    /// Every indexed struct's members that declare a unit, looking through
    /// `?` and `[..]` to the scalar inside.
    fn member_units(&mut self) -> Vec<((String, String), Dim)> {
        fn scalar(d: D) -> Option<Dim> {
            match d {
                D::Q(d) | D::Iv(d) if !d.is_none() => Some(d),
                D::Arr(inner) => scalar(*inner),
                _ => None,
            }
        }
        let mut out = Vec::new();
        let structs: Vec<(String, &'a Struct)> =
            self.structs.iter().map(|(n, s)| (n.clone(), *s)).collect();
        for (name, s) in structs {
            for f in &s.fields {
                let FieldKey::Ident(k) = &f.name else {
                    continue;
                };
                let tps = s.type_params.iter().map(|i| i.lexeme.clone()).collect();
                let d = self.with_ty_params(tps, |p| p.annotation(&f.annotation));
                if let Some(d) = scalar(d) {
                    out.push(((name.clone(), k.lexeme.clone()), d));
                }
            }
        }
        out
    }

    fn index(&mut self, stmts: &'a [Stmt]) {
        for stmt in stmts {
            if let StmtKind::Item(item) = stmt.kind() {
                self.index_item(item, None);
            }
        }
    }

    fn index_item(&mut self, item: &'a Item, owner: Option<&str>) {
        match item.kind() {
            ItemKind::Struct(s) => {
                self.structs.entry(s.name.lexeme.clone()).or_insert(s);
            }
            ItemKind::Enum(e) => {
                self.enum_names.insert(e.head.lexeme.clone());
            }
            ItemKind::Function(f) => {
                let loc = f.name.location;
                self.fns_by_site.insert((loc.file_id, loc.span.start), f);
                match owner {
                    Some(ty) => {
                        self.methods
                            .entry((ty.to_string(), f.name.lexeme.clone()))
                            .or_insert(f);
                    }
                    None => {
                        self.fns.entry(f.name.lexeme.clone()).or_insert(f);
                    }
                }
            }
            ItemKind::Impl(imp) => {
                for inner in &imp.items {
                    self.index_item(inner, Some(&imp.target.lexeme));
                }
            }
            _ => {}
        }
    }

    // ---- reporting

    fn fail(&mut self, at: Location, what: String, label: String, help: &str) {
        if !self.report {
            return;
        }
        self.errors.push(
            DimensionMismatch {
                src: self.solver.src(at),
                at: at.into(),
                what,
                label,
                help: help.to_string(),
            }
            .into(),
        );
    }

    fn describe(d: &Dim) -> String {
        d.describe()
    }

    /// `d` described with the unit as written when `e` is a quantity literal (`5ft` reads
    /// `length (ft)`); computed values fall back to the coherent unit.
    fn describe_operand(&self, e: &Expr, d: &Dim) -> String {
        let inner = match e.kind() {
            ExprKind::Unary(u) if matches!(u.op, UnaryOp::Negative | UnaryOp::Positive) => &u.right,
            _ => e,
        };
        match self.ast.quantity_unit(inner.id()) {
            Some(unit) => d.describe_unit(unit),
            None => Self::describe(d),
        }
    }

    /// Both known and different -> the error every mismatch shares.
    fn expect(&mut self, want: &D, got: &D, at: Location, what: &str) {
        if let Some((w, g)) = clash(want, got) {
            self.fail(
                at,
                format!("{what}: expected {}, found {}", Self::describe(&w), Self::describe(&g)),
                format!("this is {}", Self::describe(&g)),
                "convert with `.to(unit)`, or multiply/divide to change the dimension; a plain `float` means a number with no unit",
            );
        }
    }

    // ---- annotations

    fn annotation(&self, a: &Annotation) -> D {
        match a {
            Annotation::Kw(TyKw::Float) | Annotation::Kw(TyKw::Int) => PLAIN,
            Annotation::Ty(ident) => match self.unit_of(ident) {
                Some(d) => D::Q(d),
                None => D::Any,
            },
            Annotation::Quantity(parts) => {
                let mut dim = Dim::NONE;
                for (unit, exp) in parts {
                    let Some((d, _)) = units::lookup(&unit.lexeme) else {
                        return D::Any;
                    };
                    dim = dim.mul(d.powi(*exp));
                }
                D::Q(dim)
            }
            Annotation::Applied(ty, args)
                if ty.len() == 1 && ty[0].lexeme == "Interval" && args.len() == 1 =>
            {
                match self.annotation(&args[0]) {
                    D::Q(d) => D::Iv(d),
                    _ => D::Any,
                }
            }
            // `dec<usd>`: an exact decimal that carries a unit (r[impl mimas.dec.units])
            Annotation::Applied(ty, args) if ty.len() == 1 && ty[0].lexeme == "dec" && args.len() == 1 => {
                self.annotation(&args[0])
            }
            Annotation::Option(inner) | Annotation::Result(inner) => self.annotation(inner),
            Annotation::Array(inner, _) => D::Arr(Box::new(self.annotation(inner))),
            _ => D::Any,
        }
    }

    /// A type-position name that is a unit (and not a type the program declares).
    fn unit_of(&self, ident: &Ident) -> Option<Dim> {
        if self.structs.contains_key(&ident.lexeme)
            || self.enum_names.contains(&ident.lexeme)
            || self.ty_params.iter().flatten().any(|p| p == &ident.lexeme)
        {
            return None;
        }
        units::lookup(&ident.lexeme).map(|(d, _)| d)
    }

    /// Records the units an annotation spells (`lb`, `[usd/lb]`) as written
    /// for their dimensions, first one wins — see [`Self::shown`].
    fn note_written(&mut self, a: &Annotation) {
        match a {
            Annotation::Ty(ident) => {
                if self.unit_of(ident).is_some()
                    && let Some((dim, scale)) = units::lookup(&ident.lexeme)
                {
                    self.written.entry(dim).or_insert((ident.lexeme.clone(), scale));
                }
            }
            Annotation::Quantity(parts) => {
                let text = parts
                    .iter()
                    .enumerate()
                    .map(|(i, (u, e))| match (i, *e) {
                        (0, 1) => u.lexeme.clone(),
                        (0, e) => format!("{}^{e}", u.lexeme),
                        (_, -1) => format!("/{}", u.lexeme),
                        (_, 1) => format!("*{}", u.lexeme),
                        (_, e) if e < 0 => format!("/{}^{}", u.lexeme, -e),
                        (_, e) => format!("*{}^{e}", u.lexeme),
                    })
                    .collect::<String>();
                if let Some((dim, scale)) = units::parse(&text)
                    && !dim.is_none()
                {
                    self.written.entry(dim).or_insert((text, scale));
                }
            }
            Annotation::Array(inner, _) | Annotation::Option(inner) | Annotation::Result(inner) => {
                self.note_written(inner)
            }
            // `dec<usd>`, `Interval<kW>`: the unit inside
            Annotation::Applied(_, args) if args.len() == 1 => self.note_written(&args[0]),
            _ => {}
        }
    }

    /// Runs `f` with a generic type-param scope pushed (see `ty_params`).
    fn with_ty_params<T>(&mut self, params: Vec<String>, f: impl FnOnce(&mut Self) -> T) -> T {
        self.ty_params.push(params);
        let out = f(self);
        self.ty_params.pop();
        out
    }

    /// The declared type-param names of the adt named `name` (`List` -> `["T"]`).
    fn adt_param_names(&self, name: &str) -> Vec<String> {
        self.solver
            .adts
            .iter()
            .find(|(_, a)| a.name == name)
            .map(|(_, a)| a.type_params.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default()
    }

    // ---- items and statements

    fn run(&mut self, stmts: &'a [Stmt]) {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &'a Stmt) {
        match stmt.kind() {
            StmtKind::Let(l) => {
                // a declared unit is how the program asks to see its
                // dimension — it counts as written before the value's own
                if let Some(a) = &l.annotation {
                    self.note_written(a);
                }
                let got = self.expr(&l.right);
                if let Some(else_branch) = &l.else_branch {
                    self.expr(else_branch);
                }
                let d = match &l.annotation {
                    Some(a) => {
                        let want = self.annotation(a);
                        self.expect(
                            &want,
                            &got,
                            l.right.location(),
                            "declared type doesn't match",
                        );
                        want
                    }
                    None => got,
                };
                // a top-level `let`'s echo prints with its unit, like `print` does
                // r[impl mimas.echo.let]
                if let Some(dim) = d.dim().filter(|x| !x.is_none()) {
                    self.print_dims.insert(stmt.location(), dim);
                }
                self.echo_ds.insert(stmt.location(), d.clone());
                self.bind(&l.left, d);
            }
            StmtKind::Assignment(a) => {
                let left = self.expr(&a.left);
                let right = self.expr(&a.right);
                match a.op {
                    AssignmentOp::Identity | AssignmentOp::PlusEqual | AssignmentOp::MinusEqual => {
                        self.expect(
                            &left,
                            &right,
                            a.right.location(),
                            "assignment doesn't match",
                        );
                    }
                    _ => {}
                }
            }
            StmtKind::Expr(e) => {
                // a bare expression's echo carries its unit (`25kW * 8h` shows `720000000 J`)
                // r[impl mimas.echo]
                let d = self.expr(e);
                if let Some(dim) = d.dim().filter(|x| !x.is_none()) {
                    self.print_dims.insert(stmt.location(), dim);
                }
                self.echo_ds.insert(stmt.location(), d);
            }
            StmtKind::Item(item) => self.item(item),
            // `unit` decls register into the units table at parse — nothing to check
            StmtKind::Module(_) | StmtKind::Unit(_) | StmtKind::Needs(_) => {}
        }
    }

    fn item(&mut self, item: &'a Item) {
        match item.kind() {
            ItemKind::Function(f) => self.function(f),
            ItemKind::Const(c) => {
                let got = self.expr(&c.right);
                let d = match &c.annotation {
                    Some(a) => {
                        let want = self.annotation(a);
                        self.expect(
                            &want,
                            &got,
                            c.right.location(),
                            "declared type doesn't match",
                        );
                        want
                    }
                    None => got,
                };
                self.bind_const(&c.left, d);
            }
            ItemKind::Impl(imp) => {
                // `impl List` sees `T` -- the target's declared params -- like the solver does
                let params = self.adt_param_names(&imp.target.lexeme);
                self.with_ty_params(params, |s| {
                    for inner in &imp.items {
                        s.item(inner);
                    }
                });
            }
            // `check` lines: their bodies get the same dimension check as
            // any code, and a failing one names its values in the page's
            // units (r[impl check.report.units])
            ItemKind::Tests(tests) => {
                for case in &tests.cases {
                    self.expr(&case.expr);
                }
            }
            _ => {}
        }
    }

    fn function(&mut self, f: &'a Function) {
        let params: Vec<String> = f.type_params.iter().map(|i| i.lexeme.clone()).collect();
        self.with_ty_params(params, |s| {
            for p in &f.parameters {
                s.param(p);
            }
            let ret = f.return_type.as_ref().map_or(D::Any, |a| s.annotation(a));
            s.rets.push(ret.clone());
            let body = s.expr(&f.body);
            s.rets.pop();
            if f.return_type.is_some() {
                // a body that ends in `return` never gets here with a value worth checking
                let at = tail_location(&f.body);
                s.expect(&ret, &body, at, "returns the wrong dimension");
            }
        });
    }

    fn param(&mut self, p: &Binding) {
        let d = p.annotation.as_ref().map_or(D::Any, |a| self.annotation(a));
        self.bind(&p.left, d);
    }

    fn dec_of(&self, ident: &Ident) -> Option<DecId> {
        self.solver.node_decs.get(&ident.id).copied()
    }

    fn bind(&mut self, pat: &Pat, d: D) {
        match pat.kind() {
            PatKind::Ident(ident) => {
                if let Some(dec) = self.dec_of(ident) {
                    self.env.insert(dec, d);
                }
            }
            PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                for p in pats {
                    self.bind(p, D::Any);
                }
            }
            PatKind::Struct(_, fields) => {
                for p in fields.values() {
                    self.bind(p, D::Any);
                }
            }
            PatKind::NullBind(p) => self.bind(p, d),
            _ => {}
        }
    }

    /// `bind` for a `const`: same shape walk, but the names land in `consts`. Destructured
    /// leaves degrade to `Any`, same as a `let` tuple destructure.
    fn bind_const(&mut self, pat: &Pat, d: D) {
        match pat.kind() {
            PatKind::Ident(ident) => {
                if let Some(dec) = self.dec_of(ident) {
                    self.consts.insert(dec, d);
                }
            }
            PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                for p in pats {
                    self.bind_const(p, D::Any);
                }
            }
            PatKind::Struct(_, fields) => {
                for p in fields.values() {
                    self.bind_const(p, D::Any);
                }
            }
            PatKind::NullBind(p) => self.bind_const(p, d),
            _ => {}
        }
    }

    // ---- expressions

    fn exprs(&mut self, es: impl IntoIterator<Item = &'a Expr>) {
        for e in es {
            self.expr(e);
        }
    }

    fn is_zero(e: &Expr) -> bool {
        match e.kind() {
            ExprKind::Literal(Literal::Int(0)) => true,
            ExprKind::Literal(Literal::Float(f)) => *f == 0.0,
            ExprKind::Grouping(g) => Self::is_zero(&g.inner),
            _ => false,
        }
    }

    fn expr(&mut self, e: &'a Expr) -> D {
        let d = self.expr_dim(e);
        // record dimensionless results too: a lookup miss means "dim unknown"
        // (`b.x + BALL` joining an `Any` field), which is a different answer
        // than "provably plain" for tools naming a node's unit
        if let D::Q(dim) = d {
            self.dims.insert(e.id(), dim);
        }
        d
    }

    fn expr_dim(&mut self, e: &'a Expr) -> D {
        match e.kind() {
            ExprKind::Literal(l) => self.literal(e, l),
            ExprKind::Ident(ident) => match self.dec_of(ident) {
                Some(dec) => {
                    if let Some(name) = self.unit_decs.get(&dec)
                        && let Some((dim, scale)) = units::lookup(name)
                    {
                        self.written.entry(dim).or_insert((name.clone(), scale));
                    }
                    self
                    .env
                    .get(&dec)
                    .or_else(|| self.consts.get(&dec))
                    .cloned()
                    .unwrap_or(D::Any)
                }
                None => D::Any,
            },
            ExprKind::Grouping(g) => self.expr(&g.inner),
            ExprKind::Unary(u) => {
                let d = self.expr(&u.right);
                match u.op {
                    // a direction word's wrapper puts its axis on a known quantity;
                    // an unknown one stays unknown rather than becoming a bare tag
                    UnaryOp::Negative | UnaryOp::Positive => match (d, self.ast.direction(e.id())) {
                        (D::Q(q), Some(axis)) => D::Q(q.mul(axis)),
                        (D::Iv(q), Some(axis)) => D::Iv(q.mul(axis)),
                        (d, _) => d,
                    },
                    _ => D::Any,
                }
            }
            ExprKind::Evaluation(ev) => {
                let l = self.expr(&ev.left);
                let r = self.expr(&ev.right);
                self.evaluation(e, ev.op, l, r)
            }
            ExprKind::Equality(eq) => {
                let l = self.expr(&eq.left);
                let r = self.expr(&eq.right);
                if !Self::is_zero(&eq.left)
                    && !Self::is_zero(&eq.right)
                    && let Some((a, b)) = clash(&l, &r)
                {
                    let verb = match eq.op {
                        EqualityOp::Equal | EqualityOp::NotEqual => "compare",
                        _ => "order",
                    };
                    self.fail(
                        e.location(),
                        format!(
                            "cannot {verb} {} and {}",
                            self.describe_operand(&eq.left, &a),
                            self.describe_operand(&eq.right, &b)
                        ),
                        format!(
                            "{} against {}",
                            self.describe_operand(&eq.left, &a),
                            self.describe_operand(&eq.right, &b)
                        ),
                        "quantities are only comparable in the same dimension; convert one side with `.to(unit)`",
                    );
                }
                D::Any
            }
            ExprKind::Logical(lg) => {
                self.expr(&lg.left);
                self.expr(&lg.right);
                D::Any
            }
            ExprKind::Coalescence(c) => {
                let l = self.expr(&c.left);
                let r = self.expr(&c.right);
                self.expect(
                    &l,
                    &r,
                    c.right.location(),
                    "`??` fallback has a different dimension",
                );
                l
            }
            ExprKind::Unwrap(u) => self.expr(&u.expr),
            ExprKind::Propagate(d) => self.expr(&d.expr),
            ExprKind::Absolve(a) => {
                let l = self.expr(&a.left);
                self.expr(&a.handler);
                l
            }
            ExprKind::Block(b) => {
                self.run(&b.body);
                b.yielded_expr.as_ref().map_or(D::Any, |y| self.expr(y))
            }
            ExprKind::If(i) => {
                self.expr(&i.condition);
                if let Some(binding) = &i.binding {
                    self.bind(binding, D::Any);
                }
                let then = self.expr(&i.main_body);
                match &i.else_expr {
                    Some(other) => {
                        let els = self.expr(other);
                        self.branches(then, els, other.location())
                    }
                    None => D::Any,
                }
            }
            ExprKind::Match(m) => {
                self.expr(&m.identity);
                let mut out: Option<D> = None;
                for case in &m.cases {
                    self.bind(case.pat(), D::Any);
                    if let Some(g) = case.guard() {
                        self.expr(g);
                    }
                    let d = self.expr(case.body());
                    out = Some(match out {
                        None => d,
                        Some(prev) => self.branches(prev, d, case.body().location()),
                    });
                }
                out.unwrap_or(D::Any)
            }
            ExprKind::Call(c) => self.call(e, c),
            ExprKind::Access(a) => self.access(a),
            ExprKind::Closure(c) => {
                for p in &c.parameters {
                    self.param(p);
                }
                self.rets.push(D::Any);
                self.expr(&c.body);
                self.rets.pop();
                D::Any
            }
            ExprKind::For(f) => {
                let it = self.expr(&f.iterator);
                let item = match (&it, f.iterator.kind()) {
                    (D::Arr(inner), _) => (**inner).clone(),
                    (_, ExprKind::Range(_)) => PLAIN,
                    _ => D::Any,
                };
                self.bind(&f.binding, item);
                self.collected(|p| {
                    p.expr(&f.body);
                })
            }
            ExprKind::While(w) => {
                self.expr(&w.header);
                if let Some(b) = &w.binding {
                    self.bind(b, D::Any);
                }
                self.collected(|p| {
                    p.expr(&w.body);
                })
            }
            ExprKind::Loop(l) => self.collected(|p| {
                p.expr(&l.body);
            }),
            ExprKind::Collect(c) => {
                let d = self.expr(&c.value);
                if let Some(slot) = self.collects.last_mut() {
                    *slot = Some(match slot.take() {
                        None => d,
                        Some(prev) => join(prev, &d),
                    });
                }
                D::Any
            }
            ExprKind::Return(r) => {
                if let Some(v) = &r.value {
                    let got = self.expr(v);
                    if let Some(want) = self.rets.last().cloned() {
                        self.expect(&want, &got, v.location(), "returns the wrong dimension");
                    }
                }
                D::Any
            }
            ExprKind::Break(b) => {
                if let Some(v) = &b.value {
                    self.expr(v);
                }
                D::Any
            }
            ExprKind::Raise(r) => {
                self.expr(&r.value);
                D::Any
            }
            ExprKind::Range(r) => {
                self.expr(&r.start);
                self.expr(&r.end);
                D::Any
            }
            ExprKind::In(i) => {
                self.expr(&i.left);
                self.expr(&i.right);
                D::Any
            }
            ExprKind::FString(f) => {
                // the message `check a == b` fails with names both values
                let check_message = matches!(
                    f.parts.first(),
                    Some(FStringPart::Literal(l)) if l.starts_with("assertion failed: ")
                );
                for part in &f.parts {
                    if let FStringPart::Expr(x) = part {
                        let d = self.expr(x);
                        if check_message {
                            let loc = x.location();
                            let text = self.solver.sources.get(&loc.file_id).and_then(|s| {
                                let a = loc.span.start();
                                s.inner().get(a..a + loc.span.len()).map(str::to_string)
                            });
                            // a written number with a unit shows as written: `47lb`
                            match text {
                                Some(t)
                                    if matches!(d, D::Q(q) if !q.is_none())
                                        && t.trim_start_matches('-').starts_with(|c: char| c.is_ascii_digit())
                                        && t.ends_with(|c: char| c.is_alphabetic()) =>
                                {
                                    self.check_texts.insert(x.id(), t);
                                }
                                _ => {
                                    self.check_ds.insert(x.id(), d);
                                }
                            }
                        }
                    }
                }
                D::Any
            }
            ExprKind::Continue(_) | ExprKind::Poison(_) => D::Any,
        }
    }

    /// Two arms of an `if`/`match` that disagree are a mistake; otherwise their shared dimension.
    fn branches(&mut self, a: D, b: D, at: Location) -> D {
        if let Some((x, y)) = clash(&a, &b) {
            self.fail(
                at,
                format!(
                    "branches disagree: {} and {}",
                    Self::describe(&x),
                    Self::describe(&y)
                ),
                format!("this branch is {}", Self::describe(&y)),
                "every branch of an `if` or `match` has to give the same dimension",
            );
        }
        join(a, &b)
    }

    fn literal(&mut self, e: &'a Expr, l: &'a Literal) -> D {
        match l {
            Literal::Float(_) | Literal::Int(_) => {
                // a `dec` literal holds its number exactly or is an error,
                // never a silent 0 or a rounded float (r[impl mimas.dec.literal])
                if self.solver.dec_literals.contains(&e.id())
                    && let Some(text) = self.solver.dec_texts.get(&e.id())
                {
                    let clean = text.replace(['_', ','], "");
                    let end = clean
                        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+')))
                        .unwrap_or(clean.len());
                    if clean[..end].parse::<shared::dec::Dec>().is_err() {
                        self.fail(
                            e.location(),
                            format!("`{text}` doesn't fit in a `dec` exactly"),
                            "more digits than a `dec` holds".to_string(),
                            "a `dec` holds about 16 significant digits; use a `float` for a wider number",
                        );
                    }
                }
                // `3index` is the one int literal that carries a dimension
                let d = self.ast.quantity(e.id());
                if let (Some(dim), Some(text)) = (d, self.ast.quantity_unit(e.id()))
                    && !dim.is_none()
                    && let Some((_, scale)) = units::parse(text)
                {
                    self.written.entry(dim).or_insert((text.to_string(), scale));
                    if self.solver.dec_literals.contains(&e.id()) {
                        self.dec_notes.insert(e.id(), DecNote::Lit { dim, scale });
                    }
                }
                d.map_or(PLAIN, D::Q)
            }
            Literal::Array(items) => {
                let mut out: Option<D> = None;
                for item in items {
                    let d = self.expr(item);
                    out = Some(match out {
                        None => d,
                        Some(prev) => {
                            if let Some((x, y)) = clash(&prev, &d) {
                                self.fail(
                                    item.location(),
                                    format!(
                                        "a list mixes {} and {}",
                                        Self::describe(&x),
                                        Self::describe(&y)
                                    ),
                                    format!("this is {}", Self::describe(&y)),
                                    "the items of a list share one dimension; convert with `.to(unit)` to store plain numbers",
                                );
                            }
                            join(prev, &d)
                        }
                    });
                }
                D::Arr(Box::new(out.unwrap_or(D::Any)))
            }
            Literal::Struct(s) => {
                let name = s.name.as_ident().map(|i| i.lexeme.clone());
                let decl = name.as_ref().and_then(|n| self.structs.get(n).copied());
                if let Some(base) = &s.base {
                    self.expr(base);
                }
                for (key, value) in &s.fields {
                    let got = self.expr(value);
                    let (Some(decl), FieldKey::Ident(k)) = (decl, key) else {
                        continue;
                    };
                    let field = decl
                        .fields
                        .iter()
                        .find(|f| matches!(&f.name, FieldKey::Ident(i) if i.lexeme == k.lexeme));
                    if let Some(field) = field {
                        let want = self.with_ty_params(
                            decl.type_params.iter().map(|i| i.lexeme.clone()).collect(),
                            |s| s.annotation(&field.annotation),
                        );
                        self.expect(
                            &want,
                            &got,
                            value.location(),
                            &format!("field `{}`", k.lexeme),
                        );
                    }
                }
                D::Any
            }
            // a bare `{ … }` the solver made a struct checks its fields' units
            // like `Name { … }` does r[impl mimas.record.context]
            Literal::Record(fields) => {
                let decl = self.adt_name(e).and_then(|n| self.structs.get(&n).copied());
                for (k, value) in fields {
                    let got = self.expr(value);
                    let Some(decl) = decl else { continue };
                    let field = decl
                        .fields
                        .iter()
                        .find(|f| matches!(&f.name, FieldKey::Ident(i) if i.lexeme == k.lexeme));
                    if let Some(field) = field {
                        let want = self.with_ty_params(
                            decl.type_params.iter().map(|i| i.lexeme.clone()).collect(),
                            |s| s.annotation(&field.annotation),
                        );
                        self.expect(&want, &got, value.location(), &format!("field `{}`", k.lexeme));
                    }
                }
                D::Any
            }
            Literal::Tuple(items) => {
                self.exprs(items.iter());
                D::Any
            }
            _ => D::Any,
        }
    }

    fn evaluation(&mut self, e: &Expr, op: EvaluationOp, l: D, r: D) -> D {
        // a `dec` quantity is held in the page's unit and a float one in SI,
        // so the two can't meet (r[impl mimas.dec.units])
        if let ExprKind::Evaluation(ev) = e.kind() {
            let q = |d: &D| d.dim().is_some_and(|x| !x.is_none());
            let (ld, rd) = (self.is_dec(&ev.left), self.is_dec(&ev.right));
            let (lf, rf) = (self.is_float(&ev.left), self.is_float(&ev.right));
            if ((ld && rf) || (lf && rd)) && (q(&l) || q(&r)) {
                self.fail(
                    e.location(),
                    "a `dec` quantity meets a `float` one".to_string(),
                    "`dec` with `float`".to_string(),
                    "write both as `dec` (a number written beside a `dec` is one), or convert the `dec` with `.to_float()`",
                );
            }
        }
        let name = |op| match op {
            EvaluationOp::Plus => "add",
            EvaluationOp::Minus => "subtract",
            _ => "combine",
        };
        match op {
            EvaluationOp::Plus | EvaluationOp::Minus | EvaluationOp::Modulo => {
                if let Some((a, b)) = clash(&l, &r) {
                    let (da, db) = match e.kind() {
                        ExprKind::Evaluation(ev) => (
                            self.describe_operand(&ev.left, &a),
                            self.describe_operand(&ev.right, &b),
                        ),
                        _ => (Self::describe(&a), Self::describe(&b)),
                    };
                    self.fail(
                        e.location(),
                        format!("cannot {} {} and {}", name(op), da, db),
                        format!("{} with {}", da, db),
                        "quantities only add and subtract in the same dimension -- convert one with `.to(unit)`, or multiply/divide to make a new quantity",
                    );
                }
                match (&l, &r) {
                    (D::Iv(a), D::Iv(b) | D::Q(b)) | (D::Q(a), D::Iv(b)) if a == b => D::Iv(*a),
                    _ => join(l, &r),
                }
            }
            EvaluationOp::Multiply | EvaluationOp::Divide | EvaluationOp::Div => {
                let interval = matches!(l, D::Iv(_)) || matches!(r, D::Iv(_));
                match (l.dim(), r.dim()) {
                    (Some(a), Some(b)) => {
                        let d = if op == EvaluationOp::Multiply {
                            a.mul(b)
                        } else {
                            a.div(b)
                        };
                        if self.is_dec(e) && !(a.is_none() && b.is_none()) {
                            let mul = op == EvaluationOp::Multiply;
                            self.dec_notes.insert(e.id(), DecNote::Fix { mul, l: a, r: b, out: d });
                        }
                        if interval { D::Iv(d) } else { D::Q(d) }
                    }
                    _ => D::Any,
                }
            }
            _ => D::Any,
        }
    }

    // ---- access and calls

    /// Whether the solver typed `e` as a `dec`.
    fn is_dec(&self, e: &Expr) -> bool {
        self.solver.node_to_vid.get(&e.id()).is_some_and(|v| self.solver.is_dec(&Ty::Vid(*v)))
    }

    /// Whether the solver typed `e` as a `float`.
    fn is_float(&self, e: &Expr) -> bool {
        self.solver
            .node_to_vid
            .get(&e.id())
            .is_some_and(|v| matches!(Ty::Vid(*v).normalized(self.solver), Ty::Float))
    }

    fn adt_name(&self, e: &Expr) -> Option<String> {
        let vid = self.solver.node_to_vid.get(&e.id())?;
        match Ty::Vid(*vid).normalized(self.solver) {
            Ty::Adt(aid, _) | Ty::Identity(aid, _) => Some(self.solver.adts[aid].name.clone()),
            _ => None,
        }
    }

    fn access(&mut self, a: &'a Access) -> D {
        match a {
            Access::Dot { left, right, .. } => {
                let recv = self.expr(left);
                // a field: look its declaration up by the receiver's struct
                if let ExprKind::Ident(field) = right.kind() {
                    if let D::Iv(d) = recv
                        && matches!(field.lexeme.as_str(), "lo" | "mid" | "hi")
                    {
                        return D::Q(d);
                    }
                    if let Some(ty) = self.adt_name(left) {
                        if let Some(decl) = self.structs.get(&ty).copied()
                            && let Some(f) = decl.fields.iter().find(
                                |f| matches!(&f.name, FieldKey::Ident(i) if i.lexeme == field.lexeme),
                            )
                        {
                            return self.with_ty_params(
                                decl.type_params.iter().map(|i| i.lexeme.clone()).collect(),
                                |s| s.annotation(&f.annotation),
                            );
                        }
                        // host adts carry field dims on their api descriptor (`Input.dt` is `s`)
                        if let Some(d) = self.solver.field_dims.get(&(ty, field.lexeme.clone())) {
                            return D::Q(*d);
                        }
                    }
                }
                D::Any
            }
            Access::Square { left, key, .. } => {
                let l = self.expr(left);
                self.expr(key);
                match l {
                    D::Arr(inner) => *inner,
                    _ => D::Any,
                }
            }
            Access::Identity { .. } | Access::DoubleColon { .. } => D::Any,
        }
    }

    fn call(&mut self, e: &'a Expr, c: &'a parse::Call) -> D {
        // methods on a value: `x.abs()`, `list.push(v)`, or a user `impl` method
        if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind()
            && let ExprKind::Ident(method) = right.kind()
        {
            let recv = self.expr(left);
            let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
            if let Some(ty) = self.adt_name(left)
                && let Some(f) = self
                    .methods
                    .get(&(ty.clone(), method.lexeme.clone()))
                    .copied()
            {
                return self.apply(f, c, &args, true, Some(&ty));
            }
            if let Some(d) = self.native_apply(e, c, &args) {
                return d;
            }
            return self.builtin(e, &method.lexeme, recv, c, &args);
        }
        // `Type::assoc(...)` or a module fn like `music::play_for(...)`
        if let ExprKind::Access(Access::DoubleColon { left, right }) = c.left.kind() {
            let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
            if let ExprKind::Ident(ty) = left.kind() {
                if let Some(f) = self
                    .methods
                    .get(&(ty.lexeme.clone(), right.lexeme.clone()))
                    .copied()
                {
                    return self.apply(f, c, &args, false, Some(&ty.lexeme));
                }
                if ty.lexeme == "Interval" {
                    return self.interval_ctor(&right.lexeme, c, &args);
                }
            }
            return self.native_apply(e, c, &args).unwrap_or(D::Any);
        }
        let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
        if let ExprKind::Ident(callee) = c.left.kind() {
            let user = self.dec_of(callee).and_then(|dec| {
                let loc = self.solver.decs[dec].location;
                self.fns_by_site
                    .get(&(loc.file_id, loc.span.start))
                    .copied()
            });
            if let Some(f) = user {
                return self.apply(f, c, &args, false, None);
            }
            if let Some(d) = self.aggregate(e, c, callee, &args) {
                return d;
            }
        } else {
            self.expr(&c.left);
        }
        self.native_apply(e, c, &args).unwrap_or(D::Any)
    }

    /// A call the solver resolved to a native: check each argument against the slot's declared
    /// dimension (from unit-carrying `MimasType`s like `Secs`), and the call is worth the
    /// declared return dimension. `None` when the callee isn't native *or* declares no dims at
    /// all -- a fully dimension-agnostic native falls through so `builtin`'s special cases
    /// (`pull`, `to`, `pow`, ...) still see it.
    fn native_apply(&mut self, e: &'a Expr, c: &'a parse::Call, args: &[D]) -> Option<D> {
        let dec = match c.left.kind() {
            ExprKind::Ident(i) => self.solver.node_decs.get(&i.id).copied(),
            ExprKind::Access(Access::Dot { .. } | Access::DoubleColon { .. }) => {
                self.solver.node_decs.get(&c.left.id()).copied()
            }
            _ => None,
        }?;
        let binding = self.solver.dec_to_native.get(&dec)?;
        // `print`/`dbg`/`warn` of a quantity gets its unit in the output —
        // remember the arg's dim at the call site so the program can ship
        // it to the native.
        if let ExprKind::Ident(_) = c.left.kind()
            && matches!(
                self.solver.decs[dec].name.as_str(),
                "print" | "dbg" | "warn"
            )
            && let Some(d) = args.first().and_then(D::dim)
            && !d.is_none()
        {
            self.print_dims.insert(e.location(), d);
        }
        if binding.sig.return_dim.is_none() && binding.sig.param_dims.iter().all(Option::is_none) {
            return None;
        }
        let name = match c.left.kind() {
            ExprKind::Ident(i) => i.lexeme.as_str(),
            ExprKind::Access(Access::Dot { right, .. }) => {
                right.as_ident().map_or("?", |i| i.lexeme.as_str())
            }
            ExprKind::Access(Access::DoubleColon { right, .. }) => right.lexeme.as_str(),
            _ => "?",
        };
        for (i, (arg, got)) in c.arguments.iter().zip(args).enumerate() {
            let Some(Some(want)) = binding.sig.param_dims.get(i) else {
                continue;
            };
            if !want.is_none() {
                self.want_dims.insert(arg.value.id(), *want);
            }
            self.expect(
                &D::Q(*want),
                got,
                arg.value.location(),
                &format!("argument {} of `{name}`", i + 1),
            );
        }
        Some(binding.sig.return_dim.map_or(D::Any, D::Q))
    }

    /// Check a call to a user function against its declared parameters; its declared return is
    /// what the call is worth. `owner` names the impl target for methods and assoc fns, whose
    /// declared type params are in scope in the signature too.
    fn apply(
        &mut self,
        f: &'a Function,
        c: &'a parse::Call,
        args: &[D],
        method: bool,
        owner: Option<&str>,
    ) -> D {
        let params: Vec<&Binding> = f
            .parameters
            .iter()
            .skip(usize::from(method || f.is_method()))
            .collect();
        let mut scope: Vec<String> = f.type_params.iter().map(|i| i.lexeme.clone()).collect();
        if let Some(owner) = owner {
            scope.extend(self.adt_param_names(owner));
        }
        self.with_ty_params(scope, |s| {
            for (i, (arg, got)) in c.arguments.iter().zip(args).enumerate() {
                let param = match &arg.name {
                    Some(n) => params
                        .iter()
                        .find(|p| p.left.as_ident().is_some_and(|id| id.lexeme == n.lexeme)),
                    None => params.get(i),
                };
                let Some(p) = param else { continue };
                let Some(ann) = &p.annotation else { continue };
                let want = s.annotation(ann);
                if let Some(d) = want.dim()
                    && !d.is_none()
                {
                    s.want_dims.insert(arg.value.id(), d);
                }
                let pname = p.left.as_ident().map_or("argument", |i| i.lexeme.as_str());
                s.expect(
                    &want,
                    got,
                    arg.value.location(),
                    &format!("argument `{pname}` of `{}`", f.name.lexeme),
                );
            }
            f.return_type.as_ref().map_or(D::Any, |a| s.annotation(a))
        })
    }

    /// `Interval::pm(x, rel)` and friends: an interval in the unit of its number arguments.
    fn interval_ctor(&mut self, name: &str, c: &'a parse::Call, args: &[D]) -> D {
        // which arguments are numbers in the interval's unit (`pm`'s second is a plain fraction)
        let measured: &[usize] = match name {
            "pm" | "exact" => &[0],
            "within" | "span" => &[0, 1],
            "of" => &[0, 1, 2],
            _ => return D::Any,
        };
        let Some(first) = measured.first().and_then(|i| args.get(*i)) else {
            return D::Any;
        };
        for &i in &measured[1..] {
            if let Some(a) = args.get(i) {
                self.expect(
                    first,
                    a,
                    c.arguments[i].value.location(),
                    &format!("`Interval::{name}` bounds need the same dimension"),
                );
            }
        }
        if name == "pm"
            && let Some(rel) = args.get(1)
        {
            self.expect(
                &PLAIN,
                rel,
                c.arguments[1].value.location(),
                "`Interval::pm` takes a plain fraction",
            );
        }
        match first {
            D::Q(d) => D::Iv(*d),
            _ => D::Any,
        }
    }

    /// The prelude's free aggregates: `sum`/`avg` of a list of quantities is a
    /// quantity of their dimension, `count` is a plain number, and the `_by`
    /// forms keep the same dims per group (a grouped result reads like a list).
    /// `None` when the callee isn't one of them.
    /// r[impl mimas.agg.free] r[impl mimas.agg.by]
    fn aggregate(&mut self, e: &'a Expr, c: &'a parse::Call, callee: &Ident, args: &[D]) -> Option<D> {
        let dec = self.dec_of(callee)?;
        self.solver.dec_to_native.get(&dec)?;
        let item = |d: Option<&D>| match d {
            Some(D::Arr(inner)) => (**inner).clone(),
            _ => D::Any,
        };
        let _ = (e, c);
        Some(match self.solver.decs[dec].name.as_str() {
            "sum" | "avg" => item(args.first()),
            "count" => PLAIN,
            "sum_by" | "avg_by" => D::Arr(Box::new(item(args.first()))),
            "count_by" => D::Arr(Box::new(PLAIN)),
            _ => return None,
        })
    }

    /// Walks a loop body; the loop is worth a list of what its `collect`s hand
    /// over (unknown when it collects nothing the checker can follow).
    fn collected(&mut self, body: impl FnOnce(&mut Self)) -> D {
        self.collects.push(None);
        body(self);
        match self.collects.pop().flatten() {
            Some(d) => D::Arr(Box::new(d)),
            None => D::Any,
        }
    }

    /// Methods of the built-in float/int/list types.
    fn builtin(&mut self, e: &'a Expr, name: &str, recv: D, c: &'a parse::Call, args: &[D]) -> D {
        match (&recv, name) {
            // `df.pull("kwh")` on a schema'd frame: a numeric column reads as quantities
            // of its declared unit (or plain numbers when the column has no unit)
            (_, "pull") => {
                let recv_schema = match c.left.kind() {
                    ExprKind::Access(Access::Dot { left, .. }) => self.solver.frame_schema_of(left),
                    _ => None,
                };
                match (recv_schema, c.arguments.first().map(|a| a.value.kind())) {
                    (Some(schema), Some(ExprKind::Literal(Literal::String(name)))) => {
                        match schema.cols.get(name.as_str()) {
                            Some(col) => match col.ty {
                                Some(shared::Ty::Int) | Some(shared::Ty::Float) => {
                                    D::Arr(Box::new(D::Q(col.dim.unwrap_or(Dim::NONE))))
                                }
                                _ => D::Arr(Box::new(D::Any)),
                            },
                            None => D::Any,
                        }
                    }
                    _ => D::Any,
                }
            }
            // `df.pull_as("kwh", "kWh")`: a column of quantities in that unit
            (_, "pull_as") => match c.arguments.get(1).map(|a| a.value.kind()) {
                Some(ExprKind::Literal(Literal::String(u))) => match units::parse(u) {
                    Some((d, _)) => D::Arr(Box::new(D::Q(d))),
                    None => {
                        self.fail(
                            c.arguments[1].value.location(),
                            format!("`{u}` isn't a unit"),
                            "unknown unit".into(),
                            "units are things like m, s, kg, W, kWh, usd -- see the Units page of the book",
                        );
                        D::Any
                    }
                },
                _ => D::Any,
            },
            (D::Iv(d), "width") => D::Q(*d),
            (D::Iv(d), "sample") => D::Q(*d),
            (D::Iv(d), "contains") => {
                if let Some(a) = args.first() {
                    self.expect(
                        &D::Q(*d),
                        a,
                        c.arguments[0].value.location(),
                        "`contains` needs the same dimension",
                    );
                }
                PLAIN
            }
            (D::Iv(d), "overlaps") => {
                if let Some(a) = args.first() {
                    self.expect(
                        &D::Iv(*d),
                        a,
                        c.arguments[0].value.location(),
                        "`overlaps` needs the same dimension",
                    );
                }
                PLAIN
            }
            (D::Iv(d), "pow") => {
                let n = c.arguments.first().and_then(|a| match a.value.kind() {
                    ExprKind::Literal(Literal::Int(n)) => Some(*n as f64),
                    ExprKind::Literal(Literal::Float(n)) => Some(*n),
                    _ => None,
                });
                match n {
                    Some(n) if n.fract() == 0.0 && n.abs() < 100.0 => D::Iv(d.powi(n as i8)),
                    _ if d.is_none() => D::Iv(*d),
                    _ => D::Any,
                }
            }
            (D::Q(d), "abs" | "round" | "floor" | "ceil" | "to_int" | "to_float" | "neg") => {
                D::Q(*d)
            }
            (D::Q(_), "signum") => PLAIN,
            (D::Q(d), "min" | "max" | "hypot") => {
                if let Some(a) = args.first() {
                    self.expect(
                        &recv,
                        a,
                        c.arguments[0].value.location(),
                        &format!("`{name}` needs the same dimension"),
                    );
                }
                D::Q(*d)
            }
            (D::Q(d), "clamp") => {
                for (a, arg) in args.iter().zip(&c.arguments) {
                    self.expect(
                        &recv,
                        a,
                        arg.value.location(),
                        "`clamp` bounds need the same dimension",
                    );
                }
                D::Q(*d)
            }
            (D::Q(d), "sqrt") => match d.sqrt() {
                Some(h) => D::Q(h),
                None => {
                    self.fail(
                        e.location(),
                        format!("cannot take the square root of {}", Self::describe(d)),
                        "odd exponents".into(),
                        "a square root halves every exponent, so they must all be even",
                    );
                    D::Any
                }
            },
            (D::Q(d), "pow") => {
                let n = c.arguments.first().and_then(|a| match a.value.kind() {
                    ExprKind::Literal(Literal::Int(n)) => Some(*n as f64),
                    ExprKind::Literal(Literal::Float(n)) => Some(*n),
                    _ => None,
                });
                match n {
                    Some(n) if n.fract() == 0.0 && n.abs() < 100.0 => D::Q(d.powi(n as i8)),
                    _ if d.is_none() => PLAIN,
                    _ => D::Any,
                }
            }
            // Trig eats an angle or a plain number (plain reads as
            // radians, as it always has); the inverses take a plain
            // number and give back an angle — `x.asin().to("deg")`.
            // r[impl mimas.unit.trig]
            (D::Q(d), "sin" | "cos" | "tan") => {
                if !(d.is_none() || *d == units::ANGLE) {
                    self.fail(
                        e.location(),
                        format!("`{name}` needs an angle or a plain number, found {}", Self::describe(d)),
                        "has a unit".into(),
                        "angles are `deg`/`rad`/`turn` quantities; anything else needs `.to(unit)` first",
                    );
                }
                PLAIN
            }
            (D::Q(d), "asin" | "acos" | "atan") => {
                if !d.is_none() {
                    self.fail(
                        e.location(),
                        format!("`{name}` needs a plain number, found {}", Self::describe(d)),
                        "has a unit".into(),
                        "divide by a unit to get a plain number, or use `.to(unit)`",
                    );
                }
                D::Q(units::ANGLE)
            }
            // `y.atan2(x)`: the two legs must measure the same thing —
            // atan2(3px, 4px) is an angle, atan2(3px, 4s) is a bug.
            (D::Q(_), "atan2") => {
                if let Some(a) = args.first() {
                    self.expect(
                        &recv,
                        a,
                        c.arguments[0].value.location(),
                        "`atan2` needs both legs in the same dimension",
                    );
                }
                D::Q(units::ANGLE)
            }
            (D::Q(d), "exp" | "ln") => {
                if !d.is_none() {
                    self.fail(
                        e.location(),
                        format!("`{name}` needs a plain number, found {}", Self::describe(d)),
                        "has a unit".into(),
                        "divide by a unit to get a plain number, or use `.to(unit)`",
                    );
                }
                PLAIN
            }
            (D::Q(d), "to") => {
                // `x.to("kWh")`: the unit must measure the same thing; the result is a plain number
                if let Some(ExprKind::Literal(Literal::String(u))) =
                    c.arguments.first().map(|a| a.value.kind())
                {
                    match units::parse(u) {
                        Some((ud, _)) if !Dim::agree(ud, *d) => self.fail(
                            c.arguments[0].value.location(),
                            format!("cannot express {} in {u}", Self::describe(d)),
                            format!("{u} is {}", Self::describe(&ud)),
                            "pick a unit of the same kind (kW for power, kWh for energy, ...)",
                        ),
                        None => self.fail(
                            c.arguments[0].value.location(),
                            format!("`{u}` isn't a unit"),
                            "unknown unit".into(),
                            "units are things like m, s, kg, W, kWh, usd -- see the Units page of the book",
                        ),
                        _ => {}
                    }
                }
                // `x.to(1kWh)`: the argument itself is a quantity and `to` divides,
                // so it must measure the same thing. A `str` arrives `Any` — this
                // fires only on known numbers.
                // r[impl mimas.unit.to]
                if let Some(D::Q(ud)) = args.first()
                    && *ud != *d
                {
                    self.fail(
                        c.arguments[0].value.location(),
                        format!(
                            "cannot express {} in {}",
                            Self::describe(d),
                            Self::describe(ud)
                        ),
                        "the argument is a quantity".into(),
                        "`to` divides — pick a quantity of the same kind (x.to(1kWh)), or give the unit's name",
                    );
                }
                PLAIN
            }
            (D::Arr(inner), "push") => {
                if let Some(a) = args.first() {
                    self.expect(inner, a, c.arguments[0].value.location(), "list item");
                }
                D::Any
            }
            (D::Arr(inner), "pop" | "max" | "min" | "sum") => (**inner).clone(),
            (D::Arr(_), "len") => PLAIN,
            _ => D::Any,
        }
    }
}

/// Where a function body's value comes from: its final expression, for pointing an error at it.
fn tail_location(body: &Expr) -> Location {
    if let ExprKind::Block(b) = body.kind()
        && let Some(y) = &b.yielded_expr
    {
        return tail_location(y);
    }
    body.location()
}

#[allow(dead_code)]
fn _unused(_: NamedSource<std::sync::Arc<str>>) {}

/// A `dec` quantity the pass scales once the page's units are known: a
/// written literal (`12lb`), or a product or quotient whose operands' page
/// units don't compose to the result's.
enum DecNote {
    Lit { dim: Dim, scale: f64 },
    Fix { mul: bool, l: Dim, r: Dim, out: Dim },
}
