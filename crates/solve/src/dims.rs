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
use shared::{
    Located, Location, Ty, units,
    units::{Dim, DimTree},
};

use crate::{Result, Solver, components::DecId, errors::DimensionMismatch};

/// What the checker knows about a value.
#[derive(Debug, Clone, PartialEq)]
enum D {
    /// Can't tell (or isn't a number): never an error.
    Any,
    /// A projection carries its element dimension.
    Place(Box<D>),
    /// A number in this dimension; [`Dim::NONE`] is a plain number.
    Q(Dim),
    /// A list whose items share one dimension.
    Arr(Box<D>),
    /// A tuple whose items may each carry a dimension — `(px↔, px↕)` a
    /// point, where a list would demand one shared dimension.
    // r[impl mimas.unit.tuple]
    Tuple(Vec<D>),
    /// An `Interval` of numbers in this dimension: `lo`, `mid` and `hi` are all quantities of it,
    /// and arithmetic on it keeps the unit like arithmetic on a plain quantity does.
    Iv(Dim),
    /// A function: its parameters' dims and its return's. A closure, a
    /// `(px) -> px` annotation, a binding that holds either: calling it
    /// checks the arguments and gives the return (r[impl mimas.unit.closure]).
    Fn(Vec<D>, Box<D>),
}

const PLAIN: D = D::Q(Dim::NONE);

impl D {
    /// The dimension of a quantity or an interval of them.
    fn dim(&self) -> Option<Dim> {
        match self {
            D::Q(d) | D::Iv(d) => Some(*d),
            D::Place(inner) => inner.dim(),
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
    (angle_power(x) && y.is_none()) || (angle_power(y) && x.is_none())
}

/// A whole power of the angle unit, `rad`, `1/rad`, `rad²`: radians are
/// plain numbers that keep a name, so any power of them meets a plain
/// number (`t * 180.0 / PI - 137.0`) and feeds `ln`, `exp` and `sqrt`.
fn angle_power(d: Dim) -> bool {
    (1..=4).any(|k| d == units::ANGLE.powi(k) || d == units::ANGLE.powi(-k))
}

/// Two known dimensions that differ; `None` when either is unknown or they agree.
fn clash(a: &D, b: &D) -> Option<(Dim, Dim)> {
    match (a, b) {
        (D::Q(x) | D::Iv(x), D::Q(y) | D::Iv(y))
            if !Dim::agree(*x, *y)
                && !index_plain(x.bare(), y.bare())
                && !angle_plain(x.bare(), y.bare()) =>
        {
            Some((*x, *y))
        }
        (D::Arr(x), D::Arr(y)) => clash(x, y),
        // a place clashes on its element: `&c` of a plain `c` can't fill an
        // `inout s` slot any more than `c` could fill `x: s`
        (D::Place(x), D::Place(y)) => clash(x, y),
        // member-wise: a swapped `(px↕, px↔)` reports its first bad slot,
        // an arity miss the type solver's own error
        (D::Tuple(x), D::Tuple(y)) => x.iter().zip(y).find_map(|(a, b)| clash(a, b)),
        // a function fits a fn-typed slot when its parameters and its
        // return agree (r[impl mimas.unit.closure])
        (D::Fn(xp, xr), D::Fn(yp, yr)) => {
            xp.iter().zip(yp).find_map(|(a, b)| clash(a, b)).or_else(|| clash(xr, yr))
        }
        _ => None,
    }
}

/// A composite slot (tuple, list) whose elements claim a convention the
/// slot doesn't share — there's no one value to negate, so unlike a
/// scalar it fails rather than convert.
fn signed_uncovered(want: &D, got: &D) -> bool {
    match (want, got) {
        (D::Q(w) | D::Iv(w), D::Q(g) | D::Iv(g)) => {
            Dim::sign_flip(*w, *g) || Dim::sign_flip(*g, *w)
        }
        (D::Arr(w), D::Arr(g)) => signed_uncovered(w, g),
        (D::Tuple(w), D::Tuple(g)) => {
            w.iter().zip(g).any(|(w, g)| signed_uncovered(w, g))
        }
        _ => false,
    }
}

/// A native signature's nested dimension claim as a `D` — `Vec<Px>` checks
/// args as `[px]`, `Vec<(PxX, PxY)>` as `(px↔, px↕)` element-wise.
// r[impl mimas.unit.nested]
fn dim_tree(t: &DimTree) -> D {
    match t {
        DimTree::Q(d) => D::Q(*d),
        DimTree::Arr(inner) => D::Arr(Box::new(dim_tree(inner))),
        DimTree::Tup(parts) => D::Tuple(parts.iter().map(dim_tree).collect()),
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
        // `[(a1, b1), (a2, b2)]` keeps both columns' tags when the
        // arities match; a ragged mix degrades like any other doubt
        (D::Tuple(x), D::Tuple(y)) if x.len() == y.len() => {
            D::Tuple(x.iter().zip(y).map(|(a, b)| join(a.clone(), b)).collect())
        }
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
    /// Each display site (`print`/`dbg`/`warn` call, top-level echo) with the
    /// shown expression and its dim, made a [`units::Shape`] at the end of the
    /// walk (r[impl mimas.display.one]).
    site_ds: FxIndexMap<Location, (NodeId, D)>,
    /// Each f-string hole's dim, made a [`units::Shape`] at the end of the walk.
    hole_ds: FxIndexMap<NodeId, D>,
    /// A closure argument's parameter dims, from the fn-typed slot it fills
    /// (`f: (px) -> px`), read when the closure walks (r[impl mimas.unit.closure]).
    closure_hints: FxHashMap<NodeId, Vec<D>>,
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
    /// Exprs whose value crosses to the opposite axis convention where it
    /// lands — `40px↓` into a `wpx↑` param, `x↓ + y↑`. Emit negates them
    /// (r[impl mimas.unit.direction.signed]).
    sign_negs: rustc_hash::FxHashSet<NodeId>,
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
            site_ds: FxIndexMap::default(),
            hole_ds: FxIndexMap::default(),
            closure_hints: FxHashMap::default(),
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
            sign_negs: Default::default(),
            report: false,
            errors: Vec::new(),
        };
        pass.index(ast.stmts());
        let member_units = pass.member_units();
        let member_shapes = pass.member_shapes();
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
        let sign_negs = std::mem::take(&mut pass.sign_negs);
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
        // r[impl mimas.display.one] each display site's shape, and the unit
        // the page wrote for every dimension a shape or a struct field names
        let site_ds = std::mem::take(&mut pass.site_ds);
        let print_shapes: Vec<(Location, units::Shape)> = site_ds
            .iter()
            .map(|(loc, (id, d))| (*loc, pass.shape(solver.node_ty(*id).as_ref(), d)))
            .filter(|(_, s)| !s.is_plain())
            .collect();
        let hole_ds = std::mem::take(&mut pass.hole_ds);
        let hole_shapes: Vec<(parse::NodeId, units::Shape)> = hole_ds
            .iter()
            .map(|(id, d)| (*id, pass.shape(solver.node_ty(*id).as_ref(), d)))
            .filter(|(_, s)| !s.is_plain())
            .collect();
        let mut shown_dims: Vec<Dim> = solver.member_units.values().copied().collect();
        for shape in print_shapes.iter().map(|(_, s)| s).chain(hole_shapes.iter().map(|(_, s)| s)) {
            shape.dims_into(&mut shown_dims);
        }
        for (_, d) in &member_units {
            shown_dims.push(*d);
        }
        let shown_units: Vec<(Dim, units::Shown)> = shown_dims
            .into_iter()
            .unique()
            .filter_map(|d| pass.shown(&D::Q(d)).map(|s| (d, s)))
            .collect();
        let scale_of = |pass: &Pass, d: Dim| {
            if d.is_none() {
                1.0
            } else {
                pass.shown(&D::Q(d)).map_or(1.0, |s| s.scale)
            }
        };
        let dec_scales: Vec<(parse::NodeId, f64)> = std::mem::take(&mut pass.dec_notes)
            .into_iter()
            .map(|(id, note)| {
                let k = match note {
                    // the written number in the page's unit for its dimension
                    DecNote::Lit { dim, scale } => scale / scale_of(&pass, dim),
                    // a product or quotient of two page units, in the page's unit
                    // for the result (1 when the page's units agree)
                    DecNote::Fix {
                        mul: true,
                        l,
                        r,
                        out,
                    } => scale_of(&pass, l) * scale_of(&pass, r) / scale_of(&pass, out),
                    DecNote::Fix {
                        mul: false,
                        l,
                        r,
                        out,
                    } => scale_of(&pass, l) / scale_of(&pass, r) / scale_of(&pass, out),
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
        solver.sign_negs.extend(sign_negs);
        solver.want_dims.extend(want_dims);
        solver.print_dims.extend(print_dims);
        solver.member_units.extend(member_units);
        solver.member_shapes.extend(member_shapes);
        solver.print_shapes.extend(print_shapes);
        solver.hole_shapes.extend(hole_shapes);
        solver.shown_units.extend(shown_units);
    }
    Ok(())
}

impl<'a> Pass<'a> {
    /// How an echo of a value in `d` shows its unit: the unit written for the
    /// dimension, else a ratio or product of two written units (`usd/lb`), else
    /// the coherent unit. `None` for a value with no known dimension.
    /// r[impl mimas.echo.unit]
    /// How a value of type `ty` and dim `d` displays where its runtime value
    /// cannot tell: tuple parentheses and units (r[impl mimas.display.one]).
    fn shape(&self, ty: Option<&shared::Ty>, d: &D) -> units::Shape {
        use shared::Ty;
        use units::Shape;
        let d = match d {
            D::Place(inner) => &**inner,
            d => d,
        };
        let q = |d: &D| match d {
            D::Q(dim) if !dim.is_none() && *dim != units::INDEX => Shape::Q(*dim),
            _ => Shape::Plain,
        };
        match ty {
            Some(Ty::Option(inner)) => self.shape(Some(inner), d),
            Some(Ty::Tuple(ts)) => Shape::Tuple(
                ts.iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let di = match d {
                            D::Tuple(ds) => ds.get(i).unwrap_or(&D::Any),
                            _ => &D::Any,
                        };
                        self.shape(Some(t), di)
                    })
                    .collect(),
            ),
            Some(Ty::Array(inner, _)) => {
                let di = match d {
                    D::Arr(x) => &**x,
                    _ => &D::Any,
                };
                Shape::List(Box::new(self.shape(Some(inner), di)))
            }
            Some(Ty::Dict(_, v)) => {
                let di = match d {
                    D::Arr(x) => &**x,
                    _ => &D::Any,
                };
                Shape::Dict(Box::new(self.shape(Some(v), di)))
            }
            Some(Ty::Float | Ty::Int) => q(d),
            Some(_) => Shape::Plain,
            None => match d {
                D::Arr(x) => Shape::List(Box::new(self.shape(None, x))),
                D::Tuple(ds) => Shape::Tuple(ds.iter().map(|x| self.shape(None, x)).collect()),
                d => q(d),
            },
        }
    }

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
                        let name = if an == bn {
                            format!("{an}²")
                        } else {
                            format!("{an}*{bn}")
                        };
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

    /// Each struct field's display shape from its annotation, one unit per
    /// tuple slot: `p: (px, px)` shows `(1 px, 2 px)`
    /// (r[impl mimas.display.one]). Plain shapes are left out.
    fn member_shapes(&mut self) -> Vec<((String, String), units::Shape)> {
        let mut out = Vec::new();
        let structs: Vec<(String, &'a Struct)> =
            self.structs.iter().map(|(n, s)| (n.clone(), *s)).collect();
        for (name, s) in structs {
            for f in &s.fields {
                let key = match &f.name {
                    FieldKey::Ident(k) => k.lexeme.clone(),
                    FieldKey::Int(i) => i.to_string(),
                };
                let tps = s.type_params.iter().map(|i| i.lexeme.clone()).collect();
                let d = self.with_ty_params(tps, |p| p.annotation(&f.annotation));
                let shape = self.shape(None, &d);
                if !shape.is_plain() {
                    out.push(((name.clone(), key), shape));
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

    /// A numeric literal's value, sign included — `-1.0` is `Unary` over
    /// a literal, not a literal itself.
    fn num_lit(e: &Expr) -> Option<f64> {
        let (n, sign) = match e.kind() {
            ExprKind::Literal(Literal::Int(n)) => (*n as f64, 1.0),
            ExprKind::Literal(Literal::Float(n)) => (*n, 1.0),
            ExprKind::Unary(u) => match u.right.kind() {
                ExprKind::Literal(Literal::Int(n)) => (
                    *n as f64,
                    if u.op == UnaryOp::Negative { -1.0 } else { 1.0 },
                ),
                ExprKind::Literal(Literal::Float(n)) => {
                    (*n, if u.op == UnaryOp::Negative { -1.0 } else { 1.0 })
                }
                _ => return None,
            },
            _ => return None,
        };
        Some(n * sign)
    }

    /// A numeric literal landing in a `snorm`/`unorm` slot must sit inside
    /// the range the name promises — `let lx: snorm = 1.5` is a bug the
    /// type spells. Computed values pass for now: folding a runtime check
    /// into the same boundary is a later layer.
    // r[impl mimas.unit.norm]
    fn range_check(&mut self, e: &'a Expr, want: &D) {
        let D::Q(d) = want else { return };
        let Some((name, lo, hi)) = units::norm_bounds(*d) else {
            return;
        };
        let Some(v) = Self::num_lit(e) else { return };
        if v < lo || v > hi {
            self.fail(
                e.location(),
                format!("`{name}` is [{lo}, {hi}], found {v}"),
                format!("out of range: {v}"),
                "a norm channel is a contract, not a clamp — write a value inside it or convert deliberately",
            );
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

    /// `expect` for a value the caller has in hand, with scalar
    /// convention conversion: `40px↓` into a `wpx↑` slot negates at the
    /// boundary rather than failing. The one hard wall is a *place* —
    /// an `inout` slot writes back in its own convention, so an
    /// opposite-signed place can't stand in for it.
    // r[impl mimas.unit.direction.signed]
    fn expect_conv(&mut self, want: &D, got: &D, val: &'a Expr, what: &str) {
        if let D::Place(inner) = got
            && let D::Q(w) | D::Iv(w) = want
            && let Some(g) = inner.dim()
            && (Dim::sign_opposed(*w, g) || Dim::sign_opposed(g, *w))
        {
            self.fail(
                val.location(),
                format!(
                    "{what}: a place holds its own convention ({}), which {} can't adopt",
                    Self::describe(&g),
                    Self::describe(w)
                ),
                format!("this place is {}", Self::describe(&g)),
                "bind the converted value (`let v = -x`) and pass `v` instead",
            );
            return;
        }
        // a scalar crosses conventions where it lands: a signed value
        // reads at its positive end wherever the slot doesn't claim the
        // negative one — `40↓` is −40 under `wpx↕` or `wpx`, and negates
        // into a `wpx↑` slot
        if let (D::Q(w) | D::Iv(w), D::Q(g) | D::Iv(g)) = (want, got)
            && Dim::sign_convertible(*w, *g)
            && Dim::sign_flip(*w, *g)
        {
            self.sign_negs.insert(val.id());
            return;
        }
        // a composite (tuple element, list item) has no single value to
        // negate — signed elements it doesn't share fail outright
        if signed_uncovered(want, got) {
            self.fail(
                val.location(),
                format!(
                    "{what}: expected {}, found {}",
                    self.describe_d(want),
                    self.describe_d(got)
                ),
                "its signed elements can't convert inside a composite".into(),
                "convert the element itself (`let v = -x`) before it goes in",
            );
            return;
        }
        self.expect(want, got, val.location(), what);
    }

    /// A `D`'s own description, for messages that name a whole shape.
    fn describe_d(&self, d: &D) -> String {
        match d {
            D::Q(q) | D::Iv(q) => Self::describe(q),
            D::Arr(i) => format!("[{}]", self.describe_d(i)),
            D::Tuple(parts) => format!(
                "({})",
                parts.iter().map(|p| self.describe_d(p)).collect::<Vec<_>>().join(", ")
            ),
            D::Place(i) => self.describe_d(i),
            _ => "a value".into(),
        }
    }

    /// Two operands that agree but for opposite conventions — `x↓ + y↑`.
    /// Both convert to the positive end, whichever side holds `↓`
    /// negating, and the joined dim reports the positive convention.
    fn conv_pair(
        &mut self,
        l: &D,
        le: Option<&'a Expr>,
        r: &D,
        re: Option<&'a Expr>,
    ) -> Option<(D, D)> {
        let (D::Q(lq) | D::Iv(lq), D::Q(rq) | D::Iv(rq)) = (l, r) else {
            return None;
        };
        if !Dim::sign_convertible(*lq, *rq) {
            return None;
        }
        // no slot opposed: same conventions or an unsigned side adopting —
        // the values add as they stand, keeping their shared convention
        if !Dim::sign_opposed(*lq, *rq) && !Dim::sign_opposed(*rq, *lq) {
            return Some((l.clone(), r.clone()));
        }
        if Dim::sign_opposed(*lq, *rq)
            && let Some(le) = le
        {
            self.sign_negs.insert(le.id());
        }
        if Dim::sign_opposed(*rq, *lq)
            && let Some(re) = re
        {
            self.sign_negs.insert(re.id());
        }
        let norm = |d: &D| match d {
            D::Q(q) => D::Q(q.pos_axes()),
            D::Iv(q) => D::Iv(q.pos_axes()),
            _ => d.clone(),
        };
        Some((norm(l), norm(r)))
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
            Annotation::Applied(ty, args)
                if ty.len() == 1 && ty[0].lexeme == "dec" && args.len() == 1 =>
            {
                self.annotation(&args[0])
            }
            // `inout` lends a place: the binding's reads/writes dim at the
            // element, so the descriptor carries `Place`, like a `&` source
            Annotation::Inout(_, inner) => D::Place(Box::new(self.annotation(inner))),
            Annotation::Option(inner) | Annotation::Result(inner) => self.annotation(inner),
            Annotation::Array(inner, _) => D::Arr(Box::new(self.annotation(inner))),
            // `{px}` and `{str: px}` track their values as `[px]` does
            // (r[impl mimas.unit.dict])
            Annotation::Dictionary(inner) | Annotation::Map(_, inner, _) => {
                match self.annotation(inner) {
                    D::Any => D::Any,
                    d => D::Arr(Box::new(d)),
                }
            }
            Annotation::Tuple(parts) => {
                D::Tuple(parts.iter().map(|p| self.annotation(p)).collect())
            }
            // `(px) -> px`: calling it checks and gives units (r[impl mimas.unit.closure])
            Annotation::Function(params, ret) => D::Fn(
                params.iter().map(|p| self.annotation(p)).collect(),
                Box::new(self.annotation(ret)),
            ),
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
                    self.written
                        .entry(dim)
                        .or_insert((ident.lexeme.clone(), scale));
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
            Annotation::Array(inner, _)
            | Annotation::Option(inner)
            | Annotation::Result(inner)
            | Annotation::Inout(_, inner) => {
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
        if let Some(rewritten) = self.solver.rewrites.get(&stmt.id()) {
            if let StmtKind::Assignment(a) = stmt.kind()
                && matches!(a.op, AssignmentOp::Identity | AssignmentOp::PlusEqual | AssignmentOp::MinusEqual)
            {
                let left = self.expr(&a.left);
                let right = self.expr(&a.right);
                self.expect_conv(&left, &right, &a.right, "assignment doesn't match");
                self.range_check(&a.right, &left);
            }
            self.expr(rewritten);
            return;
        }
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
                        self.expect_conv(&want, &got, &l.right, "declared type doesn't match");
                        self.range_check(&l.right, &want);
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
                self.site_ds.insert(stmt.location(), (l.right.id(), d.clone()));
                self.bind(&l.left, d);
            }
            StmtKind::Assignment(a) => {
                let left = self.expr(&a.left);
                let right = self.expr(&a.right);
                match a.op {
                    AssignmentOp::Identity | AssignmentOp::PlusEqual | AssignmentOp::MinusEqual => {
                        self.expect_conv(&left, &right, &a.right, "assignment doesn't match");
                        self.range_check(&a.right, &left);
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
                // a `print(x)` statement shares its call's location: the
                // print's own site (the shown argument) wins
                self.site_ds.entry(stmt.location()).or_insert((e.id(), d.clone()));
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
                        self.expect_conv(&want, &got, &c.right, "declared type doesn't match");
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
                let _at = tail_location(&f.body);
                s.expect_conv(&ret, &body, tail_expr(&f.body), "returns the wrong dimension");
                s.range_check(tail_expr(&f.body), &ret);
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
            PatKind::Tuple(pats) => {
                // `let (a, b) = (x, y)` hands each name its own slot's
                // dimension; a shape miss degrades like any destructure
                if let D::Tuple(ds) = &d
                    && ds.len() == pats.len()
                {
                    for (p, d) in pats.iter().zip(ds) {
                        self.bind(p, d.clone());
                    }
                    return;
                }
                for p in pats {
                    self.bind(p, D::Any);
                }
            }
            PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
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
    /// leaves degrade to `Any` — except a tuple whose arity matches, which
    /// hands each name its own slot's dimension.
    fn bind_const(&mut self, pat: &Pat, d: D) {
        match pat.kind() {
            PatKind::Ident(ident) => {
                if let Some(dec) = self.dec_of(ident) {
                    self.consts.insert(dec, d);
                }
            }
            PatKind::Tuple(pats) => {
                if let D::Tuple(ds) = &d
                    && ds.len() == pats.len()
                {
                    for (p, d) in pats.iter().zip(ds) {
                        self.bind_const(p, d.clone());
                    }
                    return;
                }
                for p in pats {
                    self.bind_const(p, D::Any);
                }
            }
            PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
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

    fn is_zero(e: &Expr) -> bool {
        match e.kind() {
            ExprKind::Literal(Literal::Int(0)) => true,
            ExprKind::Literal(Literal::Float(f)) => *f == 0.0,
            ExprKind::Grouping(g) => Self::is_zero(&g.inner),
            _ => false,
        }
    }

    fn expr(&mut self, e: &'a Expr) -> D {
        if let Some(source) = self.solver.place_sources.get(&e.id()) {
            return D::Place(Box::new(self.expr(source)));
        }
        if let Some(rewritten) = self.solver.rewrites.get(&e.id()) {
            let d = self.expr(rewritten);
            if let D::Q(dim) = d {
                self.dims.insert(e.id(), dim);
            }
            return d;
        }
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
                    self.env
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
                    // a direction word's wrapper re-claims the axis's
                    // convention: an operand already signed the other way
                    // negates to read in the claimed end
                    UnaryOp::Negative | UnaryOp::Positive => {
                        match (d, self.ast.direction(e.id())) {
                            (D::Q(q), Some(axis)) => {
                                if Dim::sign_opposed(axis, q) || Dim::sign_opposed(q, axis) {
                                    self.sign_negs.insert(u.right.id());
                                }
                                D::Q(axis.with_axes(q))
                            }
                            (D::Iv(q), Some(axis)) => {
                                if Dim::sign_opposed(axis, q) || Dim::sign_opposed(q, axis) {
                                    self.sign_negs.insert(u.right.id());
                                }
                                D::Iv(axis.with_axes(q))
                            }
                            (d, _) => d,
                        }
                    }
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
                // opposite conventions compare fine — `40↑ < 40↓` is
                // "40 above" against "40 below", ordered in up-terms
                if !Self::is_zero(&eq.left)
                    && !Self::is_zero(&eq.right)
                    && let Some((a, b)) = clash(&l, &r)
                    && self
                        .conv_pair(&l, Some(&eq.left), &r, Some(&eq.right))
                        .is_none()
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
                self.expect_conv(&l, &r, &c.right, "`??` fallback has a different dimension");
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
                        self.branches(then, Some(&i.main_body), els, other, other.location())
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
                        Some(prev) => {
                            self.branches(prev, None, d, case.body(), case.body().location())
                        }
                    });
                }
                out.unwrap_or(D::Any)
            }
            ExprKind::Call(c) => self.call(e, c),
            ExprKind::Access(a) => self.access(a),
            // r[impl mimas.unit.closure] a closure's parameters take their
            // annotation, else the dims its slot expects (`ap(|w| …)` with
            // `f: (px) -> px`); its return is its annotation, else its body's
            ExprKind::Closure(c) => {
                let hint = self.closure_hints.get(&e.id()).cloned();
                let mut params = Vec::with_capacity(c.parameters.len());
                for (i, p) in c.parameters.iter().enumerate() {
                    let d = match &p.annotation {
                        Some(a) => self.annotation(a),
                        None => hint.as_ref().and_then(|h| h.get(i).cloned()).unwrap_or(D::Any),
                    };
                    self.bind(&p.left, d.clone());
                    params.push(d);
                }
                let want = c.return_type.as_ref().map(|a| self.annotation(a));
                self.rets.push(want.clone().unwrap_or(D::Any));
                let got = self.expr(&c.body);
                self.rets.pop();
                let ret = match want {
                    Some(want) => {
                        self.expect_conv(&want, &got, &c.body, "the closure returns the wrong dimension");
                        want
                    }
                    None => got,
                };
                D::Fn(params, Box::new(ret))
            }
            ExprKind::For(f) => {
                let it = self.expr(&f.iterator);
                let is_dict = matches!(
                    self.solver.node_ty(f.iterator.id()),
                    Some(Ty::Dict(..))
                );
                let item = match (&it, f.iterator.kind()) {
                    // a dict walks `(key, value)` pairs (r[impl mimas.unit.dict])
                    (D::Arr(inner), _) if is_dict => D::Tuple(vec![D::Any, (**inner).clone()]),
                    (D::Arr(inner), _) => (**inner).clone(),
                    (_, ExprKind::Range(_)) => PLAIN,
                    // a generator is a `() -> T?`: `for` walks its `T`s, so
                    // `subscript ws() -> px` gives `px` items
                    // (r[impl mimas.unit.generator])
                    (D::Fn(params, item), _) if params.is_empty() => (**item).clone(),
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
                let mut d = self.expr(&c.value);
                let prev = self.collects.last_mut().and_then(Option::take);
                let joined = match prev {
                    None => d.clone(),
                    Some(prev) => match self.conv_pair(&prev, None, &d, Some(&c.value)) {
                        Some((p2, d2)) => {
                            d = d2;
                            join(p2, &d)
                        }
                        None => join(prev, &d),
                    },
                };
                if let Some(slot) = self.collects.last_mut() {
                    *slot = Some(joined);
                }
                D::Any
            }
            ExprKind::Return(r) => {
                if let Some(v) = &r.value {
                    let got = self.expr(v);
                    if let Some(want) = self.rets.last().cloned() {
                        self.expect_conv(&want, &got, v, "returns the wrong dimension");
                        self.range_check(v, &want);
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
            ExprKind::Atom(a) => {
                for v in a.payload.values() {
                    self.expr(v);
                }
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
                        self.hole_ds.insert(x.id(), d.clone());
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
                                        && t.trim_start_matches('-')
                                            .starts_with(|c: char| c.is_ascii_digit())
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

    /// Two arms of an `if`/`match` that disagree are a mistake; otherwise
    /// their shared dimension. Opposite conventions normalize to the
    /// positive end, negating whichever arm reads the other way.
    fn branches(&mut self, mut a: D, ae: Option<&'a Expr>, mut b: D, be: &'a Expr, at: Location) -> D {
        // `if c { x↑ } else { y↓ }` gives an `↑`, the `↓` arm negated
        if let Some((a2, b2)) = self.conv_pair(&a, ae, &b, Some(be)) {
            (a, b) = (a2, b2);
        }
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
                        self.expect_conv(&want, &got, value, &format!("field `{}`", k.lexeme));
                        self.range_check(value, &want);
                    }
                }
                D::Any
            }
            // a bare `{ … }` the solver made a struct checks its fields' units
            // like `Name { … }` does r[impl mimas.record.context]
            Literal::Record(fields) => {
                let decl = self.adt_name(e).and_then(|n| self.structs.get(&n).copied());
                let mut gots = Vec::with_capacity(fields.len());
                for (k, value) in fields {
                    let got = self.expr(value);
                    gots.push(got.clone());
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
                        self.expect_conv(&want, &got, value, &format!("field `{}`", k.lexeme));
                        self.range_check(value, &want);
                    }
                }
                // a dict whose values share a unit tracks as a list of it, so
                // `d["a"]` keeps the unit (r[impl mimas.unit.dict])
                if decl.is_none() && !gots.is_empty() {
                    if let Some(D::Q(first)) = gots.first()
                        && gots.iter().all(|d| matches!(d, D::Q(x) if x == first))
                    {
                        return D::Arr(Box::new(D::Q(*first)));
                    }
                }
                D::Any
            }
            Literal::Tuple(items) => D::Tuple(items.iter().map(|i| self.expr(i)).collect()),
            _ => D::Any,
        }
    }

    fn evaluation(&mut self, e: &'a Expr, op: EvaluationOp, mut l: D, mut r: D) -> D {
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
                // `x↓ + y↑` — both convert to the positive end, the `↓`
                // side negating; the result reports the positive convention
                if let ExprKind::Evaluation(ev) = e.kind()
                    && let Some((l2, r2)) =
                        self.conv_pair(&l, Some(&ev.left), &r, Some(&ev.right))
                {
                    (l, r) = (l2, r2);
                }
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
                // a signed factor reads at its positive end in a product:
                // `40↓ · 2` is −80 in ↑-terms, `x↓ · y↓` composes +80
                if let ExprKind::Evaluation(ev) = e.kind() {
                    if l.dim().is_some_and(Dim::has_neg_end) {
                        self.sign_negs.insert(ev.left.id());
                    }
                    if r.dim().is_some_and(Dim::has_neg_end) {
                        self.sign_negs.insert(ev.right.id());
                    }
                }
                let interval = matches!(l, D::Iv(_)) || matches!(r, D::Iv(_));
                match (
                    l.dim().map(Dim::pos_axes),
                    r.dim().map(Dim::pos_axes),
                ) {
                    (Some(a), Some(b)) => {
                        let d = if op == EvaluationOp::Multiply {
                            a.mul(b)
                        } else {
                            a.div(b)
                        };
                        // `~/` (and `÷`'s quotient half) is integer division:
                        // a quotient of indexes stays an `index` even though
                        // the dim algebra cancels it to a plain number.
                        // r[impl mimas.index.div]
                        let d = if op == EvaluationOp::Div
                            && (a.bare() == units::INDEX || b.bare() == units::INDEX)
                        {
                            units::INDEX
                        } else {
                            d
                        };
                        if self.is_dec(e) && !(a.is_none() && b.is_none()) {
                            let mul = op == EvaluationOp::Multiply;
                            self.dec_notes.insert(
                                e.id(),
                                DecNote::Fix {
                                    mul,
                                    l: a,
                                    r: b,
                                    out: d,
                                },
                            );
                        }
                        if interval { D::Iv(d) } else { D::Q(d) }
                    }
                    _ => D::Any,
                }
            }
            // `a ≈ b` compares, so its sides must measure the same thing, as
            // `==`'s must (r[impl mimas.unit.dimension])
            EvaluationOp::Approx => {
                if let ExprKind::Evaluation(ev) = e.kind()
                    && !Self::is_zero(&ev.left)
                    && !Self::is_zero(&ev.right)
                    && let Some((a, b)) = clash(&l, &r)
                    && self.conv_pair(&l, Some(&ev.left), &r, Some(&ev.right)).is_none()
                {
                    self.fail(
                        e.location(),
                        format!(
                            "cannot compare {} and {}",
                            self.describe_operand(&ev.left, &a),
                            self.describe_operand(&ev.right, &b)
                        ),
                        format!(
                            "{} against {}",
                            self.describe_operand(&ev.left, &a),
                            self.describe_operand(&ev.right, &b)
                        ),
                        "quantities are only comparable in the same dimension; convert one side with `.to(unit)`",
                    );
                }
                D::Any
            }
            // `a ⋅ b` on two lists of quantities: the sum of products, worth
            // the product of the item units (r[impl mimas.glyph.dot.list])
            EvaluationOp::Dot => match (&l, &r) {
                (D::Arr(a), D::Arr(b)) => match (&**a, &**b) {
                    (D::Q(x), D::Q(y)) => D::Q(x.mul(*y)),
                    _ => D::Any,
                },
                _ => D::Any,
            },
            _ => D::Any,
        }
    }

    // ---- access and calls

    /// Whether the solver typed `e` as a `dec`.
    fn is_dec(&self, e: &Expr) -> bool {
        self.solver
            .node_to_vid
            .get(&e.id())
            .is_some_and(|v| self.solver.is_dec(&Ty::Vid(*v)))
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
                // `p.0`: a tuple's own slot, its own dimension
                if let ExprKind::Literal(Literal::Int(i)) = right.kind()
                    && let D::Tuple(ds) = &recv
                {
                    return ds.get(*i as usize).cloned().unwrap_or(D::Any);
                }
                if let ExprKind::Ident(field) = right.kind() {
                    // `p.x`/`p.y`/`p.z`/`p.t`: axis names spell tuple slots,
                    // so `(px↔, px↕)` keeps its tags through `p.x`/`p.y`.
                    // r[impl mimas.tuple.axis]
                    if let D::Tuple(ds) = &recv
                        && let Some(i) = ["x", "y", "z", "t"]
                            .iter()
                            .position(|n| *n == field.lexeme.as_str())
                    {
                        return ds.get(i).cloned().unwrap_or(D::Any);
                    }
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
                        if let Some(d) = self.solver.field_dims.get(&(ty.clone(), field.lexeme.clone())) {
                            return D::Q(*d);
                        }
                        // a member generator read as a field, `b.wide`: worth
                        // its `() -> px?`, so `for w in b.wide` gives `px`
                        // (r[impl mimas.unit.generator])
                        if let Some(f) = self.methods.get(&(ty.clone(), field.lexeme.clone())).copied()
                            && let Some(ret) = &f.return_type
                            && let D::Fn(ps, item) = self.with_ty_params(self.adt_param_names(&ty), |s| s.annotation(ret))
                            && ps.is_empty()
                        {
                            return D::Fn(ps, item);
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
            // a std combinator's closure learns its parameters from the
            // list's items (r[impl mimas.unit.combinator])
            let mut args: Vec<D> = Vec::with_capacity(c.arguments.len());
            for a in &c.arguments {
                if matches!(a.value.kind(), ExprKind::Closure(_))
                    && let D::Arr(item) = &recv
                {
                    let item = (**item).clone();
                    let hint = match method.lexeme.as_str() {
                        "map" | "filter" | "find" | "any" | "all" | "flat_map" | "each"
                        | "sort_by" | "min_by" | "max_by" | "sorted_by" | "count_by" => {
                            Some(vec![item])
                        }
                        "fold" => args.first().map(|init| vec![init.clone(), item]),
                        "mapi" => Some(vec![PLAIN, item]),
                        "foldi" => args.first().map(|init| vec![PLAIN, init.clone(), item]),
                        _ => None,
                    };
                    if let Some(h) = hint {
                        self.closure_hints.insert(a.value.id(), h);
                    }
                }
                args.push(self.expr(&a.value));
            }
            if let D::Place(element) = &recv {
                if method.lexeme == "read" {
                    return (**element).clone();
                }
                if method.lexeme == "write" && let Some(value) = args.first() {
                    self.expect_conv(
                        element, value, &c.arguments[0].value,
                        "place assignment doesn't match",
                    );
                    self.range_check(&c.arguments[0].value, element);
                    return D::Any;
                }
            }
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
        let user = match c.left.kind() {
            ExprKind::Ident(callee) => self.dec_of(callee).and_then(|dec| {
                let loc = self.solver.decs[dec].location;
                self.fns_by_site
                    .get(&(loc.file_id, loc.span.start))
                    .copied()
            }),
            _ => None,
        };
        if let Some(f) = user {
            self.hint_closures(f, c, false, None);
        }
        let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
        if let ExprKind::Ident(callee) = c.left.kind() {
            // `a ÷ b` is the prelude's `__op_quot_rem(a, b)` (L10's reserved
            // name; `quot_rem(a, b)` written out is the same): a whole
            // quotient worth `a / b` and a remainder worth `a`
            // (r[impl mimas.unit.combinator])
            if let Some(f) = user
                && matches!(callee.lexeme.as_str(), "__op_quot_rem" | "quot_rem")
                && f.parameters.iter().all(|p| p.annotation.is_none())
                && let [D::Q(a), D::Q(b)] = args.as_slice()
            {
                return D::Tuple(vec![D::Q(a.div(*b)), D::Q(*a)]);
            }
            if let Some(f) = user {
                return self.apply(f, c, &args, false, None);
            }
            // a binding that holds a function: a closure, a fn-typed
            // parameter (r[impl mimas.unit.closure])
            if let Some(dec) = self.dec_of(callee)
                && let Some(D::Fn(params, ret)) = self.env.get(&dec).cloned()
            {
                for (i, (arg, got)) in c.arguments.iter().zip(&args).enumerate() {
                    if let Some(want) = params.get(i) {
                        self.expect_conv(
                            want,
                            got,
                            &arg.value,
                            &format!("argument {} of `{}`", i + 1, callee.lexeme),
                        );
                    }
                }
                return *ret;
            }
            if let Some(d) = self.aggregate(e, c, callee, &args) {
                return d;
            }
        } else if let D::Fn(params, ret) = self.expr(&c.left) {
            // a call of a function value: `twice(f)(x)` (r[impl mimas.unit.closure])
            for (i, (arg, got)) in c.arguments.iter().zip(&args).enumerate() {
                if let Some(want) = params.get(i) {
                    self.expect_conv(want, got, &arg.value, &format!("argument {}", i + 1));
                }
            }
            return *ret;
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
        // r[impl mimas.display.one] the shown value's shape, for every print
        if let ExprKind::Ident(_) = c.left.kind()
            && matches!(
                self.solver.decs[dec].name.as_str(),
                "print" | "dbg" | "warn"
            )
            && let (Some(arg), Some(d)) = (c.arguments.first(), args.first())
        {
            self.site_ds.insert(e.location(), (arg.value.id(), d.clone()));
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
            if let Some(d) = want.scalar()
                && !d.is_none()
            {
                self.want_dims.insert(arg.value.id(), d);
            }
            let want = dim_tree(want);
            self.expect_conv(
                &want,
                got,
                &arg.value,
                &format!("argument {} of `{name}`", i + 1),
            );
            self.range_check(&arg.value, &want);
        }
        Some(binding.sig.return_dim.as_ref().map_or(D::Any, dim_tree))
    }

    /// Before a call's arguments walk: each closure argument whose parameter is
    /// fn-typed learns that type's parameter dims (r[impl mimas.unit.closure]).
    fn hint_closures(&mut self, f: &'a Function, c: &'a parse::Call, method: bool, owner: Option<&str>) {
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
            for (i, arg) in c.arguments.iter().enumerate() {
                if !matches!(arg.value.kind(), ExprKind::Closure(_)) {
                    continue;
                }
                let param = match &arg.name {
                    Some(n) => params
                        .iter()
                        .find(|p| p.left.as_ident().is_some_and(|id| id.lexeme == n.lexeme)),
                    None => params.get(i),
                };
                if let Some(ann) = param.and_then(|p| p.annotation.as_ref())
                    && let D::Fn(ps, _) = s.annotation(ann)
                {
                    s.closure_hints.insert(arg.value.id(), ps);
                }
            }
        });
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
        // r[impl mimas.unit.generic] a type parameter takes the unit of the
        // first argument it meets, and every later use must agree
        let tps: Vec<String> = f.type_params.iter().map(|i| i.lexeme.clone()).collect();
        let mut tmap: FxHashMap<String, D> = FxHashMap::default();
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
                let want = s.tparam_want(ann, got, &mut tmap, &tps);
                if let Some(d) = want.dim()
                    && !d.is_none()
                {
                    s.want_dims.insert(arg.value.id(), d);
                }
                let pname = p.left.as_ident().map_or("argument", |i| i.lexeme.as_str());
                s.expect_conv(
                    &want,
                    got,
                    &arg.value,
                    &format!("argument `{pname}` of `{}`", f.name.lexeme),
                );
                s.range_check(&arg.value, &want);
            }
            f.return_type.as_ref().map_or(D::Any, |a| s.tparam_subst(a, &tmap, &tps))
        })
    }

    /// The dim a parameter declared `ann` asks of an argument worth `got`,
    /// binding each type parameter it names to the first unit it meets
    /// (r[impl mimas.unit.generic]).
    fn tparam_want(&self, ann: &Annotation, got: &D, tmap: &mut FxHashMap<String, D>, tps: &[String]) -> D {
        let got = match got {
            D::Place(inner) => &**inner,
            g => g,
        };
        match ann {
            Annotation::Ty(id) if tps.contains(&id.lexeme) => match tmap.get(&id.lexeme) {
                Some(d) => d.clone(),
                None => {
                    if matches!(got, D::Q(_) | D::Arr(_) | D::Tuple(_) | D::Fn(..)) {
                        tmap.insert(id.lexeme.clone(), got.clone());
                    }
                    D::Any
                }
            },
            Annotation::Option(inner) | Annotation::Result(inner) => {
                self.tparam_want(inner, got, tmap, tps)
            }
            Annotation::Array(inner, _) => {
                let g = match got {
                    D::Arr(g) => &**g,
                    _ => &D::Any,
                };
                D::Arr(Box::new(self.tparam_want(inner, g, tmap, tps)))
            }
            Annotation::Tuple(parts) => D::Tuple(
                parts
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let g = match got {
                            D::Tuple(gs) => gs.get(i).unwrap_or(&D::Any),
                            _ => &D::Any,
                        };
                        self.tparam_want(p, g, tmap, tps)
                    })
                    .collect(),
            ),
            Annotation::Function(ps, r) => D::Fn(
                ps.iter().map(|p| self.tparam_subst(p, tmap, tps)).collect(),
                Box::new(self.tparam_subst(r, tmap, tps)),
            ),
            _ => self.annotation(ann),
        }
    }

    /// `ann` with each bound type parameter replaced by its unit, for a
    /// generic function's return (r[impl mimas.unit.generic]).
    fn tparam_subst(&self, ann: &Annotation, tmap: &FxHashMap<String, D>, tps: &[String]) -> D {
        match ann {
            Annotation::Ty(id) if tps.contains(&id.lexeme) => {
                tmap.get(&id.lexeme).cloned().unwrap_or(D::Any)
            }
            Annotation::Option(inner) | Annotation::Result(inner) => self.tparam_subst(inner, tmap, tps),
            Annotation::Array(inner, _) => D::Arr(Box::new(self.tparam_subst(inner, tmap, tps))),
            Annotation::Tuple(parts) => {
                D::Tuple(parts.iter().map(|p| self.tparam_subst(p, tmap, tps)).collect())
            }
            Annotation::Function(ps, r) => D::Fn(
                ps.iter().map(|p| self.tparam_subst(p, tmap, tps)).collect(),
                Box::new(self.tparam_subst(r, tmap, tps)),
            ),
            _ => self.annotation(ann),
        }
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
                self.expect_conv(
                    first,
                    a,
                    &c.arguments[i].value,
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
    fn aggregate(
        &mut self,
        e: &'a Expr,
        c: &'a parse::Call,
        callee: &Ident,
        args: &[D],
    ) -> Option<D> {
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
                    self.expect_conv(
                        &D::Q(*d),
                        a,
                        &c.arguments[0].value,
                        "`contains` needs the same dimension",
                    );
                }
                PLAIN
            }
            (D::Iv(d), "overlaps") => {
                if let Some(a) = args.first() {
                    self.expect_conv(
                        &D::Iv(*d),
                        a,
                        &c.arguments[0].value,
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
                    self.expect_conv(
                        &recv,
                        a,
                        &c.arguments[0].value,
                        &format!("`{name}` needs the same dimension"),
                    );
                }
                D::Q(*d)
            }
            (D::Q(d), "clamp") => {
                for (a, arg) in args.iter().zip(&c.arguments) {
                    self.expect_conv(
                        &recv,
                        a,
                        &arg.value,
                        "`clamp` bounds need the same dimension",
                    );
                }
                D::Q(*d)
            }
            (D::Q(d), "sqrt") if angle_power(d.bare()) => PLAIN,
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
            // `y.atan2(x)`, aliased `angle_to`: the two legs must measure
            // the same thing — atan2(3px, 4px) is an angle, atan2(3px, 4s)
            // is a bug (a vector's direction exists only when both legs
            // live in the same space; px-vs-s legs change value with the
            // units chosen). The legs carry perpendicular axes by design —
            // `atan2(dy, dx)` reads an (x, y) vector — so only the bare
            // dims compare; the parities are expected to differ.
            (D::Q(d), "atan2" | "angle_to") => {
                if let Some(D::Q(g)) = args.first() {
                    self.expect(
                        &D::Q(d.bare()),
                        &D::Q(g.bare()),
                        c.arguments[0].value.location(),
                        &format!("`{name}` needs both legs in the same dimension"),
                    );
                }
                D::Q(units::ANGLE)
            }
            (D::Q(d), "exp" | "ln") => {
                if !(d.is_none() || angle_power(d.bare())) {
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
                        // `x↓.to("px↑")` — the strip reads in the named
                        // convention, so the number negates too. An
                        // unsigned value reads positive-end, so
                        // `x↕.to("px↓")` flips as well.
                        Some((ud, _))
                            if Dim::sign_convertible(ud, *d)
                                && Dim::read_flip(ud, *d) =>
                        {
                            self.sign_negs.insert(e.id());
                        }
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
                // so it must measure the same thing. Same check as the string
                // path above — axes adopt, `px↔.to(1px)` reads back the
                // magnitude. A `str` arrives `Any` — this fires only on
                // known numbers.
                // r[impl mimas.unit.to]
                if let Some(D::Q(ud)) = args.first() {
                    if Dim::sign_convertible(*ud, *d) && Dim::read_flip(*ud, *d) {
                        self.sign_negs.insert(e.id());
                    } else if !Dim::agree(*ud, *d) {
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
                }
                PLAIN
            }
            (D::Arr(inner), "push") => {
                if let Some(a) = args.first() {
                    self.expect_conv(inner, a, &c.arguments[0].value, "list item");
                }
                D::Any
            }
            (D::Arr(inner), "pop" | "remove" | "get" | "max" | "min" | "sum") => (**inner).clone(),
            // `⨅ xs` and `⨆ xs` keep the items' unit (r[impl mimas.unit.combinator])
            (D::Arr(inner), "inf" | "sup" | "first" | "last") => (**inner).clone(),
            (D::Arr(_), "filter" | "sorted" | "reversed" | "sorted_by" | "shuffled") => recv.clone(),
            (D::Arr(inner), "find" | "min_by" | "max_by") => (**inner).clone(),
            // `xs.map(f)` is a list of what `f` returns
            (D::Arr(_), "map" | "mapi") => match args.first() {
                Some(D::Fn(_, ret)) => D::Arr(ret.clone()),
                _ => D::Any,
            },
            // `xs.fold(init, f)`: `f` returns the accumulator's unit
            (D::Arr(_), "fold" | "foldi") => match (args.first(), args.get(1)) {
                (Some(init), Some(D::Fn(_, ret))) => {
                    if let Some(a) = c.arguments.get(1) {
                        self.expect_conv(init, ret, &a.value, "`fold`'s step returns the wrong dimension");
                    }
                    init.clone()
                }
                (Some(init), _) => init.clone(),
                _ => D::Any,
            },
            (D::Arr(_), "len") => PLAIN,
            // `a ⋅ b` on two lists: the sum of products, worth the product
            // of the item units (r[impl mimas.glyph.dot.list])
            (D::Arr(a), "dot") => match (&**a, args.first()) {
                (D::Q(x), Some(D::Arr(b))) => match &**b {
                    D::Q(y) => D::Q(x.mul(*y)),
                    _ => D::Any,
                },
                _ => D::Any,
            },
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

/// The expression a block's value comes out of — the tail `tail_location`
/// points at, as an `Expr` so checks can read its kind.
fn tail_expr(body: &Expr) -> &Expr {
    if let ExprKind::Block(b) = body.kind()
        && let Some(y) = &b.yielded_expr
    {
        return tail_expr(y);
    }
    body
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
