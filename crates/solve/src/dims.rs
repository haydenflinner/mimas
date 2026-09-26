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

use std::collections::HashMap;

use miette::NamedSource;
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
    units::Dim,
};
use crate::components::TyExt;

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

/// Two known dimensions that differ; `None` when either is unknown or they agree.
fn clash(a: &D, b: &D) -> Option<(Dim, Dim)> {
    match (a, b) {
        (D::Q(x) | D::Iv(x), D::Q(y) | D::Iv(y)) if x != y => Some((*x, *y)),
        (D::Arr(x), D::Arr(y)) => clash(x, y),
        _ => None,
    }
}

/// The dimension two agreeing values share (unknown wins, so a doubt propagates as a doubt).
fn join(a: D, b: &D) -> D {
    match (&a, b) {
        (D::Q(x), D::Q(y)) | (D::Iv(x), D::Iv(y)) if x == y => a,
        (D::Arr(x), D::Arr(y)) => D::Arr(Box::new(join((**x).clone(), y))),
        _ => D::Any,
    }
}

struct Pass<'a> {
    solver: &'a Solver,
    ast: &'a Ast,
    /// Struct declarations by name (top level; the first of a name wins).
    structs: HashMap<String, &'a Struct>,
    /// Free functions by name, and methods / associated functions by (type, name).
    fns: HashMap<String, &'a Function>,
    methods: HashMap<(String, String), &'a Function>,
    /// A function's declaration site, so a call resolved by the solver finds its signature.
    fns_by_site: HashMap<(usize, usize), &'a Function>,
    /// Declared enum names -- needed for `unit_of` to tell `enum A` from the ampere.
    enum_names: std::collections::HashSet<String>,
    /// Generic type-param names in scope (`fn f<T>`, `struct P<A, B>` fields read at a literal,
    /// an impl's target params). They name types, not units, so `unit_of` must not see them.
    ty_params: Vec<Vec<String>>,
    env: HashMap<DecId, D>,
    consts: HashMap<DecId, D>,
    /// The declared return of each function being walked.
    rets: Vec<D>,
    report: bool,
    error: Option<miette::Report>,
}

pub(crate) fn check(solver: &Solver, asts: &[&Ast]) -> Result<()> {
    for ast in asts {
        let mut pass = Pass {
            solver,
            ast,
            structs: HashMap::new(),
            fns: HashMap::new(),
            methods: HashMap::new(),
            fns_by_site: HashMap::new(),
            enum_names: Default::default(),
            ty_params: Vec::new(),
            env: HashMap::new(),
            consts: HashMap::new(),
            rets: Vec::new(),
            report: false,
            error: None,
        };
        pass.index(ast.stmts());
        // two quiet rounds settle constants that build on each other in any order, then the
        // real one reports
        for _ in 0..2 {
            pass.run(ast.stmts());
        }
        pass.report = true;
        pass.env.clear();
        pass.run(ast.stmts());
        if let Some(e) = pass.error {
            return Err(e);
        }
    }
    Ok(())
}

impl<'a> Pass<'a> {
    // ---- indexing

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
        if !self.report || self.error.is_some() {
            return;
        }
        self.error = Some(
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
            Annotation::Option(inner) | Annotation::Result(inner) => self.annotation(inner),
            Annotation::Array(inner) => D::Arr(Box::new(self.annotation(inner))),
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
                let got = self.expr(&l.right);
                if let Some(else_branch) = &l.else_branch {
                    self.expr(else_branch);
                }
                let d = match &l.annotation {
                    Some(a) => {
                        let want = self.annotation(a);
                        self.expect(&want, &got, l.right.location(), "declared type doesn't match");
                        want
                    }
                    None => got,
                };
                self.bind(&l.left, d);
            }
            StmtKind::Assignment(a) => {
                let left = self.expr(&a.left);
                let right = self.expr(&a.right);
                match a.op {
                    AssignmentOp::Identity | AssignmentOp::PlusEqual | AssignmentOp::MinusEqual => {
                        self.expect(&left, &right, a.right.location(), "assignment doesn't match");
                    }
                    _ => {}
                }
            }
            StmtKind::Expr(e) => {
                self.expr(e);
            }
            StmtKind::Item(item) => self.item(item),
            StmtKind::Module(_) => {}
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
                        self.expect(&want, &got, c.right.location(), "declared type doesn't match");
                        want
                    }
                    None => got,
                };
                if let Some(dec) = self.dec_of(&c.left) {
                    self.consts.insert(dec, d);
                }
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
        match e.kind() {
            ExprKind::Literal(l) => self.literal(e, l),
            ExprKind::Ident(ident) => match self.dec_of(ident) {
                Some(dec) => self
                    .env
                    .get(&dec)
                    .or_else(|| self.consts.get(&dec))
                    .cloned()
                    .unwrap_or(D::Any),
                None => D::Any,
            },
            ExprKind::Grouping(g) => self.expr(&g.inner),
            ExprKind::Unary(u) => {
                let d = self.expr(&u.right);
                match u.op {
                    UnaryOp::Negative | UnaryOp::Positive => d,
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
                if !Self::is_zero(&eq.left) && !Self::is_zero(&eq.right) {
                    if let Some((a, b)) = clash(&l, &r) {
                        let verb = match eq.op {
                            EqualityOp::Equal | EqualityOp::NotEqual => "compare",
                            _ => "order",
                        };
                        self.fail(
                            e.location(),
                            format!("cannot {verb} {} and {}", Self::describe(&a), Self::describe(&b)),
                            format!("{} against {}", Self::describe(&a), Self::describe(&b)),
                            "quantities are only comparable in the same dimension; convert one side with `.to(unit)`",
                        );
                    }
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
                self.expect(&l, &r, c.right.location(), "`??` fallback has a different dimension");
                l
            }
            ExprKind::Unwrap(u) => self.expr(&u.expr),
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
                self.expr(&f.body);
                D::Any
            }
            ExprKind::While(w) => {
                self.expr(&w.header);
                if let Some(b) = &w.binding {
                    self.bind(b, D::Any);
                }
                self.expr(&w.body);
                D::Any
            }
            ExprKind::Loop(l) => {
                self.expr(&l.body);
                D::Any
            }
            ExprKind::Collect(c) => {
                self.expr(&c.value);
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
                for part in &f.parts {
                    if let FStringPart::Expr(x) = part {
                        self.expr(x);
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
                format!("branches disagree: {} and {}", Self::describe(&x), Self::describe(&y)),
                format!("this branch is {}", Self::describe(&y)),
                "every branch of an `if` or `match` has to give the same dimension",
            );
        }
        join(a, &b)
    }

    fn literal(&mut self, e: &'a Expr, l: &'a Literal) -> D {
        match l {
            Literal::Float(_) => self.ast.quantity(e.id()).map_or(PLAIN, D::Q),
            Literal::Int(_) => PLAIN,
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
                for (key, value) in &s.fields {
                    let got = self.expr(value);
                    let (Some(decl), FieldKey::Ident(k)) = (decl, key) else {
                        continue;
                    };
                    let field = decl.fields.iter().find(
                        |f| matches!(&f.name, FieldKey::Ident(i) if i.lexeme == k.lexeme),
                    );
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
            Literal::Dictionary(fields) => {
                for (_, v) in fields {
                    self.expr(v);
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
        let name = |op| match op {
            EvaluationOp::Plus => "add",
            EvaluationOp::Minus => "subtract",
            _ => "combine",
        };
        match op {
            EvaluationOp::Plus | EvaluationOp::Minus | EvaluationOp::Modulo => {
                if let Some((a, b)) = clash(&l, &r) {
                    self.fail(
                        e.location(),
                        format!("cannot {} {} and {}", name(op), Self::describe(&a), Self::describe(&b)),
                        format!("{} with {}", Self::describe(&a), Self::describe(&b)),
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
                        let d = if op == EvaluationOp::Multiply { a.mul(b) } else { a.div(b) };
                        if interval { D::Iv(d) } else { D::Q(d) }
                    }
                    _ => D::Any,
                }
            }
            _ => D::Any,
        }
    }

    // ---- access and calls

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
                    if let D::Iv(d) = recv {
                        if matches!(field.lexeme.as_str(), "lo" | "mid" | "hi") {
                            return D::Q(d);
                        }
                    }
                    if let Some(ty) = self.adt_name(left) {
                        if let Some(decl) = self.structs.get(&ty).copied() {
                            if let Some(f) = decl.fields.iter().find(
                                |f| matches!(&f.name, FieldKey::Ident(i) if i.lexeme == field.lexeme),
                            ) {
                                return self.with_ty_params(
                                    decl.type_params.iter().map(|i| i.lexeme.clone()).collect(),
                                    |s| s.annotation(&f.annotation),
                                );
                            }
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
        if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind() {
            if let ExprKind::Ident(method) = right.kind() {
                let recv = self.expr(left);
                let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
                if let Some(ty) = self.adt_name(left) {
                    if let Some(f) = self
                        .methods
                        .get(&(ty.clone(), method.lexeme.clone()))
                        .copied()
                    {
                        return self.apply(f, c, &args, true, Some(&ty));
                    }
                }
                return self.builtin(e, &method.lexeme, recv, c, &args);
            }
        }
        // `Type::assoc(...)`
        if let ExprKind::Access(Access::DoubleColon { left, right }) = c.left.kind() {
            let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
            if let ExprKind::Ident(ty) = left.kind() {
                if let Some(f) = self.methods.get(&(ty.lexeme.clone(), right.lexeme.clone())).copied() {
                    return self.apply(f, c, &args, false, Some(&ty.lexeme));
                }
                if ty.lexeme == "Interval" {
                    return self.interval_ctor(&right.lexeme, c, &args);
                }
            }
            return D::Any;
        }
        let args: Vec<D> = c.arguments.iter().map(|a| self.expr(&a.value)).collect();
        if let ExprKind::Ident(callee) = c.left.kind() {
            let user = self.dec_of(callee).and_then(|dec| {
                let loc = self.solver.decs[dec].location;
                self.fns_by_site.get(&(loc.file_id, loc.span.start)).copied()
            });
            if let Some(f) = user {
                return self.apply(f, c, &args, false, None);
            }
        } else {
            self.expr(&c.left);
        }
        D::Any
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
                self.expect(first, a, c.arguments[i].value.location(), &format!("`Interval::{name}` bounds need the same dimension"));
            }
        }
        if name == "pm" {
            if let Some(rel) = args.get(1) {
                self.expect(&PLAIN, rel, c.arguments[1].value.location(), "`Interval::pm` takes a plain fraction");
            }
        }
        match first {
            D::Q(d) => D::Iv(*d),
            _ => D::Any,
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
                    self.expect(&D::Q(*d), a, c.arguments[0].value.location(), "`contains` needs the same dimension");
                }
                PLAIN
            }
            (D::Iv(d), "overlaps") => {
                if let Some(a) = args.first() {
                    self.expect(&D::Iv(*d), a, c.arguments[0].value.location(), "`overlaps` needs the same dimension");
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
            (D::Q(d), "abs" | "round" | "floor" | "ceil" | "to_int" | "to_float" | "neg") => D::Q(*d),
            (D::Q(_), "signum") => PLAIN,
            (D::Q(d), "min" | "max" | "hypot") => {
                if let Some(a) = args.first() {
                    self.expect(&recv, a, c.arguments[0].value.location(), &format!("`{name}` needs the same dimension"));
                }
                D::Q(*d)
            }
            (D::Q(d), "clamp") => {
                for (a, arg) in args.iter().zip(&c.arguments) {
                    self.expect(&recv, a, arg.value.location(), "`clamp` bounds need the same dimension");
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
            (D::Q(d), "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "exp" | "ln") => {
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
                if let Some(ExprKind::Literal(Literal::String(u))) = c.arguments.first().map(|a| a.value.kind()) {
                    match units::parse(u) {
                        Some((ud, _)) if ud != *d => self.fail(
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
    if let ExprKind::Block(b) = body.kind() {
        if let Some(y) = &b.yielded_expr {
            return tail_location(y);
        }
    }
    body.location()
}

#[allow(dead_code)]
fn _unused(_: NamedSource<std::sync::Arc<str>>) {}
