//! `datalog { … }` — a bottom-up Datalog block, lowered to plain Mimas.
//!
//! ```text
//! let (edge_, path_) = datalog {
//!     rel edge(a: int, b: int) from pairs
//!     rel path(a: int, b: int)
//!     edge(9, 10)                                  // a fact
//!     path(a, b) if edge(a, b)                     // rules
//!     path(a, c) if edge(a, b), path(b, c), a != c
//! };
//! ```
//!
//! * `rel name(field: Type, …) [from <expr>]` declares a relation; `from` seeds it with an
//!   array of tuples. A relation is declared before any fact or rule names it.
//! * A fact is `name(e, …)`: its arguments are ordinary expressions over the enclosing scope.
//! * A rule is `head(e, …) if item, item, …`, one per line (the body may continue on the next
//!   line after a `,`, and the `if` may start a line). A body item is an *atom* `rel(arg, …)`
//!   or a bool *condition*. In an atom a bare name is a rule variable: its first occurrence
//!   binds, later ones filter by equality; `_` is a wildcard, and any other expression (a
//!   literal, `(outer)`) filters by equality. A condition sees the variables of the atoms to
//!   its left (plus the enclosing scope). The head's expressions see the body's variables; a
//!   bare-name head argument must be one of them.
//!
//! The language follows Ascent's (the Rust Datalog crate), run in mimas so rules can call the
//! page's own functions: `relation edge(int, int)` declares positionally, `<--` spells `if`,
//! and a body may also hold `if cond`, `let x = e`, `for x in e` (one derivation per
//! element), a negated atom `!rel(args)` / `not rel(args)` (every variable already bound), and
//! an aggregate `agg x = f(e) in rel(args)` with `f` one of `count()`, `sum`, `min`, `max`,
//! `mean` (the atom's new names bind per tuple, for `e` only; over no tuples only `count`
//! derives). Rules are stratified like Ascent's: a head sits at least as high as every
//! relation its body reads, and strictly higher than one it negates or aggregates, so that
//! relation is complete when asked; a relation depending on its own negation or aggregate is
//! an error. Strata run in order, each its own fixpoint. Not yet: lattices, `if let`, and
//! several heads per rule.
//!
//! The block evaluates every rule to a fixpoint (semi-naive, set semantics) and yields the
//! relations in declaration order: a tuple of arrays, each an array of tuples of the declared
//! field types. Mimas has no 1-tuples, so a one-field relation is an array of bare values, and
//! a block declaring a single relation yields that array alone.
//!
//! Lowering (no new runtime): each relation is an array `__dl_r: [(T, …)]`. Seeds, facts and
//! atom-free rules run once; then a `loop` runs rounds until no relation grows. A round
//! snapshots `__dl_hi_r = __dl_r.len()`; tuples `[__dl_lo_r, __dl_hi_r)` are the last round's
//! delta. A rule is one nest of `for` loops over its atoms (`0 .. hi`), with the conditions
//! as `if`s where they're written; the last atom only scans the delta unless an earlier atom
//! already picked a delta tuple, so each round only derives from combinations that touch
//! something new. A derived tuple is pushed unless it's already there: each relation keeps a
//! dict `__dl_x_r` bucketing its tuples by `f"{tuple}"`, so the check is a hash lookup plus a
//! scan of the (almost always one-tuple) bucket. A round costs about its derivations.

use super::*;

struct Rel {
    name: Ident,
    fields: Vec<(Ident, Annotation)>,
    seed: Option<Expr>,
}

enum Arg {
    Wild(Location),
    Var(Ident),
    Expr(Expr),
}

struct Atom {
    rel: usize,
    args: Vec<Arg>,
}

enum Item {
    Atom(Atom),
    Cond(Expr),
    /// `let x = e` -- binds `x` for the items to its right and the head.
    Let(Ident, Expr),
    /// `for x in e` -- one derivation per element of `e`.
    For(Ident, Expr),
    /// `!rel(args)` -- no tuple of `rel` matches. `rel` sits in a lower stratum, so it's
    /// complete when this asks.
    Neg(Atom),
    /// `agg x = f(e) in rel(args)` -- `f` (count/sum/min/max/mean) over the matching tuples
    /// of `rel`, also from a lower stratum.
    Agg {
        var: Ident,
        func: Ident,
        arg: Option<Expr>,
        atom: Atom,
    },
}

impl Item {
    /// The relation this item reads, and whether it needs it complete (negation and
    /// aggregates) rather than growing alongside the rule (a positive atom).
    fn reads(&self) -> Option<(usize, bool)> {
        match self {
            Item::Atom(a) => Some((a.rel, false)),
            Item::Neg(a) | Item::Agg { atom: a, .. } => Some((a.rel, true)),
            _ => None,
        }
    }
}

/// A fact (no body) or a rule.
struct Clause {
    head: Ident,
    head_args: Vec<Arg>,
    head_loc: Location,
    body: Option<Vec<Item>>,
}

impl<'s> Parser<'s> {
    pub(super) fn datalog_literal(&mut self) -> Expr {
        let start = self.next_start();
        self.advance(); // `datalog`
        self.bump(TokKind::LeftBrace);
        // rules are newline-separated, so whitespace never applies (`x\n p(..)` is two
        // things) and an operator never binds across a line end
        self.with_juxta(Juxta::Off, |p| {
            p.with_group_depth(0, |p| p.struct_literals(true, |p| p.datalog_body(start)))
        })
    }

    fn datalog_body(&mut self, start: usize) -> Expr {
        let mut rels: Vec<Rel> = vec![];
        let mut clauses: Vec<Clause> = vec![];
        while !self.at(TokKind::RightBrace) && !self.at(TokKind::Eof) {
            let before = self.next;
            match self.peek() {
                // `relation` is Ascent's spelling
                TokKind::Ident("rel" | "relation") if matches!(self.nth(1), TokKind::Ident(_)) => {
                    self.advance();
                    if let Some(rel) = self.dl_rel(&rels) {
                        rels.push(rel);
                    }
                }
                TokKind::Ident(_) if self.nth(1) == TokKind::LeftParenthesis => {
                    let clause = self.dl_clause(&rels);
                    clauses.push(clause);
                }
                _ => {
                    self.expected("a `rel` declaration, a fact, or a rule");
                    while !self.at_line_start()
                        && !self.at(TokKind::RightBrace)
                        && !self.at(TokKind::Eof)
                    {
                        self.advance();
                    }
                }
            }
            self.eat(TokKind::SemiColon);
            if self.next == before {
                self.advance();
            }
        }
        self.expect(TokKind::RightBrace);
        let here = self.location(start);
        let errors = self.errors.len();
        let lowered = self.dl_lower(rels, clauses, here);
        if self.errors.len() > errors {
            return self.poison_expr(start);
        }
        lowered
    }

    /// `name(field: Type, …) [from <expr>]`, after the `rel`.
    fn dl_rel(&mut self, rels: &[Rel]) -> Option<Rel> {
        let name = self.require_ident();
        // `name: Type` fields, or Ascent's positional `relation edge(int, int)`
        let fields = if self.expect(TokKind::LeftParenthesis) {
            let parsed = self.list(
                TokKind::RightParenthesis,
                |k| k.is_ident() || k.starts_annotation(),
                |p| {
                    if matches!(p.peek(), TokKind::Ident(_)) && p.nth(1) == TokKind::Colon {
                        let field = p.require_ident();
                        p.expect(TokKind::Colon);
                        (Some(field), p.annotation())
                    } else {
                        (None, p.annotation())
                    }
                },
            );
            parsed
                .into_iter()
                .enumerate()
                .map(|(k, (field, ann))| {
                    (field.unwrap_or_else(|| Ident::new(format!("_{k}"), name.location)), ann)
                })
                .collect()
        } else {
            vec![]
        };
        let seed = if self.at(TokKind::Ident("from")) {
            self.advance();
            Some(self.expr())
        } else {
            None
        };
        if rels.iter().any(|r| r.name.lexeme == name.lexeme) {
            self.dl_error(
                name.location,
                format!("relation `{}` is already declared", name.lexeme),
                "each `rel` needs its own name".into(),
            );
            return None;
        }
        if fields.is_empty() {
            self.dl_error(
                name.location,
                format!("relation `{}` has no fields", name.lexeme),
                "declare at least one: `rel name(field: Type)` or `relation name(Type)`".into(),
            );
            return None;
        }
        Some(Rel { name, fields, seed })
    }

    /// `head(args…) [if item, item…]`.
    fn dl_clause(&mut self, rels: &[Rel]) -> Clause {
        let start = self.next_start();
        let head = self.require_ident();
        self.bump(TokKind::LeftParenthesis);
        let head_args = self.dl_args();
        let head_loc = self.location(start);
        // `if`, or Ascent's `<--` (three adjacent tokens: `<` `-` `-`)
        let arrow = self.at(TokKind::Less)
            && self.nth(1) == TokKind::Minus
            && self.nth(2) == TokKind::Minus;
        let intro = if arrow {
            self.advance();
            self.advance();
            self.advance();
            true
        } else {
            self.eat(TokKind::If)
        };
        let body = intro.then(|| {
            let mut items = vec![];
            loop {
                let is_rel = |p: &Self, k: usize| {
                    matches!(p.nth(k), TokKind::Ident(n) if rels.iter().any(|r| r.name.lexeme == n))
                        && p.nth(k + 1) == TokKind::LeftParenthesis
                };
                let item = match self.peek() {
                    TokKind::Let => {
                        self.advance();
                        let v = self.require_ident();
                        self.expect(TokKind::Equal);
                        Item::Let(v, self.expr())
                    }
                    TokKind::For => {
                        self.advance();
                        let v = self.require_ident();
                        self.expect(TokKind::In);
                        Item::For(v, self.expr())
                    }
                    // Ascent writes its conditions `if cond`
                    TokKind::If => {
                        self.advance();
                        Item::Cond(self.expr())
                    }
                    TokKind::Bang | TokKind::NotWord if is_rel(self, 1) => {
                        self.advance();
                        Item::Neg(self.dl_atom(rels))
                    }
                    TokKind::Ident("agg")
                        if matches!(self.nth(1), TokKind::Ident(_)) && self.nth(2) == TokKind::Equal =>
                    {
                        self.advance();
                        let var = self.require_ident();
                        self.bump(TokKind::Equal);
                        let func = self.require_ident();
                        self.expect(TokKind::LeftParenthesis);
                        let arg = (!self.at(TokKind::RightParenthesis)).then(|| self.expr());
                        self.expect(TokKind::RightParenthesis);
                        self.expect(TokKind::In);
                        let atom = self.dl_atom(rels);
                        Item::Agg { var, func, arg, atom }
                    }
                    _ if is_rel(self, 0) => Item::Atom(self.dl_atom(rels)),
                    _ => Item::Cond(self.expr()),
                };
                items.push(item);
                if !self.eat(TokKind::Comma) {
                    break;
                }
            }
            items
        });
        Clause {
            head,
            head_args,
            head_loc,
            body,
        }
    }

    /// `rel(arg, …)` naming a declared relation (the caller checked), with its arity.
    fn dl_atom(&mut self, rels: &[Rel]) -> Atom {
        let atom_start = self.next_start();
        let name = self.require_ident();
        self.bump(TokKind::LeftParenthesis);
        let args = self.dl_args();
        let rel = rels
            .iter()
            .position(|r| r.name.lexeme == name.lexeme)
            .expect("dispatched on a declared relation");
        let want = rels[rel].fields.len();
        if args.len() != want {
            self.dl_arity(self.location(atom_start), &name.lexeme, args.len(), want);
        }
        Atom { rel, args }
    }

    /// An atom's or head's arguments, after the `(`; takes the `)`.
    fn dl_args(&mut self) -> Vec<Arg> {
        let exprs = self.in_group(|p| {
            p.with_juxta(Juxta::Comma, |p| {
                p.list(TokKind::RightParenthesis, TokKind::starts_expr, |p| {
                    p.struct_literals(true, Self::expr)
                })
            })
        });
        exprs
            .into_iter()
            .map(|e| match e.kind() {
                ExprKind::Ident(id) if id.lexeme == "_" => Arg::Wild(e.location()),
                ExprKind::Ident(id) => Arg::Var(id.clone()),
                _ => Arg::Expr(e),
            })
            .collect()
    }

    fn dl_error(&mut self, at: Location, msg: String, label: String) {
        // lowering happens after the `}`, where `error`'s one-per-token rule would keep only
        // the first complaint
        self.errors.push(
            Misdirection {
                src: self.src(),
                at: at.into(),
                msg,
                label,
            }
            .into(),
        );
    }

    fn dl_arity(&mut self, at: Location, rel: &str, got: usize, want: usize) {
        let plural = |n: usize| if n == 1 { "" } else { "s" };
        self.dl_error(
            at,
            format!(
                "relation `{rel}` has {want} field{}, but this names {got}",
                plural(want)
            ),
            format!("give `{rel}` exactly {want} argument{}", plural(want)),
        );
    }

    fn dl_lower(&mut self, rels: Vec<Rel>, clauses: Vec<Clause>, here: Location) -> Expr {
        let b = Build(here);
        let arr = |name: &str| format!("__dl_{name}");
        let names: Vec<String> = rels.iter().map(|r| r.name.lexeme.clone()).collect();
        let arity: Vec<usize> = rels.iter().map(|r| r.fields.len()).collect();
        let mut stmts: Vec<Stmt> = vec![];

        // the relation arrays, then their seeds
        let mut seeds = vec![];
        for rel in rels {
            let types: Vec<Annotation> = rel.fields.into_iter().map(|(_, t)| t).collect();
            let elem = if types.len() == 1 {
                types.into_iter().next().expect("one field")
            } else {
                Annotation::Tuple(types)
            };
            let ann = Annotation::Array(Box::new(elem), vec![]);
            let name = arr(&rel.name.lexeme);
            stmts.push(b.let_(&name, Some(ann), b.e(Literal::Array(vec![]))));
            // the dedup index takes its type from its inserts, so a relation nothing fills
            // goes without one
            if rel.seed.is_some() || clauses.iter().any(|c| c.head.lexeme == rel.name.lexeme) {
                let index = format!("__dl_x_{}", rel.name.lexeme);
                stmts.push(b.let_(&index, None, b.e(Literal::Dictionary(vec![]))));
            }
            if let Some(seed) = rel.seed {
                seeds.push((name, seed));
            }
        }
        for (name, seed) in seeds {
            let body = b.block(vec![b.insert(&name, b.id("__dl_s"))], None);
            stmts.push(b.stmt(For::new(b.pat("__dl_s"), seed, body)));
        }

        // facts (and clauses that read no relation) run once, first; rules go in strata
        let mut rules = vec![];
        for clause in clauses {
            let Some(rel) = names.iter().position(|n| *n == clause.head.lexeme) else {
                self.dl_error(
                    clause.head.location,
                    format!("`{}` isn't a relation of this `datalog` block", clause.head.lexeme),
                    self.dl_declared(&names),
                );
                continue;
            };
            if clause.head_args.len() != arity[rel] {
                self.dl_arity(clause.head_loc, &names[rel], clause.head_args.len(), arity[rel]);
                continue;
            }
            let reads_rel = clause
                .body
                .as_ref()
                .is_some_and(|items| items.iter().any(|i| i.reads().is_some()));
            if reads_rel {
                rules.push((rel, clause));
            } else if let Some(nest) = self.dl_nest(&b, 0, rel, clause, &names, &arity) {
                stmts.push(nest);
            }
        }

        // Stratify (as Ascent does): a rule's head sits at least as high as every relation
        // it reads, and strictly higher than one it negates or aggregates -- that one must
        // be complete before the rule asks. A relation that would have to sit above itself
        // depends on its own negation or aggregate, which has no fixpoint.
        let mut stratum = vec![0usize; names.len()];
        let mut cyclic = None;
        for _ in 0..=names.len() {
            let mut changed = false;
            for (head, clause) in &rules {
                for (read, complete) in clause.body.iter().flatten().filter_map(Item::reads) {
                    let need = stratum[read] + usize::from(complete);
                    if stratum[*head] < need {
                        stratum[*head] = need;
                        changed = true;
                        if need > names.len() {
                            cyclic = Some((*head, read));
                        }
                    }
                }
            }
            if !changed || cyclic.is_some() {
                break;
            }
        }
        if let Some((head, read)) = cyclic {
            self.dl_error(
                here,
                format!(
                    "`{}` depends on a negation or aggregate of `{}`, which depends on `{}` in turn",
                    names[head], names[read], names[head]
                ),
                "a relation can't be computed from its own absence or total -- break the cycle".into(),
            );
            return b.e(Literal::Unit);
        }
        let top = rules.iter().map(|(h, _)| stratum[*h]).max().unwrap_or(0);
        let mut by_stratum: Vec<Vec<(usize, Clause)>> = (0..=top).map(|_| vec![]).collect();
        for (head, clause) in rules {
            by_stratum[stratum[head]].push((head, clause));
        }

        for (s, rules) in by_stratum.into_iter().enumerate() {
            // clauses with no positive atom can't feed themselves: once, at the stratum's start
            let (looped, once): (Vec<_>, Vec<_>) = rules.into_iter().partition(|(_, c)| {
                c.body.iter().flatten().any(|i| matches!(i, Item::Atom(_)))
            });
            for (rel, clause) in once {
                if let Some(nest) = self.dl_nest(&b, s, rel, clause, &names, &arity) {
                    stmts.push(nest);
                }
            }
            if looped.is_empty() {
                continue;
            }
            let mut in_body = vec![false; names.len()];
            for (_, clause) in &looped {
                for item in clause.body.iter().flatten() {
                    if let Item::Atom(atom) = item {
                        in_body[atom.rel] = true;
                    }
                }
            }
            let used: Vec<usize> = (0..names.len()).filter(|&r| in_body[r]).collect();
            let lo = |r: usize| format!("__dl_lo{s}_{}", names[r]);
            let hi = |r: usize| format!("__dl_hi{s}_{}", names[r]);
            for &r in &used {
                stmts.push(b.let_(&lo(r), None, b.int(0)));
            }
            let mut round = vec![];
            let mut settled = None;
            for &r in &used {
                round.push(b.let_(&hi(r), None, b.method(b.id(&arr(&names[r])), "len", vec![])));
                let same = b.cmp(b.id(&hi(r)), EqualityOp::Equal, b.id(&lo(r)));
                settled = Some(match settled {
                    None => same,
                    Some(prev) => b.e(Logical::new(prev, LogicalOp::And, same)),
                });
            }
            let stop = b.block(vec![b.stmt(Break::new(None))], None);
            round.push(b.if_stmt(settled.expect("a looped rule has an atom"), stop));
            for (rel, clause) in looped {
                if let Some(nest) = self.dl_nest(&b, s, rel, clause, &names, &arity) {
                    round.push(nest);
                }
            }
            for &r in &used {
                round.push(b.assign(&lo(r), b.id(&hi(r))));
            }
            stmts.push(b.stmt(Loop::new(b.block(round, None))));
        }

        let mut out: Vec<Expr> = names.iter().map(|n| b.id(&arr(n))).collect();
        let result = if out.is_empty() {
            b.e(Literal::Unit)
        } else if out.len() == 1 {
            out.pop().expect("one relation")
        } else {
            b.e(Literal::Tuple(out))
        };
        b.block(stmts, Some(result))
    }

    fn dl_declared(&self, names: &[String]) -> String {
        if names.is_empty() {
            "declare it first: `rel name(field: Type, …)`".into()
        } else {
            format!(
                "declared here: {}",
                names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
            )
        }
    }

    /// One clause as a statement: its atoms' loops and conditions' `if`s, nested in body
    /// order, around the head's insert. `None` when the clause doesn't check.
    fn dl_nest(
        &mut self,
        b: &Build,
        stratum: usize,
        head_rel: usize,
        clause: Clause,
        names: &[String],
        arity: &[usize],
    ) -> Option<Stmt> {
        enum Wrap {
            /// `for __dl_i in lo .. hi { stmts; if filter { … } }`
            Scan {
                index: String,
                lo: Expr,
                hi: Expr,
                stmts: Vec<Stmt>,
                filter: Option<Expr>,
            },
            Cond(Expr),
            /// `{ stmts; if cond { … } }` (or no `if`) -- `let`, negation, aggregates
            Pre {
                stmts: Vec<Stmt>,
                cond: Option<Expr>,
            },
            /// `for v in e { … }`
            For(Ident, Expr),
        }
        let is_rule = clause.body.is_some();
        let mut scratch = 0;
        let items = clause.body.unwrap_or_default();
        let atoms = items.iter().filter(|i| matches!(i, Item::Atom(_))).count();
        let mut bound: Vec<String> = vec![];
        let mut wraps = vec![];
        let mut ok = true;
        let mut ordinal = 0;
        for item in items {
            match item {
                Item::Atom(atom) => {
                    if atom.args.len() != arity[atom.rel] {
                        // already reported where it parsed
                        ok = false;
                    }
                    let rel = &names[atom.rel];
                    let a = ordinal;
                    ordinal += 1;
                    let last = a + 1 == atoms;
                    let index = format!("__dl_i{a}");
                    let tuple = format!("__dl_t{a}");
                    let lo_name = format!("__dl_lo{stratum}_{rel}");
                    let lo = if !last {
                        b.int(0)
                    } else if a == 0 {
                        b.id(&lo_name)
                    } else {
                        // an earlier atom already took a new tuple: everything's fair game
                        let all = b.block(vec![], Some(b.int(0)));
                        let delta = b.block(vec![], Some(b.id(&lo_name)));
                        b.if_else(b.id(&format!("__dl_d{}", a - 1)), all, Some(delta))
                    };
                    let hi = b.id(&format!("__dl_hi{stratum}_{rel}"));
                    let single = arity[atom.rel] == 1;
                    let mut stmts = vec![];
                    if atom.args.iter().any(|arg| !matches!(arg, Arg::Wild(_))) {
                        let elem = b.e(Access::Square {
                            left: b.id(&format!("__dl_{rel}")),
                            key: b.id(&index),
                            kind: AccessKind::Direct,
                        });
                        stmts.push(b.let_(&tuple, None, elem));
                    }
                    let column = |k: usize| {
                        if single {
                            b.id(&tuple)
                        } else {
                            b.field(b.id(&tuple), k)
                        }
                    };
                    let mut filter: Option<Expr> = None;
                    let mut and = |test: Expr| {
                        filter = Some(match filter.take() {
                            None => test,
                            Some(prev) => b.e(Logical::new(prev, LogicalOp::And, test)),
                        });
                    };
                    for (k, arg) in atom.args.into_iter().enumerate() {
                        match arg {
                            Arg::Wild(_) => {}
                            Arg::Var(v) if bound.contains(&v.lexeme) => {
                                and(b.cmp(column(k), EqualityOp::Equal, b.e(v)));
                            }
                            Arg::Var(v) => {
                                bound.push(v.lexeme.clone());
                                stmts.push(b.let_at(v, column(k)));
                            }
                            Arg::Expr(e) => and(b.cmp(column(k), EqualityOp::Equal, e)),
                        }
                    }
                    if !last {
                        let here = b.cmp(b.id(&index), EqualityOp::GreaterOrEqual, b.id(&lo_name));
                        let delta = if a == 0 {
                            here
                        } else {
                            let prev = b.id(&format!("__dl_d{}", a - 1));
                            b.e(Logical::new(prev, LogicalOp::Or, here))
                        };
                        stmts.push(b.let_(&format!("__dl_d{a}"), None, delta));
                    }
                    wraps.push(Wrap::Scan {
                        index,
                        lo,
                        hi,
                        stmts,
                        filter,
                    });
                }
                Item::Let(v, e) => {
                    bound.push(v.lexeme.clone());
                    wraps.push(Wrap::Pre {
                        stmts: vec![b.let_at(v, e)],
                        cond: None,
                    });
                }
                Item::For(v, e) => {
                    bound.push(v.lexeme.clone());
                    wraps.push(Wrap::For(v, e));
                }
                Item::Neg(atom) => {
                    let k = scratch;
                    scratch += 1;
                    let rel = &names[atom.rel];
                    let single = arity[atom.rel] == 1;
                    let (t, j, flag) = (format!("__dl_nt{k}"), format!("__dl_nj{k}"), format!("__dl_nf{k}"));
                    let column = |c: usize| if single { b.id(&t) } else { b.field(b.id(&t), c) };
                    let mut filter: Option<Expr> = None;
                    for (c, arg) in atom.args.into_iter().enumerate() {
                        let test = match arg {
                            Arg::Wild(_) => continue,
                            Arg::Var(v) if bound.contains(&v.lexeme) => {
                                b.cmp(column(c), EqualityOp::Equal, b.e(v))
                            }
                            Arg::Var(v) => {
                                self.dl_error(
                                    v.location,
                                    format!("`{}` in a negated atom isn't bound", v.lexeme),
                                    format!("bind `{}` in an atom to its left, or write `_`", v.lexeme),
                                );
                                ok = false;
                                continue;
                            }
                            Arg::Expr(e) => b.cmp(column(c), EqualityOp::Equal, e),
                        };
                        filter = Some(match filter.take() {
                            None => test,
                            Some(prev) => b.e(Logical::new(prev, LogicalOp::And, test)),
                        });
                    }
                    let hit = vec![b.assign(&flag, b.e(Literal::True)), b.stmt(Break::new(None))];
                    let elem = b.e(Access::Square {
                        left: b.id(&format!("__dl_{rel}")),
                        key: b.id(&j),
                        kind: AccessKind::Direct,
                    });
                    let scan_body = match filter {
                        Some(f) => vec![b.let_(&t, None, elem), b.if_stmt(f, b.block(hit, None))],
                        None => hit,
                    };
                    let len = b.method(b.id(&format!("__dl_{rel}")), "len", vec![]);
                    let scan = For::new(b.pat(&j), b.e(Range::new(b.int(0), len, false)), b.block(scan_body, None));
                    wraps.push(Wrap::Pre {
                        stmts: vec![b.let_(&flag, None, b.e(Literal::False)), b.stmt(scan)],
                        cond: Some(b.e(Unary::new(UnaryOp::Not, b.id(&flag)))),
                    });
                }
                Item::Agg { var, func, arg, atom } => {
                    let k = scratch;
                    scratch += 1;
                    let rel = &names[atom.rel];
                    let single = arity[atom.rel] == 1;
                    let (t, j, acc) = (format!("__dl_at{k}"), format!("__dl_aj{k}"), format!("__dl_aa{k}"));
                    let column = |c: usize| if single { b.id(&t) } else { b.field(b.id(&t), c) };
                    // the atom's own new names bind per tuple, visible to `f`'s argument only
                    let mut locals = vec![];
                    let mut local_names: Vec<String> = vec![];
                    let mut filter: Option<Expr> = None;
                    for (c, a) in atom.args.into_iter().enumerate() {
                        let test = match a {
                            Arg::Wild(_) => continue,
                            Arg::Var(v) if bound.contains(&v.lexeme) && !local_names.contains(&v.lexeme) => {
                                b.cmp(column(c), EqualityOp::Equal, b.e(v))
                            }
                            Arg::Var(v) if local_names.contains(&v.lexeme) => {
                                b.cmp(column(c), EqualityOp::Equal, b.e(v))
                            }
                            Arg::Var(v) => {
                                local_names.push(v.lexeme.clone());
                                locals.push(b.let_at(v, column(c)));
                                continue;
                            }
                            Arg::Expr(e) => b.cmp(column(c), EqualityOp::Equal, e),
                        };
                        filter = Some(match filter.take() {
                            None => test,
                            Some(prev) => b.e(Logical::new(prev, LogicalOp::And, test)),
                        });
                    }
                    let elem = b.e(Access::Square {
                        left: b.id(&format!("__dl_{rel}")),
                        key: b.id(&j),
                        kind: AccessKind::Direct,
                    });
                    let len = b.method(b.id(&format!("__dl_{rel}")), "len", vec![]);
                    let scan = |b: &Build, step: Stmt, locals: Vec<Stmt>, filter: Option<Expr>| {
                        let mut body = vec![b.let_(&t, None, elem.clone())];
                        body.extend(locals);
                        body.push(match filter {
                            Some(f) => b.if_stmt(f, b.block(vec![step], None)),
                            None => step,
                        });
                        b.stmt(For::new(
                            b.pat(&j),
                            b.e(Range::new(b.int(0), len.clone(), false)),
                            b.block(body, None),
                        ))
                    };
                    let f = func.lexeme.as_str();
                    match (f, arg) {
                        ("count", None) => {
                            let step = b.assign(&acc, b.e(Evaluation::new(b.id(&acc), EvaluationOp::Plus, b.int(1))));
                            wraps.push(Wrap::Pre {
                                stmts: vec![
                                    b.let_(&acc, None, b.int(0)),
                                    scan(&b, step, locals, filter),
                                    b.let_at(var.clone(), b.id(&acc)),
                                ],
                                cond: None,
                            });
                        }
                        ("sum" | "min" | "max" | "mean", Some(arg)) => {
                            // gather the matching values, then fold them -- an empty match set
                            // derives nothing (there's no zero to give back for a min)
                            let vals = format!("__dl_av{k}");
                            let push = b.expr_stmt(b.method(b.id(&vals), "push", vec![arg]));
                            let nonempty = b.cmp(
                                b.method(b.id(&vals), "len", vec![]),
                                EqualityOp::Greater,
                                b.int(0),
                            );
                            wraps.push(Wrap::Pre {
                                stmts: vec![
                                    b.let_(&vals, None, b.e(Literal::Array(vec![]))),
                                    scan(&b, push, locals, filter),
                                ],
                                cond: Some(nonempty),
                            });
                            let at = |i: Expr| {
                                b.e(Access::Square {
                                    left: b.id(&vals),
                                    key: i,
                                    kind: AccessKind::Direct,
                                })
                            };
                            let step = match f {
                                "min" | "max" => {
                                    let op = if f == "min" { EqualityOp::Less } else { EqualityOp::Greater };
                                    b.if_stmt(
                                        b.cmp(at(b.id("__dl_ai")), op, b.id(&acc)),
                                        b.block(vec![b.assign(&acc, at(b.id("__dl_ai")))], None),
                                    )
                                }
                                _ => b.assign(
                                    &acc,
                                    b.e(Evaluation::new(b.id(&acc), EvaluationOp::Plus, at(b.id("__dl_ai")))),
                                ),
                            };
                            let fold = For::new(
                                b.pat("__dl_ai"),
                                b.e(Range::new(b.int(1), b.method(b.id(&vals), "len", vec![]), false)),
                                b.block(vec![step], None),
                            );
                            let value = if f == "mean" {
                                b.e(Evaluation::new(
                                    b.method(b.id(&acc), "to_float", vec![]),
                                    EvaluationOp::Divide,
                                    b.method(b.method(b.id(&vals), "len", vec![]), "to_float", vec![]),
                                ))
                            } else {
                                b.id(&acc)
                            };
                            wraps.push(Wrap::Pre {
                                stmts: vec![
                                    b.let_(&acc, None, at(b.int(0))),
                                    b.stmt(fold),
                                    b.let_at(var.clone(), value),
                                ],
                                cond: None,
                            });
                        }
                        (_, arg) => {
                            let want = match f {
                                "count" => "`count()` takes no argument",
                                "sum" | "min" | "max" | "mean" => "this aggregate takes one argument",
                                _ => "use count(), sum(e), min(e), max(e) or mean(e)",
                            };
                            let _ = arg;
                            self.dl_error(func.location, format!("`{f}` isn't an aggregate of this shape"), want.into());
                            ok = false;
                        }
                    }
                    bound.push(var.lexeme.clone());
                }
                Item::Cond(e) => {
                    // `edgee(a, _)`: a call that can't be a condition is a misspelled relation
                    if let ExprKind::Call(call) = e.kind()
                        && let ExprKind::Ident(callee) = call.left.kind()
                    {
                        let unbound = call.arguments.iter().find_map(|arg| match arg.value.kind() {
                            ExprKind::Ident(id)
                                if id.lexeme == "_" || !bound.contains(&id.lexeme) =>
                            {
                                Some(id.lexeme.clone())
                            }
                            _ => None,
                        });
                        if let Some(var) = unbound {
                            let label = format!(
                                "{} -- as a condition, `{var}` would have to be bound by an atom to its left",
                                self.dl_declared(names)
                            );
                            self.dl_error(
                                call.left.location(),
                                format!("`{}` isn't a relation of this `datalog` block", callee.lexeme),
                                label,
                            );
                            ok = false;
                        }
                    }
                    wraps.push(Wrap::Cond(e));
                }
            }
        }

        // the head: a tuple of the body's values, pushed when it's new. A body that already
        // failed would only blame the head for the variables it couldn't bind.
        let body_ok = ok;
        let mut values = vec![];
        for arg in clause.head_args {
            match arg {
                Arg::Wild(at) => {
                    self.dl_error(
                        at,
                        "`_` can't stand in a head or a fact".into(),
                        "a derived tuple needs a value in every field".into(),
                    );
                    ok = false;
                }
                Arg::Var(_) if is_rule && !body_ok => {}
                Arg::Var(v) if is_rule && !bound.contains(&v.lexeme) => {
                    self.dl_error(
                        v.location,
                        format!("head variable `{}` isn't bound by the rule's body", v.lexeme),
                        format!(
                            "bind `{}` in an atom after `if` (or write `({})` for a value from outside the block)",
                            v.lexeme, v.lexeme
                        ),
                    );
                    ok = false;
                }
                Arg::Var(v) => values.push(b.e(v)),
                Arg::Expr(e) => values.push(e),
            }
        }
        if !ok {
            return None;
        }
        let value = if values.len() == 1 {
            values.pop().expect("one value")
        } else {
            b.e(Literal::Tuple(values))
        };
        let target = format!("__dl_{}", names[head_rel]);
        let mut inner = b.insert(&target, value);
        for wrap in wraps.into_iter().rev() {
            inner = match wrap {
                Wrap::Cond(c) => b.if_stmt(c, b.block(vec![inner], None)),
                Wrap::Pre { mut stmts, cond } => {
                    stmts.push(match cond {
                        Some(c) => b.if_stmt(c, b.block(vec![inner], None)),
                        None => inner,
                    });
                    b.expr_stmt(b.block(stmts, None))
                }
                Wrap::For(v, e) => {
                    let at = v.location;
                    b.stmt(For::new(Pat::new(PatKind::Ident(v), at), e, b.block(vec![inner], None)))
                }
                Wrap::Scan {
                    index,
                    lo,
                    hi,
                    mut stmts,
                    filter,
                } => {
                    stmts.push(match filter {
                        Some(f) => b.if_stmt(f, b.block(vec![inner], None)),
                        None => inner,
                    });
                    let range = b.e(Range::new(lo, hi, false));
                    b.stmt(For::new(b.pat(&index), range, b.block(stmts, None)))
                }
            };
        }
        Some(inner)
    }
}

/// Builds synthesized nodes, all located at the `datalog` block.
struct Build(Location);
impl Build {
    fn e(&self, kind: impl Into<ExprKind>) -> Expr {
        Expr::new(kind.into(), self.0)
    }

    fn id(&self, name: &str) -> Expr {
        self.e(Ident::new(name, self.0))
    }

    fn int(&self, n: i64) -> Expr {
        self.e(Literal::Int(n))
    }

    fn pat(&self, name: &str) -> Pat {
        Pat::new(PatKind::Ident(Ident::new(name, self.0)), self.0)
    }

    fn stmt(&self, kind: impl Into<ExprKind>) -> Stmt {
        Stmt::new(StmtKind::Expr(self.e(kind)), NodeId::new(), self.0)
    }

    fn expr_stmt(&self, e: Expr) -> Stmt {
        Stmt::new(StmtKind::Expr(e), NodeId::new(), self.0)
    }

    fn let_(&self, name: &str, ann: Option<Annotation>, value: Expr) -> Stmt {
        Stmt::new(Let { left: self.pat(name), annotation: ann, right: value, else_branch: None }.into(), NodeId::new(), self.0)
    }

    /// `let v = value`, located at the user's variable so diagnostics land on it.
    fn let_at(&self, v: Ident, value: Expr) -> Stmt {
        let at = v.location;
        let pat = Pat::new(PatKind::Ident(v), at);
        Stmt::new(Let { left: pat, annotation: None, right: value, else_branch: None }.into(), NodeId::new(), at)
    }

    fn assign(&self, name: &str, value: Expr) -> Stmt {
        let assignment = Assignment::new(self.id(name), AssignmentOp::Identity, value);
        Stmt::new(assignment.into(), NodeId::new(), self.0)
    }

    fn if_else(&self, condition: Expr, main_body: Expr, else_expr: Option<Expr>) -> Expr {
        self.e(If {
            condition,
            main_body,
            else_expr,
            binding: None,
        })
    }

    fn if_stmt(&self, condition: Expr, body: Expr) -> Stmt {
        let e = self.if_else(condition, body, None);
        Stmt::new(StmtKind::Expr(e), NodeId::new(), self.0)
    }

    fn block(&self, body: Vec<Stmt>, yielded_expr: Option<Expr>) -> Expr {
        self.e(Block { body, yielded_expr })
    }

    fn method(&self, recv: Expr, name: &str, args: Vec<Expr>) -> Expr {
        let left = self.e(Access::Dot {
            left: recv,
            right: self.id(name),
            kind: AccessKind::Direct,
        });
        let arguments = args
            .into_iter()
            .map(|value| Argument { name: None, value })
            .collect();
        self.e(Call::new(left, arguments))
    }

    fn field(&self, tuple: Expr, k: usize) -> Expr {
        self.e(Access::Dot {
            left: tuple,
            right: self.int(k as i64),
            kind: AccessKind::Direct,
        })
    }

    fn cmp(&self, left: Expr, op: EqualityOp, right: Expr) -> Expr {
        self.e(Equality::new(left, op, right))
    }

    /// `{ let h = value; let k = f"{h}"; let bk = x[k] ?? []; if !bk.contains(h) { r.push(h);
    /// bk.push(h); x.insert(k, bk); } }` -- `x` buckets `r`'s tuples by their rendering, so the
    /// dedup scan is only over tuples that print the same (a rendering collision costs a
    /// longer scan, never a wrong answer).
    fn insert(&self, rel: &str, value: Expr) -> Stmt {
        let index = format!("__dl_x_{}", &rel["__dl_".len()..]);
        let key = self.e(FString::new(vec![FStringPart::Expr(self.id("__dl_h"))]));
        let bucket = self.e(Access::Square {
            left: self.id(&index),
            key: self.id("__dl_k"),
            kind: AccessKind::Direct,
        });
        let bucket = self.e(Coalescence::new(bucket, self.e(Literal::Array(vec![]))));
        let fresh = self.e(Unary::new(
            UnaryOp::Not,
            self.method(self.id("__dl_bk"), "contains", vec![self.id("__dl_h")]),
        ));
        let call = |recv: &str, name: &str, args: Vec<Expr>| {
            Stmt::new(
                StmtKind::Expr(self.method(self.id(recv), name, args)),
                NodeId::new(),
                self.0,
            )
        };
        let add = vec![
            call(rel, "push", vec![self.id("__dl_h")]),
            call("__dl_bk", "push", vec![self.id("__dl_h")]),
            call(&index, "insert", vec![self.id("__dl_k"), self.id("__dl_bk")]),
        ];
        let body = vec![
            self.let_("__dl_h", None, value),
            self.let_("__dl_k", None, key),
            self.let_("__dl_bk", None, bucket),
            self.if_stmt(fresh, self.block(add, None)),
        ];
        Stmt::new(StmtKind::Expr(self.block(body, None)), NodeId::new(), self.0)
    }
}
