//! `datalog { ... }` blocks -- Ascent-flavored rule syntax that desugars at parse time
//! into ordinary mimas control flow, so the solver/compile/vm/wasmgen need no changes:
//! relations are arrays of tuples, each rule body atom becomes a `while` join, the
//! whole block wraps in a `while changed` fixpoint, and dedup leans on the VM's deep
//! `contains` equality.
//!
//! Grammar inside the braces (entries separated by newlines, no `;`):
//!
//! ```text
//! rel name(col, col, ...)      -- declare a relation
//! name(expr, ...)              -- a fact, evaluated once before the fixpoint
//! name(var, ...) if item, ...  -- a rule; head vars must be bound by body atoms
//!   item := rel(arg, ...) | <expr>   -- declared-rel call = join atom, else a guard
//!   arg  := `_` | <var> | <expr>     -- wildcard skip; ident binds or checks; expr equality-matches
//! ```
//!
//! The block evaluates to `~{ name = [...], ... }` -- every declared relation.
//!
//! Semantics notes (v1): the fixpoint is naive (each round re-derives from the full
//! tables), which matches least-fixpoint semantics exactly -- semi-naive deltas are a
//! pure optimization for later. Guards are evaluated in body order, so a guard
//! referencing a variable bound by a *later* atom fails as unbound: write atoms first.
//! No aggregates yet -- optimization objectives are expressed by reading the result
//! dict (e.g. `res.placed.len()`) rather than inside rules.

use std::collections::{HashMap, HashSet};

use crate::{components::*, errors::DatalogError, *};
use chompy::lex::Token;
use lex::TokKind;
use shared::{Located, Location, Result};

/// A `rel` name usable as the generated backing-array variable.
fn rel_var(name: &str) -> String {
    format!("__dl_rel_{name}")
}

/// The relation's this-round-new-tuples buffer (semi-naive deltas).
fn new_var(name: &str) -> String {
    format!("__dl_new_{name}")
}

/// The relation's previous-round snapshot used as the join delta.
fn delta_var(name: &str) -> String {
    format!("__dl_delta_{name}")
}

enum DatalogEntry {
    /// `name(expr, ...)` -- a seed tuple appended before the fixpoint.
    Fact { rel: Ident, args: Vec<Expr> },
    /// `head(args) if item, ...` -- a rule.
    Rule {
        head: Ident,
        head_args: Vec<Expr>,
        body: Vec<BodyItem>,
    },
}

enum BodyItem {
    /// `rel(arg, ...)` -- one loop over that relation's tuples.
    Atom { rel: Ident, args: Vec<Expr> },
    /// `!rel(arg, ...)` -- rule fires only when no matching tuple exists
    /// (stratified negation: args must already be bound).
    NegAtom { rel: Ident, args: Vec<Expr> },
    /// anything else -- a boolean guard evaluated where it appears.
    Guard(Expr),
}

impl<'s> Parser<'s> {
    /// Parses `datalog { ... }` and returns the desugared block expression.
    pub(crate) fn datalog_block(&mut self) -> Result<Expr> {
        let start = self.next_tok_boundary();
        self.take_known(TokKind::Datalog)?;
        self.expect(TokKind::LeftBrace)?;

        let mut decls: HashMap<String, Vec<Annotation>> = HashMap::new();
        let mut order: Vec<Ident> = vec![];
        let mut entries: Vec<DatalogEntry> = vec![];

        loop {
            if self.match_take(TokKind::RightBrace).is_some() {
                break;
            }
            let name = self.require_ident()?;
            if name.lexeme == "rel" && matches!(self.peek()?.kind(), TokKind::Ident(_)) {
                // `rel name(col: ty, ...)` declaration
                let rel = self.require_ident()?;
                self.expect(TokKind::LeftParenthesis)?;
                let mut cols = vec![];
                if self.match_take(TokKind::RightParenthesis).is_none() {
                    loop {
                        self.require_ident()?;
                        self.expect(TokKind::Colon)?;
                        cols.push(self.annotation()?);
                        if self.match_take(TokKind::Comma).is_some() {
                            continue;
                        }
                        self.expect(TokKind::RightParenthesis)?;
                        break;
                    }
                }
                if decls.contains_key(&rel.lexeme) {
                    return Err(self.datalog_err(
                        rel.location,
                        format!("relation `{}` declared twice", rel.lexeme),
                    ));
                }
                decls.insert(rel.lexeme.clone(), cols);
                order.push(rel);
                continue;
            }

            // `name(args)` -- a fact, or a rule head when `if` follows.
            let Some(arity) = decls.get(&name.lexeme).map(|c| c.len()) else {
                return Err(self.datalog_err(
                    name.location.clone(),
                    format!("undeclared relation `{}`", name.lexeme),
                ));
            };
            self.expect(TokKind::LeftParenthesis)?;
            let mut args = vec![];
            if self.match_take(TokKind::RightParenthesis).is_none() {
                loop {
                    args.push(self.expr()?);
                    if self.match_take(TokKind::Comma).is_some() {
                        continue;
                    }
                    self.expect(TokKind::RightParenthesis)?;
                    break;
                }
            }
            if args.len() != arity {
                return Err(self.datalog_err(
                    name.location.clone(),
                    format!(
                        "`{}` takes {} field(s), {} given",
                        name.lexeme,
                        arity,
                        args.len()
                    ),
                ));
            }

            if self.match_take(TokKind::If).is_none() {
                entries.push(DatalogEntry::Fact { rel: name, args });
                continue;
            }

            // rule -- head args are variables or expressions over bound variables
            let mut head_args = vec![];
            for arg in &args {
                if let ExprKind::Ident(id) = arg.kind() {
                    if id.lexeme == "_" {
                        return Err(self.datalog_err(
                            arg.location(),
                            "`_` is not allowed in rule heads".to_string(),
                        ));
                    }
                }
                head_args.push(arg.clone());
            }
            let mut body = vec![];
            loop {
                let e = self.expr()?;
                let mut item = BodyItem::Guard(e.clone());
                // `!rel(args)` is a negated atom; `rel(args)` a join atom;
                // calls to anything else (and every other expr) are guards.
                let call = match e.kind() {
                    ExprKind::Unary(u)
                        if matches!(u.op, UnaryOp::Not)
                            && matches!(u.right.kind(), ExprKind::Call(_)) =>
                    {
                        match u.right.kind() {
                            ExprKind::Call(c) => Some((c, true)),
                            _ => None,
                        }
                    }
                    ExprKind::Call(c) => Some((c, false)),
                    _ => None,
                };
                if let Some((call, negated)) = call {
                    if let ExprKind::Ident(id) = call.left.kind() {
                        if let Some(cols) = decls.get(&id.lexeme) {
                            let arity = cols.len();
                            if call.arguments.len() != arity {
                                return Err(self.datalog_err(
                                    e.location(),
                                    format!(
                                        "`{}` takes {} field(s), {} given",
                                        id.lexeme,
                                        arity,
                                        call.arguments.len()
                                    ),
                                ));
                            }
                            let args: Vec<Expr> =
                                call.arguments.iter().map(|a| a.value.clone()).collect();
                            if negated {
                                item = BodyItem::NegAtom {
                                    rel: id.clone(),
                                    args,
                                };
                            } else {
                                item = BodyItem::Atom {
                                    rel: id.clone(),
                                    args,
                                };
                            }
                        }
                    }
                }
                body.push(item);
                if self.match_take(TokKind::Comma).is_none() {
                    break;
                }
            }
            // range restriction: every free variable in the head must be bound
            // by a body atom (negated atoms and guards don't bind)
            let mut bound: HashSet<String> = HashSet::new();
            for item in &body {
                match item {
                    BodyItem::Atom { args, .. } => {
                        for arg in args {
                            if let ExprKind::Ident(id) = arg.kind() {
                                if id.lexeme != "_" {
                                    bound.insert(id.lexeme.clone());
                                }
                            }
                        }
                    }
                    BodyItem::NegAtom { rel, args } => {
                        // stratified negation: only input relations may be negated,
                        // and only over already-bound variables
                        for arg in args {
                            if let ExprKind::Ident(id) = arg.kind() {
                                if id.lexeme == "_" {
                                    return Err(self.datalog_err(
                                        arg.location(),
                                        "`_` is not allowed in negated atoms".to_string(),
                                    ));
                                }
                            }
                        }
                        let _ = rel;
                    }
                    _ => {}
                }
            }
            // negated relations must be input-only (never a rule head)
            let deriving: HashSet<&str> = entries
                .iter()
                .filter_map(|e| match e {
                    DatalogEntry::Rule { head, .. } => Some(head.lexeme.as_str()),
                    _ => None,
                })
                .collect();
            for item in &body {
                if let BodyItem::NegAtom { rel, .. } = item {
                    if deriving.contains(rel.lexeme.as_str()) {
                        return Err(self.datalog_err(
                            rel.location.clone(),
                            format!(
                                "cannot negate `{}`: negated relations must be input-only",
                                rel.lexeme
                            ),
                        ));
                    }
                }
            }
            for arg in &head_args {
                let mut free = HashSet::new();
                free_idents(arg, &mut free);
                for var in free {
                    if !bound.contains(&var) {
                        return Err(self.datalog_err(
                            name.location.clone(),
                            format!("head variable `{var}` is not bound by any body atom"),
                        ));
                    }
                }
            }
            entries.push(DatalogEntry::Rule {
                head: name,
                head_args,
                body,
            });
        }

        Ok(self.datalog_desugar(&order, &decls, entries, self.location(start)))
    }

    fn datalog_err(&self, at: Location, msg: String) -> shared::Error {
        DatalogError {
            src: self.src(),
            at: at.into(),
            msg,
        }
        .into()
    }

    /// Builds the plain-AST block: relation arrays, fact seeds, the `while changed`
    /// fixpoint containing every rule's nested `while` joins, and the result dict.
    fn datalog_desugar(
        &self,
        order: &[Ident],
        decls: &HashMap<String, Vec<Annotation>>,
        entries: Vec<DatalogEntry>,
        loc: Location,
    ) -> Expr {
        let mut stmts: Vec<Stmt> = vec![];

        // let __dl_rel_<name>: [(c0, c1, ...)] = []; let __dl_new_<name>: same = []
        for rel in order {
            let ann = Annotation::Array(Box::new(Annotation::Tuple(
                decls[&rel.lexeme].clone(),
            )));
            stmts.push(let_stmt_ann(
                &rel_var(&rel.lexeme),
                ann.clone(),
                array_expr(vec![], loc),
                loc,
            ));
            stmts.push(let_stmt_ann(
                &new_var(&rel.lexeme),
                ann,
                array_expr(vec![], loc),
                loc,
            ));
        }
        // __dl_rel_<name>.push([fact args]); __dl_new_<name>.push(<same>)
        for entry in &entries {
            if let DatalogEntry::Fact { rel, args } = entry {
                for tbl in [rel_var(&rel.lexeme), new_var(&rel.lexeme)] {
                    let push = call_expr(
                        dot_expr(var_expr(&tbl, loc), "push", loc),
                        vec![tuple_expr(fresh_exprs(args), loc)],
                        loc,
                    );
                    stmts.push(expr_stmt(push, loc));
                }
            }
        }
        // let __dl_changed = true
        stmts.push(let_stmt(
            "__dl_changed",
            Expr::new(ExprKind::Literal(Literal::True), loc),
            loc,
        ));

        // while __dl_changed { __dl_changed = false; <rules> }
        let mut fix_body: Vec<Stmt> = vec![Stmt::new(
            Assignment {
                left: var_expr("__dl_changed", loc),
                op: AssignmentOp::Identity,
                right: Expr::new(ExprKind::Literal(Literal::False), loc),
            }
            .into(),
            NodeId::default(),
            loc,
        )];
        // snapshot last round's derivations as join deltas, then clear for reuse
        for rel in order {
            fix_body.push(let_stmt(
                &delta_var(&rel.lexeme),
                var_expr(&new_var(&rel.lexeme), loc),
                loc,
            ));
            fix_body.push(Stmt::new(
                StmtKind::Assignment(Assignment {
                    left: var_expr(&new_var(&rel.lexeme), loc),
                    op: AssignmentOp::Identity,
                    right: array_expr(vec![], loc),
                }),
                NodeId::default(),
                loc,
            ));
        }
        for entry in &entries {
            if let DatalogEntry::Rule {
                head,
                head_args,
                body,
            } = entry
            {
                fix_body.extend(emit_rule(head, head_args, body, loc));
            }
        }
        let fixpoint = Expr::new(
            ExprKind::While(While {
                binding: None,
                header: var_expr("__dl_changed", loc),
                body: block_expr(fix_body, None, loc),
            }),
            loc,
        );
        stmts.push(expr_stmt(fixpoint, loc));

        // ( __dl_rel_<a>, __dl_rel_<b>, ... ) in declaration order --
        // heterogeneous shapes need a tuple; destructure for named access.
        let dict = tuple_expr(
            order
                .iter()
                .map(|rel| var_expr(&rel_var(&rel.lexeme), loc))
                .collect(),
            loc,
        );
        Expr::new(
            ExprKind::Block(Block {
                body: stmts,
                yielded_expr: Some(dict),
            }),
            loc,
        )
    }
}

// desugar helpers --------------------------------------------------------------

/// One rule's join loops, semi-naive: for each atom position we emit a variant
/// where that atom iterates the previous round's delta instead of the full
/// relation -- every newly derivable tuple needs at least one new input tuple.
/// Guards nest positively (`if g { ... }`), and the innermost point dedupes
/// against the full table, pushes to both the table and this round's `new`
/// buffer, and sets `__dl_changed`.
fn emit_rule(head: &Ident, head_args: &[Expr], body: &[BodyItem], loc: Location) -> Vec<Stmt> {
    let mut stmts = vec![];
    for delta_at in 0..body.len() {
        if !matches!(body[delta_at], BodyItem::Atom { .. }) {
            continue;
        }
        let mut bound = HashSet::new();
        stmts.extend(emit_items(body, 0, delta_at, head, head_args, &mut bound, loc));
    }
    stmts
}

fn emit_items(
    items: &[BodyItem],
    idx: usize,
    delta_at: usize,
    head: &Ident,
    head_args: &[Expr],
    bound: &mut HashSet<String>,
    loc: Location,
) -> Vec<Stmt> {
    if idx == items.len() {
        // let __dl_tup = [v0, v1, ...]
        // if !__dl_rel_head.contains(__dl_tup) {
        //     __dl_rel_head.push(__dl_tup); __dl_new_head.push(__dl_tup); __dl_changed = true
        // }
        let tuple = tuple_expr(fresh_exprs(head_args), loc);
        let push_body = vec![
            expr_stmt(
                call_expr(
                    dot_expr(var_expr(&rel_var(&head.lexeme), loc), "push", loc),
                    vec![var_expr("__dl_tup", loc)],
                    loc,
                ),
                loc,
            ),
            expr_stmt(
                call_expr(
                    dot_expr(var_expr(&new_var(&head.lexeme), loc), "push", loc),
                    vec![var_expr("__dl_tup", loc)],
                    loc,
                ),
                loc,
            ),
            Stmt::new(
                Assignment {
                    left: var_expr("__dl_changed", loc),
                    op: AssignmentOp::Identity,
                    right: Expr::new(ExprKind::Literal(Literal::True), loc),
                }
                .into(),
                NodeId::default(),
                loc,
            ),
        ];
        return vec![
            let_stmt("__dl_tup", tuple, loc),
            expr_stmt(
                Expr::new(
                    If {
                        condition: not_expr(
                            call_expr(
                                dot_expr(
                                    var_expr(&rel_var(&head.lexeme), loc),
                                    "contains",
                                    loc,
                                ),
                                vec![var_expr("__dl_tup", loc)],
                                loc,
                            ),
                            loc,
                        ),
                        main_body: block_expr(push_body, None, loc),
                        else_expr: None,
                        binding: None,
                    }
                    .into(),
                    loc,
                ),
                loc,
            ),
        ];
    }

    match &items[idx] {
        BodyItem::Guard(g) => {
            // `if g { <rest> }` -- positive nesting, since a bare `continue` here
            // could target `while __dl_changed` when no atom loop encloses us.
            vec![expr_stmt(
                Expr::new(
                    ExprKind::If(If {
                        condition: fresh_clone(g),
                        main_body: block_expr(
                            emit_items(items, idx + 1, delta_at, head, head_args, bound, loc),
                            None,
                            loc,
                        ),
                        else_expr: None,
                        binding: None,
                    }),
                    loc,
                ),
                loc,
            )]
        }
        BodyItem::NegAtom { rel, args } => {
            // `if !(tbl.contains([args])) { <rest> }` -- reads the full table;
            // negated relations are required to be input-only so this is stable.
            vec![expr_stmt(
                Expr::new(
                    ExprKind::If(If {
                        condition: not_expr(
                            call_expr(
                                dot_expr(
                                    var_expr(&rel_var(&rel.lexeme), loc),
                                    "contains",
                                    loc,
                                ),
                                vec![tuple_expr(fresh_exprs(args), loc)],
                                loc,
                            ),
                            loc,
                        ),
                        main_body: block_expr(
                            emit_items(items, idx + 1, delta_at, head, head_args, bound, loc),
                            None,
                            loc,
                        ),
                        else_expr: None,
                        binding: None,
                    }),
                    loc,
                ),
                loc,
            )]
        }
        BodyItem::Atom { rel, args } => {
            let tbl = if idx == delta_at {
                delta_var(&rel.lexeme)
            } else {
                rel_var(&rel.lexeme)
            };
            let ivar = format!("__dl_i{idx}");
            let tvar = format!("__dl_t{idx}");

            // while body: let t = tbl[i]; i = i + 1; <binds/checks>; <deeper items>
            let mut inner: Vec<Stmt> = vec![
                let_stmt(
                    &tvar,
                    index_expr(var_expr(&tbl, loc), var_expr(&ivar, loc), loc),
                    loc,
                ),
                Stmt::new(
                    Assignment {
                        left: var_expr(&ivar, loc),
                        op: AssignmentOp::Identity,
                        right: bin_expr(
                            var_expr(&ivar, loc),
                            EvaluationOp::Plus,
                            int_expr(1, loc),
                            loc,
                        ),
                    }
                    .into(),
                    NodeId::default(),
                    loc,
                ),
            ];
            for (j, arg) in args.iter().enumerate() {
                let elem = tuple_index(var_expr(&tvar, loc), j, loc);
                match arg.kind() {
                    ExprKind::Ident(id) if id.lexeme == "_" => {}
                    ExprKind::Ident(id) if bound.contains(&id.lexeme) => {
                        inner.push(expr_stmt(
                            if_continue(neq_expr(elem, var_expr(&id.lexeme, loc), loc), loc),
                            loc,
                        ))
                    }
                    ExprKind::Ident(id) => {
                        bound.insert(id.lexeme.clone());
                        inner.push(let_stmt(&id.lexeme, elem, loc));
                    }
                    _ => inner.push(expr_stmt(
                        if_continue(neq_expr(elem, fresh_clone(arg), loc), loc),
                        loc,
                    )),
                }
            }
            inner.extend(emit_items(
                items,
                idx + 1,
                delta_at,
                head,
                head_args,
                bound,
                loc,
            ));

            // let i = 0; while i < tbl.len() { ... }
            let header = Expr::new(
                Equality {
                    left: var_expr(&ivar, loc),
                    op: EqualityOp::Less,
                    right: call_expr(
                        dot_expr(var_expr(&tbl, loc), "len", loc),
                        vec![],
                        loc,
                    ),
                }
                .into(),
                loc,
            );
            vec![
                let_stmt(&ivar, int_expr(0, loc), loc),
                expr_stmt(
                    Expr::new(
                        ExprKind::While(While {
                            binding: None,
                            header,
                            body: block_expr(inner, None, loc),
                        }),
                        loc,
                    ),
                    loc,
                ),
            ]
        }
    }
}

// tiny AST builders ------------------------------------------------------------

fn var_expr(name: &str, loc: Location) -> Expr {
    Expr::new(ExprKind::Ident(Ident::new(name, loc)), loc)
}

fn int_expr(v: i64, loc: Location) -> Expr {
    Expr::new(ExprKind::Literal(Literal::Int(v)), loc)
}

fn array_expr(items: Vec<Expr>, loc: Location) -> Expr {
    Expr::new(ExprKind::Literal(Literal::Array(items)), loc)
}

fn dot_expr(left: Expr, field: &str, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Access(Access::Dot {
            left,
            right: var_expr(field, loc),
            kind: AccessKind::Direct,
        }),
        loc,
    )
}

fn index_expr(left: Expr, key: Expr, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Access(Access::Square {
            left,
            key,
            kind: AccessKind::Direct,
        }),
        loc,
    )
}

fn call_expr(left: Expr, args: Vec<Expr>, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Call(Call {
            left,
            arguments: args
                .into_iter()
                .map(|value| Argument {
                    name: None,
                    value,
                })
                .collect(),
        }),
        loc,
    )
}

fn bin_expr(left: Expr, op: EvaluationOp, right: Expr, loc: Location) -> Expr {
    Expr::new(ExprKind::Evaluation(Evaluation { left, op, right }), loc)
}

fn neq_expr(left: Expr, right: Expr, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Equality(Equality {
            left,
            op: EqualityOp::NotEqual,
            right,
        }),
        loc,
    )
}

fn not_expr(inner: Expr, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Unary(Unary {
            op: UnaryOp::Not,
            right: Expr::new(ExprKind::Grouping(Grouping { inner }), loc),
        }),
        loc,
    )
}

fn block_expr(body: Vec<Stmt>, yielded: Option<Expr>, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Block(Block {
            body,
            yielded_expr: yielded,
        }),
        loc,
    )
}

fn if_continue(condition: Expr, loc: Location) -> Expr {
    Expr::new(
        ExprKind::If(If {
            condition,
            main_body: block_expr(
                vec![expr_stmt(Expr::new(ExprKind::Continue(Continue), loc), loc)],
                None,
                loc,
            ),
            else_expr: None,
            binding: None,
        }),
        loc,
    )
}

fn let_stmt(name: &str, right: Expr, loc: Location) -> Stmt {
    Stmt::new(
        StmtKind::Let(Let {
            left: Pat::new(PatKind::Ident(Ident::new(name, loc)), loc),
            annotation: None,
            right,
            else_branch: None,
        }),
        NodeId::default(),
        loc,
    )
}

fn expr_stmt(e: Expr, loc: Location) -> Stmt {
    Stmt::new(StmtKind::Expr(e), NodeId::default(), loc)
}

/// Variable names appearing in an expression -- used for the range-restriction
/// check on rule heads. Walks the common value-expression kinds; exotic kinds
/// (closures, matches, loops) contribute no visible names, which only means an
/// unbound variable inside one surfaces later as the ordinary solve-time error.
fn free_idents(e: &Expr, out: &mut HashSet<String>) {
    match e.kind() {
        ExprKind::Ident(id) => {
            if id.lexeme != "_" {
                out.insert(id.lexeme.clone());
            }
        }
        ExprKind::Evaluation(v) => {
            free_idents(&v.left, out);
            free_idents(&v.right, out);
        }
        ExprKind::Equality(v) => {
            free_idents(&v.left, out);
            free_idents(&v.right, out);
        }
        ExprKind::Logical(v) => {
            free_idents(&v.left, out);
            free_idents(&v.right, out);
        }
        ExprKind::Unary(u) => free_idents(&u.right, out),
        ExprKind::Grouping(g) => free_idents(&g.inner, out),
        ExprKind::Call(c) => {
            free_idents(&c.left, out);
            for arg in &c.arguments {
                free_idents(&arg.value, out);
            }
        }
        ExprKind::Access(a) => match a {
            Access::Dot { left, .. } => free_idents(left, out),
            Access::Square { left, key, .. } => {
                free_idents(left, out);
                free_idents(key, out);
            }
            Access::Identity { .. } | Access::DoubleColon { .. } => {}
        },
        ExprKind::Literal(Literal::Array(items) | Literal::Tuple(items)) => {
            for item in items {
                free_idents(item, out);
            }
        }
        ExprKind::Literal(Literal::Dictionary(fields)) => {
            for (_, value) in fields {
                free_idents(value, out);
            }
        }
        _ => {}
    }
}

fn tuple_expr(items: Vec<Expr>, loc: Location) -> Expr {
    Expr::new(ExprKind::Literal(Literal::Tuple(items)), loc)
}

/// `t.j` -- tuple field access, rendered via Dot with an int literal key.
fn tuple_index(left: Expr, j: usize, loc: Location) -> Expr {
    Expr::new(
        ExprKind::Access(Access::Dot {
            left,
            right: int_expr(j as i64, loc),
            kind: AccessKind::Direct,
        }),
        loc,
    )
}

fn let_stmt_ann(name: &str, ann: Annotation, right: Expr, loc: Location) -> Stmt {
    Stmt::new(
        StmtKind::Let(Let {
            left: Pat::new(PatKind::Ident(Ident::new(name, loc)), loc),
            annotation: Some(ann),
            right,
            else_branch: None,
        }),
        NodeId::default(),
        loc,
    )
}

/// Clone with fresh node ids -- the solver keys nodes by id, so the same
/// source expr emitted into multiple desugared rule variants must be
/// re-identified per copy.
fn fresh_clone(e: &Expr) -> Expr {
    let mut c = e.clone();
    c.refresh_ids();
    c
}

fn fresh_exprs(v: &[Expr]) -> Vec<Expr> {
    v.iter().map(fresh_clone).collect()
}
