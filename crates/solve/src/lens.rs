//! Known-length facts that live beside the type layer.
//!
//! `[T; n]` makes a length part of the type, but most array values are growable --
//! a literal like `[1, 2, 3]` types as plain `[int]` while still having a
//! statically knowable length. The functions here merge both sources so `where`
//! predicates and `[T; n]` contracts can be proven at compile time where
//! possible, and everything unproven falls back to entry/runtime checks.
//!
//! Only *immutable* facts count toward a compile-time answer: a declared
//! `[T; n]` can't change length, and a literal's shape is fixed at its own
//! node. Growable bindings deliberately stay out of scope -- `let xs = [1, 2]`
//! followed by `xs.push(3)` would leave a stale "2" on the binding's name, so
//! only the literal itself answers for its length.

use std::collections::HashMap;

use parse::{Argument, Expr, ExprKind, Literal, expr::Access};
use shared::{Located, Location};

use crate::{
    Result, Solver,
    components::{DecId, DecKind, FnParam, Ty, TyExt},
    errors::{ArrayLenMismatch, WhereViolation},
    utils::reduce,
};

/// What a `where`-predicate ident maps to at a call site.
enum ArgRef<'a> {
    /// The caller's expression occupying that parameter slot.
    Expr(&'a Expr),
    /// The parameter's declared default, used when the caller passed nothing.
    Default(Literal),
}

impl Solver {
    /// The lengths this expr is proven to have, outermost-first (`[]` means nothing
    /// is known -- either a non-array or an array of unknown size). Facts come from
    /// the expr's own shape, an ident's declared dec type, or an access's left
    /// operand -- but never the expr's own node vid: fulfilling `expr` against
    /// `[T; n]` binds that vid to `Const(n)`, which would make the contract prove
    /// itself instead of checking the value.
    pub(crate) fn known_dims(&self, expr: &Expr) -> Vec<Option<usize>> {
        match expr.kind() {
            ExprKind::Ident(ident) => {
                let shape = self.shape_dims(expr);
                // the *dec's* vid, not this node's: it holds the declared type,
                // already concrete before any callsite unify could touch it
                let typed = self
                    .node_decs
                    .get(&ident.id)
                    .map(|&dec| Ty::Vid(self.decs[dec].vid).normalized(self).fixed_dims())
                    .unwrap_or_default();
                let depth = shape.len().max(typed.len());
                (0..depth)
                    .map(|i| {
                        shape
                            .get(i)
                            .copied()
                            .flatten()
                            .or_else(|| typed.get(i).copied().flatten())
                    })
                    .collect()
            }
            // `m[i]` yields a row whose dims are `m`'s tail; dot access yields a
            // field or method, not an array slice
            ExprKind::Access(Access::Square { left, .. }) => {
                let mut dims = self.known_dims(left);
                if !dims.is_empty() {
                    dims.remove(0);
                }
                dims
            }
            _ => self.shape_dims(expr),
        }
    }

    /// Lengths read off the expr's own shape -- array literals and groupings. A
    /// nested literal only proves inner dims when every element agrees on them:
    /// `[[1, 2], [3]]` knows its outer 2 but nothing inside.
    fn shape_dims(&self, expr: &Expr) -> Vec<Option<usize>> {
        match expr.kind() {
            ExprKind::Literal(Literal::Array(items)) => {
                let mut dims = vec![Some(items.len())];
                if let Some(first) = items.first() {
                    let inner = self.shape_dims(first);
                    if !inner.is_empty() && items.iter().all(|i| self.shape_dims(i) == inner) {
                        dims.extend(inner);
                    }
                }
                dims
            }
            ExprKind::Grouping(g) => self.shape_dims(&g.inner),
            _ => vec![],
        }
    }

    /// Runs right after `fulfill_ty`'s unify: unification admits an unknown length
    /// against a `Const` one (the runtime prologue re-checks it), so this is where
    /// a statically-known length either proves the contract or fails it outright.
    /// Anything left unproven is marked so emit wraps the value in a check.
    pub(crate) fn check_len_contract(&mut self, expr: &Expr, expected: &Ty) -> Result<()> {
        let want = expected.fixed_dims();
        if !want.iter().any(Option::is_some) {
            return Ok(());
        }
        let have = self.known_dims(expr);
        let mut needs_runtime = false;
        for (i, w) in want.iter().enumerate() {
            let Some(&want_n) = w.as_ref() else { continue };
            match have.get(i).copied().flatten() {
                Some(found) if found != want_n => {
                    let src = self.src(expr.location());
                    return Err(ArrayLenMismatch {
                        src,
                        at: expr.location().into(),
                        expected: expected.to_string(),
                        found: render_have(&have),
                    }
                    .into());
                }
                Some(_) => {}
                None => needs_runtime = true,
            }
        }
        if needs_runtime {
            self.len_checks.insert(expr.id(), want);
        }
        Ok(())
    }

    /// `f(args)` where `f` declares where-clauses: bind each param name to the
    /// caller's arg and try to prove the predicates. Provably false is a compile
    /// error; provably true or unknown both defer to the callee's entry check
    /// (true could elide it later, but entry checks are cheap and uniform).
    pub(crate) fn check_call_wheres(
        &mut self,
        fn_dec: DecId,
        params: &[FnParam],
        args_by_slot: &[(usize, &Argument)],
        receiver: Option<&Expr>,
        call_at: Location,
    ) -> Result<()> {
        let DecKind::Item {
            wheres, defaults, ..
        } = &self.decs[fn_dec].kind
        else {
            return Ok(());
        };
        if wheres.is_empty() {
            return Ok(());
        }
        let wheres = wheres.clone();
        let defaults = defaults.clone();
        let fn_name = self.decs[fn_dec].name.clone();

        // param name -> what the caller put in its slot (expr, literal default,
        // or nothing). `receiver` covers a method's leading `self` param.
        let mut env: HashMap<String, ArgRef> = HashMap::new();
        if let Some((param, recv)) = params.first().zip(receiver) {
            if let Some(name) = &param.name {
                env.insert(name.clone(), ArgRef::Expr(recv));
            }
        }
        for (slot, arg) in args_by_slot {
            if let Some(name) = &params[*slot].name {
                env.insert(name.clone(), ArgRef::Expr(&arg.value));
            }
        }
        for (i, param) in params.iter().enumerate() {
            let Some(name) = &param.name else { continue };
            if env.contains_key(name) {
                continue;
            }
            if let Some(Some(lit)) = defaults.get(i) {
                env.insert(name.clone(), ArgRef::Default(lit.clone()));
            }
        }

        for w in &wheres {
            if let Some(Literal::False) = self.eval_where(w, &env)? {
                return Err(WhereViolation {
                    src: self.src(call_at),
                    at: call_at.into(),
                    fn_name: fn_name.clone(),
                    predicate: w.to_string(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Const-folds a `where` predicate with call-site knowledge: params resolve
    /// through `env`, `xs.len()` reads `known_dims` of whatever fills `xs`, and
    /// everything else defers to ordinary const resolution. `Ok(None)` means
    /// "can't prove it" -- never an error on its own.
    fn eval_where(&self, expr: &Expr, env: &HashMap<String, ArgRef>) -> Result<Option<Literal>> {
        reduce(expr, &|e| self.where_leaf(e, env)).map_err(|e| e.into_diag(self))
    }

    /// The `resolve` half of [`eval_where`]: idents look up `env` then the const
    /// table, and a `recv.len()` call answers from the receiver's known dims.
    fn where_leaf(&self, e: &Expr, env: &HashMap<String, ArgRef>) -> Option<Literal> {
        match e.kind() {
            ExprKind::Ident(ident) => match env.get(&ident.lexeme) {
                Some(ArgRef::Expr(arg)) => self.reduce_const_expr(arg).ok().flatten(),
                Some(ArgRef::Default(lit)) => Some(lit.clone()),
                None => self
                    .node_decs
                    .get(&e.id())
                    .and_then(|&dec| self.decs[dec].kind.const_value().cloned()),
            },
            ExprKind::Call(call) if call.arguments.is_empty() => {
                let ExprKind::Access(Access::Dot {
                    left: recv, right, ..
                }) = call.left.kind()
                else {
                    return None;
                };
                if !right.as_ident().is_some_and(|i| i.lexeme == "len") {
                    return None;
                }
                let dims = match recv.as_ident().and_then(|i| env.get(&i.lexeme)) {
                    Some(ArgRef::Expr(arg)) => self.known_dims(arg),
                    _ => self.known_dims(recv),
                };
                dims.first()
                    .copied()
                    .flatten()
                    .map(|n| Literal::Int(n as i64))
            }
            _ => None,
        }
    }
}

/// `array length 4, 8` -- how a proven-dims vector reads inside diagnostics.
fn render_have(dims: &[Option<usize>]) -> String {
    let parts = dims
        .iter()
        .map(|d| d.map_or("?".into(), |n| n.to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("array length {parts}")
}

/// Array methods that change a collection's length -- rejected on `[T; n]`
/// receivers. In-place permutations (`sort`, `reverse`, `shuffle`) aren't listed:
/// they keep the pinned length intact.
pub(crate) const LEN_MUTATING: &[&str] = &["push", "pop", "extend"];
