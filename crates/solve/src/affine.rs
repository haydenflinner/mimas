//! Posts and panels: the checks that keep a point out of the operations
//! only a difference has.
//!
//! A post (`Post<T>`, written in `library/src/group.mim`) is a point in the
//! affine space over a group `T`: a position, an instant, an index. A panel
//! is a plain `T`, the difference of two posts. The library gives `Post`
//! the operators that make sense (`p - q`, `p + d`, `d + p`, `p - d`), so a
//! form it leaves out would fail as an ordinary "cannot use `+`" error.
//! This module names the fencepost mistake instead: adding two posts,
//! scaling or negating one, integrating a stream of posts, zipping two
//! streams whose lengths differ by the one panel that `n` posts lack.
//!
//! Only the prelude's own `Post`, `Posts` and `Stream` count: a page's
//! `struct Post` (a market post, say) is another type and gets none of
//! this. r[impl mimas.group.affine.errors]

use parse::{
    Access, Call, Expr, ExprKind,
    expr::{EvaluationOp, UnaryOp},
};
use shared::{Located, Location, Ty};

use crate::{Error, Result, Solver, components::TyExt, errors::AffineMisuse, traits::Query};

/// Which std type a value is, as far as these checks care.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `Post<T>`: one point.
    Post,
    /// `Posts<T>`: a stream of points.
    Posts,
    /// `Stream<T>`: a stream of panels (any group).
    Stream,
}

impl Kind {
    fn noun(self) -> &'static str {
        match self {
            Kind::Post => "a post",
            Kind::Posts => "a stream of posts",
            Kind::Stream => "a stream",
        }
    }
}

/// The std kind of `ty`, if it is one of the prelude's `Post`, `Posts` or
/// `Stream`.
pub(crate) fn kind(solver: &Solver, ty: &Ty) -> Option<Kind> {
    let (Ty::Adt(aid, _) | Ty::Identity(aid, _)) = ty else {
        return None;
    };
    let is = |name: &str| solver.prelude_adts.get(name) == Some(aid);
    if is("Post") {
        Some(Kind::Post)
    } else if is("Posts") {
        Some(Kind::Posts)
    } else if is("Stream") {
        Some(Kind::Stream)
    } else {
        None
    }
}

fn err(solver: &Solver, at: Location, msg: &str, label: &str, help: &str) -> Error {
    AffineMisuse {
        src: solver.src(at),
        at: at.into(),
        msg: msg.into(),
        label: label.into(),
        help: help.into(),
    }
    .into()
}

const ADD_HELP: &str = "a post is a place (a position, an instant, an index), and two places do \
     not add. Subtract them to get the panel between them (`p - q`), or add a panel to one \
     (`p + d`)";

/// `a <op> b` that no operator method claimed: the affine reading of the
/// mistake, when a post is involved. `None` leaves the general error.
// r[impl mimas.group.affine.add-posts]
// r[impl mimas.group.affine.scale]
pub(crate) fn binary(
    solver: &Solver,
    op: EvaluationOp,
    lhs: &Ty,
    rhs: &Ty,
    at: Location,
) -> Option<Error> {
    let (l, r) = (kind(solver, lhs), kind(solver, rhs));
    let post = |k: Option<Kind>| matches!(k, Some(Kind::Post | Kind::Posts));
    if !post(l) && !post(r) {
        return None;
    }
    let one = if post(l) { l } else { r }.map_or("a post", Kind::noun);
    Some(match op {
        EvaluationOp::Plus | EvaluationOp::Uplus if post(l) && post(r) => err(
            solver,
            at,
            "adding two posts: subtract them to get a panel, or add a panel to one",
            "both sides are posts",
            ADD_HELP,
        ),
        EvaluationOp::Minus if !post(l) => err(
            solver,
            at,
            "subtracting a post from a panel",
            "a panel minus a post is no place and no change",
            "write `p - q` for the panel between two posts, or `p - d` for the post one panel \
             back",
        ),
        EvaluationOp::Multiply
        | EvaluationOp::Divide
        | EvaluationOp::Dot
        | EvaluationOp::Div
        | EvaluationOp::Obelus => err(
            solver,
            at,
            &format!("scaling {one}: only a panel scales"),
            "a post is a place; twice a place has no meaning",
            "scale the panel from a reference post and add it back: `o + k * (p - o)`. A \
             coordinate is that panel from the zero post: `k * p.at`",
        ),
        _ => return None,
    })
}

/// `-p` on a post. `None` leaves the general error.
// r[impl mimas.group.affine.scale]
pub(crate) fn unary(solver: &Solver, op: UnaryOp, ty: &Ty, at: Location) -> Option<Error> {
    let k = kind(solver, ty)?;
    if !matches!(k, Kind::Post | Kind::Posts) || op != UnaryOp::Negative {
        return None;
    }
    Some(err(
        solver,
        at,
        &format!("negating {}: only a panel negates", k.noun()),
        "a post is a place; minus a place has no meaning",
        "reflect it through a post instead: `o - (p - o)`, or negate its coordinate: `-p.at`",
    ))
}

/// A call, once solved: a method only a stream of panels has, called on a
/// stream of posts (`∫ ps`, `Δ⁻¹ ps`, `z⁻¹ ps` with no start, `ps.sum()`),
/// turns its "no field or method" into the affine reason; a `zip` of two
/// streams gets its alignment check. Runs after the call so the receiver's
/// type is already known and nothing solves out of its usual order.
// r[impl mimas.group.affine.from]
pub(crate) fn after_call(solver: &mut Solver, call: &Call, solved: Result<Ty>) -> Result<Ty> {
    let ExprKind::Access(Access::Dot { left, right, .. }) = call.left.kind() else {
        return solved;
    };
    let Some(name) = right.as_ident() else {
        return solved;
    };
    let name = name.lexeme.as_str();
    match (&solved, name) {
        (Ok(_), "zip") => {
            zip(solver, left, call)?;
            solved
        }
        (
            Err(_),
            "integrate" | "integrate_from" | "sum_before" | "sum_before_from" | "delay" | "sum",
        ) => {
            let Ok(ty) = left.query(solver) else {
                return solved;
            };
            let ty = ty.normalized(solver);
            if kind(solver, &ty) != Some(Kind::Posts) {
                return solved;
            }
            Err(posts_method(solver, name, call.left.location()))
        }
        _ => solved,
    }
}

fn posts_method(solver: &Solver, name: &str, at: Location) -> Error {
    match name {
        "integrate" | "integrate_from" => err(
            solver,
            at,
            "`∫` adds up every step, and posts do not add",
            "this is a stream of posts",
            "integrate the panels from a starting post: `∫ (Δ ps) from ps[0]` gives \
             `ps.after()`, and `Δ⁻¹ (Δ ps) from ps[0]` gives `ps`",
        ),
        "sum_before" | "sum_before_from" => err(
            solver,
            at,
            "`Δ⁻¹` adds up every step, and posts do not add",
            "this is a stream of posts",
            "take its panels first: `Δ⁻¹ (Δ ps) from ps[0]` gives `ps` back",
        ),
        "delay" => err(
            solver,
            at,
            "`z⁻¹` on a stream of posts needs the post before the first",
            "posts have no zero to stand in for it",
            "name the start: `z⁻¹ ps from p`",
        ),
        _ => err(
            solver,
            at,
            "summing a stream of posts: posts do not add",
            "this is a stream of posts",
            "sum its panels: `∑ (Δ ps)` is `ps.last() - ps[0]`",
        ),
    }
}

/// How many steps `e` has, as a base stream and an offset from its length:
/// `Δ s` is `(s, -1)`, `Δ⁻¹ c` is `(c, +1)`. `None` when `e` is not a chain
/// of stream glyphs over one named stream.
fn steps(solver: &mut Solver, e: &Expr) -> Option<(String, i32)> {
    match e.kind() {
        ExprKind::Grouping(g) => steps(solver, &g.inner),
        ExprKind::Ident(i) => Some((i.lexeme.clone(), 0)),
        ExprKind::Unary(u) if u.op == UnaryOp::Negative => steps(solver, &u.right),
        ExprKind::Access(Access::Dot { left, right, .. }) => {
            let (base, off) = steps(solver, left)?;
            let field = &right.as_ident()?.lexeme;
            (off == 0).then(|| (format!("{base}.{field}"), 0))
        }
        ExprKind::Call(c) => {
            let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind() else {
                return None;
            };
            let name = right.as_ident()?.lexeme.as_str();
            let shift = match name {
                "forward_difference" | "before" | "after" | "shift" => -1,
                // `∇` keeps a stream of panels' length (step 0 reads the
                // zero) and shortens a stream of posts by one
                "differentiate" => {
                    let ty = left.query(solver).ok()?.normalized(solver);
                    if kind(solver, &ty) == Some(Kind::Posts) {
                        -1
                    } else {
                        0
                    }
                }
                "integrate" | "integrate_from" | "differentiate_from" | "delay" | "delay_from"
                | "map" | "neg" => 0,
                "sum_before" | "sum_before_from" => 1,
                _ => return None,
            };
            let (base, off) = steps(solver, left)?;
            Some((base, off + shift))
        }
        _ => None,
    }
}

/// `a.zip(b)` on two streams built from the same stream by glyphs whose
/// lengths differ: the fencepost error, caught before the run. Lengths
/// known only at run time are the library's check.
// r[impl mimas.group.affine.zip]
fn zip(solver: &mut Solver, left: &Expr, call: &Call) -> Result<()> {
    let [arg] = call.arguments.as_slice() else {
        return Ok(());
    };
    let lt = left.query(solver)?.normalized(solver);
    if kind(solver, &lt).is_none_or(|k| k == Kind::Post) {
        return Ok(());
    }
    let (Some((a, da)), Some((b, db))) = (steps(solver, left), steps(solver, &arg.value)) else {
        return Ok(());
    };
    if a != b || da == db {
        return Ok(());
    }
    let len = |d: i32| match d {
        0 => "n".to_string(),
        d if d < 0 => format!("n - {}", -d),
        d => format!("n + {d}"),
    };
    let label = format!(
        "with n the steps of `{a}`, this side has {} and the other {}",
        len(da),
        len(db)
    );
    Err(err(
        solver,
        call.left.location(),
        "zipping two streams of different lengths: which alignment is meant?",
        &label,
        "`n` posts have `n - 1` panels. Pair each post with the panel after it with \
         `ps.before().zip(Δ ps)`, or with the panel that made it with `ps.after().zip(∇ ps)`",
    ))
}

/// A mismatch between a stream of panels and a stream of posts, or a panel
/// and a post: the hint that names the missing start. `None` for any other
/// pair of types.
// r[impl mimas.group.affine.from]
pub(crate) fn mismatch(
    solver: &Solver,
    expected: &str,
    found: &str,
    at: Location,
) -> Option<Error> {
    // only the generic std spellings: a page's own `struct Post` prints
    // with no `<…>`, and gets no hint
    let head = |s: &str| match s.split_once('<') {
        Some((h, _)) => h.to_string(),
        None => String::new(),
    };
    let (e, f) = (head(expected), head(found));
    let (msg, help) = match (e.as_str(), f.as_str()) {
        ("Posts", "Stream") => (
            "a stream of panels where a stream of posts goes: name the starting post",
            "`∫ c` starts from the zero and gives panels. Start from a post instead: \
             `∫ c from p` (`p` plus each running total), or `Δ⁻¹ c from p` (with `p` as post 0)",
        ),
        ("Stream", "Posts") => (
            "a stream of posts where a stream of panels goes",
            "take the panels between the posts with `Δ ps`, or the panel from one post to each \
             with `ps - p`",
        ),
        ("Post", _) if f != "Post" && kind_free(found) => (
            "a panel where a post goes",
            "make a post from a coordinate with `post(x)`, or add the panel to a post: `p + d`",
        ),
        (_, "Post") if e != "Post" && kind_free(expected) => (
            "a post where a panel goes",
            "subtract another post to get a panel (`p - q`), or read its coordinate with `p.at`",
        ),
        _ => return None,
    };
    Some(err(
        solver,
        at,
        msg,
        &format!("expected {expected} but found {found}"),
        help,
    ))
}

/// A type spelling with no unsolved parts: a hint about a post is only
/// sure when the other side is a settled type.
fn kind_free(s: &str) -> bool {
    !s.contains('?') && !s.contains('{') && !s.is_empty()
}
