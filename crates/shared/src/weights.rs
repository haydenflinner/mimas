//! The four weight kinds of a frame, shared by the checker (`solve`), the
//! runtime (`library::zset`) and the display (`vm`), so each says the same
//! thing about a table and gives the same error text.
//!
//! A frame keeps its weights in a column with a name no page can write,
//! [`WEIGHT_COL`]. Its type is the kind:
//!
//! | weight column | kind |
//! |---|---|
//! | none | plain table: a bag, one row each, repeats allowed |
//! | `bool` | set: presence, each row there or not |
//! | `int` | Z-set: whole-number weights, a row may count negative |
//! | `float` | weighted table: real weights |
//!
//! A plain table with a column of its own named `w` stays a plain table
//! until `query z { t }` lifts it; the lift reads that `w` as the weights.
//! Nothing crosses from one kind to another on its own, with one
//! exception: a set is presence, so it counts as weight 1 wherever a
//! weight is needed. Questions about presence (`x ∈ z`, emptiness) need
//! no conversion, since they give a `bool`, not a table.
//! r[impl mimas.query.zset.kinds]

/// The weight column's name inside a frame. A page writes `w` for it
/// inside `query z { … }`, and the display shows it as `w`.
pub const WEIGHT_COL: &str = "⟨w⟩";

/// The name a page writes for the weights inside `query z`, and the
/// column of its own that a plain table brings when `query z` lifts it.
pub const USER_W: &str = "w";

/// What a frame's weights are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// No weights: a bag of rows.
    Plain,
    /// Boolean weights: each row is there or not.
    Set,
    /// Whole-number weights.
    ZSet,
    /// Real weights.
    Weighted,
}

impl Kind {
    /// The kind's name in prose: "a {name}".
    pub fn name(self) -> &'static str {
        match self {
            Kind::Plain => "plain table",
            Kind::Set => "set",
            Kind::ZSet => "Z-set",
            Kind::Weighted => "weighted table",
        }
    }
}

/// Whether `a` and `b` may meet in a binary operation without a written
/// conversion: the same kind, or a set (presence) beside weights. Two
/// plain tables may only meet in `==`.
pub fn compatible(a: Kind, b: Kind) -> bool {
    use Kind::*;
    match (a, b) {
        (Plain, _) | (_, Plain) => false,
        (Set, _) | (_, Set) => true,
        (a, b) => a == b,
    }
}

/// The kind of a weight column whose type is `bool`, `int` or `float`.
pub fn kind_of_weight(is_bool: bool, is_int: bool, is_float: bool) -> Option<Kind> {
    if is_bool {
        Some(Kind::Set)
    } else if is_int {
        Some(Kind::ZSet)
    } else if is_float {
        Some(Kind::Weighted)
    } else {
        None
    }
}

/// The kind that a weighted operation on `a` and `b` gives, when they are
/// [`compatible`]: a set promotes to the other side's kind, and two sets
/// give a Z-set (their counts add).
pub fn joined(a: Kind, b: Kind) -> Kind {
    match (a, b) {
        (Kind::Set, Kind::Set) => Kind::ZSet,
        (Kind::Set, k) | (k, Kind::Set) => k,
        (k, _) => k,
    }
}

/// The phrase for `op` between a `a` (left) and a `b` (right).
fn phrase(op: &str, a: Kind, b: Kind) -> String {
    let (a, b) = (a.name(), b.name());
    match op {
        "-" => format!("subtracting a {b} from a {a}"),
        "==" | "!=" | "≈" => format!("comparing a {a} with a {b}"),
        "⋈" | "join" => format!("joining a {a} with a {b}"),
        "∪" => format!("the union of a {a} and a {b}"),
        "∩" => format!("the intersection of a {a} and a {b}"),
        "changes" => format!("the changes from a {b} to a {a}"),
        _ => format!("adding a {a} and a {b}"),
    }
}

/// The written conversion that makes `a` and `b` meet.
fn fix(a: Kind, b: Kind) -> &'static str {
    use Kind::*;
    let pair = |x: Kind, y: Kind| (a == x && b == y) || (a == y && b == x);
    if pair(Plain, Plain) {
        "a plain table has no weights. Lift both with `query z { t }` to add them as Z-sets"
    } else if pair(ZSet, Plain) {
        "lift the table with `query z { t }`, or drop the weights with `.unweighted()`"
    } else if pair(Weighted, Plain) {
        "lift the table with `query z { t }.weighted()`, or drop the weights with `.unweighted()`"
    } else if pair(Set, Plain) {
        "make the table a set with `.as_set()`, or lift it with `query z { t }`"
    } else {
        "lift the Z-set with `.weighted()`, or make the weights counts with `.counts()`"
    }
}

/// The error for `op` on a `a` and a `b` that are not [`compatible`]:
/// "comparing a Z-set with a plain table: lift the table with
/// `query z { t }`, or drop the weights with `.unweighted()`".
/// r[impl mimas.query.zset.kind-mix]
pub fn mix_message(op: &str, a: Kind, b: Kind) -> String {
    format!("{}: {}", phrase(op, a, b), fix(a, b))
}

/// The error for a weighted operation (`-t`, `k * t`, `‖t‖`, `t⁺`) on a
/// plain table. `what` names the operation: "negating", "scaling", ….
pub fn plain_message(what: &str) -> String {
    format!(
        "{what} a plain table: a plain table has no weights. Lift it with `query z {{ t }}` \
         to give each row weight 1"
    )
}

/// The error for a plain `query { … }` that joins a table with weights.
pub fn plain_join_message(kind: Kind) -> String {
    format!(
        "a plain query joins plain tables, and this one is a {}: drop the weights with \
         `.unweighted()`, make them a column with `.weights_as(\"n\")`, or join inside \
         `query z {{ … }}`",
        kind.name()
    )
}

/// The error for `k * z` with a real `k` on a Z-set.
/// r[impl mimas.query.zset.scale-real]
pub fn float_scale_message() -> String {
    "scaling a Z-set by a float: a Z-set's weights are counts. Write `k · z.weighted()` for \
     real weights, or scale by an int"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_names_the_conversion() {
        assert_eq!(
            mix_message("==", Kind::ZSet, Kind::Plain),
            "comparing a Z-set with a plain table: lift the table with `query z { t }`, \
             or drop the weights with `.unweighted()`"
        );
        assert!(mix_message("+", Kind::ZSet, Kind::Weighted).contains(".weighted()"));
        assert!(compatible(Kind::Set, Kind::Weighted));
        assert!(!compatible(Kind::ZSet, Kind::Weighted));
        assert!(!compatible(Kind::Plain, Kind::Plain));
        assert_eq!(joined(Kind::Set, Kind::Set), Kind::ZSet);
        assert_eq!(joined(Kind::Set, Kind::Weighted), Kind::Weighted);
    }
}
