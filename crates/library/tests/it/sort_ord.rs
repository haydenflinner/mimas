//! Sorting and picking over `Ord` (L5).

const P: &str = "struct P { name: str, age: int }
impl Ord for P { fn cmp(self, o: P) -> int { self.age - o.age } }
let ps = [P { name = \"b\", age = 30 }, P { name = \"a\", age = 20 }, P { name = \"c\", age = 30 }];";

test_run!(
    // r[verify mimas.list.sort-ord]
    a_list_of_an_ord_type_sorts,
    P,
    "ps.sorted().map(|p| p.name)" => "[\"a\", \"b\", \"c\"]",
    "ps.max()!.name" => "\"b\"",
    "ps.min()!.name" => "\"a\"",
    // the list itself is unchanged
    "ps.map(|p| p.name)" => "[\"b\", \"a\", \"c\"]",
);

test_run!(
    // r[verify mimas.list.sort-ord]
    by_a_key_on_any_list,
    P,
    "ps.sorted_by(|p| p.name).map(|p| p.age)" => "[20, 30, 30]",
    "ps.max_by(|p| p.name)!.name" => "\"c\"",
    "ps.min_by(|p| p.age)!.name" => "\"a\"",
    "[3, 1, 2].sorted_by(|x| 0 - x)" => "[3, 2, 1]",
    "[\"bb\", \"a\", \"ccc\"].max_by(|s| s.len())!" => "\"ccc\"",
    // the built-in sorted still answers for ints
    "[3, 1, 2].sorted()" => "[1, 2, 3]",
);

test_run!(
    // r[verify mimas.list.reversed]
    reversed_and_reverse,
    "let xs = [1, 2, 3];",
    "xs.reversed()" => "[3, 2, 1]",
    "xs" => "[1, 2, 3]",
    "{ let ys = [1, 2, 3]; ys.reverse(); ys }" => "[3, 2, 1]",
    "\"abc\".reversed()" => "\"cba\"",
);
