//! Combinators over anything that iterates (L6).

const BAG: &str = "struct Bag { xs: [int] }
impl Bag {
    fn iter(self) -> () -> int? {
        let at = [0];
        let xs = self.xs;
        || { if at[0] < xs.len() { let v = xs[at[0]]; at[0] += 1; v } else { null } }
    }
}
let bag = Bag { xs = [3, 1, 2] };
fn count_to(n: int) -> () -> int? {
    let at = [0];
    || { if at[0] < n { at[0] += 1; at[0] } else { null } }
}";

test_run!(
    // r[verify mimas.iter.combinators]
    combinators_on_a_type_with_iter,
    BAG,
    "bag.map(|x| x * 2).to_list()" => "[6, 2, 4]",
    "bag.filter(|x| x > 1).to_list()" => "[3, 2]",
    "bag.fold(0, |a, x| a + x)" => "6",
    "bag.find(|x| x < 3)!" => "1",
    "bag.any(|x| x == 2)" => "true",
    "bag.all(|x| x > 0)" => "true",
    "bag.enumerate().to_list()" => "[[0, 3], [1, 1], [2, 2]]",
    "bag.zip(bag).map(|p: (int, int)| p.0 + p.1).to_list()" => "[6, 2, 4]",
    "bag.flat_map(|x| [x, x]).to_list()" => "[3, 3, 1, 1, 2, 2]",
    "bag.to_list()" => "[3, 1, 2]",
);

test_run!(
    // r[verify mimas.iter.combinators]
    combinators_on_a_pull,
    BAG,
    "count_to(4).map(|x| x * x).to_list()" => "[1, 4, 9, 16]",
    "count_to(5).filter(|x| x % 2 == 0).to_list()" => "[2, 4]",
    "[1, 2].to_list()" => "[1, 2]",
);

test_run!(
    // r[verify mimas.iter.combinators]
    a_stream_iterates,
    "let s = stream([1, 2, 3]);",
    "{ let t = 0; for x in s { t += x; } t }" => "6",
    "s.filter(|x| x != 2).to_list()" => "[1, 3]",
    "s.to_list()" => "[1, 2, 3]",
);

