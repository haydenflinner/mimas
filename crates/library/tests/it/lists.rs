// Slicing, sorted copies, and dict keys and values.
// r[verify mimas.slice] r[verify mimas.sorted] r[verify mimas.dict.keys]

test_run!(
    lists_slice_and_sort,
    "[10, 20, 30, 40][1..3]" => "[20, 30]",
    "[10, 20, 30, 40][1..=2]" => "[20, 30]",
    "[10, 20, 30, 40][2..]" => "[30, 40]",
    "[10, 20, 30, 40][..2]" => "[10, 20]",
    "[1, 2, 3][1..99]" => "[2, 3]",
    "[1, 2, 3][2..1]" => "[]",
    "\"nina family\"[0..4]" => "\"nina\"",
    "\"nina family\"[5..]" => "\"family\"",
    "[3, 1, 2].sorted()" => "[1, 2, 3]",
    "[2.5, -1.0, 0.5].sorted()" => "[-1.0, 0.5, 2.5]",
    "[\"pear\", \"fig\", \"apple\"].sorted()" => "[\"apple\", \"fig\", \"pear\"]",
);

test_run!(
    a_sorted_copy_leaves_the_list,
    "let xs = [3, 1, 2];\nlet ys = xs.sorted();\nlet d: {int} = { ann: 31, bo: 7 };",
    "xs" => "[3, 1, 2]",
    "ys" => "[1, 2, 3]",
    "d.keys()" => "[\"ann\", \"bo\"]",
    "d.values()" => "[31, 7]",
);
