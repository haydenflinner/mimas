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

// x07 (combo C16): `a ⋅ b` on two lists is the dot product; its unit is
// the product of the item units; lists of two lengths are an error.
// r[verify mimas.glyph.dot.list]
test_run!(
    dot_of_two_lists,
    "let a = [1.0, 2.0]\nlet b = [3.0, 4.0]",
    "a ⋅ b" => "11.0",
    "[1, 2, 3] ⋅ [4, 5, 6]" => "32",
    "a.dot(b)" => "11.0",
);

// r[verify mimas.glyph.dot.list]
#[test]
fn dot_of_lists_keeps_units_and_lengths() {
    let src = "let a: [px] = [1.0px, 2.0px]\nlet bad = (a ⋅ a) + 1.0s\n";
    let err = format!("{:?}", vm::Vm::compile(src, library::std).err().expect("refused"));
    assert!(err.contains("px²"), "{err}");
    let run = vm::Vm::execute("let d = [1.0] ⋅ [1.0, 2.0]\n", library::std);
    assert!(format!("{:?}", run.err().expect("faults")).contains("one length"));
}
