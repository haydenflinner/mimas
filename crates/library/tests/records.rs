// Bare-brace literals: `{ x: 1, y: 2 }` is whatever its slot expects — the
// struct a `let` annotation, a parameter, a field, a return type or a list's
// element type names — and a dict when nothing says otherwise. Entries take
// `:` or `=`; a lone `{ x = 1 }` stays the block that assigns `x`.
#[macro_use]
mod test_runner;

const PT: &str = "struct Pt { x: float, y: float }
struct Seg { a: Pt, b: Pt }
fn xsum(p: Pt) -> float { p.x + p.y }
fn origin() -> Pt { { x: 0.0, y: 0.0 } }";

// r[verify mimas.record.context]
test_run!(
    records_fill_the_struct_their_slot_expects,
    PT,
    // a `let` annotation
    "{ let p: Pt = { x: 1.0, y: 2.0 }; p.x + p.y }" => "3.0",
    // a parameter
    "xsum({ x: 3.0, y: 4.0 })" => "7.0",
    // a return type
    "origin().y" => "0.0",
    // a list's element type, through the list literal
    "{ let ps: [Pt] = [{ x: 1.0, y: 1.0 }, { x: 2.0, y: 5.0 }]; ps[1].y }" => "5.0",
    // an option slot
    "{ let p: Pt? = { x: 9.0, y: 0.0 }; p!.x }" => "9.0",
    // a struct's own field, nested
    "{ let s: Seg = { a: { x: 0.0, y: 0.0 }, b: { x: 1.0, y: 8.0 } }; s.b.y }" => "8.0",
);

// r[verify mimas.record.literal] r[verify mimas.record.separator]
test_run!(
    records_are_dicts_by_default,
    "{ let d = { a: 1, b: 2 }; d[\"b\"]! }" => "2",
    // `=` works once a comma follows the first entry
    "{ let d = { a = 1, b = 2 }; d[\"a\"]! }" => "1",
    "{ let d = { a = 5, }; d[\"a\"]! }" => "5",
    // shorthand
    "{ let a = 3; let b = 4; let d = { a, b }; d[\"b\"]! }" => "4",
    // a quoted key holds spaces
    "{ let d = { \"brown rice\": 4, oats: 3 }; d[\"brown rice\"]! }" => "4",
    // commas stay optional between `:` entries
    "{ let d = { a: 1 b: 2 }; d[\"b\"]! }" => "2",
);

// r[verify mimas.dict.type]
test_run!(
    the_dict_type_needs_no_tilde,
    "fn total(d: {int}) -> int { d[\"a\"]! + d[\"b\"]! }",
    "total({ a: 1, b: 2 })" => "3",
    "{ let e: {int} = ~{}; e.len() }" => "0",
    // inside a tuple type and an enum payload too
    "{ let t: (int, {int}) = (1, { a: 2 }); t.1[\"a\"]! }" => "2",
);

// r[verify mimas.dict.type]
test_run!(
    an_enum_payload_takes_the_dict_type,
    "enum Box { Bag({int}), Empty }
     fn size(b: Box) -> int { match b { Box::Bag(d) => d.len(), Box::Empty => 0 } }",
    "size(Box::Bag({ a: 1, b: 2 }))" => "2",
);

// r[verify mimas.record.separator]
test_run!(
    a_lone_assignment_stays_a_block,
    // `{ x = 5 }` assigns — closures and match arms rely on it
    "{ let x = 0; let f = || { x = 5 }; f(); x }" => "0",
    "{ let x = 0; { x = 5 }; x }" => "5",
    "{ let y = { 7 }; y }" => "7",
);

// r[verify mimas.record.separator]
test_run!(
    named_literals_take_colons_too,
    PT,
    "Pt { x: 1.0, y: 2.0 }.y" => "2.0",
    "Pt { x = 1.0, y: 2.0 }.x" => "1.0",
    "~{ a: 1, b = 2 }[\"a\"]!" => "1",
);

// r[verify mimas.record.context]
test_fail!(
    record_mistakes_are_compile_errors,
    // mixed values and no struct to fill
    "let d = { a: 1, b: \"x\" };",
    // a field the struct doesn't have
    "struct Pt { x: float } let p: Pt = { x: 1.0, z: 2.0 };",
    // a field it needs, missing
    "struct Pt { x: float, y: float } let p: Pt = { x: 1.0 };",
    // the declared unit is checked through the record
    "struct Item { qty: lb } let i: Item = { qty: 2usd };",
);

// r[verify mimas.record.const]
test_run!(
    records_fold_in_consts,
    "struct V { x: float, y: float }
     struct W { v: V }
     struct Piece { cells: [V] }
     const B: V = { x: 1.0, y: 2.0 };
     const D = W { v: { x: 0.0, y: 5.0 } }
     const P = Piece { cells: [{ x: 0.0, y: -1.0 }, { x: 1.0, y: 1.0 }] }
     const TAGS = { a: 1, b: 2 }",
    // annotated: the record is the struct the annotation names
    "B.y" => "2.0",
    // in a struct field, before the struct's own types are solved
    "D.v.y" => "5.0",
    // in a list in a struct field
    "P.cells[1].x" => "1.0",
    // nothing names a struct: a dict
    "TAGS[\"b\"]!" => "2",
);
