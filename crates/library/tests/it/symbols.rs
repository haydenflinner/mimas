//! The math spellings: `≠ ≤ ≥ ¬ ∈ ∉`, the prefix reductions `∑ ∏ ⨅ ⨆`,
//! pervasive list-vs-scalar comparison and list-indexed gather. See the
//! book's Operators and Arrays pages. (`∧` `∨` `⋅` are named operators —
//! `wedge`, `vee`, `dot` methods — not spellings of `and` `or` `*`.)

test_run!(
    glyph_operators,
    "1 ≤ 2 and ¬ (3 ≥ 4)" => "true",
    "2 ≠ 2 or 1 ≠ 2" => "true",
    "3 ∈ [1, 2, 3]" => "true",
    "4 ∉ [1, 2, 3]" => "true",
    "for x in [1, 2, 3] { if x ≠ 2 { collect x; } }" => "[1, 3]",
);

test_run!(
    prefix_reductions,
    "∑ [1, 2, 3]" => "6",
    "∑ [1.5, 2.5]" => "4.0",
    "∏ [2, 3, 4]" => "24",
    "⨅ [true, true]" => "true",
    "⨅ [true, false]" => "false",
    "⨆ [false, true]" => "true",
    "⨆ [false, false]" => "false",
    // postfix binds first, as after a unary minus: `⨅ xs!` unwraps `xs`
    "(⨅ [3, 1, 2])!" => "1",
    "(⨆ [3, 1, 2])!" => "3",
    "(⨆ [0.5, 2.5])!" => "2.5",
    "∑ [1, 2, 3].map(|x| x * x)" => "14",
    // a list of lists reduces its innermost axis: one answer per row
    "∑ [[1, 2], [3, 4]]" => "[3, 7]",
    "⨅ [[true, false], [true, true]]" => "[false, true]",
    "⨆ [[1, 5], [2, 2]]" => "[5, 2]",
);

test_run!(
    pervasive_comparison,
    "[1, 2, 3] == 2" => "[false, true, false]",
    "[1, 2, 3] ≠ 2" => "[true, false, true]",
    "2 == [1, 2, 3]" => "[false, true, false]",
    "[[1, 2], [2, 2]] == 2" => "[[false, true], [true, true]]",
    "[1, 2] == [1, 2]" => "true",
    "⨆ ([1, 2, 3] == 2)" => "true",
);

test_run!(
    gather,
    "[7, 8, 9][[2, 0]]" => "[9, 7]",
    "[7, 8, 9][[[0, 1], [2, 2]]]" => "[[7, 8], [9, 9]]",
);

// tic-tac-toe's win check, the whole thing: gather the lines, compare, meet
// each line, join the lines. The players are an `opp`, so `!p` is the other one.
test_run!(
    tic_tac_toe,
    "opp Player { X, O }
     const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
     fn won(b: [Player?], p: Player) -> bool { ⨆⨅ (b[LINES] == p) }
     fn won_at(b: [Player?], p: Player, c: int) -> bool {
         ⨆ for ln in LINES { if c ∈ ln { collect ⨅ (b[ln] == p); } }
     }
     let x = Player::X;
     let o = Player::O;",
    "won([x, x, x, null, o, o, null, null, null], x)" => "true",
    "won([x, x, o, null, o, o, null, null, null], x)" => "false",
    "won([o, x, x, null, o, x, null, null, o], o)" => "true",
    "won_at([x, x, x, null, o, o, null, null, null], x, 1)" => "true",
    "won_at([x, x, x, null, o, o, null, null, null], o, 4)" => "false",
    "won([o, x, x, null, o, x, null, null, o], !x)" => "true",
);
