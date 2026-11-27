//! The failure rule and small gaps (L8).

test_run!(
    // r[verify mimas.str.parse-option]
    // `to_int`, `to_dec` and `to_float` on text agree: `null` for bad input
    text_parses_to_an_option,
    "\"12\".to_int()!" => "12",
    "\"x\".to_int() == null" => "true",
    "\"x\".to_dec() == null" => "true",
    "\"3.50\".to_dec()!.to_str()" => "\"3.50\"",
    "\" 2.5 \".to_float()!" => "2.5",
    "\"two\".to_float() == null" => "true",
);

test_run!(
    // r[verify mimas.str.re-find]
    regex_find_has_its_own_name,
    "\"x=1\".re_find(\"(\\\\w)=(\\\\d)\")!.len()" => "3",
    "\"x=1, y=2\".re_find_all(\"(\\\\w)=(\\\\d)\")!.len()" => "2",
);

test_run!(
    // r[verify mimas.str.bytes]
    text_to_bytes_and_back,
    "\"hé\".bytes()" => "[104, 195, 169]",
    "str::from_bytes(\"hé\".bytes())!" => "\"hé\"",
    "str::from_bytes([255]) == null" => "true",
);

test_run!(
    // r[verify mimas.list.filled]
    // r[verify mimas.list.shuffled]
    // r[verify mimas.dict.is-empty]
    fill_shuffle_and_empty,
    "filled(3, 0)" => "[0, 0, 0]",
    "filled(0, 1).len()" => "0",
    "[3, 1, 2].shuffled().sorted()" => "[1, 2, 3]",
    "{ let xs = [1, 2]; let ys = xs.shuffled(); xs }" => "[1, 2]",
    "dict::new().is_empty()" => "true",
    "{\"a\": 1}.is_empty()" => "false",
);
