#[macro_use]
mod test_runner;

test_run!(
    iso_round_trip,
    "let p = iso(|s: str| s.to_int()!, |n: int| f\"{n}\");",
    "un(p)(42)" => "\"42\"",
    "un(p)(7)" => "\"7\"",
);

test_run!(
    under_iso_applies_f_in_target_space,
    "let p = iso(|s: str| s.to_int()!, |n: int| f\"{n}\");",
    // to: str -> int, edit: +1, from: int -> str
    "under(p, |n| n + 1, \"41\")" => "\"42\"",
    "under(p, |n| n * 2, \"3\")" => "\"6\"",
);

test_run!(
    under_curried_returns_fn,
    "let p = iso(|s: str| s.to_int()!, |n: int| f\"{n}\");",
    "under(p, |n| n + 1)(\"41\")" => "\"42\"",
);

test_run!(
    at_lens_updates_index,
    "let xs = [8, 3, 9, 2, 0];",
    "under(at(2), |x| x * 10, xs)" => "[8, 3, 90, 2, 0]",
    "under(at(0), |x| x + 1, xs)" => "[9, 3, 9, 2, 0]",
);

test_run!(
    at_lens_leaves_source_untouched,
    "let xs = [8, 3, 9, 2, 0];
     under(at(2), |x| x * 10, xs);",
    "xs" => "[8, 3, 9, 2, 0]",
);

test_run!(
    lens_user_defined,
    "let first = lens(|xs| (xs[0], xs), |ctx, f| [f, ctx[1], ctx[2]]);",
    "under(first, |x| x + 100, [1, 2, 3])" => "[101, 2, 3]",
);

test_run!(
    un_compose_with_natives,
    "use std::parse::{from_json, to_json, Value};
     let codec = iso(|s: str| from_json(s)!, |v: Value| to_json(v)!);",
    "under(codec, |v| v, \"[1, 2, 3]\")" => "\"[1,2,3]\"",
);

test_fail!(
    un_on_lens_errors,
    "let l = at(0);
     un(l);",
    "let l = lens(|xs| (xs[0], xs), |ctx, f| ctx);
     un(l);",
);

test_fail!(
    iso_rejects_non_callable,
    "iso(1, 2);",
);
