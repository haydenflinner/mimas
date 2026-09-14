//! `std::polars` -- `to_dataframe`/`col`/`filter`/`select`/`sort` over a real polars `DataFrame`.
//! See `crates/library/src/std_lib/dataframe.rs`.
//!
//! `DataFrame`/`PlExpr` are real `Val` variants, not `#[mimas] struct`s, so the harness's
//! `Captured` inspection (used by `test_runner::render`) can't see inside them -- it reports them
//! as `Captured::Other` by design (see the comment on that variant). Every case here goes through
//! an f-string (`f"{df}"`) instead, which routes through the VM's own `Display` (the same path
//! `print` uses) and hands the harness a plain `Captured::Str` to compare. A quote *inside* an
//! f-string interpolation doesn't need escaping -- `Parser::construct_fstring` tracks `{...}`
//! depth precisely so a nested string (`f"{col("age")}"`) opens its own string instead of closing
//! the f-string early.

#[macro_use]
mod test_runner;

const EMPLOYEES: &str = "use std::polars::*;
     struct Employee { name: str, age: int, dept: str }
     let rows = [
         Employee { name = \"Alice\", age = 34, dept = \"eng\" },
         Employee { name = \"Bob\", age = 29, dept = \"sales\" },
         Employee { name = \"Carol\", age = 41, dept = \"eng\" },
         Employee { name = \"Dave\", age = 25, dept = \"sales\" },
     ];
     let df = to_dataframe(rows)!;";

test_run!(
    to_dataframe_builds_columns_named_after_struct_fields,
    EMPLOYEES,
    r#"f"{df}""# => "\"shape: (4, 3)\\n┌───────┬─────┬───────┐\\n│ name  ┆ age ┆ dept  │\\n│ ---   ┆ --- ┆ ---   │\\n│ str   ┆ i64 ┆ str   │\\n╞═══════╪═════╪═══════╡\\n│ Alice ┆ 34  ┆ eng   │\\n│ Bob   ┆ 29  ┆ sales │\\n│ Carol ┆ 41  ┆ eng   │\\n│ Dave  ┆ 25  ┆ sales │\\n└───────┴─────┴───────┘\"",
);

test_run!(
    filter_builds_a_real_expr_tree_via_gt,
    EMPLOYEES,
    r#"f"{df.filter(col("age") > 30)!}""# => "\"shape: (2, 3)\\n┌───────┬─────┬──────┐\\n│ name  ┆ age ┆ dept │\\n│ ---   ┆ --- ┆ ---  │\\n│ str   ┆ i64 ┆ str  │\\n╞═══════╪═════╪══════╡\\n│ Alice ┆ 34  ┆ eng  │\\n│ Carol ┆ 41  ┆ eng  │\\n└───────┴─────┴──────┘\"",
);

test_run!(
    filter_combines_predicates_with_single_ampersand,
    // `&`, not `&&` -- mimas's `&&`/`||` short-circuit and hard-require `Bool` on both sides
    // (see `Logical::solve`), so they never reach `bin()`'s `PlExpr` overload at all.
    EMPLOYEES,
    r#"f"{df.filter((col("age") > 25) & (col("dept") == "eng"))!}""# => "\"shape: (2, 3)\\n┌───────┬─────┬──────┐\\n│ name  ┆ age ┆ dept │\\n│ ---   ┆ --- ┆ ---  │\\n│ str   ┆ i64 ┆ str  │\\n╞═══════╪═════╪══════╡\\n│ Alice ┆ 34  ┆ eng  │\\n│ Carol ┆ 41  ┆ eng  │\\n└───────┴─────┴──────┘\"",
);

test_run!(
    select_narrows_to_the_chosen_columns,
    EMPLOYEES,
    r#"f"{df.filter(col("age") > 30)!.select([col("name"), col("age")])!}""# => "\"shape: (2, 2)\\n┌───────┬─────┐\\n│ name  ┆ age │\\n│ ---   ┆ --- │\\n│ str   ┆ i64 │\\n╞═══════╪═════╡\\n│ Alice ┆ 34  │\\n│ Carol ┆ 41  │\\n└───────┴─────┘\"",
);

test_run!(
    sort_orders_ascending_by_column_name,
    EMPLOYEES,
    r#"f"{df.sort(["age"])!}""# => "\"shape: (4, 3)\\n┌───────┬─────┬───────┐\\n│ name  ┆ age ┆ dept  │\\n│ ---   ┆ --- ┆ ---   │\\n│ str   ┆ i64 ┆ str   │\\n╞═══════╪═════╪═══════╡\\n│ Dave  ┆ 25  ┆ sales │\\n│ Bob   ┆ 29  ┆ sales │\\n│ Alice ┆ 34  ┆ eng   │\\n│ Carol ┆ 41  ┆ eng   │\\n└───────┴─────┴───────┘\"",
);

// no rows -> nothing to infer a schema from
test_fail!(
    to_dataframe_empty_array_raises,
    r#"use std::polars::*; let df = to_dataframe([])!;"#,
);

// not struct instances -> can't discover field names to name columns
test_fail!(
    to_dataframe_non_struct_elements_raises,
    r#"use std::polars::*; let df = to_dataframe([1, 2, 3])!;"#,
);
