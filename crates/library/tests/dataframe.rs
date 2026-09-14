//! `std::polars` -- `to_dataframe`/`col`/`filter`/`select`/`sort` over a real polars `DataFrame`.
//! See `crates/library/src/std_lib/dataframe.rs`.
//!
//! `DataFrame`/`PlExpr` are real `Val` variants, not `#[mimas] struct`s, so the harness's
//! `Captured` inspection (used by `test_run!`/`render`) can't see inside them -- it reports them
//! as `Captured::Other` by design. `test_run_display!`/`render_display` sidesteps that by reading
//! the value back through its real `Display` (the same path `print` uses) instead, which also
//! means the expected side here is the plain, unescaped table text.

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

test_run_display!(
    to_dataframe_builds_columns_named_after_struct_fields,
    EMPLOYEES,
    "df" => r#"shape: (4, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
│ Carol ┆ 41  ┆ eng   │
│ Dave  ┆ 25  ┆ sales │
└───────┴─────┴───────┘"#,
);

test_run_display!(
    filter_builds_a_real_expr_tree_via_gt,
    EMPLOYEES,
    r#"df.filter(col("age") > 30)!"# => r#"shape: (2, 3)
┌───────┬─────┬──────┐
│ name  ┆ age ┆ dept │
│ ---   ┆ --- ┆ ---  │
│ str   ┆ i64 ┆ str  │
╞═══════╪═════╪══════╡
│ Alice ┆ 34  ┆ eng  │
│ Carol ┆ 41  ┆ eng  │
└───────┴─────┴──────┘"#,
);

test_run_display!(
    filter_combines_predicates_with_single_ampersand,
    // `&`, not `&&` -- mimas's `&&`/`||` short-circuit and hard-require `Bool` on both sides
    // (see `Logical::solve`), so they never reach `bin()`'s `PlExpr` overload at all.
    EMPLOYEES,
    r#"df.filter((col("age") > 25) & (col("dept") == "eng"))!"# => r#"shape: (2, 3)
┌───────┬─────┬──────┐
│ name  ┆ age ┆ dept │
│ ---   ┆ --- ┆ ---  │
│ str   ┆ i64 ┆ str  │
╞═══════╪═════╪══════╡
│ Alice ┆ 34  ┆ eng  │
│ Carol ┆ 41  ┆ eng  │
└───────┴─────┴──────┘"#,
);

test_run_display!(
    select_narrows_to_the_chosen_columns,
    EMPLOYEES,
    r#"df.filter(col("age") > 30)!.select([col("name"), col("age")])!"# => r#"shape: (2, 2)
┌───────┬─────┐
│ name  ┆ age │
│ ---   ┆ --- │
│ str   ┆ i64 │
╞═══════╪═════╡
│ Alice ┆ 34  │
│ Carol ┆ 41  │
└───────┴─────┘"#,
);

test_run_display!(
    sort_orders_ascending_by_column_name,
    EMPLOYEES,
    r#"df.sort(["age"])!"# => r#"shape: (4, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Dave  ┆ 25  ┆ sales │
│ Bob   ┆ 29  ┆ sales │
│ Alice ┆ 34  ┆ eng   │
│ Carol ┆ 41  ┆ eng   │
└───────┴─────┴───────┘"#,
);

const CSV: &str = r#"use std::polars::*;
     let csv = "name,age,dept
Alice,34,eng
Bob,29,sales
";
     let df = from_csv(csv)!;"#;

test_run_display!(
    from_csv_infers_columns_from_the_header_row,
    CSV,
    "df" => r#"shape: (2, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
└───────┴─────┴───────┘"#,
);

// a from_csv DataFrame interops with the rest of std::polars just like a to_dataframe one
test_run_display!(
    from_csv_dataframe_can_be_filtered,
    CSV,
    r#"df.filter(col("age") > 30)!"# => r#"shape: (1, 3)
┌───────┬─────┬──────┐
│ name  ┆ age ┆ dept │
│ ---   ┆ --- ┆ ---  │
│ str   ┆ i64 ┆ str  │
╞═══════╪═════╪══════╡
│ Alice ┆ 34  ┆ eng  │
└───────┴─────┴──────┘"#,
);

// malformed csv (ragged row) -> from_csv raises
test_fail!(
    from_csv_malformed_raises,
    r#"use std::polars::*; let df = from_csv("a,b\n1,2,3\n")!;"#,
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
