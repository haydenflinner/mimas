//! `std::polars` -- `to_dataframe`/`col`/`filter`/`select`/`sort` over a real polars `DataFrame`.
//! See `crates/library/src/std_lib/dataframe.rs`.
//!
//! `DataFrame`/`PlExpr` are real `Val` variants, not `#[mimas] struct`s, so the harness's
//! `Captured` inspection (used by `test_run!`/`render`) can't see inside them -- it reports them
//! as `Captured::Other` by design. `test_run_display!`/`render_display` sidesteps that by reading
//! the value back through its real `Display` (the same path `print` uses) instead, which also
//! means the expected side here is the plain, unescaped table text.
//!
//! The whole file is a no-op without the `dataframe` feature: `std::polars` isn't registered, so
//! every case here would just fail to resolve rather than being skipped.
#![cfg(feature = "dataframe")]

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

test_run_display!(
    group_by_agg_reduces_per_group,
    EMPLOYEES,
    r#"df.group_by(["dept"]).agg([col("age").mean().alias("avg_age"), col("age").count().alias("n")])!.sort(["dept"])!"# => r#"shape: (2, 3)
┌───────┬─────────┬─────┐
│ dept  ┆ avg_age ┆ n   │
│ ---   ┆ ---     ┆ --- │
│ str   ┆ f64     ┆ u32 │
╞═══════╪═════════╪═════╡
│ eng   ┆ 37.5    ┆ 2   │
│ sales ┆ 27.0    ┆ 2   │
└───────┴─────────┴─────┘"#,
);

const SALES: &str = "use std::polars::*;
     struct Sale { region: str, quarter: str, revenue: int }
     let rows = [
         Sale { region = \"east\", quarter = \"Q1\", revenue = 100 },
         Sale { region = \"east\", quarter = \"Q2\", revenue = 150 },
         Sale { region = \"west\", quarter = \"Q1\", revenue = 200 },
         Sale { region = \"west\", quarter = \"Q2\", revenue = 250 },
     ];
     let df = to_dataframe(rows)!;";

test_run_display!(
    pivot_makes_a_wide_table_from_on_values,
    SALES,
    r#"df.pivot(["quarter"], ["region"], ["revenue"], "sum")!.sort(["region"])!"# => r#"shape: (2, 3)
┌────────┬─────┬─────┐
│ region ┆ Q1  ┆ Q2  │
│ ---    ┆ --- ┆ --- │
│ str    ┆ i64 ┆ i64 │
╞════════╪═════╪═════╡
│ east   ┆ 100 ┆ 150 │
│ west   ┆ 200 ┆ 250 │
└────────┴─────┴─────┘"#,
);

// unrecognized aggregate function name -> raises rather than silently defaulting
test_fail!(
    pivot_unknown_agg_raises,
    r#"use std::polars::*;
       struct S { region: str, quarter: str, revenue: int }
       let df = to_dataframe([S { region = "e", quarter = "Q1", revenue = 1 }])!;
       let p = df.pivot(["quarter"], ["region"], ["revenue"], "nonsense")!;"#,
);

const EMPLOYEES_AND_DEPTS: &str = "use std::polars::*;
     struct Employee { name: str, age: int, dept: str }
     let rows = [
         Employee { name = \"Alice\", age = 34, dept = \"eng\" },
         Employee { name = \"Bob\", age = 29, dept = \"sales\" },
     ];
     let df = to_dataframe(rows)!;
     struct Dept { dept: str, manager: str }
     let depts = [
         Dept { dept = \"eng\", manager = \"Erin\" },
         Dept { dept = \"sales\", manager = \"Sam\" },
     ];
     let dept_df = to_dataframe(depts)!;";

test_run_display!(
    join_matches_rows_on_a_shared_column_name,
    EMPLOYEES_AND_DEPTS,
    r#"df.join(dept_df, ["dept"], "inner")!.sort(["name"])!"# => r#"shape: (2, 4)
┌───────┬─────┬───────┬─────────┐
│ name  ┆ age ┆ dept  ┆ manager │
│ ---   ┆ --- ┆ ---   ┆ ---     │
│ str   ┆ i64 ┆ str   ┆ str     │
╞═══════╪═════╪═══════╪═════════╡
│ Alice ┆ 34  ┆ eng   ┆ Erin    │
│ Bob   ┆ 29  ┆ sales ┆ Sam     │
└───────┴─────┴───────┴─────────┘"#,
);

// unrecognized join type -> raises rather than silently defaulting
test_fail!(
    join_unknown_type_raises,
    r#"use std::polars::*;
       struct A { k: str } struct B { k: str }
       let a = to_dataframe([A { k = "x" }])!;
       let b = to_dataframe([B { k = "x" }])!;
       let j = a.join(b, ["k"], "sideways")!;"#,
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
