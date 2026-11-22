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

test_run_display!(
    arrange_orders_with_a_per_key_direction,
    EMPLOYEES,
    r#"df.arrange(["dept", "age"], [false, true])!"# => r#"shape: (4, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Carol ┆ 41  ┆ eng   │
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
│ Dave  ┆ 25  ┆ sales │
└───────┴─────┴───────┘"#,
);

// r[verify mimas.std.order-arrange]
test_run_display!(
    order_arrange_broadcasts_one_direction,
    EMPLOYEES,
    r#"AscDesc::Desc.arrange(df, ["age"])"# => r#"shape: (4, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Carol ┆ 41  ┆ eng   │
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
│ Dave  ┆ 25  ┆ sales │
└───────┴─────┴───────┘"#,
);

test_run_display!(
    mutate_appends_a_computed_column,
    EMPLOYEES,
    r#"df.mutate([(col("age") * 2).alias("double")])!.select_names(["name", "double"])!"# => r#"shape: (4, 2)
┌───────┬────────┐
│ name  ┆ double │
│ ---   ┆ ---    │
│ str   ┆ i64    │
╞═══════╪════════╡
│ Alice ┆ 68     │
│ Bob   ┆ 58     │
│ Carol ┆ 82     │
│ Dave  ┆ 50     │
└───────┴────────┘"#,
);

test_run_display!(
    select_names_picks_columns_by_bare_name,
    EMPLOYEES,
    r#"df.select_names(["name"])!"# => r#"shape: (4, 1)
┌───────┐
│ name  │
│ ---   │
│ str   │
╞═══════╡
│ Alice │
│ Bob   │
│ Carol │
│ Dave  │
└───────┘"#,
);

test_run_display!(
    n_counts_rows_per_group,
    EMPLOYEES,
    r#"df.group_by(["dept"]).agg([n().alias("n")])!.sort(["dept"])!"# => r#"shape: (2, 2)
┌───────┬─────┐
│ dept  ┆ n   │
│ ---   ┆ --- │
│ str   ┆ u32 │
╞═══════╪═════╡
│ eng   ┆ 2   │
│ sales ┆ 2   │
└───────┴─────┘"#,
);

test_run!(
    pull_reads_a_column_back_as_an_array,
    EMPLOYEES,
    r#"df.pull("name")!"# => r#"["Alice", "Bob", "Carol", "Dave"]"#,
);

test_run_display!(
    distinct_keeps_first_row_per_key,
    EMPLOYEES,
    r#"df.distinct(["dept"])!"# => r#"shape: (2, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
└───────┴─────┴───────┘"#,
);

test_run_display!(
    head_tail_slice_window_the_rows,
    EMPLOYEES,
    r#"df.tail(1)"# => r#"shape: (1, 3)
┌──────┬─────┬───────┐
│ name ┆ age ┆ dept  │
│ ---  ┆ --- ┆ ---   │
│ str  ┆ i64 ┆ str   │
╞══════╪═════╪═══════╡
│ Dave ┆ 25  ┆ sales │
└──────┴─────┴───────┘"#,
);

test_run_display!(
    take_is_head_under_the_domain_name,
    EMPLOYEES,
    r#"df.take(2)"# => r#"shape: (2, 3)
┌───────┬─────┬───────┐
│ name  ┆ age ┆ dept  │
│ ---   ┆ --- ┆ ---   │
│ str   ┆ i64 ┆ str   │
╞═══════╪═════╪═══════╡
│ Alice ┆ 34  ┆ eng   │
│ Bob   ┆ 29  ┆ sales │
└───────┴─────┴───────┘"#,
);

test_run_display!(
    rename_relabels_a_column,
    EMPLOYEES,
    r#"df.rename("dept", "team")!.head(1)"# => r#"shape: (1, 3)
┌───────┬─────┬──────┐
│ name  ┆ age ┆ team │
│ ---   ┆ --- ┆ ---  │
│ str   ┆ i64 ┆ str  │
╞═══════╪═════╪══════╡
│ Alice ┆ 34  ┆ eng  │
└───────┴─────┴──────┘"#,
);

// mismatched direction list -> arrange raises rather than guessing
test_fail!(
    arrange_rejects_a_mismatched_direction_list,
    r#"use std::polars::*;
       struct Employee { name: str, age: int, dept: str }
       let df = to_dataframe([Employee { name = "A", age = 1, dept = "x" }])!;
       let _ = df.arrange(["dept", "age"], [true, false, true])!;"#,
);

// a column that doesn't exist -> pull raises
test_fail!(
    pull_rejects_an_unknown_column,
    r#"use std::polars::*;
       struct Employee { name: str, age: int, dept: str }
       let df = to_dataframe([Employee { name = "A", age = 1, dept = "x" }])!;
       let _ = df.pull("nope")!;"#,
);

test_run!(
    col_names_lists_columns_in_order,
    EMPLOYEES,
    r#"df.col_names().join(",")"# => r#""name,age,dept""#,
);

// missing cells, conditional columns, text operations, wide -> tall
test_run!(
    fill_null_and_is_null,
    r#"use std::polars::*;
       let df = from_csv("a,b\nx,\ny,z\n")!;"#,
    r#"df.mutate([col("b").fill_null(lit("none")).alias("b")])!.pull("b")!.join(",")"# => r#""none,z""#,
    r#"df.filter(col("b").is_null())!.pull("a")!.join(",")"# => r#""x""#,
);

test_run!(
    when_builds_a_conditional_column,
    EMPLOYEES,
    r#"df.mutate([when(col("age") > 30, lit("senior"), lit("junior")).alias("level")])!.pull("level")!.join(",")"# => r#""senior,junior,senior,junior""#,
);

// `shift`/`diff` read neighbouring rows — the window the tidy-tables
// deck derives lap times through (sorted rider clocks, guard on the
// rider boundary, first lap counts from the gun).
test_run_display!(
    shift_and_diff_read_neighbouring_rows,
    EMPLOYEES,
    r#"df.mutate([col("age").shift(1).alias("prev")])!.select_names(["name", "prev"])"# => r#"shape: (4, 2)
┌───────┬──────┐
│ name  ┆ prev │
│ ---   ┆ ---  │
│ str   ┆ i64  │
╞═══════╪══════╡
│ Alice ┆ null │
│ Bob   ┆ 34   │
│ Carol ┆ 29   │
│ Dave  ┆ 41   │
└───────┴──────┘"#,
    r#"df.mutate([col("age").diff(1).alias("gap")])!.select_names(["name", "gap"])"# => r#"shape: (4, 2)
┌───────┬──────┐
│ name  ┆ gap  │
│ ---   ┆ ---  │
│ str   ┆ i64  │
╞═══════╪══════╡
│ Alice ┆ null │
│ Bob   ┆ -5   │
│ Carol ┆ 12   │
│ Dave  ┆ -16  │
└───────┴──────┘"#,
    // a negative shift is a lead
    r#"df.mutate([col("age").shift(-1).alias("next")])!.select_names(["name", "next"])"# => r#"shape: (4, 2)
┌───────┬──────┐
│ name  ┆ next │
│ ---   ┆ ---  │
│ str   ┆ i64  │
╞═══════╪══════╡
│ Alice ┆ 29   │
│ Bob   ┆ 41   │
│ Carol ┆ 25   │
│ Dave  ┆ null │
└───────┴──────┘"#,
);

test_run_display!(
    laptime_per_rider_via_shift_diff_and_when,
    r#"use std::polars::*;
       const START = 36000;
       let samples = table {
           rider    clock
           "bo"     36061
           "ada"    36060
           "ada"    36122
           "bo"     36124
       };"#,
    r#"samples
       .arrange(["rider", "clock"], [false, false])!
       .mutate([when(col("rider") == col("rider").shift(1),
                      col("clock").diff(1),
                      col("clock") - lit(START)).alias("laptime")])!
       .select_names(["rider", "laptime"])!"# => r#"shape: (4, 2)
┌───────┬─────────┐
│ rider ┆ laptime │
│ ---   ┆ ---     │
│ str   ┆ i64     │
╞═══════╪═════════╡
│ ada   ┆ 60      │
│ ada   ┆ 62      │
│ bo    ┆ 61      │
│ bo    ┆ 63      │
└───────┴─────────┘"#,
);

test_run!(
    string_expressions,
    EMPLOYEES,
    r#"df.mutate([col("name").str_to_upper().alias("name")])!.pull("name")!.join(",")"# => r#""ALICE,BOB,CAROL,DAVE""#,
    r#"df.filter(col("name").str_starts_with("C"))!.pull("name")!.join(",")"# => r#""Carol""#,
    r#"df.filter(col("name").str_ends_with("e"))!.pull("name")!.join(",")"# => r#""Alice,Dave""#,
    r#"df.filter(col("name").str_contains("ar"))!.pull("name")!.join(",")"# => r#""Carol""#,
    r#"df.mutate([col("name").str_len().alias("n")])!.pull("n")!.len()"# => "4",
);

test_run!(
    unpivot_makes_a_wide_table_tall,
    r#"use std::polars::*;
       let df = from_csv("who,q1,q2\nann,1,2\nbob,3,4\n")!;"#,
    r#"df.unpivot(["who"], ["q1", "q2"])!.pull("value")!.len()"# => "4",
    r#"df.unpivot(["who"], ["q1", "q2"])!.col_names().join(",")"# => r#""who,variable,value""#,
);

test_run!(
    cast_and_is_in,
    r#"use std::polars::*;
       let df = from_csv("a,n\nx,1\ny,three\nz,3\n")!;"#,
    r#"df.filter(col("a").is_in(["x", "z"]))!.pull("a")!.join(",")"# => r#""x,z""#,
    r#"df.filter(col("n") != "three")!.mutate([col("n").cast("int").alias("n")])!.pull("n")!.len()"# => "2",
);

// the `table { … }` literal: header line, then one row per line; columns can differ in type
test_run!(
    table_literal_builds_columns,
    r#"use std::polars::*;
       let df = table {
           name        age   "full name"
           "Denmark"   25    "example1"
           "Hello"     42    "yep!"
       };"#,
    r#"df.col_names().join(",")"# => r#""name,age,full name""#,
    r#"df.pull("age")!.len()"# => "2",
    r#"df.filter(col("age") > 30)!.pull("name")!.join(",")"# => r#""Hello""#,
);

test_run!(
    table_literal_accepts_commas_negatives_and_parens,
    r#"use std::polars::*;
       let df = table { a, b
           1, -2.5
           (1 + 2), 0.5
       };"#,
    r#"df.pull("a")!.len()"# => "2",
    r#"df.filter(col("b") < 0.0)!.pull("a")!.len()"# => "1",
);

test_fail!(
    table_literal_rejects_a_ragged_row,
    r#"let df = table { a b
           1 2
           3
       };"#,
);

// ---- `query { … }`: PRQL-style verb lines over a table --------------------------------------
// Bare names are columns, `&&`/`||` combine column tests, `if` is a conditional column. A query
// needs no `use` -- it lowers to `__q_*` natives (the table/preamble below uses `std::polars`
// only for `pull`/`col_names` in the assertions).

const QEVENTS: &str = r#"let events = table {
        name      numtix   delivery
        "Ellie"   2        "email"
        "Bonnie"  1        "pickup"
        "Sam"     5        "pickup"
        "Zach"    0        "email"
        "Parrot"  10       "pickup"
        "Ana"     3        "email"
    };"#;

test_run_display!(
    query_filter_sort_take_select,
    QEVENTS,
    r#"query {
        events
        filter numtix > 0 && delivery == "pickup"
        derive cost = numtix * 25
        sort -cost
        take 2
        select name cost
    }"# => r#"shape: (2, 2)
┌────────┬──────┐
│ name   ┆ cost │
│ ---    ┆ ---  │
│ str    ┆ i64  │
╞════════╪══════╡
│ Parrot ┆ 250  │
│ Sam    ┆ 125  │
└────────┴──────┘"#,
);

test_run_display!(
    query_group_aggregate,
    QEVENTS,
    r#"query {
        events
        group delivery {
            aggregate { tickets = sum(numtix), orders = count() }
        }
        sort delivery
    }"# => r#"shape: (2, 3)
┌──────────┬─────────┬────────┐
│ delivery ┆ tickets ┆ orders │
│ ---      ┆ ---     ┆ ---    │
│ str      ┆ i64     ┆ u32    │
╞══════════╪═════════╪════════╡
│ email    ┆ 5       ┆ 3      │
│ pickup   ┆ 16      ┆ 3      │
└──────────┴─────────┴────────┘"#,
);

test_run_display!(
    query_lag_lead_and_difference,
    QEVENTS,
    // difference(numtix) is numtix minus the row above — sort first!
    r#"query {
        events
        sort name
        derive gap = difference(numtix)
        select name gap
    }"# => r#"shape: (6, 2)
┌────────┬──────┐
│ name   ┆ gap  │
│ ---    ┆ ---  │
│ str    ┆ i64  │
╞════════╪══════╡
│ Ana    ┆ null │
│ Bonnie ┆ -2   │
│ Ellie  ┆ 1    │
│ Parrot ┆ 8    │
│ Sam    ┆ -5   │
│ Zach   ┆ -5   │
└────────┴──────┘"#,
    // lag reads the row above, lead the row below
    r#"query {
        events
        sort name
        derive prev = lag(delivery)
        select name prev
    }"# => r#"shape: (6, 2)
┌────────┬────────┐
│ name   ┆ prev   │
│ ---    ┆ ---    │
│ str    ┆ str    │
╞════════╪════════╡
│ Ana    ┆ null   │
│ Bonnie ┆ email  │
│ Ellie  ┆ pickup │
│ Parrot ┆ email  │
│ Sam    ┆ pickup │
│ Zach   ┆ pickup │
└────────┴────────┘"#,
    r#"query {
        events
        sort name
        derive nxt = lead(numtix)
        select name nxt
    }"# => r#"shape: (6, 2)
┌────────┬──────┐
│ name   ┆ nxt  │
│ ---    ┆ ---  │
│ str    ┆ i64  │
╞════════╪══════╡
│ Ana    ┆ 1    │
│ Bonnie ┆ 2    │
│ Ellie  ┆ 10   │
│ Parrot ┆ 5    │
│ Sam    ┆ 0    │
│ Zach   ┆ null │
└────────┴──────┘"#,
);

test_run!(
    query_if_rename_distinct_join,
    r#"use std::polars::*;
       let events = table { name, numtix, delivery
           "Ellie", 2, "email"
           "Bonnie", 1, "pickup"
       };
       let fees = table { delivery, fee
           "email", 0
           "pickup", 3
       };"#,
    // if c { a } else { b } is a conditional column
    r#"query {
        events
        derive size = if numtix >= 2 { "large" } else { "small" }
        sort name
    }.pull("size")!.join(",")"# => r#""small,large""#,
    // join takes the other table then key names; rename and distinct take bare names
    r#"query {
        events
        join fees delivery
        rename delivery ship_via
        distinct ship_via
    }.pull("ship_via")!.join(",")"# => r#""email,pickup""#,
);

test_run!(
    query_column_functions_and_splices,
    // `$expr` is the escape: inside a verb every bare name is a column, so an outside
    // value (a list, a computed scalar) splices in with a leading `$`
    r#"use std::polars::*;
       fn known_codes() -> [str] { ["birthday", "student"] }
       let events = from_csv("name,discount
Ellie,Birthday
Bonnie,STUDENT
Sam,
Zach,none")!;"#,
    r#"query {
        events
        derive discount = to_lower(discount)
        derive discount = if is_in(discount, $known_codes()) { discount } else { "" }
        sort name
    }.pull("discount")!.join(",")"# => r#""student,birthday,,""#,
    r#"query {
        events
        filter contains(name, "a")
        sort -name
    }.pull("name")!.join(",")"# => r#""Zach,Sam""#,
);

// `name == $who` -- the `$` splice evaluates `who` outside the query's column
// namespace, so a column can meet a parameter or a `let`
test_run!(
    query_splices_reach_outside_values,
    r#"use std::polars::*;
       fn people() -> DataFrame {
           table { name, age
               "Anna", 28
               "Susan", 54
           }
       }
       fn row_for(t: DataFrame, who: str) -> str {
           let row = query {
               t
               filter name == $who
           };
           row.pull("name")![0]
       }
       let decade = 10;"#,
    r#"row_for(people(), "Susan")"# => r#""Susan""#,
    r#"query {
        people()
        filter name != $"Anna"
    }.pull("name")!.join(",")"# => r#""Susan""#,
    // a splice is a whole outside expression, not just a name
    r#"query {
        people()
        derive score = age * $decade
    }.pull("score")![1]"# => "540",
    // `eq` still works -- now uniformly a column-vs-column (or vs `$`-value) test
    r#"query {
        people()
        filter eq(name, $"Anna")
    }.pull("name")!.join(",")"# => r#""Anna""#,
);

// `$` outside a `query { }` isn't an escape from anything -- the marker reaches
// the `__q_splice` native, which says so
test_fail!(
    query_splice_outside_a_query_raises,
    r#"let who = "Anna";
       let x = $who;"#,
);

// a verb line that isn't a verb -> a parse error pointing at it
test_fail!(
    query_unknown_verb_raises,
    r#"let df = query {
        table { a
            1
        }
        frobnicate a
    };"#,
);

// ---- schemas ---------------------------------------------------------------
// `table {}` and `.schema("…")` give the checker a column map, so `pull` returns a
// concrete `[T]` (no annotation, no `?mimas<T>`) and a typo'd name is a check-time
// error. Opaque frames (`from_csv` with no schema) keep the old generic pull.

const TRIPS: &str = "use std::polars::*;
     let trips = table {
         zone  fare  tip
         1     12.5  2.0
         2     8.0   1.0
         1     20.0  5.0
     };";

// `.format(2)` on a pulled element only compiles because `pull` knows `fare: float`
test_run!(
    table_literal_schema_types_pull,
    TRIPS,
    r#"trips.pull("fare")![0].format(2)"# => r#""12.50""#,
    r#"trips.pull("zone")![0] + 1"# => "2",
);

// `.schema(…)` declares types on an opaque frame; the same string drives the runtime
// check/cast and the solver's column types
test_run!(
    schema_declares_types_on_an_opaque_frame,
    "use std::polars::*;",
    r#"from_csv("zone,fare\n1,10.5\n2,9.0")!.schema("zone:int fare:float")!.pull("zone")![0] + 1"# => "2",
);

// schemas propagate through the query verbs
test_run!(
    query_verbs_propagate_the_schema,
    TRIPS,
    r#"query {
        trips
        filter fare > 5
        derive gross = fare + tip
        group zone {
            aggregate {
                n = count(),
                revenue = sum(gross),
            }
        }
        sort -revenue
    }.pull("revenue")![0].format(1)"# => r#""39.5""#,
);

// and through the method api
test_run!(
    method_calls_propagate_the_schema,
    TRIPS,
    r#"trips.filter(col("fare") > 5)!.mutate([col("fare").sum().alias("rev")])!.pull("rev")![0]"# => "40.5",
);

// a `schema()`-declared unit makes pulled elements quantities the dims checker tracks
test_run_display!(
    schema_units_track_through_pull,
    "use std::polars::*;
     let df = from_csv(\"kwh\\n3.5\\n4.0\")!.schema(\"kwh:kWh\")!;
     let kwh = df.pull(\"kwh\")!;",
    "kwh[0] + kwh[1]" => "7.5",
);

test_fail!(
    pull_of_a_typo_column_is_a_check_error,
    r#"use std::polars::*;
       let trips = table {
           zone  fare
           1     12.5
       };
       trips.pull("zome")!;"#,
);

test_fail!(
    query_verb_rejects_a_typo_column_at_check_time,
    r#"use std::polars::*;
       let trips = table {
           zone  fare
           1     12.5
       };
       query {
           trips
           filter frae > 5
       };"#,
);

test_fail!(
    schema_spec_rejects_an_unknown_type,
    r#"use std::polars::*;
       let trips = table {
           zone
           1
       };
       trips.schema("zone:integer")!;"#,
);

test_fail!(
    schema_spec_rejects_an_unknown_column,
    r#"use std::polars::*;
       let trips = table {
           zone
           1
       };
       trips.schema("zome:int")!;"#,
);

// unit dims on a pulled column still check -- kwh + seconds is a dims error
test_fail!(
    schema_units_reject_mismatched_dims,
    r#"use std::polars::*;
       let df = from_csv("kwh\n3.5")!.schema("kwh:kWh")!;
       df.pull("kwh")![0] + 2s;"#,
);

// unit literals in `table {}` cells carry their dims into the schema too
test_run_display!(
    table_literal_unit_columns,
    "use std::polars::*;
     let df = table {
         dist
         5km
         3km
     };",
    r#"df.pull("dist")![0] + 100m"# => "5100.0",
);

// ---- unit columns inside query/mutate expressions -------------------------------------
// A schema'd column keeps its unit inside `query { … }` verbs and the `col(…)` method
// api: comparisons and `+`/`-` demand the same dimension, `*`/`/` compose it onto the
// derived column in the propagated schema, and a `$` splice or opaque call stays
// permissive -- the checker enforces dims; the polars columns underneath stay raw
// numbers.

const MASSES: &str = "use std::polars::*;
     let df = from_csv(\"item,mass\\na,1.5\\nb,3.0\\nc,0.5\\n\")!.schema(\"item:str mass:kg\")!;";

test_run!(
    query_verbs_apply_unit_dims,
    MASSES,
    r#"query {
        df
        filter mass > 2kg
    }.pull("item")!.join(",")"# => r#""b""#,
    // literal 0 is allowed on either side of a comparison, like the scalar dims pass
    r#"query {
        df
        filter mass > 0
    }.pull("item")!.join(",")"# => r#""a,b,c""#,
    r#"query {
        df
        filter mass + 1kg > 2kg
    }.pull("item")!.join(",")"# => r#""a,b""#,
);

// `price * n` gives `cost` the composed dimension -- `.to("usd")` on the sum only
// compiles because `pull("cost")` hands back usd-quantity elements
test_run!(
    query_derive_lands_the_composed_dim_in_the_schema,
    "use std::polars::*;
     let df = table {
         item  price  n
         \"a\"   2usd   3
         \"b\"   5usd   2
     };
     let out = query {
         df
         derive cost = price * n
     };",
    r#"out.pull("cost")!.sum().to("usd")"# => "16.0",
);

// the method api runs the same checks on its explicit column expressions
test_run!(
    method_api_units_filter_and_is_in,
    MASSES,
    r#"df.filter(col("mass") > 2kg)!.pull("item")!.join(",")"# => r#""b""#,
    r#"df.filter(col("mass").is_in([0.5kg, 3kg]))!.pull("item")!.join(",")"# => r#""b,c""#,
    r#"df.mutate([(col("mass") / col("mass")).alias("one")])!.pull("one")![0] + 1.0"# => "2.0",
);

// mismatched dims inside a column expr are check-time errors in the query verbs…
test_fail!(
    query_verbs_reject_mismatched_dims,
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       query { df
           filter mass > 2s
       };"#,
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       query { df
           derive bad = mass + 2s
       };"#,
    // a bare number isn't interchangeable with a quantity column either
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       query { df
           derive bad = mass + 3
       };"#,
);

// …and in the method api
test_fail!(
    method_api_rejects_mismatched_dims,
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       df.filter(col("mass") > 2s)!;"#,
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       df.mutate([(col("mass") + 2s).alias("x")])!;"#,
    r#"use std::polars::*;
       let df = from_csv("item,mass\na,1.5\n")!.schema("item:str mass:kg")!;
       df.filter(col("mass").is_in([2kg, 3s]))!;"#,
);

// an opaque frame stays permissive -- pull keeps its old generic array type and a
// bad name still only fails at runtime, not at check time
test_fail!(
    unschematized_pull_of_a_bad_column_fails_at_runtime,
    r#"use std::polars::*;
       let df = from_csv("a,b\n1,2")!;
       df.pull("nope")!;"#,
);

// ---- rows as records ---------------------------------------------------------------
// `df.row(i)`/`df.rows()` hand a row back as an instance of the declared struct whose
// fields are exactly the frame's columns, filled by name. On a `table {}` literal the
// closed schema resolves the record type at check time (so `row.field` type-checks);
// on an opaque frame the return stays anonymous and unifies where it's used -- the
// runtime matches the struct by field names either way.

const SHUTTLE: &str = "use std::polars::*;
     struct Shuttle { month: str, riders: int }
     let df = table {
         month    riders
         \"Jan\"    1123
         \"Feb\"    1045
         \"Mar\"    1087
         \"Apr\"    999
     };";

test_run!(
    row_reads_a_row_back_as_the_matching_record,
    SHUTTLE,
    r#"df.row(0)!"# => r#"{ "Jan", 1123 }"#,
    r#"df.row(3)!.riders"# => "999",
);

// the struct's field order doesn't have to match the column order -- fields fill by name
test_run!(
    row_fills_fields_by_name_not_position,
    r#"use std::polars::*;
       struct Swap { riders: int, month: str }
       let df = table { month  riders
           "Jan"  1123
       };
       let r = df.row(0)!;"#,
    "r" => r#"{ 1123, "Jan" }"#,
);

test_run!(
    rows_reads_every_row_in_order,
    SHUTTLE,
    r#"df.rows()!.len()"# => "4",
    r#"df.rows()![2].month"# => r#""Mar""#,
);

// the point of the api: a row goes straight into a fn typed on the record. This frame
// is opaque (from_csv, no schema) so the record type comes from unifying the anonymous
// return with `s: Shuttle`, not the schema fast path; the free-fn form does the same.
test_run!(
    row_passes_into_a_function_typed_on_the_record,
    r#"use std::polars::*;
       struct Shuttle { month: str, riders: int }
       fn cleared_1k(s: Shuttle) -> bool { s.riders < 1000 }
       let df = from_csv("month,riders\nJan,1123\nApr,999\n")!;"#,
    "cleared_1k(df.row(0)!)" => "false",
    "cleared_1k(df.row(1)!)" => "true",
    "cleared_1k(row(df, 0)!)" => "false",
);

// a missing cell materializes as `null` in the field's slot
test_run!(
    row_materializes_a_missing_cell_as_null,
    r#"use std::polars::*;
       struct CsvRow { a: str, b: str }
       let df = from_csv("a,b\nx,\ny,z\n")!.schema("a:str b:str")!;"#,
    r#"df.row(0)!"# => r#"{ "x", null }"#,
);

// no declared struct has the frame's column set -> raise, not a guess. (`int` forces
// the anonymous return to resolve so the failure is the runtime match, not inference.)
test_fail!(
    row_rejects_a_frame_with_no_matching_record_shape,
    r#"use std::polars::*;
       let df = table { a b
           1 2
       };
       let _: int = df.row(0)!;"#,
);

// two structs with the same fields -> the match is ambiguous and raises
test_fail!(
    row_rejects_an_ambiguous_record_shape,
    r#"use std::polars::*;
       struct A { x: int, y: int }
       struct B { x: int, y: int }
       let df = to_dataframe([A { x = 1, y = 2 }])!;
       let _: int = df.row(0)!;"#,
);

test_fail!(
    row_out_of_bounds_raises,
    r#"use std::polars::*;
       struct Shuttle { month: str, riders: int }
       let df = table { month riders
           "Jan" 1123
       };
       let _ = df.row(5)!;"#,
);

// ---- `from_json` + datetimes ---------------------------------------------------
// A JSON document loads straight into a DataFrame: the rows are an array of
// objects (a bare array, or the envelope's single array field -- `key` picks
// one when there are several). Scalar fields become typed columns, a scalar
// array flattens to `f_0`/`f_1`, deeper arrays become list columns, and mixed
// values render their JSON text. `to_datetime` then turns an epoch column into
// a real datetime the rest of the verbs can take apart.

const LOBBY: &str = r#"use std::polars::*;
    let games = from_json("{\"total\":2,\"games\":[{\"p\":[\"ann\",\"bo\"],\"s\":[3,1],\"w\":0,\"kind\":\"tour\",\"endedAt\":1704103200000,\"actions\":5,\"moves\":[[0,1,2,100],[1,3,4,200]]},{\"p\":[\"cy\",\"di\"],\"s\":[2,2],\"w\":\"tie\",\"kind\":\"practice\",\"endedAt\":1704153600000,\"actions\":8,\"moves\":null}]}")!;"#;

test_run!(
    from_json_reads_an_envelopes_array_and_flattens_scalar_arrays,
    LOBBY,
    // `p: [blue, red]` flattens to p_0/p_1; `s` likewise
    r#"games.pull("p_0")!.join(",")"# => r#""ann,cy""#,
    r#"games.pull("s_1")"# => "[1, 2]",
    // `w` mixes an int seat and a "tie" string -- the whole column is text
    r#"games.pull("w")!.join(",")"# => r#""0,tie""#,
    r#"games.pull("endedAt")!.len()"# => "2",
);

test_run!(
    from_json_nested_arrays_become_list_columns,
    LOBBY,
    // `moves` is an array of arrays -> a list column you can measure and unpack
    r#"games.mutate([col("moves").list_len().alias("nm")])!.pull("nm")!"# => "[2, null]",
    r#"games.pull("moves")![0][0]"# => "[0, 1, 2, 100]",
    // explode opens the lists into one row per move (rows only grow via select)
    r#"games.select([col("moves").explode().alias("m")])!.pull("m")"# => "[[0, 1, 2, 100], [1, 3, 4, 200], null]",
    r#"games.select([col("moves").explode().alias("m")])!.filter(col("m").is_not_null())!.pull("m")![1]"# => "[1, 3, 4, 200]",
);

test_run!(
    from_json_query_space_flattens_too,
    LOBBY,
    // inside query { } the flattened names are ordinary columns
    r#"query {
        games
        derive winner = if w == "0" { p_0 } else { if w == "1" { p_1 } else { "tie" } }
        sort -s_0
    }.pull("winner")!.join(",")"# => r#""ann,tie""#,
);

// epoch ms -> datetime -> parts. 1704103200000 is 2024-01-01T10:00:00Z.
test_run!(
    datetime_parts_off_an_epoch_column,
    LOBBY,
    r#"query {
        games
        derive ended = to_datetime(endedAt, "ms")
        sort ended
    }.pull("ended")![0]"# => "1704103200000",
    r#"query {
        games
        derive ended = to_datetime(endedAt, "ms")
        sort -ended
        derive hr = hour(ended), yr = year(ended), dow = strftime(ended, "%a")
    }.pull("hr")![0]"# => "0",
    r#"query {
        games
        derive ended = to_datetime(endedAt, "ms")
        sort ended
        derive hr = hour(ended), yr = year(ended), dow = strftime(ended, "%a")
    }.pull("dow")!.join(",")"# => r#""Mon,Tue""#,
    r#"query {
        games
        derive ended = to_datetime(endedAt, "ms")
        sort ended
        derive yr = year(ended)
    }.pull("yr")"# => "[2024, 2024]",
    // date() gives a real date dtype; pull renders it as ISO text
    r#"query {
        games
        derive d = date(to_datetime(endedAt, "ms"))
        sort -d
    }.pull("d")![0]"# => r#""2024-01-02""#,
    // truncate buckets a timestamp -- the 10:00 game snaps to midnight
    r#"query {
        games
        derive day_ms = epoch_ms(truncate(to_datetime(endedAt, "ms"), "1d"))
        sort day_ms
    }.pull("day_ms")![0]"# => "1704067200000",
);

test_run!(
    datetime_fluent_api_mirrors_the_query_functions,
    LOBBY,
    r#"games.mutate([col("endedAt").to_datetime("ms").dt_strftime("%Y-%m-%d").alias("d")])!.pull("d")!.join(",")"# => r#""2024-01-01,2024-01-02""#,
    r#"games.mutate([col("endedAt").to_datetime("ms").dt_epoch_ms().alias("ms")])!.pull("ms")![0]"# => "1704103200000",
    // "s" rescales the ints into milliseconds before the cast
    r#"table { t
        1704103200
    }.mutate([col("t").to_datetime("s").dt_year().alias("y")])!.pull("y")![0]"# => "2024",
);

// a bare top-level array needs no key; several arrays do
test_run!(
    from_json_array_and_key_forms,
    "use std::polars::*;",
    r#"from_json("[{\"a\":1},{\"a\":2}]")!.pull("a")"# => "[1, 2]",
    r#"from_json("{\"x\":[{\"a\":1}],\"y\":[{\"b\":2}]}", "y")!.pull("b")"# => "[2]",
);

test_fail!(
    from_json_rejects_a_multi_array_document_without_a_key,
    r#"use std::polars::*;
       let _ = from_json("{\"x\":[{}],\"y\":[{}]}")!;"#,
);

test_fail!(
    from_json_rejects_non_object_rows,
    r#"use std::polars::*;
       let _ = from_json("[1,2,3]")!;"#,
);

// A data frame held in program state snapshots and restores: game pages
// carry their top-level `let`s across a replay checkpoint or an edit,
// and a table is a fine thing for a top-level `let` to hold.
// r[verify game.state.dataframe]
#[test]
fn a_dataframe_survives_snapshot_and_restore() {
    let src = format!("{EMPLOYEES}\nlet same = df;\nlet q = col(\"age\") > 30;");
    let mut vm = vm::Vm::execute(&src, library::std).expect("compiled and ran");
    let before = vm.resolve_name_to_string("df").unwrap().unwrap();
    let snap = vm.snapshot().expect("a dataframe snapshots");
    vm.restore(&snap).expect("restore");
    assert_eq!(vm.resolve_name_to_string("df").unwrap().unwrap(), before);
    assert_eq!(vm.resolve_name_to_string("same").unwrap().unwrap(), before);
    assert!(vm.resolve_name_to_string("q").unwrap().is_ok());
}

// ---- `data "…"` -- external sources with a check-time schema -----------
// `data "name"` is the type-provider form: the host's resolver (the ambient
// hook `solve::set_data_resolver`, which every compile path reads) answers
// the name with a column schema, and `pull`/`query` code checks columns by
// name. The runtime half is the host's own `__data` binding: a page-level
// `fn __data` shadows the prelude fallback, which is how the web host's
// generated source wires a fetched payload in.

/// Install `f` as the ambient data resolver for the test's duration; the
/// previous resolver goes back on drop (tests share the thread).
struct ResolverGuard(Option<solve::DataResolver>);

impl ResolverGuard {
    fn install(f: impl Fn(&str) -> Option<solve::DataSchema> + 'static) -> Self {
        Self(solve::set_data_resolver(Some(std::rc::Rc::new(f))))
    }
}

impl Drop for ResolverGuard {
    fn drop(&mut self) {
        solve::set_data_resolver(self.0.take());
    }
}

fn data_resolver(name: &str) -> Option<solve::DataSchema> {
    (name == "trips").then(|| solve::DataSchema::Spec("zone:int fare:float".to_string()))
}

/// The host's runtime half, as its generated source would write it: a
/// page-level `fn __data` shadows the prelude fallback and answers the
/// name with a real frame.
const TRIPS_DATA: &str = "use std::polars::*;
    fn __data(name: str) -> DataFrame {
        from_csv(\"zone,fare\\n1,12.5\\n2,9.0\")!
    }";

fn compile_err(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("expected a compile error:\n{src}"),
        Err(e) => format!("{e}"),
    }
}

#[test]
fn data_source_schema_types_pull_and_query() {
    let _g = ResolverGuard::install(data_resolver);
    // `.format(2)` on a pulled element compiles only because `pull` knows `fare: float`
    let got = test_runner::render(
        TRIPS_DATA,
        r#"query { data "trips" filter fare > 10 }.pull("fare")![0].format(2)"#,
    );
    assert_eq!(got, r#""12.50""#);
}

#[test]
fn data_source_schema_propagates_through_a_let() {
    let _g = ResolverGuard::install(data_resolver);
    let got = test_runner::render(
        "use std::polars::*;
         fn __data(name: str) -> DataFrame { from_csv(\"zone,fare\\n1,12.5\")! }
         let trips = data \"trips\";",
        r#"trips.pull("zone")![0] + 10"#,
    );
    assert_eq!(got, "11");
}

#[test]
fn data_source_join_merges_resolved_schemas() {
    let _g = ResolverGuard::install(|name| match name {
        "trips" => Some(solve::DataSchema::Spec("zone:int fare:float".into())),
        "zones" => Some(solve::DataSchema::Spec("zone:int label:str".into())),
        _ => None,
    });
    let got = test_runner::render(
        "use std::polars::*;
         fn __data(name: str) -> DataFrame {
             if name == \"zones\" { from_csv(\"zone,label\\n1,a\\n2,b\")! }
             else { from_csv(\"zone,fare\\n1,12.5\\n2,9.0\")! }
         }",
        r#"query { data "trips" join data "zones" zone
           select zone fare label }
           .pull("label")![0]"#,
    );
    assert_eq!(got, r#""a""#);
}

#[test]
fn data_source_typo_column_is_a_check_error() {
    let _g = ResolverGuard::install(data_resolver);
    let err = compile_err(r#"let df = query { data "trips" filter frae > 5 };"#);
    assert!(err.contains("no column `frae`"), "{err}");
    assert!(err.contains("zone, fare"), "{err}");
}

#[test]
fn data_source_unknown_name_is_a_check_error() {
    let _g = ResolverGuard::install(data_resolver);
    let err = compile_err(r#"let df = data "trpis";"#);
    assert!(err.contains("no data source `trpis`"), "{err}");
}

#[test]
fn data_source_opaque_answer_stays_permissive() {
    let _g = ResolverGuard::install(|name| (name == "trips").then_some(solve::DataSchema::Opaque));
    let src = "use std::polars::*;
        fn __data(name: str) -> DataFrame { from_csv(\"zone,fare\\n1,12.5\")! }
        let df = data \"trips\";
        df.pull(\"frae\")!;";
    // compiles (no schema to check against), fails at runtime like a
    // `from_csv` frame
    assert!(vm::Vm::compile(src, library::std).is_ok());
    assert!(test_runner::try_execute(src).is_err());
}

#[test]
fn data_source_unbound_faults_at_runtime() {
    let _g = ResolverGuard::install(data_resolver); // schema resolves, but no `fn __data` binds it
    let err = match vm::Vm::execute(r#"let df = data "trips";"#, library::std) {
        Ok(_) => panic!("expected the unbound `data` source to fault"),
        Err(e) => format!("{e}"),
    };
    assert!(err.contains("no source named `trips`"), "{err}");
}

#[test]
fn data_source_without_resolver_compiles_opaque() {
    // no resolver installed: `data "…"` is an ordinary opaque DataFrame --
    // a bare `mimas check` on a page never fails for want of a host.
    assert!(vm::Vm::compile(r#"let df = data "trips";"#, library::std).is_ok());
}

// ---- `query z { … }`: the same verbs over Z-sets --------------------------------------------
// Each row has an integer weight, `w`. A plain table enters at weight 1 per row, identical
// rows merge by summing their weights, and a row whose weights sum to zero drops out.

const ZROWS: &str = r#"use std::polars::*;
    let dup = table { k, v
        "a", 1
        "a", 1
        "b", 2
    };
    let three = table { k, v
        "a", 5
        "a", 5
        "a", 5
        "b", 1
    };
    let tags = table { k, tag
        "a", "x"
        "a", "x"
        "b", "y"
    };"#;

// r[verify mimas.query.zset.weight]
// r[verify mimas.query.zset.consolidate]
test_run!(
    zset_duplicate_rows_merge_into_one_weight,
    ZROWS,
    r#"query z { dup }.pull("k")!"# => r#"["a", "b"]"#,
    r#"query z { dup }.pull("w")!"# => "[2, 1]",
    // the weight is the last column, and it is an int
    r#"query z { dup }.col_names()"# => r#"["k", "v", "w"]"#,
    // plain `query { }` keeps every row and adds no weight
    r#"query { dup }.pull("k")!"# => r#"["a", "a", "b"]"#,
    r#"query { dup }.col_names()"# => r#"["k", "v"]"#,
);

// r[verify mimas.query.zset.weight]
test_run!(
    zset_filter_derive_rename_select_keep_the_weight,
    ZROWS,
    r#"query z {
        three
        filter v > 1
        derive v10 = v * 10
        rename k key
    }.col_names()"# => r#"["key", "v", "v10", "w"]"#,
    r#"query z {
        three
        filter v > 1
    }.pull("w")!"# => "[3]",
    // a projection that makes rows equal merges them
    r#"query z {
        tags
        select tag
    }.pull("w")!"# => "[2, 1]",
    r#"query z {
        tags
        select tag
    }.col_names()"# => r#"["tag", "w"]"#,
);

// r[verify mimas.query.zset.join]
test_run!(
    zset_join_multiplies_weights,
    ZROWS,
    // a (weight 2) meets x (weight 2): one row of weight 4
    r#"{ let dz = query z { dup };
       let tz = query z { tags };
       query z { dz join tz k }.pull("w")! }"# => "[4, 1]",
    r#"{ let dz = query z { dup };
       let tz = query z { tags };
       query z { dz join tz k }.col_names() }"# => r#"["k", "v", "tag", "w"]"#,
    // an unmatched row of a left join keeps its own weight
    r#"{ let dz = query z { dup };
       let who = table { k
           "a"
           "c"
       };
       query z { who join dz k "left" }.pull("w")! }"# => "[2, 1]",
);

// r[verify mimas.query.zset.aggregate]
test_run!(
    zset_aggregate_counts_each_row_by_its_weight,
    ZROWS,
    r#"{ let z3 = query z { three };
       query z {
           z3
           group k { aggregate { n = count(), s = sum(v) } }
           sort k
       }.pull("n")! }"# => "[3, 1]",
    r#"{ let z3 = query z { three };
       query z {
           z3
           group k { aggregate { n = count(), s = sum(v) } }
           sort k
       }.pull("s")! }"# => "[15, 1]",
    // each group is one row of weight 1
    r#"{ let z3 = query z { three };
       query z {
           z3
           group k { aggregate { n = count() } }
       }.pull("w")! }"# => "[1, 1]",
    // a row of negative weight counts against the total
    r#"{ let gone = changes(table { k
           "a"
       }, table { k
           "a"
           "b"
       });
       query z {
           gone
           group k { aggregate { n = count() } }
       }.pull("n")! }"# => "[-1]",
    // `mean` is the weighted mean: (5 * 3 + 1) / 4
    r#"{ let z3 = query z { three };
       query z {
           z3
           group { aggregate { m = mean(v) } }
       }.pull("m")! }"# => "[4.0]",
);

// The grocery basket: an aggregate reads the weights directly, so a change of
// -1 bread takes a bread's price off the total.
const BASKET: &str = r#"use std::polars::*;
    let basket = table { item, w
        "apple", 2
        "bread", 1
    };
    let change = table { item, w
        "apple", 1
        "bread", -1
    };
    let prices = table { item, price
        "apple", 0.50
        "bread", 2.00
    };
    fn total(b: DataFrame) -> str {
        let t: [float] = query z {
            b
            join prices item
            group { aggregate { t = sum(price) } }
        }.pull("t")!;
        t[0].format(2)
    }"#;

// r[verify mimas.query.zset.aggregate]
test_run!(
    zset_aggregate_sums_by_weight,
    BASKET,
    r#"total(basket)"# => r#""3.00""#,
    // a delta Z-set totals the change: +0.50 for an apple, -2.00 for a bread
    r#"total(change)"# => r#""-1.50""#,
    r#"total(basket + change)"# => r#""1.50""#,
    // a delta of one bread taken away
    r#"total(table { item, w
        "bread", -1
    })"# => r#""-2.00""#,
    // `count()` adds the weights
    r#"query z {
        basket
        group { aggregate { n = count() } }
    }.pull("n")!"# => "[3]",
);

// r[verify mimas.query.zset.aggregate.other]
test_run!(
    zset_min_max_read_only_rows_of_positive_weight,
    BASKET,
    // bread has weight -1 in `change`, so it is not a value for `max`
    r#"query z {
        change
        join prices item
        group { aggregate { hi = max(price), lo = min(price) } }
    }.pull("hi")!"# => "[0.5]",
    r#"query z {
        basket
        join prices item
        group { aggregate { hi = max(price), lo = min(price) } }
    }.pull("hi")!"# => "[2.0]",
);

// `median` in `query z` is the median of the copies the weights stand for,
// and `unweighted_median` counts each row of positive weight once.
const ZMEDIAN: &str = r#"use std::polars::*;
    let skew = table { v, w
        1, 3
        10, 1
    };
    let even = table { v, w
        1, 1
        2, 1
        10, 2
    };
    let gone = table { v, w
        1, 1
        5, 1
        7, -4
        9, 1
    };
    let grouped = table { k, v, w
        "a", 1, 3
        "a", 10, 1
        "b", 2, 1
        "b", 4, 1
        "b", 6, 5
    };
    let plain = table { v
        1
        2
        3
        4
    };
    fn med(t: DataFrame) -> [float] {
        query z {
            t
            group { aggregate { m = median(v) } }
        }.pull("m")!
    }
    fn umed(t: DataFrame) -> [float] {
        query z {
            t
            group { aggregate { m = unweighted_median(v) } }
        }.pull("m")!
    }"#;

// r[verify mimas.query.zset.aggregate.median]
test_run!(
    zset_median_weighs_each_row,
    ZMEDIAN,
    // the copies are 1, 1, 1, 10: the middle two are both 1
    r#"med(skew)"# => "[1.0]",
    // the copies are 1, 2, 10, 10: the middle falls between 2 and 10
    r#"med(even)"# => "[6.0]",
    // the row of weight -4 is not a value: the copies are 1, 5, 9
    r#"med(gone)"# => "[5.0]",
    // weight 1 on every row gives the plain median
    r#"med(plain)"# => "[2.5]",
    r#"query {
        plain
        group { aggregate { m = median(v) } }
    }.pull("m")!"# => "[2.5]",
    // one median per group: a is 1, 1, 1, 10 and b is 2, 4, 6, 6, 6, 6, 6
    r#"query z {
        grouped
        group k { aggregate { m = median(v) } }
        sort k
    }.pull("m")!"# => "[1.0, 6.0]",
);

// r[verify mimas.query.zset.aggregate.other]
test_run!(
    zset_unweighted_median_counts_each_row_once,
    ZMEDIAN,
    r#"umed(skew)"# => "[5.5]",
    r#"umed(even)"# => "[2.0]",
    // the row of weight -4 is still not a value
    r#"umed(gone)"# => "[5.0]",
    r#"query z {
        grouped
        group k { aggregate { m = unweighted_median(v) } }
        sort k
    }.pull("m")!"# => "[5.5, 4.0]",
    // outside `query z`, `unweighted_median` is `median`
    r#"query {
        plain
        group { aggregate { m = unweighted_median(v) } }
    }.pull("m")!"# => "[2.5]",
);

// r[verify mimas.query.zset.add]
test_run!(
    zset_tables_add_and_subtract,
    r#"use std::polars::*;
       let z1 = table { item, w
           "apple", 2
           "pear", 1
       };
       let z2 = table { item, w
           "apple", 1
           "pear", -1
           "fig", 1
       };
       let plain = table { item
           "apple"
           "fig"
       };"#,
    // equal rows add their weights, and pear comes to 0 and drops out
    r#"(z1 + z2).pull("item")!"# => r#"["apple", "fig"]"#,
    r#"(z1 + z2).pull("w")!"# => "[3, 1]",
    // a Z-set minus itself is empty
    r#"(z1 - z1).pull("item")!.len()"# => "0",
    // `-` is `changes`
    r#"(z1 - z2).pull("w")! == changes(z1, z2).pull("w")!"# => "true",
    r#"(z1 - z2).pull("item")!"# => r#"["apple", "pear", "fig"]"#,
    r#"(z1 - z2).pull("w")!"# => "[1, 2, -1]",
    // a plain frame counts each row at weight 1
    r#"(z1 + plain).pull("w")!"# => "[3, 1, 1]",
    r#"(plain + plain).pull("w")!"# => "[2, 2]",
    r#"(plain + plain).col_names()"# => r#"["item", "w"]"#,
    // column order does not matter
    r#"{ let a = table { k, v
           "x", 1
       };
       let b = table { v, k
           1, "x"
       };
       (a + b).pull("w")! }"# => "[2]",
);

// r[verify mimas.query.zset.add]
#[test]
fn zset_add_with_different_columns_is_an_error() {
    let src = r#"use std::polars::*;
        let a = table { item, w
            "apple", 2
        };
        let b = table { name, w
            "apple", 1
        };
        let c = a + b;"#;
    let err = compile_err(src);
    // the report wraps its lines, so check short pieces
    assert!(err.contains("same column names"), "{err}");
    assert!(err.contains("has `item`"), "{err}");
    assert!(err.contains("has `name`"), "{err}");
    // frames whose columns the checker cannot see fault at run time
    let src = r#"use std::polars::*;
        let a = from_csv("item,w\napple,2")!;
        let b = from_csv("name,w\napple,1")!;
        let c = a - b;"#;
    let err = match test_runner::try_execute(src) {
        Ok(_) => panic!("expected different columns to fault"),
        Err(e) => format!("{e}"),
    };
    assert!(err.contains("same column names"), "{err}");
    assert!(err.contains("`item`"), "{err}");
}

// r[verify mimas.query.zset.distinct]
test_run!(
    zset_distinct_sets_each_weight_to_one,
    ZROWS,
    r#"query z {
        three
        distinct
    }.pull("w")!"# => "[1, 1]",
    // a row of negative weight is not in the set
    r#"{ let d = changes(table { k
           "a"
       }, table { k
           "c"
       });
       query z {
           d
           distinct
       }.pull("k")! }"# => r#"["a"]"#,
);

// r[verify mimas.query.zset.consolidate]
test_run!(
    zset_sort_and_take_see_consolidated_rows,
    ZROWS,
    // two copies of a are one row, so `take 2` reaches b
    r#"query z {
        dup
        take 2
    }.pull("k")!"# => r#"["a", "b"]"#,
    r#"query {
        dup
        take 2
    }.pull("k")!"# => r#"["a", "a"]"#,
    r#"query z {
        dup
        sort -k
    }.pull("w")!"# => "[1, 2]",
);

// r[verify mimas.query.zset.changes]
test_run!(
    zset_changes_is_new_minus_old,
    r#"use std::polars::*;
       let old = table { k, n
           "a", 1
           "b", 2
       };
       let new = table { k, n
           "a", 1
           "b", 3
           "c", 4
       };"#,
    r#"changes(new, old).pull("k")!"# => r#"["b", "c", "b"]"#,
    r#"changes(new, old).pull("n")!"# => "[3, 4, 2]",
    r#"changes(new, old).pull("w")!"# => "[1, 1, -1]",
    // nothing changed: no rows
    r#"changes(new, new).pull("k")!.len()"# => "0",
    // a Z-set's own weights carry: two copies gone is weight -2, and
    // b, in both once, cancels
    r#"{ let two = query z {
           table { k, v
               "a", 1
               "a", 2
               "b", 3
           }
           select k
       };
       let one_b = table { k
           "b"
       };
       changes(one_b, two).pull("w")! }"# => "[-2]",
);

// r[verify mimas.query.zset.adopt]
test_run!(
    zset_int_w_column_is_the_weights,
    r#"use std::polars::*;
       let weighed = table { k, w
           "a", 2
           "b", 1
           "a", 3
       };"#,
    // a table with an int `w` is a Z-set already: consolidation sums its weights
    r#"query z { weighed }.pull("w")!"# => "[5, 1]",
    r#"query z { weighed }.col_names()"# => r#"["k", "w"]"#,
    // a join multiplies the two `w`s and keeps no `w_right`
    r#"{ let other = table { k, w
           "a", 10
       };
       query z { weighed join other k }.col_names() }"# => r#"["k", "w"]"#,
    r#"{ let other = table { k, w
           "a", 10
       };
       query z { weighed join other k }.pull("w")! }"# => "[50]",
    // plain `query { }` leaves `w` an ordinary column
    r#"query { weighed }.pull("w")!"# => "[2, 1, 3]",
);

// r[verify mimas.query.zset.reject]
#[test]
fn zset_float_w_column_is_a_check_error() {
    let err = compile_err(
        r#"use std::polars::*;
           let boxes = table { k, w
               "a", 1.5
           };
           let z = query z { boxes };"#,
    );
    assert!(err.contains("this table's `w` is float"), "{err}");
    assert!(err.contains(r#".rename("w", "width")!"#), "{err}");
    // plain `query { }` takes the same table
    assert!(
        vm::Vm::compile(
            r#"use std::polars::*;
               let boxes = table { k, w
                   "a", 1.5
               };
               let q = query { boxes };"#,
            library::std
        )
        .is_ok()
    );
}

// r[verify mimas.query.zset.reject]
#[test]
fn zset_float_w_column_faults_at_run_time() {
    // a csv frame's columns are unknown to the checker; the runtime catches it
    let src = r#"use std::polars::*;
        let boxes = from_csv("k,w\na,1.5")!;
        let z = query z { boxes };"#;
    let err = match test_runner::try_execute(src) {
        Ok(_) => panic!("expected a float `w` to fault"),
        Err(e) => format!("{e}"),
    };
    // the report wraps its lines, so check short pieces
    assert!(err.contains("is float"), "{err}");
    assert!(err.contains("whole numbers"), "{err}");
}

// ---- Z-sets: abelian group operations on weighted tables -------------------------------------
// `-z`, `k * z`, `z.added()`, `z.removed()`, `z.size()`, `z.unweighted()`, `z.as_set()`, the
// `distinct` keep policy, and `integrate`/`differentiate` over a list. Two Z-sets are equal when
// their difference is empty, `(a - b).size() == 0`, whatever order their rows come in.

const ZGROUP: &str = r#"use std::polars::*;
    let a = table { item, w
        "apple", 2
        "bread", -1
        "fig", 3
    };
    let b = table { item, w
        "apple", 1
        "bread", 1
        "pear", -2
    };
    let plain = table { item
        "apple"
        "apple"
        "fig"
    };
    fn same(x: DataFrame, y: DataFrame) -> bool { (x - y).size() == 0 }
    fn all_same(xs: [DataFrame], ys: [DataFrame]) -> bool {
        if xs.len() != ys.len() { return false; }
        for i in 0..xs.len() {
            if !same(xs[i], ys[i]) { return false; }
        }
        true
    }"#;

// r[verify mimas.query.zset.negate]
test_run!(
    zset_negate_undoes_a_change,
    ZGROUP,
    r#"(-a).pull("w")!"# => "[-2, 1, -3]",
    r#"(-a).col_names()"# => r#"["item", "w"]"#,
    // z + (-z) is the empty Z-set
    r#"(a + -a).size()"# => "0",
    r#"(a + -a).pull("item")!.len()"# => "0",
    r#"same(-(-a), a)"# => "true",
    // a plain table counts each row at weight 1, consolidated first
    r#"(-plain).pull("w")!"# => "[-2, -1]",
    // `+z` is `z` consolidated
    r#"(+plain).pull("w")!"# => "[2, 1]",
);

// r[verify mimas.query.zset.scale]
test_run!(
    zset_scale_applies_a_change_k_times,
    ZGROUP,
    r#"(3 * a).pull("w")!"# => "[6, -3, 9]",
    r#"(a * 3).pull("w")!"# => "[6, -3, 9]",
    r#"(-2 * a).pull("w")!"# => "[-4, 2, -6]",
    // k = 0 is the empty Z-set, with the same columns
    r#"(0 * a).size()"# => "0",
    r#"(0 * a).col_names()"# => r#"["item", "w"]"#,
    r#"same(2 * a, a + a)"# => "true",
    r#"same(-1 * a, -a)"# => "true",
    // k * (a + b) == k * a + k * b
    r#"same(3 * (a + b), 3 * a + 3 * b)"# => "true",
    r#"same(-2 * (a + b), -2 * a + -2 * b)"# => "true",
    // (j + k) * a == j * a + k * a
    r#"same((2 + 5) * a, 2 * a + 5 * a)"# => "true",
    r#"{ let k = 4; (k * plain).pull("w")! }"# => "[8, 4]",
);

// r[verify mimas.query.zset.scale]
#[test]
fn zset_scale_by_a_float_is_an_error() {
    let err = compile_err(
        r#"use std::polars::*;
           let a = table { item
               "apple"
           };
           let z = 1.5 * a;"#,
    );
    assert!(err.contains("whole number"), "{err}");
    assert!(err.contains("float"), "{err}");
    let err = compile_err(
        r#"use std::polars::*;
           let a = table { item
               "apple"
           };
           let z = a * 2.0;"#,
    );
    assert!(err.contains("whole number"), "{err}");
}

// r[verify mimas.query.zset.parts]
test_run!(
    zset_added_and_removed_are_the_two_parts,
    ZGROUP,
    r#"a.added().pull("item")!"# => r#"["apple", "fig"]"#,
    r#"a.added().pull("w")!"# => "[2, 3]",
    r#"a.removed().pull("item")!"# => r#"["bread"]"#,
    // Z⁻ holds positive weights
    r#"a.removed().pull("w")!"# => "[1]",
    r#"a.removed().col_names()"# => r#"["item", "w"]"#,
    // a == a.added() - a.removed()
    r#"same(a, a.added() - a.removed())"# => "true",
    r#"same(b, b.added() - b.removed())"# => "true",
    // a plain table has no removed part
    r#"plain.removed().size()"# => "0",
    r#"plain.added().pull("w")!"# => "[2, 1]",
    // the free-function spelling
    r#"added(a).pull("w")!"# => "[2, 3]",
);

// r[verify mimas.query.zset.size]
test_run!(
    zset_size_is_the_sum_of_absolute_weights,
    ZGROUP,
    r#"a.size()"# => "6",
    r#"b.size()"# => "4",
    r#"plain.size()"# => "3",
    // size(a) == size(a.added()) + size(a.removed())
    r#"a.size() == a.added().size() + a.removed().size()"# => "true",
    r#"b.size() == b.added().size() + b.removed().size()"# => "true",
    // weights that cancel are not part of the size
    r#"(a + b).size()"# => "8",
    r#"{ let s: int = size(a); s }"# => "6",
);

// r[verify mimas.query.zset.unweighted]
test_run!(
    zset_unweighted_drops_the_weights,
    ZGROUP,
    r#"a.unweighted().col_names()"# => r#"["item"]"#,
    // the rows of positive weight, each once
    r#"a.unweighted().pull("item")!"# => r#"["apple", "fig"]"#,
    r#"plain.unweighted().pull("item")!"# => r#"["apple", "fig"]"#,
    // the checker knows `w` is gone
    r#"{ let u = a.unweighted(); u.col_names().len() }"# => "1",
);

// r[verify mimas.query.zset.unweighted]
#[test]
fn zset_unweighted_has_no_w_at_check_time() {
    let err = compile_err(
        r#"use std::polars::*;
           let a = table { item, w
               "apple", 2
           };
           let ws = a.unweighted().pull("w")!;"#,
    );
    assert!(err.contains("`w`"), "{err}");
}

// r[verify mimas.query.zset.as_set]
test_run!(
    zset_as_set_is_distinct,
    ZGROUP,
    r#"a.as_set().pull("item")!"# => r#"["apple", "fig"]"#,
    r#"a.as_set().pull("w")!"# => "[1, 1]",
    r#"same(a.as_set(), query z { a distinct })"# => "true",
    r#"same(plain.as_set(), query z { plain distinct })"# => "true",
);

const ZKEEP: &str = r#"use std::polars::*;
    let orders = table { item, qty
        "apple", 1
        "bread", 4
        "apple", 2
        "apple", 3
        "bread", 5
    };
    let gone = table { item, qty, w
        "apple", 1, -1
        "apple", 2, 1
        "apple", 3, 1
    };"#;

// r[verify mimas.query.zset.coalesce]
test_run!(
    zset_distinct_keeps_the_row_its_policy_names,
    ZKEEP,
    // the default is `first`
    r#"query z { orders distinct item }.pull("qty")!"# => "[1, 4]",
    r#"query z { orders distinct item first }.pull("qty")!"# => "[1, 4]",
    r#"query z { orders distinct item last }.pull("qty")!"# => "[3, 5]",
    r#"query z { orders distinct item last }.pull("w")!"# => "[1, 1]",
    // `any` keeps one row of each key, in no set order
    r#"query z { orders distinct item any }.pull("item")!.len()"# => "2",
    r#"query z { orders distinct item any }.pull("w")!"# => "[1, 1]",
    r#"query z {
        orders
        distinct item any
        sort item
    }.pull("item")!"# => r#"["apple", "bread"]"#,
    // a row of negative weight is not a candidate
    r#"query z { gone distinct item first }.pull("qty")!"# => "[2]",
    r#"query z { gone distinct item last }.pull("qty")!"# => "[3]",
    // plain `query { }` takes the same policy
    r#"query { orders distinct item last }.pull("qty")!"# => "[3, 5]",
    r#"query { orders distinct item }.pull("qty")!"# => "[1, 4]",
    // a key named like a policy is quoted: `"last"` is a second key here
    r#"{ let t = table { k, last
           "a", 1
           "a", 2
           "b", 1
       };
       query { t distinct k "last" }.pull("k")! }"# => r#"["a", "a", "b"]"#,
    r#"{ let t = table { k, last
           "a", 1
           "a", 2
           "b", 1
       };
       query { t distinct k last }.pull("last")! }"# => "[2, 1]",
);

// r[verify mimas.query.zset.integrate]
// r[verify mimas.query.zset.differentiate]
// r[verify mimas.query.zset.delay]
test_run!(
    zset_integrate_and_differentiate_a_stream,
    ZGROUP,
    r#"[a, b].integrate().len()"# => "2",
    r#"[a, b].integrate()[1].pull("item")!"# => r#"["apple", "fig", "pear"]"#,
    r#"[a, b].integrate()[1].pull("w")!"# => "[3, 3, -2]",
    r#"same([a, b, plain].integrate()[2], a + b + plain)"# => "true",
    r#"[a, b].differentiate()[1].pull("w")! == (b - a).pull("w")!"# => "true",
    r#"same([a, b].differentiate()[0], a)"# => "true",
    // D undoes I, and I undoes D
    r#"{ let zs = [a, b, plain, -a]; all_same(zs.integrate().differentiate(), zs) }"# => "true",
    r#"{ let zs = [a, b, plain, -a]; all_same(zs.differentiate().integrate(), zs) }"# => "true",
    r#"{ let zs: [DataFrame] = []; zs.integrate().len() }"# => "0",
    // z⁻¹: the stream one step late, from the empty Z-set
    r#"[a, b, plain].delay().len()"# => "3",
    r#"[a, b, plain].delay()[0].size()"# => "0",
    r#"[a, b, plain].delay()[0].col_names()"# => r#"["item", "w"]"#,
    r#"same([a, b, plain].delay()[2], b)"# => "true",
    r#"{ let zs: [DataFrame] = []; zs.delay().len() }"# => "0",
    // D = 1 - z⁻¹, item by item (a list minus a list is not elementwise, so loop)
    r#"{ let zs = [a, b, plain, -a];
       let d = zs.differentiate();
       let late = zs.delay();
       let ok = true;
       for i in 0..zs.len() {
           if !same(d[i], zs[i] - late[i]) { ok = false; }
       }
       ok }"# => "true",
    r#"integrate([a]).len()"# => "1",
);

// The laws of the group, and two uses of them. A sum aggregate is linear, so it
// distributes over `+`; a join is bilinear, so its change follows the product rule.
const ZLAWS: &str = r#"use std::polars::*;
    let a = table { item, w
        "apple", 2
        "bread", -1
    };
    let b = table { item, w
        "apple", 1
        "fig", 4
    };
    let prices = table { item, price
        "apple", 1
        "bread", 3
        "fig", 5
    };
    let stock = table { item, shelf, w
        "apple", "A", 1
        "fig", "B", 2
    };
    let dstock = table { item, shelf, w
        "apple", "A", 1
        "bread", "C", 1
        "fig", "B", -1
    };
    fn same(x: DataFrame, y: DataFrame) -> bool { (x - y).size() == 0 }
    fn total(z: DataFrame) -> int {
        let t: [int] = query z {
            z
            join prices item
            group { aggregate { t = sum(price) } }
        }.pull("t")!;
        t[0]
    }
    fn j(x: DataFrame, y: DataFrame) -> DataFrame {
        query z { x join y item }
    }"#;

// r[verify mimas.query.zset.negate]
// r[verify mimas.query.zset.scale]
// r[verify mimas.query.zset.parts]
// r[verify mimas.query.zset.size]
// r[verify mimas.query.zset.add]
test_run!(
    zset_abelian_group_laws,
    ZLAWS,
    // zero, inverse, commutative, associative
    r#"(a + -a).size()"# => "0",
    r#"same(a + b, b + a)"# => "true",
    r#"same((a + b) + stock.unweighted().select_names(["item"])!, a + (b + stock.unweighted().select_names(["item"])!))"# => "true",
    r#"same(a - b, a + -b)"# => "true",
    // k * (a + b) == k * a + k * b
    r#"same(5 * (a + b), 5 * a + 5 * b)"# => "true",
    r#"same(a, a.added() - a.removed())"# => "true",
    r#"a.size() == a.added().size() + a.removed().size()"# => "true",
    // linearity: a weighted sum over a + b is the sum over a plus the sum over b
    r#"total(a + b) == total(a) + total(b)"# => "true",
    // apple 3 at 1, bread -1 at 3, fig 4 at 5
    r#"total(a + b)"# => "20",
    r#"total(3 * a) == 3 * total(a)"# => "true",
    r#"total(-a) == -total(a)"# => "true",
    // the product rule: Δ(A ⋈ B) = ΔA ⋈ B + A ⋈ ΔB + ΔA ⋈ ΔB
    r#"same(
        j(a + b, stock + dstock) - j(a, stock),
        j(b, stock) + j(a, dstock) + j(b, dstock)
    )"# => "true",
);
