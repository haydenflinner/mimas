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
    r#"df.pull("dist")![0] + 100m"# => "5100",
);

// an opaque frame stays permissive -- pull keeps its old generic array type and a
// bad name still only fails at runtime, not at check time
test_fail!(
    unschematized_pull_of_a_bad_column_fails_at_runtime,
    r#"use std::polars::*;
       let df = from_csv("a,b\n1,2")!;
       df.pull("nope")!;"#,
);
