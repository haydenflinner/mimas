//! Automatic semicolon insertion (Go's rules, applied where mimas's own grammar already knows a
//! `;` was mandatory -- see `Parser::newline_before_next` and its call sites in
//! `crates/parse/src/parser.rs`). These are end-to-end (compile + run) rather than parse-tree
//! shape checks, since the point of the feature is that ordinary semicolon-free code behaves
//! identically to its semicolon'd equivalent.

#[macro_use]
mod vm_test_utils;

use vm::Captured::*;

test_vm!(
    sequential_statements_without_semicolons,
    "{
        let a = 1
        let b = 2
        a + b
    }" => Int(3),
);

test_vm!(
    // Go itself requires `} else {` cuddled onto one line: a lexer that blindly inserts a `;`
    // after every trigger-token-then-newline would insert one right between the `}` and `else`,
    // breaking the chain. `newline_before_next` is only ever consulted at an already-identified
    // statement/item boundary (inside `semicolon_check`/`ok_if_semicolon`), and `if`/`else` are
    // parsed as one expression at the grammar level regardless of what's between them -- so this
    // was never at risk here to begin with. This test pins that down.
    uncuddled_if_else_chain_without_semicolons,
    "fn classify(n: int) -> int {
        if n < 0 { 0 }
        else if n == 0 { 1 }
        else { 2 }
    }",
    "classify(-5)" => Int(0),
    "classify(0)" => Int(1),
    "classify(5)" => Int(2),
);

test_vm!(
    // Regression: `break`/`return` with no value, as a block's last statement with no `;`,
    // used to make `optional_expr` try to parse the following `}` as the start of a value
    // expression -- see the fix in `Parser::optional_expr`.
    bare_break_before_closing_brace,
    "{
        let i = 0
        loop {
            if i >= 3 {
                break
            }
            i += 1
        }
        i
    }" => Int(3),
);

test_vm!(
    bare_return_before_closing_brace,
    "fn f(x: int) -> int {
        if x > 0 {
            return x
        }
        0
    }",
    "f(5)" => Int(5),
    "f(-5)" => Int(0),
);

test_vm!(
    // A multi-line argument list's last argument, alone on its own line right before the
    // closing `)`, must not have a semicolon inserted after it -- it's not even a statement
    // boundary, so `newline_before_next` (only ever consulted at an *already-identified*
    // statement/item boundary) never gets a chance to misfire here regardless.
    multi_line_call_args_without_semicolons,
    "fn add(a: int, b: int) -> int { a + b }",
    "add(
        1,
        2
    )" => Int(3),
);

test_vm!(
    // A block's tail expression (no trailing `;`, yields the block's value) spanning its own
    // line right before the closing `}` must keep yielding, not get treated as a
    // semicolon-needing statement.
    tail_expression_before_closing_brace_still_yields,
    "fn add(a: int, b: int) -> int {
        a + b
    }",
    "add(2, 3)" => Int(5),
);
