use crate::tests::utils::parse;
use shared::Located;

fn docs_at(source: &str, token: &str) -> Option<String> {
    let (ast, errors) = parse(source);
    assert!(errors.is_empty(), "{errors:?}");
    ast.docs_for(source.find(token).unwrap())
        .map(str::to_string)
}

#[test]
fn doc_is_at_the_start_of_its_item() {
    let source = "/// Adds.
                  pub fn add() {}";
    let (ast, errors) = parse(source);
    assert!(errors.is_empty(), "{errors:?}");
    let start = ast.stmts()[0].span().start;
    assert_eq!(ast.docs_for(start), Some("Adds."));
}

#[test]
fn doc_lines_lose_their_slashes() {
    let source = "/// Adds two numbers.
                  ///
                  /// Returns the sum.
                  fn add() {}";
    let docs = docs_at(source, "fn").unwrap();
    assert_eq!(
        docs.lines().collect::<Vec<_>>(),
        ["Adds two numbers.", "", "Returns the sum."]
    );
}

#[test]
fn only_the_closest_doc_counts() {
    let source = "/// a

                  /// b
                  fn foo() {}";
    assert_eq!(docs_at(source, "fn").as_deref(), Some("b"));
}

#[test]
fn plain_comment_between_doc_and_item() {
    let source = "/// Docs.
                  // todo
                  fn foo() {}";
    assert_eq!(docs_at(source, "fn").as_deref(), Some("Docs."));
}

#[test]
fn plain_comment_is_not_a_doc() {
    let source = "// not docs
                  fn foo() {}";
    assert_eq!(docs_at(source, "fn"), None);
}

/// `>>` closes two generics at once: the parser splits it into two `>`
/// tokens. Every token after it then sits one place later, and a doc
/// comment after it used to hang on the token before its item.
#[test]
fn doc_after_a_nested_generic_stays_on_its_item() {
    let source = "struct Pair<A, B> { a: A, b: B }
                  fn wrap(p: Pair<int, Pair<int, int>>) -> int { 1 }
                  /// After the split.
                  fn after() {}";
    assert_eq!(
        docs_at(source, "fn after").as_deref(),
        Some("After the split.")
    );
}

#[test]
fn block_doc_attaches() {
    let source = "/** Adds. */
                  fn add() {}";
    assert_eq!(docs_at(source, "fn").as_deref(), Some("Adds."));
}

#[test]
fn block_doc_lines_lose_their_stars() {
    let source = "/** Adds two.
                   * Returns the sum.
                   */
                  fn add() {}";
    let docs = docs_at(source, "fn").unwrap();
    assert_eq!(
        docs.lines().collect::<Vec<_>>(),
        ["Adds two.", "Returns the sum."]
    );
}

#[test]
fn inner_doc_is_not_the_items_doc() {
    let source = "//! scope docs
                  fn foo() {}";
    assert_eq!(docs_at(source, "fn"), None);
}

#[test]
fn literal_comment_is_not_a_doc() {
    let source = "//// dead code
                  /*** also dead */
                  fn foo() {}";
    assert_eq!(docs_at(source, "fn"), None);
}
