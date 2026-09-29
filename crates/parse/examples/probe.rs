use parse::Parser;
use parse::lex::Lexer;
fn main() {
    for src in [
        "check 1 |> add3(10, 20, 30) == 61\n",
        "check f(2) == 3\n",
        "let x = 1 |> add3(10, 20, 30);\n",
    ] {
        let p = Parser::new(Lexer::new(src, 0, "t".into()));
        let (ast, diags) = p.into_ast();
        eprintln!(
            "{:?} => stmts={} diags={:?}",
            src,
            ast.stmts().len(),
            diags.iter().map(|d| format!("{}", d)).collect::<Vec<_>>()
        );
    }
}
