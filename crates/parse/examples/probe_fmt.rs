use parse::Parser;
use parse::lex::Lexer;
fn main() {
    for src in [
        "x ! + y\n",
        "x ?(1)\n",
        "check -1\n",
        "check - 1\n",
        "check [-1]\n",
        "a-1\n",
        "unit tick = 9.8 m / s ^ 2\n",
        "let x: {int} = dict\n",
        "{a: 1}\n",
        "x?(1)\n",
        "d?[k] \u{b7} x?.f()\n",
        "a % 2\n",
        "x = -y\n",
        "let xs = [0 -1]\n",
        "let xs = [0 - 1]\n",
        "fn f() {\n    x\n        // odd\n    y\n}\n",
    ] {
        let p = Parser::new(Lexer::new(src, 0, "t".into()));
        let (ast, diags) = p.into_ast();
        eprintln!(
            "{:?} => stmts={} diags={:?}",
            src.replace('\n', "\\n"),
            ast.stmts().len(),
            diags.iter().map(|d| format!("{d}")).collect::<Vec<_>>()
        );
    }
}
