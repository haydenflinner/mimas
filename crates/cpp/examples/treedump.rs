fn show(n: tree_sitter::Node, src: &str, d: usize, out: &mut String) {
    for i in 0..n.child_count() {
        let c = n.child(i).unwrap();
        let f = n.field_name_for_child(i as u32).unwrap_or("-");
        let text: String = c.utf8_text(src.as_bytes()).unwrap_or("").chars().take(40).collect();
        out.push_str(&format!("{}{} [{}] {:?}\n", "  ".repeat(d), f, c.kind(), text));
        show(c, src, d + 1, out);
    }
}
fn main() {
    let src = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).unwrap();
    let t = p.parse(&src, None).unwrap();
    let r = t.root_node();
    let mut out = String::new();
    show(r, &src, 0, &mut out);
    print!("{out}");
}
