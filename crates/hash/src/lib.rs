//! Content addressing for Mimas items: pull top-level `fn` items out of
//! Mimas source and hash them.
//!
//! Extraction uses the real `mimas-parse` parser; a brace matcher is kept as
//! the fallback for source that doesn't parse (mid-edit states). Hashes cover
//! the item's token stream, not its bytes — reformatting or comment edits do
//! not change a function's content hash.
//!
//! This crate is the shared home of the hashing machinery: `mimas hash` on
//! the CLI and the literate host's blob store both build on it. It must never
//! depend on Automerge — namespace/blob persistence stays in the host.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub name: String,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

impl Item {
    /// An item whose byte span is unknown. `apply_item` looks the name up again.
    pub fn named(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: 0,
            end: 0,
            source: source.into(),
        }
    }

    pub fn line(&self, source: &str) -> usize {
        let start = self.start.min(source.len());
        source[..start].bytes().filter(|&b| b == b'\n').count() + 1
    }

    pub fn contains_byte(&self, byte: usize) -> bool {
        byte >= self.start && byte < self.end
    }

    pub fn n_lines(&self) -> usize {
        self.source.lines().count().max(1)
    }

    /// Inclusive 1-based line range of this item in `source`.
    pub fn line_span(&self, source: &str) -> (usize, usize) {
        let start = self.line(source);
        let n = self.n_lines();
        (start, start + n - 1)
    }

    pub fn contains_line(&self, source: &str, line: usize) -> bool {
        let (lo, hi) = self.line_span(source);
        line >= lo && line <= hi
    }

    pub fn hash(&self) -> String {
        hash_item(&self.source)
    }

    /// The blob-stored form of this item: canonical token text when it
    /// lexes, else the raw source. The hash covers these bytes.
    pub fn canonical_source(&self) -> String {
        canonical(&self.source).unwrap_or_else(|| self.source.clone())
    }

    /// Inclusive-exclusive char offsets of this item in `source`.
    pub fn char_span(&self, source: &str) -> (usize, usize) {
        let lo = source
            .get(..self.start)
            .map(|s| s.chars().count())
            .unwrap_or(0);
        let hi = source
            .get(..self.end.min(source.len()))
            .map(|s| s.chars().count())
            .unwrap_or_else(|| source.chars().count());
        (lo, hi)
    }

    pub fn byte_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn mentions(&self, name: &str) -> bool {
        mentions(&self.source, name)
    }

    pub fn overlay_line(&self, source: &str) -> String {
        format!(
            "{}  L{}  [{}..{}]",
            self.name,
            self.line(source),
            self.start,
            self.end
        )
    }

    pub fn hash_line(&self) -> String {
        format!("{}  {}", self.name, self.hash())
    }

    pub fn counts_overlay(&self, source: &str) -> String {
        let (lo, hi) = self.line_span(source);
        format!(
            "name      {}\nlines     {}\nbytes     {}\nspan      L{}-L{}",
            self.name,
            self.n_lines(),
            self.byte_len(),
            lo,
            hi
        )
    }
}

pub fn named<'a>(items: &'a [Item], name: &str) -> Option<&'a Item> {
    items.iter().find(|i| i.name == name)
}

pub fn names(items: &[Item]) -> Vec<&str> {
    items.iter().map(|i| i.name.as_str()).collect()
}

pub fn at_byte(items: &[Item], byte: usize) -> Option<&Item> {
    items.iter().find(|i| i.contains_byte(byte))
}

pub fn at_line<'a>(items: &'a [Item], source: &str, line: usize) -> Option<&'a Item> {
    items.iter().find(|i| i.contains_line(source, line))
}

pub fn unique_name(items: &[Item], base: &str) -> String {
    if named(items, base).is_none() {
        return base.to_string();
    }
    for n in 2..10_000 {
        let candidate = format!("{base}_{n}");
        if named(items, &candidate).is_none() {
            return candidate;
        }
    }
    format!("{base}_{}", items.len() + 1)
}

pub fn unique_copy_name(items: &[Item], name: &str) -> String {
    unique_name(items, &format!("{name}_copy"))
}

/// Rewrite the leading `fn old` of a hashed item body.
pub fn rename_fn(source: &str, old: &str, new: &str) -> String {
    source.replacen(&format!("fn {old}"), &format!("fn {new}"), 1)
}

/// Replace the named item in `source`, or append it. Byte offsets on `item`
/// are ignored; the name is looked up again.
pub fn apply_item(source: &str, item: &Item) -> String {
    let (lo, del, ins) = apply_item_span(source, item);
    splice_chars(source, lo, del, &ins)
}

/// Peritext edit that applies `item`: char index, chars to delete, insert.
pub fn apply_item_span(source: &str, item: &Item) -> (usize, usize, String) {
    let items = extract(source);
    if let Some(existing) = named(&items, &item.name) {
        let (lo, hi) = existing.char_span(source);
        (lo, hi.saturating_sub(lo), item.source.clone())
    } else {
        let mut ins = String::new();
        if !source.is_empty() && !source.ends_with('\n') {
            ins.push('\n');
        }
        if !source.is_empty() {
            ins.push('\n');
        }
        ins.push_str(&item.source);
        if !item.source.ends_with('\n') {
            ins.push('\n');
        }
        (source.chars().count(), 0, ins)
    }
}

fn splice_chars(source: &str, lo: usize, del: usize, ins: &str) -> String {
    let start = source
        .char_indices()
        .nth(lo)
        .map(|(i, _)| i)
        .unwrap_or(source.len());
    let end = source
        .char_indices()
        .nth(lo + del)
        .map(|(i, _)| i)
        .unwrap_or(source.len());
    let mut out = String::with_capacity(source.len() + ins.len());
    out.push_str(&source[..start]);
    out.push_str(ins);
    out.push_str(&source[end..]);
    out
}

pub fn fn_count(source: &str) -> usize {
    extract(source).len()
}

pub fn fn_counts_overlay(source: &str) -> String {
    format!("fns       {}", fn_count(source))
}

pub fn hashes_overlay(source: &str) -> String {
    let items = extract(source);
    if items.is_empty() {
        return "no fn items".into();
    }
    items
        .iter()
        .map(Item::hash_line)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn extract(source: &str) -> Vec<Item> {
    extract_parsed(source).unwrap_or_else(|| extract_braces(source))
}

/// Parser-backed extraction: real item spans (attributes included), no
/// `where:`/`examples` blocks or `// example:` comments. `None` when the
/// source doesn't parse — callers then fall back to the brace matcher.
fn extract_parsed(source: &str) -> Option<Vec<Item>> {
    use parse::lex::Lexer;
    use parse::{ItemKind, Parser, StmtKind};
    use shared::Located;

    let ast = Parser::new(Lexer::new(source, 0, "page".into())).try_into_ast().ok()?;
    let mut items = Vec::new();
    for stmt in ast.stmts() {
        let StmtKind::Item(item) = stmt.kind() else {
            continue;
        };
        let ItemKind::Function(f) = item.kind() else {
            continue;
        };
        let span = item.location().span;
        let (start, end) = (span.start.min(source.len()), span.end.min(source.len()));
        if start >= end || !source.is_char_boundary(start) || !source.is_char_boundary(end) {
            continue;
        }
        items.push(Item {
            name: f.name.lexeme.clone(),
            start,
            end,
            source: source[start..end].to_string(),
        });
    }
    Some(items)
}

/// Brace-matcher fallback for source that doesn't parse.
fn extract_braces(source: &str) -> Vec<Item> {
    let mut items = Vec::new();
    let mut depth: i32 = 0;
    let mut i = 0;
    let bytes = source.as_bytes();
    while i < bytes.len() {
        if skip_comment(source, &mut i) {
            continue;
        }
        if skip_string(source, &mut i) {
            continue;
        }
        let ch = bytes[i] as char;
        if ch == '{' {
            depth += 1;
            i += 1;
            continue;
        }
        if ch == '}' {
            depth = (depth - 1).max(0);
            i += 1;
            continue;
        }
        if depth == 0 && skip_where_block(source, &mut i) {
            continue;
        }
        if depth == 0 && is_fn_start(source, i) {
            if let Some(item) = parse_fn(source, i) {
                i = item.end;
                items.push(item);
                continue;
            }
        }
        i += 1;
    }
    items
}

fn skip_where_block(source: &str, i: &mut usize) -> bool {
    let rest = &source[*i..];
    let trimmed = rest.trim_start();
    if !(trimmed.starts_with("where:") || trimmed.starts_with("where {") || trimmed == "where") {
        return false;
    }
    let indent_skipped = rest.len() - trimmed.len();
    if indent_skipped > 0 {
        // `where:` only starts a block at column 0 of a line-ish position
        // (whitespace after newline is fine; mid-identifier is not).
        let before = &source[..*i];
        if !before.ends_with('\n') && !before.is_empty() {
            return false;
        }
    }
    *i += indent_skipped;
    if let Some(nl) = source[*i..].find('\n') {
        *i += nl + 1;
    } else {
        *i = source.len();
        return true;
    }
    while *i < source.len() {
        let line = source[*i..].lines().next().unwrap_or("");
        let trimmed = line.trim();
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if trimmed.is_empty() || indented || trimmed == "}" || trimmed.starts_with("//") {
            *i += line.len();
            if source.as_bytes().get(*i) == Some(&b'\n') {
                *i += 1;
            }
            continue;
        }
        break;
    }
    true
}

fn is_fn_start(source: &str, i: usize) -> bool {
    let rest = &source[i..];
    if !rest.starts_with("fn") {
        return false;
    }
    let before_ok = i == 0
        || source[..i]
            .chars()
            .next_back()
            .map_or(true, |c| c.is_whitespace());
    let after = rest.get(2..).and_then(|s| s.chars().next());
    before_ok && after.is_some_and(|c| c.is_whitespace() || c == '_')
}

fn parse_fn(source: &str, start: usize) -> Option<Item> {
    let after_fn = start + 2;
    let name_start = skip_ws(source, after_fn);
    let name_end = scan_ident(source, name_start)?;
    let name = source[name_start..name_end].to_string();
    if name.is_empty() {
        return None;
    }
    let brace = source[name_end..].find('{')?;
    let body_open = name_end + brace;
    let end = match_brace(source, body_open)? + 1;
    Some(Item {
        name,
        start,
        end,
        source: source[start..end].to_string(),
    })
}

fn skip_ws(source: &str, mut i: usize) -> usize {
    while i < source.len() {
        let ch = source[i..].chars().next().unwrap();
        if !ch.is_whitespace() {
            break;
        }
        i += ch.len_utf8();
    }
    i
}

fn scan_ident(source: &str, start: usize) -> Option<usize> {
    let mut chars = source[start..].char_indices();
    let (_, first) = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    let mut end = start + first.len_utf8();
    for (off, ch) in chars {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            end = start + off + ch.len_utf8();
        } else {
            break;
        }
    }
    Some(end)
}

fn match_brace(source: &str, open: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    if bytes.get(open).copied() != Some(b'{') {
        return None;
    }
    let mut depth = 0;
    let mut i = open;
    while i < bytes.len() {
        if skip_comment(source, &mut i) {
            continue;
        }
        if skip_string(source, &mut i) {
            continue;
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn skip_comment(source: &str, i: &mut usize) -> bool {
    let rest = &source[*i..];
    if rest.starts_with("//") {
        if let Some(nl) = rest.find('\n') {
            *i += nl + 1;
        } else {
            *i = source.len();
        }
        return true;
    }
    if rest.starts_with("/*") {
        if let Some(end) = rest[2..].find("*/") {
            *i += 2 + end + 2;
        } else {
            *i = source.len();
        }
        return true;
    }
    false
}

fn skip_string(source: &str, i: &mut usize) -> bool {
    let rest = &source[*i..];
    let quote = match rest.as_bytes().first() {
        Some(b'"') | Some(b'\'') => rest.as_bytes()[0],
        _ => return false,
    };
    let bytes = rest.as_bytes();
    let mut j = 1;
    while j < bytes.len() {
        if bytes[j] == b'\\' {
            j += 2;
            continue;
        }
        if bytes[j] == quote {
            *i += j + 1;
            return true;
        }
        j += 1;
    }
    *i = source.len();
    true
}

/// Names that mention `callee` as a call, excluding `callee` itself.
pub fn callers<'a>(items: &'a [Item], callee: &str) -> Vec<&'a str> {
    items
        .iter()
        .filter(|item| item.name != callee)
        .filter(|item| item.mentions(callee))
        .map(|item| item.name.as_str())
        .collect()
}

pub fn caller_count(items: &[Item], callee: &str) -> usize {
    callers(items, callee).len()
}

pub fn callers_counts_overlay(source: &str, callee: &str) -> String {
    format!(
        "callee    {}\ncallers   {}",
        callee,
        caller_count(&extract(source), callee)
    )
}

/// Inspect text for callers of `callee` on this page.
pub fn callers_overlay(source: &str, callee: &str) -> String {
    let items = extract(source);
    callers_overlay_items(&items, source, callee)
}

/// Callers of every top-level function on the page.
pub fn callers_page_overlay(source: &str) -> String {
    let items = extract(source);
    if items.is_empty() {
        return "no fn items".into();
    }
    items
        .iter()
        .map(|item| {
            let found = callers(&items, &item.name);
            if found.is_empty() {
                format!("{}  no callers", item.name)
            } else {
                format!("{}  <-  {}", item.name, found.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn callers_overlay_items(items: &[Item], source: &str, callee: &str) -> String {
    let found = callers(items, callee);
    if found.is_empty() {
        return format!("{callee}  no callers on this page");
    }
    let mut lines = vec![format!("{callee}  <-  {} caller(s)", found.len())];
    for c in &found {
        if let Some(item) = named(items, c) {
            lines.push(format!("  {}", item.overlay_line(source)));
        } else {
            lines.push(format!("  {c}"));
        }
    }
    lines.join("\n")
}

/// Every identifier referenced in a source slice, via the real AST — calls,
/// value positions, member names. String/comment false positives are
/// impossible; the parse path misses nothing the compiler would see.
/// `None` when the slice doesn't parse.
fn references(source: &str) -> Option<std::collections::HashSet<String>> {
    use parse::{Parser, lex::Lexer};
    let ast = Parser::new(Lexer::new(source, 0, "refs".into())).try_into_ast().ok()?;
    let mut out = std::collections::HashSet::new();
    for stmt in ast.stmts() {
        stmt_refs(stmt, &mut out);
    }
    Some(out)
}

type RefSet = std::collections::HashSet<String>;

fn stmt_refs(stmt: &parse::Stmt, out: &mut RefSet) {
    use parse::StmtKind;
    match stmt.kind() {
        StmtKind::Let(l) => {
            pat_refs(&l.left, out);
            expr_refs(&l.right, out);
            if let Some(e) = &l.else_branch {
                expr_refs(e, out);
            }
        }
        StmtKind::Assignment(a) => {
            expr_refs(&a.left, out);
            expr_refs(&a.right, out);
        }
        StmtKind::Expr(e) => expr_refs(e, out),
        StmtKind::Item(item) => {
            // Nested items contribute their bodies too.
            if let parse::ItemKind::Function(f) = item.kind() {
                expr_refs(&f.body, out);
            }
        }
        StmtKind::Module(_) => {}
    }
}

fn pat_refs(pat: &parse::components::Pat, out: &mut RefSet) {
    use parse::components::PatKind;
    match pat.kind() {
        PatKind::Ident(ident) => {
            out.insert(ident.lexeme.clone());
        }
        PatKind::Tuple(pats) | PatKind::Or(pats) => {
            for p in pats {
                pat_refs(p, out);
            }
        }
        PatKind::Struct(path, fields) => {
            expr_refs(path, out);
            for p in fields.values() {
                pat_refs(p, out);
            }
        }
        PatKind::TupleVariant(path, pats) => {
            expr_refs(path, out);
            for p in pats {
                pat_refs(p, out);
            }
        }
        PatKind::Variant(path) => expr_refs(path, out),
        PatKind::NullBind(p) => pat_refs(p, out),
        PatKind::Literal(lit) => literal_refs(lit, out),
        PatKind::Poison(_) => {}
    }
}

fn literal_refs(lit: &parse::Literal, out: &mut RefSet) {
    use parse::Literal;
    match lit {
        Literal::Array(exprs) | Literal::Tuple(exprs) => {
            for e in exprs {
                expr_refs(e, out);
            }
        }
        Literal::Dictionary(fields) => {
            for (_, e) in fields {
                expr_refs(e, out);
            }
        }
        Literal::Struct(s) => {
            expr_refs(&s.name, out);
            for (_, e) in &s.fields {
                expr_refs(e, out);
            }
        }
        _ => {}
    }
}

fn expr_refs(expr: &parse::Expr, out: &mut RefSet) {
    use parse::{Access, ExprKind, FStringPart};
    match expr.kind() {
        ExprKind::Ident(ident) => {
            out.insert(ident.lexeme.clone());
        }
        ExprKind::Call(c) => {
            expr_refs(&c.left, out);
            for arg in &c.arguments {
                expr_refs(&arg.value, out);
            }
        }
        ExprKind::Access(access) => match access {
            Access::Identity { right } => {
                out.insert(right.lexeme.clone());
            }
            Access::Dot { left, right, .. } => {
                expr_refs(left, out);
                expr_refs(right, out);
            }
            Access::DoubleColon { left, right } => {
                expr_refs(left, out);
                out.insert(right.lexeme.clone());
            }
            Access::Square { left, key, .. } => {
                expr_refs(left, out);
                expr_refs(key, out);
            }
        },
        ExprKind::Block(b) => {
            for s in &b.body {
                stmt_refs(s, out);
            }
            if let Some(e) = &b.yielded_expr {
                expr_refs(e, out);
            }
        }
        ExprKind::Break(b) => {
            if let Some(e) = &b.value {
                expr_refs(e, out);
            }
        }
        ExprKind::Closure(c) => {
            for b in &c.parameters {
                pat_refs(&b.left, out);
                if let Some(e) = &b.right {
                    expr_refs(e, out);
                }
            }
            expr_refs(&c.body, out);
        }
        ExprKind::Collect(c) => expr_refs(&c.value, out),
        ExprKind::Continue(_) => {}
        ExprKind::Coalescence(c) => {
            expr_refs(&c.left, out);
            expr_refs(&c.right, out);
        }
        ExprKind::Equality(e) => {
            expr_refs(&e.left, out);
            expr_refs(&e.right, out);
        }
        ExprKind::Evaluation(e) => {
            expr_refs(&e.left, out);
            expr_refs(&e.right, out);
        }
        ExprKind::Logical(e) => {
            expr_refs(&e.left, out);
            expr_refs(&e.right, out);
        }
        ExprKind::For(f) => {
            pat_refs(&f.binding, out);
            expr_refs(&f.iterator, out);
            expr_refs(&f.body, out);
        }
        ExprKind::FString(f) => {
            for part in &f.parts {
                if let FStringPart::Expr(e) = part {
                    expr_refs(e, out);
                }
            }
        }
        ExprKind::Grouping(g) => expr_refs(&g.inner, out),
        ExprKind::If(i) => {
            expr_refs(&i.condition, out);
            expr_refs(&i.main_body, out);
            if let Some(e) = &i.else_expr {
                expr_refs(e, out);
            }
            if let Some(p) = &i.binding {
                pat_refs(p, out);
            }
        }
        ExprKind::In(i) => {
            expr_refs(&i.left, out);
            expr_refs(&i.right, out);
        }
        ExprKind::Literal(lit) => literal_refs(lit, out),
        ExprKind::Loop(l) => expr_refs(&l.body, out),
        ExprKind::Match(m) => {
            expr_refs(&m.identity, out);
            for case in &m.cases {
                pat_refs(case.pat(), out);
                if let Some(g) = case.guard() {
                    expr_refs(g, out);
                }
                expr_refs(case.body(), out);
            }
        }
        ExprKind::Raise(r) => expr_refs(&r.value, out),
        ExprKind::Range(r) => {
            expr_refs(&r.start, out);
            expr_refs(&r.end, out);
        }
        ExprKind::Return(r) => {
            if let Some(e) = &r.value {
                expr_refs(e, out);
            }
        }
        ExprKind::Unary(u) => expr_refs(&u.right, out),
        ExprKind::Unwrap(u) => expr_refs(&u.expr, out),
        ExprKind::While(w) => {
            if let Some(p) = &w.binding {
                pat_refs(p, out);
            }
            expr_refs(&w.header, out);
            expr_refs(&w.body, out);
        }
        ExprKind::Absolve(a) => {
            expr_refs(&a.left, out);
            expr_refs(&a.handler, out);
        }
        ExprKind::Poison(_) => {}
    }
}

fn mentions(source: &str, name: &str) -> bool {
    if let Some(refs) = references(source) {
        return refs.contains(name);
    }
    // Unparseable mid-edit source: substring scan as before.
    let needle = format!("{name}(");
    source.contains(&needle)
}

/// The canonical form of a Mimas source slice: its token stream, trivia
/// dropped, tokens joined by single spaces. Whitespace and comments don't
/// move the content hash; every meaningful token does. `None` when the slice
/// doesn't lex cleanly (e.g. an unterminated string mid-edit).
///
/// Rename-is-free: the item's own name — the ident right after the first
/// `fn` — erases to `_`, so `fn square` and `fn sq` with the same body share
/// a hash and a blob. A *self-reference* inside the body keeps the name, so
/// renaming a recursive fn still changes its hash (a known limit until
/// scope-aware renaming lands; see the content-addressing roadmap).
pub fn canonical(source: &str) -> Option<String> {
    let mut out = String::new();
    let mut lexer = parse::lex::Lexer::new(source, 0, "canonical".into());
    let mut name_erased = false;
    let mut after_fn = false;
    for tok in &mut lexer {
        if matches!(tok.kind, parse::lex::TokKind::Invalid(_)) {
            return None;
        }
        if tok.kind.is_comment() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        // `Float(1.0)` Displays as `1`, colliding with `Int(1)` — keep the
        // point so an int and a float never share a canonical token.
        match tok.kind {
            parse::lex::TokKind::Float(v) => {
                let s = v.to_string();
                if s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E'))
                    || s.contains("inf")
                    || s.contains("NaN")
                {
                    out.push_str(&s);
                } else {
                    out.push_str(&s);
                    out.push_str(".0");
                }
            }
            // The first ident after the first `fn` is the item's own name.
            kind if !name_erased && after_fn && matches!(kind, parse::lex::TokKind::Ident(_)) => {
                out.push('_');
                name_erased = true;
            }
            kind => out.push_str(&kind.to_string()),
        }
        after_fn = matches!(tok.kind, parse::lex::TokKind::Fn);
    }
    // an unterminated string/comment lexes as a token but records an error
    if !lexer.take_errors().is_empty() {
        return None;
    }
    Some(out)
}

/// Content hash of an item: blake3 of the canonical token stream, falling
/// back to the raw bytes when the source doesn't lex.
pub fn hash_item(source: &str) -> String {
    let canonical = canonical(source).unwrap_or_else(|| source.to_string());
    blake3::hash(canonical.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"fn square(n: int) -> int {
    n * n
}

fn twice(n: int) -> int {
    square(n) + square(n)
}

// example: square(4) is 16
"#;

    #[test]
    fn extracts_two_functions() {
        let items = extract(SRC);
        assert_eq!(names(&items), vec!["square", "twice"]);
        assert_eq!(fn_count(SRC), 2);
        assert_eq!(fn_counts_overlay(SRC), "fns       2");
        assert!(hashes_overlay(SRC).contains("square  "));
        assert!(items[0].hash_line().starts_with("square  "));
        assert!(items[0].source.contains("n * n"));
        assert!(items[1].source.contains("square(n)"));
        assert_eq!(items[0].line(SRC), 1);
        assert!(items[0].overlay_line(SRC).contains("L1"));
        assert_eq!(items[1].line(SRC), 5);
        assert_eq!(items[0].line_span(SRC), (1, 3));
        assert_eq!(items[0].n_lines(), 3);
        assert!(
            items[0].counts_overlay(SRC).contains("lines     3"),
            "{}",
            items[0].counts_overlay(SRC)
        );
        assert!(items[0].contains_line(SRC, 2));
        assert!(!items[0].contains_line(SRC, 5));
        assert_eq!(items[1].line_span(SRC), (5, 7));
    }

    #[test]
    fn nested_braces_and_strings() {
        let src = r#"fn talk() {
    let s = "fn decoy() { }";
    if true {
        print(s);
    }
}
"#;
        let items = extract(src);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "talk");
        assert!(items[0].source.contains("decoy"));
    }

    #[test]
    fn twice_calls_square() {
        let items = extract(SRC);
        assert_eq!(callers(&items, "square"), vec!["twice"]);
        assert_eq!(caller_count(&items, "square"), 1);
        assert!(
            callers_counts_overlay(SRC, "square").contains("callers   1"),
            "{}",
            callers_counts_overlay(SRC, "square")
        );
        assert_eq!(caller_count(&items, "twice"), 0);
        assert!(named(&items, "twice").unwrap().mentions("square"));
        assert!(!named(&items, "square").unwrap().mentions("twice"));
        assert!(callers(&items, "twice").is_empty());
        let overlay = callers_overlay(SRC, "square");
        assert!(overlay.contains("twice"), "{overlay}");
        assert!(overlay.contains("L5"), "{overlay}");
        assert!(callers_overlay(SRC, "twice").contains("no callers"));
        let page = callers_page_overlay(SRC);
        assert!(page.contains("square  <-  twice"), "{page}");
        assert!(page.contains("twice  no callers"), "{page}");
    }

    #[test]
    fn callers_use_the_ast_not_substrings() {
        // `square(` in a string or comment must not count as a call; a bare
        // reference (stored/passed value) does.
        let src = r#"fn square(n: int) -> int { n * n }
fn mentions_it() -> string { "call square(n) now" }
// fn in_comment() { square(2) }
fn holds_it() -> _ { square }
fn calls_it() -> int { square(3) }
"#;
        let items = extract(src);
        let found = callers(&items, "square");
        assert!(found.contains(&"calls_it"), "{found:?}");
        assert!(found.contains(&"holds_it"), "{found:?}");
        assert!(!found.contains(&"mentions_it"), "{found:?}");
    }

    #[test]
    fn hash_changes_when_body_changes() {
        let a = extract("fn x() { 1 }\n")[0].source.clone();
        let b = extract("fn x() { 2 }\n")[0].source.clone();
        assert_ne!(hash_item(&a), hash_item(&b));
        assert_eq!(hash_item(&a), hash_item(&a));
        assert_eq!(extract("fn x() { 1 }\n")[0].hash(), hash_item(&a));
    }

    #[test]
    fn rename_is_free_for_the_items_own_name() {
        // 2a: the header name erases to `_` in the canonical form, so the
        // same body hashes identically under any name — and the stored blob
        // is the name-erased text, with `rename_fn` splicing a name back in.
        let a = extract("fn square(n: int) -> int { n * n }\n");
        let b = extract("fn sq(n: int) -> int { n * n }\n");
        assert_eq!(a[0].hash(), b[0].hash());
        let canon = a[0].canonical_source();
        assert!(canon.starts_with("fn _"), "{canon}");
        assert_eq!(
            rename_fn(&canon, "_", "square"),
            "fn square ( n : int ) -> int { n * n }"
        );
        // `pub`/attribute prefixes don't change the property either.
        let c = extract("pub fn square(n: int) -> int { n * n }\n");
        let d = extract("pub fn sq(n: int) -> int { n * n }\n");
        assert_eq!(c[0].hash(), d[0].hash());
        let e = extract("#[test]\nfn square() { 1 }\n");
        assert!(e[0].canonical_source().contains("fn _"), "{canon}");
    }

    #[test]
    fn recursive_self_reference_still_hashes_on_the_name() {
        // KNOWN LIMIT (2a): only the header name erases. A recursive call in
        // the body keeps its callee token, so renaming a self-recursive fn
        // still moves the hash — scope-aware renaming (roadmap 2b/3) is the
        // fix. Meanwhile renames of fns that only mention *other* names are
        // already free.
        let f = "fn f(n: int) -> int { if n <= 0 { 0 } else { f(n - 1) } }\n";
        let g = "fn g(n: int) -> int { if n <= 0 { 0 } else { g(n - 1) } }\n";
        assert_ne!(hash_item(f), hash_item(g));
        let a = "fn a(n: int) -> int { helper(n) }\n";
        let b = "fn b(n: int) -> int { helper(n) }\n";
        assert_eq!(hash_item(a), hash_item(b));
    }

    #[test]
    fn named_and_at_byte_find_the_item() {
        let items = extract(SRC);
        assert_eq!(
            named(&items, "twice").map(|i| i.name.as_str()),
            Some("twice")
        );
        assert!(named(&items, "nope").is_none());
        let twice = named(&items, "twice").unwrap();
        assert!(twice.byte_len() > 0);
        assert_eq!(twice.byte_len(), twice.end - twice.start);
        let (lo, hi) = twice.char_span(SRC);
        assert!(hi > lo);
        assert!(SRC
            .chars()
            .skip(lo)
            .take(hi - lo)
            .collect::<String>()
            .contains("fn twice"));
        assert_eq!(
            at_byte(&items, twice.start).map(|i| i.name.as_str()),
            Some("twice")
        );
        assert!(at_byte(&items, twice.end).is_none());
    }

    #[test]
    fn at_line_and_unique_name() {
        let items = extract(SRC);
        assert_eq!(
            at_line(&items, SRC, 2).map(|i| i.name.as_str()),
            Some("square")
        );
        assert_eq!(
            at_line(&items, SRC, 6).map(|i| i.name.as_str()),
            Some("twice")
        );
        assert!(at_line(&items, SRC, 9).is_none());
        assert_eq!(unique_name(&items, "cube"), "cube");
        assert_eq!(unique_name(&items, "square"), "square_2");
        assert_eq!(unique_copy_name(&items, "square"), "square_copy");
        assert_eq!(unique_copy_name(&items, "twice"), "twice_copy");
        assert!(rename_fn(&items[0].source, "square", "cube").starts_with("fn cube"));
        let replaced = apply_item(
            SRC,
            &Item::named("square", "fn square(n: int) -> int { n }\n"),
        );
        assert!(
            replaced.contains("fn square(n: int) -> int { n }"),
            "{replaced}"
        );
        assert!(replaced.contains("fn twice"), "{replaced}");
        let added = apply_item(
            SRC,
            &Item::named("cube", "fn cube(n: int) -> int { n * n * n }\n"),
        );
        assert!(added.contains("fn cube"), "{added}");
        let (lo, del, ins) = apply_item_span(
            SRC,
            &Item::named("square", "fn square(n: int) -> int { n }\n"),
        );
        assert_eq!(lo, 0);
        assert!(del > 0);
        assert!(ins.contains("fn square(n: int) -> int { n }"));
    }

    #[test]
    fn where_block_is_not_extracted_as_code() {
        let src = "fn square(n: int) -> int { n * n }\nwhere:\n    square(0) is 0\n    square(2) is 4\n\n// fn decoy() { 1 }\n";
        let items = extract(src);
        assert_eq!(names(&items), vec!["square"]);
    }

    #[test]
    fn parser_extraction_covers_attrs_and_pub() {
        // The brace matcher missed `#[test]`/`pub` prefixes; the parser's
        // item span covers them.
        let src = "#[test]\nfn checked() { 1 }\npub fn open() { 2 }\n";
        let items = extract(src);
        assert_eq!(names(&items), vec!["checked", "open"]);
        assert!(items[0].source.starts_with("#[test]"), "{}", items[0].source);
        assert!(items[1].source.starts_with("pub fn"), "{}", items[1].source);
    }


    #[test]
    fn dbg_canon_probe() {
        let a = "fn square(n: int) -> int {\n    n * n // comment\n}\n";
        let b = "fn square(n:int)->int{n*n}\n";
        let c = "fn square(n: int) -> int { n * n * 1 }\n";
        eprintln!("A: {:?}", canonical(a));
        eprintln!("B: {:?}", canonical(b));
        eprintln!("C: {:?}", canonical(c));
        eprintln!("I: {:?}", canonical("fn f() { 1 }\n"));
        eprintln!("F: {:?}", canonical("fn f() { 1.0 }\n"));
        eprintln!("K: {:?}", canonical("fn x() { \"unterminated"));
    }
    #[test]
    fn canonical_hash_ignores_formatting_and_comments() {
        let a = "fn square(n: int) -> int {\n    n * n // comment\n}\n";
        let b = "fn square(n:int)->int{n*n}\n";
        assert_eq!(hash_item(a), hash_item(b));
        let c = "fn square(n: int) -> int { n * n * 1 }\n";
        assert_ne!(hash_item(a), hash_item(c));
        // Unlexable source falls back to hashing raw bytes.
        let broken = "fn x() { \"unterminated";
        assert!(canonical(broken).is_none());
        assert_eq!(
            hash_item(broken),
            blake3::hash(broken.as_bytes()).to_hex().to_string()
        );
        // `1` and `1.0` are different types — the canonical form keeps the
        // float's point so they can't share a hash.
        assert_ne!(
            hash_item("fn f() { 1 }\n"),
            hash_item("fn f() { 1.0 }\n")
        );
        // … while `1.0` and `1.00` are the same float.
        assert_eq!(
            hash_item("fn f() { 1.0 }\n"),
            hash_item("fn f() { 1.00 }\n")
        );
    }
}
