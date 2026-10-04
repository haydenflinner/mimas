//! Expression lowering.
//!
//! `expr` returns mimas *source text* — never lines. Constructs that need
//! statements (postfix `++`, out-param calls, `std::swap`) fold into inline
//! block expressions `{ ..; v }`, which mimas supports everywhere an
//! expression is expected.

use tree_sitter::Node;

use super::cty::{spec_type, unqual};
use super::stmt::unwrap_parens;
use super::{FnSig, Lower};
use crate::ty::{CTy, SeqKind};
use crate::DiagKind;

/// mimas keywords — a C++ identifier colliding with one gets `_` appended.
const RESERVED: &[&str] = &[
    "if", "else", "for", "while", "loop", "match", "break", "continue", "return", "raise",
    "absolve", "collect", "in", "let", "const", "fn", "struct", "enum", "impl", "pact", "pub",
    "module", "use", "self", "true", "false", "null", "and", "or", "not", "is", "as", "type",
    "trait", "where", "yield",
];

fn sanitize(name: &str) -> String {
    if RESERVED.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

impl<'a> Lower<'a> {
    /// Is this node kind an expression at all?
    pub(crate) fn is_expr(&self, node: Node<'a>) -> bool {
        matches!(
            node.kind(),
            "identifier" | "this" | "number_literal" | "string_literal" | "char_literal"
                | "concatenated_string_literal" | "true" | "false" | "null" | "nullptr"
                | "user_defined_literal" | "binary_expression" | "unary_expression"
                | "update_expression" | "assignment_expression" | "conditional_expression"
                | "call_expression" | "field_expression" | "subscript_expression"
                | "parenthesized_expression" | "comma_expression" | "cast_expression"
                | "new_expression" | "delete_expression" | "sizeof_expression"
                | "lambda_expression" | "initializer_list" | "qualified_identifier"
                | "template_function" | "template_method" | "throw_expression"
                | "co_await_expression" | "fold_expression"
        )
    }

    /// Strip parens, then lower.
    pub(crate) fn expr_unparen(&mut self, node: Node<'a>) -> String {
        self.expr(unwrap_parens(self, node))
    }

    /// Lower an expression to mimas text.
    pub(crate) fn expr(&mut self, node: Node<'a>) -> String {
        match node.kind() {
            "identifier" => self.ident(node),
            "this" => self.self_name.clone(),
            "true" | "false" => self.node_text(node).to_string(),
            "null" | "nullptr" => "null".into(),
            "number_literal" => self.number(node),
            "string_literal" | "concatenated_string_literal" => self.string_lit(node),
            "char_literal" => self.char_lit(node),
            "user_defined_literal" => {
                // `42s`, `1.5f` — drop the suffix
                let inner = self.first(node).map(|c| self.expr(c));
                inner.unwrap_or_else(|| self.node_text(node).to_string())
            }
            "parenthesized_expression" => {
                format!("({})", self.first(node).map(|e| self.expr(e)).unwrap_or_default())
            }
            "binary_expression" => self.binary(node),
            "unary_expression" | "pointer_expression" => self.unary(node),
            "update_expression" => self.update(node),
            "assignment_expression" => self.assign(node),
            "conditional_expression" => {
                let c = self.f(node, "condition").map(|c| self.expr_unparen(c)).unwrap_or_default();
                let t = self.f(node, "consequence").map(|c| self.expr(c)).unwrap_or_default();
                let f = self.f(node, "alternative").map(|c| self.expr(c)).unwrap_or_default();
                format!("(if {c} {{ {t} }} else {{ {f} }})")
            }
            "call_expression" => self.call(node),
            "field_expression" => self.field(node),
            "subscript_expression" => self.subscript(node),
            "comma_expression" => {
                let parts = self.comma_exprs(node);
                let last = parts.len() - 1;
                let body = parts
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let e = self.expr(*p);
                        if i == last { e } else { format!("{e};") }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{{ {body} }}")
            }
            "cast_expression" => self.cast(node),
            "new_expression" => self.new_expr(node),
            "delete_expression" => {
                self.diag(DiagKind::Note, node, "`delete` dropped — mimas is GC'd");
                "()".into()
            }
            "sizeof_expression" => {
                let inner = self
                    .f(node, "value")
                    .or_else(|| self.f(node, "type"))
                    .or_else(|| self.first(node));
                match inner.map(|i| (i, self.expr_ty(i))) {
                    Some((i, CTy::Arr(..) | CTy::Str | CTy::Dict(_) | CTy::Set(_))) => {
                        format!("{}.len()", self.expr(i))
                    }
                    _ => {
                        self.diag(DiagKind::Approximate, node, "sizeof -> 8");
                        "8".into()
                    }
                }
            }
            "lambda_expression" => self.lambda(node),
            "initializer_list" => {
                let items = self
                    .children(node)
                    .into_iter()
                    .filter(|c| c.kind() != "comment" && c.kind() != "initializer_pair")
                    .collect::<Vec<_>>();
                format!(
                    "[{}]",
                    items.iter().map(|i| self.expr(*i)).collect::<Vec<_>>().join(", ")
                )
            }
            "qualified_identifier" => self.qualified(node),
            "template_function" | "template_method" => {
                let inner = self
                    .f(node, "name")
                    .or_else(|| self.f(node, "function"))
                    .or_else(|| self.first(node));
                inner.map(|i| self.expr(i)).unwrap_or_default()
            }
            "namespace_identifier" => unqual(self.node_text(node)).to_string(),
            "throw_expression" => {
                let e = self.first(node).map(|e| self.expr(e)).unwrap_or_default();
                format!("panic({e})")
            }
            _ => {
                self.diag(
                    DiagKind::Unsupported,
                    node,
                    format!("expression `{}`", node.kind()),
                );
                format!("panic(\"cpp: unsupported {}\")", node.kind())
            }
        }
    }

    /// `a` — with member access rewriting inside methods.
    fn ident(&mut self, node: Node<'a>) -> String {
        let name = self.node_text(node);
        match name {
            "INT_MAX" => return "2147483647".into(),
            "INT_MIN" => return "-2147483648".into(),
            "LLONG_MAX" | "LONG_LONG_MAX" => return "9223372036854775807".into(),
            "LLONG_MIN" | "LONG_LONG_MIN" => return "-9223372036854775808".into(),
            "UINT_MAX" | "UINT32_MAX" => return "4294967295".into(),
            "npos" => return "-1".into(),
            _ => {}
        }
        // `x` inside a method where `x` is a member and not a local -> `self.x`
        if let Some(owner) = self.cur_struct.clone()
            && self.lookup(name).is_none()
                && let Some(info) = self.structs.get(&owner)
                    && info.fields.iter().any(|(f, _)| f == name) {
                        return format!("{}.{}", self.self_name, sanitize(name));
                    }
        sanitize(name)
    }

    /// `std::x` / `ns::x` — the namespace is flattened; `std` silently.
    fn qualified(&mut self, node: Node<'a>) -> String {
        let scope = self.f(node, "scope").map(|s| self.node_text(s)).unwrap_or("");
        let name = self
            .f(node, "name")
            .or_else(|| self.f(node, "value"))
            .map(|n| self.node_text(n))
            .unwrap_or_else(|| self.node_text(node));
        let name = unqual(name);
        if name == "npos" || name == "endl" {
            return if name == "npos" { "-1".into() } else { "\"\\n\"".into() };
        }
        if let Some(e) = self.enum_const(node, scope, name) {
            return e;
        }
        if scope != "std" && !scope.is_empty() {
            self.diag(
                DiagKind::Note,
                node,
                format!("`{scope}::` flattened to `{name}`"),
            );
        }
        // `Struct::CONST` / `Struct::method` keep the type path in mimas
        if self.structs.contains_key(scope) {
            return format!("{scope}::{}", sanitize(name));
        }
        sanitize(name)
    }

    /// `E::V` for a `enum class` we emitted as a mimas enum stays `E::V`.
    /// Returns Some when the qualifier is a known enum — enums recorded via
    /// `enum E { .. }` live in `structs`? no — separate. Keep it simple:
    /// scoped enum access already reads `E::V`, which is valid mimas.
    fn enum_const(&mut self, _node: Node<'a>, _scope: &str, _name: &str) -> Option<String> {
        None
    }

    fn number(&mut self, node: Node<'a>) -> String {
        let t = self.node_text(node);
        let t = t.trim_end_matches(['u', 'U', 'l', 'L', 'f', 'F', 'z', 'Z']);
        let t = t.replace('\'', "");
        if t.starts_with("0x") || t.starts_with("0X") {
            return u128::from_str_radix(&t[2..], 16)
                .map(|v| v.to_string())
                .unwrap_or(t);
        }
        if t.starts_with("0b") || t.starts_with("0B") {
            return u128::from_str_radix(&t[2..], 2)
                .map(|v| v.to_string())
                .unwrap_or(t);
        }
        // leading-zero octal: `012` -> 10 (careful: `0` itself and `0.5`)
        if t.len() > 1 && t.starts_with('0') && !t.contains('.') && !t.contains('e') {
            return u128::from_str_radix(&t[1..], 8)
                .map(|v| v.to_string())
                .unwrap_or(t);
        }
        t
    }

    fn string_lit(&mut self, node: Node<'a>) -> String {
        let mut t = self.node_text(node).to_string();
        if t.starts_with("R\"") || t.starts_with("u8\"") || t.starts_with("u\"") || t.starts_with("U\"") || t.starts_with("L\"") {
            // raw/unicode prefix -> keep the quoted body
            if let Some(start) = t.find('"') {
                if t.starts_with("R\"") {
                    // R"delim(body)delim"
                    let inner_start = t.find('(').map(|i| i + 1).unwrap_or(start + 1);
                    let inner_end = t.rfind(')').unwrap_or(t.len() - 1);
                    let body = &t[inner_start..inner_end];
                    return format!("\"{}\"", body.replace('\\', "\\\\").replace('"', "\\\""));
                }
                t = t[start..].to_string();
            }
        }
        // adjacent `"a" "b"` concat arrives as concatenated_string_literal
        if node.kind() == "concatenated_string_literal" {
            let parts: Vec<String> = node
                .named_children(&mut node.walk())
                .map(|c| self.string_lit(c))
                .collect();
            // merge the string bodies
            let merged = parts
                .iter()
                .map(|p| p.trim_matches('"').to_string())
                .collect::<Vec<_>>()
                .join("");
            return format!("\"{merged}\"");
        }
        t
    }

    /// `'c'` -> `"c"`; in arithmetic position callers emit `.ord()!`.
    fn char_lit(&mut self, node: Node<'a>) -> String {
        let t = self.node_text(node);
        // encoding prefix: u8'c', u'c', U'c', L'c'
        let t = if t.len() > 1 && t.as_bytes()[1] == b'\'' {
            &t[1..]
        } else if t.len() > 2 && t.starts_with("u8'") {
            &t[2..]
        } else {
            t
        };
        let inner = t.trim_matches('\'');
        let body = match inner {
            "\\n" => "\\n",
            "\\t" => "\\t",
            "\\0" => "\\0",
            "\\\\" => "\\\\",
            "\\'" => "'",
            "\\\"" => "\\\"",
            other => other,
        };
        format!("\"{body}\"")
    }

    /// `'c'` where an int is needed -> `("c").ord()!`.
    fn char_ord(&mut self, node: Node<'a>) -> String {
        format!("{}.ord()!", self.char_lit(node))
    }

    /// Operator text of a binary_expression / assignment_expression.
    pub(crate) fn bin_op(&self, node: Node<'a>) -> &str {
        if let Some(op) = self.f(node, "operator") {
            return self.node_text(op);
        }
        // fallback: the anonymous child between left and right
        let left = self.f(node, "left");
        let right = self.f(node, "right");
        for c in node.children(&mut node.walk()) {
            if Some(c) == left || Some(c) == right {
                continue;
            }
            if c.is_named() {
                continue;
            }
            return self.node_text(c);
        }
        "?"
    }

    pub(crate) fn assign_op(&self, node: Node<'a>) -> &str {
        self.bin_op(node)
    }

    fn binary(&mut self, node: Node<'a>) -> String {
        let l = self.f(node, "left").unwrap();
        let r = self.f(node, "right").unwrap();
        let op = self.bin_op(node).to_string();

        // `s.find(x) != npos` / `s.find(x) != s.end()` -> `x in s`
        if op == "!=" || op == "==" {
            if let Some(found) = self.find_vs_end(l, r) {
                return if op == "!=" { found } else { format!("!({found})") };
            }
            if let Some(found) = self.find_vs_end(r, l) {
                return if op == "!=" { found } else { format!("!({found})") };
            }
        }
        // cout/cin chains can nest (`cout << (a << b)` is nonsense anyway) —
        // a top-level `<<` chain is handled at the statement level; nested
        // ones are just shifts.
        let lt = self.expr_ty(l);
        let rt = self.expr_ty(r);
        match op.as_str() {
            "/" if lt.is_int() && rt.is_int() => {
                format!("{} ~/ {}", self.expr(l), self.expr(r))
            }
            "+" | "-" | "*" | "&" | "|" | "^" | "<<" | ">>" | "%" | "/" => {
                // mimas spells modulo `mod` — `%` is its percent marker
                let op = if op == "%" { "mod" } else { op.as_str() };
                let le = self.arith_operand(l, &lt, &rt);
                let re = self.arith_operand(r, &rt, &lt);
                format!("{le} {op} {re}")
            }
            _ => {
                format!("{} {op} {}", self.expr(l), self.expr(r))
            }
        }
    }

    /// In an arithmetic op, a `char` operand needs `.ord()!` to act on ints.
    fn arith_operand(&mut self, node: Node<'a>, self_ty: &CTy, other: &CTy) -> String {
        if matches!(self_ty, CTy::Char) && other.is_int() {
            if node.kind() == "char_literal" {
                return self.char_ord(node);
            }
            let e = self.expr(node);
            return format!("{e}.ord()!");
        }
        self.expr(node)
    }

    /// `a.find(x) != a.end()` / `!= npos` detection — returns `x in a`.
    fn find_vs_end(&mut self, call: Node<'a>, other: Node<'a>) -> Option<String> {
        let call = unwrap_parens(self, call);
        if call.kind() != "call_expression" {
            return None;
        }
        let f = self.f(call, "function")?;
        if f.kind() != "field_expression" {
            return None;
        }
        let method = self.f(f, "field").map(|m| self.node_text(m))?;
        if method != "find" {
            return None;
        }
        let other = unwrap_parens(self, other);
        let is_end = match other.kind() {
            "call_expression" => self
                .f(other, "function")
                .and_then(|f| self.f(f, "field"))
                .is_some_and(|m| self.node_text(m) == "end"),
            "identifier" | "qualified_identifier" => self.node_text(other).contains("npos"),
            _ => false,
        };
        if !is_end {
            return None;
        }
        let recv = self.f(f, "argument").map(|a| self.expr(a))?;
        let arg = self
            .f(call, "arguments")
            .and_then(|a| self.first(a))
            .map(|a| self.expr(a))?;
        Some(format!("({arg} in {recv})"))
    }

    fn unary(&mut self, node: Node<'a>) -> String {
        let op = self
            .f(node, "operator")
            .map(|o| self.node_text(o))
            .unwrap_or_else(|| node.child(0).map(|c| self.node_text(c)).unwrap_or(""));
        let arg = self.f(node, "argument").unwrap();
        match op {
            "!" | "-" | "+" | "~" => format!("{op}{}", self.expr_paren(arg)),
            "*" => {
                self.diag(DiagKind::Approximate, node, "dereference erased — mimas shares refs");
                self.expr(arg)
            }
            "&" => {
                self.diag(DiagKind::Approximate, node, "address-of erased");
                self.expr(arg)
            }
            _ => format!("{}{}", op, self.expr_paren(arg)),
        }
    }

    /// Parenthesize an operand when it's a composite expr.
    fn expr_paren(&mut self, node: Node<'a>) -> String {
        let e = self.expr(node);
        match node.kind() {
            "identifier" | "this" | "number_literal" | "string_literal" | "char_literal"
            | "true" | "false" | "call_expression" | "field_expression"
            | "subscript_expression" | "parenthesized_expression" | "qualified_identifier" => e,
            _ => format!("({e})"),
        }
    }

    /// `++x` / `x++` in expression position — statement position is handled by
    /// `expr_stmt_inner` (`x += 1`).
    fn update(&mut self, node: Node<'a>) -> String {
        let arg = self.f(node, "argument").map(|a| self.expr(a)).unwrap_or_default();
        let op = self.node_text(node);
        let delta = if op.contains("--") { "-= 1" } else { "+= 1" };
        let prefix = op.starts_with("++") || op.starts_with("--");
        if prefix {
            format!("{{ {arg} {delta}; {arg} }}")
        } else {
            let t = self.fresh();
            format!("{{ let {t} = {arg}; {arg} {delta}; {t} }}")
        }
    }

    /// `lhs op rhs`.
    fn assign(&mut self, node: Node<'a>) -> String {
        let lhs_n = self.f(node, "left").unwrap();
        let rhs_n = self.f(node, "right").unwrap();
        let op = self.assign_op(node).to_string();

        // `std::tie(a, b) = e` / `(a, b) = e`
        if lhs_n.kind() == "call_expression"
            && self.call_name(lhs_n).as_deref() == Some("tie")
        {
            let vars: Vec<String> = self
                .f(lhs_n, "arguments")
                .map(|a| self.children(a))
                .unwrap_or_default()
                .iter()
                .map(|a| self.expr(*a))
                .collect();
            let rhs = self.expr(rhs_n);
            let tmps: Vec<String> = vars.iter().map(|_| self.fresh()).collect();
            let mut s = format!("{{ let ({}) = {rhs}; ", tmps.join(", "));
            for (v, t) in vars.iter().zip(tmps.iter()) {
                s.push_str(&format!("{v} = {t}; "));
            }
            s.push_str(&vars.last().cloned().unwrap_or_default());
            s.push_str(" }");
            return s;
        }

        let lhs_ty = self.expr_ty(lhs_n);
        let lhs = self.lvalue(lhs_n);
        let rhs = self.expr(rhs_n);
        match op.as_str() {
            "=" => format!("{lhs} = {rhs}"),
            "/=" if lhs_ty.is_int() => format!("{lhs} ~/= {rhs}"),
            "+=" | "-=" | "*=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" | "/=" => {
                // dict subscript lvalue: `m[k] += 1` -> `m[k] = (m[k] ?? 0) + 1`
                if let CTy::Dict(v) = &lhs_ty
                    && lhs_n.kind() == "subscript_expression" {
                        let base_op = op.trim_end_matches('=');
                        let def = v.default_init(&self.struct_defaults());
                        return format!("{lhs} = ({lhs} ?? {def}) {base_op} {rhs}");
                    }
                format!("{lhs} {op} {rhs}")
            }
            "??=" => format!("{lhs} ??= {rhs}"),
            _ => format!("{lhs} {op} {rhs}"),
        }
    }

    /// LHS target text — same as expr but dict subscripts stay bare `m[k]`
    /// (assignment, not a `??` read).
    fn lvalue(&mut self, node: Node<'a>) -> String {
        match node.kind() {
            "subscript_expression" => {
                let a = self.f(node, "argument").unwrap();
                let i = self.subscript_index(node);
                format!("{}[{}]", self.expr(a), self.expr(i))
            }
            _ => self.expr(node),
        }
    }

    /// The index node of `a[i]` — `indices` is a `subscript_argument_list`
    /// (multi-arg `a[i, j]` takes the first and warns).
    fn subscript_index(&mut self, node: Node<'a>) -> Node<'a> {
        let list = self.f(node, "indices").unwrap();
        let args: Vec<Node<'a>> = self
            .children(list)
            .into_iter()
            .filter(|c| c.kind() != "comment")
            .collect();
        if args.len() > 1 {
            self.diag(DiagKind::Unsupported, node, "multi-dimensional subscript");
        }
        args.first().copied().unwrap_or(node)
    }

    /// `a[i]` — dict reads get `?? default`; everything else is plain indexing.
    fn subscript(&mut self, node: Node<'a>) -> String {
        let a = self.f(node, "argument").unwrap();
        let i = self.subscript_index(node);
        let at = self.expr_ty(a);
        let ie = self.expr(i);
        let ae = self.expr(a);
        if let CTy::Dict(v) = at {
            let def = v.default_init(&self.struct_defaults());
            return format!("({ae}[{ie}] ?? {def})");
        }
        format!("{ae}[{ie}]")
    }

    /// `a.b` / `a->b` non-call member access.
    fn field(&mut self, node: Node<'a>) -> String {
        let arg = self.f(node, "argument").unwrap();
        let field = self.f(node, "field").unwrap();
        let fname = self.node_text(field);
        let aty = self.expr_ty(arg);
        let a = self.expr(arg);
        match (fname, &aty) {
            ("first", CTy::Tuple(_)) => format!("{a}.0"),
            ("second", CTy::Tuple(_)) => format!("{a}.1"),
            ("first" | "second", _) => {
                let idx = if fname == "first" { 0 } else { 1 };
                format!("{a}.{idx}")
            }
            _ => {
                if a == "this" || a == self.self_name {
                    format!("{}.{}", self.self_name, sanitize(fname))
                } else {
                    format!("{}.{}", a, sanitize(fname))
                }
            }
        }
    }

    /// `(T)e` and `T(e)` casts.
    fn cast(&mut self, node: Node<'a>) -> String {
        let ty = spec_type(self, self.f(node, "type"));
        let v = self
            .f(node, "value")
            .or_else(|| self.first(node))
            .map(|v| self.expr(v))
            .unwrap_or_default();
        self.cast_to(&ty, v, node)
    }

    fn cast_to(&mut self, ty: &CTy, v: String, node: Node<'a>) -> String {
        match ty {
            CTy::Int => {
                if v.starts_with('"') {
                    format!("{v}.ord()!")
                } else {
                    format!("{v}.to_int()")
                }
            }
            CTy::Float => format!("{v}.to_float()"),
            CTy::Str | CTy::Char => format!("f\"{{{v}}}\""),
            CTy::Bool => format!("{v} != 0"),
            _ => {
                let _ = node;
                v
            }
        }
    }

    /// `new X(args)` / `new int[n]`.
    fn new_expr(&mut self, node: Node<'a>) -> String {
        let ty = spec_type(self, self.f(node, "type"));
        // `new T[n]` — the declarator is a `new_declarator` with a `length`
        if let Some(len) = self
            .f(node, "declarator")
            .and_then(|d| self.f(d, "length"))
        {
            let len = self.expr(len);
            return format!("array::new_filled({}, {len})", ty.default_init(&self.struct_defaults()));
        }
        let args = self
            .f(node, "arguments")
            .map(|a| {
                self.children(a)
                    .into_iter()
                    .filter(|c| c.kind() != "comment")
                    .map(|c| self.expr(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        match ty {
            CTy::Struct(name) => format!("{name}::new({args})"),
            CTy::Arr(elem, _) if !args.is_empty() => {
                format!("array::new_filled({}, {args})", elem.default_init(&self.struct_defaults()))
            }
            _ => {
                self.diag(DiagKind::Approximate, node, "`new` erased to a value");
                ty.default_init(&self.struct_defaults())
            }
        }
    }

    /// `[cap](params) -> ret { body }` -> `|p: T| { body }`.
    fn lambda(&mut self, node: Node<'a>) -> String {
        let decl = self.f(node, "declarator");
        let params = decl.and_then(|d| self.f(d, "parameters"));
        let mut parts = Vec::new();
        if let Some(pl) = params {
            for p in pl.named_children(&mut pl.walk()) {
                if p.kind() == "parameter_declaration" || p.kind() == "optional_parameter_declaration" {
                    let base = spec_type(self, self.f(p, "type"));
                    let d = super::cty::declarator(self, self.f(p, "declarator").unwrap_or(p));
                    let ty = self.wrap_type(base, &d);
                    let annot = ty.annot().unwrap_or_else(|| "_".into());
                    parts.push(format!("{}: {annot}", d.name));
                }
            }
        }
        let body = self.f(node, "body");
        // a lambda's body is a compound_statement — but mimas closures are
        // `|args| expr-or-block`; emit `{ stmts }` by lowering inline.
        match body {
            Some(b) => {
                let inner = self.block_to_string(b);
                format!("|{}| {inner}", parts.join(", "))
            }
            None => format!("|{}| ()", parts.join(", ")),
        }
    }

    /// Render a compound_statement's contents as a `{ stmts }` expr string.
    fn block_to_string(&mut self, body: Node<'a>) -> String {
        // reuse the statement emitter into a scratch buffer
        let saved = std::mem::take(&mut self.out);
        let saved_indent = self.indent;
        self.indent = 0;
        self.push_scope();
        self.block_stmts(body);
        self.pop_scope();
        self.indent = saved_indent;
        let inner = std::mem::replace(&mut self.out, saved);
        let text = inner.trim().replace('\n', " ");
        format!("{{ {text} }}")
    }

    /// comma_expression children, flattened.
    fn comma_exprs<'n>(&mut self, node: Node<'n>) -> Vec<Node<'n>> {
        if node.kind() == "comma_expression" {
            self.children(node)
                .into_iter()
                .flat_map(|c| self.comma_exprs(c))
                .collect()
        } else {
            vec![node]
        }
    }

    // ---------- calls ----------

    /// The function name being called, if statically known.
    fn call_name(&mut self, call: Node<'a>) -> Option<String> {
        let f = self.f(call, "function")?;
        Some(match f.kind() {
            "identifier" => self.node_text(f).to_string(),
            "qualified_identifier" => unqual(self.node_text(f)).to_string(),
            "template_function" | "template_method" => {
                // `f<T>(args)` / `static_cast<T>(x)`
                let text = self.node_text(f);
                text.split('<').next().map(unqual).unwrap_or(text).to_string()
            }
            _ => return None,
        })
    }

    /// A `template_function`'s first template argument as a [`CTy`]
    /// (`static_cast<int>(x)` -> int).
    fn template_arg_ty(&mut self, f: Node<'a>) -> CTy {
        self.f(f, "arguments")
            .and_then(|a| self.first(a))
            .map(|a| spec_type(self, Some(a)))
            .unwrap_or(CTy::Unknown)
    }

    fn call(&mut self, node: Node<'a>) -> String {
        let f = self.f(node, "function").unwrap();
        let arg_nodes: Vec<Node<'a>> = self
            .f(node, "arguments")
            .map(|a| {
                self.children(a)
                    .into_iter()
                    .filter(|c| c.kind() != "comment")
                    .collect()
            })
            .unwrap_or_default();

        match f.kind() {
            "field_expression" => return self.method_call(node, f, &arg_nodes),
            "qualified_identifier" | "identifier" | "template_function" => {}
            "parenthesized_expression" | "field_identifier" => {}
            "primitive_type" | "type_identifier" | "template_type" | "sized_type_specifier" => {
                // `int(x)` / `Foo(args)` / `vector<int>(n)`
                let ty = spec_type(self, Some(f));
                let argv: Vec<String> = arg_nodes.iter().map(|a| self.expr(*a)).collect();
                return match ty {
                    CTy::Struct(name) => format!("{name}::new({})", argv.join(", ")),
                    CTy::Arr(elem, _) => match argv.len() {
                        0 => "array::new()".into(),
                        1 => format!("array::new_filled({}, {})", elem.default_init(&self.struct_defaults()), argv[0]),
                        _ => format!("array::new_filled({}, {})", argv[1], argv[0]),
                    },
                    _ => self.cast_to(&ty, argv.first().cloned().unwrap_or_default(), node),
                };
            }
            _ => {}
        }

        let name = self.call_name(node).unwrap_or_default();
        // `static_cast<T>(e)` / `const_cast` / `reinterpret_cast` / `dynamic_cast`
        if name.ends_with("_cast") {
            let ty = self.template_arg_ty(f);
            let v = arg_nodes.first().map(|a| self.expr(*a)).unwrap_or_default();
            return self.cast_to(&ty, v, node);
        }
        self.free_call(&name, &arg_nodes, node)
    }

    /// `f(args)` — std library rewrites first, then user functions (with the
    /// out-param desugar), then a passthrough.
    fn free_call(&mut self, name: &str, args: &[Node<'a>], node: Node<'a>) -> String {
        let argv: Vec<String> = args.iter().map(|a| self.expr(*a)).collect();
        match name {
            // ----- iostream / misc -----
            "printf" | "fprintf" | "sprintf" | "snprintf" => {
                self.diag(DiagKind::Approximate, node, "printf-style call -> print");
                let parts = argv.iter().skip(1.min(argv.len())).collect::<Vec<_>>();
                let fmt = parts
                    .iter()
                    .map(|p| format!("{{{p}}}"))
                    .collect::<Vec<_>>()
                    .join("");
                return format!("print(f\"{fmt}\")");
            }
            "puts" | "putchar" => return format!("print({})", argv[0]),
            "assert" => {
                return format!(
                    "(if !({}) {{ panic(\"assertion failed\") }} else {{ () }})",
                    argv.first().cloned().unwrap_or_else(|| "true".into())
                );
            }
            "exit" => return format!("std::sys::exit({})", argv.first().cloned().unwrap_or_else(|| "0".into())),
            // ----- math -----
            "abs" | "fabs" | "labs" => return format!("{}.abs()", argv[0]),
            "sqrt" | "sqrtf" => return format!("{}.sqrt()", argv[0]),
            "pow" | "powf" => return format!("{}.pow({})", argv[0], argv.get(1).cloned().unwrap_or("2".into())),
            "floor" => return format!("{}.floor()", argv[0]),
            "ceil" => return format!("{}.ceil()", argv[0]),
            "round" | "lround" | "llround" => return format!("{}.round()", argv[0]),
            "fmod" => return format!("{} % {}", argv[0], argv.get(1).cloned().unwrap_or_default()),
            "min" | "fmin" => return format!("{}.min({})", argv[0], argv.get(1).cloned().unwrap_or_default()),
            "max" | "fmax" => return format!("{}.max({})", argv[0], argv.get(1).cloned().unwrap_or_default()),
            "gcd" | "lcm" => {
                self.diag(DiagKind::Unsupported, node, format!("std::{name}"));
                return format!("panic(\"cpp: std::{name}\")");
            }
            // ----- conversions -----
            "to_string" | "to_wstring" => return format!("{}.to_str()", argv[0]),
            "stoi" | "stol" | "stoll" | "atoi" | "atol" | "atoll" => {
                return format!("{}.to_int()!", argv[0]);
            }
            "stof" | "stod" | "stold" | "atof" => return format!("{}.to_float()!", argv[0]),
            "isdigit" => return format!("({} >= \"0\" && {} <= \"9\")", argv[0], argv[0]),
            "isalpha" => {
                return format!(
                    "(({0} >= \"a\" && {0} <= \"z\") || ({0} >= \"A\" && {0} <= \"Z\"))",
                    argv[0]
                );
            }
            "isupper" => return format!("({0} >= \"A\" && {0} <= \"Z\")", argv[0]),
            "islower" => return format!("({0} >= \"a\" && {0} <= \"z\")", argv[0]),
            "isspace" => return format!("({0} == \" \" || {0} == \"\\t\" || {0} == \"\\n\")", argv[0]),
            "toupper" => return format!("{}.to_upper()", argv[0]),
            "tolower" => return format!("{}.to_lower()", argv[0]),
            "size" | "ssize" => return format!("{}.len()", argv[0]),
            "data" => return argv[0].clone(),
            "move" | "forward" | "as_const" => return argv[0].clone(),
            "swap" => {
                self.diag(
                    DiagKind::Note,
                    node,
                    "std::swap in expression position",
                );
                return format!(
                    "{{ let __t = {0}; {0} = {1}; {1} = __t; () }}",
                    argv[0],
                    argv.get(1).cloned().unwrap_or_default()
                );
            }
            "make_pair" | "make_tuple" => return format!("({})", argv.join(", ")),
            "get" => {
                // std::get<0>(t) — the index is in the template arg
                let t = argv[0].clone();
                let idx = self
                    .f(node, "function")
                    .and_then(|f| self.f(f, "arguments"))
                    .and_then(|a| self.first(a))
                    .map(|a| self.node_text(a).to_string())
                    .unwrap_or_else(|| "0".into());
                return format!("{t}.{idx}");
            }
            "tie" => return format!("({})", argv.join(", ")),
            "endl" => return "\"\\n\"".into(),
            // ----- <algorithm> -----
            "sort" => {
                if args.len() >= 2 {
                    let v = self.iter_arg(args[0]);
                    let ty = self.iter_elem_ty(args[0]);
                    return match ty {
                        CTy::Float => format!("{v}.sort_by_float({v})"),
                        _ => format!("{v}.sort_by_int({v})"),
                    };
                }
            }
            "stable_sort" => {
                if args.len() >= 2 {
                    let v = self.iter_arg(args[0]);
                    return format!("{v}.sort_by_int({v})");
                }
            }
            "reverse" => {
                let v = self.iter_arg(args[0]);
                return format!(
                    "{{ {v} = for (i, x) in {v}.enumerate() collect {v}[{v}.len() - 1 - i]; () }}"
                );
            }
            "accumulate" => {
                if args.len() == 3 {
                    let v = self.iter_arg(args[0]);
                    let init = self.expr(args[2]);
                    let _ty = self.iter_elem_ty(args[0]);
                    return format!("({v}.sum() + {init})");
                }
            }
            "count" | "count_if" => {
                if args.len() >= 3 {
                    let v = self.iter_arg(args[0]);
                    let x = self.expr(args[2]);
                    return format!(
                        "(for __e in {v} {{ if __e == {x} {{ collect 1; }} }}).len()"
                    );
                }
            }
            "find" => {
                if args.len() >= 3 {
                    let v = self.iter_arg(args[0]);
                    let x = self.expr(args[2]);
                    return format!(
                        "((for (__i, __e) in {v}.enumerate() {{ if __e == {x} {{ break __i; }} }}) ?? -1)"
                    );
                }
            }
            "binary_search" | "lower_bound" | "upper_bound" => {
                if args.len() >= 3 {
                    let v = self.iter_arg(args[0]);
                    let x = self.expr(args[2]);
                    self.diag(
                        DiagKind::Approximate,
                        node,
                        format!("std::{name} -> membership test"),
                    );
                    return format!("({x} in {v})");
                }
            }
            "min_element" | "max_element" => {
                if args.len() >= 2 {
                    let v = self.iter_arg(args[0]);
                    // value context mimics `*it`; statement context discards
                    return if name == "min_element" {
                        format!("{v}.min()!")
                    } else {
                        format!("{v}.max()!")
                    };
                }
            }
            "fill" | "fill_n" => {
                if args.len() >= 3 {
                    let v = self.iter_arg(args[0]);
                    let x = self.expr(args[2]);
                    return format!(
                        "{{ for (__i, __e) in {v}.enumerate() {{ {v}[__i] = {x}; }}; () }}"
                    );
                }
            }
            "iota" => {
                if args.len() >= 3 {
                    let v = self.iter_arg(args[0]);
                    let x = self.expr(args[2]);
                    return format!(
                        "{{ for __i in {v}.len() {{ {v}[__i] = {x} + __i; }}; () }}"
                    );
                }
            }
            "next_permutation" | "prev_permutation" | "nth_element" | "partial_sort"
            | "partition" | "unique" | "merge" | "set_union" | "set_intersection" => {
                self.diag(DiagKind::Unsupported, node, format!("std::{name}"));
                return format!("panic(\"cpp: std::{name}\")");
            }
            "begin" | "end" | "rbegin" | "rend" | "cbegin" | "cend" => {
                if let Some(a) = argv.first() {
                    return if name.ends_with("end") {
                        format!("{a}.len()")
                    } else {
                        "0".into()
                    };
                }
            }
            _ => {}
        }

        // member function called unqualified inside a method (`m(x)` where `m`
        // is a member) -> `self.m(x)`
        if let Some(owner) = self.cur_struct.clone()
            && self.lookup(name).is_none()
                && self
                    .structs
                    .get(&owner)
                    .is_some_and(|i| i.methods.iter().any(|m| m == name))
            {
                let key = format!("{owner}::{name}");
                let call = format!("{}.{}({})", self.self_name, sanitize(name), argv.join(", "));
                if let Some(sig) = self.fns.get(&key).cloned() {
                    return self.out_call_rewrite(call, &sig, args, node);
                }
                return call;
            }
        // user function — check for out-param desugar
        if let Some(sig) = self.fns.get(name).cloned() {
            let call = format!("{}({})", sanitize(name), argv.join(", "));
            if sig.out_params().is_empty() {
                return call;
            }
            return self.out_call_rewrite(call, &sig, args, node);
        }
        // constructor call `Foo(args)` — struct name in call position
        if let Some(info) = self.structs.get(name)
            && (!info.fields.is_empty() || info.has_ctor) {
                if info.has_ctor {
                    return format!("{name}::new({})", argv.join(", "));
                }
                // aggregate positional init
                let fields = info
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, (fname, _))| {
                        let v = argv.get(i).cloned().unwrap_or_else(|| "0".into());
                        format!("{fname} = {v}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                return format!("{name} {{ {fields} }}");
            }
        format!("{name}({})", argv.join(", "))
    }

    // ---------- member calls ----------

    fn method_call(&mut self, node: Node<'a>, f: Node<'a>, args: &[Node<'a>]) -> String {
        let recv_n = self.f(f, "argument").unwrap();
        let mname = self
            .f(f, "field")
            .map(|m| self.node_text(m).to_string())
            .unwrap_or_default();
        let rty = self.expr_ty(recv_n);
        let recv = self.expr(recv_n);
        let argv: Vec<String> = args.iter().map(|a| self.expr(*a)).collect();
        let arg0 = || argv.first().cloned().unwrap_or_default();

        match mname.as_str() {
            // universal
            "size" | "length" => return format!("{recv}.len()"),
            "empty" => return format!("{recv}.is_empty()"),
            "clear" => {
                return format!("{{ {recv} = array::new(); () }}");
            }
            "data" | "c_str" => return recv,
            "capacity" | "max_size" => return format!("{recv}.len()"),
            "reserve" | "shrink_to_fit" => return "()".into(),
            "front" => return format!("{recv}[0]"),
            "back" => return format!("{recv}[{recv}.len() - 1]"),
            _ => {}
        }

        match &rty {
            CTy::Arr(_, SeqKind::Stack) => match mname.as_str() {
                "push" | "emplace" => return format!("{recv}.push({})", arg0()),
                "pop" => return format!("{recv}.pop()"),
                "top" => return format!("{recv}[{recv}.len() - 1]"),
                _ => {}
            },
            CTy::Arr(_, SeqKind::Queue) => match mname.as_str() {
                "push" | "emplace" => return format!("{recv}.push({})", arg0()),
                "pop" => {
                    return format!(
                        "{{ {recv} = for (i, x) in {recv}.enumerate() {{ if i > 0 {{ collect x; }} }}; () }}"
                    );
                }
                "front" => return format!("{recv}[0]"),
                "back" => return format!("{recv}[{recv}.len() - 1]"),
                _ => {}
            },
            CTy::Arr(_, SeqKind::PQueue { min }) => match mname.as_str() {
                "push" | "emplace" => {
                    let _ = min; // min/max order is not preserved — queue stays sorted
                    return format!("{{ {recv}.push({}); {recv}.sort_by_int({recv}); () }}", arg0());
                }
                "top" => {
                    return if *min {
                        format!("{recv}[0]")
                    } else {
                        format!("{recv}[{recv}.len() - 1]")
                    };
                }
                "pop" => {
                    return if *min {
                        format!(
                            "{{ {recv} = for (i, x) in {recv}.enumerate() {{ if i > 0 {{ collect x; }} }}; () }}"
                        )
                    } else {
                        format!("{recv}.pop()")
                    };
                }
                _ => {}
            },
            CTy::Arr(..) => match mname.as_str() {
                "push_back" | "push" | "emplace_back" | "emplace" | "append" => {
                    return format!("{recv}.push({})", arg0());
                }
                "pop_back" | "pop" => return format!("{recv}.pop()"),
                "at" => return format!("{recv}[{}]", arg0()),
                "insert" => {
                    // v.insert(v.begin() + k, x)
                    let idx = args
                        .first()
                        .map(|a| self.iter_index(*a))
                        .unwrap_or_else(|| "0".into());
                    let x = argv.get(1).cloned().unwrap_or_else(&arg0);
                    return format!(
                        "{{ let __i = {idx}; let __x = {x}; {recv} = for (j, y) in {recv}.enumerate() {{ if j == __i {{ collect __x; }} collect y; }}; if __i == {recv}.len() {{ {recv}.push(__x); }}; () }}"
                    );
                }
                "erase" => {
                    let idx = args.first().map(|a| self.iter_index(*a)).unwrap_or_else(|| "0".into());
                    return format!(
                        "{{ {recv} = for (j, y) in {recv}.enumerate() {{ if j != {idx} {{ collect y; }} }}; () }}"
                    );
                }
                "resize" => {
                    let n = arg0();
                    let fill = argv.get(1).cloned().unwrap_or_else(|| "0".into());
                    return format!(
                        "{{ while {recv}.len() < {n} {{ {recv}.push({fill}); }} while {recv}.len() > {n} {{ {recv}.pop(); }}; () }}"
                    );
                }
                "assign" => {
                    if argv.len() == 2 {
                        return format!(
                            "{{ {recv} = array::new_filled({}, {}); () }}",
                            argv[1], argv[0]
                        );
                    }
                }
                "swap" => {
                    return format!(
                        "{{ let __t = {recv}; {recv} = {0}; {0} = __t; () }}",
                        arg0()
                    );
                }
                "begin" | "cbegin" | "rbegin" => return "0".into(),
                "end" | "cend" | "rend" => return format!("{recv}.len()"),
                _ => {}
            },
            CTy::Str => match mname.as_str() {
                "push_back" | "append" | "operator+=" => return format!("{recv} += {}", arg0()),
                "pop_back" => {
                    return format!(
                        "{{ {recv} = (for (i, c) in {recv}.enumerate() {{ if i < {recv}.len() - 1 {{ collect c; }} }}).join(\"\"); () }}"
                    );
                }
                "substr" => {
                    let i = arg0();
                    let len = argv.get(1).cloned().unwrap_or_else(|| format!("{recv}.len()"));
                    return format!(
                        "(for __j in ({i}..({i} + {len})) collect {recv}[__j]).join(\"\")"
                    );
                }
                "at" => return format!("{recv}[{}]", arg0()),
                "starts_with" => return format!("{recv}.starts_with({})", arg0()),
                "ends_with" => return format!("{recv}.ends_with({})", arg0()),
                "compare" => return format!("({recv} == {})", arg0()),
                "count" => {
                    return format!("({recv}.find_all({})! ?? []).len()", arg0())
                }
                "find" => {
                    // C++ find returns a position (-1 = npos); split is literal
                    return format!(
                        "({{ let __p = {recv}.split({}); if __p.len() > 1 {{ __p[0].len() }} else {{ -1 }} }})",
                        arg0()
                    )
                }
                "rfind" | "find_first_of" | "find_last_of" => {
                    self.diag(DiagKind::Unsupported, node, format!("string::{mname}"));
                    return format!("panic(\"cpp: str::{mname}\")");
                }
                "insert" | "replace" | "erase" => {
                    self.diag(DiagKind::Approximate, node, format!("str::{mname} approximated"));
                    return recv;
                }
                _ => {}
            },
            CTy::Dict(v) => match mname.as_str() {
                "at" => return format!("{recv}[{}]!", arg0()),
                "contains" => return format!("({} in {recv})", arg0()),
                "count" => {
                    return format!("(if {} in {recv} {{ 1 }} else {{ 0 }})", arg0())
                }
                "insert" | "emplace" | "insert_or_assign" => {
                    // m.insert({k, v}) / m[k] — `{k, v}` arrives as initializer_list
                    if let Some(a) = args.first() {
                        if a.kind() == "initializer_list" {
                            let kv = self.children(*a);
                            if kv.len() == 2 {
                                let k = self.expr(kv[0]);
                                let v2 = self.expr(kv[1]);
                                return format!("{{ {recv}[{k}] = {v2}; () }}");
                            }
                        }
                        // make_pair(k, v)
                        if a.kind() == "call_expression" {
                            let inner = self.call_args(*a);
                            if inner.len() == 2 {
                                let k = self.expr(inner[0]);
                                let v2 = self.expr(inner[1]);
                                return format!("{{ {recv}[{k}] = {v2}; () }}");
                            }
                        }
                    }
                    self.diag(DiagKind::Approximate, node, "map::insert approximated");
                    return "()".into();
                }
                "erase" => {
                    return format!(
                        "{{ let __d = ~{{}}; for (__k, __v) in {recv} {{ if __k != {} {{ __d[__k] = __v; }} }} {recv} = __d; () }}",
                        arg0()
                    );
                }
                "find" => {
                    self.diag(
                        DiagKind::Approximate,
                        node,
                        "map::find — use `!= m.end()` pattern for `in`",
                    );
                    return format!("({} in {recv})", arg0());
                }
                "operator[]" => return format!("{recv}[{}]", arg0()),
                _ => {
                    let _ = v;
                }
            },
            CTy::Set(_) => match mname.as_str() {
                "insert" | "emplace" => {
                    return format!("{{ if !({} in {recv}) {{ {recv}.push({}); }}; () }}", arg0(), arg0());
                }
                "contains" => return format!("({} in {recv})", arg0()),
                "count" => {
                    return format!("(if {} in {recv} {{ 1 }} else {{ 0 }})", arg0())
                }
                "erase" => {
                    return format!(
                        "{{ {recv} = for __e in {recv} {{ if __e != {} {{ collect __e; }} }}; () }}",
                        arg0()
                    );
                }
                _ => {}
            },
            CTy::Tuple(_) => match mname.as_str() {
                "first" => return format!("{recv}.0"),
                "second" => return format!("{recv}.1"),
                _ => {}
            },
            CTy::Struct(name) => {
                let key = format!("{}::{}", name.clone(), mname);
                if let Some(sig) = self.fns.get(&key).cloned() {
                    let outs = sig.out_params();
                    let call = format!("{recv}.{}({})", sanitize(&mname), argv.join(", "));
                    if outs.is_empty() {
                        return call;
                    }
                    return self.out_call_rewrite(call, &sig, args, node);
                }
            }
            CTy::Opt(_) => match mname.as_str() {
                "value" => return format!("{recv}!"),
                "value_or" => return format!("({recv} ?? {})", arg0()),
                "has_value" => return format!("({recv} != null)"),
                "reset" => return format!("{{ {recv} = null; () }}"),
                _ => {}
            },
            _ => {}
        }

        // generic fallbacks by name
        match mname.as_str() {
            "push_back" | "push" | "emplace_back" | "emplace" | "append" => {
                format!("{recv}.push({})", arg0())
            }
            "pop_back" => format!("{recv}.pop()"),
            "at" => format!("{recv}[{}]", arg0()),
            "begin" | "cbegin" => "0".into(),
            "end" | "cend" => format!("{recv}.len()"),
            "find" => format!("({} in {recv})", arg0()),
            "count" | "contains" => format!("({} in {recv})", arg0()),
            "insert" | "emplace_hint" => {
                self.diag(DiagKind::Approximate, node, format!("`{mname}` on unknown receiver"));
                format!("{recv}.push({})", argv.last().cloned().unwrap_or_default())
            }
            _ => {
                // user method on a known struct (receiver type unknown), or
                // genuinely missing — emit the call through; mimas methods
                // resolve dynamically enough that many will just work.
                format!("{recv}.{}({})", sanitize(&mname), argv.join(", "))
            }
        }
    }

    /// The shared out-param call rewrite: `f(a, b)` with out-params becomes a
    /// destructuring bind of the tuple return.
    fn out_call_rewrite(&mut self, call: String, sig: &FnSig, args: &[Node<'a>], _node: Node<'a>) -> String {
        let outs = sig.out_params();
        let mut lvals = Vec::new();
        for &i in &outs {
            let Some(a) = args.get(i) else { continue };
            if !matches!(a.kind(), "identifier" | "subscript_expression" | "field_expression") {
                self.diag(
                    DiagKind::Unsupported,
                    *a,
                    "out-param arg is not an lvalue",
                );
                continue;
            }
            lvals.push(self.lvalue(*a));
        }
        // `let` would shadow the caller's vars inside the block — bind temps
        // and assign outward so `f(x, &y)` really updates `y`.
        let mut stmts = String::from("{ ");
        let tmps: Vec<String> = lvals.iter().map(|_| self.fresh()).collect();
        let ret_tmp = (!sig.ret_is_unit).then(|| self.fresh());
        let mut bind_parts = tmps.clone();
        if let Some(r) = &ret_tmp {
            bind_parts.insert(0, r.clone());
        }
        let bind = if bind_parts.len() == 1 {
            bind_parts[0].clone()
        } else {
            format!("({})", bind_parts.join(", "))
        };
        stmts.push_str(&format!("let {bind} = {call}; "));
        for (lv, t) in lvals.iter().zip(tmps.iter()) {
            stmts.push_str(&format!("{lv} = {t}; "));
        }
        match ret_tmp {
            Some(r) => {
                stmts.push_str(&r);
                stmts.push_str(" }");
            }
            None => stmts.push('}'),
        }
        stmts
    }

    /// `v.begin()` / `v.end()` / `v.begin() + k` inside an algorithm call →
    /// the container expression.
    fn iter_arg(&mut self, node: Node<'a>) -> String {
        let node = unwrap_parens(self, node);
        match node.kind() {
            "call_expression" => {
                if let Some(f) = self.f(node, "function")
                    && f.kind() == "field_expression"
                        && let Some(a) = self.f(f, "argument") {
                            return self.expr(a);
                        }
                self.expr(node)
            }
            "binary_expression" => {
                // `v.begin() + 2` — subrange unsupported, take the container
                let l = self.f(node, "left").unwrap();
                self.iter_arg(l)
            }
            _ => self.expr(node),
        }
    }

    /// Iterator expr -> index int: `v.begin() + k` -> `k`, `v.end()` -> len.
    fn iter_index(&mut self, node: Node<'a>) -> String {
        let node = unwrap_parens(self, node);
        match node.kind() {
            "call_expression" => {
                let f = self.f(node, "function");
                let m = f
                    .and_then(|f| self.f(f, "field"))
                    .map(|m| self.node_text(m).to_string())
                    .unwrap_or_default();
                let recv = f
                    .and_then(|f| self.f(f, "argument"))
                    .map(|a| self.expr(a))
                    .unwrap_or_default();
                match m.as_str() {
                    "end" | "cend" | "rend" => format!("{recv}.len()"),
                    _ => "0".into(),
                }
            }
            "binary_expression" => {
                let l = self.f(node, "left").unwrap();
                let r = self.f(node, "right").unwrap();
                let op = self.bin_op(node).to_string();
                let li = self.iter_index(l);
                let ri = self.expr(r);
                format!("({li} {op} {ri})")
            }
            _ => self.expr(node),
        }
    }

    /// Element type of an iterator/container expr.
    fn iter_elem_ty(&mut self, node: Node<'a>) -> CTy {
        let node = unwrap_parens(self, node);
        if node.kind() == "call_expression"
            && let Some(f) = self.f(node, "function")
                && let Some(a) = self.f(f, "argument") {
                    return self.expr_ty(a).elem();
                }
        if node.kind() == "binary_expression"
            && let Some(l) = self.f(node, "left") {
                return self.iter_elem_ty(l);
            }
        self.expr_ty(node).elem()
    }

    fn call_args<'n>(&mut self, call: Node<'n>) -> Vec<Node<'n>> {
        self.f(call, "arguments")
            .map(|a| {
                self.children(a)
                    .into_iter()
                    .filter(|c| c.kind() != "comment")
                    .collect()
            })
            .unwrap_or_default()
    }

    // ---------- ostream/istream chains ----------

    /// `cout << a << b << endl` — returns the value args (endl filtered) when
    /// the chain's head is an output stream.
    pub(crate) fn ostream_chain(&mut self, node: Node<'a>) -> Option<Vec<Node<'a>>>
    {
        let node = unwrap_parens(self, node);
        if node.kind() != "binary_expression" {
            return None;
        }
        // collect left-to-right
        let mut chain = Vec::new();
        let mut cur = node;
        loop {
            if cur.kind() == "binary_expression" && self.bin_op(cur) == "<<" {
                chain.push(self.f(cur, "right").unwrap());
                cur = self.f(cur, "left").unwrap();
            } else {
                chain.push(cur);
                break;
            }
        }
        chain.reverse();
        let head = chain[0];
        let head_name = match head.kind() {
            "identifier" => self.node_text(head).to_string(),
            "qualified_identifier" | "field_expression" => {
                unqual(self.node_text(head)).to_string()
            }
            _ => return None,
        };
        if !matches!(head_name.as_str(), "cout" | "cerr" | "clog" | "wcout") {
            return None;
        }
        let args: Vec<Node<'a>> = chain[1..]
            .iter()
            .copied()
            .filter(|a| {
                let t = self.node_text(*a);
                !matches!(t, "endl" | "std::endl" | "flush" | "std::flush" | "'\\n'")
                    && t != "\"\\n\""
            })
            .collect();
        Some(args)
    }

    /// `cin >> a >> b` — returns (lvalue text, type) per var.
    pub(crate) fn istream_chain(&mut self, node: Node<'a>) -> Option<Vec<(String, CTy)>> {
        let node = unwrap_parens(self, node);
        if node.kind() != "binary_expression" {
            return None;
        }
        let mut chain = Vec::new();
        let mut cur = node;
        loop {
            if cur.kind() == "binary_expression" && self.bin_op(cur) == ">>" {
                chain.push(self.f(cur, "right").unwrap());
                cur = self.f(cur, "left").unwrap();
            } else {
                chain.push(cur);
                break;
            }
        }
        chain.reverse();
        let head = chain[0];
        let head_name = match head.kind() {
            "identifier" => self.node_text(head).to_string(),
            _ => return None,
        };
        if head_name != "cin" && head_name != "wcin" {
            return None;
        }
        let mut out = Vec::new();
        for a in &chain[1..] {
            let ty = self.expr_ty(*a);
            let v = self.lvalue(*a);
            out.push((v, ty));
        }
        Some(out)
    }

    // ---------- best-effort type inference ----------

    /// The C++ type of an expression, for `/`->`~/` and method dispatch.
    pub(crate) fn expr_ty(&mut self, node: Node<'a>) -> CTy {
        match node.kind() {
            "number_literal" => {
                let t = self.node_text(node);
                if t.contains('.') || t.contains('e') || t.contains('E') || t.ends_with('f') {
                    CTy::Float
                } else {
                    CTy::Int
                }
            }
            "char_literal" => CTy::Char,
            "string_literal" | "concatenated_string_literal" => CTy::Str,
            "true" | "false" => CTy::Bool,
            "null" | "nullptr" => CTy::Opt(Box::new(CTy::Unknown)),
            "identifier" => self
                .lookup(self.node_text(node))
                .cloned()
                .unwrap_or(CTy::Unknown),
            "qualified_identifier" => {
                let name = unqual(self.node_text(node));
                self.lookup(name).cloned().unwrap_or(CTy::Unknown)
            }
            "this" => self
                .cur_struct
                .as_ref()
                .map(|s| CTy::Struct(s.clone()))
                .unwrap_or(CTy::Unknown),
            "parenthesized_expression" | "condition_clause" => self
                .first(node)
                .map(|c| self.expr_ty(c))
                .unwrap_or(CTy::Unknown),
            "unary_expression" => {
                let op = self.f(node, "operator").map(|o| self.node_text(o)).unwrap_or("");
                if op == "!" {
                    CTy::Bool
                } else {
                    self.f(node, "argument")
                        .map(|a| self.expr_ty(a))
                        .unwrap_or(CTy::Unknown)
                }
            }
            "update_expression" | "assignment_expression" => {
                let l = self
                    .f(node, "left")
                    .or_else(|| self.f(node, "argument"));
                l.map(|a| self.expr_ty(a)).unwrap_or(CTy::Unknown)
            }
            "binary_expression" => {
                let op = self.bin_op(node);
                match op {
                    "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" => CTy::Bool,
                    "/" => CTy::Float, // mimas `/` floats; `~/` only for int/int
                    _ => {
                        let lt = self
                            .f(node, "left")
                            .map(|l| self.expr_ty(l))
                            .unwrap_or(CTy::Unknown);
                        let rt = self
                            .f(node, "right")
                            .map(|r| self.expr_ty(r))
                            .unwrap_or(CTy::Unknown);
                        match (&lt, &rt) {
                            (CTy::Str, _) | (_, CTy::Str) => CTy::Str,
                            (CTy::Float, _) | (_, CTy::Float) => CTy::Float,
                            (CTy::Int, CTy::Int) | (CTy::Char, _) | (_, CTy::Char) => CTy::Int,
                            _ => lt,
                        }
                    }
                }
            }
            "conditional_expression" => self
                .f(node, "consequence")
                .map(|c| self.expr_ty(c))
                .unwrap_or(CTy::Unknown),
            "subscript_expression" => {
                let a = self.f(node, "argument").unwrap();
                self.expr_ty(a).elem()
            }
            "field_expression" => {
                let a = self.f(node, "argument").unwrap();
                let fname = self
                    .f(node, "field")
                    .map(|f| self.node_text(f).to_string())
                    .unwrap_or_default();
                if let CTy::Struct(name) = self.expr_ty(a)
                    && let Some(info) = self.structs.get(&name)
                        && let Some((_, t)) = info.fields.iter().find(|(f, _)| *f == fname) {
                            return t.clone();
                        }
                match fname.as_str() {
                    "first" => self.expr_ty(a).elem(),
                    "second" => {
                        if let CTy::Tuple(ts) = self.expr_ty(a) {
                            ts.get(1).cloned().unwrap_or(CTy::Unknown)
                        } else {
                            CTy::Unknown
                        }
                    }
                    _ => CTy::Unknown,
                }
            }
            "call_expression" => self.call_ty(node),
            "cast_expression" => spec_type(self, self.f(node, "type")),
            "new_expression" => spec_type(self, self.f(node, "type")),
            "initializer_list" => CTy::Arr(Box::new(CTy::Unknown), SeqKind::Vector),
            "comma_expression" => self
                .comma_exprs(node)
                .last()
                .map(|c| self.expr_ty(*c))
                .unwrap_or(CTy::Unknown),
            "lambda_expression" => CTy::Unknown,
            _ => CTy::Unknown,
        }
    }

    fn call_ty(&mut self, node: Node<'a>) -> CTy {
        let f = self.f(node, "function");
        let Some(f) = f else { return CTy::Unknown };
        match f.kind() {
            "field_expression" => {
                let m = self
                    .f(f, "field")
                    .map(|m| self.node_text(m).to_string())
                    .unwrap_or_default();
                match m.as_str() {
                    "len" | "size" | "length" | "count" | "find" => CTy::Int,
                    "empty" | "is_empty" | "contains" | "has_value" | "starts_with"
                    | "ends_with" => CTy::Bool,
                    "pop" | "pop_back" | "front" | "back" | "top" | "at" | "value" => {
                        self.f(f, "argument")
                            .map(|a| self.expr_ty(a).elem())
                            .unwrap_or(CTy::Unknown)
                    }
                    "substr" | "c_str" | "data" | "to_str" => CTy::Str,
                    "to_int" => CTy::Int,
                    "to_float" => CTy::Float,
                    _ => CTy::Unknown,
                }
            }
            _ => {
                let name = self.call_name(node).unwrap_or_default();
                match name.as_str() {
                    "min" | "max" | "abs" | "stoi" | "atoi" | "size" | "ssize" => CTy::Int,
                    "sqrt" | "pow" | "stod" | "atof" => CTy::Float,
                    "to_string" => CTy::Str,
                    _ => self
                        .fns
                        .get(&name)
                        .map(|s| s.ret.clone())
                        .unwrap_or(CTy::Unknown),
                }
            }
        }
    }
}
