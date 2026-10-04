//! Statement lowering.

use tree_sitter::Node;

use super::cty::{declarator, spec_type};
use super::{Lower, one_line};
use crate::ty::CTy;
use crate::DiagKind;

impl<'a> Lower<'a> {
    /// Lower one statement node.
    pub(crate) fn stmt(&mut self, node: Node<'a>) {
        match node.kind() {
            "compound_statement" => self.block(node),
            "declaration" => self.decl(node, false),
            "expression_statement" => self.expr_stmt(node),
            "if_statement" => self.if_stmt(node),
            "while_statement" => self.while_stmt(node),
            "do_statement" => self.do_stmt(node),
            "for_statement" => self.for_stmt(node),
            "for_range_loop" => self.range_for(node),
            "return_statement" => self.return_stmt(node),
            "switch_statement" => self.switch_stmt(node),
            "case_statement" => self.unsupported(node, "`case` outside switch"),
            "break_statement" => self.line("break;"),
            "continue_statement" => self.line("continue;"),
            "comment" => self.comment_stmt(node),
            "labeled_statement" => {
                // `label:` — only reachable via goto, which we don't support.
                self.diag(DiagKind::Note, node, "label dropped (no goto)");
                let inner = self.children(node).into_iter().last();
                if let Some(inner) = inner {
                    self.stmt(inner);
                }
            }
            "try_statement" => {
                self.diag(
                    DiagKind::Approximate,
                    node,
                    "try/catch: body kept, handlers dropped",
                );
                if let Some(body) = self.f(node, "body") {
                    self.block_stmts(body);
                }
            }
            "throw_statement" => {
                let e = self.first(node).map(|e| self.expr(e)).unwrap_or_default();
                self.line(&format!("panic({e});"));
            }
            "goto_statement" => self.unsupported(node, "goto"),
            "type_definition" | "alias_declaration" | "using_declaration" => {
                self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
            }
            "struct_specifier" | "class_specifier" | "enum_specifier" => {
                // local struct — unsupported, rare in practice
                self.unsupported(node, "local type declaration");
            }
            "static_assert" | "static_assert_declaration" => {}
            "ERROR" => self.unsupported(node, "parse error"),
            _ => {
                // maybe an expression used as a statement without a wrapper
                if self.is_expr(node) {
                    self.expr_stmt_inner(node);
                } else {
                    self.unsupported(node, node.kind());
                }
            }
        }
    }

    /// `{ ... }` as a statement — pushes a lexical scope.
    pub(crate) fn block(&mut self, node: Node<'a>) {
        self.open("{");
        self.push_scope();
        if node.kind() == "compound_statement" {
            self.block_stmts(node);
        } else {
            self.stmt(node);
        }
        self.pop_scope();
        self.close("");
    }

    fn comment_stmt(&mut self, node: Node<'a>) {
        for t in self.node_text(node).lines() {
            let t = t
                .trim()
                .trim_start_matches("/*")
                .trim_start_matches('*')
                .trim_start_matches("//")
                .trim_end_matches("*/")
                .trim();
            self.line(&format!("// {t}"));
        }
    }

    /// `e;` — with the ostream/istream chain special cases first.
    fn expr_stmt(&mut self, node: Node<'a>) {
        let Some(inner) = self.first(node) else { return };
        self.expr_stmt_inner(inner);
    }

    pub(crate) fn expr_stmt_inner(&mut self, inner: Node<'a>) {
        if let Some(args) = self.ostream_chain(inner) {
            // `cout << "sum=" << s << endl;` -> `print(f"sum={s}");` — string
            // literals land in the f-string text, other exprs interpolate.
            let body: Vec<String> = args
                .iter()
                .map(|a| {
                    if a.kind() == "string_literal" {
                        // keep the literal's content as f-string text
                        let t = self.node_text(*a);
                        let inner = t.trim_matches('"');
                        inner.replace("{", "{{").replace("}", "}}")
                    } else {
                        let e = self.expr(*a);
                        // `{{` is the f-string escape — block exprs need parens
                        if e.starts_with('{') {
                            format!("{{({e})}}")
                        } else {
                            format!("{{{e}}}")
                        }
                    }
                })
                .collect();
            self.line(&format!("print(f\"{}\");", body.join("")));
            return;
        }
        if let Some(vars) = self.istream_chain(inner) {
            for (v, ty) in vars {
                let read = match ty {
                    CTy::Int | CTy::Char => "std::sys::stdin()!.to_int()!".to_string(),
                    CTy::Float => "std::sys::stdin()!.to_float()!".to_string(),
                    _ => "std::sys::stdin()!".to_string(),
                };
                self.line(&format!("{v} = {read};"));
            }
            return;
        }
        let e = self.expr(inner);
        self.line(&format!("{e};"));
    }

    /// `int x = 5, y = x + 1;` — also globals (file scope).
    pub(crate) fn decl(&mut self, node: Node<'a>, global: bool) {
        let Some((base, decls)) = super::item::decl_parts(self, node) else {
            return;
        };
        let is_const = self.node_text(node).contains("const")
            || self.node_text(node).starts_with("constexpr");
        let is_static = self.node_text(node).starts_with("static");
        if is_static && !global {
            self.diag(
                DiagKind::Approximate,
                node,
                "function-local `static` behaves like a normal local",
            );
        }
        for (d, ty) in decls {
            self.decl_one(node, &d, ty, is_const, global);
        }
        let _ = base;
    }

    fn decl_one(&mut self, node: Node<'a>, d: &super::cty::Decl<'a>, ty: CTy, is_const: bool, _g: bool) {
        // structured binding: `auto [a, b] = t`
        if let Some(names) = &d.binding_names {
            if let Some(init) = d.init {
                let e = self.expr(init);
                self.line(&format!("let ({}) = {e};", names.join(", ")));
                for n in names {
                    self.bind(n, CTy::Unknown);
                }
            }
            return;
        }
        let name = &d.name;
        // array dims from the declarator (`int a[3][4]`)
        let ty = ty.clone();
        // pick keyword
        let kw = if is_const && d.init.as_ref().is_some_and(|i| self.is_const_literal(*i)) {
            "const"
        } else {
            "let"
        };
        self.bind(name, ty.clone());
        let annot = ty
            .annot()
            .map(|a| format!(": {a}"))
            .unwrap_or_default();
        if let Some(init) = d.init {
            let init = self.init_expr(node, &ty, init, d);
            self.line(&format!("{kw} {name}{annot} = {init};"));
        } else if !d.dims.is_empty() {
            // `int a[3][4]` without init
            let init = self.dim_init(&ty, &d.dims);
            self.line(&format!("{kw} {name}{annot} = {init};"));
        } else {
            // `Point p;` invokes the default ctor — honor a user-defined one
            let init = match ty {
                CTy::Struct(name)
                    if self.structs.get(name.as_str()).is_some_and(|s| s.has_ctor) =>
                {
                    format!("{name}::new()")
                }
                _ => ty.default_init(&self.struct_defaults()),
            };
            if kw == "const" {
                self.diag(
                    DiagKind::Approximate,
                    node,
                    format!("`const {name}` has no initializer — emitted `let`"),
                );
                self.line(&format!("let {name}{annot} = {init};"));
            } else {
                self.line(&format!("let {name}{annot} = {init};"));
            }
        }
    }

    /// Whether an init expression is literal enough for `const`.
    fn is_const_literal(&mut self, node: Node<'a>) -> bool {
        match node.kind() {
            "number_literal" | "string_literal" | "true" | "false" | "char_literal"
            | "concatenated_string_literal" => true,
            "unary_expression" => self
                .f(node, "argument")
                .is_some_and(|a| self.is_const_literal(a)),
            "parenthesized_expression" => self
                .first(node)
                .is_some_and(|a| self.is_const_literal(a)),
            "binary_expression" => {
                let l = self.f(node, "left");
                let r = self.f(node, "right");
                l.is_some_and(|x| self.is_const_literal(x))
                    && r.is_some_and(|x| self.is_const_literal(x))
            }
            "identifier" | "qualified_identifier" => true, // may name a const
            _ => false,
        }
    }

    /// `let a = <nested new_filled>` for `int a[d1][d2]...`.
    ///
    /// `dims` arrive outermost-declarator first (`a[3][4]` pushes `[4, 3]` —
    /// the outer node's size is the innermost array's length), which is also
    /// inside-out construction order for `new_filled`.
    fn dim_init(&mut self, ty: &CTy, dims: &[Node<'a>]) -> String {
        let sizes: Vec<String> = dims.iter().map(|d| self.expr(*d)).collect();
        // innermost element type
        let mut elem = ty.clone();
        let n = dims.len();
        for _ in 0..n {
            elem = elem.elem();
        }
        let zero = elem.default_init(&self.struct_defaults());
        // build inside-out: innermost dim first
        let mut e = format!("array::new_filled({zero}, {})", sizes[0]);
        for size in &sizes[1..] {
            e = format!("(for __i in {size} collect {e})");
        }
        e
    }

    /// The right-hand side of `let name = <init>` from `= e`, `(args)`, or
    /// `{a, b}` initializer forms.
    fn init_expr(&mut self, _node: Node<'a>, ty: &CTy, init: Node<'a>, d: &super::cty::Decl<'a>) -> String {
        match init.kind() {
            "initializer_list" => self.brace_init(ty, init),
            "argument_list" => self.ctor_args(ty, init), // `T x(a, b)`
            _ => {
                // plain `= expr` — but `T x(e)` also lands here for single args
                // on some grammar versions (function_declarator style handled
                // by caller as fn_params)
                let e = self.expr(init);
                // value-semantics copy for containers/structs bound from an lvalue
                if !d.init_is_paren
                    && matches!(ty, CTy::Arr(..) | CTy::Dict(_) | CTy::Set(_) | CTy::Struct(_))
                    && matches!(init.kind(), "identifier" | "field_expression" | "subscript_expression" | "qualified_identifier")
                {
                    return ty.copy_expr(&e, &self.struct_defaults());
                }
                e
            }
        }
    }

    /// `{a, b, c}` — array literal, tuple, or aggregate struct init.
    fn brace_init(&mut self, ty: &CTy, init: Node<'a>) -> String {
        let items: Vec<Node<'a>> = self
            .children(init)
            .into_iter()
            .filter(|c| c.kind() != "comment")
            .collect();
        match ty {
            CTy::Arr(..) | CTy::Set(_) => {
                format!("[{}]", items.iter().map(|i| self.expr(*i)).collect::<Vec<_>>().join(", "))
            }
            CTy::Tuple(_) => {
                format!("({})", items.iter().map(|i| self.expr(*i)).collect::<Vec<_>>().join(", "))
            }
            CTy::Dict(_) => {
                // `{{k, v}, ...}` pairs -> `~{ k = v }`? dict literal keys are
                // identifiers-as-strings; emit `~{}` + inserts via expr map.
                let _ = items;
                self.diag(DiagKind::Approximate, init, "map brace-init -> `~{}`");
                "~{}".into()
            }
            CTy::Struct(name) => {
                // positional aggregate init `X{a, b}` -> `X { f1 = a, f2 = b }`
                if let Some(info) = self.structs.get(name).cloned() {
                    let fields = info
                        .fields
                        .iter()
                        .enumerate()
                        .map(|(i, (fname, fty))| {
                            let v = items
                                .get(i)
                                .map(|n| self.expr(*n))
                                .unwrap_or_else(|| fty.default_init(&self.struct_defaults()));
                            format!("{fname} = {v}")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{name} {{ {fields} }}")
                } else {
                    self.diag(DiagKind::Unsupported, init, "aggregate init of unknown struct");
                    format!("{name} {{}}")
                }
            }
            _ => format!("[{}]", items.iter().map(|i| self.expr(*i)).collect::<Vec<_>>().join(", ")),
        }
    }

    /// `T x(args)` constructor-call form.
    fn ctor_args(&mut self, ty: &CTy, args: Node<'a>) -> String {
        let argv: Vec<String> = self
            .children(args)
            .into_iter()
            .filter(|c| c.kind() != "comment")
            .map(|a| self.expr(a))
            .collect();
        match ty {
            CTy::Arr(elem, _) => {
                match argv.len() {
                    // vector<T>(n) -> filled with elem default
                    1 => format!(
                        "array::new_filled({}, {})",
                        elem.default_init(&self.struct_defaults()),
                        argv[0]
                    ),
                    // vector<T>(n, x)
                    _ => format!("array::new_filled({}, {})", argv[1], argv[0]),
                }
            }
            CTy::Str => {
                // string(n, 'c') -> repeat
                if argv.len() == 2 {
                    format!("{}.repeat({})", argv[1], argv[0])
                } else {
                    argv.first().cloned().unwrap_or_else(|| "\"\"".into())
                }
            }
            CTy::Struct(name) => format!("{name}::new({})", argv.join(", ")),
            CTy::Int => {
                let a = argv.first().cloned().unwrap_or_else(|| "0".into());
                // a string/char arg needs the conversion; an int expr doesn't
                if a.starts_with('"') {
                    format!("{a}.to_int()!")
                } else {
                    a
                }
            }
            CTy::Float => {
                format!("{}.to_float()", argv.first().cloned().unwrap_or_else(|| "0".into()))
            }
            _ => argv.first().cloned().unwrap_or_else(|| "0".into()),
        }
    }

    /// `if/else if/else` — collect the chain, then emit mimas's native shape.
    fn if_stmt(&mut self, node: Node<'a>) {
        let mut arms: Vec<(String, Node<'a>)> = Vec::new();
        let mut cur = node;
        let else_body = loop {
            let cond = self.cond(cur);
            if let Some(c) = self.f(cur, "consequence") {
                arms.push((cond, c));
            }
            match self
                .f(cur, "alternative")
                // `alternative` is an else_clause — its named child is either
                // an if_statement (else-if chain) or the else body.
                .and_then(|ec| self.first(ec))
            {
                Some(a) if a.kind() == "if_statement" => cur = a,
                other => break other,
            }
        };
        for (i, (cond, cons)) in arms.iter().enumerate() {
            if i == 0 {
                self.open(&format!("if {cond}"));
            } else {
                self.indent = self.indent.saturating_sub(1);
                self.line(&format!("}} else if {cond} {{"));
                self.indent += 1;
            }
            self.push_scope();
            self.braced_inner(*cons);
            self.pop_scope();
        }
        if let Some(eb) = else_body {
            self.indent = self.indent.saturating_sub(1);
            self.line("} else {");
            self.indent += 1;
            self.push_scope();
            self.braced_inner(eb);
            self.pop_scope();
        }
        self.close("");
    }

    /// `if (cond)` — `condition` is a `condition_clause` whose `value` field
    /// holds the expression (plus an optional C++17 init-statement).
    pub(crate) fn cond(&mut self, node: Node<'a>) -> String {
        let c = self.f(node, "condition").unwrap_or(node);
        if c.kind() == "condition_clause" {
            if let Some(init) = self.f(c, "initializer") {
                self.diag(
                    DiagKind::Approximate,
                    init,
                    "if/switch initializer dropped — bind it before the statement instead",
                );
            }
            if let Some(v) = self.f(c, "value") {
                if v.kind() == "declaration" {
                    self.diag(DiagKind::Unsupported, v, "declaration in condition");
                    return "true".into();
                }
                return self.expr_unparen(v);
            }
        }
        self.expr_unparen(c)
    }

    fn while_stmt(&mut self, node: Node<'a>) {
        let cond = self.cond(node);
        self.open(&format!("while {cond}"));
        self.push_scope();
        if let Some(b) = self.f(node, "body") {
            self.braced_inner(b);
        }
        self.pop_scope();
        self.close("");
    }

    /// Inside an opened block: emit `b`'s statements (it may be a
    /// compound_statement or a bare statement).
    fn braced_inner(&mut self, b: Node<'a>) {
        match b.kind() {
            "compound_statement" => self.block_stmts(b),
            _ => self.stmt(b),
        }
    }

    /// `do { b } while (c)` -> `loop { b; if !(c) { break } }`.
    fn do_stmt(&mut self, node: Node<'a>) {
        self.open("loop");
        self.push_scope();
        if let Some(b) = self.f(node, "body") {
            match b.kind() {
                "compound_statement" => self.block_stmts(b),
                _ => self.stmt(b),
            }
        }
        let cond = self.cond(node);
        self.line(&format!("if !({cond}) {{ break; }}"));
        self.pop_scope();
        self.close("");
    }

    /// `for (init; cond; update) body`.
    ///
    /// The common `for (i = a; i < b; i++)` becomes `for i in (a..b)`.
    fn for_stmt(&mut self, node: Node<'a>) {
        if let Some((var, lo, hi, inclusive)) = self.range_for_parts(node) {
            let op = if inclusive { "..=" } else { ".." };
            self.open(&format!("for {var} in {lo}{op}{hi}"));
            self.push_scope();
            self.bind(&var, CTy::Int);
            if let Some(b) = self.f(node, "body") {
                match b.kind() {
                    "compound_statement" => self.block_stmts(b),
                    _ => self.stmt(b),
                }
            }
            self.pop_scope();
            self.close("");
            return;
        }
        // general case: `{ init; while cond { body; update; } }`
        self.open("{");
        self.push_scope();
        if let Some(init) = self.f(node, "initializer") {
            self.stmt(init);
        }
        let cond = self
            .f(node, "condition")
            .map(|c| self.expr_unparen(c))
            .unwrap_or_else(|| "true".into());
        self.open(&format!("while {cond}"));
        if let Some(b) = self.f(node, "body") {
            match b.kind() {
                "compound_statement" => self.block_stmts(b),
                _ => self.stmt(b),
            }
        }
        if let Some(u) = self.f(node, "update") {
            // update may be a comma list
            for e in comma_parts(self, u) {
                let e_str = self.expr(e);
                self.line(&format!("{e_str};"));
            }
        }
        self.close("");
        self.pop_scope();
        self.close("");
    }

    /// Detect `for (i = lo; i < hi; ++i)` / `i++` / `i += 1` / `i <= hi`.
    fn range_for_parts(&mut self, node: Node<'a>) -> Option<(String, String, String, bool)> {
        let init = self.f(node, "initializer")?;
        let cond = self.f(node, "condition")?;
        let update = self.f(node, "update")?;

        // init: declaration `T i = e` or assignment `i = e`
        let (var, lo): (String, Node<'a>) = match init.kind() {
            "declaration" => {
                let (_, decls) = super::item::decl_parts(self, init)?;
                let (d, _) = decls.into_iter().next()?;
                if d.binding_names.is_some() || d.fn_params.is_some() {
                    return None;
                }
                (d.name.clone(), d.init?)
            }
            "expression_statement" => {
                let inner = self.first(init)?;
                if inner.kind() != "assignment_expression" {
                    return None;
                }
                let lhs = self.f(inner, "left")?;
                if lhs.kind() != "identifier" {
                    return None;
                }
                (
                    self.node_text(lhs).to_string(),
                    self.f(inner, "right")?,
                )
            }
            _ => return None,
        };
        // cond: `i < hi` or `i <= hi`
        let cond = unwrap_parens(self, cond);
        if cond.kind() != "binary_expression" {
            return None;
        }
        let lhs = self.f(cond, "left")?;
        if lhs.kind() != "identifier" || self.node_text(lhs) != var {
            return None;
        }
        let op = self.bin_op(cond);
        let inclusive = match op {
            "<" => false,
            "<=" => true,
            _ => return None,
        };
        let hi = self.f(cond, "right")?;
        // update: `i++`, `++i`, `i += 1`
        let ok = match update.kind() {
            "update_expression" => self
                .f(update, "argument")
                .is_some_and(|a| a.kind() == "identifier" && self.node_text(a) == var),
            "assignment_expression" => {
                let l = self.f(update, "left");
                let op = self.assign_op(update);
                let r = self.f(update, "right");
                l.is_some_and(|a| a.kind() == "identifier" && self.node_text(a) == var)
                    && op == "+="
                    && r.is_some_and(|r| self.node_text(r) == "1")
            }
            _ => false,
        };
        if !ok {
            return None;
        }
        Some((
            var,
            self.expr(lo),
            self.expr(hi),
            inclusive,
        ))
    }

    /// `for (T x : v)` -> `for x in v`.
    fn range_for(&mut self, node: Node<'a>) {
        let right = self.f(node, "right").map(|r| self.expr(r)).unwrap_or_default();
        // binding: declaration-ish `type declarator`
        let decl = self.f(node, "declarator");
        let name = decl
            .map(|d| {
                let d = declarator(self, d);
                if let Some(names) = d.binding_names.clone() {
                    format!("({})", names.join(", "))
                } else {
                    let ty = spec_type(self, self.f(node, "type"));
                    self.bind(&d.name, ty);
                    d.name
                }
            })
            .unwrap_or_else(|| "_".into());
        self.open(&format!("for {name} in {right}"));
        self.push_scope();
        if let Some(b) = self.f(node, "body") {
            match b.kind() {
                "compound_statement" => self.block_stmts(b),
                _ => self.stmt(b),
            }
        }
        self.pop_scope();
        self.close("");
    }

    fn return_stmt(&mut self, node: Node<'a>) {
        let e = self.first(node).map(|v| self.expr(v));
        if let Some(sig) = self.cur_fn.clone() {
            let outs = sig.out_params();
            if !outs.is_empty() {
                let names: Vec<String> = outs.iter().map(|i| sig.params[*i].0.clone()).collect();
                let ret = match e {
                    Some(e) if !sig.ret_is_unit => {
                        if names.is_empty() {
                            e
                        } else {
                            format!("({e}, {})", names.join(", "))
                        }
                    }
                    _ => {
                        if names.len() == 1 {
                            names[0].clone()
                        } else {
                            format!("({})", names.join(", "))
                        }
                    }
                };
                self.line(&format!("return {ret};"));
                return;
            }
        }
        match e {
            Some(e) => self.line(&format!("return {e};")),
            None => self.line("return;"),
        }
    }

    /// `switch (e) { case a: s.. break; default: .. }` -> a `loop` wrapping an
    /// if/else-if chain so `break` inside cases exits the dispatch.
    fn switch_stmt(&mut self, node: Node<'a>) {
        let cond = self.cond(node);
        self.open("loop");
        self.push_scope();
        let mut first = true;
        if let Some(body) = self.f(node, "body") {
            let mut cases: Vec<Node<'a>> = self
                .children(body)
                .into_iter()
                .filter(|c| c.kind() == "case_statement")
                .collect();
            // a `default:` that isn't last must still emit last — `} else if`
            // can't follow `} else`
            cases.sort_by_key(|c| usize::from(self.f(*c, "value").is_none()));
            for case in cases {
                let value = self.f(case, "value");
                // statements of a case are its trailing children
                let stmts: Vec<Node<'a>> = self
                    .children(case)
                    .into_iter()
                    .filter(|c| c.kind() != "comment" && Some(*c) != value)
                    .collect();
                let ends_break = stmts
                    .last()
                    .is_some_and(|s| s.kind() == "break_statement");
                let head = match value {
                    Some(v) => {
                        let v = self.expr(v);
                        format!("{}if {cond} == {v} {{", if first { "" } else { "} else " })
                    }
                    None => "} else {".to_string(),
                };
                let _ = first;
                self.line(&head);
                self.indent += 1;
                // warn on fallthrough
                if !ends_break && !stmts.is_empty() {
                    self.diag(
                        DiagKind::Approximate,
                        case,
                        "switch case without `break` — fallthrough is not preserved",
                    );
                }
                for s in stmts {
                    if s.kind() == "break_statement" {
                        continue;
                    }
                    self.stmt(s);
                }
                self.indent -= 1;
                first = false;
            }
        }
        if first {
            self.line("// cpp: empty switch");
        } else {
            self.line("}");
        }
        self.line("break;");
        self.pop_scope();
        self.close("");
    }
}

/// `comma_expression` children.
fn comma_parts<'n>(l: &mut Lower<'n>, node: Node<'n>) -> Vec<Node<'n>> {
    if node.kind() == "comma_expression" {
        l.children(node)
            .into_iter()
            .flat_map(|c| comma_parts(l, c))
            .collect()
    } else {
        vec![node]
    }
}

/// Strip `parenthesized_expression`/`condition_clause` wrappers.
pub(crate) fn unwrap_parens<'n>(_l: &Lower<'n>, node: Node<'n>) -> Node<'n> {
    let mut cur = node;
    loop {
        match cur.kind() {
            "parenthesized_expression" | "condition_clause" => {
                match cur.named_child(0) {
                    Some(inner) => cur = inner,
                    None => return cur,
                }
            }
            _ => return cur,
        }
    }
}
