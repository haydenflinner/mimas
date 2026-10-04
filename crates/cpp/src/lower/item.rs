//! Top-level items: functions, structs, globals, enums, using/typedef.

use tree_sitter::Node;

use super::cty::{declarator, spec_type, unqual};
use super::{Alias, FnSig, Lower, StructInfo, one_line};
use crate::ty::CTy;
use crate::DiagKind;

impl<'a> Lower<'a> {
    /// Entry: collect signatures first (call sites need them for the out-param
    /// desugar), then lower each item.
    pub fn translation_unit(&mut self, root: Node<'a>) {
        let items = self.children(root);
        for node in &items {
            self.prescan(*node);
        }
        for node in items {
            self.mark(node);
            self.item(node);
        }
    }

    /// Signature-only pass: fns, structs (fields/methods), aliases, globals.
    fn prescan(&mut self, node: Node<'a>) {
        match node.kind() {
            "function_definition" | "declaration" => {
                if let Some(sig) = self.fn_sig(node) {
                    let name = sig_name(self, node);
                    self.fns.insert(name, sig);
                } else {
                    // global declaration: register names for `x / y` typing
                    if let Some((_, decl)) = decl_parts(self, node) {
                        for (d, ty) in decl {
                            if !d.name.is_empty() {
                                self.bind(&d.name, ty);
                            }
                        }
                    }
                }
            }
            "struct_specifier" | "class_specifier" => {
                self.prescan_struct(node);
            }
            "type_definition" | "alias_declaration" => {
                self.prescan_alias(node);
            }
            "enum_specifier" | "preproc_def" => {}
            "template_declaration" => {
                // prescan the inner item so signatures exist even though we
                // can't specialize — calls inside the file still resolve.
                if let Some(inner) = self.children(node).last() {
                    self.prescan(*inner);
                }
            }
            "linkage_specification" | "namespace_definition" => {
                for c in self.children(node) {
                    self.prescan(c);
                }
            }
            _ => {}
        }
    }

    /// `using ll = long long;` / `typedef int T;` / `namespace fs = std::x;`
    fn prescan_alias(&mut self, node: Node<'a>) {
        match node.kind() {
            "alias_declaration" => {
                let name = self
                    .f(node, "name")
                    .map(|n| self.node_text(n).to_string())
                    .unwrap_or_default();
                let ty = spec_type(self, self.f(node, "type"));
                self.aliases.insert(name, Alias::Ty(ty));
            }
            "type_definition" => {
                // typedef <type> <declarator>;
                let ty = spec_type(self, self.f(node, "type"));
                if let Some(d) = self.f(node, "declarator") {
                    let d = declarator(self, d);
                    self.aliases.insert(d.name, Alias::Ty(ty));
                }
            }
            _ => {}
        }
    }

    /// Record a struct's fields + method names without lowering bodies.
    fn prescan_struct(&mut self, node: Node<'a>) {
        let Some(name) = self.f(node, "name").map(|n| self.node_text(n).to_string()) else {
            return;
        };
        let mut info = StructInfo::default();
        let access_private = node.kind() == "class_specifier";
        let _ = access_private; // all members are visible in mimas
        if let Some(body) = self.f(node, "body") {
            for member in self.children(body) {
                match member.kind() {
                    "field_declaration" => {
                        let base = spec_type(self, self.f(member, "type"));
                        // `int y = 5` / `int y{5}` put the default in a
                        // `default_value` field on the declaration itself
                        let default_node = self.f(member, "default_value");
                        for c in self.children(member) {
                            // each declarator child
                            if matches!(
                                c.kind(),
                                "init_declarator"
                                    | "identifier"
                                    | "field_identifier"
                                    | "array_declarator"
                                    | "reference_declarator"
                                    | "pointer_declarator"
                                    | "parenthesized_declarator"
                            ) {
                                if c.kind() == "init_declarator"
                                    && let Some(v) = self.f(c, "value") {
                                        let inner = self.f(c, "declarator").unwrap();
                                        let d = declarator(self, inner);
                                        let ty = self.wrap_type(base.clone(), &d);
                                        info.fields.push((d.name.clone(), ty));
                                        let val = self.expr(v);
                                        info.defaults.insert(d.name, val);
                                        continue;
                                    }
                                let d = declarator(self, c);
                                let ty = self.wrap_type(base.clone(), &d);
                                if let Some(dv) = default_node {
                                    // `y{5}` — initializer_list around a scalar
                                    let dv = if dv.kind() == "initializer_list" {
                                        self.children(dv)
                                            .into_iter()
                                            .find(|c| c.kind() != "comment")
                                            .unwrap_or(dv)
                                    } else {
                                        dv
                                    };
                                    let val = self.expr(dv);
                                    info.defaults.insert(d.name.clone(), val);
                                }
                                info.fields.push((d.name, ty));
                            }
                        }
                    }
                    "function_definition" => {
                        let dname = self
                            .f(member, "declarator")
                            .and_then(|d| self.f(d, "declarator"))
                            .map(|d| self.node_text(d))
                            .unwrap_or("");
                        let dname = unqual(dname);
                        if dname.starts_with('~') {
                            continue; // destructor — GC handles it
                        }
                        if dname == name {
                            info.has_ctor = true;
                        } else {
                            info.methods.push(dname.to_string());
                            if let Some(d) = self.f(member, "declarator") {
                                let sig = self.sig_from(self.f(member, "type"), d);
                                self.fns.insert(format!("{name}::{dname}"), sig);
                            }
                        }
                    }
                    "template_declaration" | "access_specifier" | "comment"
                    | "friend_declaration" | "using_declaration" | "declaration"
                    | "type_definition" | "alias_declaration" | "enum_specifier"
                    | "struct_specifier" | "field_declaration_list" | "storage_class_specifier"
                    | "preproc_if" | "preproc_ifdef" | "preproc_else" | "preproc_endif" => {}
                    _ => {}
                }
            }
        }
        self.structs.insert(name, info);
    }

    /// Apply declarator dims (arrays) onto a base type.
    pub(crate) fn wrap_type(&mut self, mut base: CTy, d: &super::cty::Decl<'a>) -> CTy {
        for _ in &d.dims {
            base = CTy::Arr(Box::new(base), crate::ty::SeqKind::Vector);
        }
        base
    }

    /// Extract a signature from a `function_definition`, or from a
    /// `declaration` whose declarator is a `function_declarator` (a prototype).
    fn fn_sig(&mut self, node: Node<'a>) -> Option<FnSig> {
        let (ret_node, decl_node) = match node.kind() {
            "function_definition" => (self.f(node, "type"), self.f(node, "declarator")?),
            "declaration" => {
                let t = self.f(node, "type");
                let d = self.f(node, "declarator")?;
                // `declaration` covers prototypes AND `int x = 5`; only a
                // function_declarator (possibly under ref/ptr wrappers) is a sig.
                let mut cur = d;
                loop {
                    match cur.kind() {
                        "function_declarator" => return Some(self.sig_from(t, cur)),
                        "reference_declarator" | "pointer_declarator"
                        | "parenthesized_declarator" => {
                            cur = self.f(cur, "declarator")?;
                        }
                        "init_declarator" => {
                            cur = self.f(cur, "declarator")?;
                        }
                        _ => return None,
                    }
                }
            }
            _ => return None,
        };
        self.sig_from(ret_node, decl_node).into()
    }

    fn sig_from(&mut self, ret_node: Option<Node<'a>>, fdecl: Node<'a>) -> FnSig {
        let mut ret = spec_type(self, ret_node);
        let mut ret_is_unit = matches!(ret, CTy::Unit) || ret_node.is_none();
        // `auto f() -> T` trailing return
        if matches!(ret, CTy::Unknown)
            && let Some(tr) = self.f(fdecl, "return_type").or_else(|| self.f(fdecl, "type")) {
                ret = spec_type(self, Some(tr));
            }
        let _ = &mut ret_is_unit;
        let params_node = self.f(fdecl, "parameters");
        let mut params = Vec::new();
        if let Some(pl) = params_node {
            for p in pl.named_children(&mut pl.walk()) {
                match p.kind() {
                    "parameter_declaration" | "optional_parameter_declaration" => {
                        let base = spec_type(self, self.f(p, "type"));
                        let d = declarator(self, self.f(p, "declarator").unwrap_or(p));
                        let ty = self.wrap_type(base.clone(), &d);
                        // `const T&` is a borrow — strip; mutable `T&` on a
                        // scalar becomes an out-param.
                        let out = d.is_ref
                            && !d.is_const
                            && !matches!(
                                ty,
                                CTy::Arr(..) | CTy::Dict(_) | CTy::Set(_) | CTy::Struct(_)
                                    | CTy::Tuple(_)
                            );
                        params.push((d.name, ty, out));
                    }
                    "variadic_parameter_declaration" => {
                        self.diag(DiagKind::Unsupported, p, "variadic `...`");
                    }
                    _ => {}
                }
            }
        }
        FnSig {
            ret,
            params,
            ret_is_unit,
        }
    }

    /// Lower one top-level node.
    fn item(&mut self, node: Node<'a>) {
        match node.kind() {
            "comment" => self.comment(node),
            "preproc_include" | "preproc_call" | "preproc_include_next" => {
                self.diag(
                    DiagKind::Note,
                    node,
                    format!("dropped {}", one_line(self.node_text(node))),
                );
                self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
            }
            "preproc_def" => self.preproc_def(node),
            "preproc_function_def" | "preproc_if" | "preproc_ifdef" | "preproc_else"
            | "preproc_elif" | "preproc_endif" | "preproc_defined" => {
                self.diag(
                    DiagKind::Note,
                    node,
                    format!("dropped {}", one_line(self.node_text(node))),
                );
                self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
            }
            "using_declaration" => {
                let text = self.node_text(node);
                if !text.contains("std") {
                    self.diag(
                        DiagKind::Note,
                        node,
                        format!("`{}` — namespaces are flattened", one_line(text)),
                    );
                }
                self.line(&format!("// cpp: {}", one_line(text)));
            }
            "namespace_definition" => {
                self.diag(
                    DiagKind::Note,
                    node,
                    "namespace contents flattened into the top level",
                );
                if let Some(body) = self.f(node, "body") {
                    for c in self.children(body) {
                        self.item(c);
                    }
                }
            }
            "alias_declaration" | "type_definition" => {
                self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
            }
            "linkage_specification" => {
                // `extern "C" { ... }` — emit contents.
                for c in self.children(node) {
                    if c.kind() == "declaration_list" || c.kind() == "compound_statement" {
                        for g in self.children(c) {
                            self.item(g);
                        }
                    }
                }
            }
            "template_declaration" => {
                self.diag(
                    DiagKind::Approximate,
                    node,
                    "template stripped — single instantiation assumed",
                );
                if let Some(inner) = self.children(node).last() {
                    self.item(*inner);
                }
            }
            "struct_specifier" | "class_specifier" => self.lower_struct(node),
            "enum_specifier" => self.lower_enum(node),
            "function_definition" => self.lower_fn(node, None),
            "declaration" => {
                // prototype or global variable
                if self.is_prototype(node) {
                    self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
                } else {
                    self.decl(node, true);
                }
            }
            "static_assert" | "static_assert_declaration" => {
                self.diag(DiagKind::Note, node, "static_assert dropped");
            }
            "friend_declaration" | "using_static_assert_declaration" => {}
            "expression_statement" | "statement" => self.stmt(node),
            "ERROR" => self.unsupported(node, "parse error"),
            _ => self.unsupported(node, node.kind()),
        }
    }

    fn is_prototype(&mut self, node: Node<'a>) -> bool {
        if node.kind() != "declaration" {
            return false;
        }
        let Some(d) = self.f(node, "declarator") else {
            return false;
        };
        let mut cur = d;
        loop {
            match cur.kind() {
                "function_declarator" => return true,
                "reference_declarator" | "pointer_declarator" | "parenthesized_declarator"
                | "init_declarator" => {
                    let Some(inner) = self.f(cur, "declarator") else {
                        return false;
                    };
                    cur = inner;
                }
                _ => return false,
            }
        }
    }

    /// `#define NAME value` -> `const NAME = value` when the value is a simple
    /// literal/expression; otherwise dropped with a note.
    fn preproc_def(&mut self, node: Node<'a>) {
        let name = self
            .f(node, "name")
            .map(|n| self.node_text(n).to_string())
            .unwrap_or_default();
        if let Some(v) = self.f(node, "value") {
            let init = self.expr(v);
            self.line(&format!("const {name} = {init};"));
        } else {
            self.line(&format!("// cpp: {}", one_line(self.node_text(node))));
        }
    }

    /// `//` and `/* */` comments pass through as mimas `//` lines.
    fn comment(&mut self, node: Node<'a>) {
        for l in self.node_text(node).lines() {
            let l = l.trim().trim_start_matches("/*").trim_start_matches('*').trim_start_matches("//").trim_end_matches("*/").trim();
            self.line(&format!("// {l}"));
        }
    }

    fn lower_enum(&mut self, node: Node<'a>) {
        let name = self.f(node, "name").map(|n| self.node_text(n).to_string());
        let scoped = self.node_text(node).contains("enum class") || self.node_text(node).contains("enum struct");
        let Some(body) = self.f(node, "body") else { return };
        if scoped && let Some(name) = name.clone() {
            // `enum class E { A, B }` -> `enum E { A, B }`
            let mut variants = Vec::new();
            for e in body.named_children(&mut body.walk()) {
                if e.kind() == "enumerator" {
                    let n = self
                        .f(e, "name")
                        .map(|n| self.node_text(n).to_string())
                        .unwrap_or_default();
                    if self.f(e, "value").is_some() {
                        self.diag(DiagKind::Approximate, e, "explicit enum value dropped");
                    }
                    variants.push(n);
                }
            }
            self.line(&format!("enum {name} {{ {} }}", variants.join(", ")));
            return;
        }
        // unscoped enum -> consecutive consts
        let mut next = 0i64;
        for e in body.named_children(&mut body.walk()) {
            if e.kind() != "enumerator" {
                continue;
            }
            let n = self
                .f(e, "name")
                .map(|n| self.node_text(n).to_string())
                .unwrap_or_default();
            if let Some(v) = self.f(e, "value") {
                let init = self.expr(v);
                self.line(&format!("const {n} = {init};"));
                next = init.parse::<i64>().map(|i| i + 1).unwrap_or(0);
            } else {
                self.line(&format!("const {n} = {next};"));
                next += 1;
            }
        }
    }

    /// `struct X { fields; methods; }` -> mimas `struct` + `impl` blocks.
    fn lower_struct(&mut self, node: Node<'a>) {
        let Some(name) = self.f(node, "name").map(|n| self.node_text(n).to_string()) else {
            self.unsupported(node, "anonymous struct");
            return;
        };
        let info = self.structs.get(&name).cloned().unwrap_or_default();
        self.mark(node);
        self.open(&format!("struct {name}"));
        for (fname, fty) in &info.fields {
            let annot = fty.annot().unwrap_or_else(|| "_".into());
            self.line(&format!("{fname}: {annot},"));
        }
        self.close("");
        let prev = self.cur_struct.replace(name.clone());
        // methods + ctors
        if let Some(body) = self.f(node, "body") {
            let mut impl_open = false;
            for member in self.children(body) {
                if member.kind() != "function_definition" {
                    continue;
                }
                let dname = self
                    .f(member, "declarator")
                    .and_then(|d| self.f(d, "declarator"))
                    .map(|d| self.node_text(d).to_string())
                    .unwrap_or_default();
                let dname = unqual(&dname).to_string();
                if dname.starts_with('~') {
                    self.diag(DiagKind::Note, member, "destructor dropped — mimas is GC'd");
                    continue;
                }
                if !impl_open {
                    self.open(&format!("impl {name}"));
                    impl_open = true;
                }
                if dname == name {
                    self.lower_ctor(member, &name);
                } else {
                    self.lower_fn(member, Some(&name));
                }
            }
            if impl_open {
                self.close("");
            }
        }
        self.cur_struct = prev;
    }

    /// `X(args) : f(e), ... { body }` -> `fn new(args) -> Self { let o = Self { ... }; ...; o }`.
    ///
    /// Inside the body, member names resolve to `o.` via `self_name`.
    fn lower_ctor(&mut self, node: Node<'a>, name: &str) {
        let decl = self.f(node, "declarator").unwrap();
        let params = self.f(decl, "parameters");
        let plist = self.param_list(params);
        self.mark(node);
        self.open(&format!("fn new({plist}) -> Self"));
        self.push_scope();
        // member initializer list + declared defaults
        let mut field_exprs: Vec<(String, String)> = Vec::new();
        if let Some(info) = self.structs.get(name) {
            for (fname, fty) in &info.fields {
                let d = info
                    .defaults
                    .get(fname)
                    .cloned()
                    .unwrap_or_else(|| fty.default_init(&self.struct_defaults()));
                field_exprs.push((fname.clone(), d));
            }
        }
        // ctor init list: `X() : f(v), ...` arrives as a field_initializer_list
        // child of the function_definition
        let init_list = self
            .children(node)
            .into_iter()
            .find(|c| c.kind() == "field_initializer_list");
        if let Some(inits) = init_list {
            for init in self.children(inits) {
                if init.kind() != "field_initializer" {
                    continue;
                }
                // `x(1)` — field_identifier + argument_list
                let fname = init
                    .named_child(0)
                    .map(|n| unqual(self.node_text(n)).to_string())
                    .unwrap_or_default();
                let arg = self
                    .children(init)
                    .into_iter()
                    .find(|c| c.kind() == "argument_list")
                    .and_then(|a| {
                        self.children(a)
                            .into_iter()
                            .find(|c| c.kind() != "comment")
                    })
                    .map(|v| self.expr(v))
                    .unwrap_or_else(|| "0".into());
                if let Some(e) = field_exprs.iter_mut().find(|(n, _)| *n == fname) {
                    e.1 = arg;
                } else {
                    field_exprs.push((fname, arg));
                }
            }
        }
        let body_init = field_exprs
            .iter()
            .map(|(n, e)| format!("{n} = {e}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.line(&format!("let o = Self {{ {body_init} }};"));
        let prev_self = std::mem::replace(&mut self.self_name, "o".into());
        if let Some(body) = self.f(node, "body") {
            for s in self.children(body) {
                self.stmt(s);
            }
        }
        self.self_name = prev_self;
        self.line("o");
        self.pop_scope();
        self.close("");
    }

    /// `ret name(params) { body }` -> `fn name(name: ty, ...) -> ret { }`.
    ///
    /// `owner` is the struct name when lowering a method (`self` is prepended
    /// and `T&` out-params still apply).
    pub(crate) fn lower_fn(&mut self, node: Node<'a>, owner: Option<&str>) {
        let decl = self.f(node, "declarator").unwrap();
        let mut dname = self
            .f(decl, "declarator")
            .map(|d| self.node_text(d).to_string())
            .unwrap_or_default();
        dname = unqual(&dname).to_string();
        let sig = if let Some(owner) = owner {
            if self.structs.contains_key(owner) {
                Some(self.sig_from(self.f(node, "type"), decl))
            } else {
                None
            }
        } else {
            self.fns.get(&dname).cloned()
        }
        .unwrap_or_else(|| FnSig {
            ret: CTy::Unit,
            ret_is_unit: true,
            params: Vec::new(),
        });
        let params = self.f(decl, "parameters");
        let mut plist = self.param_list(params);
        if owner.is_some() {
            plist = if plist.is_empty() {
                "self".into()
            } else {
                format!("self, {plist}")
            };
        }
        self.mark(node);
        let ret = sig.mimas_ret().unwrap_or_else(|| "()".into());
        self.open(&format!("fn {dname}({plist}) -> {ret}"));
        self.push_scope();
        for (pname, pty, _) in &sig.params {
            self.bind(pname, pty.clone());
        }
        self.cur_fn = Some(sig.clone());
        if let Some(body) = self.f(node, "body") {
            self.block_stmts(body);
        }
        // out-param epilogue: a void function's fallthrough returns the outs.
        let outs = sig.out_params();
        if !outs.is_empty() && sig.ret_is_unit {
            let names: Vec<String> = outs
                .iter()
                .map(|i| sig.params[*i].0.clone())
                .collect();
            self.line(&format!("return {};", tuple_or_single(&names)));
        } else if !outs.is_empty() {
            self.diag(
                DiagKind::Approximate,
                node,
                "non-void function with `T&` params: fallthrough returns default value",
            );
            let mut parts = vec![sig.ret.default_init(&self.struct_defaults())];
            parts.extend(outs.iter().map(|i| sig.params[*i].0.clone()));
            self.line(&format!("return ({});", parts.join(", ")));
        }
        self.cur_fn = None;
        self.pop_scope();
        self.close("");
        if dname == "main" && owner.is_none() {
            self.has_main = true;
        }
    }

    /// `param_list` node -> `name: ty, name=default` text.
    fn param_list(&mut self, params: Option<Node<'a>>) -> String {
        let mut parts = Vec::new();
        let Some(pl) = params else { return String::new() };
        for p in pl.named_children(&mut pl.walk()) {
            match p.kind() {
                "parameter_declaration" | "optional_parameter_declaration" => {
                    let base = spec_type(self, self.f(p, "type"));
                    let d = declarator(self, self.f(p, "declarator").unwrap_or(p));
                    let ty = self.wrap_type(base, &d);
                    let annot = ty.annot().unwrap_or_else(|| "_".into());
                    if let Some(def) = self.f(p, "default_value") {
                        let dflt = self.expr(def);
                        parts.push(format!("{} = {dflt}", d.name));
                    } else {
                        parts.push(format!("{}: {annot}", d.name));
                    }
                }
                _ => {}
            }
        }
        parts.join(", ")
    }

    /// Lower the statements of a `compound_statement` (no braces emitted).
    pub(crate) fn block_stmts(&mut self, body: Node<'a>) {
        for s in self.children(body) {
            self.mark(s);
            self.stmt(s);
        }
    }
}

fn sig_name<'a>(l: &mut Lower<'a>, node: Node<'a>) -> String {
    let decl = match l.f(node, "declarator") {
        Some(d) => d,
        None => return String::new(),
    };
    let mut cur = decl;
    loop {
        match cur.kind() {
            "function_declarator" | "reference_declarator" | "pointer_declarator"
            | "parenthesized_declarator" | "init_declarator" => {
                let Some(inner) = l.f(cur, "declarator") else { return String::new() };
                cur = inner;
            }
            _ => return unqual(l.node_text(cur)).to_string(),
        }
    }
}

/// `(a, b)` or bare `a` for a single out-param return.
fn tuple_or_single(names: &[String]) -> String {
    if names.len() == 1 {
        names[0].clone()
    } else {
        format!("({})", names.join(", "))
    }
}

/// Split a `declaration` into (base type, [declarators]) — used by both the
/// signature prescan and statement lowering.
pub(crate) fn decl_parts<'n>(
    l: &mut Lower<'n>,
    node: Node<'n>,
) -> Option<(CTy, Vec<(super::cty::Decl<'n>, CTy)>)> {
    if node.kind() != "declaration" && node.kind() != "field_declaration" {
        return None;
    }
    let base = spec_type(l, l.f(node, "type"));
    let mut out = Vec::new();
    for c in node.named_children(&mut node.walk()) {
        match c.kind() {
            "primitive_type" | "sized_type_specifier" | "type_identifier" | "template_type"
            | "qualified_identifier" | "auto" | "placeholder_type_specifier" | "type_qualifier"
            | "storage_class_specifier" | "decltype" | "struct_specifier" | "class_specifier"
            | "enum_specifier" | "union_specifier" | "comment" | "virtual" | "virtual_specifier" => {}
            _ => {
                let d = declarator(l, c);
                let ty = l.wrap_type(base.clone(), &d);
                out.push((d, ty));
            }
        }
    }
    Some((base, out))
}
