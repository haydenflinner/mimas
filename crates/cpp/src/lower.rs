//! tree-sitter-cpp AST -> mimas source lowering.
//!
//! The lowerer is deliberately statement-at-a-time and source-producing: every
//! C++ statement becomes one or more mimas lines, and each emitted statement
//! records a `(generated line, c++ line)` pair in [`crate::Output::line_map`]
//! so a host can map debugger positions back to the C++ file.

use std::collections::HashMap;

use tree_sitter::Node;

use crate::{Diag, DiagKind, Output, ty::CTy};

mod cty;
mod expr;
mod item;
mod stmt;

/// A lowered function signature — needed at call sites for the `T&`
/// out-param desugar before bodies are lowered.
#[derive(Debug, Clone)]
pub(crate) struct FnSig {
    pub ret: CTy,
    /// `(name, type, is_out_param)`.
    pub params: Vec<(String, CTy, bool)>,
    /// Whether `ret` was explicitly `void`/omitted. Non-void + out params make
    /// the mimas return type a tuple `(ret, outs...)`.
    pub ret_is_unit: bool,
}

impl FnSig {
    /// Indices of params that are mutable scalar `T&` (emit as extra returns).
    pub fn out_params(&self) -> Vec<usize> {
        self.params
            .iter()
            .enumerate()
            .filter(|(_, (_, _, out))| *out)
            .map(|(i, _)| i)
            .collect()
    }

    /// The mimas return-type annotation for this signature.
    pub fn mimas_ret(&self) -> Option<String> {
        let outs = self.out_params();
        if outs.is_empty() {
            return self.ret.annot().or(Some("()".into()));
        }
        let mut parts: Vec<CTy> = Vec::new();
        if !self.ret_is_unit {
            parts.push(self.ret.clone());
        }
        parts.extend(outs.iter().map(|i| self.params[*i].1.clone()));
        if parts.len() == 1 {
            return parts[0].annot().or(Some("()".into()));
        }
        Some(format!(
            "({})",
            parts
                .iter()
                .map(|t| t.annot().unwrap_or_else(|| "_".into()))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// What we learned about a `struct`/`class` during the pre-pass.
#[derive(Debug, Default, Clone)]
pub(crate) struct StructInfo {
    /// `(name, type)` in declaration order.
    pub fields: Vec<(String, CTy)>,
    /// `name -> default mimas expr` for C++ default member initializers.
    pub defaults: HashMap<String, String>,
    /// Method names (for `this`-less member calls inside methods).
    pub methods: Vec<String>,
    /// Any constructor declared.
    pub has_ctor: bool,
}

/// A namespace alias — `namespace fs = std::filesystem` — or a `using` alias
/// for a type.
#[derive(Debug)]
pub(crate) enum Alias {
    /// Type alias (`using ll = long long`, `typedef int T`).
    Ty(CTy),
}

pub(crate) struct Lower<'a> {
    pub src: &'a str,
    out: String,
    indent: usize,
    pub diags: Vec<Diag>,
    pub line_map: Vec<(u32, u32)>,
    /// Fresh-name counter for `__cpp` temporaries.
    tmp: usize,
    /// Lexical scopes of variable types, innermost last.
    pub scopes: Vec<HashMap<String, CTy>>,
    /// All functions seen in the signature pre-pass.
    pub fns: HashMap<String, FnSig>,
    pub structs: HashMap<String, StructInfo>,
    pub aliases: HashMap<String, Alias>,
    /// The function being lowered: its out-param names for `return` rewrites.
    pub cur_fn: Option<FnSig>,
    /// The struct whose method is being lowered (member names -> `self.`).
    pub cur_struct: Option<String>,
    /// What `self` is called in the current context — `o` inside a
    /// constructor (which builds a fresh `Self`), `self` in methods.
    pub self_name: String,
    /// Emitted `fn main()` exists — emit a trailing `main();` call.
    has_main: bool,
}

impl<'a> Lower<'a> {
    pub fn new(src: &'a str) -> Self {
        Lower {
            src,
            out: String::new(),
            indent: 0,
            diags: Vec::new(),
            line_map: Vec::new(),
            tmp: 0,
            scopes: vec![HashMap::new()],
            fns: HashMap::new(),
            structs: HashMap::new(),
            aliases: HashMap::new(),
            cur_fn: None,
            cur_struct: None,
            self_name: "self".into(),
            has_main: false,
        }
    }

    pub fn finish(self) -> Output {
        let mut out = self.out;
        if self.has_main {
            out.push_str("main();\n");
        }
        Output {
            source: out,
            diagnostics: self.diags,
            line_map: self.line_map,
        }
    }

    // ---------- emission ----------

    pub fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("__cpp{}", self.tmp)
    }

    /// Current generated line number (1-based).
    fn gen_line(&self) -> u32 {
        self.out.matches('\n').count() as u32 + 1
    }

    /// Record `cpp_node` -> current generated line for the host srcmap.
    pub fn mark(&mut self, node: Node<'a>) {
        self.line_map
            .push((self.gen_line(), node.start_position().row as u32 + 1));
    }

    pub fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// `text {` then indent.
    pub fn open(&mut self, text: &str) {
        self.line(&format!("{text} {{"));
        self.indent += 1;
    }

    /// `}` (dedent). `suffix` appends e.g. ` else {`.
    pub fn close(&mut self, suffix: &str) {
        self.indent = self.indent.saturating_sub(1);
        self.line(&format!("}}{suffix}"));
    }

    /// Emit a diagnostic and (for statement position) a visible comment.
    pub fn diag(&mut self, kind: DiagKind, node: Node<'a>, msg: impl Into<String>) {
        self.diags.push(Diag::new(kind, node, msg.into()));
    }

    /// Unsupported construct: comment + diagnostic, no code.
    pub fn unsupported(&mut self, node: Node<'a>, what: &str) {
        self.diag(DiagKind::Unsupported, node, format!("{what} not supported"));
        self.mark(node);
        self.line(&format!(
            "// cpp: unsupported {}: {}",
            what,
            one_line(self.node_text(node))
        ));
    }

    // ---------- node helpers ----------

    /// Source text of a node.
    pub fn node_text(&self, node: Node<'a>) -> &'a str {
        node.utf8_text(self.src.as_bytes()).unwrap_or("")
    }

    /// `child_by_field_name` shorthand.
    pub fn f<'n>(&self, node: Node<'n>, name: &str) -> Option<Node<'n>> {
        node.child_by_field_name(name)
    }

    /// Iterate named children of a field that may repeat.
    pub fn children<'n>(&self, node: Node<'n>) -> Vec<Node<'n>> {
        node.named_children(&mut node.walk()).collect()
    }

    /// First named child, skipping `comment`s.
    pub fn first<'n>(&self, node: Node<'n>) -> Option<Node<'n>> {
        self.children(node).into_iter().next()
    }

    /// Register a variable's type in the current scope.
    pub fn bind(&mut self, name: &str, ty: CTy) {
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&CTy> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// `struct` field names+defaults for `CTy::default_init`.
    pub fn struct_defaults(&self) -> HashMap<String, Vec<String>> {
        self.structs
            .iter()
            .map(|(name, info)| {
                let fields = info
                    .fields
                    .iter()
                    .map(|(fname, fty)| {
                        let d = info
                            .defaults
                            .get(fname)
                            .cloned()
                            .unwrap_or_else(|| fty.default_init(&HashMap::new()));
                        format!("{fname} = {d}")
                    })
                    .collect::<Vec<_>>();
                (name.clone(), fields)
            })
            .collect()
    }
}

/// Collapse a node to a single line for embedding in comments.
pub(crate) fn one_line(s: &str) -> String {
    let s = s.trim();
    if let Some(i) = s.find('\n') { &s[..i] } else { s }.to_string()
}
