//! C++ type specifier + declarator parsing into [`CTy`].

use tree_sitter::Node;

use super::Lower;
use crate::{DiagKind, ty::CTy, ty::SeqKind};

/// A declaration's declarator, unwrapped:
/// `const vector<int>& v = xs` -> name `v`, ref, init `xs`.
#[derive(Debug)]
pub(crate) struct Decl<'n> {
    pub name: String,
    /// `&` or `&&` around the declarator.
    pub is_ref: bool,
    /// `const T&` / `const T` — immutable borrow, value semantics in mimas.
    pub is_const: bool,
    /// `*` — unsupported, diagnosed.
    pub is_ptr: bool,
    /// `T a[N]` dims, outermost last (`int g[3][4]` -> `[3, 4]`... stored in
    /// C++ order; the caller wraps `Arr` inside-out).
    pub dims: Vec<Node<'n>>,
    /// `= x` / `(x)` / `{x}` initializer.
    pub init: Option<Node<'n>>,
    /// Which brace an init used, to tell `T x(e)` apart from `T x{e}`.
    pub init_is_paren: bool,
    /// `auto [a, b] = t` — the bound names.
    pub binding_names: Option<Vec<String>>,
    /// Function declarator (parameter_list) — means this declares a function.
    pub fn_params: Option<Node<'n>>,
}

/// Unwrap the declarator chain of a `declaration`/`parameter_declaration`.
pub(crate) fn declarator<'n>(l: &mut Lower<'n>, mut node: Node<'n>) -> Decl<'n> {
    let mut d = Decl {
        name: String::new(),
        is_ref: false,
        is_const: false,
        is_ptr: false,
        dims: Vec::new(),
        init: None,
        init_is_paren: false,
        binding_names: None,
        fn_params: None,
    };
    // walk inside-out: init_declarator wraps the inner declarator + value
    loop {
        match node.kind() {
            "init_declarator" => {
                let value = l.f(node, "value");
                if let Some(v) = value {
                    // `T x(a, b)` — value is an `argument_list`; `T x = e` is
                    // the expr itself
                    if v.kind() == "argument_list" {
                        d.init_is_paren = true;
                    }
                    d.init = Some(v);
                }
                if let Some(inner) = l.f(node, "declarator") {
                    node = inner;
                    continue;
                }
                break;
            }
            "reference_declarator" => {
                d.is_ref = true;
                if let Some(inner) = l
                    .f(node, "declarator")
                    .or_else(|| node.named_child(0))
                {
                    node = inner;
                    continue;
                }
                break;
            }
            "pointer_declarator" => {
                d.is_ptr = true;
                if let Some(inner) = l
                    .f(node, "declarator")
                    .or_else(|| node.named_child(0))
                {
                    node = inner;
                    continue;
                }
                break;
            }
            "array_declarator" => {
                if let Some(size) = l.f(node, "size") {
                    d.dims.push(size);
                }
                if let Some(inner) = l
                    .f(node, "declarator")
                    .or_else(|| node.named_child(0))
                {
                    node = inner;
                    continue;
                }
                break;
            }
            "parenthesized_declarator" | "attributed_declarator" => {
                if let Some(inner) = l
                    .f(node, "declarator")
                    .or_else(|| node.named_child(0))
                {
                    node = inner;
                    continue;
                }
                break;
            }
            "structured_binding_declarator" => {
                let names = node
                    .named_children(&mut node.walk())
                    .map(|c| l.node_text(c).to_string())
                    .collect();
                d.binding_names = Some(names);
                break;
            }
            "abstract_reference_declarator" | "abstract_pointer_declarator" => {
                // unnamed `int&` / `int*` — just the ref-ness matters
                if node.kind().contains("reference") {
                    d.is_ref = true;
                } else {
                    d.is_ptr = true;
                }
                break;
            }
            "function_declarator" => {
                // `T f(args)` — a function declarator. Also `T x(a)` where x is a
                // variable parses as function_declarator; callers distinguish.
                d.fn_params = l.f(node, "parameters");
                if let Some(inner) = l.f(node, "declarator") {
                    node = inner;
                    continue;
                }
                break;
            }
            "identifier" | "field_identifier" | "type_identifier" | "destructor_name"
            | "qualified_identifier" => {
                d.name = unqual(l.node_text(node)).to_string();
                break;
            }
            "operator_name" => {
                d.name = l.node_text(node).to_string();
                break;
            }
            _ => {
                // abstract declarators (unnamed) and anything else — take text.
                let t = l.node_text(node);
                if !t.is_empty() {
                    d.name = unqual(t).to_string();
                }
                break;
            }
        }
    }
    // `const` anywhere on the declarator/type side marks it immutable for our
    // purposes (the `type` node's qualifier is checked by the caller too).
    let text = l.node_text(node);
    d.is_const = text.starts_with("const ") || text.contains(" const");
    if d.name.is_empty() {
        d.name = "_".into();
    }
    d
}

/// Strip a leading `std::`/namespace scope from a name.
pub(crate) fn unqual(name: &str) -> &str {
    name.rsplit("::").next().unwrap_or(name)
}

/// Map a type-specifier node (`primitive_type`, `type_identifier`,
/// `template_type`, ...) to a [`CTy`].
pub(crate) fn spec_type<'n>(l: &mut Lower<'n>, node: Option<Node<'n>>) -> CTy {
    let Some(node) = node else { return CTy::Unknown };
    match node.kind() {
        "auto" | "placeholder_type_specifier" => CTy::Unknown,
        "primitive_type" | "sized_type_specifier" => prim(l.node_text(node), l, node),
        "type_identifier" => named(l, l.node_text(node), node),
        "qualified_identifier" => {
            let name = l
                .f(node, "name")
                .map(|n| l.node_text(n))
                .unwrap_or_else(|| l.node_text(node));
            let scope = l.f(node, "scope").map(|n| l.node_text(n)).unwrap_or("");
            if scope != "std" && !scope.is_empty() {
                l.diag(
                    DiagKind::Note,
                    node,
                    format!("namespace `{scope}::` flattened"),
                );
            }
            named(l, name, node)
        }
        "template_type" => {
            let raw_name = l
                .f(node, "name")
                .map(|n| l.node_text(n))
                .unwrap_or("?");
            let name = unqual(raw_name);
            let args: Vec<CTy> = l
                .f(node, "arguments")
                .map(|args| {
                    args.named_children(&mut args.walk())
                        .map(|a| template_arg(l, a))
                        .collect()
                })
                .unwrap_or_default();
            template(l, node, name, args)
        }
        "type_descriptor" => {
            // wraps the real specifier (e.g. inside template_argument_list)
            spec_type(l, l.f(node, "type"))
        }
        "decltype" | "dependent_type" => {
            l.diag(DiagKind::Approximate, node, "dependent type erased");
            CTy::Unknown
        }
        "optional_type_declaration" | "enum_specifier" => named(l, l.node_text(node), node),
        "struct_specifier" | "class_specifier" => {
            // inline `struct X {...} v;` — the specifier carries a body; the item
            // pass will have recorded it already.
            match l.f(node, "name") {
                Some(n) => CTy::Struct(l.node_text(n).to_string()),
                None => CTy::Unknown,
            }
        }
        _ => named(l, l.node_text(node), node),
    }
}

fn prim<'a>(text: &str, l: &mut Lower<'a>, node: Node<'a>) -> CTy {
    let t: String = text.split_whitespace().collect();
    let t = t.as_str();
    if t.contains("double") || t.contains("float") || t.contains("longdouble") {
        CTy::Float
    } else if t.contains("bool") {
        CTy::Bool
    } else if t == "void" {
        CTy::Unit
    } else if t.contains("char") {
        CTy::Char
    } else if t.contains("unsigned") {
        l.diag(
            DiagKind::Approximate,
            node,
            "unsigned type mapped to int — C++ wrap-around is not preserved",
        );
        CTy::Int
    } else {
        CTy::Int
    }
}

fn named<'a>(l: &mut Lower<'a>, name: &str, node: Node<'a>) -> CTy {
    match unqual(name) {
        "string" | "string_view" | "wstring" | "u8string" => CTy::Str,
        "size_t" | "ssize_t" | "ptrdiff_t" | "intptr_t" | "uintptr_t" | "int8_t" | "int16_t"
        | "int32_t" | "int64_t" | "uint8_t" | "uint16_t" | "uint32_t" | "uint64_t"
        | "time_t" => CTy::Int,
        "auto" => CTy::Unknown,
        other => {
            if let Some(super::Alias::Ty(t)) = l.aliases.get(other) {
                return t.clone();
            }
            if l.structs.contains_key(other) {
                CTy::Struct(other.to_string())
            } else {
                // Unknown user type — keep the name so mimas either resolves a
                // struct declared later or errors clearly.
                let _ = node;
                CTy::Struct(other.to_string())
            }
        }
    }
}

/// One `template_argument_list` child: a type, or a value (`array<int, 5>`).
fn template_arg<'n>(l: &mut Lower<'n>, node: Node<'n>) -> CTy {
    match node.kind() {
        "number_literal" | "identifier" => CTy::Unknown,
        "type_descriptor" | "template_type" | "type_identifier" | "primitive_type"
        | "sized_type_specifier" | "qualified_identifier" | "auto"
        | "placeholder_type_specifier" => spec_type(l, Some(node)),
        _ => CTy::Unknown,
    }
}

fn template<'a>(l: &mut Lower<'a>, node: Node<'a>, name: &str, args: Vec<CTy>) -> CTy {
    match name {
        "vector" | "array" | "deque" | "list" | "initializer_list" | "valarray" => {
            CTy::Arr(Box::new(args.first().cloned().unwrap_or(CTy::Unknown)), SeqKind::Vector)
        }
        "stack" => CTy::Arr(
            Box::new(args.first().cloned().unwrap_or(CTy::Unknown)),
            SeqKind::Stack,
        ),
        "queue" => CTy::Arr(
            Box::new(args.first().cloned().unwrap_or(CTy::Unknown)),
            SeqKind::Queue,
        ),
        "priority_queue" => {
            // `greater<>` in the third template arg flips the heap direction.
            let min = node_text_contains(l, node, "greater");
            CTy::Arr(
                Box::new(args.first().cloned().unwrap_or(CTy::Unknown)),
                SeqKind::PQueue { min },
            )
        }
        "set" | "unordered_set" | "multiset" | "unordered_multiset" => CTy::Set(Box::new(
            args.first().cloned().unwrap_or(CTy::Unknown),
        )),
        "map" | "unordered_map" | "multimap" | "unordered_multimap" => {
            let key_ok = matches!(args.first(), Some(CTy::Str) | Some(CTy::Char) | None);
            if !key_ok {
                l.diag(
                    DiagKind::Unsupported,
                    node,
                    "map with non-string key — mimas dicts are str-keyed",
                );
                return CTy::Unknown;
            }
            CTy::Dict(Box::new(args.get(1).cloned().unwrap_or(CTy::Unknown)))
        }
        "pair" => CTy::Tuple(vec![
            args.first().cloned().unwrap_or(CTy::Unknown),
            args.get(1).cloned().unwrap_or(CTy::Unknown),
        ]),
        "tuple" => CTy::Tuple(args),
        "optional" | "expected" => CTy::Opt(Box::new(args.first().cloned().unwrap_or(CTy::Unknown))),
        "shared_ptr" | "unique_ptr" | "weak_ptr" => {
            l.diag(
                DiagKind::Approximate,
                node,
                "smart pointer erased to the pointee type — mimas is GC'd",
            );
            args.first().cloned().unwrap_or(CTy::Unknown)
        }
        "function" => {
            l.diag(DiagKind::Unsupported, node, "std::function type");
            CTy::Unknown
        }
        _ => {
            // User template class — no generics in mimas.
            l.diag(
                DiagKind::Approximate,
                node,
                format!("template type `{name}<..>` used as a struct of that name"),
            );
            CTy::Struct(name.to_string())
        }
    }
}

fn node_text_contains<'a>(l: &Lower<'a>, node: Node<'a>, needle: &str) -> bool {
    l.node_text(node).contains(needle)
}
