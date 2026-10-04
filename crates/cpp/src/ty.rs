//! C++ types, mapped to the mimas type vocabulary.
//!
//! The transpiler does a light, best-effort type analysis: enough to lower
//! `a / b` to `~/` when both sides are `int`-family, to pick the right
//! container method (`v.size()` on a `std::queue` behaves like `std::vector`),
//! and to tell scalar `T&` out-params apart from immutable `const T&` borrows.

/// The sequencing flavour of an `[T]`-mapped C++ container, which decides how
/// member calls like `pop()`/`top()`/`front()` are lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqKind {
    /// `std::vector`, `std::array`, C arrays, `std::deque`.
    Vector,
    /// `std::stack` — push/pop at the back.
    Stack,
    /// `std::queue` — push at the back, pop at the front.
    Queue,
    /// `std::priority_queue` — kept sorted; `top` is the greatest element
    /// (or least when declared with `greater<>`).
    PQueue {
        /// `priority_queue<T, vector<T>, greater<T>>` — a min-heap.
        min: bool,
    },
}

/// A C++ type as far as the transpiler models it.
#[derive(Debug, Clone, PartialEq)]
pub enum CTy {
    /// `void`.
    Unit,
    Bool,
    /// Every integer type, signed or not, including `size_t`.
    Int,
    /// `float`, `double`, `long double`.
    Float,
    /// `std::string`, `char*`, string literals. `char` lowers to `str` too;
    /// see [`Self::Char`] for arithmetic-position chars.
    Str,
    /// A `char` literal or variable used arithmetically — lowered to `ord("c")`
    /// so `'a' + 1` keeps working.
    Char,
    /// A sequence container; `kind` refines member-call lowering.
    Arr(Box<CTy>, SeqKind),
    /// `std::map`/`std::unordered_map`/`std::set` over `str` keys — mimas `~{V}`.
    Dict(Box<CTy>),
    /// `std::set`/`std::unordered_set` (non-str keys emulated by a `[T]` +
    /// dedup; sorted-order iteration is approximated and warned about).
    Set(Box<CTy>),
    /// `std::pair`, `std::tuple`, structured bindings.
    Tuple(Vec<CTy>),
    /// `std::optional<T>`.
    Opt(Box<CTy>),
    /// A user-defined `struct`/`class`.
    Struct(String),
    /// Type we could not or did not map — the annotation is omitted and mimas
    /// infers it.
    Unknown,
}

impl CTy {
    /// The mimas type annotation for this type, or `None` when it should be
    /// left to inference (`auto`, unmapped types).
    pub fn annot(&self) -> Option<String> {
        Some(match self {
            CTy::Unit => "()".into(),
            CTy::Bool => "bool".into(),
            CTy::Int => "int".into(),
            CTy::Float => "float".into(),
            CTy::Str | CTy::Char => "str".into(),
            CTy::Arr(t, _) => format!("[{}]", t.annot()?),
            CTy::Dict(v) => format!("~{{{}}}", v.annot()?),
            CTy::Set(t) => format!("[{}]", t.annot()?),
            CTy::Tuple(ts) => format!(
                "({})",
                ts.iter()
                    .map(|t| t.annot().unwrap_or_else(|| "_".into()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            CTy::Opt(t) => format!("{}?", t.annot()?),
            CTy::Struct(n) => n.clone(),
            CTy::Unknown => return None,
        })
    }

    /// A mimas expression evaluating to the C++ default value of this type.
    /// `struct_defaults` resolves `Struct(name)` via the recorded fields.
    pub fn default_init(&self, structs: &std::collections::HashMap<String, Vec<String>>) -> String {
        match self {
            CTy::Unit => "()".into(),
            CTy::Bool => "false".into(),
            CTy::Int => "0".into(),
            CTy::Float => "0.0".into(),
            CTy::Str | CTy::Char => "\"\"".into(),
            CTy::Arr(..) | CTy::Set(_) => "array::new()".into(),
            CTy::Dict(_) => "~{}".into(),
            CTy::Tuple(ts) => format!(
                "({})",
                ts.iter()
                    .map(|t| t.default_init(structs))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            CTy::Opt(_) => "null".into(),
            CTy::Struct(n) => match structs.get(n) {
                Some(fields) => format!("{} {{ {} }}", n, fields.join(", ")),
                None => format!("{n} {{}}"),
            },
            CTy::Unknown => "0 /* cpp: unknown default */".into(),
        }
    }

    /// Element type for containers, `self` otherwise.
    pub fn elem(&self) -> CTy {
        match self {
            CTy::Arr(t, _) | CTy::Set(t) | CTy::Opt(t) => (**t).clone(),
            _ => self.clone(),
        }
    }

    /// Whether the type behaves like an `int` for `/` → `~/` lowering.
    pub fn is_int(&self) -> bool {
        matches!(self, CTy::Int | CTy::Char)
    }

    /// `T x = <clone of e>` so `vector`/`map`/`set`/struct copies keep C++
    /// value semantics (mimas shares the reference otherwise).
    pub fn copy_expr(&self, e: &str, structs: &std::collections::HashMap<String, Vec<String>>) -> String {
        match self {
            CTy::Arr(..) | CTy::Set(_) => format!("(for __e in {e} collect __e)"),
            CTy::Dict(_) => format!(
                "{{ let __d = ~{{}}; for (__k, __v) in {e} {{ __d[__k] = __v; }} __d }}"
            ),
            CTy::Struct(n) => match structs.get(n) {
                Some(fields) => {
                    let assigns = fields
                        .iter()
                        .map(|f| {
                            let name = f.split(' ').next().unwrap_or(f);
                            format!("{name} = {e}.{name}")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{n} {{ {assigns} }}")
                }
                None => e.to_string(),
            },
            _ => e.to_string(),
        }
    }
}
