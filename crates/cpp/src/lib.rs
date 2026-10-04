//! A C++ frontend for mimas: parses a C++ source subset with
//! [tree-sitter](https://tree-sitter.github.io) and emits mimas source that
//! runs through the normal `parse -> solve -> compile -> vm` pipeline.
//!
//! The point of going through source (rather than constructing AST or bytecode
//! directly) is that the generated `.mim` is inspectable, passes through the
//! ordinary type checker, and inherits every host integration mimas programs
//! get — stepping, the variable inspector, `check` lines, scene drawing.
//!
//! # Coverage
//!
//! The supported subset targets algorithm-style C++: functions, structs,
//! `std::vector`/`string`/`map`/`set`/`stack`/`queue`/`priority_queue`,
//! `std::pair`/`tuple`, loops (including range-for), `switch`, references
//! (mutable scalar `T&` params become mimas tuple out-returns), lambdas,
//! `std::cout`, and the usual `<algorithm>`/`<cmath>` calls. Unsupported
//! constructs emit a diagnostic and a `// cpp:` comment in the output instead
//! of failing, so partial programs still run.
//!
//! ```
//! let out = cpp::transpile("int main() { std::cout << 1 + 2; }").unwrap();
//! assert!(out.source.contains("fn main()"));
//! ```

mod lower;
mod ty;

pub use ty::{CTy, SeqKind};

/// How severe a lowering diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagKind {
    /// The construct has no mimas equivalent; code was emitted that will not
    /// preserve C++ semantics (usually a `panic` or a passthrough marker).
    Unsupported,
    /// Lowered with a semantic difference (e.g. unsigned wrap, set ordering).
    Approximate,
    /// Informational — e.g. `#include` dropped, `using namespace` ignored.
    Note,
}

/// One lowering diagnostic, located in the C++ source.
#[derive(Debug, Clone)]
pub struct Diag {
    /// 1-based line in the C++ input.
    pub line: u32,
    /// 1-based column in the C++ input.
    pub col: u32,
    pub kind: DiagKind,
    pub message: String,
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            DiagKind::Unsupported => "unsupported",
            DiagKind::Approximate => "approximate",
            DiagKind::Note => "note",
        };
        write!(f, "{}:{}: {}: {}", self.line, self.col, kind, self.message)
    }
}

/// The result of transpiling a C++ translation unit.
#[derive(Debug)]
pub struct Output {
    /// The generated mimas source.
    pub source: String,
    /// Lowering diagnostics, in source order.
    pub diagnostics: Vec<Diag>,
    /// `(generated line, c++ line)` pairs, 1-based, recorded at each emitted
    /// statement and item — enough for a host to map stepping positions back
    /// to the C++ source.
    pub line_map: Vec<(u32, u32)>,
}

/// Hard failure: the input didn't parse at all.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("C++ parse error at {line}:{col}")]
    Parse {
        /// 1-based line of the first error node.
        line: u32,
        /// 1-based column of the first error node.
        col: u32,
    },
}

/// Transpile a C++ source file to mimas.
///
/// Succeeds on any input tree-sitter parses (errors inside the tree are
/// tolerated — they lower to diagnostics + `panic` stubs). Fails only when the
/// tree itself is broken.
pub fn transpile(source: &str) -> Result<Output, Error> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .expect("tree-sitter-cpp should load");
    let tree = parser.parse(source, None).ok_or(Error::Parse {
        line: 0,
        col: 0,
    })?;
    let root = tree.root_node();
    if root.has_error() {
        // find the shallowest error for a decent position
        let mut cur = root.walk();
        let mut err = root;
        'outer: loop {
            for child in err.children(&mut cur) {
                if child.is_error() || child.is_missing() {
                    err = child;
                    continue 'outer;
                }
                if child.has_error() {
                    err = child;
                    if !err.is_error() {
                        continue 'outer;
                    }
                }
            }
            break;
        }
        // Still produce output for the parseable parts? For now treat a broken
        // tree as fatal: solve errors on the generated text would be noise.
        return Err(Error::Parse {
            line: err.start_position().row as u32 + 1,
            col: err.start_position().column as u32 + 1,
        });
    }
    let mut l = lower::Lower::new(source);
    l.translation_unit(root);
    Ok(l.finish())
}

/// Emit a helper for diagnostics during lowering.
impl Diag {
    pub(crate) fn new(kind: DiagKind, node: tree_sitter::Node, message: String) -> Self {
        let p = node.start_position();
        Diag {
            line: p.row as u32 + 1,
            col: p.column as u32 + 1,
            kind,
            message,
        }
    }
}

