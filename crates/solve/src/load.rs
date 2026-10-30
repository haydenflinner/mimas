use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use api::Library;
use miette::NamedSource;
use parse::{Ast, Parser, lex::Lexer};
use shared::FileId;
use walkdir::WalkDir;

use crate::{Error, Solver};

/// The files of one program, parsed and solved together. File ids follow the order the files
/// were given in.
pub struct Loaded {
    pub asts: Vec<Ast>,
    pub sources: HashMap<FileId, NamedSource<Arc<str>>>,
    pub errors: Vec<Error>,
    /// Severity-warning diagnostics from the grades pass's lint suite -- the solve
    /// succeeded, these are for the host to render (or gate page eval on).
    pub warnings: Vec<miette::Report>,
    pub solver: Solver,
}

/// What every program has, in mimas source. `Opp` is the pact an `opp`
/// declaration implements: two ends and a flip — `!x` is `x.inv()`, the
/// other end, and `!(!x) == x`. `bool` is the first opp (its `!` is
/// built in; `impl Opp for bool` waits on impls for primitive types); the
/// rest are the pairs pages kept spelling as bools or ±1 — a shared
/// vocabulary, so `Size::Big` on one page means what it means on the next.
pub const BUILTIN_SRC: &str = "\
pact Opp {
    fn inv(self) -> Self;
}
opp Sign { Pos, Neg }
opp Size { Big, Small }
opp Shade { Light, Dark }
opp Daytime { Day, Night }
opp Flow { Push, Pull }
opp Fill { Full, Empty }
";
/// The built-in source's file name.
pub const BUILTIN_NAME: &str = "<builtin>";

/// Parses and solves `files` (each a name and its text) against `library`. The solver can't
/// take poison yet, so a parse error in any file skips the solve.
pub fn load_files<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
    library: &Library<()>,
) -> Loaded {
    let mut asts = Vec::new();
    let mut sources = HashMap::new();
    let mut errors = Vec::new();
    // One `unit` table per program: clear once, then every file parses
    // into the same table so a `unit tile` in the first file lexes `4tile`
    // in the second.
    shared::units::clear();
    // and per-file shadowed operator words: a kit's `fn not` makes `not` a
    // name in the kit and in the files that `use` it, never upstream
    let mut files: Vec<(&str, &str)> = files.into_iter().collect();
    // every program has the built-in source: last, so user files keep
    // their file ids (the first is still 0)
    files.push((BUILTIN_NAME, BUILTIN_SRC));
    let shadow = parse::lex::shadow_sets(&files);
    for (file_id, (name, text)) in files.into_iter().enumerate() {
        sources.insert(file_id, NamedSource::new(name, Arc::from(text)));
        let lexer = Lexer::new(text, file_id, name.to_owned()).with_shadowed(shadow[file_id]);
        let (ast, parse_errors) = Parser::new_extending(lexer).into_ast();
        errors.extend(parse_errors);
        asts.push(ast);
    }

    let mut solver = Solver::new();
    solver.install_library(library);
    solver.set_sources(sources.clone());
    if errors.is_empty()
        && let Err(error) = solver.solve_all(&asts)
    {
        errors.push(error);
    }
    let warnings = std::mem::take(&mut solver.warnings);

    Loaded {
        asts,
        sources,
        errors,
        warnings,
        solver,
    }
}

/// Every `.mim` file under `root`, sorted by path, with whatever the walk couldn't read.
pub fn mim_files(root: &Path) -> (Vec<PathBuf>, Vec<std::io::Error>) {
    let mut files = Vec::new();
    let mut errors = Vec::new();
    for entry in WalkDir::new(root) {
        match entry {
            Ok(entry) => {
                let is_mim = entry.file_type().is_file()
                    && entry.path().extension().and_then(|e| e.to_str()) == Some("mim");
                if is_mim {
                    files.push(entry.into_path());
                }
            }
            Err(e) => errors.push(std::io::Error::other(format!("{}: {e}", root.display()))),
        }
    }
    files.sort();
    (files, errors)
}
