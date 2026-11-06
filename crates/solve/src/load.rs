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
/// vocabulary, so `LightDark::Dark` on one page means what it means on the next.
/// Declarations only — a program built without std has it too. Std's
/// opp-taking operations (`StartEnd::End.of(xs)`, `AscDesc::Desc.sort(xs)`) ship
/// with std as library source (`Library::add_source`).
// r[impl mimas.opp.builtin]
pub const BUILTIN_SRC: &str = include_str!("builtin.mim");
/// The built-in source's file name.
pub const BUILTIN_NAME: &str = "<builtin>";

/// Shipped files (the built-in source, the library's own `.mim`) take ids
/// from here up, so a program's own files keep `0..` whatever was prepared
/// before them. Nothing indexes a table by file id past a program's own
/// files; the id only keys `sources` and locates diagnostics.
pub const SHIPPED_FILE_BASE: FileId = 1 << 20;

/// A solver with a library and the shipped sources already taken in —
/// everything a program's solve needs before reading its first file, and
/// the fixed cost of every solve (hoisting a few hundred natives, solving
/// the prelude). Build one per library and `solve` any number of programs
/// from it: each starts from a clone of this state, so programs never see
/// each other. Sans I/O: it reads nothing and touches no global but the
/// thread's unit table and type-name registries, which it installs itself.
#[derive(Clone)]
pub struct Prepared {
    solver: Solver,
    /// `(name, text)` of each shipped file, in id order from `SHIPPED_FILE_BASE`.
    shipped: Vec<(String, String)>,
    shipped_asts: Vec<Ast>,
    shipped_sources: HashMap<FileId, NamedSource<Arc<str>>>,
    /// The unit table as the shipped sources left it; a solve starts from
    /// it, and a program's own `unit` declarations copy it before writing.
    units: std::rc::Rc<shared::units::CustomUnits>,
    /// What the shipped sources themselves warned (nothing, by policy).
    shipped_warnings: Vec<std::rc::Rc<miette::Report>>,
    fingerprint: u64,
}

impl Prepared {
    /// Take in `library` and its shipped sources. `Err` carries the shipped
    /// sources' own parse or solve errors — a broken prelude has no program.
    pub fn new(library: &Library<()>) -> std::result::Result<Self, Vec<Error>> {
        shared::units::clear();
        let shipped: Vec<(String, String)> = std::iter::once((BUILTIN_NAME, BUILTIN_SRC))
            .chain(library.sources().iter().copied())
            .map(|(n, t)| (n.to_string(), t.to_string()))
            .collect();
        let files: Vec<(&str, &str)> = shipped.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
        let inherited = parse::lex::inherited_shadow(&files);
        let mut asts = Vec::new();
        let mut sources = HashMap::new();
        let mut errors = Vec::new();
        for (i, (name, text)) in files.iter().enumerate() {
            let file_id = SHIPPED_FILE_BASE + i;
            sources.insert(file_id, NamedSource::new(*name, Arc::from(*text)));
            let lexer = Lexer::new(text, file_id, name.to_string()).with_inherited(inherited[i]);
            let (ast, parse_errors) = Parser::new_extending(lexer).into_ast();
            errors.extend(parse_errors);
            asts.push(ast);
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let mut solver = Solver::new();
        solver.install_library(library);
        solver.set_sources(sources.clone());
        // r[impl mimas.prelude.shadow]
        // the built-in and library sources solve first, to completion: their
        // impls bind to their own types before a program is even hoisted, so
        // a user `enum PosNeg` shadows the built-in `PosNeg` for user code
        // without catching the built-in `impl Opp for PosNeg` mid-hoist.
        solver.in_prelude = true;
        let prelude = solver.solve_all(asts.iter());
        solver.in_prelude = false;
        prelude.map_err(|e| vec![e])?;
        let shipped_warnings = std::mem::take(&mut solver.warnings);
        Ok(Self {
            solver,
            shipped,
            shipped_asts: asts,
            shipped_sources: sources,
            units: shared::units::snapshot(),
            shipped_warnings,
            fingerprint: library.fingerprint(),
        })
    }

    /// The library this was prepared for, by [`Library::fingerprint`].
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Parse and solve one program's `files` (each a name and its text)
    /// on top of the prepared state. File ids follow the order given; the
    /// shipped asts follow the program's in `asts`, as `load_files` always
    /// returned them. The solver can't take poison yet, so a parse error in
    /// any file skips the solve.
    pub fn solve<'a>(&self, files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Loaded {
        // the program's `unit` declarations build on the shipped table and
        // copy it before writing, so the next program starts clean
        // `MIMAS_SOLVE_TIMING=1`: the solve's fixed costs, to stderr
        let timing = std::env::var_os("MIMAS_SOLVE_TIMING").is_some();
        let mut last = std::time::Instant::now();
        let mut lap = move |step: &str| {
            if timing {
                let now = std::time::Instant::now();
                eprintln!("PREPARED {step}: {:?}", now - last);
                last = now;
            }
        };
        shared::units::install(&self.units);
        let user: Vec<(&str, &str)> = files.into_iter().collect();
        // shadowed operator words: a kit's `fn not` makes `not` a name in the
        // kit and in the files that `use` it, never upstream — computed over
        // the program and the shipped files together, as a page may `use` one
        let all: Vec<(&str, &str)> = user
            .iter()
            .copied()
            .chain(self.shipped.iter().map(|(n, t)| (n.as_str(), t.as_str())))
            .collect();
        let inherited = parse::lex::inherited_shadow(&all);
        let mut asts = Vec::with_capacity(user.len() + self.shipped_asts.len());
        let mut sources = self.shipped_sources.clone();
        let mut errors = Vec::new();
        for (file_id, (name, text)) in user.iter().enumerate() {
            sources.insert(file_id, NamedSource::new(*name, Arc::from(*text)));
            let lexer = Lexer::new(text, file_id, name.to_string()).with_inherited(inherited[file_id]);
            let (ast, parse_errors) = Parser::new_extending(lexer).into_ast();
            errors.extend(parse_errors);
            asts.push(ast);
        }
        lap("parse");
        let mut solver = self.solver.clone();
        lap("clone solver");
        solver.set_sources(sources.clone());
        lap("set_sources");
        if errors.is_empty() {
            if let Err(error) = solver.solve_all(asts.iter()) {
                errors.push(error);
            }
        }
        lap("solve_all");
        let mut warnings = self.shipped_warnings.clone();
        warnings.append(&mut solver.warnings);
        asts.extend(self.shipped_asts.iter().cloned());
        lap("shipped asts");
        Loaded {
            asts,
            sources,
            errors,
            warnings: unwrap_warnings(warnings),
            solver,
        }
    }
}

/// `Rc<Report>`s back to the reports a host renders: a warning minted in
/// this solve is unique and unwraps; one shared with a prepared state is
/// re-rendered from its own text.
pub(crate) fn unwrap_warnings(warnings: Vec<std::rc::Rc<miette::Report>>) -> Vec<miette::Report> {
    warnings
        .into_iter()
        .map(|w| std::rc::Rc::try_unwrap(w).unwrap_or_else(|shared| miette::miette!("{shared:?}")))
        .collect()
}

thread_local! {
    /// The solvers prepared on this thread, by library fingerprint — one
    /// per distinct library (a host has one or two), built on first use.
    static PREPARED: std::cell::RefCell<HashMap<u64, std::rc::Rc<Prepared>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// The thread's prepared solver for `library`, built on first use.
pub fn prepared(library: &Library<()>) -> std::result::Result<std::rc::Rc<Prepared>, Vec<Error>> {
    let key = library.fingerprint();
    if let Some(p) = PREPARED.with(|m| m.borrow().get(&key).cloned()) {
        return Ok(p);
    }
    let p = std::rc::Rc::new(Prepared::new(library)?);
    PREPARED.with(|m| m.borrow_mut().insert(key, p.clone()));
    Ok(p)
}

/// Parses and solves `files` (each a name and its text) against `library`,
/// on the thread's [`prepared`] solver for it. The solver can't take poison
/// yet, so a parse error in any file skips the solve.
pub fn load_files<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
    library: &Library<()>,
) -> Loaded {
    match prepared(library) {
        Ok(p) => p.solve(files),
        Err(errors) => Loaded {
            asts: Vec::new(),
            sources: HashMap::new(),
            errors,
            warnings: Vec::new(),
            solver: Solver::new(),
        },
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
