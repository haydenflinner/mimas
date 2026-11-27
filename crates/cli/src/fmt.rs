//! `mimas fmt` — the canonical formatter.
//!
//! A token-stream respacer, not an AST reprint: the lexer keeps comments
//! as tokens, so every comment survives verbatim and each token is emitted
//! from its own source slice (numbers like `1e-6`, `#1.10`, `1_000` keep
//! their written form). Line structure is preserved — mimas's semicolon
//! insertion makes newlines semantic (`c\n= a` is not an assignment), so
//! the formatter only ever rewrites whitespace: gaps between tokens on
//! one line, indentation, blank runs, trailing whitespace, the final
//! newline.
//!
//! Spacing is semantic in a few places and is *preserved*, not
//! normalized, wherever it can change the parse:
//!
//! * `x!`, `x?`, `x%` postfix forms only apply when glued (`x !` parses
//!   differently), so the gap next to `!`/`?`/`%` keeps the original
//!   adjacency.
//! * `[a -1]` is two elements where `a - 1` is subtraction — a `-` glued
//!   to a number on its right but not its left keeps that shape.
//! * `check (x)` is a check of `(x)` but `check(x)` is a call — the gap
//!   after `check` before `(`/`[`/`-` keeps the original adjacency.
//! * `<`/`>`/`>>` keep source adjacency: `Vec<int>` glued stays glued,
//!   `a < b` spaced stays spaced (a token-level formatter can't tell a
//!   generic argument list from a comparison).
//!
//! Word and glyph spellings canonicalize through each token's `Display`:
//! `and`→`&&`, `or`→`||`, `not`→`!`, `≠`→`!=`, `≤`→`<=`, `⊻`→`⊕`,
//! `~=`→`≈`, `·`→`⋅` (the lexer itself calls the glyphs canonical).
//!
//! Layout rules, taken from the shipped `.mim`/`.mimas` corpus:
//!
//! * 4-space indents, one level per open `(`, `[`, `{`, `?[`; a line
//!   opening with closers dedents by that many; a line that can't start
//!   a statement (`&&`, `.`, `in`, `=>`, `=`, …) takes one
//!   continuation level, and a line the author over-indented keeps one.
//! * `{` spaces around except dict-*type* braces and `use` paths —
//!   `x: {int}`, `-> {Nd}`, `({bool}, …)`, `Vec<{int}>`,
//!   `use a::{b, c}` stay tight; records `{ a: 1 }`, blocks, match and
//!   enum arms space.
//! * `table {}` rows are glued cell runs, re-aligned into columns with a
//!   4-space gutter; `query {}` keeps its written cell runs;
//!   `datalog {}` is ordinary code.
//! * `macro` decls glue `$x:expr` fragments and `),*`/`)*`/`)+`/`)?`
//!   repeats; `unit` specs glue `m/s`-style operators (but `^`, `-`, `!`
//!   stay spaced — `s^2` would lex as one identifier).
//! * Blank runs collapse to one line; blanks right after an opener line
//!   or before a closer line drop; one trailing newline.
//!
//! Two more rules sit on top of that respacing:
//!
//! * **Opt-out regions** — `// fmt: off` suspends formatting until a
//!   `// fmt: on` comment; between the two the source is emitted
//!   verbatim, indentation and blank lines included. `// fmt: skip`
//!   exempts the one statement or item after it (the whole thing when
//!   it carries a `{…}` body).
//! * **Columnar alignment** — consecutive lines of the same shape and
//!   indent pad a shared pivot into a column: `=` in `let`/`const`/
//!   assignment runs, values after `:` in field lists, `=>` in match
//!   arms, `->` in `fn` signature lines, and cells of comma-separated
//!   rows. A group ends at a blank line, a comment-only line, an
//!   item, or a shape change, and is dropped whole when its column
//!   would pass `MAX_ALIGN_COL` or a member needs more than
//!   `MAX_ALIGN_PAD` spaces. Emitting is two passes: a canonical
//!   single-space pass measures each pivot's column, the second pass
//!   writes the pads — so `fmt(fmt(x)) == fmt(x)` holds by
//!   construction.
use parse::{
    Parser,
    lex::{Lexer, TokKind},
};
use std::io::Read as _;
use std::path::{Path, PathBuf};

/// Spaces per indent level; the shipped corpus is uniformly 4.
const LEVEL: usize = 4;

/// Gutter between `table {}` columns (widest cell + this).
const TABLE_GUTTER: usize = 4;

/// An alignment column further right than this is dropped: past it the
/// padding costs more reading than it saves.
const MAX_ALIGN_COL: usize = 80;

/// The most padding one group member may need before the group is left
/// alone — a badly overlong member splits the eye, not helps it.
const MAX_ALIGN_PAD: usize = 32;

/// `mimas fmt` entry point: files/dirs to rewrite, or `-`/nothing for
/// stdin → stdout.
pub fn run(paths: &[PathBuf], check: bool) -> i32 {
    if paths.is_empty() || paths.iter().any(|p| p.as_os_str() == "-") {
        return run_stdio(check);
    }
    let mut files = vec![];
    let mut failures = 0;
    for path in paths {
        collect(path, &mut files, &mut failures);
    }
    files.sort();
    files.dedup();
    let mut changed = 0usize;
    for file in &files {
        let src = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("fmt: {}: {e}", file.display());
                failures += 1;
                continue;
            }
        };
        match format_source(&src, &file.to_string_lossy()) {
            Err(_) => {
                eprintln!("fmt: {}: skipped (doesn't parse)", file.display());
                failures += 1;
            }
            Ok(out) if out == src => {}
            Ok(out) => {
                changed += 1;
                println!("{}", file.display());
                if !check && let Err(e) = std::fs::write(file, &out) {
                    eprintln!("fmt: {}: {e}", file.display());
                    failures += 1;
                }
            }
        }
    }
    if failures > 0 {
        2
    } else if check && changed > 0 {
        1
    } else {
        0
    }
}

/// `.mim`/`.mimas` files under a dir, or the file itself.
fn collect(path: &Path, files: &mut Vec<PathBuf>, failures: &mut usize) {
    if path.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    if path.is_dir() {
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            match std::fs::read_dir(&dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.is_dir() {
                            stack.push(p);
                        } else if matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("mim" | "mimas")
                        ) {
                            files.push(p);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("fmt: {}: {e}", dir.display());
                    *failures += 1;
                }
            }
        }
        return;
    }
    eprintln!("fmt: {}: no such file or directory", path.display());
    *failures += 1;
}

/// stdin → stdout, for editor integration.
fn run_stdio(check: bool) -> i32 {
    let mut src = String::new();
    if std::io::stdin().read_to_string(&mut src).is_err() {
        eprintln!("fmt: could not read stdin");
        return 2;
    }
    match format_source(&src, "<stdin>") {
        Ok(out) => {
            if check {
                i32::from(out != src)
            } else {
                print!("{out}");
                0
            }
        }
        Err(_) => {
            eprintln!("fmt: stdin doesn't parse");
            2
        }
    }
}

/// Format one source to canonical text. `Err(n)` when the source has `n`
/// diagnostics — a file that doesn't parse is left alone.
// r[impl mimas.fmt]
pub fn format_source(src: &str, file_name: &str) -> Result<String, usize> {
    // Parse first: it runs the unit-decl prescan, so the relex below
    // reads page-declared quantities (`18tile`) the same way. Any
    // diagnostic is a reason to keep hands off.
    let (_, diags) = Parser::new(Lexer::new(src, 0, file_name.into())).into_ast();
    if !diags.is_empty() {
        return Err(diags.len());
    }
    let mut lexer = Lexer::new(src, 0, file_name.into());
    let mut toks = Vec::new();
    for tok in lexer.by_ref() {
        let span = tok.location.span();
        toks.push(T {
            kind: tok.kind,
            start: span.start(),
            end: span.end(),
        });
    }
    if !lexer.take_errors().is_empty() {
        return Err(1);
    }
    let f = Fmt::new(src, toks);
    // `class`/`record` in decl position relex as `struct` after the
    // rewrite — their starts normalize the comparison, like `not`→`!`.
    let aliases: std::collections::HashSet<usize> = (0..f.toks.len())
        .filter(|&i| f.alias_kw[i])
        .map(|i| f.toks[i].start)
        .collect();
    let out = f.finish();
    if !same_token_stream(src, &out, file_name, &aliases) {
        // The respacer only writes whitespace — a token-stream diff means a
        // formatter bug; refuse to touch the file rather than risk it.
        return Err(1);
    }
    Ok(out)
}

/// The semantic guard: the formatted text must relex to the same
/// non-comment token stream (`and`→`&&`, `·`→`⋅` and friends are the
/// lexer's own canonical spellings, so kinds compare equal across the
/// rewrite; `not` respells to `!`, and a `class`/`record` whose start
/// sits in `aliases` respells to `struct`).
fn same_token_stream(
    before: &str,
    after: &str,
    file_name: &str,
    aliases: &std::collections::HashSet<usize>,
) -> bool {
    fn kinds<'a>(
        src: &'a str,
        file_name: &str,
        aliases: &std::collections::HashSet<usize>,
    ) -> Option<Vec<TokKind<'a>>> {
        let mut lexer = Lexer::new(src, 0, file_name.into());
        let mut ks = Vec::new();
        for tok in lexer.by_ref() {
            if !tok.kind.is_comment() {
                ks.push(match tok.kind {
                    TokKind::NotWord => TokKind::Bang,
                    TokKind::Ident("class" | "record")
                        if aliases.contains(&tok.location.span().start()) =>
                    {
                        TokKind::Struct
                    }
                    k => k,
                });
            }
        }
        lexer.take_errors().is_empty().then_some(ks)
    }
    kinds(before, file_name, aliases) == kinds(after, file_name, aliases)
}

/// A token as the formatter sees it: kind + byte span into the source.
struct T<'s> {
    kind: TokKind<'s>,
    start: usize,
    end: usize,
}

// --- token classification ---------------------------------------------------

/// Tokens that can end an expression — what follows them reads as
/// postfix, index, call or infix.
fn is_ender(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::Ident(_)
            | TokKind::Int(_)
            | TokKind::Float(_)
            | TokKind::Quantity(..)
            | TokKind::String(_)
            | TokKind::FString(_)
            | TokKind::Tagged(..)
            | TokKind::Hex(_)
            | TokKind::True
            | TokKind::False
            | TokKind::Null
            | TokKind::SelfKeyword
            | TokKind::TyKw(_)
            | TokKind::RightParenthesis
            | TokKind::RightSquare
            | TokKind::RightBrace
            | TokKind::Bang
            | TokKind::Hook
            // `z⁺ - y`, `q^Δ (s)`: the Z-set postfix glyphs end an operand
            | TokKind::SupPlus
            | TokKind::SupMinus
            | TokKind::Incremental
            | TokKind::Percent
    )
}

fn is_opener(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::LeftParenthesis
            | TokKind::LeftSquare
            | TokKind::LeftBrace
            | TokKind::HookLeftSquare
    )
}

fn is_closer(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::RightParenthesis | TokKind::RightSquare | TokKind::RightBrace
    )
}

/// Kinds that can never begin a statement — a line led by one continues
/// the line above and indents one extra level.
fn is_continuation(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::DoubleAmpersand
            | TokKind::DoubleHook
            | TokKind::DoubleHookEqual
            | TokKind::In
            | TokKind::NotIn
            | TokKind::FatArrow
            | TokKind::Arrow
            | TokKind::Equal
            | TokKind::DoubleEqual
            | TokKind::BangEqual
            | TokKind::Less
            | TokKind::LessEqual
            | TokKind::Greater
            | TokKind::GreaterEqual
            | TokKind::DoubleLeftCaret
            | TokKind::DoubleRightCaret
            | TokKind::Comma
            | TokKind::SemiColon
            | TokKind::Star
            | TokKind::Slash
            | TokKind::TildeSlash
            | TokKind::TildeSlashEqual
            | TokKind::PercentEqual
            | TokKind::Ampersand
            | TokKind::AmpersandEqual
            | TokKind::PipeEqual
            | TokKind::CaretEqual
            | TokKind::CircledPlus
            | TokKind::CircledPlusEqual
            | TokKind::PlusEqual
            | TokKind::MinusEqual
            | TokKind::StarEqual
            | TokKind::SlashEqual
            | TokKind::Nand
            | TokKind::Nor
            | TokKind::Wedge
            | TokKind::Vee
            | TokKind::DotOp
            | TokKind::Cross
            | TokKind::Compose
            | TokKind::Tensor
            | TokKind::SqDot
            | TokKind::Intersect
            | TokKind::Union
            | TokKind::Ominus
            | TokKind::Uplus
            | TokKind::SqPlus
            | TokKind::SqMinus
            | TokKind::Approx
            | TokKind::PlusMinus
            | TokKind::Bowtie
            | TokKind::Percent
            | TokKind::Dot
            | TokKind::HookDot
            | TokKind::DoubleColon
            | TokKind::Hook
            | TokKind::HookLeftSquare
            | TokKind::Absolve
            | TokKind::Dollar
    )
}

/// Always glued to the token after them.
fn tight_after(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::LeftParenthesis
            | TokKind::LeftSquare
            | TokKind::HookLeftSquare
            | TokKind::Dot
            | TokKind::HookDot
            | TokKind::DoubleColon
            | TokKind::DoubleDot
            | TokKind::DoubleDotEqual
            | TokKind::Dollar
            | TokKind::Hash
            | TokKind::At
    )
}

/// Always glued to the token before them.
fn tight_before(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::RightParenthesis
            | TokKind::RightSquare
            | TokKind::Comma
            | TokKind::SemiColon
            | TokKind::Colon
            | TokKind::Dot
            | TokKind::HookDot
            | TokKind::DoubleColon
            | TokKind::HookLeftSquare
            // `z⁺`, `z⁻`, `q^Δ`: postfix, only ever glued
            | TokKind::SupPlus
            | TokKind::SupMinus
            | TokKind::Incremental
    )
}

/// Kinds that carry source text — emit their slice, not their Display
/// (`1e-6`, `#1.10`, `25kW`, `"a\n"`, `1_000`, comments all round-trip).
fn has_text(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::Ident(_)
            | TokKind::Int(_)
            | TokKind::Float(_)
            | TokKind::Quantity(..)
            | TokKind::String(_)
            | TokKind::FString(_)
            | TokKind::Tagged(..)
            | TokKind::Hex(_)
            | TokKind::Invalid(_)
            | TokKind::Comment(_)
            | TokKind::DocComment(_)
            | TokKind::InnerDocComment(_)
            | TokKind::LiteralComment(_)
            | TokKind::ProseIsland(_)
    )
}

/// One space on both sides, always. `|`, `!`, `?`, `%`, `+`, `-`, `<`,
/// `>`, `>>`, `~`, `:` get their own rules; `and`/`or`/`not` lex to the
/// symbol kinds and re-emit through Display.
fn spaced(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::Equal
            | TokKind::DoubleEqual
            | TokKind::BangEqual
            | TokKind::LessEqual
            | TokKind::GreaterEqual
            | TokKind::DoubleLeftCaret
            | TokKind::Arrow
            | TokKind::FatArrow
            | TokKind::DoubleAmpersand
            | TokKind::DoublePipe
            | TokKind::DoubleHook
            | TokKind::DoubleHookEqual
            | TokKind::Plus
            | TokKind::Star
            | TokKind::Slash
            | TokKind::TildeSlash
            | TokKind::TildeSlashEqual
            | TokKind::PercentEqual
            | TokKind::Ampersand
            | TokKind::AmpersandEqual
            | TokKind::PipeEqual
            | TokKind::Caret
            | TokKind::CaretEqual
            | TokKind::CircledPlus
            | TokKind::CircledPlusEqual
            | TokKind::PlusEqual
            | TokKind::MinusEqual
            | TokKind::StarEqual
            | TokKind::SlashEqual
            | TokKind::Nand
            | TokKind::Nor
            | TokKind::Wedge
            | TokKind::Vee
            | TokKind::DotOp
            | TokKind::Cross
            | TokKind::Compose
            | TokKind::Tensor
            | TokKind::SqDot
            | TokKind::Intersect
            | TokKind::Union
            | TokKind::Ominus
            | TokKind::Uplus
            | TokKind::SqPlus
            | TokKind::SqMinus
            | TokKind::Approx
            | TokKind::PlusMinus
            | TokKind::Bowtie
            | TokKind::In
            | TokKind::NotIn
    )
}

/// The unit-spec set (the parser's `is_unit_spec` is crate-private):
/// `unit x = 1852 m / s ^ 2` reads operands glued by `/`/`*`.
fn is_unit_tok(k: TokKind) -> bool {
    matches!(
        k,
        TokKind::Ident(_)
            | TokKind::Int(_)
            | TokKind::Float(_)
            | TokKind::Quantity(..)
            | TokKind::Slash
            | TokKind::Star
            | TokKind::Caret
            | TokKind::NotWord
            | TokKind::LeftParenthesis
            | TokKind::RightParenthesis
    )
}

/// How a single `|` reads: `|x|` delimits a closure (open then close),
/// `a | b` is the infix operator.
#[derive(Clone, Copy, PartialEq)]
enum PipeRole {
    Open,
    Close,
    Infix,
}

/// What a `// fmt:` comment line may say — `fmt: off` alone or followed
/// by more words (`// fmt: off — generated table`).
fn directive(body: &str, name: &str) -> bool {
    body == name
        || body
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with(char::is_whitespace))
}

/// Line shapes that can share a column.
#[derive(Clone, Copy, PartialEq)]
enum Cls {
    /// `=` of a `let`/`const`/assignment or an `=` record field.
    Eq,
    /// The value after a `name:` field colon.
    Colon,
    /// `=>` of a match-style arm.
    Arm,
    /// `->` of a signature line.
    Arrow,
    /// Comma-separated cells of one row.
    Cells,
}

/// Where a line's pad goes: before the one pivot token, or before each
/// cell from the second on (`Cells` lists every cell's first token).
enum Anchor {
    One(usize),
    Cells(Vec<usize>),
}

struct Fmt<'s> {
    src: &'s str,
    toks: Vec<T<'s>>,
    line_starts: Vec<usize>,
    /// For each token, the previous non-comment token's index (`MAX`
    /// when none).
    sig_prev: Vec<usize>,
    /// Per-`{`/`}`: tight (dict-type `{int}`, `use` path) or spaced.
    tight_brace: Vec<bool>,
    /// Token indices inside `table {}` cells, `query {}`, a `unit` spec,
    /// a `macro` decl.
    in_table: Vec<bool>,
    in_query: Vec<bool>,
    in_unit: Vec<bool>,
    in_macro: Vec<bool>,
    pipe: Vec<PipeRole>,
    /// `table {}`: spaces to emit after the last token of a non-final
    /// cell (column padding).
    cell_pad: std::collections::HashMap<usize, usize>,
    /// Unmatched openers before each token — the token's nesting depth,
    /// with a leading `}` counted at its outer level.
    depth: Vec<usize>,
    /// `class`/`record` tokens sitting in `struct` decl position —
    /// canonical spelling is `struct` (r[impl mimas.struct.alias]).
    alias_kw: Vec<bool>,
    /// Source lines a `// fmt: off`/`// fmt: on` region or a
    /// `// fmt: skip` node keeps verbatim.
    off_line: Vec<bool>,
}

impl<'s> Fmt<'s> {
    fn new(src: &'s str, toks: Vec<T<'s>>) -> Self {
        let mut line_starts = vec![0usize];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        let n = toks.len();
        let mut f = Self {
            src,
            toks,
            line_starts,
            sig_prev: vec![usize::MAX; n],
            tight_brace: vec![false; n],
            in_table: vec![false; n],
            in_query: vec![false; n],
            in_unit: vec![false; n],
            in_macro: vec![false; n],
            pipe: vec![PipeRole::Infix; n],
            cell_pad: Default::default(),
            depth: vec![0; n],
            alias_kw: vec![false; n],
            off_line: vec![false; src.lines().count() + 1],
        };
        f.prepare();
        f
    }

    // --- positions -----------------------------------------------------------

    fn line_of(&self, pos: usize) -> usize {
        self.line_starts.partition_point(|&s| s <= pos).max(1) - 1
    }

    fn kind(&self, i: usize) -> TokKind<'s> {
        self.toks[i].kind
    }

    fn start_line(&self, i: usize) -> usize {
        self.line_of(self.toks[i].start)
    }

    /// The line this token's last byte sits on.
    fn end_line(&self, i: usize) -> usize {
        let t = &self.toks[i];
        self.line_of(t.end.saturating_sub(1).max(t.start))
    }

    /// Bytes between the previous token and this one.
    fn gap_bytes(&self, i: usize) -> usize {
        if i == 0 {
            0
        } else {
            self.toks[i].start - self.toks[i - 1].end
        }
    }

    fn glued_left(&self, i: usize) -> bool {
        i > 0 && self.gap_bytes(i) == 0
    }

    fn is_comment(&self, i: usize) -> bool {
        self.kind(i).is_comment()
    }

    /// The nearest non-comment token before `i`.
    fn sig(&self, i: usize) -> Option<usize> {
        (self.sig_prev[i] != usize::MAX).then_some(self.sig_prev[i])
    }

    /// The next non-comment token after `i`.
    fn sig_next(&self, i: usize) -> Option<usize> {
        (i + 1..self.toks.len()).find(|&j| !self.is_comment(j))
    }

    /// Whether token `i` spells a unit name (`px`, `s`, `tile`,
    /// `540.0px`'s unit) — `s^2`-style `^n` tails split off first, the
    /// way `unit_name_exp` reads them.
    fn unit_named(&self, i: usize) -> bool {
        let name = match self.kind(i) {
            TokKind::Ident(name) => name.split('^').next().unwrap_or(name),
            TokKind::Quantity(_, name) => name,
            _ => return false,
        };
        shared::units::lookup(name).is_some()
    }

    // --- preparation ---------------------------------------------------------

    fn prepare(&mut self) {
        let mut prev = usize::MAX;
        for i in 0..self.toks.len() {
            self.sig_prev[i] = prev;
            if !self.is_comment(i) {
                prev = i;
            }
        }
        self.match_braces();
        self.mark_regions();
        self.pipe_roles();
        self.table_cells();
        self.bracket_depths();
        self.off_regions();
        self.struct_aliases();
    }

    /// `class`/`record` where the parser's decl gate fires canonicalize
    /// to `struct` (r[impl mimas.struct.alias]): the word opens a
    /// statement — after `;`/`{`/`}`/`pub`, or at a line head outside
    /// parens/brackets — and a name then `{`, a same-line `(`/`;`, or a
    /// balanced generic `<…>` the `{`/`(`/`;` of a decl follows.
    fn struct_aliases(&mut self) {
        let mut paren = vec![false; self.toks.len()];
        let mut depth = 0usize;
        for i in 0..self.toks.len() {
            let k = self.kind(i);
            if matches!(k, TokKind::RightParenthesis | TokKind::RightSquare) {
                depth = depth.saturating_sub(1);
            }
            paren[i] = depth > 0;
            if matches!(
                k,
                TokKind::LeftParenthesis | TokKind::LeftSquare | TokKind::HookLeftSquare
            ) {
                depth += 1;
            }
        }
        for i in 0..self.toks.len() {
            let TokKind::Ident("class" | "record") = self.kind(i) else {
                continue;
            };
            self.alias_kw[i] = self.alias_decl_at(i, &paren);
        }
    }

    /// Whether the `class`/`record` token at `i` sits where the parser
    /// would read a `struct` decl. The statement-start half is the
    /// formatter's own question (the parser answers it by position);
    /// the decl-shape half is `parse::alias_decl_at` itself, so the two
    /// can never drift (r[impl mimas.struct.alias]).
    fn alias_decl_at(&self, i: usize, paren: &[bool]) -> bool {
        let stmt_start = match self.sig(i) {
            None => true,
            Some(p) => {
                matches!(
                    self.kind(p),
                    TokKind::SemiColon | TokKind::LeftBrace | TokKind::RightBrace | TokKind::Pub
                ) || (self.start_line(i) > self.end_line(p) && !paren[i])
            }
        };
        if !stmt_start {
            return false;
        }
        parse::alias_decl_at(
            |j| self.toks.get(j).map(|t| t.kind),
            |j| {
                self.sig(j)
                    .is_some_and(|p| self.start_line(j) > self.end_line(p))
            },
            i,
        )
    }

    /// Nesting depth per token: a `{`/`(`/`[`/`?[` raises it for what
    /// follows, a closer lowers it first, so `}` measures at its outer
    /// level. Well-nested by the time this runs — the source parsed.
    fn bracket_depths(&mut self) {
        let mut d = 0usize;
        for i in 0..self.toks.len() {
            let k = self.kind(i);
            if is_closer(k) {
                d = d.saturating_sub(1);
            }
            self.depth[i] = d;
            if is_opener(k) {
                d += 1;
            }
        }
    }

    /// `// fmt: off` … `// fmt: on` and `// fmt: skip` regions
    /// (r[impl mimas.fmt.off]). The directive lives in a `//` comment
    /// (a run of `//` lines is one token; the marker sits on any of
    /// them). `off` covers the lines after the marker's token up to and
    /// including the `on` comment's lines; an unclosed `off` runs to
    /// end of file. `skip` marks the next statement or item's lines —
    /// bracket depth decides where it ends, so `fn`/`if`/`match`
    /// bodies come along whole.
    fn off_regions(&mut self) {
        let mut off_start = usize::MAX;
        for i in 0..self.toks.len() {
            let TokKind::Comment(text) = self.kind(i) else {
                continue;
            };
            if !text.starts_with("//") {
                continue;
            }
            for line in text.split('\n') {
                let body = line.trim_start().strip_prefix("//").unwrap_or("").trim();
                if directive(body, "fmt: off") && off_start == usize::MAX {
                    off_start = self.end_line(i) + 1;
                } else if directive(body, "fmt: on") && off_start != usize::MAX {
                    // the `fmt: on` comment's own lines stay verbatim —
                    // code sharing the line with it keeps its spacing
                    self.mark_lines(off_start, self.end_line(i) + 1);
                    off_start = usize::MAX;
                } else if directive(body, "fmt: skip") && off_start == usize::MAX {
                    self.skip_node(i);
                }
            }
        }
        if off_start != usize::MAX {
            self.mark_lines(off_start, self.line_starts.len());
        }
    }

    fn mark_lines(&mut self, from: usize, through: usize) {
        for l in from..through.min(self.off_line.len()) {
            self.off_line[l] = true;
        }
    }

    /// The statement or item after a `// fmt: skip` comment: from the
    /// next non-comment token until the first later line that starts
    /// back at its depth without continuing the statement — so a `{…}`
    /// body or a `.`-chain comes along, an `else` arm too.
    fn skip_node(&mut self, comment: usize) {
        let Some(first) = self.sig_next(comment) else {
            return;
        };
        let base = self.depth[first];
        let mut last = first;
        while let Some(k) = self.sig_next(last) {
            if self.start_line(k) > self.end_line(last)
                && self.depth[k] <= base
                && !is_continuation(self.kind(k))
                && self.kind(k) != TokKind::Else
            {
                break;
            }
            last = k;
        }
        self.mark_lines(self.start_line(first), self.end_line(last) + 1);
    }

    /// Match every `(`/`[`/`{`/`?[` with its closer and classify each
    /// `{`: tight when it's a dict *type* or `use` path — after `:`,
    /// `->`, `,`, `(`, `[`, `<` with no inner `:`/`=`/`=>`, or after
    /// `::` — spaced otherwise (blocks, records, struct literals, enum
    /// arms). The kind only shows for same-line braces anyway.
    fn match_braces(&mut self) {
        let mut stack: Vec<usize> = vec![];
        for i in 0..self.toks.len() {
            let k = self.kind(i);
            if is_opener(k) {
                stack.push(i);
            } else if is_closer(k)
                && let Some(open) = stack.pop()
                && self.kind(open) == TokKind::LeftBrace
            {
                let tight = self.tight_open_brace(open, i);
                self.tight_brace[open] = tight;
                self.tight_brace[i] = tight;
            }
        }
    }

    /// Whether the `{` at `open` (closed at `close`) is a dict type or
    /// `use` path and so keeps tight braces.
    fn tight_open_brace(&self, open: usize, close: usize) -> bool {
        if let Some(p) = self.sig(open)
            && self.kind(p) == TokKind::DoubleColon
        {
            return true; // `use a::{b, c}`
        }
        let candidate = matches!(
            self.sig(open).map(|p| self.kind(p)),
            Some(
                TokKind::Colon
                    | TokKind::Arrow
                    | TokKind::Comma
                    | TokKind::LeftParenthesis
                    | TokKind::LeftSquare
                    | TokKind::Less
            )
        );
        if !candidate || self.start_line(open) != self.end_line(close) {
            return false;
        }
        // inside `{…}` at annotation position, a `:`/`=`/`=>` makes it a
        // record or block, not a `{T}` dict type
        !(open + 1..close).any(|i| {
            matches!(
                self.kind(i),
                TokKind::Colon | TokKind::Equal | TokKind::FatArrow
            )
        })
    }

    /// The closer matching the opener at `i` (a rescan; only run for
    /// `Ident("…")` intros so it's rare).
    fn match_of(&self, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for j in open..self.toks.len() {
            let k = self.kind(j);
            if is_opener(k) {
                depth += 1;
            } else if is_closer(k) {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(j);
                }
            }
        }
        None
    }

    /// Whether `i` starts a statement: first non-comment of its line, or
    /// right after a `;`/`{`/`}`.
    fn stmt_start(&self, i: usize) -> bool {
        match self.sig(i) {
            None => true,
            Some(p) if self.end_line(p) != self.start_line(i) => true,
            Some(p) => matches!(
                self.kind(p),
                TokKind::SemiColon | TokKind::LeftBrace | TokKind::RightBrace
            ),
        }
    }

    /// Mark `table {}`/`query {}` interiors, `unit` spec tokens, `macro`
    /// decl spans.
    fn mark_regions(&mut self) {
        for i in 0..self.toks.len() {
            let k = self.kind(i);
            // `table {`/`query {`/`query z {`: rows/cells, whitespace-separated
            if let TokKind::Ident(name @ ("table" | "query")) = k
                && let Some(next) = self.sig_next(i)
                && let Some(open) = if name == "query" && self.kind(next) == TokKind::Ident("z") {
                    self.sig_next(next)
                } else {
                    Some(next)
                }
                && self.kind(open) == TokKind::LeftBrace
                && let Some(close) = self.match_of(open)
            {
                for t in open + 1..close {
                    if name == "table" {
                        self.in_table[t] = true;
                    } else {
                        self.in_query[t] = true;
                    }
                }
            }
            // `macro name(` / `macro name {`: the decl is macro space
            if k == TokKind::Ident("macro") {
                let name = self.sig_next(i);
                let open = name.and_then(|j| self.sig_next(j));
                if let (Some(_), Some(open)) = (name, open)
                    && matches!(
                        self.kind(open),
                        TokKind::LeftParenthesis | TokKind::LeftBrace
                    )
                    && let Some(mut last) = self.match_of(open)
                {
                    if self.kind(open) == TokKind::LeftParenthesis
                        && let Some(tmpl) = self.sig_next(last)
                        && self.kind(tmpl) == TokKind::LeftBrace
                        && let Some(end) = self.match_of(tmpl)
                    {
                        last = end;
                    }
                    for t in i..=last {
                        self.in_macro[t] = true;
                    }
                }
            }
            // `unit name = spec` at a statement start: the spec runs to
            // the line's end (or `;`) and glues `/`/`*` between operands
            if k == TokKind::Ident("unit")
                && self.stmt_start(i)
                && let Some(name) = self.sig_next(i)
                && matches!(self.kind(name), TokKind::Ident(_))
            {
                let line = self.start_line(i);
                let mut j = name;
                while let Some(next) = self.sig_next(j) {
                    j = next;
                    if self.start_line(j) != line || self.kind(j) == TokKind::SemiColon {
                        break;
                    }
                    if self.kind(j) != TokKind::Equal {
                        continue;
                    }
                    let mut t = j;
                    while let Some(spec) = self.sig_next(t) {
                        t = spec;
                        if self.start_line(t) != line
                            || self.kind(t) == TokKind::SemiColon
                            || !is_unit_tok(self.kind(t))
                        {
                            break;
                        }
                        self.in_unit[t] = true;
                    }
                    break;
                }
            }
        }
    }

    /// `|` roles: a `|` after a non-ender opens a closure, the next one
    /// closes it; after an ender it's the infix `|`.
    fn pipe_roles(&mut self) {
        let mut open = false;
        for i in 0..self.toks.len() {
            if self.kind(i) != TokKind::Pipe {
                continue;
            }
            if open {
                self.pipe[i] = PipeRole::Close;
                open = false;
            } else {
                match self.sig(i) {
                    Some(p) if is_ender(self.kind(p)) => self.pipe[i] = PipeRole::Infix,
                    _ => {
                        self.pipe[i] = PipeRole::Open;
                        open = true;
                    }
                }
            }
        }
    }

    /// `table {}` rows: cells are runs of glued tokens; each column pads
    /// to the row-max width plus the gutter.
    fn table_cells(&mut self) {
        let n = self.toks.len();
        let mut rows: Vec<Vec<(usize, usize)>> = vec![];
        let mut i = 0;
        while i < n {
            if !self.in_table[i] || self.is_comment(i) {
                i += 1;
                continue;
            }
            // a row: the in_table tokens sharing a line, split into
            // glued cell runs
            let line = self.start_line(i);
            let mut cells: Vec<(usize, usize)> = vec![];
            let mut j = i;
            while j < n && self.in_table[j] && self.start_line(j) == line {
                if self.is_comment(j) {
                    j += 1;
                    continue;
                }
                let cell_start = j;
                let mut last = j;
                j += 1;
                while j < n
                    && self.in_table[j]
                    && self.start_line(j) == line
                    && !self.is_comment(j)
                    && self.gap_bytes(j) == 0
                {
                    last = j;
                    j += 1;
                }
                cells.push((cell_start, last));
            }
            if !cells.is_empty() {
                rows.push(cells);
            }
            i = j.max(i + 1);
        }
        let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut widths = vec![0usize; cols];
        for row in &rows {
            for (c, &(s, e)) in row.iter().enumerate() {
                let w = self.src[self.toks[s].start..self.toks[e].end]
                    .chars()
                    .count();
                widths[c] = widths[c].max(w);
            }
        }
        for row in rows {
            for (c, &(s, e)) in row.iter().enumerate() {
                if c + 1 == row.len() {
                    break;
                }
                let w = self.src[self.toks[s].start..self.toks[e].end]
                    .chars()
                    .count();
                self.cell_pad.insert(e, widths[c] - w + TABLE_GUTTER);
            }
        }
    }

    // --- emission ------------------------------------------------------------

    /// Emit twice (r[impl mimas.fmt.align]): a first pass with the
    /// canonical single-space gaps measures every alignment pivot's
    /// column; the second writes the pads [`alignment_pads`] computed
    /// from it. Each pad replaces one gap wholesale, so a second
    /// `fmt` measures the same columns and writes the same text.
    fn finish(self) -> String {
        let (once, pos) = self.emit(&Default::default());
        let pads = self.alignment_pads(&pos);
        if pads.is_empty() {
            return once;
        }
        self.emit(&pads).0
    }

    /// One emission pass. Returns the text and, for every token written
    /// through the normal path, its `(output line, output column)` in
    /// chars — `usize::MAX` for tokens inside a verbatim region.
    fn emit(
        &self,
        pads: &std::collections::HashMap<usize, usize>,
    ) -> (String, Vec<(usize, usize)>) {
        let mut out = String::with_capacity(self.src.len() + 16);
        let mut pos = vec![(usize::MAX, 0usize); self.toks.len()];
        // the emitted line each still-open bracket was opened on — a
        // line that opens `(` *and* `[` on one line still adds a single
        // level, which is how the corpus writes nested call/array args
        let mut open_stack: Vec<usize> = vec![];
        let mut cur_line = 0usize;
        let mut cur_col = 0usize;
        let mut line_indent = 0usize;
        let mut i = 0usize;
        while i < self.toks.len() {
            if self.off_line[self.start_line(i)] {
                // a verbatim run: the source bytes from the end of the
                // last formatted token through the last off line's end,
                // newline excluded (the next token's own line-start
                // writes it)
                let mut j = i;
                while j + 1 < self.toks.len() && self.off_line[self.start_line(j + 1)] {
                    j += 1;
                }
                let last_line = self.end_line(j);
                let run_end = if last_line + 1 < self.line_starts.len() {
                    self.line_starts[last_line + 1] - 1
                } else {
                    self.src.len()
                };
                let from = if i == 0 { 0 } else { self.toks[i - 1].end };
                let slice = &self.src[from..run_end];
                out.push_str(slice);
                cur_line += slice.matches('\n').count();
                cur_col = slice.rsplit('\n').next().unwrap_or("").chars().count();
                i = j + 1;
                continue;
            }
            if i > 0 {
                if self.start_line(i) == self.end_line(i - 1) {
                    let g = self.gap(i, pads);
                    out.push_str(&g);
                    cur_col += g.chars().count();
                } else {
                    // a new line: collapse blanks, then indent
                    let between = self.start_line(i) - self.end_line(i - 1) - 1;
                    let drop_blank = self.line_starts_closer(i) || self.prev_line_ends_opener(i);
                    out.push('\n');
                    cur_line += 1;
                    if !drop_blank {
                        for _ in 0..between.min(1) {
                            out.push('\n');
                            cur_line += 1;
                        }
                    }
                    // leading closers pop their brackets first — the
                    // line then sits at the outermost closed bracket's
                    // host line's level
                    let lead = self.leading_closers(i);
                    let remaining = open_stack.len().saturating_sub(lead);
                    let base = open_stack[..remaining]
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len();
                    line_indent = self.line_indent(i, lead, base);
                    out.push_str(&" ".repeat(line_indent));
                    cur_col = line_indent;
                }
            }
            pos[i] = (cur_line, cur_col);
            let at = out.len();
            self.emit_tok(&mut out, i, line_indent);
            let written = &out[at..];
            match written.rfind('\n') {
                Some(_) => {
                    cur_line += written.matches('\n').count();
                    cur_col = written.rsplit('\n').next().unwrap_or("").chars().count();
                }
                None => cur_col += written.chars().count(),
            }
            match self.kind(i) {
                k if is_opener(k) => open_stack.push(cur_line),
                k if is_closer(k) => {
                    open_stack.pop();
                }
                _ => {}
            }
            i += 1;
        }
        let trimmed = out.trim_end().len();
        out.truncate(trimmed);
        if !out.is_empty() {
            out.push('\n');
        }
        (out, pos)
    }

    // --- alignment -------------------------------------------------------------

    /// From the single-space pass's `(line, col)` per token: the pad map
    /// for pass two. Lines of one class, one indent, and adjacent
    /// numbers form a group; a group of one emits nothing. For a
    /// single-pivot class the pad before the pivot is `target - col +
    /// 1`; for cell rows each column's target chains off the previous
    /// column's (r[impl mimas.fmt.align]).
    fn alignment_pads(&self, pos: &[(usize, usize)]) -> std::collections::HashMap<usize, usize> {
        let mut pads = std::collections::HashMap::new();
        // the sig token indices of each emitted line, in order
        let mut lines: Vec<(usize, Vec<usize>)> = vec![];
        for (i, &(line, _)) in pos.iter().enumerate() {
            if line == usize::MAX || self.is_comment(i) {
                continue;
            }
            match lines.last_mut() {
                Some(l) if l.0 == line => l.1.push(i),
                _ => lines.push((line, vec![i])),
            }
        }
        let mut group: Vec<Anchor> = vec![];
        let mut key: Option<(Cls, usize, usize)> = None;
        let mut prev_line = usize::MAX;
        for (no, sigs) in lines {
            let member = self.line_class(&sigs).map(|(c, a)| {
                let cells = match &a {
                    Anchor::Cells(c) => c.len(),
                    _ => 0,
                };
                (c, a, pos[sigs[0]].1, cells)
            });
            let fits = match (&member, key) {
                (Some((c, _, ind, cells)), Some((kc, kind, kcells))) => {
                    *c == kc && *ind == kind && *cells == kcells && no == prev_line + 1
                }
                _ => false,
            };
            if !fits {
                self.apply_group(&group, pos, &mut pads);
                group.clear();
                key = member.as_ref().map(|(c, _, ind, cells)| (*c, *ind, *cells));
            }
            if let Some((_, a, _, _)) = member {
                group.push(a);
            }
            prev_line = no;
        }
        self.apply_group(&group, pos, &mut pads);
        pads
    }

    /// Write one finished group's pads — nothing under two members or
    /// past the column bounds.
    fn apply_group(
        &self,
        group: &[Anchor],
        pos: &[(usize, usize)],
        pads: &mut std::collections::HashMap<usize, usize>,
    ) {
        if group.len() < 2 {
            return;
        }
        // all-or-nothing: stage the pads, merge them only when every
        // member stays inside the bounds
        let mut staged: Vec<(usize, usize)> = vec![];
        let ok = (|pads: &mut Vec<(usize, usize)>| {
            match &group[0] {
                Anchor::One(_) => {
                    let col = |g: &Anchor| match g {
                        Anchor::One(t) => pos[*t].1,
                        _ => unreachable!(),
                    };
                    let target = group.iter().map(&col).max().unwrap_or(0);
                    if target > MAX_ALIGN_COL {
                        return false;
                    }
                    for g in group {
                        let pad = target + 1 - col(g);
                        if pad > MAX_ALIGN_PAD {
                            return false;
                        }
                        let Anchor::One(t) = g else { unreachable!() };
                        pads.push((*t, pad));
                    }
                }
                Anchor::Cells(first_cells) => {
                    // each column's target chains off the previous one's:
                    // cell k's start is `prev_target + dist`, `dist` the
                    // single-spaced distance between the two cell starts
                    let cell_col = |g: &Anchor, k: usize| match g {
                        Anchor::Cells(c) => pos[c[k]].1,
                        _ => unreachable!(),
                    };
                    let mut prev_target = 0usize;
                    for k in 1..first_cells.len() {
                        let target = group
                            .iter()
                            .map(|g| prev_target + cell_col(g, k) - cell_col(g, k - 1))
                            .max()
                            .unwrap_or(0);
                        if target > MAX_ALIGN_COL {
                            return false;
                        }
                        for g in group {
                            let would = prev_target + cell_col(g, k) - cell_col(g, k - 1);
                            let pad = target + 1 - would;
                            if pad > MAX_ALIGN_PAD {
                                return false;
                            }
                            let Anchor::Cells(c) = g else { unreachable!() };
                            pads.push((c[k], pad));
                        }
                        prev_target = target;
                    }
                }
            }
            true
        })(&mut staged);
        if ok {
            pads.extend(staged);
        }
    }

    /// A line's alignment class, or `None` for a line that breaks
    /// groups — blank lines and comment-only lines never reach here
    /// (they leave no sig run), the rest fail the shapes below.
    fn line_class(&self, sigs: &[usize]) -> Option<(Cls, Anchor)> {
        let first = sigs[0];
        if self.in_table[first]
            || self.in_query[first]
            || self.in_unit[first]
            || self.in_macro[first]
        {
            return None;
        }
        let d0 = self.depth[first];
        let k0 = self.kind(first);
        // `let`/`const` bindings: the first `=` at the binding's own
        // depth. A `let` with no `=` breaks a run either way.
        if matches!(k0, TokKind::Let | TokKind::Const) {
            return sigs
                .iter()
                .find(|&&t| self.kind(t) == TokKind::Equal && self.depth[t] == d0)
                .map(|&t| (Cls::Eq, Anchor::One(t)));
        }
        // `name:` field lines — struct decls, record fields, fn params,
        // annotations — pad the value that follows the colon
        if matches!(k0, TokKind::Ident(_) | TokKind::String(_))
            && sigs.len() >= 3
            && self.kind(sigs[1]) == TokKind::Colon
            && self.depth[sigs[1]] == d0
        {
            return Some((Cls::Colon, Anchor::One(sigs[2])));
        }
        // `P => v` arms pivot on `=>`
        if let Some(&t) = sigs
            .iter()
            .find(|&&t| self.kind(t) == TokKind::FatArrow && self.depth[t] == d0)
        {
            return Some((Cls::Arm, Anchor::One(t)));
        }
        // `x = …`, `x.f = …`, `self.x = …` — assignments and `=` record
        // fields — pivot on their first depth-0 `=`
        if matches!(k0, TokKind::Ident(_) | TokKind::SelfKeyword)
            && let Some(&t) = sigs
                .iter()
                .find(|&&t| self.kind(t) == TokKind::Equal && self.depth[t] == d0)
        {
            return Some((Cls::Eq, Anchor::One(t)));
        }
        // `->` on a line a signature leads — `fn`, `pub fn`, `notation`
        let fn_led = k0 == TokKind::Fn
            || (k0 == TokKind::Pub && sigs.get(1).is_some_and(|&t| self.kind(t) == TokKind::Fn))
            || matches!(k0, TokKind::Ident("notation"));
        if fn_led
            && let Some(&t) = sigs
                .iter()
                .find(|&&t| self.kind(t) == TokKind::Arrow && self.depth[t] == d0)
        {
            return Some((Cls::Arrow, Anchor::One(t)));
        }
        // comma-separated rows — `[1, 2,]`, `f(a, b)` — cells split at
        // the line's shallowest commas that a cell follows
        if matches!(
            k0,
            TokKind::Ident(_)
                | TokKind::Int(_)
                | TokKind::Float(_)
                | TokKind::Quantity(..)
                | TokKind::String(_)
                | TokKind::FString(_)
                | TokKind::Tagged(..)
                | TokKind::Hex(_)
                | TokKind::True
                | TokKind::False
                | TokKind::Null
                | TokKind::SelfKeyword
                | TokKind::TyKw(_)
                | TokKind::LeftParenthesis
                | TokKind::LeftSquare
                | TokKind::LeftBrace
                | TokKind::Minus
                | TokKind::Hash
        ) {
            let d = sigs[..sigs.len() - 1]
                .iter()
                .filter(|&&t| self.kind(t) == TokKind::Comma)
                .map(|&t| self.depth[t])
                .min()?;
            let mut cells = vec![first];
            for w in 1..sigs.len() {
                let p = sigs[w - 1];
                if self.kind(p) == TokKind::Comma && self.depth[p] == d {
                    cells.push(sigs[w]);
                }
            }
            if cells.len() >= 2 {
                return Some((Cls::Cells, Anchor::Cells(cells)));
            }
        }
        None
    }

    /// The first non-comment token on `i`'s line (`i` opens the line) —
    /// `None` for a comment-only line.
    fn first_sig_of_line(&self, i: usize) -> Option<usize> {
        let line = self.start_line(i);
        (i..self.toks.len())
            .find(|&j| !(self.start_line(j) == line && self.is_comment(j)))
            .filter(|&j| self.start_line(j) == line)
    }

    fn line_starts_closer(&self, i: usize) -> bool {
        self.first_sig_of_line(i)
            .is_some_and(|j| is_closer(self.kind(j)))
    }

    /// The previous emitted line's last *non-comment* token opens a
    /// group — a blank line under it is dropped (`{\n\n x` → `{\n x`).
    fn prev_line_ends_opener(&self, i: usize) -> bool {
        let line = self.end_line(i - 1);
        let mut j = i - 1;
        loop {
            if self.end_line(j) != line {
                return false;
            }
            if !self.is_comment(j) {
                return is_opener(self.kind(j));
            }
            if j == 0 {
                return false;
            }
            j -= 1;
        }
    }

    /// Consecutive closer tokens leading `i`'s line.
    fn leading_closers(&self, i: usize) -> usize {
        let Some(first) = self.first_sig_of_line(i) else {
            return 0;
        };
        let line = self.start_line(i);
        let mut n = 0usize;
        let mut j = first;
        while j < self.toks.len() && self.start_line(j) == line && is_closer(self.kind(j)) {
            n += 1;
            j += 1;
        }
        n
    }

    /// Indent columns for a line starting at token `i`: `base` levels
    /// (distinct host lines of open brackets, after its leading closers
    /// pop theirs), plus one continuation level when the line can't
    /// start a statement or the author over-indented it.
    fn line_indent(&self, i: usize, lead_closers: usize, base: usize) -> usize {
        let bonus = match self.first_sig_of_line(i) {
            Some(j) if is_continuation(self.kind(j)) => 1,
            Some(_) if lead_closers > 0 => 0,
            _ => self.bonus(i, base),
        };
        (base + bonus) * LEVEL
    }

    /// The preserve-bonus: +1 when the author's original indent already
    /// exceeded the canonical base (a wrapped `if`/`from` continuation,
    /// a deliberately nested comment).
    fn bonus(&self, i: usize, base: usize) -> usize {
        let line = self.start_line(i);
        let orig: usize = self.src[self.line_starts[line]..self.toks[i].start]
            .chars()
            .map(|c| if c == '\t' { LEVEL } else { 1 })
            .sum();
        usize::from(orig > base * LEVEL)
    }

    /// The gap between tokens `i-1` and `i` on one line: `""` or `" "`,
    /// an alignment pad, or `table {}` column padding.
    fn gap(&self, i: usize, pads: &std::collections::HashMap<usize, usize>) -> String {
        let (pk, ck) = (self.kind(i - 1), self.kind(i));
        // comment adjacency: one space, always
        if pk.is_comment() || ck.is_comment() {
            return " ".into();
        }
        // an alignment group's pad replaces the single space wholesale —
        // every padded spot (`=`, `=>`, `->`, a `:` value, a cell start)
        // is one the rules below would give a space anyway
        if let Some(&w) = pads.get(&i) {
            return " ".repeat(w.max(1));
        }
        // table cells: glued runs stay, cells get column padding
        if self.in_table[i] && self.in_table[i - 1] {
            if self.gap_bytes(i) == 0 {
                return String::new();
            }
            return " ".repeat(self.cell_pad.get(&(i - 1)).copied().unwrap_or(1));
        }
        // query cells: keep the written adjacency
        if self.in_query[i] && self.in_query[i - 1] {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        // unit specs: `/`/`*`/parens glue operands, but `^`, `-`, `+`,
        // `!` space — glued `s^2`/`s-2` would lex as one identifier
        if self.in_unit[i] && self.in_unit[i - 1] {
            if [pk, ck].iter().any(|k| {
                matches!(
                    k,
                    TokKind::Caret | TokKind::NotWord | TokKind::Minus | TokKind::Plus
                )
            }) {
                return " ".into();
            }
            let operand = |k: TokKind| {
                matches!(
                    k,
                    TokKind::Int(_) | TokKind::Float(_) | TokKind::Quantity(..) | TokKind::Ident(_)
                )
            };
            let glue = matches!(
                pk,
                TokKind::Slash | TokKind::Star | TokKind::LeftParenthesis
            ) || matches!(
                ck,
                TokKind::Slash | TokKind::Star | TokKind::RightParenthesis
            );
            return if operand(pk) && operand(ck) && !glue {
                " "
            } else {
                ""
            }
            .into();
        }
        // `‖z‖`: the bars keep the written adjacency on both sides (a
        // token formatter can't tell an opening bar from a closing one)
        if pk == TokKind::Norm || ck == TokKind::Norm {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        // tight-after ( `(`, `[`, `.`, `::`, `..`, `$`, `#`, `@`, `?[` )
        if tight_after(pk) {
            return String::new();
        }
        // `{:}`, the empty dict, stays glued
        let kind_at = |j: usize| self.toks.get(j).map(|t| t.kind);
        if (pk == TokKind::LeftBrace
            && ck == TokKind::Colon
            && kind_at(i + 1) == Some(TokKind::RightBrace))
            || (pk == TokKind::Colon
                && ck == TokKind::RightBrace
                && i >= 2
                && kind_at(i - 2) == Some(TokKind::LeftBrace))
        {
            return String::new();
        }
        // `{`'s interior: `{ a }` but `{int}` and `::{a, b}` tight
        if pk == TokKind::LeftBrace {
            return if self.tight_brace[i - 1] { "" } else { " " }.into();
        }
        // tight-before ( `)`, `]`, `,`, `;`, `:` )
        if tight_before(ck) {
            return String::new();
        }
        // `{}` empty braces stay glued
        if ck == TokKind::RightBrace && pk == TokKind::LeftBrace {
            return String::new();
        }
        // `}` mirrors its `{`'s kind
        if ck == TokKind::RightBrace {
            return if self.tight_brace[i] { "" } else { " " }.into();
        }
        // `..`/`..=` hug operands (`0..8`, `..rest`); after a separator
        // they space (`, ..rest`)
        if matches!(ck, TokKind::DoubleDot | TokKind::DoubleDotEqual) {
            return if is_ender(pk) { "" } else { " " }.into();
        }
        // `/`/`*` between unit names keeps the written adjacency:
        // `vx: px/s` stays glued (a unit type), `s * m` stays spaced —
        // `s` and `m` are everyday names as much as unit names
        if matches!(ck, TokKind::Slash | TokKind::Star)
            && self.unit_named(i - 1)
            && self.sig_next(i).is_some_and(|n| self.unit_named(n))
        {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        if matches!(pk, TokKind::Slash | TokKind::Star)
            && self.unit_named(i)
            && self.sig(i - 1).is_some_and(|p| self.unit_named(p))
        {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        // `{` after `!` is a macro call's template (`name!{`)
        if ck == TokKind::LeftBrace {
            return if pk == TokKind::Bang { "" } else { " " }.into();
        }
        // `(`/`[`: call/index after an ender, spaced otherwise — except
        // after `check`, where the gap decides check-expr vs call
        if matches!(ck, TokKind::LeftParenthesis | TokKind::LeftSquare) {
            if matches!(pk, TokKind::Ident("check")) || pk == TokKind::Hash {
                return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
            }
            // `X<(`: after an all-glued `<` the paren stays glued too —
            // spacing it would unglue the generic on the next pass
            let glued_generic = pk == TokKind::Less
                && self.glued_left(i - 1)
                && self.glued_left(i);
            return if is_ender(pk) || glued_generic { "" } else { " " }.into();
        }
        // `!`/`?`/`%`: postfix only when glued — keep the written gap
        if matches!(ck, TokKind::Bang | TokKind::Hook | TokKind::Percent) {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        if matches!(pk, TokKind::Bang | TokKind::Hook | TokKind::Percent) && self.gap_bytes(i) == 0
        {
            return String::new();
        }
        // macro repeats: `),*` / `)*` / `)+` / `)?`
        if self.in_macro[i]
            && matches!(ck, TokKind::Star | TokKind::Plus | TokKind::Hook)
            && (pk == TokKind::RightParenthesis
                || (pk == TokKind::Comma
                    && matches!(
                        self.sig(i - 1),
                        Some(b) if self.kind(b) == TokKind::RightParenthesis
                    )))
        {
            return String::new();
        }
        // `check -` keeps its gap (check-expr vs subtraction)
        if ck == TokKind::Minus && matches!(pk, TokKind::Ident("check")) {
            return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
        }
        // `-`/`+` after: `check`-preserve, then `[a -1]`'s element split
        // stays glued, binary spaces, prefix hugs
        if matches!(pk, TokKind::Minus | TokKind::Plus) {
            if self
                .sig(i - 1)
                .is_some_and(|p| matches!(self.kind(p), TokKind::Ident("check")))
            {
                return if self.gap_bytes(i) == 0 { "" } else { " " }.into();
            }
            let glued_minus = pk == TokKind::Minus
                && matches!(
                    ck,
                    TokKind::Int(_) | TokKind::Float(_) | TokKind::Quantity(..) | TokKind::Hex(_)
                )
                && self.gap_bytes(i) == 0
                && self.sig(i - 1).is_some_and(|p| {
                    is_ender(self.kind(p)) && self.toks[i - 1].start > self.toks[p].end
                });
            if glued_minus {
                return String::new();
            }
            let binary = self.sig(i - 1).is_some_and(|p| is_ender(self.kind(p)));
            return if binary { " " } else { "" }.into();
        }
        // `-`/`+` before: binary or prefix both read one space (the `(`/
        // `[`/`{`/`/` tight cases already returned)
        if matches!(ck, TokKind::Minus | TokKind::Plus) {
            return " ".into();
        }
        // `<` all-glued stays glued (generics vs comparisons); any gap
        // opens both sides. `>`/`>>` hug left, space right
        if ck == TokKind::Less {
            return if self.glued_left(i) && self.glued_left(i + 1) {
                ""
            } else {
                " "
            }
            .into();
        }
        if matches!(ck, TokKind::Greater | TokKind::DoubleRightCaret) {
            return if self.glued_left(i) { "" } else { " " }.into();
        }
        if pk == TokKind::Less || matches!(pk, TokKind::Greater | TokKind::DoubleRightCaret) {
            return if self.glued_left(i - 1) && self.glued_left(i) {
                ""
            } else {
                " "
            }
            .into();
        }
        // `:` after `$x` is a macro fragment spec (`$x:expr`)
        if pk == TokKind::Colon
            && let Some(b) = self.sig(i - 1)
            && matches!(self.kind(b), TokKind::Ident(_))
            && self.sig(b).is_some_and(|d| self.kind(d) == TokKind::Dollar)
        {
            return String::new();
        }
        // `,`/`;` then more on the line
        if matches!(pk, TokKind::Comma | TokKind::SemiColon) {
            return " ".into();
        }
        // `|`: an opening one hugs right, a closing one hugs left,
        // infix spaces both sides
        if pk == TokKind::Pipe {
            return match self.pipe[i - 1] {
                PipeRole::Open => String::new(),
                _ => " ".into(),
            };
        }
        if ck == TokKind::Pipe {
            return if self.pipe[i] == PipeRole::Close {
                ""
            } else {
                " "
            }
            .into();
        }
        // `~` reads like a word (`~ rem`)
        if pk == TokKind::Tilde {
            return " ".into();
        }
        // `not` re-emits `!` and hugs
        if pk == TokKind::NotWord {
            return String::new();
        }
        // the plain binary operators: one space either side
        if spaced(pk) || spaced(ck) {
            return " ".into();
        }
        " ".into()
    }

    /// One token's text: its source slice when it carries one, the
    /// canonical spelling otherwise (`and`→`&&`, `≠`→`!=`, `⊻`→`⊕`…).
    fn emit_tok(&self, out: &mut String, i: usize, line_indent: usize) {
        let k = self.kind(i);
        match k {
            TokKind::Comment(text)
            | TokKind::DocComment(text)
            | TokKind::InnerDocComment(text)
            | TokKind::LiteralComment(text)
            | TokKind::ProseIsland(text) => {
                emit_comment(out, text, line_indent);
            }
            TokKind::NotWord => out.push('!'),
            TokKind::Ident("class" | "record") if self.alias_kw[i] => out.push_str("struct"),
            _ if has_text(k) => out.push_str(&self.src[self.toks[i].start..self.toks[i].end]),
            _ => out.push_str(&k.to_string()),
        }
    }
}

/// A comment's text. `//`-family comments merge their continuation lines
/// into the token; interior lines re-indent to the current line's
/// indent. `/* */` interiors stay as written (they're often art) —
/// only trailing whitespace is trimmed.
fn emit_comment(out: &mut String, text: &str, line_indent: usize) {
    let interior = !text.starts_with("//");
    for (n, line) in text.split('\n').enumerate() {
        if n == 0 {
            out.push_str(line.trim_end());
        } else if interior {
            out.push('\n');
            out.push_str(line.trim_end());
        } else {
            out.push('\n');
            out.push_str(&" ".repeat(line_indent));
            out.push_str(line.trim());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(src: &str) -> String {
        format_source(src, "test.mim").expect("source should parse")
    }

    fn unchanged(src: &str) {
        assert_eq!(fmt(src), src, "expected {src:?} to be canonical");
    }

    // r[verify mimas.dict.empty-literal]
    #[test]
    fn empty_dict_stays_glued() {
        unchanged("let d: {int} = {:}\n");
        assert_eq!(fmt("let d: {int} = { : }\n"), "let d: {int} = {:}\n");
        unchanged("let u = a ∪ b ∩ c ⊎ d\n");
    }

    #[test]
    fn operator_spacing() {
        assert_eq!(fmt("let x=1+2*3\n"), "let x = 1 + 2 * 3\n");
        assert_eq!(fmt("x==y and y !=z\n"), "x == y && y != z\n");
        assert_eq!(fmt("a&&b&&c\n"), "a && b && c\n");
    }

    #[test]
    fn calls_and_access() {
        unchanged("w.hist.push(x)\n");
        unchanged("std::parse::Value::String(\"pass\")\n");
        unchanged("MOVE_CP[f][j]\n");
        assert_eq!(fmt("d?[k] \u{00b7} x?.f()\n"), "d?[k] \u{22c5} x?.f()\n");
        unchanged("[j for j in 0..8 if xs[j] == i]\n");
        assert_eq!(fmt("f (a ,b)\n"), "f(a, b)\n");
        assert_eq!(fmt("a [ 0 ]\n"), "a[0]\n");
    }

    #[test]
    fn postfix_glue_preserved() {
        // `x!`/`x?`/`x%` postfix only parse when glued — the gap is
        // semantic (and an unglued one doesn't parse, so nothing
        // normalizes it)
        unchanged("x! + y\n");
        unchanged("x?(1) + 50%\n");
    }

    #[test]
    fn zset_glyphs_kept() {
        // the notation keeps its glyphs: prefix glyphs space before their
        // operand, the postfix ones and the norm bars stay glued
        unchanged("check ∇ (∫ zs) == zs\n");
        unchanged("check Δ⁻¹ (Δ s) == s - z⁻¹ s\n");
        unchanged("check z == z⁺ - z⁻\n");
        unchanged("check ‖z⁺‖ + ‖z⁻‖ == ‖z‖\n");
        unchanged("let j = a ⋈ b\n");
        unchanged("let i = (q1 ∘ q2)^Δ(s)\n");
        assert_eq!(fmt("let j = a⋈b\n"), "let j = a ⋈ b\n");
        assert_eq!(fmt("let k = 2 · z\n"), "let k = 2 ⋅ z\n");
        assert_eq!(fmt("let d = ∇  s\n"), "let d = ∇ s\n");
    }

    #[test]
    fn minus_glue_preserved() {
        unchanged("let xs = [0 -1]\n"); // two elements
        unchanged("let xs = [0 - 1]\n"); // subtraction
        unchanged("a-1\n"); // one identifier, not arithmetic
        unchanged("check -1\n"); // a check of -1
        unchanged("check - 1\n"); // `check - 1`
        assert_eq!(fmt("x -  1\n"), "x - 1\n");
    }

    #[test]
    fn check_intro_preserved() {
        // `check (x)` checks `x`; `check(x)` is a call — the gap is
        // semantic
        unchanged("check (x > 0)\n");
        unchanged("check(x > 0)\n");
        unchanged("check [-1]\n");
    }

    #[test]
    fn generics_vs_comparison() {
        unchanged("fn f(x: Vec<int>) -> Vec<Vec<int>> { x }\n");
        unchanged("a < b\n");
        unchanged("a<b\n"); // glued comparison keeps its shape
    }

    #[test]
    fn braces() {
        unchanged("let x: {int} = dict\n");
        unchanged("use std::math::{PI, TAU};\n");
        assert_eq!(fmt("{a: 1}\n"), "{ a: 1 }\n");
        unchanged("fn f() { 1 }\n");
    }

    #[test]
    fn indentation() {
        let src = "fn f() {\n    let x = 1\n    if c {\n        x\n    }\n}\n";
        unchanged(src);
        let messy = "fn f() {\nlet x = 1\n        if c {\nx\n}\n}\n";
        let want = "fn f() {\n    let x = 1\n        if c {\n        x\n    }\n}\n";
        // over-indented `if` keeps one bonus level (author's choice)
        assert_eq!(fmt(messy), want);
    }

    #[test]
    fn continuation_lines() {
        unchanged("let x = a\n    .f()\n    .g()\n");
        unchanged("match x {\n    P => 1,\n    Q => 2,\n}\n");
    }

    #[test]
    fn comments_survive() {
        unchanged("// leading\nlet x = 1 // trailing\n// tail\n");
        let src = "let x = 1   // note\n";
        assert_eq!(fmt(src), "let x = 1 // note\n");
        // a `//` run's interior lines re-indent
        let src = "fn f() {\n    let x = 1 // a\n        // b\n}\n";
        let want = "fn f() {\n    let x = 1 // a\n    // b\n}\n";
        assert_eq!(fmt(src), want);
    }

    #[test]
    fn literals_preserved() {
        unchanged("let x = #1.10\n");
        unchanged("let p = 25kW\n");
        unchanged("let e = 1e-6 + 1_000\n");
        unchanged("let s = f\"{name}:co{i}\"\n");
        unchanged("let t = pat\"{a}, {b}\"\n");
    }

    #[test]
    fn table_alignment() {
        let src = "let s = table {\n    rider  clock  scorer\n    \"ada\" 36058 \"spot\"\n}\n";
        let want =
            "let s = table {\n    rider    clock    scorer\n    \"ada\"    36058    \"spot\"\n}\n";
        assert_eq!(fmt(src), want);
    }

    #[test]
    fn query_cells_keep_adjacency() {
        // `sort -elapsed` keeps its glued `-`, in both query modes
        unchanged("let s = query {\n    t\n    sort -elapsed rider\n}\n");
        unchanged("let s = query z {\n    t\n    sort -elapsed rider\n}\n");
    }

    #[test]
    fn decls() {
        unchanged("unit league = 3mi;\n");
        assert_eq!(
            fmt("unit tick = 9.8 m / s ^ 2\n"),
            "unit tick = 9.8 m/s ^ 2\n"
        );
        unchanged("opp Sky { ☀, ☾ }\n");
        unchanged("space Politics { econ: Dial, power: Dial }\n");
        unchanged("macro double($e:expr) { $e + $e }\n");
        unchanged("use \"page\";\n");
    }

    // r[verify mimas.struct.alias]
    #[test]
    fn struct_aliases_canonicalize() {
        assert_eq!(fmt("class P { x: int }"), "struct P { x: int }\n");
        assert_eq!(fmt("record P { x: int }"), "struct P { x: int }\n");
        assert_eq!(fmt("record P(int, int)"), "struct P(int, int)\n");
        assert_eq!(fmt("record P<A> { x: A }"), "struct P<A> { x: A }\n");
        assert_eq!(fmt("pub record P { x: int }"), "pub struct P { x: int }\n");
        assert_eq!(fmt("record P;"), "struct P;\n");
        // a `record`/`class` that isn't decl-shaped stays a name
        unchanged("fn record(x: int) -> int { x + 1 }\nlet y = record 10\nlet z = record y < 4\nlet c = class 3\n");
    }

    #[test]
    fn blank_lines() {
        let src = "let a = 1\n\n\n\nlet b = 2\n";
        assert_eq!(fmt(src), "let a = 1\n\nlet b = 2\n");
        let src = "fn f() {\n\n    x\n\n}\n";
        assert_eq!(fmt(src), "fn f() {\n    x\n}\n");
    }

    #[test]
    fn trailing_and_final() {
        assert_eq!(fmt("let x = 1   \n\n"), "let x = 1\n");
        assert_eq!(fmt(""), "");
        assert_eq!(fmt("// just a comment"), "// just a comment\n");
    }

    #[test]
    fn closures_and_pipes() {
        unchanged("let f = |x| x + 1\n");
        unchanged("let g = || 0\n");
        unchanged("a | b\n");
    }

    #[test]
    fn words_become_symbols() {
        assert_eq!(fmt("a and b or c\n"), "a && b || c\n");
        assert_eq!(fmt("not x\n"), "!x\n");
        assert_eq!(fmt("a ≠ b\n"), "a != b\n");
        assert_eq!(fmt("a ⊻ b\n"), "a ⊕ b\n");
    }

    #[test]
    fn unparseable_is_refused() {
        assert!(format_source("let = = =", "x.mim").is_err());
        assert!(format_source("fn f( {", "x.mim").is_err());
    }

    // r[verify mimas.fmt.off]
    #[test]
    fn fmt_off_region() {
        // canonical outside, verbatim inside — blank runs included
        let src = "let x = 1\n// fmt: off\n   weird    spacing\n\n\n// fmt: on\nlet y = 2\n";
        assert_eq!(fmt(src), src);
        // the region's lines come through byte-for-byte, markers too
        let src = "let   x=1\n// fmt: off\n   weird    spacing\nlet   z=3\n// fmt: on\nlet   y=2\n";
        let want =
            "let x = 1\n// fmt: off\n   weird    spacing\nlet   z=3\n// fmt: on\nlet y = 2\n";
        assert_eq!(fmt(src), want);
        // an unclosed `off` runs to end of file
        assert_eq!(
            fmt("let x = 1\n// fmt: off\n  a    b\nlet   y=2\n"),
            "let x = 1\n// fmt: off\n  a    b\nlet   y=2\n"
        );
        // a marker in a comment run still reads; trailing code shares it
        let src = "// heads up\n// fmt: off\n  a   b\n// fmt: on\nlet   y=2\n";
        let want = "// heads up\n// fmt: off\n  a   b\n// fmt: on\nlet y = 2\n";
        assert_eq!(fmt(src), want);
    }

    // r[verify mimas.fmt.off]
    #[test]
    fn fmt_skip_node() {
        // one statement verbatim, the next formatted
        assert_eq!(
            fmt("// fmt: skip\nlet   weird    =    1\nlet y = 2\n"),
            "// fmt: skip\nlet   weird    =    1\nlet y = 2\n"
        );
        // an item with a `{…}` body comes along whole
        let src = "// fmt: skip\nfn  weird()   {\n      let   x=1\n}\nlet y = 2\n";
        assert_eq!(fmt(src), src);
        // a `.`-chain is one statement
        let src = "// fmt: skip\nlet   x = a\n     .f()\nlet y = 2\n";
        assert_eq!(fmt(src), src);
    }

    // r[verify mimas.fmt.align]
    #[test]
    fn align_binding_runs() {
        assert_eq!(
            fmt("let x = 1\nlet longer = 2\nconst Z = 3\n"),
            "let x      = 1\nlet longer = 2\nconst Z    = 3\n"
        );
        // assignment `=`s join the same class
        assert_eq!(
            fmt("let j = int::random(9)\ns = s + 1\n"),
            "let j = int::random(9)\ns     = s + 1\n"
        );
        // a blank line, a comment, or another shape ends the group
        assert_eq!(
            fmt("let x = 1\n\nlet yy = 2\nlet z = 3\n"),
            "let x = 1\n\nlet yy = 2\nlet z  = 3\n"
        );
        assert_eq!(
            fmt("let x = 1\n// note\nlet yy = 2\nlet z = 3\n"),
            "let x = 1\n// note\nlet yy = 2\nlet z  = 3\n"
        );
        assert_eq!(
            fmt("let x = 1\nf()\nlet yy = 2\n"),
            "let x = 1\nf()\nlet yy = 2\n"
        );
        // an overlong member drops the group wholesale
        let long = "let a_very_long_variable_name_that_keeps_going_and_going_past = 1\nlet b = 2\n";
        assert_eq!(fmt(long), long);
    }

    // r[verify mimas.fmt.align]
    #[test]
    fn align_fields_and_arms() {
        // `:` fields: the values share a column
        assert_eq!(
            fmt("struct P {\n    name: str,\n    age: int,\n}\n"),
            "struct P {\n    name: str,\n    age:  int,\n}\n"
        );
        // `=>` arms pivot on the arrow
        assert_eq!(
            fmt("let m = match x {\n    P => 1,\n    LongerPattern => 2,\n}\n"),
            "let m = match x {\n    P             => 1,\n    LongerPattern => 2,\n}\n"
        );
        // `->` on signature lines led by `fn`/`pub fn`/`notation`
        assert_eq!(
            fmt("fn f(a: int) -> int { a }\nfn longer_name(b: int) -> int { b }\n"),
            "fn f(a: int)           -> int { a }\nfn longer_name(b: int) -> int { b }\n"
        );
        assert_eq!(
            fmt("pub fn f() -> int { 1 }\npub fn longer() -> int { 2 }\n"),
            "pub fn f()      -> int { 1 }\npub fn longer() -> int { 2 }\n"
        );
    }

    // r[verify mimas.fmt.align]
    #[test]
    fn align_cell_rows() {
        assert_eq!(
            fmt("let xs = [\n    1, 222, 33,\n    4444, 5, 6,\n]\n"),
            "let xs = [\n    1,    222, 33,\n    4444, 5,   6,\n]\n"
        );
        // a ragged last row without a trailing comma still joins
        assert_eq!(
            fmt("let xs = [\n    1, 222,\n    4444, 5\n]\n"),
            "let xs = [\n    1,    222,\n    4444, 5\n]\n"
        );
        // different cell counts never mix
        assert_eq!(
            fmt("let xs = [\n    1, 2, 3,\n    4, 5,\n]\n"),
            "let xs = [\n    1, 2, 3,\n    4, 5,\n]\n"
        );
    }

    // r[verify mimas.fmt]
    #[test]
    fn fmt_idempotent() {
        for messy in [
            "fn   f( a: int, b: int) -> int { a+b }\nlet x = [1, 2 ,3]\n",
            "let x = 1\nlet longer = 2\nstruct P {\n    a: int,\n    bb: int,\n}\n",
            "// fmt: off\n   x\n// fmt: on\nlet   y=2\n",
        ] {
            let once = fmt(messy);
            assert_eq!(fmt(&once), once, "fmt not idempotent on {once:?}");
        }
    }

    /// The law pass's helpers are machine-written Mimas — the shapes
    /// from `laws.rs` must round-trip.
    // r[verify mimas.fmt]
    #[test]
    fn generated_law_helpers() {
        let src = concat!(
            "fn __law_impl_0_7(x: Flip) -> bool { x.turned(r).turned(r) == x }\n",
            "fn __law_check_0_7() -> bool { let __ss = [0, 1]; for x in __ss { if !__law_impl_0_7(x) { panic(f\"the law fails for x = {x}\"); } } true }\n",
            "check __law_check_0_7()\n",
            "fn __arb_gen_Point_3(size: int) -> Point { let __sz = if size < 0 { 0 } else { size }; Point { x = int::random(__sz), y = int::random(__sz) } }\n",
            "fn __arb_samples_Point_3() -> [Point] { let __ss: [Point] = []; for __i in 64 { __ss.push(__arb_gen_Point_3(10)); } __ss }\n",
            "fn __arb_min_0_7(x: Flip) -> Flip { for __step in 4096 { let __moved = false; for __c0 in __arb_shrink_Flip_7(x) { if !__law_impl_0_7(__c0) && __c0 != x { x = __c0; __moved = true; break; } } if !__moved { break; } } x }\n",
        );
        let once = fmt(src);
        assert_eq!(fmt(&once), once, "generated helpers not stable: {once:?}");
    }

    /// The idempotence corpus: every shipped `.mim`/`.mimas` under the
    /// mimas tree. `fmt(fmt(x)) == fmt(x)`, and `fmt(x)` must still
    /// parse — `format_source` itself refuses a token-stream diff, so
    /// an `Ok` here is already a semantics round-trip.
    // r[verify mimas.fmt]
    #[test]
    fn corpus_idempotent() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut files = vec![];
        let mut fails = 0;
        collect(&root, &mut files, &mut fails);
        files.sort();
        files.dedup();
        assert!(files.len() > 40, "corpus too small: {}", files.len());
        let mut skipped = vec![];
        let mut unstable = vec![];
        for file in &files {
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let Ok(once) = format_source(&src, &file.to_string_lossy()) else {
                skipped.push(file.display().to_string());
                continue;
            };
            let twice = format_source(&once, &file.to_string_lossy())
                .expect("formatted output must still parse");
            if once != twice {
                unstable.push(file.display().to_string());
            }
        }
        eprintln!(
            "corpus: {} files, {} skipped (unparseable)",
            files.len(),
            skipped.len()
        );
        assert!(
            unstable.is_empty(),
            "fmt not idempotent on:\n{}",
            unstable.join("\n")
        );
    }
}
