use chompy::{
    define_error,
    diagnostics::{Builder, DiagBox, Result as ChompyResult},
    lex::{CharStream, Lex, Tok, UnterminatedString},
    utils::*,
};

use crate::lex::{TokKind, TyKw};

/// Lexing error for a `/*` that never meets its `*/`.
pub struct UnterminatedBlockComment(pub Location);
define_error!(
    UnterminatedBlockComment {
        fn build(&self, builder: Builder) -> Builder {
            builder.label(self.0.primary("this block comment was never closed with `*/`"))
        }

        fn location(&self) -> Location {
            self.0
        }
    }
);

/// Lexing error for a whole number too big to hold.
pub struct IntegerOutOfRange(pub Location);
define_error!(
    IntegerOutOfRange {
        fn build(&self, builder: Builder) -> Builder {
            builder.label(self.0.primary("this number doesn't fit in an int"))
        }

        fn location(&self) -> Location {
            self.0
        }
    }
);

#[derive(Debug)]
pub struct Lexer<'s> {
    source: &'s str,
    char_stream: CharStream<'s>,
    file_id: FileId,
    file_name: String,
    offset: usize,
    errors: Vec<(DiagBox, Location)>,
    cut_short: bool,
    /// Which operator words lex as names, by byte offset: `(start, [and,
    /// or, not])` per page of a spliced source (`shadow_table`), the last
    /// entry at or before a token's start applies.
    shadow: Vec<(usize, [bool; 3])>,
}

/// The operator words a source may take back as names, with the symbol
/// each one spells: `and` `&&`, `or` `||`, `not` `!`.
pub const SHADOWABLE_WORDS: [(&str, &str); 3] = [("and", "&&"), ("or", "||"), ("not", "!")];

/// `Some(symbol)` when `name` is an operator word a fn may shadow.
pub fn shadowable_word_symbol(name: &str) -> Option<&'static str> {
    SHADOWABLE_WORDS
        .iter()
        .find(|(w, _)| *w == name)
        .map(|(_, s)| *s)
}

/// Which operator words `source` declares as fns -- a line-level scan for
/// `fn and` / `fn or` / `fn not` (optionally `pub`), comments skipped. A
/// page that writes its own `not` (a nand2tetris gate, a DSL) gets the
/// name in its own text and in the pages that `use` it, and a warning
/// says so -- never in the pages *it* uses, which were written without it.
pub fn shadowed_words(source: &str) -> [bool; 3] {
    let mut out = [false; 3];
    shadowed_words_into(source, &mut out);
    out
}

/// The line a host's splice writes where each page's text begins:
/// `//# page: game-kit`. A comment to everything else; to the lexer it
/// bounds a page, so shadowing stays per page in one spliced source.
pub const PAGE_MARK: &str = "//# page: ";

/// Per-page shadow sets for a program's pages (`(name, text)`): a page's
/// set is its own `fn and`/`or`/`not` declarations plus the sets of the
/// pages it `use "…"`s, transitively -- a declaration reaches downstream
/// (its users), never upstream (its dependencies).
pub fn shadow_sets(pages: &[(&str, &str)]) -> Vec<[bool; 3]> {
    fn set_of(
        i: usize,
        pages: &[(&str, &str)],
        memo: &mut [Option<[bool; 3]>],
        busy: &mut [bool],
    ) -> [bool; 3] {
        if let Some(s) = memo[i] {
            return s;
        }
        if busy[i] {
            return shadowed_words(pages[i].1); // a `use` cycle: own decls only
        }
        busy[i] = true;
        let mut out = shadowed_words(pages[i].1);
        for name in uses(pages[i].1) {
            if let Some(j) = pages.iter().position(|(n, _)| n.to_lowercase() == name) {
                let dep = set_of(j, pages, memo, busy);
                for k in 0..3 {
                    out[k] |= dep[k];
                }
            }
        }
        busy[i] = false;
        memo[i] = Some(out);
        out
    }
    let mut memo = vec![None; pages.len()];
    let mut busy = vec![false; pages.len()];
    (0..pages.len())
        .map(|i| set_of(i, pages, &mut memo, &mut busy))
        .collect()
}

/// The `use "…"` names `text` includes, lowercased.
fn uses(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| {
            let rest = l.trim().strip_prefix("use")?;
            if !rest.starts_with([' ', '\t']) {
                return None;
            }
            let q = rest.trim_start().strip_prefix('"')?;
            Some(q.split_once('"')?.0.trim().to_lowercase())
        })
        .collect()
}

/// What each file of a multi-file program inherits from the files it
/// `use`s (their own declarations, transitively) -- ORed into the file's own
/// per-page table (`Lexer::with_inherited`), never replacing it, so a
/// spliced file's page marks still bound its own pages.
pub fn inherited_shadow(files: &[(&str, &str)]) -> Vec<[bool; 3]> {
    let sets = shadow_sets(files);
    files
        .iter()
        .map(|(_, text)| {
            let mut out = [false; 3];
            for name in uses(text) {
                if let Some(j) = files.iter().position(|(n, _)| n.to_lowercase() == name) {
                    for k in 0..3 {
                        out[k] |= sets[j][k];
                    }
                }
            }
            out
        })
        .collect()
}

/// `source` split at its `PAGE_MARK` lines into pages, each with its
/// shadow set (`shadow_sets`), as `(byte start, set)`. A source without
/// marks is one page.
// r[impl mimas.op.shadow]
pub fn shadow_table(source: &str) -> Vec<(usize, [bool; 3])> {
    let mut starts: Vec<(usize, &str)> = vec![(0, "")];
    let mut at = 0;
    for line in source.split_inclusive('\n') {
        if let Some(name) = line.strip_prefix(PAGE_MARK) {
            starts.push((at, name.trim()));
        }
        at += line.len();
    }
    let pages: Vec<(&str, &str)> = starts
        .iter()
        .enumerate()
        .map(|(k, &(s, name))| {
            let end = starts.get(k + 1).map_or(source.len(), |n| n.0);
            (name, &source[s..end])
        })
        .collect();
    starts
        .iter()
        .map(|s| s.0)
        .zip(shadow_sets(&pages))
        .collect()
}

/// The set of the last table entry starting at or before `pos`.
pub fn shadow_at(table: &[(usize, [bool; 3])], pos: usize) -> [bool; 3] {
    table
        .iter()
        .rev()
        .find(|(s, _)| *s <= pos)
        .map_or([false; 3], |(_, set)| *set)
}

fn shadowed_words_into(source: &str, out: &mut [bool; 3]) {
    for line in source.lines() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        let mut rest = t;
        while let Some(i) = rest.find("fn ") {
            let before_ok = i == 0
                || !rest[..i]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let after = rest[i + 3..].trim_start();
            if before_ok {
                for (k, (w, _)) in SHADOWABLE_WORDS.iter().enumerate() {
                    if let Some(tail) = after.strip_prefix(w)
                        && !tail.starts_with(|c: char| c.is_alphanumeric() || c == '_')
                    {
                        out[k] = true;
                    }
                }
            }
            rest = &rest[i + 3..];
        }
    }
}

impl<'s> Lexer<'s> {
    /// A number directly followed by a unit (`25kW`, `0.14usd/kWh`) is a quantity: consume the
    /// suffix and say so. Anything else after a number is left for the parser, as before.
    fn with_unit(&mut self, plain: TokKind<'s>, value: f64) -> TokKind<'s> {
        let source = self.source;
        let pos = self.char_stream.position();
        let rest = &source[pos..];
        if let Some(len) = shared::units::scan_suffix(rest) {
            let unit = &rest[..len];
            for c in unit.chars() {
                self.match_chomp(c);
            }
            return TokKind::Quantity(value, unit);
        }
        // `3_i` — the subscript-index suffix. The number scan already
        // took the `_` as a digit separator, so it sits just before
        // `rest`: reattach it as the unit iff the tail parses as one.
        if pos > 0
            && source.as_bytes()[pos - 1] == b'_'
            && rest.starts_with(|c: char| c.is_ascii_alphabetic())
        {
            let ilen = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            let candidate = &source[pos - 1..pos + ilen];
            if shared::units::parse(candidate).is_some() {
                for c in candidate[1..].chars() {
                    self.match_chomp(c);
                }
                return TokKind::Quantity(value, candidate);
            }
        }
        plain
    }

    /// An `e[+-]?digits` exponent riding on a mantissa, chomped only when it's really
    /// there. chompy's own scan advances twice per exponent digit, which drops a char each
    /// time and can swallow what follows the number (`1.5e2+1` came out as `1.5`), so the
    /// exponent is done here instead. `_` separates exponent digits like anywhere else, and
    /// an `e` that isn't followed by `digits` (or `sign + digits`) is simply left alone --
    /// `1.5else` is `1.5` then the identifier `else`. Returns `None` when no exponent was
    /// consumed, so the caller knows whether to keep an `Int` or mint a `Float`.
    ///
    /// The value is the whole literal's text parsed in one go, not the
    /// mantissa times `10^exp`: that product rounds twice, so
    /// `1.7976931348623157e308` (`float::MAX` as it prints) overflowed to
    /// infinity and `1e-7` could miss the nearest double by a bit. A
    /// float's text has to read back as the float that printed it.
    // r[impl mimas.num.text]
    fn with_exponent(&mut self, start: usize, mantissa: f64) -> Option<f64> {
        let source = self.source;
        let stream = &mut self.char_stream;
        if !stream.match_peek('e') {
            return None;
        }
        stream.match_peek_with(|c: char| c == '+' || c == '-');
        while stream.match_peek_with(|c: char| c.is_ascii_digit() || c == '_') {}
        // `e`, an optional sign, then the digits -- `i32` takes the sign itself
        let exp = stream.inspect_peeks()[1..].replace('_', "").parse::<i32>();
        match exp {
            Ok(exp) => {
                stream.chomp_peeks();
                let text = source[start..stream.position()].replace('_', "");
                Some(text.parse::<f64>().unwrap_or(mantissa * 10f64.powi(exp)))
            }
            Err(_) => {
                stream.reset_peeks();
                None
            }
        }
    }

    /// Creates a new Lexer, taking a string of mimas source.
    pub fn new(source: &'s str, file_id: FileId, file_name: String) -> Self {
        Self {
            source,
            char_stream: CharStream::new(source),
            file_id,
            file_name,
            offset: 0,
            errors: vec![],
            cut_short: false,
            shadow: shadow_table(source),
        }
    }

    /// Lex with one shadow set throughout: a file of a multi-file program
    /// (`load_files` computes it with `shadow_sets`), or an f-string
    /// interpolation's slice (its page's set).
    pub fn with_shadowed(mut self, shadowed: [bool; 3]) -> Self {
        self.shadow = vec![(0, shadowed)];
        self
    }

    /// OR `inherited` (what the files this one `use`s declare) into every page of this
    /// lexer's own table.
    pub fn with_inherited(mut self, inherited: [bool; 3]) -> Self {
        for (_, set) in &mut self.shadow {
            for k in 0..3 {
                set[k] |= inherited[k];
            }
        }
        self
    }

    /// Keep another lexer's whole table (a relex of the same source).
    pub(crate) fn with_shadow_table(mut self, table: Vec<(usize, [bool; 3])>) -> Self {
        self.shadow = table;
        self
    }

    pub(crate) fn shadow_table(&self) -> &[(usize, [bool; 3])] {
        &self.shadow
    }

    /// The shadow set in force at byte `pos` of this lexer's source.
    pub(crate) fn shadowed_at(&self, pos: usize) -> [bool; 3] {
        shadow_at(&self.shadow, pos)
    }

    /// Shifts every location the lexer produces by `offset`. For lexing a slice of a larger
    /// source (an f-string's interpolation) with spans that still point into the whole thing.
    pub(crate) fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Nonzero when this lexer is on a slice of a bigger source (an
    /// f-string interpolation) — `unit` decls are top-level only.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Takes the errors found since the last call, each with where it happened.
    pub fn take_errors(&mut self) -> Vec<(DiagBox, Location)> {
        std::mem::take(&mut self.errors)
    }

    /// Whether a string or comment that never closed took the rest of the source with it.
    pub(crate) fn cut_short(&self) -> bool {
        self.cut_short
    }

    /// Where the source ends.
    pub(crate) fn end(&self) -> Location {
        self.location(self.source.len(), self.source.len())
    }

    /// `start..end` of our own source, as a location in the whole thing.
    fn location(&self, start: usize, end: usize) -> Location {
        Location::new(
            self.file_id,
            Span::new(self.offset + start, self.offset + end),
        )
    }

    /// Reports whatever `start..body` opened as never closed. Only the opener is marked (the
    /// rest of the file would be otherwise).
    fn unterminated(&mut self, start: usize, body: usize, diag: impl FnOnce(Location) -> DiagBox) {
        let location = self.location(start, body);
        self.errors.push((diag(location), location));
        self.cut_short = true;
    }

    /// Chomps the string or f-string opened at `start`. `body` is everything after the opening
    /// quote, and `end` is where in it the closing quote is (when there is one). Returns what's
    /// between the quotes.
    fn construct_quoted(&mut self, start: usize, body: &'s str, end: Option<usize>) -> &'s str {
        fn chomp_to(stream: &mut CharStream, end: usize) {
            while stream.position() < end {
                stream.chomp();
            }
        }
        let body_start = self.source.len() - body.len();
        match end {
            Some(end) => {
                chomp_to(&mut self.char_stream, body_start + end + 1);
                &body[..end]
            }
            None => {
                self.unterminated(start, body_start, |location| {
                    UnterminatedString(location).into()
                });
                chomp_to(&mut self.char_stream, self.source.len());
                body
            }
        }
    }

    /// Chomps the `"""`-delimited docstring opened at `start`; `body` is everything after the
    /// opener. Inside a docstring a lone `"` is just a character and nothing escapes the closer,
    /// so the first `"""` ends it — same split Python's tokenizer makes.
    fn construct_docstring(&mut self, start: usize, body: &'s str) -> &'s str {
        let body_start = self.source.len() - body.len();
        match body.find("\"\"\"") {
            Some(end) => {
                let close = body_start + end + 3;
                while self.char_stream.position() < close {
                    self.char_stream.chomp();
                }
                &body[..end]
            }
            None => {
                self.unterminated(start, body_start, |location| {
                    UnterminatedString(location).into()
                });
                while self.char_stream.position() < self.source.len() {
                    self.char_stream.chomp();
                }
                body
            }
        }
    }

    /// Chomps the rest of the block comment whose `/*` is at `start`. Nests like Rust's, so
    /// commenting out a region that already contains a block comment closes at the right `*/`.
    /// Returns the whole comment.
    fn construct_block_comment(&mut self, start: usize) -> &'s str {
        // `prev` is cleared after each delimiter so a shared slash can't count twice --
        // `/*/` must not open-and-close, matching rustc.
        let mut depth = 1usize;
        let mut prev = None;
        while depth > 0 {
            let Some(c) = self.char_stream.chomp() else {
                self.unterminated(start, start + 2, |location| {
                    UnterminatedBlockComment(location).into()
                });
                break;
            };
            match (prev, c) {
                (Some('/'), '*') => {
                    depth += 1;
                    prev = None;
                }
                (Some('*'), '/') => {
                    depth -= 1;
                    prev = None;
                }
                _ => prev = Some(c),
            }
        }
        &self.source[start..self.char_stream.position()]
    }
}

impl<'s> Iterator for Lexer<'s> {
    type Item = Tok<TokKind<'s>>;

    fn next(&mut self) -> Option<Self::Item> {
        // chompy's trait makes `lex` a result, but ours has no way to fail
        self.lex().ok().flatten()
    }
}

impl<'s> Lex<'s, Tok<TokKind<'s>>, TokKind<'s>> for Lexer<'s> {
    fn source(&self) -> &'s str {
        self.source
    }

    fn file_id(&self) -> FileId {
        self.file_id
    }

    fn char_stream(&mut self) -> &mut CharStream<'s> {
        &mut self.char_stream
    }

    /// Overrides `Lex`'s default (alphanumeric-or-`_` only) to also allow internal `-`s, kebab-
    /// case style (`my-var`) -- but only when the `-` sits directly between two identifier chars
    /// with no whitespace on either side. `-` next to whitespace (`foo - bar`, `foo- bar`, `foo
    /// -bar`) is never absorbed here, so it's left for the ordinary `Minus` token exactly as
    /// before -- which is what actually disambiguates `foo-bar` (one identifier) from `foo - bar`
    /// (subtraction): write the operator with spaces, or it reads as part of the name.
    ///
    /// A leading `-`/`^` (nothing consumed yet) is never treated as a continuation, whatever
    /// follows it -- `can_extend` guards that, so a bare `-foo` still lexes as `Minus` then
    /// `Ident(foo)`, same as before. A `-`/`^` not immediately followed by an identifier char
    /// (`foo-)`, `foo- `, `x^ 2`, `foo--bar`) rolls back to just after the last committed
    /// segment via `reset_peeks`, leaving the operator itself untouched for the next `lex()`
    /// call. `^` joins a name the same way `foo-bar`'s `-` does: `x^2` is one ident, which the
    /// editor typesets raised; `x ^ 2` spaced still lexes `Caret` (the parser's misdirection
    /// points at `⊕`/`xor`/`.pow`).
    fn construct_ident(&mut self) -> Option<&'s str> {
        let is_continue = |c: char| c.is_alphanumeric() || c == '_' || is_symbol_ident(c);
        let stream = self.char_stream();
        let start = stream.position();
        loop {
            while stream.match_peek_with(is_continue) {}
            stream.chomp_peeks();
            let can_extend = stream.position() > start;
            if can_extend
                && stream.match_peek_with(|c: char| c == '-' || c == '^')
                && stream.match_peek_with(is_continue)
            {
                continue;
            }
            stream.reset_peeks();
            break;
        }
        let end = stream.position();
        (end > start).then(|| stream.slice(start..end))
    }

    fn lex(&mut self) -> ChompyResult<Option<Tok<TokKind<'s>>>> {
        // whitespace in one go -- massive amounts of whitespace can cause enormously deep stacks
        while self.char_stream.match_chomp_with(|c| c.is_whitespace()) {}
        let start = self.char_stream.position();
        let rest = &self.source[start..];
        let kind = if let Some(hex) = self.construct_hex("0x") {
            match hex {
                Ok(hex) => TokKind::Hex(hex),
                // a `0x` with no digits after it is still a hex literal, just an empty one
                Err(diag) => {
                    let location = self.location(start, self.char_stream.position());
                    self.errors.push((diag, location));
                    TokKind::Hex("")
                }
            }
        } else if let Some((len, text, is_float)) = grouped_number(rest) {
            // `1,000,000`, `1,234.50`: commas group thousands (r[impl mimas.num.groups])
            for _ in 0..len {
                self.char_stream.match_chomp_with(|_| true);
            }
            if is_float {
                let float = text.parse::<f64>().unwrap_or(f64::NAN);
                self.with_unit(TokKind::Float(float), float)
            } else {
                match text.parse::<i64>() {
                    Ok(int) => self.with_unit(TokKind::Int(int), int as f64),
                    Err(_) => {
                        let location = self.location(start, self.char_stream.position());
                        self.errors.push((IntegerOutOfRange(location).into(), location));
                        TokKind::Int(0)
                    }
                }
            }
        } else if let Some(mantissa) = self.construct_float(true, false) {
            let float = self.with_exponent(start, mantissa).unwrap_or(mantissa);
            self.with_unit(TokKind::Float(float), float)
        } else if rest.starts_with(|c: char| c.is_ascii_digit()) {
            match self.construct_integer(true) {
                // r[impl mimas.num.exponent]
                // `1e-6`, `2e3`: an exponent on an int mantissa makes a float;
                // `1e` or `1else` is an int then whatever follows, as before
                Some(int) => match self.with_exponent(start, int as f64) {
                    Some(float) => self.with_unit(TokKind::Float(float), float),
                    None => self.with_unit(TokKind::Int(int), int as f64),
                },
                // on a digit, the only way for that to fail is a number too big to hold
                None => {
                    let location = self.location(start, self.char_stream.position());
                    self.errors
                        .push((IntegerOutOfRange(location).into(), location));
                    TokKind::Int(0)
                }
            }
        } else if let Some(body) = rest.strip_prefix("\"\"\"") {
            // `"""…"""` — the Python-flavored docstring: a str literal
            // that may span lines and whose unassigned statement form
            // is documentation; the editor typesets it.
            TokKind::String(self.construct_docstring(start, body))
        } else if let Some(body) = rest.strip_prefix("f\"") {
            TokKind::FString(self.construct_quoted(start, body, fstring_end(body)))
        } else if let Some(body) = rest.strip_prefix('"') {
            TokKind::String(self.construct_quoted(start, body, string_end(body)))
        } else if let Some(ident) = self.construct_ident() {
            let kind = match ident {
                "self" => TokKind::SelfKeyword,
                "let" => TokKind::Let,
                "return" => TokKind::Return,
                "raise" => TokKind::Raise,
                "absolve" => TokKind::Absolve,
                "if" => TokKind::If,
                "else" => TokKind::Else,
                "null" => TokKind::Null,
                "struct" => TokKind::Struct,
                "pact" => TokKind::Pact,
                "impl" => TokKind::Impl,
                "const" => TokKind::Const,
                "pub" => TokKind::Pub,
                "module" => TokKind::Module,
                "break" => TokKind::Break,
                "collect" => TokKind::Collect,
                "fn" => TokKind::Fn,
                "true" => TokKind::True,
                "false" => TokKind::False,
                "for" => TokKind::For,
                "in" => TokKind::In,
                "match" => TokKind::Match,
                "continue" => TokKind::Continue,
                "enum" => TokKind::Enum,
                "loop" => TokKind::Loop,
                "while" => TokKind::While,
                "use" => TokKind::Use,
                // the word forms of the logical operators — the same tokens
                // as the symbols (C++'s alternative tokens). Source canonically
                // spells `&&`/`||`/`!`; the editor can show and accept the
                // words (a per-person option, apart from the code style).
                // A source that declares `fn and`/`fn or`/`fn not` takes the
                // word back as a name (see `shadowed_words`).
                "and" if !self.shadowed_at(start)[0] => TokKind::DoubleAmpersand,
                "or" if !self.shadowed_at(start)[1] => TokKind::DoublePipe,
                "not" if !self.shadowed_at(start)[2] => TokKind::NotWord,
                "int" => TokKind::TyKw(TyKw::Int),
                "float" => TokKind::TyKw(TyKw::Float),
                "str" => TokKind::TyKw(TyKw::Str),
                "bool" => TokKind::TyKw(TyKw::Bool),
                "" => TokKind::TyKw(TyKw::Bool),

                lexeme => TokKind::Ident(lexeme),
            };
            // r[impl mimas.str.tagged]
            // `tag"…"` -- a name glued to a string (no blank between) is a
            // tagged literal: `pat"bd sd"`, `keys"right@5 up@2"`. Keywords
            // never tag (`f"…"` was taken above as the f-string).
            match kind {
                TokKind::Ident(tag)
                    if self.source[self.char_stream.position()..].starts_with('"') =>
                {
                    self.match_chomp('"');
                    let body = &self.source[self.char_stream.position()..];
                    TokKind::Tagged(tag, self.construct_quoted(start, body, string_end(body)))
                }
                kind => kind,
            }
        } else {
            let Some(chr) = self.chomp() else {
                return Ok(None);
            };
            match chr {
                '.' => {
                    if self.match_chomp('.') {
                        if self.match_chomp('=') {
                            TokKind::DoubleDotEqual
                        } else {
                            TokKind::DoubleDot
                        }
                    } else {
                        TokKind::Dot
                    }
                }
                ',' => TokKind::Comma,
                ';' => TokKind::SemiColon,
                '(' => TokKind::LeftParenthesis,
                ')' => TokKind::RightParenthesis,
                '{' => TokKind::LeftBrace,
                '}' => TokKind::RightBrace,
                '[' => TokKind::LeftSquare,
                ']' => TokKind::RightSquare,
                '@' => TokKind::At,
                '#' => TokKind::Hash,
                '$' => TokKind::Dollar,
                '=' => {
                    if self.match_chomp('=') {
                        TokKind::DoubleEqual
                    } else if self.match_chomp('>') {
                        TokKind::FatArrow
                    } else {
                        TokKind::Equal
                    }
                }
                '-' => {
                    if self.match_chomp('=') {
                        TokKind::MinusEqual
                    } else if self.match_chomp('>') {
                        TokKind::Arrow
                    } else {
                        TokKind::Minus
                    }
                }
                '+' => {
                    if self.match_chomp('=') {
                        TokKind::PlusEqual
                    } else {
                        TokKind::Plus
                    }
                }
                '*' => {
                    if self.match_chomp('=') {
                        TokKind::StarEqual
                    } else {
                        TokKind::Star
                    }
                }
                '<' => {
                    if self.match_chomp('=') {
                        TokKind::LessEqual
                    } else if self.match_chomp('<') {
                        TokKind::DoubleLeftCaret
                    } else {
                        TokKind::Less
                    }
                }
                '>' => {
                    if self.match_chomp('=') {
                        TokKind::GreaterEqual
                    } else if self.match_chomp('>') {
                        TokKind::DoubleRightCaret
                    } else {
                        TokKind::Greater
                    }
                }
                '&' => {
                    if self.match_chomp('&') {
                        TokKind::DoubleAmpersand
                    } else if self.match_chomp('=') {
                        TokKind::AmpersandEqual
                    } else {
                        TokKind::Ampersand
                    }
                }
                '|' => {
                    if self.match_chomp('|') {
                        TokKind::DoublePipe
                    } else if self.match_chomp('=') {
                        TokKind::PipeEqual
                    } else if self.match_chomp('>') {
                        TokKind::PipeGreater
                    } else {
                        TokKind::Pipe
                    }
                }
                ':' => {
                    if self.match_chomp(':') {
                        TokKind::DoubleColon
                    } else {
                        TokKind::Colon
                    }
                }
                '~' => {
                    if self.match_chomp('/') {
                        if self.match_chomp('=') {
                            TokKind::TildeSlashEqual
                        } else {
                            TokKind::TildeSlash
                        }
                    } else if self.match_chomp('{') {
                        TokKind::TildeLeftBrace
                    } else if self.match_chomp('=') {
                        // `~=`, the typed spelling of `≈`
                        TokKind::Approx
                    } else {
                        TokKind::Tilde
                    }
                }
                '%' => {
                    if self.match_chomp('=') {
                        TokKind::PercentEqual
                    } else {
                        TokKind::Percent
                    }
                }
                '±' => TokKind::PlusMinus,
                '?' => {
                    if self.match_chomp('?') {
                        if self.match_chomp('=') {
                            TokKind::DoubleHookEqual
                        } else {
                            TokKind::DoubleHook
                        }
                    } else if self.match_chomp('.') {
                        TokKind::HookDot
                    } else if self.match_chomp('[') {
                        TokKind::HookLeftSquare
                    } else {
                        TokKind::Hook
                    }
                }
                '^' => {
                    if self.match_chomp('=') {
                        TokKind::CaretEqual
                    } else {
                        TokKind::Caret
                    }
                }
                // `⊕`/`⊻` both spell xor; `⊕=`/`⊻=` its assignment. They share
                // the one token each so the AST always prints `⊕`/`⊕=`.
                '⊕' | '⊻' => {
                    if self.match_chomp('=') {
                        TokKind::CircledPlusEqual
                    } else {
                        TokKind::CircledPlus
                    }
                }
                // `⍲`/`⍱` — APL's nand/nor glyphs; `a nand b`/`a nor b` are
                // the word spellings. No assignment forms.
                '⍲' => TokKind::Nand,
                '⍱' => TokKind::Nor,
                // r[impl mimas.op.glyphs]
                // the named operators (see `TokKind::Wedge`): symbols the
                // lexer knows and only a method gives meaning to
                '∧' => TokKind::Wedge,
                '∨' => TokKind::Vee,
                '⋅' | '·' => TokKind::DotOp,
                '×' => TokKind::Cross,
                '∘' => TokKind::Compose,
                '⊗' => TokKind::Tensor,
                '⊙' => TokKind::Odot,
                '∩' => TokKind::Intersect,
                '∪' => TokKind::Union,
                '⊖' => TokKind::Ominus,
                '⊎' => TokKind::Uplus,
                // r[impl mimas.op.approx]
                '≈' => TokKind::Approx,
                // r[impl mimas.op.cmp-glyphs]
                // the comparison and membership glyphs are the ASCII tokens:
                // the AST prints `!=`, `<=`, `>=`, `in`, `!in`
                '≠' => TokKind::BangEqual,
                '≤' => TokKind::LessEqual,
                '≥' => TokKind::GreaterEqual,
                '∈' => TokKind::In,
                '∉' => TokKind::NotIn,
                '!' => {
                    // `!in` is the operator only when the `in` ends there: `!inside` is `!`
                    // `inside`
                    let rest = &self.source()[self.char_stream().peek_position()..];
                    let not_in = rest.starts_with("in")
                        && !rest[2..].starts_with(|c: char| c.is_alphanumeric() || c == '_');
                    if self.match_chomp('=') {
                        TokKind::BangEqual
                    } else if not_in && self.chomp_pattern("in") {
                        TokKind::NotIn
                    } else {
                        TokKind::Bang
                    }
                }
                '/' => {
                    if self.match_chomp('/') {
                        let is_doc = self.match_chomp('/');
                        let stream = &mut self.char_stream;
                        loop {
                            while stream.match_chomp_with(|c| c != '\n' && c != '\r') {}
                            // carry on into the next line if it's the same kind of comment
                            stream.match_peek('\r');
                            stream.match_peek('\n');
                            stream.peek_while(|c| c == ' ' || c == '\t');
                            if stream.match_peek('/')
                                && stream.match_peek('/')
                                && stream.match_peek('/') == is_doc
                            {
                                stream.chomp_peeks();
                            } else {
                                stream.reset_peeks();
                                break;
                            }
                        }
                        let text = &self.source[start..self.char_stream.position()];
                        if is_doc {
                            TokKind::DocComment(text)
                        } else {
                            TokKind::Comment(text)
                        }
                    } else if self.match_chomp('*') {
                        TokKind::Comment(self.construct_block_comment(start))
                    } else if self.match_chomp('=') {
                        TokKind::SlashEqual
                    } else {
                        TokKind::Slash
                    }
                }
                _ => TokKind::Invalid(&self.source[start..self.char_stream.position()]),
            }
        };

        let location = self.location(start, self.char_stream.position());
        Ok(Some(Tok::new(kind, location)))
    }
}

// r[impl mimas.ident.symbols]
/// Symbols a name may be made of besides letters, digits and `_`: the
/// arrows, geometric shapes, miscellaneous symbols, dingbats and emoji
/// blocks, plus the variation selector and joiner emoji sequences need --
/// so `opp Sky { ☀, ☾ }` and `let 🌙 = 1` lex as names. The operator
/// symbols stay operators: `±` (Latin-1), `⊕`/`⊻` (Mathematical
/// Operators) and `⍲`/`⍱` (the APL slice of Miscellaneous Technical,
/// cut out of that block below) are not in any of these ranges.
pub fn is_symbol_ident(c: char) -> bool {
    matches!(
        c as u32,
        0x2190..=0x21FF        // arrows: ↑ ↗ ⇤ ⇥
            | 0x2300..=0x236F  // misc technical: ⌂ ⏮ ⏭ (below the APL run)
            | 0x2380..=0x23FF  // misc technical, above it
            | 0x25A0..=0x25FF  // geometric shapes: ● ◆ ▲
            | 0x2600..=0x27BF  // misc symbols + dingbats: ☀ ☾ ★ ➕ ➖
            | 0x27F0..=0x27FF  // supplemental arrows-A: ⟲ ⟳
            | 0x2900..=0x297F  // supplemental arrows-B: ⤡ ⤢ (the moves' transpose)
            | 0x2B00..=0x2BFF  // misc symbols and arrows: ⬤ ⬆ ⬇
            | 0x1F000..=0x1FAFF // emoji: 🌞 🌙 😀
            | 0xFE0F           // variation selector-16
            | 0x200D           // zero-width joiner
    )
}

/// Where the string whose body is the start of `text` has its closing quote.
fn string_end(text: &str) -> Option<usize> {
    let mut chars = text.char_indices();
    while let Some((at, c)) = chars.next() {
        match c {
            '\\' => {
                chars.next();
            }
            '"' => return Some(at),
            _ => {}
        }
    }
    None
}

/// Where the f-string whose body is the start of `text` has its closing quote. A `"` inside an
/// interpolation (`f"{m["k"]}"`) opens a nested string rather than closing the f-string.
fn fstring_end(text: &str) -> Option<usize> {
    let mut at = 0;
    loop {
        let rest = &text[at..];
        let mut chars = rest.chars();
        match chars.next()? {
            '"' => return Some(at),
            '\\' => at += 1 + chars.next().map_or(0, char::len_utf8),
            '{' if rest.starts_with("{{") => at += 2,
            '{' => match interp_end(&rest[1..]) {
                Some(end) => at += 1 + end + 1,
                // an interpolation that never closes holds no strings, so the next quote is ours
                None => return rest.find('"').map(|quote| at + quote),
            },
            c => at += c.len_utf8(),
        }
    }
}

/// Where the `}` closing an f-string interpolation is, given the text right after its `{`.
/// Braces nest, and one inside a nested string doesn't count.
pub(crate) fn interp_end(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut at = 0;
    loop {
        let c = text[at..].chars().next()?;
        at += c.len_utf8();
        match c {
            '}' if depth == 0 => return Some(at - 1),
            '}' => depth -= 1,
            '{' => depth += 1,
            '"' => {
                at += string_end(&text[at..])? + 1;
            }
            _ => {}
        }
    }
}

/// A number with thousands grouped by commas at the start of `rest`:
/// digits, then one or more `,ddd` (a comma and exactly three digits, not
/// followed by another digit), then an optional fraction and exponent. The
/// answer is its length in bytes, its text without the commas, and whether
/// it is a float. `None` when there is no comma group, so `[1, 2]` and
/// `f(a,b)` lex as before; `f(1,234)` is the one number 1234.
// r[impl mimas.num.groups]
fn grouped_number(rest: &str) -> Option<(usize, String, bool)> {
    let b = rest.as_bytes();
    let digit = |i: usize| i < b.len() && b[i].is_ascii_digit();
    let mut i = 0;
    while digit(i) || (i > 0 && i < b.len() && b[i] == b'_') {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    let mut groups = 0;
    while i < b.len() && b[i] == b',' && digit(i + 1) && digit(i + 2) && digit(i + 3) && !digit(i + 4) {
        i += 4;
        groups += 1;
    }
    if groups == 0 {
        return None;
    }
    let mut is_float = false;
    if i < b.len() && b[i] == b'.' && digit(i + 1) {
        is_float = true;
        i += 1;
        while digit(i) || (i < b.len() && b[i] == b'_') {
            i += 1;
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        if digit(j) {
            while digit(j) {
                j += 1;
            }
            is_float = true;
            i = j;
        }
    }
    let text: String = rest[..i].chars().filter(|c| *c != ',' && *c != '_').collect();
    Some((i, text, is_float))
}
