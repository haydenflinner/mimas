use crate::{Literal, UnaryOp};
use chompy::lex::TokenKind;
use std::fmt::Display;

/// An individual tok of mimas.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TokKind<'s> {
    Match,
    Break,
    Collect,
    Return,
    Raise,
    Absolve,
    Struct,
    Impl,
    Pact,
    Const,
    Pub,
    Module,
    At,
    Hash,
    Colon,
    DoubleColon,
    Dot,
    DoubleDot,
    DoubleDotEqual,
    LeftSquare,
    RightSquare,
    Enum,
    LeftBrace,
    RightBrace,
    TildeLeftBrace,
    LeftParenthesis,
    RightParenthesis,
    Comma,
    Ampersand,
    DoubleAmpersand,
    DoublePipe,
    Caret,
    Equal,
    DoubleEqual,
    BangEqual,
    Fn,
    SelfKeyword,
    Percent,
    PercentEqual,
    TildeSlash,
    TildeSlashEqual,
    Slash,
    Star,
    True,
    False,
    Plus,
    Minus,
    /// `±` — the plus-minus interval literal, `x ± d` / `x ± d%`.
    PlusMinus,
    Bang,
    /// The word `not` — prefix only, the same operator as a prefix `!`
    /// (never a postfix unwrap). `and`/`or` lex straight to `&&`/`||`.
    NotWord,
    Hook,
    DoubleHook,
    DoubleHookEqual,
    HookDot,
    HookLeftSquare,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Pipe,
    PipeGreater,
    SemiColon,
    If,
    Else,
    For,
    NotIn,
    In,
    Loop,
    While,
    Let,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PipeEqual,
    AmpersandEqual,
    CaretEqual,
    /// `⊕` (also `⊻`) — xor. `^` left the operator table for identifier
    /// continuation (`x^2` is a name); the parser's misdirection teaches it.
    CircledPlus,
    /// `⊕=` (also `⊻=`).
    CircledPlusEqual,
    Tilde,
    Dollar,
    DoubleLeftCaret,
    DoubleRightCaret,
    Continue,
    Null,
    Use,
    Arrow,
    FatArrow,
    TyKw(TyKw),
    Ident(&'s str),
    Int(i64),
    Float(f64),
    /// A number with a unit suffix, `25kW`: the value as written and the unit text.
    Quantity(f64, &'s str),
    String(&'s str),
    FString(&'s str),
    Hex(&'s str),
    Invalid(&'s str),
    Comment(&'s str),
    DocComment(&'s str),
    /// Ends every token stream the parser reads. The lexer never produces one.
    Eof,
}

#[mutants::skip]
impl Display for TokKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TokKind::Match => "match",
            TokKind::Break => "break",
            TokKind::Collect => "collect",
            TokKind::Return => "return",
            TokKind::Raise => "raise",
            TokKind::Absolve => "absolve",
            TokKind::Struct => "struct",
            TokKind::Pact => "pact",
            TokKind::Impl => "impl",
            TokKind::Const => "const",
            TokKind::Pub => "pub",
            TokKind::Module => "module",
            TokKind::At => "@",
            TokKind::Hash => "#",
            TokKind::Colon => ":",
            TokKind::DoubleColon => "::",
            TokKind::Dot => ".",
            TokKind::DoubleDot => "..",
            TokKind::DoubleDotEqual => "..=",
            TokKind::LeftSquare => "[",
            TokKind::RightSquare => "]",
            TokKind::Enum => "enum",
            TokKind::LeftBrace => "{",
            TokKind::RightBrace => "}",
            TokKind::TildeLeftBrace => "~{",
            TokKind::LeftParenthesis => "(",
            TokKind::RightParenthesis => ")",
            TokKind::Comma => ",",
            TokKind::Ampersand => "&",
            TokKind::DoubleAmpersand => "&&",
            TokKind::DoublePipe => "||",
            TokKind::Caret => "^",
            TokKind::Equal => "=",
            TokKind::DoubleEqual => "==",
            TokKind::BangEqual => "!=",
            TokKind::Fn => "fn",
            TokKind::SelfKeyword => "self",
            TokKind::Percent => "%",
            TokKind::PercentEqual => "%=",
            TokKind::TildeSlash => "~/",
            TokKind::TildeSlashEqual => "~/=",
            TokKind::Slash => "/",
            TokKind::Star => "*",
            TokKind::True => "true",
            TokKind::False => "false",
            TokKind::Plus => "+",
            TokKind::Minus => "-",
            TokKind::PlusMinus => "±",
            TokKind::Bang => "!",
            TokKind::NotWord => "not",
            TokKind::Hook => "?",
            TokKind::DoubleHook => "??",
            TokKind::DoubleHookEqual => "??=",
            TokKind::HookDot => "?.",
            TokKind::HookLeftSquare => "?[",
            TokKind::Greater => ">",
            TokKind::GreaterEqual => ">=",
            TokKind::Less => "<",
            TokKind::LessEqual => "<=",
            TokKind::Pipe => "|",
            TokKind::PipeGreater => "|>",
            TokKind::SemiColon => ";",
            TokKind::If => "if",
            TokKind::Else => "else",
            TokKind::For => "for",
            TokKind::In => "in",
            TokKind::NotIn => "!in",
            TokKind::Loop => "loop",
            TokKind::While => "while",
            TokKind::Let => "let",
            TokKind::PlusEqual => "+=",
            TokKind::MinusEqual => "-=",
            TokKind::StarEqual => "*=",
            TokKind::SlashEqual => "/=",
            TokKind::PipeEqual => "|=",
            TokKind::AmpersandEqual => "&=",
            TokKind::CaretEqual => "^=",
            TokKind::CircledPlus => "⊕",
            TokKind::CircledPlusEqual => "⊕=",
            TokKind::Tilde => "~",
            TokKind::Dollar => "$",
            TokKind::DoubleLeftCaret => "<<",
            TokKind::DoubleRightCaret => ">>",
            TokKind::Continue => "continue",
            TokKind::Null => "null",
            TokKind::Use => "use",
            TokKind::Arrow => "->",
            TokKind::FatArrow => "=>",
            TokKind::TyKw(tym) => return f.pad(&tym.to_string()),
            TokKind::Ident(iden) => iden,
            TokKind::Int(r) => return f.pad(&r.to_string()),
            TokKind::Float(r) => return f.pad(&r.to_string()),
            TokKind::Quantity(r, unit) => return f.pad(&format!("{r}{unit}")),
            TokKind::String(s) => return f.pad(&format!("\"{s}\"")),
            TokKind::FString(s) => return f.pad(&format!("f\"{s}\"")),
            TokKind::Hex(hex) => hex,
            TokKind::Invalid(_) => "INVALID_TOKEN",
            TokKind::Comment(s) => return f.pad(s.as_ref()),
            TokKind::DocComment(s) => return f.pad(s.as_ref()),
            TokKind::Eof => return f.pad("end of input"),
        };
        f.pad(s)
    }
}

impl TokenKind for TokKind<'_> {}

#[derive(Debug, Copy, PartialEq, Clone)]
pub enum TyKw {
    Int,
    Float,
    Str,
    Bool,
}

#[mutants::skip]
impl std::fmt::Display for TyKw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TyKw::Int => f.pad("int"),
            TyKw::Float => f.pad("float"),
            TyKw::Str => f.pad("str"),
            TyKw::Bool => f.pad("bool"),
        }
    }
}

// Parsing logic
impl TokKind<'_> {
    /// Returns if this can start a statement.
    pub(crate) fn starts_stmt(self) -> bool {
        self.is_stmt_keyword() || self.starts_expr()
    }

    /// Returns if this can start an item.
    pub(crate) fn starts_item(self) -> bool {
        matches!(
            self,
            TokKind::Hash
                | TokKind::Pub
                | TokKind::Fn
                | TokKind::Struct
                | TokKind::Pact
                | TokKind::Enum
                | TokKind::Impl
                | TokKind::Const
                | TokKind::Use
        )
    }

    /// Returns if this can start an expression.
    pub(crate) fn starts_expr(self) -> bool {
        Literal::try_from(self).is_ok()
            || UnaryOp::try_from(self).is_ok()
            || matches!(
                self,
                TokKind::Ident(_)
                    | TokKind::NotWord
                    | TokKind::SelfKeyword
                    | TokKind::TyKw(_)
                    | TokKind::FString(_)
                    | TokKind::LeftParenthesis
                    | TokKind::LeftSquare
                    | TokKind::LeftBrace
                    | TokKind::TildeLeftBrace
                    | TokKind::Dollar
                    | TokKind::Pipe
                    | TokKind::DoublePipe
                    | TokKind::Loop
                    | TokKind::While
                    | TokKind::For
                    | TokKind::If
                    | TokKind::Match
                    | TokKind::Return
                    | TokKind::Raise
                    | TokKind::Break
                    | TokKind::Collect
                    | TokKind::Continue
            )
    }

    /// Whether this can open a juxtaposition argument — atoms only: a
    /// name, a literal, or a parenthesised expression. `{`, `[`, `|`,
    /// keyword-exprs and unary heads stay out, so `Foo {` keeps its
    /// struct-literal read, `x [i]` stays an index, `f - x` stays
    /// subtraction, and `x if` stays a stray token. Operator words
    /// ([is_operator_word]) are not atoms — `x mod 2` and `a xor b` stay
    /// infix, and `a and b` / `x as int` keep their teaching errors.
    pub(crate) fn starts_juxta_arg(self) -> bool {
        !matches!(self, TokKind::Ident(name) if is_operator_word(name))
            && (Literal::try_from(self).is_ok()
                || matches!(
                    self,
                    TokKind::Ident(_) | TokKind::SelfKeyword | TokKind::LeftParenthesis
                ))
    }

    /// Returns if this can start a pattern.
    pub(crate) fn starts_pattern(self) -> bool {
        Literal::try_from(self).is_ok()
            || matches!(
                self,
                TokKind::Ident(_) | TokKind::LeftParenthesis | TokKind::Minus
            )
    }

    /// Returns if this can start a type annotation.
    pub(crate) fn starts_annotation(self) -> bool {
        matches!(
            self,
            TokKind::Ident(_)
                | TokKind::TyKw(_)
                | TokKind::LeftParenthesis
                | TokKind::LeftSquare
                | TokKind::TildeLeftBrace
        )
    }

    /// Returns if this is an identifier.
    pub(crate) fn is_ident(self) -> bool {
        matches!(self, TokKind::Ident(_))
    }

    /// Returns if this is a comment.
    pub fn is_comment(self) -> bool {
        matches!(self, TokKind::Comment(_) | TokKind::DocComment(_))
    }

    /// Returns if this is a keyword that starts a statement wherever it shows up. Recovery
    /// never skips these.
    pub(crate) fn is_stmt_keyword(self) -> bool {
        matches!(self, TokKind::Let | TokKind::Module) || self.starts_item()
    }

    /// Returns if this closes a group.
    pub(crate) fn is_closer(self) -> bool {
        matches!(
            self,
            TokKind::RightParenthesis | TokKind::RightSquare | TokKind::RightBrace
        )
    }

    /// Returns if this is a token recovery leaves where it is, for whatever's waiting on it.
    pub(crate) fn is_anchor(self) -> bool {
        self.ends_list() || self == TokKind::Comma
    }

    /// Returns if a list that finds this where an element should start is over. These can't
    /// be part of a list unless an element takes them, so something further out is waiting
    /// for each.
    pub(crate) fn ends_list(self) -> bool {
        self.is_boundary()
            || matches!(
                self,
                TokKind::Equal
                    | TokKind::Arrow
                    | TokKind::FatArrow
                    | TokKind::LeftBrace
                    | TokKind::In
                    | TokKind::Else
            )
    }

    /// Returns if no skip inside a list runs past this.
    pub(crate) fn is_boundary(self) -> bool {
        self.ends_stmt() || self.is_closer()
    }

    /// Returns if a broken statement never runs past this.
    pub(crate) fn ends_stmt(self) -> bool {
        matches!(
            self,
            TokKind::SemiColon | TokKind::RightBrace | TokKind::Eof
        ) || self.is_stmt_keyword()
    }

    /// Whether this can appear in a `unit` decl's spec — `= 3mi`, `= 1852 m/h`,
    /// `= 12`, `= 1 m/s^2`. Used to bound the same-line token collection;
    /// `unit_spec` decides whether the collected tokens are a valid quantity.
    pub(crate) fn is_unit_spec(self) -> bool {
        matches!(
            self,
            TokKind::Int(_) | TokKind::Float(_) | TokKind::Quantity(..) | TokKind::Ident(_)
                | TokKind::Star | TokKind::Slash | TokKind::Caret | TokKind::Minus
        )
    }
}

/// Words that read as operators but lex as names — mimas's own infix
/// words (`mod`, `xor`) and other languages' (`as`, `is`, `elif`, which get
/// a teaching error). They never join a juxtaposition, as head or
/// argument. (`and`/`or`/`not` aren't names at all: they lex as the
/// `&&`/`||`/prefix-`!` operators — see the lexer's keyword table.)
pub(crate) fn is_operator_word(name: &str) -> bool {
    matches!(name, "mod" | "xor" | "as" | "is" | "elif")
}
