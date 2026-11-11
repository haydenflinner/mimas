//! Laws as checks (`mimas.pact.law`). The `check |x, y| …` lines of a pact
//! run against every implementer. The parser already lowered each law to a
//! top-level `fn __law_<Pact>_<i><T: Pact>(x: T, …) -> bool`. This pass runs
//! once the files of a program parse. It gives each `impl Pact for T` one
//! check per law, and the check calls the law over the samples of `T`:
//!
//! - an enum (or `opp`) whose variants carry no data: every variant
//! - `int`, `float`, `bool`, `str`: a few fixed values
//! - any other type: `T::samples()`, which the type supplies
//!   (`impl T { fn samples() -> [T] { … } }`), or else a compile error
//!
//! The pass writes the checks as source and parses them under a file id of
//! their own (the returned `sources`). So a diagnostic in a check shows the
//! generated text, not an unrelated offset of the page. The statements join
//! the file of the impl, so names resolve as they do at the impl. Each check
//! has the name `law <Pact> for <T>: <law as written>`.

use crate::{
    Ast, ItemKind, Law, Member, Parser, StmtKind, TestCase,
    errors::Misdirection,
    lex::Lexer,
};
use miette::NamedSource;
use shared::Located;
use std::{collections::HashMap, sync::Arc};

/// Generated law sources start at this file id, one id per file: above the
/// shipped files' `1 << 20` and inside a 32-bit `usize` (the wasm build).
pub const LAW_FILE_BASE: usize = 1 << 28;

/// What [`expand_laws`] added: the generated files (id, name, text) for the
/// host's source map, and the errors (a type with no samples).
#[derive(Default)]
pub struct LawExpansion {
    pub sources: Vec<(usize, String, String)>,
    pub errors: Vec<shared::Error>,
}

/// Expands the laws of every pact visible in `user` and `shipped` into
/// checks on each `impl` in `user`.
// r[impl mimas.pact.law.impl] r[impl mimas.pact.samples]
pub fn expand_laws(user: &mut [Ast], shipped: &[Ast]) -> LawExpansion {
    let mut out = LawExpansion::default();
    let all = || user.iter().chain(shipped.iter());

    let mut laws: HashMap<String, Vec<Law>> = HashMap::new();
    let mut variants: HashMap<String, Vec<String>> = HashMap::new();
    let mut has_samples: std::collections::HashSet<String> = Default::default();
    for ast in all() {
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else { continue };
            match item.kind() {
                ItemKind::Pact(p) if !p.laws.is_empty() => {
                    laws.entry(p.name.lexeme.clone())
                        .or_default()
                        .extend(p.laws.iter().cloned());
                }
                ItemKind::Enum(e) => {
                    let plain = e
                        .members
                        .iter()
                        .all(|(_, m)| matches!(m, Member::Struct(f) if f.is_empty()));
                    if plain && e.type_params.is_empty() {
                        variants.insert(
                            e.head.lexeme.clone(),
                            e.members.iter().map(|(n, _)| n.lexeme.clone()).collect(),
                        );
                    }
                }
                ItemKind::Impl(i) if i.pact.is_none() => {
                    let samples = i.items.iter().any(|it| {
                        matches!(it.kind(), ItemKind::Function(f) if f.name.lexeme == "samples")
                    });
                    if samples {
                        has_samples.insert(i.target.lexeme.clone());
                    }
                }
                _ => {}
            }
        }
    }
    if laws.is_empty() {
        return out;
    }

    let mut next_id = LAW_FILE_BASE;
    for ast in user.iter_mut() {
        let mut text = String::new();
        let mut names: Vec<String> = vec![];
        let mut first_impl = None;
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else { continue };
            let ItemKind::Impl(imp) = item.kind() else { continue };
            let Some(pact) = &imp.pact else { continue };
            let Some(pact_laws) = laws.get(&pact.lexeme) else { continue };
            let ty = imp.target.lexeme.as_str();
            let samples = match ty {
                "int" => "[0, 1, -2, 7]".to_string(),
                "float" => "[0.0, 1.0, -2.5, 0.5, 3.0]".to_string(),
                "bool" => "[false, true]".to_string(),
                "str" => "[\"\", \"a\", \"hello\"]".to_string(),
                _ if variants.contains_key(ty) => format!(
                    "[{}]",
                    variants[ty]
                        .iter()
                        .map(|v| format!("{ty}::{v}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ if has_samples.contains(ty) => format!("{ty}::samples()"),
                _ => {
                    out.errors.push(
                        Misdirection {
                            src: ast.src(),
                            at: item.location().into(),
                            msg: format!(
                                "`{ty}` implements `{}`, whose laws need sample values of `{ty}`",
                                pact.lexeme
                            ),
                            label: format!(
                                "add a few: `impl {ty} {{ fn samples() -> [{ty}] {{ [ … ] }} }}`"
                            ),
                        }
                        .into(),
                    );
                    continue;
                }
            };
            first_impl.get_or_insert(item.location());
            for law in pact_laws {
                let k = names.len();
                names.push(format!("law {} for {ty}: {}", pact.lexeme, law.text));
                // nested loops over the samples, one per law parameter; a
                // failure names the values that broke the law
                let mut body = String::new();
                for p in &law.params {
                    body.push_str(&format!("for {p} in {samples} {{ "));
                }
                let shown: Vec<String> = law.params.iter().map(|p| format!("{p} = {{{p}}}")).collect();
                let quoted = law.text.replace('\\', "\\\\").replace('"', "\\\"").replace('{', "{{").replace('}', "}}");
                body.push_str(&format!(
                    "if !{}({}) {{ panic(f\"the law `{quoted}` fails for {}\"); }} ",
                    law.fn_name,
                    law.params.join(", "),
                    shown.join(", ")
                ));
                for _ in &law.params {
                    body.push_str("} ");
                }
                text.push_str(&format!(
                    "fn __law_check_{k}_{}() -> bool {{ {body}true }}\ncheck __law_check_{k}_{}()\n",
                    next_id - LAW_FILE_BASE,
                    next_id - LAW_FILE_BASE,
                ));
            }
        }
        if text.is_empty() {
            continue;
        }
        let id = next_id;
        next_id += 1;
        let name = format!("{} (laws)", ast.name());
        let (mut generated, errors) = Parser::new(Lexer::new(&text, id, name.clone())).into_ast();
        out.errors.extend(errors);
        // the checks take the law's own name, in impl order
        let mut names = names.into_iter();
        for stmt in generated.stmts_mut() {
            if let StmtKind::Item(item) = stmt.kind_mut()
                && let ItemKind::Tests(tests) = item.kind_mut()
            {
                for case in &mut tests.cases {
                    rename(case, names.next().unwrap_or_default());
                }
            }
        }
        ast.absorb(generated);
        out.sources.push((id, name, text));
        let _ = first_impl;
    }
    out
}

fn rename(case: &mut TestCase, name: String) {
    case.name.lexeme = name;
}

/// The generated sources as `NamedSource`s, for a host's source map.
pub fn law_sources(exp: &LawExpansion) -> impl Iterator<Item = (usize, NamedSource<Arc<str>>)> + '_ {
    exp.sources
        .iter()
        .map(|(id, name, text)| (*id, NamedSource::new(name.clone(), Arc::from(text.as_str()))))
}
