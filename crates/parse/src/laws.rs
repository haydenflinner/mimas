//! Laws as checks (`mimas.pact.law`). The `check |x, y| …` lines of a pact
//! run against every implementer. The parser already lowered each law to a
//! top-level `fn __law_<Pact>_<i><T: Pact>(x: T, …) -> bool`. This pass runs
//! once the files of a program parse. It gives each `impl Pact for T` one
//! check per law, and the check calls the law over the samples of `T`:
//!
//! - an enum (or `opp`) whose variants carry no data: every variant
//! - `int`, `float`, `bool`, `str`: a few fixed values
//! - `impl T { fn samples() -> [T] { … } }`: that list
//! - a type that implements `Arbitrary` (`arb.mim`): `T::gen(size)` draws
//!   off the program's random stream
//! - any other non-generic struct or enum whose fields and variant
//!   payloads are all generatable: a `gen`/`shrink` pair this pass writes
//! - anything else: a compile error naming what to add
//!
//! A witness that breaks a law shrinks before it's named: `x.shrink()` (or
//! the derived helper) lists simpler candidates and a greedy loop keeps
//! the first one that still breaks the law until none does, so a failure
//! reports a minimal counterexample the way the tape shrinker reports a
//! minimal tape. Generated samples come off the seeded stream, reseeded at
//! the top of each check — same seed, same samples, same minimal witness.
//!
//! The pass writes the checks as source and parses them under a file id of
//! their own (the returned `sources`). So a diagnostic in a check shows the
//! generated text, not an unrelated offset of the page. The statements join
//! the file of the impl, so names resolve as they do at the impl. Each check
//! has the name `law <Pact> for <T>: <law as written>`.

use crate::{
    Ast, FieldKey, ItemKind, Law, Member, Parser, StmtKind, StructField, TestCase,
    components::Annotation,
    errors::Misdirection,
    lex::Lexer,
};
use miette::NamedSource;
use shared::Located;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// Generated law sources start at this file id, one id per file: above the
/// shipped files' `1 << 20` and inside a 32-bit `usize` (the wasm build).
pub const LAW_FILE_BASE: usize = 1 << 28;

/// Draws per generated sample set — a two-parameter law pairs every draw,
/// so the count stays modest.
const DRAWS: usize = 24;
/// The `size` handed to `gen` for the top draws — ints land in ±size,
/// strings grow up to it.
const DRAW_SIZE: usize = 12;

/// What [`expand_laws`] added: the generated files (id, name, text) for the
/// host's source map, and the errors (a type with no samples).
#[derive(Default)]
pub struct LawExpansion {
    pub sources: Vec<(usize, String, String)>,
    pub errors: Vec<shared::Error>,
}

/// A non-generic struct or enum a `gen`/`shrink` pair can be derived for.
/// `Applied` annotations (`Pair<int>`) and generic defs stay out — a
/// derived helper would have to substitute type params it can't see.
#[derive(Clone)]
enum Adt {
    /// Field order, exactly as declared. Tuple-struct fields
    /// (`FieldKey::Int`) are kept but make the type non-derivable — the
    /// positional constructor isn't written here.
    Struct(Vec<StructField>),
    Enum(Vec<(String, Member)>),
}

/// Derived-generation state for one file's law checks: the shapes a
/// `gen`/`shrink` pair can be written for, which of them the file's checks
/// ended up needing, and the helpers minted for anonymous shapes (`[F]`,
/// `F?`, tuples — shapes an `impl` can't name, so the pact can't reach
/// them).
struct Gen<'a> {
    defs: &'a HashMap<String, Adt>,
    /// Types with a declared `impl Arbitrary for T`, std's or the program's.
    arb: &'a HashSet<String>,
    /// `file_id - LAW_FILE_BASE` — keeps helper names unique to the file.
    here: usize,
    /// Worklists of derived types whose `__arb_gen_*`/`__arb_shrink_*`
    /// helpers are still owed, each drained once per name.
    gen_needed: Vec<String>,
    gen_done: HashSet<String>,
    shrink_needed: Vec<String>,
    shrink_done: HashSet<String>,
    /// `Annotation` text → helper index, for `F?`/`[F]`/tuple shrinks.
    shapes: HashMap<String, usize>,
    shape_order: Vec<Annotation>,
    emitted_shapes: usize,
    /// `__arb_samples_<T>` fns already written this file.
    samples_done: HashSet<String>,
    /// A mint for block-local names (`__v0`, `__v1`, …) so nested shape
    /// gens never shadow a sibling's variable.
    fresh: usize,
}

impl<'a> Gen<'a> {
    fn fresh(&mut self) -> usize {
        self.fresh += 1;
        self.fresh
    }

    /// `gen` can be written for `ann`. `stack` holds the derived-type
    /// names being unfolded on this path; `carried` says a `?`/`[]` wraps
    /// the annotation, so meeting a name already on `stack` is fine — the
    /// derived gen spends a size step on it and the carrier bottoms out.
    /// A *bare* repeat is survivable only behind an enum's variant choice
    /// (its ground variants are the way out); a struct that must hold
    /// itself has no finite value at all.
    fn generatable(&self, ann: &Annotation, stack: &mut Vec<String>, carried: bool) -> bool {
        match ann {
            Annotation::Kw(_) => true,
            Annotation::Option(inner) => self.generatable(inner, stack, true),
            Annotation::Array(inner, dims) => {
                dims.is_empty() && self.generatable(inner, stack, true)
            }
            Annotation::Tuple(parts) => {
                parts.iter().all(|p| self.generatable(p, stack, carried))
            }
            Annotation::Ty(name) => {
                let n = name.lexeme.as_str();
                if self.arb.contains(n) {
                    return true;
                }
                let Some(def) = self.defs.get(n) else { return false };
                if stack.iter().any(|s| s == n) {
                    return carried || matches!(def, Adt::Enum(_));
                }
                stack.push(n.to_string());
                let ok = match def {
                    Adt::Struct(fields) => fields.iter().all(|f| {
                        matches!(f.name, FieldKey::Ident(_))
                            && self.generatable(&f.annotation, stack, false)
                    }),
                    Adt::Enum(variants) => {
                        variants.iter().all(|(_, m)| self.member_gen(m, stack))
                            && variants.iter().any(|(_, m)| self.member_ground(m))
                    }
                };
                stack.pop();
                ok
            }
            _ => false,
        }
    }

    /// Every payload annotation of one enum variant is generatable.
    fn member_gen(&self, member: &Member, stack: &mut Vec<String>) -> bool {
        match member {
            Member::Tuple(anns) => anns.iter().all(|a| self.generatable(a, stack, false)),
            Member::Struct(fields) => fields.iter().all(|f| {
                matches!(f.name, FieldKey::Ident(_))
                    && self.generatable(&f.annotation, stack, false)
            }),
        }
    }

    /// Constructible at `size <= 0` without calling a derived `gen`: the
    /// enum variants a recursive `gen` bottoms out on. `?` and `[]` are
    /// ground at any depth (null / empty at size 0); a named type only
    /// when it carries `Arbitrary` itself.
    fn member_ground(&self, member: &Member) -> bool {
        match member {
            Member::Tuple(anns) => anns.iter().all(|a| self.ground(a)),
            Member::Struct(fields) => fields.iter().all(|f| self.ground(&f.annotation)),
        }
    }

    fn ground(&self, ann: &Annotation) -> bool {
        match ann {
            Annotation::Kw(_) => true,
            Annotation::Option(_) | Annotation::Array(..) => true,
            Annotation::Tuple(parts) => parts.iter().all(|p| self.ground(p)),
            Annotation::Ty(name) => self.arb.contains(name.lexeme.as_str()),
            _ => false,
        }
    }

    /// The mimas expression that draws an `ann` value at `size` (an
    /// expression string — composite shapes size-step it). Assumes
    /// `generatable` already passed.
    fn gen_expr(&mut self, ann: &Annotation, size: &str) -> String {
        let here = self.here;
        match ann {
            Annotation::Kw(kw) => self.prim_gen(&kw.to_string(), size),
            Annotation::Ty(name) => {
                let n = name.lexeme.as_str();
                if self.arb.contains(n) {
                    format!("{n}::gen({size})")
                } else {
                    // each edge into a derived type spends a size step, so
                    // recursion through carriers bottoms out
                    self.gen_needed.push(n.to_string());
                    format!("__arb_gen_{n}_{here}({size} - 1)")
                }
            }
            Annotation::Option(inner) => {
                let inner = self.gen_expr(inner, &format!("({size} - 1)"));
                // null gets likelier as size runs out, bounding `T?` chains
                format!(
                    "if {size} <= 0 {{ null }} else if int::random(3) == 0 {{ null }} else {{ {inner} }}"
                )
            }
            Annotation::Array(inner, _) => {
                let inner = self.gen_expr(inner, &format!("({size} - 1)"));
                let f = self.fresh();
                format!(
                    "{{ let __v{f}: {ann} = []; let __n{f} = if {size} <= 0 {{ 0 }} else {{ int::random({size} + 1) }}; for __i{f} in __n{f} {{ __v{f}.push({inner}); }} __v{f} }}"
                )
            }
            Annotation::Tuple(parts) => {
                let parts: Vec<String> =
                    parts.iter().map(|p| self.gen_expr(p, size)).collect();
                format!("({})", parts.join(", "))
            }
            _ => "null".to_string(), // unreachable past `generatable`
        }
    }

    /// A draw for a `TyKw` — `int::gen(size)` where `arb.mim` shipped, an
    /// inline recipe otherwise, so generated code never names a pact a
    /// host's library didn't bring.
    fn prim_gen(&mut self, kw: &str, size: &str) -> String {
        if self.arb.contains(kw) {
            return format!("{kw}::gen({size})");
        }
        match kw {
            "int" => format!("(int::random(2 * ({size}) + 1) - ({size}))"),
            "float" => format!(
                "(float::random(2.0 * ({size}).to_float()) - ({size}).to_float())"
            ),
            "bool" => "bool::random()".to_string(),
            "str" => {
                let f = self.fresh();
                format!(
                    "{{ let __s{f} = \"\"; let __p{f} = \"abcd efgh ijkl mnop 0123\"; for __i{f} in int::random(({size}) + 1) {{ let __j{f} = int::random(__p{f}.len()); __s{f} = __s{f} + __p{f}[__j{f} .. __j{f} + 1]; }} __s{f} }}"
                )
            }
            _ => "null".to_string(),
        }
    }

    /// The `[ann]` of simpler candidates for `subj`, or `None` when the
    /// shape has nowhere simpler to go — the caller skips `None` fields.
    fn shrink_expr(&mut self, ann: &Annotation, subj: &str) -> Option<String> {
        let here = self.here;
        match ann {
            Annotation::Kw(kw) if self.arb.contains(kw.to_string().as_str()) => {
                Some(format!("{subj}.shrink()"))
            }
            Annotation::Kw(_) => None,
            Annotation::Ty(name) => {
                let n = name.lexeme.as_str();
                if self.arb.contains(n) {
                    Some(format!("{subj}.shrink()"))
                } else if self.defs.contains_key(n) {
                    self.shrink_needed.push(n.to_string());
                    Some(format!("__arb_shrink_{n}_{here}({subj})"))
                } else {
                    None
                }
            }
            Annotation::Option(_) | Annotation::Tuple(_) => {
                let i = self.shape_index(ann);
                Some(format!("__arb_shrinkshape_{i}_{here}({subj})"))
            }
            Annotation::Array(_, dims) if dims.is_empty() => {
                let i = self.shape_index(ann);
                Some(format!("__arb_shrinkshape_{i}_{here}({subj})"))
            }
            _ => None,
        }
    }

    /// The helper index for an anonymous shape's shrink fn — minted on
    /// first use, emitted by [`Gen::drain`].
    fn shape_index(&mut self, ann: &Annotation) -> usize {
        let sig = ann.to_string();
        if let Some(i) = self.shapes.get(&sig) {
            return *i;
        }
        let i = self.shape_order.len();
        self.shapes.insert(sig, i);
        self.shape_order.push(ann.clone());
        i
    }

    /// Every helper the checks queued — derived `gen`/`shrink` helpers and
    /// the anonymous-shape shrinks — until all three worklists sit empty
    /// (each helper can owe the other kind).
    fn drain(&mut self, out: &mut String) {
        loop {
            let mut progressed = false;
            while let Some(n) = self.gen_needed.pop() {
                if self.gen_done.insert(n.clone()) {
                    self.emit_gen(&n, out);
                    progressed = true;
                }
            }
            while let Some(n) = self.shrink_needed.pop() {
                if self.shrink_done.insert(n.clone()) {
                    self.emit_shrink(&n, out);
                    progressed = true;
                }
            }
            while self.emitted_shapes < self.shape_order.len() {
                let i = self.emitted_shapes;
                let ann = self.shape_order[i].clone();
                self.emitted_shapes += 1;
                self.emit_shape(&ann, i, out);
                progressed = true;
            }
            if !progressed {
                return;
            }
        }
    }

    /// `__arb_gen_<n>` for a derived type: clamp `size`, then build the
    /// constructor field-first / variant-first.
    fn emit_gen(&mut self, n: &str, out: &mut String) {
        let here = self.here;
        match self.defs[n].clone() {
            Adt::Struct(fields) => {
                let mut ctor = String::new();
                for (ix, f) in fields.iter().enumerate() {
                    let e = self.gen_expr(&f.annotation, "__sz");
                    if ix > 0 {
                        ctor.push_str(", ");
                    }
                    ctor.push_str(&format!("{} = {e}", f.name));
                }
                out.push_str(&format!(
                    "fn __arb_gen_{n}_{here}(size: int) -> {n} {{
                        let __sz = if size < 0 {{ 0 }} else {{ size }};
                        {n} {{ {ctor} }}
                    }}\n"
                ));
            }
            Adt::Enum(variants) => {
                let n_all = variants.len();
                let ground_ix: Vec<usize> = variants
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, m))| self.member_ground(m))
                    .map(|(i, _)| i)
                    .collect();
                let mut arms = String::new();
                for (i, (vname, m)) in variants.iter().enumerate() {
                    let ctor = self.variant_ctor(n, vname, m, "__sz");
                    let pat = if i + 1 == n_all {
                        "_".to_string()
                    } else {
                        i.to_string()
                    };
                    arms.push_str(&format!("{pat} => {ctor}, "));
                }
                // at size 0 only ground variants are reachable — the
                // recursive ones would walk off the bottom
                let pick = if ground_ix.len() == n_all {
                    format!("let __k = int::random({n_all});")
                } else {
                    format!(
                        "let __safe: [int] = [{}]; let __k = if __sz <= 0 {{ __safe[int::random({})] }} else {{ int::random({n_all}) }};",
                        ground_ix
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                        ground_ix.len()
                    )
                };
                out.push_str(&format!(
                    "fn __arb_gen_{n}_{here}(size: int) -> {n} {{
                        let __sz = if size < 0 {{ 0 }} else {{ size }};
                        {pick}
                        match __k {{ {arms} }}
                    }}\n"
                ));
            }
        }
    }

    /// `__arb_shrink_<n>` for a derived type: a struct walks each field
    /// with the rest held; an enum tries the unit variants first, then
    /// each payload position.
    fn emit_shrink(&mut self, n: &str, out: &mut String) {
        let here = self.here;
        match self.defs[n].clone() {
            Adt::Struct(fields) => {
                let mut body = String::new();
                for (ix, f) in fields.iter().enumerate() {
                    let name = f.name.to_string();
                    let Some(cand) = self.shrink_expr(&f.annotation, &format!("v.{name}"))
                    else {
                        continue;
                    };
                    let rebuilt: Vec<String> = fields
                        .iter()
                        .enumerate()
                        .map(|(j, g)| {
                            let k = g.name.to_string();
                            if j == ix {
                                format!("{k} = __c")
                            } else {
                                format!("{k} = v.{k}")
                            }
                        })
                        .collect();
                    body.push_str(&format!(
                        "for __c in {cand} {{ out.push({n} {{ {} }}); }}\n",
                        rebuilt.join(", ")
                    ));
                }
                out.push_str(&format!(
                    "fn __arb_shrink_{n}_{here}(v: {n}) -> [{n}] {{
                        let out: [{n}] = [];
                        {body}
                        out
                    }}\n"
                ));
            }
            Adt::Enum(variants) => {
                let units: Vec<String> = variants
                    .iter()
                    .filter_map(|(vname, m)| match m {
                        Member::Struct(f) if f.is_empty() => Some(format!("{n}::{vname}")),
                        Member::Tuple(a) if a.is_empty() => Some(format!("{n}::{vname}")),
                        _ => None,
                    })
                    .collect();
                let mut body = String::new();
                if !units.is_empty() {
                    body.push_str(&format!(
                        "let __us: [{n}] = [{}]; for __u in __us {{ if __u != v {{ out.push(__u); }} }}\n",
                        units.join(", ")
                    ));
                }
                let mut arms = String::new();
                for (vname, m) in variants.iter() {
                    match m {
                        Member::Tuple(anns) if !anns.is_empty() => {
                            let binds: Vec<String> =
                                (0..anns.len()).map(|i| format!("__p{i}")).collect();
                            let mut loops = String::new();
                            for (ix, a) in anns.iter().enumerate() {
                                let Some(cand) = self.shrink_expr(a, &format!("__p{ix}"))
                                else {
                                    continue;
                                };
                                let rebuilt: Vec<String> = binds
                                    .iter()
                                    .enumerate()
                                    .map(|(j, b)| {
                                        if j == ix { "__c".to_string() } else { b.clone() }
                                    })
                                    .collect();
                                loops.push_str(&format!(
                                    "for __c in {cand} {{ out.push({n}::{vname}({})); }}\n",
                                    rebuilt.join(", ")
                                ));
                            }
                            if !loops.is_empty() {
                                arms.push_str(&format!(
                                    "{n}::{vname}({}) => {{ {loops} }}, ",
                                    binds.join(", ")
                                ));
                            }
                        }
                        Member::Struct(fields)
                            if !fields.is_empty()
                                && fields
                                    .iter()
                                    .all(|f| matches!(f.name, FieldKey::Ident(_))) =>
                        {
                            let binds: Vec<String> =
                                fields.iter().map(|f| f.name.to_string()).collect();
                            let mut loops = String::new();
                            for (ix, f) in fields.iter().enumerate() {
                                let fname = f.name.to_string();
                                let Some(cand) = self.shrink_expr(&f.annotation, &fname)
                                else {
                                    continue;
                                };
                                let rebuilt: Vec<String> = fields
                                    .iter()
                                    .enumerate()
                                    .map(|(j, g)| {
                                        let gn = g.name.to_string();
                                        if j == ix {
                                            format!("{gn} = __c")
                                        } else {
                                            format!("{gn} = {gn}")
                                        }
                                    })
                                    .collect();
                                loops.push_str(&format!(
                                    "for __c in {cand} {{ out.push({n}::{vname} {{ {} }}); }}\n",
                                    rebuilt.join(", ")
                                ));
                            }
                            if !loops.is_empty() {
                                arms.push_str(&format!(
                                    "{n}::{vname} {{ {} }} => {{ {loops} }}, ",
                                    binds.join(", ")
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                if !arms.is_empty() {
                    body.push_str(&format!("match v {{ {arms} _ => () }}\n"));
                }
                out.push_str(&format!(
                    "fn __arb_shrink_{n}_{here}(v: {n}) -> [{n}] {{
                        let out: [{n}] = [];
                        {body}
                        out
                    }}\n"
                ));
            }
        }
    }

    /// `E::V`, `E::V(<draws>)` or `E::V { f = <draw>, .. }` — one enum
    /// constructor with fresh payloads at `size`.
    fn variant_ctor(&mut self, en: &str, vname: &str, member: &Member, size: &str) -> String {
        match member {
            Member::Tuple(anns) => {
                if anns.is_empty() {
                    return format!("{en}::{vname}");
                }
                let args: Vec<String> =
                    anns.iter().map(|a| self.gen_expr(a, size)).collect();
                format!("{en}::{vname}({})", args.join(", "))
            }
            Member::Struct(fields) => {
                if fields.is_empty() {
                    return format!("{en}::{vname}");
                }
                let args: Vec<String> = fields
                    .iter()
                    .map(|f| format!("{} = {}", f.name, self.gen_expr(&f.annotation, size)))
                    .collect();
                format!("{en}::{vname} {{ {} }}", args.join(", "))
            }
        }
    }

    /// One `__arb_shrinkshape_<i>` helper for an `F?`/`[F]`/tuple `ann`.
    fn emit_shape(&mut self, ann: &Annotation, i: usize, out: &mut String) {
        let here = self.here;
        match ann {
            Annotation::Option(inner) => {
                // toward `null` first, then whatever the payload can lose
                let inner_loop = self
                    .shrink_expr(inner, "v!")
                    .map(|c| {
                        format!(
                            "for __c in {c} {{ let __w: {ann} = __c; out.push(__w); }}"
                        )
                    })
                    .unwrap_or_default();
                out.push_str(&format!(
                    "fn __arb_shrinkshape_{i}_{here}(v: {ann}) -> [{ann}] {{
                        if v == null {{ return []; }}
                        let out: [{ann}] = [null];
                        {inner_loop}
                        out
                    }}\n"
                ));
            }
            Annotation::Array(inner, _) => {
                // toward shorter first (`[]`, then each single deletion),
                // then each element with the rest held
                let mut elem = String::new();
                if let Some(cand) = self.shrink_expr(inner, "v[__i]") {
                    elem = format!(
                        "for __i in v.len() {{
                            for __c in {cand} {{
                                let __d: {ann} = [];
                                __d.extend(v);
                                __d[__i] = __c;
                                out.push(__d);
                            }}
                        }}"
                    );
                }
                out.push_str(&format!(
                    "fn __arb_shrinkshape_{i}_{here}(v: {ann}) -> [{ann}] {{
                        let out: [{ann}] = [];
                        let __e: {ann} = [];
                        out.push(__e);
                        for __i in v.len() {{
                            let __d: {ann} = [];
                            __d.extend(v.slice(0, __i));
                            __d.extend(v.slice_from(__i + 1));
                            out.push(__d);
                        }}
                        {elem}
                        out
                    }}\n"
                ));
            }
            Annotation::Tuple(parts) => {
                let mut loops = String::new();
                for (ix, p) in parts.iter().enumerate() {
                    let Some(cand) = self.shrink_expr(p, &format!("v.{ix}")) else {
                        continue;
                    };
                    let rebuilt: Vec<String> = (0..parts.len())
                        .map(|j| {
                            if j == ix {
                                "__c".to_string()
                            } else {
                                format!("v.{j}")
                            }
                        })
                        .collect();
                    loops.push_str(&format!(
                        "for __c in {cand} {{ out.push(({})); }}",
                        rebuilt.join(", ")
                    ));
                }
                out.push_str(&format!(
                    "fn __arb_shrinkshape_{i}_{here}(v: {ann}) -> [{ann}] {{
                        let out: [{ann}] = [];
                        {loops}
                        out
                    }}\n"
                ));
            }
            _ => {}
        }
    }
}

/// Expands the laws of every pact visible in `user` and `shipped` into
/// checks on each `impl` in `user`.
// r[impl mimas.pact.law.impl] r[impl mimas.pact.samples] r[impl mimas.pact.samples.arb]
pub fn expand_laws(user: &mut [Ast], shipped: &[Ast]) -> LawExpansion {
    let mut out = LawExpansion::default();
    let all = || user.iter().chain(shipped.iter());

    let mut laws: HashMap<String, Vec<Law>> = HashMap::new();
    let mut variants: HashMap<String, Vec<String>> = HashMap::new();
    let mut has_samples: HashSet<String> = Default::default();
    // `impl Arbitrary for T`, std's or the program's — `T::gen`/`x.shrink()`
    let mut arb: HashSet<String> = Default::default();
    // non-generic structs and enums, for derived generation
    let mut defs: HashMap<String, Adt> = HashMap::new();
    for ast in all() {
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else { continue };
            match item.kind() {
                ItemKind::Pact(p) if !p.laws.is_empty() => {
                    laws.entry(p.name.lexeme.clone())
                        .or_default()
                        .extend(p.laws.iter().cloned());
                }
                ItemKind::Struct(s) if s.type_params.is_empty() => {
                    defs.insert(s.name.lexeme.clone(), Adt::Struct(s.fields.clone()));
                }
                ItemKind::Enum(e) if e.type_params.is_empty() => {
                    let plain = e
                        .members
                        .iter()
                        .all(|(_, m)| matches!(m, Member::Struct(f) if f.is_empty()));
                    if plain {
                        variants.insert(
                            e.head.lexeme.clone(),
                            e.members.iter().map(|(n, _)| n.lexeme.clone()).collect(),
                        );
                    }
                    defs.insert(
                        e.head.lexeme.clone(),
                        Adt::Enum(
                            e.members
                                .iter()
                                .map(|(n, m)| (n.lexeme.clone(), m.clone()))
                                .collect(),
                        ),
                    );
                }
                ItemKind::Impl(i) => match &i.pact {
                    None => {
                        let samples = i.items.iter().any(|it| {
                            matches!(it.kind(), ItemKind::Function(f) if f.name.lexeme == "samples")
                        });
                        if samples {
                            has_samples.insert(i.target.lexeme.clone());
                        }
                    }
                    Some(p) if p.lexeme == "Arbitrary" => {
                        arb.insert(i.target.lexeme.clone());
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    if laws.is_empty() {
        return out;
    }

    let mut next_id = LAW_FILE_BASE;
    for ast in user.iter_mut() {
        let here = next_id - LAW_FILE_BASE;
        let mut g = Gen {
            defs: &defs,
            arb: &arb,
            here,
            gen_needed: vec![],
            gen_done: HashSet::new(),
            shrink_needed: vec![],
            shrink_done: HashSet::new(),
            shapes: HashMap::new(),
            shape_order: vec![],
            emitted_shapes: 0,
            samples_done: HashSet::new(),
            fresh: 0,
        };
        let mut text = String::new();
        // helpers the checks rely on — sample fns, minimal-witness fns,
        // derived `gen`/`shrink` pairs, shape shrinks — collected beside
        // `text` and appended before the generated file is parsed
        let mut helpers = String::new();
        let mut names: Vec<String> = vec![];
        for stmt in ast.stmts() {
            let StmtKind::Item(item) = stmt.kind() else { continue };
            let ItemKind::Impl(imp) = item.kind() else { continue };
            let Some(pact) = &imp.pact else { continue };
            let Some(pact_laws) = laws.get(&pact.lexeme) else { continue };
            let ty = imp.target.lexeme.as_str();
            let ty_ann = Annotation::Ty(imp.target.clone());
            // how a witness of `ty` shrinks — `x.shrink()` where `ty`
            // implements `Arbitrary`, a derived helper where the pass can
            // write one, or nothing: the report names the raw witness
            let shrink: Option<String> = if arb.contains(ty) {
                Some("{p}.shrink()".to_string())
            } else if g.generatable(&ty_ann, &mut vec![], false) {
                g.shrink_needed.push(ty.to_string());
                Some(format!("__arb_shrink_{ty}_{here}({{p}})"))
            } else {
                None
            };
            let mut generated = false;
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
                _ if g.generatable(&ty_ann, &mut vec![], false) => {
                    generated = true;
                    if g.samples_done.insert(ty.to_string()) {
                        let draw = if arb.contains(ty) {
                            format!("{ty}::gen({DRAW_SIZE})")
                        } else {
                            g.gen_needed.push(ty.to_string());
                            format!("__arb_gen_{ty}_{here}({DRAW_SIZE})")
                        };
                        helpers.push_str(&format!(
                            "fn __arb_samples_{ty}_{here}() -> [{ty}] {{
                                let __ss: [{ty}] = [];
                                for __i in {DRAWS} {{ __ss.push({draw}); }}
                                __ss
                            }}\n"
                        ));
                    }
                    format!("__arb_samples_{ty}_{here}()")
                }
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
                                "add a few: `impl {ty} {{ fn samples() -> [{ty}] {{ [ … ] }} }}`, or make its parts generatable with `impl Arbitrary for {ty}`"
                            ),
                        }
                        .into(),
                    );
                    continue;
                }
            };
            for law in pact_laws {
                let k = names.len();
                names.push(format!("law {} for {ty}: {}", pact.lexeme, law.text));
                // the law's body typed for this impl: `x ≈ x` on a `float`
                // is `float`'s `approx`, chosen now, not by a runtime test of
                // each implementer (which wasmgen can't make on a primitive)
                let typed: Vec<String> =
                    law.params.iter().map(|p| format!("{p}: {ty}")).collect();
                text.push_str(&format!(
                    "fn __law_impl_{k}_{here}({}) -> bool {{ {} }}\n",
                    typed.join(", "),
                    law.body
                ));
                let args = law.params.join(", ");
                let quoted = law
                    .text
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('{', "{{")
                    .replace('}', "}}");
                // one list of samples, shared by every parameter loop
                let mut body = format!("let __ss = {samples}; ");
                for p in &law.params {
                    body.push_str(&format!("for {p} in __ss {{ "));
                }
                match &shrink {
                    Some(tmpl) => {
                        // `__law_min` walks each parameter's candidates in
                        // turn until none still breaks the law — the
                        // reported values are the minimal counterexample
                        let mut min_loops = String::new();
                        for (i, p) in law.params.iter().enumerate() {
                            let mut call = law.params.clone();
                            call[i] = format!("__c{i}");
                            min_loops.push_str(&format!(
                                "for __c{i} in {} {{ if !__law_impl_{k}_{here}({}) && __c{i} != {p} {{ {p} = __c{i}; __moved = true; break; }} }} ",
                                tmpl.replace("{p}", p),
                                call.join(", ")
                            ));
                        }
                        let (ret, given) = if law.params.len() == 1 {
                            (ty.to_string(), args.clone())
                        } else {
                            let tys: Vec<&str> = law.params.iter().map(|_| ty).collect();
                            (format!("({})", tys.join(", ")), format!("({args})"))
                        };
                        // a user-written `shrink` owes no promise of a strictly
                        // shrinking chain -- equal candidates are skipped, and
                        // the step cap ends a ping-pong (a -> b -> a) walk
                        helpers.push_str(&format!(
                            "fn __arb_min_{k}_{here}({}) -> {ret} {{
                                for __step in 4096 {{
                                    let __moved = false;
                                    {min_loops}
                                    if !__moved {{ break; }}
                                }}
                                {given}
                            }}\n",
                            typed.join(", ")
                        ));
                        let ws: Vec<String> =
                            (0..law.params.len()).map(|i| format!("__w{i}")).collect();
                        let bound = if ws.len() == 1 {
                            ws[0].clone()
                        } else {
                            format!("({})", ws.join(", "))
                        };
                        let shown: Vec<String> = law
                            .params
                            .iter()
                            .enumerate()
                            .map(|(i, p)| format!("{p} = {{{}}}", ws[i]))
                            .collect();
                        body.push_str(&format!(
                            "if !__law_impl_{k}_{here}({args}) {{
                                let {bound} = __arb_min_{k}_{here}({args});
                                panic(f\"the law `{quoted}` fails for {}\");
                            }} ",
                            shown.join(", ")
                        ));
                    }
                    None => {
                        let shown: Vec<String> = law
                            .params
                            .iter()
                            .map(|p| format!("{p} = {{{p}}}"))
                            .collect();
                        body.push_str(&format!(
                            "if !__law_impl_{k}_{here}({args}) {{ panic(f\"the law `{quoted}` fails for {}\"); }} ",
                            shown.join(", ")
                        ));
                    }
                }
                for _ in &law.params {
                    body.push_str("} ");
                }
                // same seed → same draws → same minimal witness
                let seed = if generated {
                    format!("random::seed({}); ", 0x5EED5EED_u64.wrapping_add(k as u64))
                } else {
                    String::new()
                };
                text.push_str(&format!(
                    "fn __law_check_{k}_{here}() -> bool {{ {seed}{body}true }}\ncheck __law_check_{k}_{here}()\n",
                ));
            }
        }
        g.drain(&mut helpers);
        text.push_str(&helpers);
        if text.is_empty() {
            continue;
        }
        let id = next_id;
        next_id += 1;
        let name = format!("{} (laws)", ast.name());
        let (mut generated_ast, errors) =
            Parser::new_extending(Lexer::new(&text, id, name.clone())).into_ast();
        out.errors.extend(errors);
        // the checks take the law's own name, in impl order
        let mut names = names.into_iter();
        for stmt in generated_ast.stmts_mut() {
            if let StmtKind::Item(item) = stmt.kind_mut()
                && let ItemKind::Tests(tests) = item.kind_mut()
            {
                for case in &mut tests.cases {
                    rename(case, names.next().unwrap_or_default());
                }
            }
        }
        ast.absorb(generated_ast);
        out.sources.push((id, name, text));
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

#[cfg(test)]
mod dump_tests {
    #[test]
    fn dump_generated() {
        let src = "pact Small {
            fn v(self) -> int;
            check |x| x.v() <= 3
        }
        struct S { v: int }
        impl Small for S { fn v(self) -> int { self.v } }";
        let (ast, errs) = crate::Parser::new_extending(crate::lex::Lexer::new(src, 1, "t".into())).into_ast();
        assert!(errs.is_empty(), "{errs:?}");
        let exp = super::expand_laws(&mut [ast], &[]);
        for (_, name, text) in &exp.sources {
            println!("=== {name} ===\n{text}");
        }
        for e in &exp.errors {
            println!("ERR {e:?}");
        }
    }

    #[test]
    fn dump_enum() {
        let src = "enum D { N, V(int) }\nstruct S { v: int }";
        let (ast, errs) = crate::Parser::new_extending(crate::lex::Lexer::new(src, 1, "t".into())).into_ast();
        assert!(errs.is_empty(), "{errs:?}");
        let _ = ast;
    }
}
