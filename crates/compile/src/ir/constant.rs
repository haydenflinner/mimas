use itertools::Itertools;
use parse::{Expr, ExprKind, Literal};
use shared::StrId;
use solve::{components::DecId, utils::reduce_simple};

use super::{Inst, Ir, IrDisplay};

// the discriminants here are the on-wire tag -- keep them in sync with ConstantCode.
#[repr(u8)]
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Bool(bool) = 0,
    Int(i64) = 1,
    Float(f64) = 2,
    Str(StrId) = 3,
    Array(Vec<Constant>) = 4,
    Null = 5,
    /// An exact decimal, as its 64 bits (`shared::dec::Dec::to_bits`).
    Dec(i64) = 6,
}

/// By value, a float by its bits (for `bodycache`'s keys: two constants
/// that hash alike must compile alike, and `0.0`/`-0.0` do not).
impl std::hash::Hash for Constant {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        std::mem::discriminant(self).hash(h);
        match self {
            Constant::Bool(b) => b.hash(h),
            Constant::Int(i) | Constant::Dec(i) => i.hash(h),
            Constant::Float(f) => f.to_bits().hash(h),
            Constant::Str(s) => s.hash(h),
            Constant::Array(a) => a.hash(h),
            Constant::Null => {}
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ConstantCode {
    Bool = 0,
    Int = 1,
    Float = 2,
    Str = 3,
    Array = 4,
    Null = 5,
    Dec = 6,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CachedConstant {
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(StrId),
    Null,
    Dec(i64),
}

impl CachedConstant {
    /// Back to a full [Constant] so the prologue can emit its load.
    pub(crate) fn to_constant(&self) -> Constant {
        match self {
            CachedConstant::Bool(b) => Constant::Bool(*b),
            CachedConstant::Int(i) => Constant::Int(*i),
            CachedConstant::Float(bits) => Constant::Float(f64::from_bits(*bits)),
            CachedConstant::Str(s) => Constant::Str(*s),
            CachedConstant::Null => Constant::Null,
            CachedConstant::Dec(b) => Constant::Dec(*b),
        }
    }
}

#[inline(always)]
pub fn constant_discriminant(c: &Constant) -> u8 {
    unsafe { *(c as *const Constant as *const u8) }
}

impl Constant {
    pub fn byte_len(&self) -> usize {
        match self {
            Constant::Bool(_) => 1,
            Constant::Int(_) | Constant::Float(_) | Constant::Dec(_) => 8,
            Constant::Str(_) => 4,
            Constant::Array(a) => 4 + a.iter().map(|v| 1 + v.byte_len()).sum::<usize>(),
            Constant::Null => 0,
        }
    }

    pub(crate) fn encode(&self, e: &mut crate::Encoder) {
        match self {
            Constant::Bool(b) => e.u8(*b as u8),
            Constant::Int(i) => e.i64(*i),
            Constant::Float(f) => e.f64(*f),
            Constant::Dec(b) => e.i64(*b),
            Constant::Str(str_id) => e.u32(*str_id),
            Constant::Array(a) => {
                e.u32(a.len() as u32);
                for v in a {
                    e.u8(constant_discriminant(v));
                    v.encode(e);
                }
            }
            Constant::Null => {}
        }
    }

    pub(crate) fn from_dec(ir: &mut Ir, dec: DecId) -> Self {
        let solve::ResolvedDeclKind::Constant(lit) = &ir.resolutions.decs[dec].kind else {
            unreachable!("Constant::from_dec called on non-constant dec {dec:?}");
        };
        let lit = lit.clone();
        Self::from_literal(ir, lit)
    }

    pub(crate) fn from_literal(ir: &mut Ir, lit: Literal) -> Self {
        match lit {
            Literal::True => Self::Bool(true),
            Literal::False => Self::Bool(false),
            Literal::Null | Literal::Unit => Self::Null,
            Literal::Int(i) => Self::Int(i),
            Literal::Float(f) => Self::Float(f),
            Literal::Hex(h) => Self::Int(i64::from_str_radix(&h, 16).unwrap()),
            Literal::String(s) => Self::Str(ir.intern_str(&s)),
            // arrays and tuples share the same runtime shape -- both lower to a `NewArray` + pushes
            // at use sites today (see `Literal::emit`), so the const form mirrors that.
            Literal::Array(a) | Literal::Tuple(a) => {
                Self::Array(a.iter().map(|e| Self::from_expr(ir, e)).collect())
            }
            // dicts and structs allocate distinct runtime types; the `const` site can't inline
            // them as a bytecode Constant. callers should use `Self::emit_const_dec` (which
            // routes through `Literal::emit`) instead of trying to fold to a `Constant`.
            Literal::Struct(_) | Literal::Record(_) => unreachable!(
                "Constant::from_literal: dict/struct have no scalar constant form -- use emit_const_dec",
            ),
        }
    }

    /// Emit a use-site reference to a const dec. Scalar/array/tuple literals lower to a
    /// `LoadConst` inst; dict/struct literals lower to a fresh runtime allocation at the
    /// use site (same `NewDict`/`NewInstance` path as inline literals would take). `id` is
    /// the use-site's node id, needed by `Literal::Struct` to resolve the ADT type.
    pub(crate) fn emit_const_dec(
        ir: &mut Ir,
        dec: DecId,
        id: parse::NodeId,
    ) -> Option<crate::InstId> {
        let solve::ResolvedDeclKind::Constant(lit) = &ir.resolutions.decs[dec].kind else {
            unreachable!("emit_const_dec called on non-constant dec {dec:?}");
        };
        // r[impl mimas.const.shared]
        // a stored const: the one value, built and frozen when the program started
        if let Some(&local) = ir.const_slots.get(&dec) {
            return Some(if ir.current_body == super::BodyId::ZERO {
                ir.current().get_local(local)
            } else {
                ir.current().get_entry(local)
            });
        }
        let lit = lit.clone();
        // a unit variant (`const ME = Team::Blue`) is its path, lowered the
        // ordinary way — the struct-literal path would build the enum's type,
        // not the variant's
        if let Literal::Struct(s) = &lit
            && s.fields.is_empty()
            && matches!(
                ir.resolutions
                    .node_decs
                    .get(&s.name.id())
                    .map(|&d| &ir.resolutions.decs[d].kind),
                Some(solve::ResolvedDeclKind::Variant { .. })
            )
        {
            use super::Lower;
            return s.name.lower(ir);
        }
        // a tuple that is a struct (`const O: V = (0, 0)`) allocates the struct
        let struct_tuple = matches!(lit, Literal::Tuple(_))
            && matches!(
                ir.resolutions.node_tys.get(&id),
                Some(solve::components::Ty::Adt(..) | solve::components::Ty::Identity(..))
            );
        // A const that is not stored (declared in a fn or an impl, or destructured)
        // is built here. Its value is frozen all the same: a constant cannot be
        // changed wherever it is written.
        if struct_tuple || Self::needs_alloc(ir, &lit) {
            use super::Emit;
            let value = lit.emit(id, ir)?;
            Some(Self::frozen(ir, value))
        } else {
            let aggregate = matches!(lit, Literal::Array(_) | Literal::Tuple(_));
            let constant = Self::from_literal(ir, lit);
            let value = ir.current().constant(constant);
            Some(if aggregate {
                Self::frozen(ir, value)
            } else {
                value
            })
        }
    }

    /// `value` made read-only, when std has the `__freeze` native.
    // r[impl mimas.const.frozen]
    fn frozen(ir: &mut Ir, value: crate::InstId) -> crate::InstId {
        match ir.resolutions.freeze_native {
            Some(native) => ir.current().call_native(native, vec![value]),
            None => value,
        }
    }

    /// A const whose value holds a dict or struct -- itself, an element, or
    /// through a const ref (`const DIRS4 = [RIGHT DOWN LEFT UP]` over struct
    /// consts) -- has no scalar `Constant` form: each use site allocates it
    /// fresh, the way a struct const already does.
    pub(crate) fn needs_alloc(ir: &Ir, lit: &Literal) -> bool {
        match lit {
            Literal::Struct(_) | Literal::Record(_) => true,
            Literal::Array(es) | Literal::Tuple(es) => {
                es.iter().any(|e| Self::expr_needs_alloc(ir, e))
            }
            _ => false,
        }
    }

    /// The literal an expression is at compile time, when it is a const or a field
    /// of one: `WELL`, `WELL.origin`, `(WELL)`. `None` for anything that has to run.
    /// `inside` says the expression is part of a const's own literal. There a
    /// written-out literal counts too. Outside it does not: `S { a = f(), b = 2 }.b`
    /// in ordinary code must still call `f`.
    fn view(ir: &Ir, expr: &Expr, inside: bool) -> Option<Literal> {
        match expr.kind() {
            ExprKind::Literal(lit) if inside => Some(lit.clone()),
            ExprKind::Grouping(g) => Self::view(ir, &g.inner, inside),
            ExprKind::Ident(_) => match ir
                .resolutions
                .node_decs
                .get(&expr.id())
                .map(|&d| &ir.resolutions.decs[d].kind)
            {
                Some(solve::ResolvedDeclKind::Constant(lit)) => Some(lit.clone()),
                _ => None,
            },
            ExprKind::Access(parse::Access::Dot {
                left,
                right,
                kind: parse::AccessKind::Direct,
            }) => {
                let ExprKind::Ident(field) = right.kind() else {
                    return None;
                };
                let part = Self::part_of(ir, left, &field.lexeme, inside)?;
                Self::view(ir, &part, true)
            }
            _ => None,
        }
    }

    /// The initializer of one field of a const struct: for `WELL.rows`, the `20` in
    /// `const WELL = Lattice { rows = 20, .. }`. A read of a const's field lowers
    /// this one expression and never builds the struct around it. `None` when
    /// `left` is not a const struct literal with that field written out (a `..base`
    /// or a field left to its default has no initializer to take).
    // r[impl mimas.const.read]
    pub(crate) fn field_of(ir: &Ir, left: &Expr, field: &str) -> Option<Expr> {
        Self::part_of(ir, left, field, false)
    }

    fn part_of(ir: &Ir, left: &Expr, field: &str, inside: bool) -> Option<Expr> {
        let Literal::Struct(s) = Self::view(ir, left, inside)? else {
            return None;
        };
        if s.base.is_some() {
            return None;
        }
        s.fields.iter().find_map(|(key, value)| match key {
            parse::FieldKey::Ident(name) if name.lexeme == field => Some(value.clone()),
            _ => None,
        })
    }

    /// One field of every element of a const list of structs, as a list constant:
    /// for `MENU[i].row`, the `row` of each button. The read is then one stored
    /// list and an index, and the buttons are never built. `None` unless every
    /// element is a struct literal with that field written out as a plain value
    /// (a number, a string, a bool).
    // r[impl mimas.const.read]
    pub(crate) fn column(ir: &mut Ir, list: &Expr, field: &str) -> Option<Self> {
        let Literal::Array(elems) = Self::view(ir, list, false)? else {
            return None;
        };
        let mut parts = Vec::with_capacity(elems.len());
        for elem in &elems {
            let part = Self::part_of(ir, elem, field, true)?;
            if !Self::is_plain(ir, &part) {
                return None;
            }
            parts.push(part);
        }
        Some(Self::Array(
            parts.iter().map(|e| Self::from_expr(ir, e)).collect(),
        ))
    }

    /// Whether an expression has a scalar `Constant` form `from_expr` can make.
    pub(crate) fn is_plain(ir: &Ir, expr: &Expr) -> bool {
        match expr.kind() {
            ExprKind::Literal(
                Literal::True
                | Literal::False
                | Literal::Int(_)
                | Literal::Float(_)
                | Literal::Hex(_)
                | Literal::String(_),
            ) => true,
            ExprKind::Grouping(g) => Self::is_plain(ir, &g.inner),
            ExprKind::Ident(_) => match ir
                .resolutions
                .node_decs
                .get(&expr.id())
                .map(|&d| &ir.resolutions.decs[d].kind)
            {
                Some(solve::ResolvedDeclKind::Constant(lit)) => matches!(
                    lit,
                    Literal::True
                        | Literal::False
                        | Literal::Int(_)
                        | Literal::Float(_)
                        | Literal::Hex(_)
                        | Literal::String(_)
                ),
                _ => false,
            },
            _ => false,
        }
    }

    fn expr_needs_alloc(ir: &Ir, expr: &Expr) -> bool {
        match expr.kind() {
            ExprKind::Literal(lit) => Self::needs_alloc(ir, lit),
            ExprKind::Grouping(g) => Self::expr_needs_alloc(ir, &g.inner),
            ExprKind::Ident(_) => match &ir.resolutions.decs[ir.node_dec(expr.id())].kind {
                solve::ResolvedDeclKind::Constant(lit) => Self::needs_alloc(ir, lit),
                _ => false,
            },
            // a unit variant (`Team::Blue`) is an instance, built at each use
            ExprKind::Access(parse::Access::DoubleColon { .. }) => matches!(
                ir.resolutions
                    .node_decs
                    .get(&expr.id())
                    .map(|&d| &ir.resolutions.decs[d].kind),
                Some(solve::ResolvedDeclKind::Variant { .. })
            ),
            // r[impl mimas.const.ctor]
            // a `#[ctor]` call (`Vec2::new(..)`) is the only call solve lets into
            // a const; it builds a struct, so each use site emits it
            ExprKind::Call(_) => true,
            _ => false,
        }
    }

    // inside a const array, each element is: ident-to-const, nested array, or a reducible expr.
    // safe to unwrap both layers: solve has already verified every element folds (`is_const_expr`)
    // and would have surfaced any divide-by-zero / overflow as a diagnostic before IR runs.
    fn from_expr(ir: &mut Ir, expr: &Expr) -> Self {
        match expr.kind() {
            ExprKind::Ident(_) => Self::from_dec(ir, ir.node_dec(expr.id())),
            ExprKind::Literal(Literal::Array(inner)) => {
                Self::Array(inner.iter().map(|e| Self::from_expr(ir, e)).collect())
            }
            // a whole-number literal in a `float` slot (`mimas.num.literal`)
            _ if ir.resolutions.int_as_float.contains(&expr.id()) => match reduce_simple(expr) {
                Ok(Some(parse::Literal::Int(n))) => Self::Float(n as f64),
                _ => Self::from_literal(ir, reduce_simple(expr).ok().flatten().unwrap()),
            },
            _ => Self::from_literal(
                ir,
                // solve already vetted every element; reduce_simple is source-free and
                // either succeeds or the expr is malformed (which would've failed type-check).
                // both unwraps lean on that.
                reduce_simple(expr)
                    .expect("solve verified every const array element folds")
                    .expect("solve verified every const array element folds to Some"),
            ),
        }
    }

    /// Returns the [CachedConstant] version of this const, if available. These are Hash friendly.
    pub(crate) fn cached(&self) -> Option<CachedConstant> {
        match self {
            Constant::Bool(b) => Some(CachedConstant::Bool(*b)),
            Constant::Int(i) => Some(CachedConstant::Int(*i)),
            Constant::Str(str_id) => Some(CachedConstant::Str(*str_id)),
            Constant::Null => Some(CachedConstant::Null),
            Constant::Float(f) => Some(CachedConstant::Float(f.to_bits())),
            Constant::Dec(b) => Some(CachedConstant::Dec(*b)),
            Constant::Array(_) => None,
        }
    }
}

#[mutants::skip]
impl IrDisplay for Constant {
    fn ir_display(&self, ir: &Ir) -> String {
        match self {
            Constant::Bool(b) => format!("{b:?}"),
            Constant::Int(i) => i.to_string(),
            Constant::Float(float) => float.to_string(),
            Constant::Dec(b) => format!("{}dec", shared::dec::Dec::from_bits(*b)),
            Constant::Str(str_id) => format!("\"{}\"", ir.str(*str_id)),
            Constant::Array(_) => self.to_string(),
            Constant::Null => "null".into(),
        }
    }
}

#[mutants::skip]
impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Bool(b) => f.pad(&format!("{b}")),
            Constant::Int(i) => f.pad(&format!("{i}")),
            Constant::Float(v) => f.pad(&format!("{v}")),
            Constant::Dec(b) => f.pad(&format!("{}dec", shared::dec::Dec::from_bits(*b))),
            Constant::Str(str_id) => f.pad(&format!("str {}", str_id.index())),
            Constant::Array(a) => {
                f.pad(&format!("[{}]", a.iter().map(|v| v.to_string()).join(", ")))
            }
            Constant::Null => f.pad("null"),
        }
    }
}

// todo: could macro the numericals
impl From<Constant> for Inst {
    fn from(value: Constant) -> Self {
        Self::Constant(value)
    }
}

impl From<usize> for Constant {
    fn from(value: usize) -> Self {
        Self::Int(value as i64)
    }
}

impl From<i64> for Constant {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<i32> for Constant {
    fn from(value: i32) -> Self {
        Self::Int(value as i64)
    }
}

impl From<f64> for Constant {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for Constant {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
