//! Parallel emitter to Zig source, compiled to wasm via
//! `zig build-lib -target wasm32-freestanding`. Shares `analyze`/`scopes` with
//! the wasm-encoder lane; Zig labeled blocks stand in for wasm `block`, and
//! labeled `while (true)` loops for wasm `loop` (`br`→`break :l`/`continue :l`).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

use compile::{BinOp, Op, Program};

use crate::{Bail, Body, K, Scope, ScopeKind, Sig, Skip, analyze, scopes};

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

pub struct Ziggen {
    pub source: String,
    pub bodies: Vec<Body>,
    pub skipped: Vec<Skip>,
}

fn zt(k: K) -> &'static str {
    match k {
        K::Int | K::Word => "i64",
        K::Float => "f64",
        K::Bool => "bool",
    }
}

fn zero(k: K) -> &'static str {
    match k {
        K::Int | K::Word => "0",
        K::Float => "0.0",
        K::Bool => "false",
    }
}

fn coerce(expr: &str, from: K, to: K) -> Result<String, Bail> {
    Ok(match (from, to) {
        (a, b) if a == b => expr.to_string(),
        (K::Int, K::Float) => format!("@floatFromInt({expr})"),
        (K::Float, K::Int) => format!("@intFromFloat(@trunc({expr}))"),
        (K::Int, K::Bool) => format!("({expr} != 0)"),
        (K::Float, K::Bool) => format!("({expr} != 0.0)"),
        (K::Bool, K::Int) => format!("@intFromBool({expr})"),
        (K::Bool, K::Float) => format!("@floatFromInt(@intFromBool({expr}))"),
        _ => bail!("coerce"),
    })
}

struct ZEm<'a> {
    out: String,
    ops: &'a [(usize, Op)],
    class: &'a HashMap<u32, K>,
    sigs: &'a [Option<Sig>],
    callee: &'a HashMap<usize, usize>,
    ret_k: Option<K>,
    stack: Vec<Scope>,
    ind: usize,
}

impl ZEm<'_> {
    fn ln(&mut self, s: &str) {
        for _ in 0..self.ind {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn k(&self, r: compile::Reg) -> Result<K, Bail> {
        self.class
            .get(&(r.index() as u32))
            .copied()
            .ok_or_else(|| format!("reg {} has no scalar class", r.index()))
    }

    fn rg(&self, r: compile::Reg, want: K) -> Result<String, Bail> {
        coerce(&format!("r{}", r.index()), self.k(r)?, want)
    }

    /// wasm-depth-style: find the scope in the stack whose landing is `t` —
    /// by construction each open scope has a unique target index.
    fn label(&self, t: usize) -> Result<(usize, ScopeKind), Bail> {
        for s in self.stack.iter().rev() {
            if s.target == t {
                return Ok((t, s.kind));
            }
        }
        bail!("no open scope for target op {t}")
    }

    fn br(&mut self, t: usize) -> Result<String, Bail> {
        let (t, kind) = self.label(t)?;
        Ok(match kind {
            ScopeKind::Loop => format!("continue :s{t}"),
            ScopeKind::Block => format!("break :s{t}"),
        })
    }

    fn br_if(&mut self, cond: &str, t: usize, is_true: bool) -> Result<(), Bail> {
        let c = if is_true {
            cond.to_string()
        } else {
            format!("!({cond})")
        };
        let b = self.br(t)?;
        self.ln(&format!("if ({c}) {b};"));
        Ok(())
    }

    fn tgt(
        &self,
        t: &compile::BlockTarget,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<usize, Bail> {
        let compile::BlockTarget::ByteOffset(to) = t else {
            bail!("unresolved block target");
        };
        off2idx
            .get(to)
            .copied()
            .ok_or_else(|| format!("jump target {to} mid-op"))
    }

    fn icmp(&mut self, l: compile::Reg, r: compile::Reg, op: &str) -> Result<String, Bail> {
        Ok(format!(
            "{} {op} {}",
            self.rg(l, K::Int)?,
            self.rg(r, K::Int)?
        ))
    }

    fn emit_op(&mut self, i: usize, off2idx: &HashMap<usize, usize>) -> Result<bool, Bail> {
        let (_, op) = &self.ops[i];
        // dead scalar writes (unclassed dst) are skipped, as in the wasm lane
        if let Some(d) = crate::op_dst(op) {
            if !self.class.contains_key(&(d.index() as u32)) {
                return Ok(false);
            }
        }
        match op {
            Op::Move { dst, src } => {
                let k = self.k(*dst)?;
                self.ln(&format!("r{} = {};", dst.index(), self.rg(*src, k)?));
            }
            Op::LoadConst { dst, constant } => {
                let s = match (self.k(*dst)?, constant) {
                    (K::Int, compile::Constant::Int(v)) => v.to_string(),
                    (K::Float, compile::Constant::Float(v)) => {
                        format!("@bitCast(@as(u64, {}))", v.to_bits())
                    }
                    (K::Bool, compile::Constant::Bool(v)) => v.to_string(),
                    (k, c) => bail!("LoadConst {c:?} into {k:?}"),
                };
                self.ln(&format!("r{} = {s};", dst.index()));
            }
            Op::AddInt { dst, left, right }
            | Op::SubInt { dst, left, right }
            | Op::MultInt { dst, left, right } => {
                let builtin = match op {
                    Op::AddInt { .. } => "add",
                    Op::SubInt { .. } => "sub",
                    _ => "mul",
                };
                let l = self.rg(*left, K::Int)?;
                let r = self.rg(*right, K::Int)?;
                self.ln(&format!(
                    "{{ const ov = @{builtin}WithOverflow({l}, {r}); if (ov[1] != 0) unreachable; r{} = ov[0]; }}",
                    dst.index()
                ));
            }
            Op::ModInt { dst, left, right } => {
                // @rem traps on zero divisor, matching RtErr's halt
                self.ln(&format!(
                    "r{} = @rem({}, {});",
                    dst.index(),
                    self.rg(*left, K::Int)?,
                    self.rg(*right, K::Int)?
                ));
            }
            Op::AddIntImm { dst, left, val }
            | Op::SubIntImm { dst, left, val }
            | Op::MultIntImm { dst, left, val } => {
                let builtin = match op {
                    Op::AddIntImm { .. } => "add",
                    Op::SubIntImm { .. } => "sub",
                    _ => "mul",
                };
                let l = self.rg(*left, K::Int)?;
                self.ln(&format!(
                    "{{ const ov = @{builtin}WithOverflow({l}, {val}); if (ov[1] != 0) unreachable; r{} = ov[0]; }}",
                    dst.index()
                ));
            }
            Op::ModIntImm { dst, left, val } => {
                let l = self.rg(*left, K::Int)?;
                if *val == 0 {
                    self.ln("unreachable;");
                } else {
                    self.ln(&format!("r{} = @rem({l}, {val});", dst.index()));
                }
            }
            Op::IntLt { dst, left, right }
            | Op::IntLe { dst, left, right }
            | Op::IntGt { dst, left, right }
            | Op::IntGe { dst, left, right }
            | Op::IntEq { dst, left, right }
            | Op::IntNe { dst, left, right } => {
                let zs = match op {
                    Op::IntLt { .. } => "<",
                    Op::IntLe { .. } => "<=",
                    Op::IntGt { .. } => ">",
                    Op::IntGe { .. } => ">=",
                    Op::IntEq { .. } => "==",
                    _ => "!=",
                };
                let c = self.icmp(*left, *right, zs)?;
                self.ln(&format!("r{} = {};", dst.index(), c));
            }
            Op::IntLtImm { dst, left, val }
            | Op::IntLeImm { dst, left, val }
            | Op::IntGtImm { dst, left, val }
            | Op::IntGeImm { dst, left, val }
            | Op::IntEqImm { dst, left, val }
            | Op::IntNeImm { dst, left, val } => {
                let zs = match op {
                    Op::IntLtImm { .. } => "<",
                    Op::IntLeImm { .. } => "<=",
                    Op::IntGtImm { .. } => ">",
                    Op::IntGeImm { .. } => ">=",
                    Op::IntEqImm { .. } => "==",
                    _ => "!=",
                };
                self.ln(&format!(
                    "r{} = {} {zs} {};",
                    dst.index(),
                    self.rg(*left, K::Int)?,
                    val
                ));
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                let zs = match op {
                    Op::AddFloat { .. } => "+",
                    Op::SubFloat { .. } => "-",
                    Op::MultFloat { .. } => "*",
                    _ => "/",
                };
                self.ln(&format!(
                    "r{} = {} {zs} {};",
                    dst.index(),
                    self.rg(*left, K::Float)?,
                    self.rg(*right, K::Float)?
                ));
            }
            Op::FloatLt { dst, left, right }
            | Op::FloatLe { dst, left, right }
            | Op::FloatGt { dst, left, right }
            | Op::FloatGe { dst, left, right }
            | Op::FloatEq { dst, left, right }
            | Op::FloatNe { dst, left, right } => {
                let zs = match op {
                    Op::FloatLt { .. } => "<",
                    Op::FloatLe { .. } => "<=",
                    Op::FloatGt { .. } => ">",
                    Op::FloatGe { .. } => ">=",
                    Op::FloatEq { .. } => "==",
                    _ => "!=",
                };
                self.ln(&format!(
                    "r{} = {} {zs} {};",
                    dst.index(),
                    self.rg(*left, K::Float)?,
                    self.rg(*right, K::Float)?
                ));
            }
            Op::AddFloatImm { dst, left, val }
            | Op::SubFloatImm { dst, left, val }
            | Op::MultFloatImm { dst, left, val }
            | Op::ModFloatImm { dst, left, val } => {
                let v = f64::from_bits(*val as u64);
                let l = self.rg(*left, K::Float)?;
                let expr = match op {
                    Op::ModFloatImm { .. } => format!("{l} - @trunc({l} / {v:?}) * {v:?}"),
                    _ => format!(
                        "{l} {} {v:?}",
                        match op {
                            Op::AddFloatImm { .. } => "+",
                            Op::SubFloatImm { .. } => "-",
                            _ => "*",
                        }
                    ),
                };
                self.ln(&format!("r{} = {expr};", dst.index()));
            }
            Op::FloatLtImm { dst, left, val }
            | Op::FloatLeImm { dst, left, val }
            | Op::FloatGtImm { dst, left, val }
            | Op::FloatGeImm { dst, left, val }
            | Op::FloatEqImm { dst, left, val }
            | Op::FloatNeImm { dst, left, val } => {
                let v = f64::from_bits(*val as u64);
                let zs = match op {
                    Op::FloatLtImm { .. } => "<",
                    Op::FloatLeImm { .. } => "<=",
                    Op::FloatGtImm { .. } => ">",
                    Op::FloatGeImm { .. } => ">=",
                    Op::FloatEqImm { .. } => "==",
                    _ => "!=",
                };
                self.ln(&format!(
                    "r{} = {} {zs} {v:?};",
                    dst.index(),
                    self.rg(*left, K::Float)?
                ));
            }
            Op::BoolEq { dst, left, right } | Op::BoolNe { dst, left, right } => {
                let zs = if matches!(op, Op::BoolEq { .. }) {
                    "=="
                } else {
                    "!="
                };
                self.ln(&format!(
                    "r{} = {} {zs} {};",
                    dst.index(),
                    self.rg(*left, K::Bool)?,
                    self.rg(*right, K::Bool)?
                ));
            }
            Op::ToFloat { dst, src } => {
                let k = self.k(*dst)?;
                self.ln(&format!("r{} = {};", dst.index(), self.rg(*src, k)?));
            }
            Op::Sqrt { dst, src } => {
                self.ln(&format!(
                    "r{} = @sqrt({});",
                    dst.index(),
                    self.rg(*src, K::Float)?
                ));
            }
            Op::Unary { dst, op: uop, src } => {
                let sk = self.k(*src)?;
                let dk = self.k(*dst)?;
                let s = self.rg(*src, sk)?;
                match uop {
                    compile::UnaryOp::Negative if sk == K::Int => {
                        self.ln(&format!(
                            "{{ const ov = @subWithOverflow(@as(i64, 0), {s}); if (ov[1] != 0) unreachable; r{} = ov[0]; }}",
                            dst.index()
                        ));
                    }
                    _ => {
                        let expr = match uop {
                            compile::UnaryOp::Negative => format!("-({s})"),
                            compile::UnaryOp::Not => format!("!({s})"),
                            compile::UnaryOp::BitwiseNot => format!("~({s})"),
                            compile::UnaryOp::Positive => s,
                        };
                        self.ln(&format!("r{} = {};", dst.index(), coerce(&expr, sk, dk)?));
                    }
                }
            }
            Op::Bin {
                dst,
                left,
                op: bop,
                right,
            } => match (self.k(*left)?, self.k(*right)?) {
                (K::Int, K::Int) => match bop {
                    BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod | BinOp::IDiv => {
                        let l = self.rg(*left, K::Int)?;
                        let r = self.rg(*right, K::Int)?;
                        match bop {
                            BinOp::Mod => self.ln(&format!("r{} = @rem({l}, {r});", dst.index())),
                            BinOp::IDiv => {
                                self.ln(&format!("r{} = @divTrunc({l}, {r});", dst.index()))
                            }
                            _ => {
                                let b = match bop {
                                    BinOp::Add => "add",
                                    BinOp::Sub => "sub",
                                    _ => "mul",
                                };
                                self.ln(&format!(
                                        "{{ const ov = @{b}WithOverflow({l}, {r}); if (ov[1] != 0) unreachable; r{} = ov[0]; }}",
                                        dst.index()
                                    ));
                            }
                        }
                    }
                    _ => bail!("Bin {bop:?} int,int"),
                },
                (K::Float, K::Float) => match bop {
                    BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Div => {
                        let zs = match bop {
                            BinOp::Add => "+",
                            BinOp::Sub => "-",
                            BinOp::Mult => "*",
                            _ => "/",
                        };
                        self.ln(&format!(
                            "r{} = {} {zs} {};",
                            dst.index(),
                            self.rg(*left, K::Float)?,
                            self.rg(*right, K::Float)?
                        ));
                    }
                    BinOp::LessThan
                    | BinOp::LessEqual
                    | BinOp::GreaterThan
                    | BinOp::GreaterEqual
                    | BinOp::Identity
                    | BinOp::NotEqual => {
                        let zs = match bop {
                            BinOp::LessThan => "<",
                            BinOp::LessEqual => "<=",
                            BinOp::GreaterThan => ">",
                            BinOp::GreaterEqual => ">=",
                            BinOp::Identity => "==",
                            _ => "!=",
                        };
                        self.ln(&format!(
                            "r{} = {} {zs} {};",
                            dst.index(),
                            self.rg(*left, K::Float)?,
                            self.rg(*right, K::Float)?
                        ));
                    }
                    _ => bail!("Bin {bop:?} float,float"),
                },
                (K::Bool, K::Bool) => match bop {
                    BinOp::Identity | BinOp::NotEqual | BinOp::Xor | BinOp::And | BinOp::Or => {
                        let zs = match bop {
                            BinOp::Identity => "==",
                            BinOp::Xor | BinOp::NotEqual => "!=",
                            BinOp::And => "and",
                            _ => "or",
                        };
                        self.ln(&format!(
                            "r{} = {} {zs} {};",
                            dst.index(),
                            self.rg(*left, K::Bool)?,
                            self.rg(*right, K::Bool)?
                        ));
                    }
                    _ => bail!("Bin {bop:?} bool,bool"),
                },
                (a, b) => bail!("Bin {bop:?} {a:?},{b:?}"),
            },
            Op::Jump { target } => {
                let t = self.tgt(target, off2idx)?;
                let b = self.br(t)?;
                self.ln(&format!("{b};"));
            }
            Op::JumpIf {
                cond,
                target,
                is_true,
            } => {
                let t = self.tgt(target, off2idx)?;
                self.br_if(&self.rg(*cond, K::Bool)?, t, *is_true)?;
            }
            Op::ForNext { idx, bound, target } => {
                let t = self.tgt(target, off2idx)?;
                self.ln(&format!("r{} += 1;", idx.index()));
                self.br_if(
                    &format!("{} < {}", self.rg(*idx, K::Int)?, self.rg(*bound, K::Int)?),
                    t,
                    true,
                )?;
            }
            Op::CallDirect { dst, body, args } => {
                self.emit_call(*dst, body.index(), args)?;
            }
            Op::Call { dst, args, .. } => {
                let b = *self
                    .callee
                    .get(&i)
                    .ok_or("dynamic Call to non-const callee")?;
                self.emit_call(*dst, b, args)?;
            }
            Op::CallNative { dst, id, args } => {
                let mut params = Vec::new();
                let mut argv = Vec::new();
                for a in args {
                    let k = self.k(*a)?;
                    params.push(k);
                    argv.push(format!("r{}", a.index()));
                }
                let dk = self.class.get(&(dst.index() as u32)).copied();
                let ret = dk.unwrap_or(K::Int);
                let sig_desc: String = params
                    .iter()
                    .map(|k| match k {
                        K::Int | K::Word => 'i',
                        K::Float => 'f',
                        K::Bool => 'b',
                    })
                    .collect();
                let call = format!("n{}_{sig_desc}({})", id.index(), argv.join(", "));
                match dk {
                    Some(k) => self.ln(&format!("r{} = {};", dst.index(), coerce(&call, ret, k)?)),
                    None => self.ln(&format!("_ = {call};")),
                }
            }
            Op::LoadBody { .. } => {}
            Op::Return { val } => {
                if let Some(k) = self.ret_k {
                    self.ln(&format!("return {};", self.rg(*val, k)?));
                } else {
                    self.ln("return;");
                }
            }
            Op::BIntLt {
                target,
                left,
                right,
                is_true,
            }
            | Op::BIntLe {
                target,
                left,
                right,
                is_true,
            }
            | Op::BIntGt {
                target,
                left,
                right,
                is_true,
            }
            | Op::BIntGe {
                target,
                left,
                right,
                is_true,
            }
            | Op::BIntEq {
                target,
                left,
                right,
                is_true,
            }
            | Op::BIntNe {
                target,
                left,
                right,
                is_true,
            } => {
                let zs = match op {
                    Op::BIntLt { .. } => "<",
                    Op::BIntLe { .. } => "<=",
                    Op::BIntGt { .. } => ">",
                    Op::BIntGe { .. } => ">=",
                    Op::BIntEq { .. } => "==",
                    _ => "!=",
                };
                let t = self.tgt(target, off2idx)?;
                let c = self.icmp(*left, *right, zs)?;
                self.br_if(&c, t, *is_true)?;
            }
            Op::BIntLtImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BIntLeImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BIntGtImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BIntGeImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BIntEqImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BIntNeImm {
                target,
                left,
                val,
                is_true,
            } => {
                let zs = match op {
                    Op::BIntLtImm { .. } => "<",
                    Op::BIntLeImm { .. } => "<=",
                    Op::BIntGtImm { .. } => ">",
                    Op::BIntGeImm { .. } => ">=",
                    Op::BIntEqImm { .. } => "==",
                    _ => "!=",
                };
                let t = self.tgt(target, off2idx)?;
                self.br_if(
                    &format!("{} {zs} {val}", self.rg(*left, K::Int)?),
                    t,
                    *is_true,
                )?;
            }
            Op::BFloatLt {
                target,
                left,
                right,
                is_true,
            }
            | Op::BFloatLe {
                target,
                left,
                right,
                is_true,
            }
            | Op::BFloatGt {
                target,
                left,
                right,
                is_true,
            }
            | Op::BFloatGe {
                target,
                left,
                right,
                is_true,
            }
            | Op::BFloatEq {
                target,
                left,
                right,
                is_true,
            }
            | Op::BFloatNe {
                target,
                left,
                right,
                is_true,
            } => {
                let zs = match op {
                    Op::BFloatLt { .. } => "<",
                    Op::BFloatLe { .. } => "<=",
                    Op::BFloatGt { .. } => ">",
                    Op::BFloatGe { .. } => ">=",
                    Op::BFloatEq { .. } => "==",
                    _ => "!=",
                };
                let t = self.tgt(target, off2idx)?;
                self.br_if(
                    &format!(
                        "{} {zs} {}",
                        self.rg(*left, K::Float)?,
                        self.rg(*right, K::Float)?
                    ),
                    t,
                    *is_true,
                )?;
            }
            Op::BFloatLtImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BFloatLeImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BFloatGtImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BFloatGeImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BFloatEqImm {
                target,
                left,
                val,
                is_true,
            }
            | Op::BFloatNeImm {
                target,
                left,
                val,
                is_true,
            } => {
                let v = f64::from_bits(*val as u64);
                let zs = match op {
                    Op::BFloatLtImm { .. } => "<",
                    Op::BFloatLeImm { .. } => "<=",
                    Op::BFloatGtImm { .. } => ">",
                    Op::BFloatGeImm { .. } => ">=",
                    Op::BFloatEqImm { .. } => "==",
                    _ => "!=",
                };
                let t = self.tgt(target, off2idx)?;
                self.br_if(
                    &format!("{} {zs} {v:?}", self.rg(*left, K::Float)?),
                    t,
                    *is_true,
                )?;
            }
            other => bail!("unsupported op {other:?}"),
        }
        Ok(matches!(
            op,
            Op::Jump { .. } | Op::Return { .. } | Op::ModIntImm { val: 0, .. }
        ))
    }

    fn emit_call(
        &mut self,
        dst: compile::Reg,
        b: usize,
        args: &[compile::Reg],
    ) -> Result<(), Bail> {
        let sig = self.sigs[b]
            .as_ref()
            .ok_or_else(|| format!("callee body {b} not emitted"))?;
        let dk = self.class.get(&(dst.index() as u32)).copied();
        if let Some(k) = dk {
            let ret = sig.ret.ok_or("callee returns void but dst is read")?;
            if k != ret {
                bail!("call dst class != callee ret");
            }
        }
        if sig.params.len() != args.len() {
            bail!("arity mismatch calling body {b}");
        }
        let mut argv = Vec::new();
        for (a, pk) in args.iter().zip(&sig.params) {
            argv.push(self.rg(*a, *pk)?);
        }
        let call = format!("b{b}({})", argv.join(", "));
        match dk {
            Some(k) => self.ln(&format!(
                "r{} = {};",
                dst.index(),
                coerce(&call, sig.ret.unwrap(), k)?
            )),
            None => self.ln(&format!("_ = {call};")),
        }
        Ok(())
    }
}

fn try_zig_body(
    b: usize,
    ops: &[(usize, Op)],
    program: &Program,
    ana: &crate::Ana,
    sigs: &[Option<Sig>],
) -> Result<String, Bail> {
    let chunk = &program.chunks[compile::BodyId::from(b as u32)];
    let Some(sig) = &sigs[b] else { bail!("no sig") };
    let sc = scopes(ops, ops.len())?;
    let mut opens: BTreeMap<usize, Vec<Scope>> = BTreeMap::new();
    for s in sc {
        opens.entry(s.open).or_default().push(s);
    }
    let mut em = ZEm {
        out: String::new(),
        ops,
        class: &ana.class,
        sigs,
        callee: &ana.callee,
        ret_k: sig.ret,
        stack: Vec::new(),
        ind: 1,
    };
    let params = chunk
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| format!("r{}: {}", p.index(), zt(sig.params[i])))
        .collect::<Vec<_>>()
        .join(", ");
    let ret_ty = sig.ret.map(zt).unwrap_or("void");
    writeln!(em.out, "export fn b{b}({params}) {ret_ty} {{").unwrap();
    // declare all classed regs (params already bound)
    let mut decls: Vec<u32> = ana.class.keys().copied().collect();
    decls.sort();
    let params_set: HashMap<u32, ()> = chunk
        .params
        .iter()
        .map(|p| (p.index() as u32, ()))
        .collect();
    for r in decls {
        if params_set.contains_key(&r) {
            continue;
        }
        em.ln(&format!(
            "var r{r}: {} = {};",
            zt(ana.class[&r]),
            zero(ana.class[&r])
        ));
    }
    let mut off2idx = HashMap::new();
    for (i, (off, _)) in ops.iter().enumerate() {
        off2idx.insert(*off, i);
    }
    // Zig hard-errors on unreachable statements. A diverging op (Jump,
    // Return, trap) kills everything until the next scope close, which is
    // always a reachable landing (block end = br target, loop end = fallthrough
    // exit). Scopes opening inside the dead span are skipped entirely.
    let mut dead = false;
    let mut phantom: Vec<usize> = Vec::new(); // closes of skipped scopes
    for i in 0..ops.len() {
        loop {
            if dead && phantom.last() == Some(&i) {
                phantom.pop();
                continue;
            }
            match em.stack.last() {
                Some(s) if s.close <= i => {
                    let s = *s;
                    em.stack.pop();
                    if s.kind == ScopeKind::Loop {
                        // wasm `loop` exits at end; `while (true)` needs it
                        em.ln("break;");
                    }
                    em.ind -= 1;
                    em.ln("}");
                    dead = false;
                }
                _ => break,
            }
        }
        if let Some(mut group) = opens.remove(&i) {
            group.sort_by_key(|s| (std::cmp::Reverse(s.close), s.kind));
            for s in group {
                if dead {
                    phantom.push(s.close);
                    continue;
                }
                match s.kind {
                    ScopeKind::Loop => em.ln(&format!("s{}: while (true) {{", s.target)),
                    ScopeKind::Block => em.ln(&format!("s{}: {{", s.target)),
                }
                em.ind += 1;
                em.stack.push(s);
            }
        }
        if dead {
            continue;
        }
        if em.emit_op(i, &off2idx)? {
            dead = true;
        }
    }
    loop {
        if phantom.pop().is_some() {
            continue;
        }
        match em.stack.pop() {
            Some(s) => {
                if s.kind == ScopeKind::Loop {
                    em.ln("break;");
                }
                em.ind -= 1;
                em.ln("}");
            }
            None => break,
        }
    }
    if sig.ret.is_none() && !dead {
        em.ln("return;");
    }
    em.ln("}");
    Ok(em.out)
}

/// Emit Zig source for the same scalar subset `emit` covers.
pub fn emit_zig(program: &Program) -> Result<Ziggen, Bail> {
    let nbodies = program.chunks.len();
    let bodies_ops: Vec<Vec<(usize, Op)>> = (0..nbodies)
        .map(|b| program.ops(compile::BodyId::from(b as u32)))
        .collect();

    let anas: Vec<Option<crate::Ana>> = (0..nbodies)
        .map(|b| {
            let chunk = &program.chunks[compile::BodyId::from(b as u32)];
            Some(analyze(&bodies_ops[b], chunk))
        })
        .collect();

    let mut skipped: Vec<Skip> = Vec::new();
    let mut sigs: Vec<Option<Sig>> = (0..nbodies).map(|_| None).collect();
    for b in 0..nbodies {
        let chunk = &program.chunks[compile::BodyId::from(b as u32)];
        let Some(ana) = &anas[b] else {
            skipped.push(Skip {
                body: b,
                reason: "analysis failed".into(),
            });
            continue;
        };
        let mut params = Vec::new();
        let mut ok = true;
        for p in &chunk.params {
            match ana.class.get(&(p.index() as u32)) {
                Some(&k) => params.push(k),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            skipped.push(Skip {
                body: b,
                reason: "param has no scalar class".into(),
            });
            continue;
        }
        sigs[b] = Some(Sig {
            params,
            ret: ana.ret,
        });
    }

    // natives over all sig'd bodies
    let mut natives: HashMap<(u32, Vec<K>, Option<K>), u32> = HashMap::new();
    let mut native_list: Vec<(u32, Vec<K>, Option<K>)> = Vec::new();
    for b in 0..nbodies {
        let Some(ana) = &anas[b] else { continue };
        if sigs[b].is_none() {
            continue;
        }
        for (_, op) in &bodies_ops[b] {
            if let Op::CallNative { dst, id, args } = op {
                let params: Vec<K> = args
                    .iter()
                    .filter_map(|a| ana.class.get(&(a.index() as u32)).copied())
                    .collect();
                let ret = Some(
                    ana.class
                        .get(&(dst.index() as u32))
                        .copied()
                        .unwrap_or(K::Int),
                );
                let key = (id.index() as u32, params, ret);
                if !natives.contains_key(&key) {
                    natives.insert(key.clone(), native_list.len() as u32);
                    native_list.push(key);
                }
            }
        }
    }

    let mut emitted: Vec<bool> = sigs.iter().map(|s| s.is_some()).collect();
    let mut reasons: Vec<String> = (0..nbodies).map(|_| String::new()).collect();
    loop {
        let mut changed = false;
        for b in 0..nbodies {
            if !emitted[b] {
                continue;
            }
            let ana = anas[b].as_ref().unwrap();
            let native_ok = bodies_ops[b].iter().all(|(_, op)| match op {
                Op::CallNative { args, .. } => args
                    .iter()
                    .all(|a| ana.class.contains_key(&(a.index() as u32))),
                _ => true,
            });
            let mut why = String::new();
            if let Some(c) = ana.callee.values().find(|c| !emitted[**c]) {
                why = format!("calls skipped body {c}");
            } else if !native_ok {
                why = "native arg has no scalar class".into();
            } else if let Err(e) = try_zig_body(b, &bodies_ops[b], program, ana, &sigs) {
                why = e;
            }
            if !why.is_empty() {
                emitted[b] = false;
                reasons[b] = why;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for b in 0..nbodies {
        if sigs[b].is_some() && !emitted[b] {
            skipped.push(Skip {
                body: b,
                reason: reasons[b].clone(),
            });
            sigs[b] = None;
        }
    }

    let mut source = String::from("// generated by mimas-wasmgen zig lane\n");
    for (id, params, ret) in native_list.iter() {
        let sig_desc: String = params
            .iter()
            .map(|k| match k {
                K::Int => 'i',
                K::Float => 'f',
                K::Bool => 'b',
                K::Word => 'i',
            })
            .collect();
        let args = (0..params.len())
            .map(|i| format!("a{i}: {}", zt(params[i])))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            source,
            "extern \"env\" fn n{id}_{sig_desc}({args}) {};",
            zt(ret.unwrap())
        )
        .unwrap();
    }
    let mut bodies = Vec::new();
    for b in 0..nbodies {
        if !emitted[b] {
            continue;
        }
        let ana = anas[b].as_ref().unwrap();
        let src = try_zig_body(b, &bodies_ops[b], program, ana, &sigs)
            .map_err(|e| format!("body {b} passed trial but failed emit: {e}"))?;
        source.push_str(&src);
        bodies.push(Body {
            body: b,
            func: b as u32,
            name: format!("b{b}"),
        });
    }
    Ok(Ziggen {
        source,
        bodies,
        skipped,
    })
}
