//! Resumable emission: every emitted body is a pc-dispatch state machine with
//! registers in linear memory instead of wasm locals.
//!
//! Why: handles irreducible CFGs (no `block`/`loop` nesting needed), pauses
//! mid-function (set `__status`, return, re-enter via `pc`), and makes a
//! checkpoint a memcpy — the whole live state is the frame stack + pc_table.
//!
//! Linear memory layout:
//!   [0, nbodies*8)        pc_table — body -> region index to resume at (0=fresh)
//!   [STACK, +STACK_CAP)   frame stack — one frame per live call, reg r at fp+r*8
//!   [COV, +total_ops)     coverage bitmap when `Opts::coverage`
//!
//! Globals imported from `env`: `__fuel` i64 — charged by op count per region,
//! suspends with `__status=2` when negative; `__pause` i32 — checked per region,
//! suspends with `__status=1`. Defined+exported: `__status` i32 (0=done),
//! `__sp` i32 (frame stack top; on suspend left at the outermost live frame's
//! base so resume re-derives every fp).
//!
//! Call convention: the caller sets `__sp = myfp + fsize` before every call —
//! fresh or resume, same value. Callee reads `myfp = __sp`, bumps `__sp` by its
//! own frame size, and only stores its params when `pc_table[b] == 0` (fresh
//! entry). On suspend each level returns without popping, unwinding the wasm
//! stack while memory frames persist; re-calling an export resumes it at
//! `pc_table[b]` and the paused call chain rebuilds itself as callers replay
//! their call sites. On normal return: `__sp = myfp`, `pc_table[b] = 0`,
//! `__status = 0`.
//!
//! A region is a maximal straight-line run ending at a control op; every call
//! is alone in its region so a suspend propagating through it resumes by
//! re-entering exactly at the call — no op that suspends sits mid-region, so no
//! host-visible side effect (natives) is ever replayed on resume.

use std::collections::{BTreeSet, HashMap, HashSet};

use compile::{BinOp, Op, Program, Reg, UnaryOp};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, CustomSection, Encode, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, GlobalSection, GlobalType, ImportSection,
    Instruction, MemArg, MemorySection, MemoryType, Module, NameMap, NameSection, TypeSection,
    ValType,
};

use crate::{analyze, op_dst, Bail, Body, K, Opts, Sig, Skip, Wasmgen};

const STACK_CAP: u32 = 1 << 20;
const PAGE: u64 = 65536;
const STATUS_PAUSED: i32 = 1;
const STATUS_FUEL: i32 = 2;

fn mem(offset: u32, align: u32) -> MemArg {
    MemArg {
        offset: offset as u64,
        align,
        memory_index: 0,
    }
}

struct RE<'a> {
    ops: &'a [(usize, Op)],
    mask: &'a HashMap<u32, u8>,
    reads_w: &'a HashSet<u32>,
    sigs: &'a [Option<Sig>],
    func_map: &'a HashMap<usize, u32>,
    callee: &'a HashMap<usize, usize>,
    natives: &'a HashMap<(u32, Vec<K>, Option<K>), u32>,
    ret_k: Option<K>,
    body: usize,
    /// op index -> region index
    region_of: &'a [u32],
    /// bytes so far (srcmap offsets)
    c: Vec<u8>,
    srcmap: Vec<(u32, u32)>,
    opts: &'a Opts,
    cov_base: u32,
    g_fuel: Option<u32>,
    g_pause: Option<u32>,
    g_status: u32,
    g_sp: u32,
    g_hp: u32,
    l_pc: u32,
    l_myfp: u32,
    l_tmp: u32,
    l_tmpf: u32,
    l_tmpb: u32,
    l_tmpc: u32,
    l_hp: u32,
    l_sz: u32,
    region: u32,
}

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

impl RE<'_> {
    fn ins(&mut self, i: Instruction) {
        i.encode(&mut self.c);
    }

    fn pc_addr(&self) -> u32 {
        self.body as u32 * 8
    }

    /// push frame address of reg r (store/load MemArg carries the r*8 offset)
    fn fp(&mut self) {
        self.ins(Instruction::LocalGet(self.l_myfp));
    }

    /// load reg r's raw 8-byte word (heap handles, Move, field elements —
    /// values whose class the reader's op decides, not the slot)
    fn ld_word(&mut self, r: Reg) {
        self.fp();
        self.ins(Instruction::I64Load(mem(r.index() as u32 * 8, 3)));
    }

    /// load reg r in context `want`. A multi-class reg is read at `want`'s
    /// width — on any path that reaches this op in a valid program the last
    /// write had that class, so the bytes are already in the right encoding.
    /// A single-class reg of a different class gets the coercion.
    fn get(&mut self, r: Reg, want: K) -> Result<(), Bail> {
        let idx = r.index() as u32;
        let off = idx * 8;
        if want == K::Word {
            self.fp();
            self.ins(Instruction::I64Load(mem(off, 3)));
            return Ok(());
        }
        let m = *self
            .mask
            .get(&idx)
            .ok_or_else(|| format!("reg {} has no scalar class", idx))?;
        if m & crate::kbit(want) != 0 {
            self.fp();
            self.ins(match want {
                K::Int | K::Word => Instruction::I64Load(mem(off, 3)),
                K::Float => Instruction::F64Load(mem(off, 3)),
                K::Bool => Instruction::I32Load(mem(off, 2)),
            });
            return Ok(());
        }
        self.fp();
        match (m, want) {
            (crate::K_INT, K::Float) => {
                self.ins(Instruction::I64Load(mem(off, 3)));
                self.ins(Instruction::F64ConvertI64S);
            }
            (crate::K_FLOAT, K::Int) => {
                self.ins(Instruction::F64Load(mem(off, 3)));
                self.ins(Instruction::I64TruncF64S);
            }
            (crate::K_BOOL, K::Int) => {
                self.ins(Instruction::I32Load(mem(off, 2)));
                self.ins(Instruction::I64ExtendI32U);
            }
            (crate::K_BOOL, K::Float) => {
                self.ins(Instruction::I32Load(mem(off, 2)));
                self.ins(Instruction::F64ConvertI32S);
            }
            (crate::K_INT, K::Bool) => {
                self.ins(Instruction::I64Load(mem(off, 3)));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64Ne);
            }
            (crate::K_FLOAT, K::Bool) => {
                self.ins(Instruction::F64Load(mem(off, 3)));
                self.ins(Instruction::F64Const(0.0f64.into()));
                self.ins(Instruction::F64Ne);
            }
            _ => bail!("coerce mask {m:#x}->{want:?}"),
        }
        Ok(())
    }

    /// store the stack-top value of class `k` into reg r's slot
    fn st(&mut self, r: Reg, k: K) {
        let off = r.index() as u32 * 8;
        self.ins(match k {
            K::Int | K::Word => Instruction::I64Store(mem(off, 3)),
            K::Float => Instruction::F64Store(mem(off, 3)),
            K::Bool => Instruction::I32Store(mem(off, 2)),
        });
    }

    /// `frame[dst] = <produced k>`: emit addr first, then producer, then store.
    fn setv(
        &mut self,
        dst: Reg,
        k: K,
        produce: impl FnOnce(&mut Self) -> Result<(), Bail>,
    ) -> Result<(), Bail> {
        self.fp();
        produce(self)?;
        self.st(dst, k);
        Ok(())
    }

    /// a class for regs whose producer's class is implicit (CallNative dst,
    /// Unary/Bin operands on untyped heap words): prefer Float (heap values
    /// in mimas programs are overwhelmingly numeric-float), then Int, then
    /// Bool — a wrong pick only mis-encodes a read a valid program wouldn't
    /// reach anyway
    fn pick(&self, r: Reg) -> Option<K> {
        let m = self.mask.get(&(r.index() as u32))?;
        Some(if m & crate::K_FLOAT != 0 {
            K::Float
        } else if m & crate::K_INT != 0 {
            K::Int
        } else {
            K::Bool
        })
    }

    /// suspend: pc_table[b]=region, __status=code, __sp=myfp, return dummy.
    /// `keep_status` is for the post-call propagation path, where the callee
    /// already wrote the status we must NOT overwrite.
    fn suspend(&mut self, status: i32) {
        self.suspend_keep(status, false);
    }

    fn suspend_keep(&mut self, status: i32, keep_status: bool) {
        let pc = self.region;
        self.ins(Instruction::I32Const(self.pc_addr() as i32));
        self.ins(Instruction::I32Const(pc as i32));
        self.ins(Instruction::I32Store(mem(0, 2)));
        if !keep_status {
            self.ins(Instruction::I32Const(status));
            self.ins(Instruction::GlobalSet(self.g_status));
        }
        self.ins(Instruction::LocalGet(self.l_myfp));
        self.ins(Instruction::GlobalSet(self.g_sp));
        match self.ret_k {
            Some(K::Word) => self.ins(Instruction::I64Const(0)),
            Some(K::Int) => self.ins(Instruction::I64Const(0)),
            Some(K::Float) => self.ins(Instruction::F64Const(0.0f64.into())),
            Some(K::Bool) => self.ins(Instruction::I32Const(0)),
            None => {}
        }
        self.ins(Instruction::Return);
    }

    /// fuel charge + pause check at a region head; `nops` ops in the region
    fn region_head(&mut self, nops: usize) {
        if let Some(g) = self.g_fuel {
            self.ins(Instruction::GlobalGet(g));
            self.ins(Instruction::I64Const(nops as i64));
            self.ins(Instruction::I64Sub);
            self.ins(Instruction::GlobalSet(g));
            self.ins(Instruction::GlobalGet(g));
            self.ins(Instruction::I64Const(0));
            self.ins(Instruction::I64LtS);
            self.ins(Instruction::If(BlockType::Empty));
            self.suspend(STATUS_FUEL);
            self.ins(Instruction::End);
        }
        if let Some(g) = self.g_pause {
            self.ins(Instruction::GlobalGet(g));
            self.ins(Instruction::If(BlockType::Empty));
            self.suspend(STATUS_PAUSED);
            self.ins(Instruction::End);
        }
    }

    /// `pc = r; br $L` — dispatch loop depth from region j is j+1.
    fn goto(&mut self, region: u32) {
        self.ins(Instruction::I32Const(region as i32));
        self.ins(Instruction::LocalSet(self.l_pc));
        self.ins(Instruction::Br(self.region + 1));
    }

    /// `pc = f; if (cond) pc = t; br $L` — cond producer runs first.
    fn goto_if(
        &mut self,
        t: u32,
        f: u32,
        is_true: bool,
        cond: impl FnOnce(&mut Self) -> Result<(), Bail>,
    ) -> Result<(), Bail> {
        cond(self)?;
        self.ins(Instruction::LocalSet(self.l_tmpb));
        self.ins(Instruction::I32Const(f as i32));
        self.ins(Instruction::LocalSet(self.l_pc));
        self.ins(Instruction::LocalGet(self.l_tmpb));
        if !is_true {
            self.ins(Instruction::I32Eqz);
        }
        self.ins(Instruction::If(BlockType::Empty));
        self.ins(Instruction::I32Const(t as i32));
        self.ins(Instruction::LocalSet(self.l_pc));
        self.ins(Instruction::End);
        self.ins(Instruction::Br(self.region + 1));
        Ok(())
    }

    fn checked_int(&mut self, l: Reg, r: Reg, op: BinOp) -> Result<(), Bail> {
        match op {
            BinOp::Mod | BinOp::IDiv => {
                self.get(l, K::Int)?;
                self.get(r, K::Int)?;
                self.ins(if op == BinOp::Mod {
                    Instruction::I64RemS
                } else {
                    Instruction::I64DivS
                });
                return Ok(());
            }
            _ => {}
        }
        self.get(l, K::Int)?;
        self.get(r, K::Int)?;
        self.ins(match op {
            BinOp::Add => Instruction::I64Add,
            BinOp::Sub => Instruction::I64Sub,
            BinOp::Mult => Instruction::I64Mul,
            _ => bail!("checked_int {op:?}"),
        });
        self.ins(Instruction::LocalSet(self.l_tmp));
        match op {
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (Instruction::I64LtS, Instruction::I64GtS)
                } else {
                    (Instruction::I64GtS, Instruction::I64LtS)
                };
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64GtS);
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.get(l, K::Int)?;
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.get(l, K::Int)?;
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
            }
            BinOp::Mult => {
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Eqz);
                self.ins(Instruction::I32Eqz);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.get(r, K::Int)?;
                self.ins(Instruction::I64DivS);
                self.get(l, K::Int)?;
                self.ins(Instruction::I64Ne);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
                self.ins(Instruction::End);
            }
            _ => unreachable!(),
        }
        self.ins(Instruction::LocalGet(self.l_tmp));
        Ok(())
    }

    fn checked_int_imm(&mut self, l: Reg, v: i64, op: BinOp) -> Result<(), Bail> {
        match op {
            BinOp::Mod | BinOp::IDiv => {
                if v == 0 {
                    self.ins(Instruction::Unreachable);
                    return Ok(());
                }
                self.get(l, K::Int)?;
                self.ins(Instruction::I64Const(v));
                self.ins(if op == BinOp::Mod {
                    Instruction::I64RemS
                } else {
                    Instruction::I64DivS
                });
                return Ok(());
            }
            _ => {}
        }
        self.get(l, K::Int)?;
        self.ins(Instruction::I64Const(v));
        self.ins(match op {
            BinOp::Add => Instruction::I64Add,
            BinOp::Sub => Instruction::I64Sub,
            BinOp::Mult => Instruction::I64Mul,
            _ => bail!("checked_int_imm {op:?}"),
        });
        self.ins(Instruction::LocalSet(self.l_tmp));
        match op {
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (Instruction::I64LtS, Instruction::I64GtS)
                } else {
                    (Instruction::I64GtS, Instruction::I64LtS)
                };
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64GtS);
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.get(l, K::Int)?;
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.get(l, K::Int)?;
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
            }
            BinOp::Mult => {
                if v == 0 {
                    self.ins(Instruction::LocalGet(self.l_tmp));
                    return Ok(());
                }
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64DivS);
                self.get(l, K::Int)?;
                self.ins(Instruction::I64Ne);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
            }
            _ => unreachable!(),
        }
        self.ins(Instruction::LocalGet(self.l_tmp));
        Ok(())
    }

    fn emit_bin(&mut self, dst: Reg, l: Reg, op: BinOp, r: Reg) -> Result<(), Bail> {
        // operand class for heap-loaded words: best-known class per operand,
        // falling back to the sibling's, the dst's, then Float
        let kl = self
            .pick(dst)
            .or_else(|| self.pick(l))
            .or_else(|| self.pick(r))
            .unwrap_or(K::Float);
        let kr = kl;
        match (kl, kr) {
            (K::Word, _) | (_, K::Word) => unreachable!("bin operands resolve to scalar"),
            (K::Int, K::Int) => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod | BinOp::IDiv => {
                    self.setv(dst, K::Int, |s| s.checked_int(l, r, op))?;
                }
                _ => {
                    let (i, dk) = match op {
                        BinOp::LessThan => (Instruction::I64LtS, K::Bool),
                        BinOp::LessEqual => (Instruction::I64LeS, K::Bool),
                        BinOp::GreaterThan => (Instruction::I64GtS, K::Bool),
                        BinOp::GreaterEqual => (Instruction::I64GeS, K::Bool),
                        BinOp::Identity => (Instruction::I64Eq, K::Bool),
                        BinOp::NotEqual => (Instruction::I64Ne, K::Bool),
                        BinOp::And => (Instruction::I64And, K::Int),
                        BinOp::Or => (Instruction::I64Or, K::Int),
                        BinOp::Xor => (Instruction::I64Xor, K::Int),
                        _ => bail!("Bin int {op:?}"),
                    };
                    self.setv(dst, dk, |s| {
                        s.get(l, K::Int)?;
                        s.get(r, K::Int)?;
                        s.ins(i);
                        Ok(())
                    })?;
                }
            },
            (K::Float, K::Float) => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Div => {
                    let i = match op {
                        BinOp::Add => Instruction::F64Add,
                        BinOp::Sub => Instruction::F64Sub,
                        BinOp::Mult => Instruction::F64Mul,
                        _ => Instruction::F64Div,
                    };
                    self.setv(dst, K::Float, |s| {
                        s.get(l, K::Float)?;
                        s.get(r, K::Float)?;
                        s.ins(i);
                        Ok(())
                    })?;
                }
                BinOp::LessThan | BinOp::LessEqual | BinOp::GreaterThan | BinOp::GreaterEqual
                | BinOp::Identity | BinOp::NotEqual => {
                    let i = match op {
                        BinOp::LessThan => Instruction::F64Lt,
                        BinOp::LessEqual => Instruction::F64Le,
                        BinOp::GreaterThan => Instruction::F64Gt,
                        BinOp::GreaterEqual => Instruction::F64Ge,
                        BinOp::Identity => Instruction::F64Eq,
                        _ => Instruction::F64Ne,
                    };
                    self.setv(dst, K::Bool, |s| {
                        s.get(l, K::Float)?;
                        s.get(r, K::Float)?;
                        s.ins(i);
                        Ok(())
                    })?;
                }
                _ => bail!("Bin float {op:?}"),
            },
            (K::Bool, K::Bool) => {
                let i = match op {
                    BinOp::Identity => Instruction::I32Eq,
                    BinOp::NotEqual | BinOp::Xor => Instruction::I32Ne,
                    BinOp::And => Instruction::I32And,
                    BinOp::Or => Instruction::I32Or,
                    _ => bail!("Bin bool {op:?}"),
                };
                self.setv(dst, K::Bool, |s| {
                    s.get(l, K::Bool)?;
                    s.get(r, K::Bool)?;
                    s.ins(i);
                    Ok(())
                })?;
            }
            (a, b) => bail!("Bin {op:?} {a:?},{b:?}"),
        }
        Ok(())
    }

    /// bump-allocate `[size]` bytes from the __hp arena; pops size (i32) from
    /// the wasm stack, pushes the block handle (i32). Grows memory on demand.
    fn alloc(&mut self) {
        self.ins(Instruction::LocalSet(self.l_sz));
        // if __hp + size > memory.size * 65536 → grow
        self.ins(Instruction::GlobalGet(self.g_hp));
        self.ins(Instruction::LocalGet(self.l_sz));
        self.ins(Instruction::I32Add);
        self.ins(Instruction::MemorySize(0));
        self.ins(Instruction::I32Const(16));
        self.ins(Instruction::I32Shl);
        self.ins(Instruction::I32GtU);
        self.ins(Instruction::If(BlockType::Empty));
        // pages = ceil((__hp + size - membytes) / 65536)
        self.ins(Instruction::GlobalGet(self.g_hp));
        self.ins(Instruction::LocalGet(self.l_sz));
        self.ins(Instruction::I32Add);
        self.ins(Instruction::MemorySize(0));
        self.ins(Instruction::I32Const(16));
        self.ins(Instruction::I32Shl);
        self.ins(Instruction::I32Sub);
        self.ins(Instruction::I32Const(65535));
        self.ins(Instruction::I32Add);
        self.ins(Instruction::I32Const(16));
        self.ins(Instruction::I32ShrU);
        self.ins(Instruction::MemoryGrow(0));
        self.ins(Instruction::I32Const(-1));
        self.ins(Instruction::I32Eq);
        self.ins(Instruction::If(BlockType::Empty));
        self.ins(Instruction::Unreachable);
        self.ins(Instruction::End);
        self.ins(Instruction::End);
        self.ins(Instruction::GlobalGet(self.g_hp));
        self.ins(Instruction::GlobalGet(self.g_hp));
        self.ins(Instruction::LocalGet(self.l_sz));
        self.ins(Instruction::I32Add);
        self.ins(Instruction::GlobalSet(self.g_hp));
    }

    /// only `AccessKind::Direct` field/index reads lower to bare word loads —
    /// Option access would need a Null representation we don't model
    fn direct(&self, kind: &compile::AccessKind) -> Result<(), Bail> {
        match kind {
            compile::AccessKind::Direct => Ok(()),
            _ => bail!("optional access needs Null values"),
        }
    }

    /// one NON-terminator op (regions handle jumps/calls/returns)
    fn emit_op(&mut self, i: usize) -> Result<(), Bail> {
        let (_, op) = &self.ops[i];
        // Move and heap ops are raw word operations — they emit even when the
        // dst has no scalar mask (a handle is never scalar-read)
        if let Some(d) = op_dst(op) {
            if !matches!(op, Op::Move { .. })
                && !self.mask.contains_key(&(d.index() as u32))
                && !self.reads_w.contains(&(d.index() as u32))
            {
                return Ok(());
            }
        }
        match op {
            Op::Move { dst, src } => {
                // raw 8-byte slot copy — class- and handle-agnostic
                self.fp();
                self.ld_word(*src);
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            Op::LoadConst { dst, constant } => match constant {
                compile::Constant::Int(v) => {
                    let v = *v;
                    self.setv(*dst, K::Int, |s| {
                        s.ins(Instruction::I64Const(v));
                        Ok(())
                    })?;
                }
                compile::Constant::Float(v) => {
                    let v = (*v).into();
                    self.setv(*dst, K::Float, |s| {
                        s.ins(Instruction::F64Const(v));
                        Ok(())
                    })?;
                }
                compile::Constant::Bool(v) => {
                    let v = *v as i32;
                    self.setv(*dst, K::Bool, |s| {
                        s.ins(Instruction::I32Const(v));
                        Ok(())
                    })?;
                }
                compile::Constant::Null => {
                    // raw zero word — Null is only meaningful to word-reads
                    self.fp();
                    self.ins(Instruction::I64Const(0));
                    self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
                }
                c => bail!("LoadConst {c:?}"),
            },
            Op::AddInt { dst, left, right } => {
                self.setv(*dst, K::Int, |s| s.checked_int(*left, *right, BinOp::Add))?;
            }
            Op::SubInt { dst, left, right } => {
                self.setv(*dst, K::Int, |s| s.checked_int(*left, *right, BinOp::Sub))?;
            }
            Op::MultInt { dst, left, right } => {
                self.setv(*dst, K::Int, |s| s.checked_int(*left, *right, BinOp::Mult))?;
            }
            Op::ModInt { dst, left, right } => {
                self.setv(*dst, K::Int, |s| s.checked_int(*left, *right, BinOp::Mod))?;
            }
            Op::AddIntImm { dst, left, val } => {
                self.setv(*dst, K::Int, |s| s.checked_int_imm(*left, *val, BinOp::Add))?;
            }
            Op::SubIntImm { dst, left, val } => {
                self.setv(*dst, K::Int, |s| s.checked_int_imm(*left, *val, BinOp::Sub))?;
            }
            Op::MultIntImm { dst, left, val } => {
                self.setv(*dst, K::Int, |s| s.checked_int_imm(*left, *val, BinOp::Mult))?;
            }
            Op::ModIntImm { dst, left, val } => {
                self.setv(*dst, K::Int, |s| s.checked_int_imm(*left, *val, BinOp::Mod))?;
            }
            Op::IntLt { dst, left, right }
            | Op::IntLe { dst, left, right }
            | Op::IntGt { dst, left, right }
            | Op::IntGe { dst, left, right }
            | Op::IntEq { dst, left, right }
            | Op::IntNe { dst, left, right } => {
                let i = match op {
                    Op::IntLt { .. } => Instruction::I64LtS,
                    Op::IntLe { .. } => Instruction::I64LeS,
                    Op::IntGt { .. } => Instruction::I64GtS,
                    Op::IntGe { .. } => Instruction::I64GeS,
                    Op::IntEq { .. } => Instruction::I64Eq,
                    _ => Instruction::I64Ne,
                };
                self.setv(*dst, K::Bool, |s| {
                    s.get(*left, K::Int)?;
                    s.get(*right, K::Int)?;
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::IntLtImm { dst, left, val }
            | Op::IntLeImm { dst, left, val }
            | Op::IntGtImm { dst, left, val }
            | Op::IntGeImm { dst, left, val }
            | Op::IntEqImm { dst, left, val }
            | Op::IntNeImm { dst, left, val } => {
                let (i, v) = match op {
                    Op::IntLtImm { .. } => (Instruction::I64LtS, *val),
                    Op::IntLeImm { .. } => (Instruction::I64LeS, *val),
                    Op::IntGtImm { .. } => (Instruction::I64GtS, *val),
                    Op::IntGeImm { .. } => (Instruction::I64GeS, *val),
                    Op::IntEqImm { .. } => (Instruction::I64Eq, *val),
                    _ => (Instruction::I64Ne, *val),
                };
                self.setv(*dst, K::Bool, |s| {
                    s.get(*left, K::Int)?;
                    s.ins(Instruction::I64Const(v));
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                let i = match op {
                    Op::AddFloat { .. } => Instruction::F64Add,
                    Op::SubFloat { .. } => Instruction::F64Sub,
                    Op::MultFloat { .. } => Instruction::F64Mul,
                    _ => Instruction::F64Div,
                };
                self.setv(*dst, K::Float, |s| {
                    s.get(*left, K::Float)?;
                    s.get(*right, K::Float)?;
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::FloatLt { dst, left, right }
            | Op::FloatLe { dst, left, right }
            | Op::FloatGt { dst, left, right }
            | Op::FloatGe { dst, left, right }
            | Op::FloatEq { dst, left, right }
            | Op::FloatNe { dst, left, right } => {
                let i = match op {
                    Op::FloatLt { .. } => Instruction::F64Lt,
                    Op::FloatLe { .. } => Instruction::F64Le,
                    Op::FloatGt { .. } => Instruction::F64Gt,
                    Op::FloatGe { .. } => Instruction::F64Ge,
                    Op::FloatEq { .. } => Instruction::F64Eq,
                    _ => Instruction::F64Ne,
                };
                self.setv(*dst, K::Bool, |s| {
                    s.get(*left, K::Float)?;
                    s.get(*right, K::Float)?;
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::AddFloatImm { dst, left, val }
            | Op::SubFloatImm { dst, left, val }
            | Op::MultFloatImm { dst, left, val } => {
                let i = match op {
                    Op::AddFloatImm { .. } => Instruction::F64Add,
                    Op::SubFloatImm { .. } => Instruction::F64Sub,
                    _ => Instruction::F64Mul,
                };
                let v = f64::from_bits(*val as u64).into();
                self.setv(*dst, K::Float, |s| {
                    s.get(*left, K::Float)?;
                    s.ins(Instruction::F64Const(v));
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::ModFloatImm { dst, left, val } => {
                return self.mod_float(*dst, *left, f64::from_bits(*val as u64));
            }
            Op::FloatLtImm { dst, left, val }
            | Op::FloatLeImm { dst, left, val }
            | Op::FloatGtImm { dst, left, val }
            | Op::FloatGeImm { dst, left, val }
            | Op::FloatEqImm { dst, left, val }
            | Op::FloatNeImm { dst, left, val } => {
                let (i, v) = match op {
                    Op::FloatLtImm { .. } => (Instruction::F64Lt, f64::from_bits(*val as u64)),
                    Op::FloatLeImm { .. } => (Instruction::F64Le, f64::from_bits(*val as u64)),
                    Op::FloatGtImm { .. } => (Instruction::F64Gt, f64::from_bits(*val as u64)),
                    Op::FloatGeImm { .. } => (Instruction::F64Ge, f64::from_bits(*val as u64)),
                    Op::FloatEqImm { .. } => (Instruction::F64Eq, f64::from_bits(*val as u64)),
                    _ => (Instruction::F64Ne, f64::from_bits(*val as u64)),
                };
                self.setv(*dst, K::Bool, |s| {
                    s.get(*left, K::Float)?;
                    s.ins(Instruction::F64Const(v.into()));
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::BoolEq { dst, left, right } | Op::BoolNe { dst, left, right } => {
                let i = if matches!(op, Op::BoolEq { .. }) {
                    Instruction::I32Eq
                } else {
                    Instruction::I32Ne
                };
                self.setv(*dst, K::Bool, |s| {
                    s.get(*left, K::Bool)?;
                    s.get(*right, K::Bool)?;
                    s.ins(i);
                    Ok(())
                })?;
            }
            Op::ToFloat { dst, src } => {
                self.setv(*dst, K::Float, |s| {
                    s.get(*src, K::Int)?;
                    s.ins(Instruction::F64ConvertI64S);
                    Ok(())
                })?;
            }
            Op::Sqrt { dst, src } => {
                self.setv(*dst, K::Float, |s| {
                    s.get(*src, K::Float)?;
                    s.ins(Instruction::F64Sqrt);
                    Ok(())
                })?;
            }
            Op::Unary { dst, op: uop, src } => {
                // operand class for heap-loaded words: best-known class,
                // else Float (numbers dominate untyped unary use)
                let sk = self
                    .pick(*dst)
                    .or_else(|| self.pick(*src))
                    .unwrap_or(K::Float);
                match uop {
                    UnaryOp::Negative => match sk {
                        K::Word => unreachable!(),
                        K::Int => self.setv(*dst, K::Int, |s| {
                            s.ins(Instruction::I64Const(0));
                            s.get(*src, K::Int)?;
                            s.ins(Instruction::I64Sub);
                            s.ins(Instruction::LocalSet(s.l_tmp));
                            s.get(*src, K::Int)?;
                            s.ins(Instruction::I64Const(i64::MIN));
                            s.ins(Instruction::I64Eq);
                            s.ins(Instruction::If(BlockType::Empty));
                            s.ins(Instruction::Unreachable);
                            s.ins(Instruction::End);
                            s.ins(Instruction::LocalGet(s.l_tmp));
                            Ok(())
                        })?,
                        K::Float => self.setv(*dst, K::Float, |s| {
                            s.get(*src, K::Float)?;
                            s.ins(Instruction::F64Neg);
                            Ok(())
                        })?,
                        K::Bool => bail!("neg bool"),
                    },
                    // mimas `+x` is checked abs on ints, fabs on floats
                    UnaryOp::Positive => match sk {
                        K::Word => unreachable!(),
                        K::Int => self.setv(*dst, K::Int, |s| {
                            s.get(*src, K::Int)?;
                            s.ins(Instruction::LocalSet(s.l_tmp));
                            s.ins(Instruction::LocalGet(s.l_tmp));
                            s.ins(Instruction::I64Const(i64::MIN));
                            s.ins(Instruction::I64Eq);
                            s.ins(Instruction::If(BlockType::Empty));
                            s.ins(Instruction::Unreachable);
                            s.ins(Instruction::End);
                            s.ins(Instruction::LocalGet(s.l_tmp));
                            s.ins(Instruction::I64Const(0));
                            s.ins(Instruction::I64LtS);
                            s.ins(Instruction::If(BlockType::Result(ValType::I64)));
                            s.ins(Instruction::I64Const(0));
                            s.ins(Instruction::LocalGet(s.l_tmp));
                            s.ins(Instruction::I64Sub);
                            s.ins(Instruction::Else);
                            s.ins(Instruction::LocalGet(s.l_tmp));
                            s.ins(Instruction::End);
                            Ok(())
                        })?,
                        K::Float => self.setv(*dst, K::Float, |s| {
                            s.get(*src, K::Float)?;
                            s.ins(Instruction::F64Abs);
                            Ok(())
                        })?,
                        K::Bool => bail!("abs bool"),
                    },
                    UnaryOp::Not => match sk {
                        K::Word => unreachable!(),
                        K::Bool => self.setv(*dst, K::Bool, |s| {
                            s.get(*src, K::Bool)?;
                            s.ins(Instruction::I32Eqz);
                            Ok(())
                        })?,
                        _ => bail!("not on {sk:?}"),
                    },
                    UnaryOp::BitwiseNot => self.setv(*dst, K::Int, |s| {
                        s.get(*src, K::Int)?;
                        s.ins(Instruction::I64Const(-1));
                        s.ins(Instruction::I64Xor);
                        Ok(())
                    })?,
                }
            }
            Op::Bin { dst, left, op: bop, right } => {
                self.emit_bin(*dst, *left, *bop, *right)?;
            }
            Op::CallNative { dst, id, args } => {
                let mut params = Vec::new();
                for a in args {
                    params.push(self.pick(*a).unwrap_or(K::Word));
                }
                // classless dst that a heap op reads later must round-trip the
                // word — passthrough natives (cov::pass & co) rely on it
                let dk = self.pick(*dst).or_else(|| {
                    self.reads_w
                        .contains(&(dst.index() as u32))
                        .then_some(K::Word)
                });
                let ret = dk.unwrap_or(K::Int);
                let fi = self.natives[&(id.index() as u32, params.clone(), Some(ret))];
                for (a, &k) in args.iter().zip(&params) {
                    self.get(*a, k)?;
                }
                self.ins(Instruction::Call(fi));
                match dk {
                    Some(k) => {
                        self.ins(match k {
                            K::Int | K::Word => Instruction::LocalSet(self.l_tmp),
                            K::Float => Instruction::LocalSet(self.l_tmpf),
                            K::Bool => Instruction::LocalSet(self.l_tmpb),
                        });
                        self.fp();
                        self.ins(match k {
                            K::Int | K::Word => Instruction::LocalGet(self.l_tmp),
                            K::Float => Instruction::LocalGet(self.l_tmpf),
                            K::Bool => Instruction::LocalGet(self.l_tmpb),
                        });
                        self.st(*dst, k);
                    }
                    None => self.ins(Instruction::Drop),
                }
            }
            Op::LoadBody { .. } => {}
            // ---------- heap ops: bump-arena objects in linear memory ----------
            // array object = 16B header [data_ptr][len][cap][pad] (indirect —
            // Move'd aliases share the header, so realloc updates every alias)
            // instance = [adt u32][pad u32][field words] — fields at +8+slot*8
            Op::NewArray { dst } => {
                self.ins(Instruction::I32Const(16));
                self.alloc();
                self.ins(Instruction::LocalSet(self.l_hp));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64Store(mem(0, 3)));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64Store(mem(8, 3)));
                self.fp();
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I64ExtendI32U);
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            Op::Push { array, value } => {
                // l_hp = array header
                self.ld_word(*array);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::LocalSet(self.l_hp));
                // l_tmpb = len
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(4, 2)));
                self.ins(Instruction::LocalSet(self.l_tmpb));
                // grow when len == cap
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(4, 2)));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(8, 2)));
                self.ins(Instruction::I32Eq);
                self.ins(Instruction::If(BlockType::Empty));
                // newcap = cap ? cap*2 : 4
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(8, 2)));
                self.ins(Instruction::I32Eqz);
                self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                self.ins(Instruction::I32Const(4));
                self.ins(Instruction::Else);
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(8, 2)));
                self.ins(Instruction::I32Const(2));
                self.ins(Instruction::I32Mul);
                self.ins(Instruction::End);
                self.ins(Instruction::LocalSet(self.l_tmpc));
                // l_tmp = newptr (i64 form)
                self.ins(Instruction::LocalGet(self.l_tmpc));
                self.ins(Instruction::I32Const(8));
                self.ins(Instruction::I32Mul);
                self.alloc();
                self.ins(Instruction::I64ExtendI32U);
                self.ins(Instruction::LocalSet(self.l_tmp));
                // memory.copy(newptr, ptr, len*8)
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(0, 2)));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ins(Instruction::I32Const(8));
                self.ins(Instruction::I32Mul);
                self.ins(Instruction::MemoryCopy { src_mem: 0, dst_mem: 0 });
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::LocalGet(self.l_tmp));
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Store(mem(0, 2)));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::LocalGet(self.l_tmpc));
                self.ins(Instruction::I32Store(mem(8, 2)));
                self.ins(Instruction::End);
                // ptr[len] = value word; len += 1
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Load(mem(0, 2)));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ins(Instruction::I32Const(8));
                self.ins(Instruction::I32Mul);
                self.ins(Instruction::I32Add);
                self.ld_word(*value);
                self.ins(Instruction::I64Store(mem(0, 3)));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ins(Instruction::I32Const(1));
                self.ins(Instruction::I32Add);
                self.ins(Instruction::I32Store(mem(4, 2)));
            }
            Op::GetIndex { dst, set, index, kind } => {
                self.direct(kind)?;
                // l_tmpb = index (bounds-checked below)
                self.get(*index, K::Int)?;
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::LocalSet(self.l_tmpb));
                // if idx >= len → trap (interp: IndexOutOfBounds)
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ld_word(*set);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Load(mem(4, 2)));
                self.ins(Instruction::I32GeU);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
                // dst = word(ptr + idx*8)
                self.fp();
                self.ld_word(*set);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Load(mem(0, 2)));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ins(Instruction::I32Const(8));
                self.ins(Instruction::I32Mul);
                self.ins(Instruction::I32Add);
                self.ins(Instruction::I64Load(mem(0, 3)));
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            Op::SetIndex { set, index, value, .. } => {
                self.get(*index, K::Int)?;
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::LocalSet(self.l_tmpb));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ld_word(*set);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Load(mem(4, 2)));
                self.ins(Instruction::I32GeU);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::Unreachable);
                self.ins(Instruction::End);
                self.ld_word(*set);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Load(mem(0, 2)));
                self.ins(Instruction::LocalGet(self.l_tmpb));
                self.ins(Instruction::I32Const(8));
                self.ins(Instruction::I32Mul);
                self.ins(Instruction::I32Add);
                self.ld_word(*value);
                self.ins(Instruction::I64Store(mem(0, 3)));
            }
            Op::NewInstance { dst, adt, fields } => {
                let n = fields.len() as i32;
                self.ins(Instruction::I32Const(8 + n * 8));
                self.alloc();
                self.ins(Instruction::LocalSet(self.l_hp));
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I32Const(adt.index() as i32));
                self.ins(Instruction::I32Store(mem(0, 2)));
                for (fi, f) in fields.iter().enumerate() {
                    self.ins(Instruction::LocalGet(self.l_hp));
                    self.ld_word(*f);
                    self.ins(Instruction::I64Store(mem(8 + fi as u32 * 8, 3)));
                }
                self.fp();
                self.ins(Instruction::LocalGet(self.l_hp));
                self.ins(Instruction::I64ExtendI32U);
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            Op::GetField { dst, src, slot, kind } => {
                self.direct(kind)?;
                self.fp();
                self.ld_word(*src);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I64Load(mem(8 + *slot * 8, 3)));
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            Op::SetField { receiver, slot, value } => {
                self.ld_word(*receiver);
                self.ins(Instruction::I32WrapI64);
                self.ld_word(*value);
                self.ins(Instruction::I64Store(mem(8 + *slot * 8, 3)));
            }
            Op::Len { dst, src } => {
                self.fp();
                self.ld_word(*src);
                self.ins(Instruction::I32WrapI64);
                self.ins(Instruction::I32Load(mem(4, 2)));
                self.ins(Instruction::I64ExtendI32U);
                self.ins(Instruction::I64Store(mem(dst.index() as u32 * 8, 3)));
            }
            other => bail!("unsupported op {other:?}"),
        }
        if self.opts.coverage {
            let cov = self.cov_base + i as u32;
            self.ins(Instruction::I32Const(cov as i32));
            self.ins(Instruction::I32Const(1));
            self.ins(Instruction::I32Store8(mem(0, 0)));
        }
        Ok(())
    }

    fn mod_float(&mut self, dst: Reg, l: Reg, v: f64) -> Result<(), Bail> {
        self.setv(dst, K::Float, |s| {
            s.get(l, K::Float)?;
            s.ins(Instruction::LocalSet(s.l_tmpf));
            s.ins(Instruction::LocalGet(s.l_tmpf));
            s.ins(Instruction::LocalGet(s.l_tmpf));
            s.ins(Instruction::F64Const(v.into()));
            s.ins(Instruction::F64Div);
            s.ins(Instruction::F64Trunc);
            s.ins(Instruction::F64Const(v.into()));
            s.ins(Instruction::F64Mul);
            s.ins(Instruction::F64Sub);
            Ok(())
        })
    }
}

fn is_term(op: &Op) -> bool {
    matches!(
        op,
        Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::ForNext { .. }
            | Op::Switch { .. }
            | Op::Return { .. }
            | Op::CallDirect { .. }
            | Op::Call { .. }
            | Op::BIntLt { .. }
            | Op::BIntLe { .. }
            | Op::BIntGt { .. }
            | Op::BIntGe { .. }
            | Op::BIntEq { .. }
            | Op::BIntNe { .. }
            | Op::BIntLtImm { .. }
            | Op::BIntLeImm { .. }
            | Op::BIntGtImm { .. }
            | Op::BIntGeImm { .. }
            | Op::BIntEqImm { .. }
            | Op::BIntNeImm { .. }
            | Op::BFloatLt { .. }
            | Op::BFloatLe { .. }
            | Op::BFloatGt { .. }
            | Op::BFloatGe { .. }
            | Op::BFloatEq { .. }
            | Op::BFloatNe { .. }
            | Op::BFloatLtImm { .. }
            | Op::BFloatLeImm { .. }
            | Op::BFloatGtImm { .. }
            | Op::BFloatGeImm { .. }
            | Op::BFloatEqImm { .. }
            | Op::BFloatNeImm { .. }
    )
}

/// op index of a jump target (absolute byte offset -> op index)
fn tgt_idx(t: &compile::BlockTarget, off2idx: &HashMap<usize, usize>) -> Result<usize, Bail> {
    let compile::BlockTarget::ByteOffset(to) = t else {
        bail!("unresolved block target");
    };
    off2idx.get(to).copied().ok_or_else(|| format!("jump target {to} mid-op"))
}

impl RE<'_> {
    /// normal completion: status=0, pc_table=0, sp=myfp, return
    fn finish(&mut self, val: Reg) -> Result<(), Bail> {
        self.ins(Instruction::I32Const(0));
        self.ins(Instruction::GlobalSet(self.g_status));
        self.ins(Instruction::I32Const(self.pc_addr() as i32));
        self.ins(Instruction::I32Const(0));
        self.ins(Instruction::I32Store(mem(0, 2)));
        self.ins(Instruction::LocalGet(self.l_myfp));
        self.ins(Instruction::GlobalSet(self.g_sp));
        if let Some(k) = self.ret_k {
            self.get(val, k)?;
        }
        self.ins(Instruction::Return);
        Ok(())
    }

    /// a call terminator: set callee frame base, call, propagate suspend,
    /// store dst, goto next region
    fn emit_call(&mut self, i: usize, fsize: u32) -> Result<(), Bail> {
        let (dst, args, fi, params, ret_k) = {
            let (_, op) = &self.ops[i];
            let (dst, b, args) = match op {
                Op::CallDirect { dst, body, args } => (*dst, body.index(), args.as_slice()),
                Op::Call { dst, args, .. } => {
                    let b = *self
                        .callee
                        .get(&i)
                        .ok_or("dynamic Call to non-const callee")?;
                    (*dst, b, args.as_slice())
                }
                _ => bail!("emit_call on non-call"),
            };
            let fi = self.func_map[&b];
            let sig = self.sigs[b].as_ref().ok_or("callee sig missing")?;
            if sig.params.len() != args.len() {
                bail!("arity mismatch calling body {b}");
            }
            (dst, args.to_vec(), fi, sig.params.clone(), sig.ret)
        };
        // __sp = myfp + fsize (callee's frame base)
        self.ins(Instruction::LocalGet(self.l_myfp));
        self.ins(Instruction::I32Const(fsize as i32));
        self.ins(Instruction::I32Add);
        self.ins(Instruction::GlobalSet(self.g_sp));
        for (a, &pk) in args.iter().zip(&params) {
            self.get(*a, pk)?;
        }
        self.ins(Instruction::Call(fi));
        // propagate suspend
        self.ins(Instruction::GlobalGet(self.g_status));
        self.ins(Instruction::If(BlockType::Empty));
        self.suspend_keep(0, true); // callee already set the status
        self.ins(Instruction::End);
        let store_k = if self.mask.contains_key(&(dst.index() as u32)) {
            ret_k
        } else if self.reads_w.contains(&(dst.index() as u32)) {
            ret_k.map(|_| K::Word)
        } else {
            None
        };
        match store_k {
            Some(k) => {
                if k == K::Word {
                    // callee's declared ret → raw i64 word
                    match ret_k {
                        Some(K::Float) => self.ins(Instruction::I64ReinterpretF64),
                        Some(K::Bool) => self.ins(Instruction::I64ExtendI32U),
                        _ => {}
                    }
                    self.ins(Instruction::LocalSet(self.l_tmp));
                    self.fp();
                    self.ins(Instruction::LocalGet(self.l_tmp));
                } else {
                    self.ins(match k {
                        K::Int | K::Word => Instruction::LocalSet(self.l_tmp),
                        K::Float => Instruction::LocalSet(self.l_tmpf),
                        K::Bool => Instruction::LocalSet(self.l_tmpb),
                    });
                    self.fp();
                    self.ins(match k {
                        K::Int | K::Word => Instruction::LocalGet(self.l_tmp),
                        K::Float => Instruction::LocalGet(self.l_tmpf),
                        K::Bool => Instruction::LocalGet(self.l_tmpb),
                    });
                }
                self.st(dst, k);
            }
            None => {
                if ret_k.is_some() {
                    self.ins(Instruction::Drop);
                }
            }
        }
        Ok(())
    }

    /// region terminator at op index i
    fn emit_term(&mut self, i: usize, fsize: u32) -> Result<(), Bail> {
        let (_, op) = &self.ops[i];
        let mut off2idx = HashMap::new();
        for (ix, (off, _)) in self.ops.iter().enumerate() {
            off2idx.insert(*off, ix);
        }
        match op {
            Op::Jump { target } => {
                let r = self.region_of[tgt_idx(target, &off2idx)?];
                self.goto(r);
            }
            Op::JumpIf {
                cond,
                target,
                is_true,
            } => {
                let c = *cond;
                let it = *is_true;
                let r = self.region_of[tgt_idx(target, &off2idx)?];
                let f = self.region + 1;
                self.goto_if(r, f, it, |s| s.get(c, K::Bool))?;
            }
            Op::ForNext { idx, bound, target } => {
                let (ix, bound) = (*idx, *bound);
                // idx += 1
                self.fp();
                self.get(ix, K::Int)?;
                self.ins(Instruction::I64Const(1));
                self.ins(Instruction::I64Add);
                self.st(ix, K::Int);
                let r = self.region_of[tgt_idx(target, &off2idx)?];
                let f = self.region + 1;
                self.goto_if(r, f, true, |s| {
                    s.get(ix, K::Int)?;
                    s.get(bound, K::Int)?;
                    s.ins(Instruction::I64LtS);
                    Ok(())
                })?;
            }
            Op::Switch {
                scrut,
                base,
                default,
                table,
            } => {
                // nested blocks: innermost = last table case; each case's end
                // lands on `pc = region(table[k]); br $L`
                let n = table.len() as u32;
                self.ins(Instruction::Block(BlockType::Empty)); // default
                for _ in 0..n {
                    self.ins(Instruction::Block(BlockType::Empty));
                }
                self.get(*scrut, K::Int)?;
                self.ins(Instruction::I64Const(*base as i64));
                self.ins(Instruction::I64Sub);
                self.ins(Instruction::I32WrapI64);
                let entries: Vec<u32> = (0..n).map(|k| n - 1 - k).collect();
                self.ins(Instruction::BrTable(entries.into(), n));
                for k in 0..n {
                    self.ins(Instruction::End); // close case block k (innermost first)
                    let r = self.region_of[tgt_idx(&table[k as usize], &off2idx)?];
                    self.goto_case(r, (n - 1 - k) + 1);
                }
                self.ins(Instruction::End); // close default
                let r = self.region_of[tgt_idx(default, &off2idx)?];
                self.ins(Instruction::I32Const(r as i32));
                self.ins(Instruction::LocalSet(self.l_pc));
                self.ins(Instruction::Br(1));
            }
            Op::Return { val } => {
                let v = *val;
                self.finish(v)?;
            }
            Op::CallDirect { .. } | Op::Call { .. } => {
                self.emit_call(i, fsize)?;
                let f = self.region + 1;
                self.ins(Instruction::I32Const(f as i32));
                self.ins(Instruction::LocalSet(self.l_pc));
                self.ins(Instruction::Br(self.region + 1));
            }
            _ => {
                // B* conditional branches
                let (target, is_true) = match op {
                    Op::BIntLt { target, is_true, .. }
                    | Op::BIntLe { target, is_true, .. }
                    | Op::BIntGt { target, is_true, .. }
                    | Op::BIntGe { target, is_true, .. }
                    | Op::BIntEq { target, is_true, .. }
                    | Op::BIntNe { target, is_true, .. }
                    | Op::BIntLtImm { target, is_true, .. }
                    | Op::BIntLeImm { target, is_true, .. }
                    | Op::BIntGtImm { target, is_true, .. }
                    | Op::BIntGeImm { target, is_true, .. }
                    | Op::BIntEqImm { target, is_true, .. }
                    | Op::BIntNeImm { target, is_true, .. }
                    | Op::BFloatLt { target, is_true, .. }
                    | Op::BFloatLe { target, is_true, .. }
                    | Op::BFloatGt { target, is_true, .. }
                    | Op::BFloatGe { target, is_true, .. }
                    | Op::BFloatEq { target, is_true, .. }
                    | Op::BFloatNe { target, is_true, .. }
                    | Op::BFloatLtImm { target, is_true, .. }
                    | Op::BFloatLeImm { target, is_true, .. }
                    | Op::BFloatGtImm { target, is_true, .. }
                    | Op::BFloatGeImm { target, is_true, .. }
                    | Op::BFloatEqImm { target, is_true, .. }
                    | Op::BFloatNeImm { target, is_true, .. } => (target.clone(), *is_true),
                    _ => bail!("emit_term on non-terminator"),
                };
                let r = self.region_of[tgt_idx(&target, &off2idx)?];
                let f = self.region + 1;
                match op {
                    Op::BIntLt { left, right, .. }
                    | Op::BIntLe { left, right, .. }
                    | Op::BIntGt { left, right, .. }
                    | Op::BIntGe { left, right, .. }
                    | Op::BIntEq { left, right, .. }
                    | Op::BIntNe { left, right, .. } => {
                        let (l, rr) = (*left, *right);
                        let i = match op {
                            Op::BIntLt { .. } => Instruction::I64LtS,
                            Op::BIntLe { .. } => Instruction::I64LeS,
                            Op::BIntGt { .. } => Instruction::I64GtS,
                            Op::BIntGe { .. } => Instruction::I64GeS,
                            Op::BIntEq { .. } => Instruction::I64Eq,
                            _ => Instruction::I64Ne,
                        };
                        self.goto_if(r, f, is_true, |s| {
                            s.get(l, K::Int)?;
                            s.get(rr, K::Int)?;
                            s.ins(i);
                            Ok(())
                        })?;
                    }
                    Op::BIntLtImm { left, val, .. }
                    | Op::BIntLeImm { left, val, .. }
                    | Op::BIntGtImm { left, val, .. }
                    | Op::BIntGeImm { left, val, .. }
                    | Op::BIntEqImm { left, val, .. }
                    | Op::BIntNeImm { left, val, .. } => {
                        let (l, v) = (*left, *val);
                        let i = match op {
                            Op::BIntLtImm { .. } => Instruction::I64LtS,
                            Op::BIntLeImm { .. } => Instruction::I64LeS,
                            Op::BIntGtImm { .. } => Instruction::I64GtS,
                            Op::BIntGeImm { .. } => Instruction::I64GeS,
                            Op::BIntEqImm { .. } => Instruction::I64Eq,
                            _ => Instruction::I64Ne,
                        };
                        self.goto_if(r, f, is_true, |s| {
                            s.get(l, K::Int)?;
                            s.ins(Instruction::I64Const(v));
                            s.ins(i);
                            Ok(())
                        })?;
                    }
                    Op::BFloatLt { left, right, .. }
                    | Op::BFloatLe { left, right, .. }
                    | Op::BFloatGt { left, right, .. }
                    | Op::BFloatGe { left, right, .. }
                    | Op::BFloatEq { left, right, .. }
                    | Op::BFloatNe { left, right, .. } => {
                        let (l, rr) = (*left, *right);
                        let i = match op {
                            Op::BFloatLt { .. } => Instruction::F64Lt,
                            Op::BFloatLe { .. } => Instruction::F64Le,
                            Op::BFloatGt { .. } => Instruction::F64Gt,
                            Op::BFloatGe { .. } => Instruction::F64Ge,
                            Op::BFloatEq { .. } => Instruction::F64Eq,
                            _ => Instruction::F64Ne,
                        };
                        self.goto_if(r, f, is_true, |s| {
                            s.get(l, K::Float)?;
                            s.get(rr, K::Float)?;
                            s.ins(i);
                            Ok(())
                        })?;
                    }
                    Op::BFloatLtImm { left, val, .. }
                    | Op::BFloatLeImm { left, val, .. }
                    | Op::BFloatGtImm { left, val, .. }
                    | Op::BFloatGeImm { left, val, .. }
                    | Op::BFloatEqImm { left, val, .. }
                    | Op::BFloatNeImm { left, val, .. } => {
                        let (l, bits) = (*left, *val);
                        let i = match op {
                            Op::BFloatLtImm { .. } => Instruction::F64Lt,
                            Op::BFloatLeImm { .. } => Instruction::F64Le,
                            Op::BFloatGtImm { .. } => Instruction::F64Gt,
                            Op::BFloatGeImm { .. } => Instruction::F64Ge,
                            Op::BFloatEqImm { .. } => Instruction::F64Eq,
                            _ => Instruction::F64Ne,
                        };
                        self.goto_if(r, f, is_true, |s| {
                            s.get(l, K::Float)?;
                            s.ins(Instruction::F64Const(f64::from_bits(bits as u64).into()));
                            s.ins(i);
                            Ok(())
                        })?;
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }

    /// `pc = r; br` — used inside Switch cases where $L's depth differs
    fn goto_case(&mut self, region: u32, l_depth: u32) {
        self.ins(Instruction::I32Const(region as i32));
        self.ins(Instruction::LocalSet(self.l_pc));
        self.ins(Instruction::Br(l_depth));
    }
}

/// partition `ops` into regions: boundaries at 0, every jump target, every
/// call op, and the op after every terminator.
fn regions(ops: &[(usize, Op)]) -> Result<(Vec<(usize, usize)>, Vec<u32>), Bail> {
    let mut off2idx = HashMap::new();
    for (i, (off, _)) in ops.iter().enumerate() {
        off2idx.insert(*off, i);
    }
    let n = ops.len();
    let mut bounds = BTreeSet::from([0usize]);
    for (i, (_, op)) in ops.iter().enumerate() {
        match op {
            Op::Jump { target } | Op::JumpIf { target, .. } | Op::ForNext { target, .. } => {
                bounds.insert(tgt_idx(target, &off2idx)?);
            }
            Op::Switch { table, default, .. } => {
                bounds.insert(tgt_idx(default, &off2idx)?);
                for t in table {
                    bounds.insert(tgt_idx(t, &off2idx)?);
                }
            }
            _ => {
                if let Some(t) = match op {
                    Op::BIntLt { target, .. }
                    | Op::BIntLe { target, .. }
                    | Op::BIntGt { target, .. }
                    | Op::BIntGe { target, .. }
                    | Op::BIntEq { target, .. }
                    | Op::BIntNe { target, .. }
                    | Op::BIntLtImm { target, .. }
                    | Op::BIntLeImm { target, .. }
                    | Op::BIntGtImm { target, .. }
                    | Op::BIntGeImm { target, .. }
                    | Op::BIntEqImm { target, .. }
                    | Op::BIntNeImm { target, .. }
                    | Op::BFloatLt { target, .. }
                    | Op::BFloatLe { target, .. }
                    | Op::BFloatGt { target, .. }
                    | Op::BFloatGe { target, .. }
                    | Op::BFloatEq { target, .. }
                    | Op::BFloatNe { target, .. }
                    | Op::BFloatLtImm { target, .. }
                    | Op::BFloatLeImm { target, .. }
                    | Op::BFloatGtImm { target, .. }
                    | Op::BFloatGeImm { target, .. }
                    | Op::BFloatEqImm { target, .. }
                    | Op::BFloatNeImm { target, .. } => Some(target.clone()),
                    _ => None,
                } {
                    bounds.insert(tgt_idx(&t, &off2idx)?);
                }
            }
        }
        if is_term(op) {
            if i + 1 < n {
                bounds.insert(i + 1);
            }
        }
    }
    let bvec: Vec<usize> = bounds.into_iter().collect();
    let mut rs = Vec::new();
    let mut region_of = vec![0u32; n];
    for (j, &start) in bvec.iter().enumerate() {
        let end = bvec.get(j + 1).copied().unwrap_or(n);
        if start < end {
            rs.push((start, end));
        }
    }
    // renumber: start boundary -> its index in rs
    let mut ix_of_start = HashMap::new();
    for (j, (s, _)) in rs.iter().enumerate() {
        ix_of_start.insert(*s, j as u32);
    }
    for (j, (s, e)) in rs.iter().enumerate() {
        for i in *s..*e {
            region_of[i] = j as u32;
        }
    }
    Ok((rs, region_of))
}

fn try_resume_body(
    b: usize,
    ops: &[(usize, Op)],
    program: &Program,
    ana: &crate::Ana,
    sigs: &[Option<Sig>],
    func_map: &HashMap<usize, u32>,
    natives: &HashMap<(u32, Vec<K>, Option<K>), u32>,
    opts: &Opts,
    cov_base: u32,
    g_fuel: Option<u32>,
    g_pause: Option<u32>,
    g_status: u32,
    g_sp: u32,
    g_hp: u32,
) -> Result<(Function, Vec<(u32, u32)>), Bail> {
    let chunk = &program.chunks[compile::BodyId::from(b as u32)];
    let Some(sig) = &sigs[b] else { bail!("no sig") };
    let (rs, region_of) = regions(ops)?;
    let nparams = chunk.params.len() as u32;
    let mut re = RE {
        ops,
        mask: &ana.mask,
        reads_w: &ana.reads_w,
        sigs,
        func_map,
        callee: &ana.callee,
        natives,
        ret_k: sig.ret,
        body: b,
        region_of: &region_of,
        c: Vec::new(),
        srcmap: Vec::new(),
        opts,
        cov_base,
        g_fuel,
        g_pause,
        g_status,
        g_sp,
        g_hp,
        l_pc: nparams,
        l_myfp: nparams + 1,
        l_tmp: nparams + 2,
        l_tmpf: nparams + 3,
        l_tmpb: nparams + 4,
        l_tmpc: nparams + 5,
        l_hp: nparams + 6,
        l_sz: nparams + 7,
        region: 0,
    };
    let fsize = (chunk.regs as u32) * 8 + 8;

    // entry: myfp = __sp; __sp += fsize; pc = pc_table[b]; fresh → params+zero-init
    re.ins(Instruction::GlobalGet(g_sp));
    re.ins(Instruction::LocalSet(re.l_myfp));
    re.ins(Instruction::LocalGet(re.l_myfp));
    re.ins(Instruction::I32Const(fsize as i32));
    re.ins(Instruction::I32Add);
    re.ins(Instruction::GlobalSet(g_sp));
    re.ins(Instruction::I32Const(re.pc_addr() as i32));
    re.ins(Instruction::I32Load(mem(0, 2)));
    re.ins(Instruction::LocalSet(re.l_pc));
    re.ins(Instruction::LocalGet(re.l_pc));
    re.ins(Instruction::I32Eqz);
    re.ins(Instruction::If(BlockType::Empty));
    for (pi, p) in chunk.params.iter().enumerate() {
        let k = sig.params[pi];
        let off = p.index() as u32 * 8;
        re.fp();
        re.ins(Instruction::LocalGet(pi as u32));
        re.ins(match k {
            K::Int | K::Word => Instruction::I64Store(mem(off, 3)),
            K::Float => Instruction::F64Store(mem(off, 3)),
            K::Bool => Instruction::I32Store(mem(off, 2)),
        });
    }
    re.ins(Instruction::End);

    let r = rs.len();
    re.ins(Instruction::Loop(BlockType::Empty));
    re.ins(Instruction::Block(BlockType::Empty)); // $exit
    for _ in 0..r {
        re.ins(Instruction::Block(BlockType::Empty));
    }
    // dispatch: pc=j → depth r-1-j ; default → $exit (depth r)
    re.ins(Instruction::LocalGet(re.l_pc));
    let entries: Vec<u32> = (0..r as u32).map(|j| r as u32 - 1 - j).collect();
    re.ins(Instruction::BrTable(entries.into(), r as u32));
    re.ins(Instruction::Unreachable);

    for j in (0..r).rev() {
        re.ins(Instruction::End); // close B_j → region j code follows
        re.region = j as u32;
        let (start, end) = rs[j];
        re.region_head(end - start);
        for i in start..end {
            re.srcmap.push((re.c.len() as u32, i as u32));
            let (_, op) = &re.ops[i];
            if is_term(op) {
                re.emit_term(i, fsize)?;
            } else {
                re.emit_op(i)?;
            }
        }
        // fallthrough to next region
        if !is_term(&re.ops[end - 1].1) {
            re.goto(j as u32 + 1);
        }
    }
    re.ins(Instruction::End); // close $exit
    re.ins(Instruction::Unreachable);
    re.ins(Instruction::End); // close $L — a loop can always "fall through"
    // its end in the validator's eyes, so seal the func's stack-type here
    re.ins(Instruction::Unreachable);
    re.ins(Instruction::End); // function end

    let mut locals: Vec<(u32, ValType)> = vec![
        (1, ValType::I32), // pc
        (1, ValType::I32), // myfp
        (1, ValType::I64), // tmp
        (1, ValType::F64), // tmpf
        (1, ValType::I32), // tmpb
        (1, ValType::I32), // tmpc
        (1, ValType::I32), // hp
        (1, ValType::I32), // sz
    ];
    let mut f = Function::new(locals.drain(..));
    f.raw(re.c.iter().copied());
    Ok((f, re.srcmap))
}

/// Emit the resumable lane: pc-dispatch + memory-resident registers. Same
/// coverage as `emit` for op semantics, plus ALL control flow (no irreducible
/// skips). Every emitted body suspends/resumes on `__pause`/`__fuel`.
/// emit-side `pick` (mask-based, Float>Int>Bool) for pre-pass collection
fn ana_pick(ana: &crate::Ana, r: Reg) -> Option<K> {
    let m = *ana.mask.get(&(r.index() as u32))?;
    Some(if m & crate::K_FLOAT != 0 {
        K::Float
    } else if m & crate::K_INT != 0 {
        K::Int
    } else {
        K::Bool
    })
}

pub fn emit_resumable(program: &Program, opts: &Opts) -> Result<Wasmgen, Bail> {
    let nbodies = program.chunks.len();
    let bodies_ops: Vec<Vec<(usize, Op)>> = (0..nbodies)
        .map(|b| program.ops(compile::BodyId::from(b as u32)))
        .collect();

    let mut ret: Vec<Option<K>> = vec![None; nbodies];
    let mut anas: Vec<Option<crate::Ana>> = (0..nbodies).map(|_| None).collect();
    for _ in 0..16 {
        let mut stable = true;
        for b in 0..nbodies {
            let nregs = program.chunks[compile::BodyId::from(b as u32)].regs as u32;
            match analyze(&bodies_ops[b], nregs, &ret) {
                Ok(a) => {
                    if a.ret != ret[b] {
                        ret[b] = a.ret;
                        stable = false;
                    }
                    anas[b] = Some(a);
                }
                Err(_) => {}
            }
        }
        if stable {
            break;
        }
    }
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
        let params: Vec<K> = chunk
            .params
            .iter()
            // param never read: any class works — i64 word is the widest
            .map(|p| ana_pick(ana, *p).unwrap_or(K::Int))
            .collect();
        sigs[b] = Some(Sig {
            params,
            ret: ana.ret,
        });
    }

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
                    .map(|a| ana_pick(ana, *a).unwrap_or(K::Word))
                    .collect();
                let ret = Some(
                    ana_pick(ana, *dst)
                        .or_else(|| {
                            ana.reads_w
                                .contains(&(dst.index() as u32))
                                .then_some(K::Word)
                        })
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
    let nimports = native_list.len() as u32;

    // settle: a body survives iff its callees are emitted AND a trial
    // emission succeeds — emit failures are skips, not fatal
    let mut emitted: Vec<bool> = sigs.iter().map(|s| s.is_some()).collect();
    let dummy_map: HashMap<usize, u32> = (0..nbodies).map(|b| (b, b as u32)).collect();
    let mut reasons: Vec<String> = (0..nbodies).map(|_| String::new()).collect();
    loop {
        let mut changed = false;
        for b in 0..nbodies {
            if !emitted[b] {
                continue;
            }
            let ana = anas[b].as_ref().unwrap();
            let mut why = String::new();
            if let Some(c) = ana.callee.values().find(|c| !emitted[**c]) {
                why = format!("calls skipped body {c}");
            } else {
                if let Err(e) = try_resume_body(
                    b,
                    &bodies_ops[b],
                    program,
                    ana,
                    &sigs,
                    &dummy_map,
                    &natives,
                    opts,
                    0,
                    None,
                    None,
                    0,
                    1,
                    2,
                ) {
                    why = e;
                }
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
    let mut func_map: HashMap<usize, u32> = HashMap::new();
    for (fi, b) in emitted
        .iter()
        .enumerate()
        .filter(|(_, e)| **e)
        .map(|(i, _)| i)
        .enumerate()
    {
        func_map.insert(b, nimports + fi as u32);
    }

    let mut types = TypeSection::new();
    let mut imports = ImportSection::new();
    let mut funcs = FunctionSection::new();
    let mut code = CodeSection::new();
    let mut exports = ExportSection::new();
    let mut globals = GlobalSection::new();
    let mut names = NameMap::new();
    let mut out_bodies: Vec<Body> = Vec::new();
    let mut type_ids: HashMap<(Vec<K>, Option<K>), u32> = HashMap::new();
    let mut srcmaps: Vec<(u32, Vec<(u32, u32)>)> = Vec::new();
    let mut locs: Vec<(u32, u32, u32)> = Vec::new();
    let mut loc_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut cov_next = 0u32;

    for (id, params, ret) in &native_list {
        let key = (params.clone(), *ret);
        let ntypes = types.len();
        let ty = *type_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types.ty().function(
                params.iter().map(|k| k.val_type()),
                ret.map(|k| k.val_type()),
            );
        }
        let sig_desc = params
            .iter()
            .map(|k| match k {
                K::Int => 'i',
                K::Float => 'f',
                K::Bool => 'b',
                K::Word => 'w',
            })
            .collect::<String>();
        let ret_desc = match ret {
            Some(K::Int) => 'i',
            Some(K::Float) => 'f',
            Some(K::Bool) => 'b',
            Some(K::Word) => 'w',
            None => 'v',
        };
        imports.import(
            "env",
            &format!("n{id}_{sig_desc}_{ret_desc}"),
            EntityType::Function(ty),
        );
    }

    let mut gnext = 0u32;
    let mut import_global = |imports: &mut ImportSection, name: &str, vt| {
        let g = gnext;
        gnext += 1;
        imports.import(
            "env",
            name,
            EntityType::Global(GlobalType {
                val_type: vt,
                mutable: true,
                shared: false,
            }),
        );
        g
    };
    let g_fuel = opts
        .fuel
        .then(|| import_global(&mut imports, "__fuel", ValType::I64));
    let g_pause = opts
        .pause
        .then(|| import_global(&mut imports, "__pause", ValType::I32));
    let g_status = gnext;
    gnext += 1;
    let g_sp = gnext;
    gnext += 1;
    let g_hp = gnext;
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &ConstExpr::i32_const(0),
    );
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &ConstExpr::i32_const(nbodies as i32 * 8), // __sp starts at STACK
    );
    // heap arena follows the coverage region: arrays/instances are bump-
    // allocated blocks of 8-byte words
    let total_ops: u32 = (0..nbodies).map(|b| bodies_ops[b].len() as u32).sum();
    let stack_base = nbodies as u32 * 8;
    let cov_addr = stack_base + STACK_CAP;
    let cov_end = if opts.coverage { cov_addr + total_ops + 64 } else { cov_addr };
    let heap_base = (cov_end + 7) & !7;
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &ConstExpr::i32_const(heap_base as i32), // __hp: heap bump pointer
    );

    for b in 0..nbodies {
        let Some(sig) = &sigs[b] else { continue };
        let ana = anas[b].as_ref().unwrap();
        let key = (sig.params.clone(), sig.ret);
        let ntypes = types.len();
        let ty = *type_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types.ty().function(
                sig.params.iter().map(|k| k.val_type()),
                sig.ret.map(|k| k.val_type()),
            );
        }
        funcs.function(ty);
        let cov_base = if opts.coverage {
            let cb = cov_next;
            cov_next += bodies_ops[b].len() as u32;
            nbodies as u32 * 8 + STACK_CAP + cb // coverage region base + body offset
        } else {
            0
        };
        let (f, sm) = try_resume_body(
            b,
            &bodies_ops[b],
            program,
            ana,
            &sigs,
            &func_map,
            &natives,
            opts,
            cov_base,
            g_fuel,
            g_pause,
            g_status,
            g_sp,
            g_hp,
        )
        .map_err(|e| format!("body {b}: {e}"))?;
        let fidx = func_map[&b];
        code.function(&f);
        let name = format!("b{b}");
        exports.export(&name, ExportKind::Func, fidx);
        names.append(fidx, &name);
        let chunk = &program.chunks[compile::BodyId::from(b as u32)];
        let sm_loc: Vec<(u32, u32)> = sm
            .iter()
            .map(|&(off, i)| {
                let (boff, _) = bodies_ops[b][i as usize];
                let loc = chunk.loc_at((boff - chunk.offset) as u32);
                let key = (
                    loc.file_id as u32,
                    loc.span.start as u32,
                    loc.span.end as u32,
                );
                let li = *loc_map.entry(key).or_insert_with(|| {
                    locs.push(key);
                    (locs.len() - 1) as u32
                });
                (off, li)
            })
            .collect();
        srcmaps.push((fidx, sm_loc));
        out_bodies.push(Body {
            body: b,
            func: fidx,
            name,
        });
    }

    let pages = (heap_base as u64 + PAGE) / PAGE;

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&mems);
    module.section(&globals);
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("__status", ExportKind::Global, g_status);
    exports.export("__sp", ExportKind::Global, g_sp);
    exports.export("__hp", ExportKind::Global, g_hp);
    module.section(&exports);
    module.section(&code);
    let mut ns = NameSection::new();
    ns.functions(&names);
    module.section(&ns);
    if !srcmaps.is_empty() {
        let mut d = Vec::new();
        d.extend_from_slice(&0x4D534D31u32.to_le_bytes());
        d.extend_from_slice(&(srcmaps.len() as u32).to_le_bytes());
        for (fi, entries) in &srcmaps {
            d.extend_from_slice(&fi.to_le_bytes());
            d.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for (off, li) in entries {
                d.extend_from_slice(&off.to_le_bytes());
                d.extend_from_slice(&li.to_le_bytes());
            }
        }
        d.extend_from_slice(&(locs.len() as u32).to_le_bytes());
        for (f, lo, hi) in &locs {
            d.extend_from_slice(&f.to_le_bytes());
            d.extend_from_slice(&lo.to_le_bytes());
            d.extend_from_slice(&hi.to_le_bytes());
        }
        module.section(&CustomSection {
            name: std::borrow::Cow::Borrowed("mimas.srcmap"),
            data: std::borrow::Cow::Borrowed(&d),
        });
    }
    Ok(Wasmgen {
        bytes: module.finish(),
        bodies: out_bodies,
        skipped,
    })
}
