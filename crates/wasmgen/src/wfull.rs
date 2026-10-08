//! `emit_waffle_ir` — the production lane: `compile::Ir` bodies -> waffle
//! `FunctionBody`s -> wasm. Inst semantics (the tagged heap, dict/str
//! helpers, coverage sink, host ABI) live in the per-inst arms below;
//! waffle's reducify/stackify/localify backend owns CFG lowering, so any
//! CFG shape — irreducible included — emits.
//!
//! ## Representation
//!
//! Mimas locals become SSA by threading one typed blockparam per *live*
//! local through every CFG edge (max-SSA; `localify` prunes); `Phi` insts
//! become real blockparams. Each SSA value carries its stored class
//! (`class`/`lclass`, else `Word`).
//!
//! Wasm scratch locals (`tmp`/`sz`/`hp`/`tb`/`tc`) become plain SSA values.
//! Mid-block `if`s inside inst arms (`Len`/`GetIndex`/`Push`/`In`/checked
//! arithmetic) become CondBr splits via `branch`/`join`/`trap_if`; a
//! value-producing `if` lowers to a merge blockparam.
//!
//! The runtime helpers (`__alloc`, `__str_*`, `__dict_*`, `__cov_*`) ride
//! along as `FuncDecl::Compiled` — `wrt::emit_helpers` produces raw
//! wasm_encoder bytes spliced into the code section.

use std::collections::{HashMap, HashSet};

use crate::wrt::{
    AnaI, COV_HELPER_NAMES, CovCtx, DCELL, HELPER_NAMES, OPSTK_N, SINK_CAP, STACK_CAP, Statics,
    TAG_ARRAY, TAG_CLOSURE, TAG_DICT, TAG_INSTANCE, TAG_STR, analyze_body, binop_prod,
    emit_helpers, emit_trampoline, native_key,
};
use crate::{Bail, Body, K, Opts, Sig, Skip, Wasmgen};
use compile::{
    BinOp, BlockId, Body as IrBody, Constant, FormatPart, Inst, InstId, Ir, OperandKind, UnaryOp,
};
use shared::{StrId, StrInterner};
use waffle::{
    Block as WBlock, BlockTarget, Export, ExportKind, Func, FuncDecl, FunctionBody, Global,
    GlobalData, Import, ImportKind, Memory, MemoryData, MemorySegment, Module, Operator as WOp,
    Signature, SignatureData, Table, TableData, Terminator, Type as WTy, Value, entity::EntityRef,
};

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

/// body-block successors mirroring codegen's convention: `Jump` is the
/// fallthrough (unconditional edge), `JumpIfFalse`/`ForNext`/`Switch` are the
/// branch edges.
fn succs(body: &IrBody, bid: BlockId) -> (Option<BlockId>, Vec<BlockId>) {
    let mut ft = None;
    let mut brs = Vec::new();
    for &iid in &body.blocks[bid].stream {
        match &body.instructions[iid] {
            Inst::Jump { target } => ft = Some(*target),
            Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => brs.push(*target),
            Inst::Switch { table, default, .. } => {
                brs.extend(table.iter().copied());
                brs.push(*default);
            }
            _ => {}
        }
    }
    (ft, brs)
}

/// fallthrough-first DFS from entry — matches codegen's serialization order.
fn dfs_order(body: &IrBody) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut placed = vec![false; body.blocks.len()];
    let mut stack = vec![BlockId::ZERO];
    while let Some(b) = stack.pop() {
        if placed[b.index()] {
            continue;
        }
        placed[b.index()] = true;
        order.push(b);
        let (ft, brs) = succs(body, b);
        for t in brs {
            stack.push(t);
        }
        if let Some(f) = ft {
            stack.push(f);
        }
    }
    order
}

/// Word bytes for one const-array element — Float rides as raw f64 bits,
/// Str as its static object address, nested arrays recurse (children bake
/// first so their addresses exist when the parent's words are written).
fn const_word(ctx: &Statics, c: &Constant) -> Result<i64, Bail> {
    Ok(match c {
        Constant::Int(v) => *v,
        Constant::Float(v) => v.to_bits() as i64,
        Constant::Bool(v) => *v as i64,
        Constant::Null => 0,
        Constant::Str(s) => *ctx
            .str_objs
            .get(&(s.index() as u32))
            .ok_or("str in const array not laid out")? as i64,
        Constant::Array(es) => bake_const_array_obj(ctx, es)? as i64,
    })
}

/// Bake `elems` as a tagged `TAG_ARRAY` object in the statics hole:
/// `[tag][data][len][cap]` header followed by the 8-byte element words.
/// Identical arrays dedup on their serialized element words.
fn bake_const_array_obj(ctx: &Statics, elems: &[Constant]) -> Result<u32, Bail> {
    let mut words = Vec::with_capacity(elems.len() * 8);
    for e in elems {
        words.extend(const_word(ctx, e)?.to_le_bytes());
    }
    if let Some(&a) = ctx.arr_objs.borrow().get(&words) {
        return Ok(a);
    }
    let n = elems.len() as u32;
    let addr = ctx.arr_cur.get();
    let end = addr + 16 + n * 8;
    if end > ctx.arr_cap {
        return Err("const-array statics overflow the 1MB hole".into());
    }
    let mut b = Vec::with_capacity((end - addr) as usize);
    b.extend(TAG_ARRAY.to_le_bytes());
    b.extend((addr + 16).to_le_bytes());
    b.extend(n.to_le_bytes());
    b.extend(n.to_le_bytes());
    b.extend_from_slice(&words);
    ctx.arr_cur.set(end);
    ctx.arr_statics.borrow_mut().push((addr, b));
    ctx.arr_objs.borrow_mut().insert(words, addr);
    Ok(addr)
}

/// `cov_base` sentinel — no per-inst coverage bytes.
const NO_COV: u32 = u32::MAX;

fn wty(k: K) -> WTy {
    match k {
        K::Int | K::Word => WTy::I64,
        K::Float => WTy::F64,
        K::Bool => WTy::I32,
    }
}

fn vty2wty(v: wasm_encoder::ValType) -> WTy {
    match v {
        wasm_encoder::ValType::I32 => WTy::I32,
        wasm_encoder::ValType::I64 => WTy::I64,
        wasm_encoder::ValType::F32 => WTy::F32,
        wasm_encoder::ValType::F64 => WTy::F64,
        _ => WTy::I64,
    }
}

fn marg(offset: u32, align: u32) -> waffle::MemoryArg {
    waffle::MemoryArg {
        align,
        offset,
        memory: Memory::new(0),
    }
}

// ---------- module ----------

/// Emit the module from `ir` through waffle. Skipped bodies keep trapping
/// stubs (callers are skipped by dep propagation, so the interpreter lane
/// owns them).
pub fn emit_waffle_ir(
    ir: &Ir,
    strs: &StrInterner,
    opts: &Opts,
    sink: Option<&crate::CovSink>,
    math: Option<&crate::MathNatives>,
    cov_points: u32,
    cov_decs: u32,
) -> Result<Wasmgen, Bail> {
    let empty_sink = crate::CovSink::new();
    let sink = sink.unwrap_or(&empty_sink);
    let empty_math = crate::MathNatives::new();
    let math = math.unwrap_or(&empty_math);
    let nbodies = ir.bodies.len();

    // locals addressed by GetEntry/SetEntry anywhere — shared memory region
    let mut entry_locals: HashSet<u32> = HashSet::new();
    for (_, body) in ir.bodies.iter() {
        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                match &body.instructions[iid] {
                    Inst::GetEntry(l) | Inst::SetEntry(l, _) => {
                        entry_locals.insert(l.index() as u32);
                    }
                    _ => {}
                }
            }
        }
    }
    let bodies: Vec<&IrBody> = ir.bodies.iter().map(|(_, b)| b).collect();
    let ninsts: Vec<usize> = bodies
        .iter()
        .map(|b| {
            b.blocks
                .iter()
                .map(|(_, bl)| bl.stream.len())
                .sum::<usize>()
        })
        .collect();

    // cross-body fixpoint on return classes
    let mut ret: Vec<Option<K>> = vec![None; nbodies];
    let mut anas: Vec<Option<AnaI>> = (0..nbodies).map(|_| None).collect();
    for _ in 0..16 {
        let mut stable = true;
        for b in 0..nbodies {
            match analyze_body(bodies[b], &ret) {
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

    // ---- static string objects: bake tagged Str objects into memory ----
    let mut str_objs: HashMap<u32, u32> = HashMap::new();
    {
        let mut ids: Vec<u32> = Vec::new();
        for (_, body) in ir.bodies.iter() {
            for (_, block) in body.blocks.iter() {
                for &iid in &block.stream {
                    match &body.instructions[iid] {
                        Inst::Constant(c) => {
                            // Str consts nested in const arrays need objects too
                            let mut stack = vec![c];
                            while let Some(c) = stack.pop() {
                                match c {
                                    Constant::Str(s) => ids.push(s.index() as u32),
                                    Constant::Array(es) => stack.extend(es.iter()),
                                    _ => {}
                                }
                            }
                        }
                        Inst::Format(parts) => {
                            for p in parts {
                                if let FormatPart::Literal(s) = p {
                                    ids.push(s.index() as u32);
                                }
                            }
                        }
                        Inst::Insert { key, .. } => ids.push(key.index() as u32),
                        _ => {}
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            str_objs.insert(id, 0);
        }
    }

    let mut skipped: Vec<Skip> = Vec::new();
    let mut sigs: Vec<Option<Sig>> = (0..nbodies).map(|_| None).collect();
    for b in 0..nbodies {
        let body = bodies[b];
        let Some(ana) = &anas[b] else {
            skipped.push(Skip {
                body: b,
                reason: "analysis failed".into(),
            });
            continue;
        };
        let mut params: Vec<K> = body
            .captures
            .iter()
            .map(|p| {
                ana.lclass
                    .get(&(p.index() as u32))
                    .copied()
                    .unwrap_or(K::Word)
            })
            .collect();
        params.extend(body.params.iter().map(|p| {
            ana.lclass
                .get(&(p.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        }));
        sigs[b] = Some(Sig {
            params,
            ret: ana.ret,
        });
    }

    // ---- memory layout: statics / heap base / cov plane ----
    let cov_total: u32 = if opts.coverage {
        (0..nbodies)
            .filter(|b| sigs[*b].is_some())
            .map(|b| ninsts[b] as u32)
            .sum()
    } else {
        0
    };
    let entry_base = nbodies as u32 * 8;
    let entry_bytes = bodies[0].locals.len() as u32 * 8;
    let sp_init = entry_base + entry_bytes;
    let mut off = (sp_init + 15) & !15;
    let mut statics: Vec<MemorySegment> = Vec::new();
    let mut put_str = |off: &mut u32, s: &str| -> u32 {
        let addr = *off;
        let mut b = Vec::with_capacity(16 + s.len());
        b.extend(TAG_STR.to_le_bytes());
        b.extend((addr + 16).to_le_bytes());
        b.extend((s.len() as u32).to_le_bytes());
        b.extend((s.len() as u32).to_le_bytes());
        b.extend(s.as_bytes());
        while b.len() % 8 != 0 {
            b.push(0);
        }
        *off += b.len() as u32;
        statics.push(MemorySegment {
            offset: addr as usize,
            data: b,
        });
        addr
    };
    let true_obj = put_str(&mut off, "true");
    let false_obj = put_str(&mut off, "false");
    let obj_obj = put_str(&mut off, "<obj>");
    let null_obj = put_str(&mut off, "null");
    let str_ids: Vec<u32> = str_objs.keys().copied().collect();
    for id in str_ids {
        let addr = put_str(&mut off, strs.get(StrId::from(id)));
        *str_objs.get_mut(&id).unwrap() = addr;
    }
    let max_call_args = ir
        .bodies
        .iter()
        .flat_map(|(_, body)| {
            body.blocks.iter().flat_map(|(_, block)| {
                block
                    .stream
                    .iter()
                    .filter_map(|&iid| match &body.instructions[iid] {
                        Inst::Call { args, .. } => Some(args.len()),
                        _ => None,
                    })
            })
        })
        .max()
        .unwrap_or(0)
        .max(8) as u32;
    let call_scratch = off;
    off += (max_call_args * 8 + 15) & !15;
    let i64_scratch = off;
    off += 32;
    if off > sp_init + STACK_CAP {
        return Err("static data overflows the 1MB stack hole".into());
    }
    let cov0 = sp_init + STACK_CAP;
    let ptmap_base = (cov0 + cov_total + 7) & !7;
    let decv_base = (ptmap_base + cov_points + 7) & !7;
    let opstk_base = decv_base + cov_decs * DCELL;
    let sink_base = (opstk_base + OPSTK_N * 16 + 7) & !7;
    let heap_base = sink_base + if sink.is_empty() { 0 } else { SINK_CAP };
    let ctx = Statics {
        str_objs,
        true_obj,
        false_obj,
        obj_obj,
        null_obj,
        call_scratch,
        i64_scratch,
        // const arrays bake lazily past the fixed statics, capped at cov0
        arr_cur: std::cell::Cell::new(off),
        arr_cap: cov0,
        arr_objs: std::cell::RefCell::new(HashMap::new()),
        arr_statics: std::cell::RefCell::new(Vec::new()),
    };

    let mut module = Module::empty();
    // signature dedup keyed on the (params, ret) class tuple
    let mut sig_map: HashMap<(Vec<K>, Option<K>), Signature> = HashMap::new();
    let sig_of = |module: &mut Module,
                  sig_map: &mut HashMap<(Vec<K>, Option<K>), Signature>,
                  params: &[K],
                  ret: Option<K>|
     -> Signature {
        *sig_map.entry((params.to_vec(), ret)).or_insert_with(|| {
            module.signatures.push(SignatureData {
                params: params.iter().map(|k| wty(*k)).collect(),
                returns: ret.iter().map(|k| wty(*k)).collect(),
            })
        })
    };

    // ---- native imports: (id, params, ret) per call site, same ABI names ----
    let mut natives: HashMap<(u32, Vec<K>, Option<K>), Func> = HashMap::new();
    let mut native_list: Vec<((u32, Vec<K>, Option<K>), Func)> = Vec::new();
    for b in 0..nbodies {
        if sigs[b].is_none() {
            continue;
        }
        let ana = anas[b].as_ref().unwrap();
        for (_, block) in bodies[b].blocks.iter() {
            for &iid in &block.stream {
                if let Inst::CallNative { id, .. } = &bodies[b].instructions[iid] {
                    if sink.contains_key(&(id.index() as u32)) {
                        continue;
                    }
                    let key = native_key(bodies[b], ana, iid);
                    if !natives.contains_key(&key) {
                        let (id, params, ret) = key.clone();
                        let sig = sig_of(&mut module, &mut sig_map, &params, ret);
                        let name = native_name(id, &params, ret);
                        let f = Func::new(module.funcs.len());
                        module.funcs.push(FuncDecl::Import(sig, name.clone()));
                        module.imports.push(Import {
                            module: "env".into(),
                            name,
                            kind: ImportKind::Func(f),
                        });
                        natives.insert(key, f);
                        native_list.push(((id, params, ret), f));
                    }
                }
            }
        }
    }
    let nimports = module.funcs.len();

    // ---- imported instrumentation globals (fuel/pause), then defined ----
    let fuel_g = opts.fuel.then(|| {
        let g = Global::new(module.globals.len());
        module.globals.push(GlobalData {
            ty: WTy::I64,
            value: None,
            mutable: true,
        });
        module.imports.push(Import {
            module: "env".into(),
            name: "__fuel".into(),
            kind: ImportKind::Global(g),
        });
        g
    });
    let pause_g = opts.pause.then(|| {
        let g = Global::new(module.globals.len());
        module.globals.push(GlobalData {
            ty: WTy::I32,
            value: None,
            mutable: true,
        });
        module.imports.push(Import {
            module: "env".into(),
            name: "__pause".into(),
            kind: ImportKind::Global(g),
        });
        g
    });
    let hp_g = Global::new(module.globals.len());
    let g_status = Global::new(module.globals.len() + 1);
    let g_sp = Global::new(module.globals.len() + 2);
    let g_covp = Global::new(module.globals.len() + 3);
    let g_covbuf = Global::new(module.globals.len() + 4);
    let g_ptmap = Global::new(module.globals.len() + 5);
    let g_osp = Global::new(module.globals.len() + 6);
    let g_dcov = Global::new(module.globals.len() + 7);
    for (ty_v, mutable) in [
        (heap_base as u64, true),   // __hp
        (0, true),                  // __status
        (sp_init as u64, true),     // __sp
        (0, true),                  // __covp
        (sink_base as u64, false),  // __covbuf
        (ptmap_base as u64, false), // __ptmap
        (0, true),                  // __osp
        (decv_base as u64, false),  // __dcov
    ] {
        module.globals.push(GlobalData {
            ty: WTy::I32,
            value: Some(ty_v),
            mutable,
        });
    }

    // ---- body func slots: stubs for every body, replaced on emit ----
    let mut funcs: HashMap<usize, Func> = HashMap::new();
    let mut wsig: Vec<Option<Signature>> = (0..nbodies).map(|_| None).collect();
    for (bid, _) in ir.bodies.iter() {
        let b = bid.index();
        let Some(sig) = &sigs[b] else { continue };
        let ws = sig_of(&mut module, &mut sig_map, &sig.params, sig.ret);
        wsig[b] = Some(ws);
        let mut stub = FunctionBody::new(&module, ws);
        stub.set_terminator(stub.entry, Terminator::Unreachable);
        let f = Func::new(module.funcs.len());
        module.funcs.push(FuncDecl::Body(ws, format!("b{b}"), stub));
        funcs.insert(b, f);
    }
    let nbodies_slots = funcs.len() as u32;

    // ---- helper funcs come after body slots; cov helpers follow ----
    let ncov = if sink.is_empty() {
        0
    } else {
        COV_HELPER_NAMES.len()
    } as u32;
    let helper_base = nimports as u32 + nbodies_slots;
    let mut helpers_u32: HashMap<&'static str, u32> = HELPER_NAMES
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, helper_base + i as u32))
        .collect();
    for (i, &n) in COV_HELPER_NAMES.iter().enumerate().take(ncov as usize) {
        helpers_u32.insert(n, helper_base + HELPER_NAMES.len() as u32 + i as u32);
    }
    let helpers: HashMap<&'static str, Func> = helpers_u32
        .iter()
        .map(|(&n, &i)| (n, Func::new(i as usize)))
        .collect();
    let covctx = (!sink.is_empty()).then(|| CovCtx {
        ptmap_base,
        decv_base,
        opstk_base,
        sink_base,
        g_covp: g_covp.index() as u32,
        g_osp: g_osp.index() as u32,
        h: COV_HELPER_NAMES
            .iter()
            .map(|n| helpers_u32[*n])
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap(),
    });

    // ---- trampolines: one table slot per RefBody/MakeClosure target ----
    let mut tramp_targets: Vec<usize> = Vec::new();
    let mut has_dyn_call = false;
    for b in 0..nbodies {
        if sigs[b].is_none() {
            continue;
        }
        for (_, block) in bodies[b].blocks.iter() {
            for &iid in &block.stream {
                match &bodies[b].instructions[iid] {
                    Inst::RefBody(t) | Inst::MakeClosure { body: t, .. } => {
                        let t = t.index();
                        if sigs[t].is_some() && !tramp_targets.contains(&t) {
                            tramp_targets.push(t);
                        }
                    }
                    Inst::Call { .. } => has_dyn_call = true,
                    _ => {}
                }
            }
        }
    }
    let tramp: HashMap<usize, u32> = tramp_targets
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i as u32))
        .collect();
    let tramp_sig = sig_of(
        &mut module,
        &mut sig_map,
        &[K::Bool, K::Bool, K::Bool],
        Some(K::Int),
    ); // (i32,i32,i32)->i64 uniform trampoline sig
    let needs_table = has_dyn_call || !tramp_targets.is_empty();

    // ---- probe emit: build every body; settle on call/ref deps ----
    let mut built: Vec<Option<FunctionBody>> = (0..nbodies).map(|_| None).collect();
    let mut ok: Vec<bool> = sigs.iter().map(|s| s.is_some()).collect();
    let mut reasons: Vec<String> = (0..nbodies).map(|_| String::new()).collect();
    for b in 0..nbodies {
        if sigs[b].is_none() {
            continue;
        }
        match wbody_full(
            &module,
            wsig[b].unwrap(),
            ir,
            b,
            &anas[b].as_ref().unwrap(),
            &sigs,
            &funcs,
            &natives,
            &helpers,
            &tramp,
            tramp_sig,
            &entry_locals,
            entry_base,
            &ctx,
            fuel_g,
            pause_g,
            if opts.coverage { cov0 + 0 } else { NO_COV },
            sink,
            math,
            covctx.as_ref(),
        ) {
            Ok(fb) => built[b] = Some(fb),
            Err(e) => {
                ok[b] = false;
                reasons[b] = e;
            }
        }
    }
    // fixpoint: a body survives iff its static call/ref targets survive
    loop {
        let mut changed = false;
        for b in 0..nbodies {
            if !ok[b] {
                continue;
            }
            let ana = anas[b].as_ref().unwrap();
            let mut why = String::new();
            if let Some(&c) = ana.callee.values().find(|c| !ok[**c]) {
                why = format!("calls skipped body {c}");
            } else {
                for (_, block) in bodies[b].blocks.iter() {
                    for &iid in &block.stream {
                        match &bodies[b].instructions[iid] {
                            Inst::RefBody(t) | Inst::MakeClosure { body: t, .. }
                                if !ok[t.index()] =>
                            {
                                why = format!("references skipped body {}", t.index());
                            }
                            _ => {}
                        }
                    }
                }
            }
            if !why.is_empty() {
                ok[b] = false;
                reasons[b] = why;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let wf_debug = std::env::var("WF_DEBUG").is_ok();
    let wf_only: Option<usize> = std::env::var("WF_ONLY").ok().and_then(|s| s.parse().ok());
    let mut out_bodies: Vec<Body> = Vec::new();
    for b in 0..nbodies {
        if sigs[b].is_some() && !ok[b] {
            skipped.push(Skip {
                body: b,
                reason: reasons[b].clone(),
            });
            continue;
        }
        if let Some(fb) = built[b].take() {
            if wf_debug {
                if let Err(e) = fb.validate() {
                    eprintln!("wfull body {b}: {e}");
                }
            }
            if wf_only == Some(b) && std::env::var("WF_DUMP").is_ok() {
                eprintln!("{}", fb.display_verbose("| ", None));
            }
            let f = funcs[&b];
            let ws = wsig[b].unwrap();
            // WF_ONLY bisects: leave non-target bodies as unreachable stubs
            let fb = if wf_only.is_none() || wf_only == Some(b) {
                fb
            } else {
                let mut stub = FunctionBody::new(&module, ws);
                stub.set_terminator(stub.entry, Terminator::Unreachable);
                stub
            };
            module.funcs[f] = FuncDecl::Body(ws, format!("b{b}"), fb);
            module.exports.push(Export {
                name: format!("b{b}"),
                kind: ExportKind::Func(f),
            });
            out_bodies.push(Body {
                body: b,
                func: f.index() as u32,
                name: format!("b{b}"),
            });
        }
    }
    out_bodies.sort_by_key(|b| b.body);

    // ---- helpers as precompiled bodies; trampolines too ----
    let helper_fns = emit_helpers(&ctx, &helpers_u32, hp_g.index() as u32, covctx.as_ref());
    debug_assert_eq!(helper_fns.len(), HELPER_NAMES.len() + ncov as usize);
    for h in helper_fns {
        let sig = module.signatures.push(SignatureData {
            params: h.params.iter().map(|v| vty2wty(*v)).collect(),
            returns: h.rets.iter().map(|v| vty2wty(*v)).collect(),
        });
        let buf = h.f.into_raw_body();
        let f = Func::new(module.funcs.len());
        module
            .funcs
            .push(FuncDecl::Compiled(sig, h.name.into(), buf));
        debug_assert_eq!(f.index() as u32, helpers_u32[h.name]);
    }
    // trampolines: emit with real callee indices; table position = tramp idx
    let mut tramp_funcs: Vec<Func> = Vec::new();
    for &t in &tramp_targets {
        let f_enc = emit_trampoline(
            bodies[t],
            sigs[t].as_ref().unwrap(),
            funcs[&t].index() as u32,
        );
        let buf = f_enc.into_raw_body();
        let f = Func::new(module.funcs.len());
        module
            .funcs
            .push(FuncDecl::Compiled(tramp_sig, format!("__tr_b{t}"), buf));
        tramp_funcs.push(f);
    }

    // ---- table / memory / globals / exports ----
    if needs_table {
        module.tables.push(TableData {
            ty: WTy::FuncRef,
            initial: tramp_targets.len() as u64,
            max: None,
            func_elements: Some(tramp_funcs),
        });
    }
    let pages = (heap_base as u64 + 65535) / 65536;
    // lazily-baked const-array objects join the fixed statics
    for (addr, bytes) in ctx.arr_statics.borrow_mut().drain(..) {
        statics.push(MemorySegment {
            offset: addr as usize,
            data: bytes,
        });
    }
    module.memories.push(MemoryData {
        initial_pages: pages.max(1) as usize,
        maximum_pages: None,
        segments: statics,
    });
    module.exports.push(Export {
        name: "memory".into(),
        kind: ExportKind::Memory(Memory::new(0)),
    });
    for (name, g) in [
        ("__status", g_status),
        ("__sp", g_sp),
        ("__hp", hp_g),
        ("__covp", g_covp),
        ("__covbuf", g_covbuf),
        ("__ptmap", g_ptmap),
        ("__dcov", g_dcov),
    ] {
        module.exports.push(Export {
            name: name.into(),
            kind: ExportKind::Global(g),
        });
    }

    let bytes = module
        .to_wasm_bytes()
        .map_err(|e| format!("waffle backend: {e:?}"))?;
    Ok(Wasmgen {
        bytes,
        bodies: out_bodies,
        skipped,
    })
}

/// `n{id}_{param-classes}_{ret}` — the wgame dispatcher parses this name.
fn native_name(id: u32, params: &[K], ret: Option<K>) -> String {
    let sig_desc: String = params
        .iter()
        .map(|k| match k {
            K::Int => 'i',
            K::Float => 'f',
            K::Bool => 'b',
            K::Word => 'w',
        })
        .collect();
    let ret_desc = match ret {
        Some(K::Int) => 'i',
        Some(K::Float) => 'f',
        Some(K::Bool) => 'b',
        Some(K::Word) => 'w',
        None => 'v',
    };
    format!("n{id}_{sig_desc}_{ret_desc}")
}

// ---------- body translator ----------

struct WFx<'a> {
    body: &'a IrBody,
    ana: &'a AnaI,
    sigs: &'a [Option<Sig>],
    fb: FunctionBody,
    /// mimas BlockId -> waffle Block
    wb: Vec<WBlock>,
    /// current emission point (moves on mid-inst splits)
    w: WBlock,
    /// phi insts per block, in stream order
    phis: Vec<Vec<InstId>>,
    /// Phi inst -> its blockparam value
    phi_val: HashMap<u32, Value>,
    /// live locals, sorted — blockparam order after phis
    live: Vec<u32>,
    /// block -> blockparam values for `live`
    local_bp: Vec<Vec<Value>>,
    /// inst -> (SSA value, stored class)
    iv: HashMap<u32, (Value, K)>,
    /// local -> current SSA value (typed at its class)
    cur: HashMap<u32, Value>,
    funcs: &'a HashMap<usize, Func>,
    natives: &'a HashMap<(u32, Vec<K>, Option<K>), Func>,
    helpers: &'a HashMap<&'static str, Func>,
    tramp: &'a HashMap<usize, u32>,
    tramp_sig: Signature,
    is_entry: bool,
    entry_locals: &'a HashSet<u32>,
    entry_base: u32,
    ctx: &'a Statics,
    fuel_g: Option<Global>,
    pause_g: Option<Global>,
    cov_base: u32,
    seq_i: usize,
    sink: &'a crate::CovSink,
    math: &'a crate::MathNatives,
    cov: Option<&'a CovCtx>,
}

impl<'a> WFx<'a> {
    // ----- op sugar: `emit*` appends at self.w, `op_at` at an explicit block -----

    fn emit(&mut self, op: WOp, args: &[Value], tys: &[WTy]) -> Value {
        let w = self.w;
        self.fb.add_op(w, op, args, tys)
    }
    fn op_at(&mut self, w: WBlock, op: WOp, args: &[Value], tys: &[WTy]) -> Value {
        self.fb.add_op(w, op, args, tys)
    }
    fn k64(&mut self, v: i64) -> Value {
        self.emit(WOp::I64Const { value: v as u64 }, &[], &[WTy::I64])
    }
    fn k32(&mut self, v: i32) -> Value {
        self.emit(WOp::I32Const { value: v as u32 }, &[], &[WTy::I32])
    }
    fn kf(&mut self, f: f64) -> Value {
        self.emit(WOp::F64Const { value: f.to_bits() }, &[], &[WTy::F64])
    }
    fn load64(&mut self, addr: Value, off: u32) -> Value {
        self.emit(
            WOp::I64Load {
                memory: marg(off, 3),
            },
            &[addr],
            &[WTy::I64],
        )
    }
    fn load32(&mut self, addr: Value, off: u32) -> Value {
        self.emit(
            WOp::I32Load {
                memory: marg(off, 2),
            },
            &[addr],
            &[WTy::I32],
        )
    }
    fn store64(&mut self, addr: Value, v: Value, off: u32) {
        self.emit(
            WOp::I64Store {
                memory: marg(off, 3),
            },
            &[addr, v],
            &[],
        );
    }
    fn store32(&mut self, addr: Value, v: Value, off: u32) {
        self.emit(
            WOp::I32Store {
                memory: marg(off, 2),
            },
            &[addr, v],
            &[],
        );
    }
    fn store8(&mut self, addr: Value, v: Value) {
        self.emit(WOp::I32Store8 { memory: marg(0, 0) }, &[addr, v], &[]);
    }
    fn wrap(&mut self, v: Value) -> Value {
        self.emit(WOp::I32WrapI64, &[v], &[WTy::I32])
    }
    fn extend(&mut self, v: Value) -> Value {
        self.emit(WOp::I64ExtendI32U, &[v], &[WTy::I64])
    }
    fn call_h(&mut self, h: &str, args: &[Value], rtys: &[WTy]) -> Value {
        let f = self.helpers[h];
        self.emit(WOp::Call { function_index: f }, args, rtys)
    }

    /// split current block on i32 `cond` -> (then, else); caller fills arms.
    fn branch(&mut self, cond: Value) -> (WBlock, WBlock) {
        let t = self.fb.add_block();
        let f = self.fb.add_block();
        self.fb.set_terminator(
            self.w,
            Terminator::CondBr {
                cond,
                if_true: BlockTarget {
                    block: t,
                    args: vec![],
                },
                if_false: BlockTarget {
                    block: f,
                    args: vec![],
                },
            },
        );
        (t, f)
    }

    /// merge open arm blocks into a new block with `tys` blockparams;
    /// returns the blockparam values; leaves self.w at the merge.
    fn join(&mut self, arms: &[(WBlock, Vec<Value>)], tys: &[WTy]) -> Vec<Value> {
        let m = self.fb.add_block();
        let mut outs = Vec::with_capacity(tys.len());
        for &ty in tys {
            outs.push(self.fb.add_blockparam(m, ty));
        }
        for (a, vs) in arms {
            self.fb.set_terminator(
                *a,
                Terminator::Br {
                    target: BlockTarget {
                        block: m,
                        args: vs.clone(),
                    },
                },
            );
        }
        self.w = m;
        outs
    }

    /// `if (cond) unreachable`. Continues in a new block.
    fn trap_if(&mut self, cond: Value) {
        let (t, cont) = self.branch(cond);
        self.fb.set_terminator(t, Terminator::Unreachable);
        self.w = cont;
    }

    // ----- value access: get/emit_const/coerce/store_dst -----

    fn stored_k(&self, iid: InstId) -> K {
        if let Inst::GetLocal(l) = &self.body.instructions[iid] {
            self.ana
                .lclass
                .get(&(l.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        } else {
            self.ana
                .class
                .get(&(iid.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        }
    }

    fn lk(&self, l: u32) -> K {
        self.ana.lclass.get(&l).copied().unwrap_or(K::Word)
    }

    fn coerce_val(&mut self, v: Value, k: K, want: K) -> Result<Value, Bail> {
        Ok(match (k, want) {
            _ if k == want => v,
            (K::Int, K::Float) => self.emit(WOp::F64ConvertI64S, &[v], &[WTy::F64]),
            (K::Float, K::Int) => self.emit(WOp::I64TruncF64S, &[v], &[WTy::I64]),
            (K::Bool, K::Int) => self.emit(WOp::I64ExtendI32U, &[v], &[WTy::I64]),
            (K::Bool, K::Float) => self.emit(WOp::F64ConvertI32S, &[v], &[WTy::F64]),
            (K::Int, K::Bool) => {
                let z = self.k64(0);
                self.emit(WOp::I64Ne, &[v, z], &[WTy::I32])
            }
            (K::Float, K::Bool) => {
                let z = self.kf(0.0);
                self.emit(WOp::F64Ne, &[v, z], &[WTy::I32])
            }
            (K::Int, K::Word) | (K::Word, K::Int) => v,
            (K::Float, K::Word) => self.emit(WOp::I64ReinterpretF64, &[v], &[WTy::I64]),
            (K::Word, K::Float) => self.emit(WOp::F64ReinterpretI64, &[v], &[WTy::F64]),
            (K::Bool, K::Word) => self.emit(WOp::I64ExtendI32U, &[v], &[WTy::I64]),
            (K::Word, K::Bool) => self.emit(WOp::I32WrapI64, &[v], &[WTy::I32]),
            _ => bail!("coerce {k:?}->{want:?}"),
        })
    }

    /// Push `iid`'s value at `want` — `emit_const` inline, `GetLocal` alias,
    /// inst results from `iv`. Entry-shared locals load the memory slot.
    fn get(&mut self, iid: InstId, want: K) -> Result<Value, Bail> {
        match &self.body.instructions[iid] {
            Inst::Constant(c) => self.emit_const(c, want),
            Inst::GetLocal(l) => {
                let lx = l.index() as u32;
                if self.is_entry && self.entry_locals.contains(&lx) {
                    let a = self.k32((self.entry_base + lx * 8) as i32);
                    let v = self.load64(a, 0);
                    self.coerce_val(v, K::Word, want)
                } else {
                    let Some(&v) = self.cur.get(&lx) else {
                        bail!("read of dead local {lx}")
                    };
                    let lk = self.lk(lx);
                    self.coerce_val(v, lk, want)
                }
            }
            _ => {
                let ix = iid.index() as u32;
                let (v, k) = self.iv.get(&ix).copied().ok_or_else(|| {
                    format!(
                        "inst {ix} not materialized: {:?}",
                        self.body.instructions[iid]
                    )
                })?;
                self.coerce_val(v, k, want)
            }
        }
    }

    fn emit_const(&mut self, c: &Constant, want: K) -> Result<Value, Bail> {
        Ok(match (c, want) {
            (_, K::Word) => match c {
                Constant::Int(v) => self.k64(*v),
                Constant::Float(v) => self.k64(v.to_bits() as i64),
                Constant::Bool(v) => self.k64(*v as i64),
                Constant::Str(s) => {
                    let a = self
                        .ctx
                        .str_objs
                        .get(&(s.index() as u32))
                        .copied()
                        .ok_or("str const not laid out")?;
                    self.k64(a as i64)
                }
                Constant::Null => self.k64(0),
                Constant::Array(es) => {
                    let a = bake_const_array_obj(self.ctx, es)?;
                    self.k64(a as i64)
                }
            },
            (Constant::Int(v), K::Int) => self.k64(*v),
            (Constant::Int(v), K::Float) => self.kf(*v as f64),
            (Constant::Int(v), K::Bool) => self.k32((*v != 0) as i32),
            (Constant::Float(v), K::Float) => self.kf(*v),
            (Constant::Float(v), K::Int) => self.k64(*v as i64),
            (Constant::Float(v), K::Bool) => self.k32((*v != 0.0) as i32),
            (Constant::Bool(v), K::Bool) => self.k32(*v as i32),
            (Constant::Bool(v), K::Int) => self.k64(*v as i64),
            (Constant::Bool(v), K::Float) => self.kf(*v as i32 as f64),
            _ => bail!("const {c:?} can't serve {want:?}"),
        })
    }

    /// Record `iid`'s produced value (class `produced`) coerced to its slot
    /// class; a dead result is simply not recorded (SSA has no drop).
    fn store_dst(&mut self, iid: InstId, v: Value, produced: K) -> Result<(), Bail> {
        let ix = iid.index() as u32;
        let dk = match self.ana.class.get(&ix) {
            Some(&k) => k,
            None if self.ana.wused.contains(&ix) => K::Word,
            None => return Ok(()),
        };
        let v = self.coerce_val(v, produced, dk)?;
        self.iv.insert(ix, (v, dk));
        Ok(())
    }

    /// only `AccessKind::Direct` lowers to bare loads
    fn direct(&self, kind: &compile::AccessKind) -> Result<(), Bail> {
        match kind {
            compile::AccessKind::Direct => Ok(()),
            _ => bail!("optional access needs Null values"),
        }
    }

    /// bump-allocate `[sz]` bytes — pops the i32 size arg.
    fn alloc(&mut self, sz: Value) -> Value {
        self.call_h("__alloc", &[sz], &[WTy::I32])
    }

    // ----- entry-shared locals (memory region at entry_base) -----

    fn eaddr(&mut self, l: u32) -> Value {
        self.k32((self.entry_base + l * 8) as i32)
    }

    // ----- fuel + coverage tick, once per stream inst -----
    fn tick(&mut self, i: usize) {
        if let Some(g) = self.fuel_g {
            let cur = self.emit(WOp::GlobalGet { global_index: g }, &[], &[WTy::I64]);
            let one = self.k64(1);
            let n = self.emit(WOp::I64Sub, &[cur, one], &[WTy::I64]);
            self.emit(WOp::GlobalSet { global_index: g }, &[n], &[]);
            let cur = self.emit(WOp::GlobalGet { global_index: g }, &[], &[WTy::I64]);
            let z = self.k64(0);
            let c = self.emit(WOp::I64LtS, &[cur, z], &[WTy::I32]);
            self.trap_if(c);
        }
        if self.cov_base != NO_COV {
            let a = self.k32((self.cov_base + i as u32) as i32);
            let one = self.k32(1);
            self.store8(a, one);
        }
    }

    // ----- edge args: phis (stream order) then live locals -----

    fn targs(&mut self, pred: usize, target: usize) -> Result<Vec<Value>, Bail> {
        let mut args = Vec::new();
        for &piid in &self.phis[target].clone() {
            let Inst::Phi(branches) = &self.body.instructions[piid] else {
                unreachable!()
            };
            let pk = self.stored_k(piid);
            let v = match branches.iter().find(|(b, _)| b.index() == pred) {
                Some((_, i)) => self
                    .get(*i, pk)
                    .map_err(|e| format!("{e} (phi {piid:?} arg, b{pred}->b{target})"))?,
                None => self.zero(wty(pk)),
            };
            args.push(v);
        }
        for &l in &self.live.clone() {
            match self.cur.get(&l) {
                Some(&v) => args.push(v),
                None => args.push(self.zero(wty(self.lk(l)))),
            }
        }
        Ok(args)
    }

    fn zero(&mut self, ty: WTy) -> Value {
        match ty {
            WTy::I32 => self.k32(0),
            WTy::F64 => self.kf(0.0),
            _ => self.k64(0),
        }
    }

    fn tgt(&mut self, pred: usize, target: usize) -> Result<BlockTarget, Bail> {
        Ok(BlockTarget {
            block: self.wb[target],
            args: self.targs(pred, target)?,
        })
    }

    /// Pause check before a backward edge.
    fn pause_check(&mut self) {
        if let Some(g) = self.pause_g {
            let c = self.emit(WOp::GlobalGet { global_index: g }, &[], &[WTy::I32]);
            self.trap_if(c);
        }
    }

    fn imm(&self, iid: InstId, kind: OperandKind) -> Option<i64> {
        match (kind, &self.body.instructions[iid]) {
            (OperandKind::Int, Inst::Constant(Constant::Int(v))) => Some(*v),
            (OperandKind::Float, Inst::Constant(Constant::Float(f))) => Some(f.to_bits() as i64),
            _ => None,
        }
    }

    // ================== inst arms: verbatim emit_one transcriptions ==================

    fn emit_inst(&mut self, iid: InstId) -> Result<(), Bail> {
        let inst = &self.body.instructions[iid];
        let ix = iid.index() as u32;
        let live = self.ana.class.contains_key(&ix) || self.ana.wused.contains(&ix);
        match inst {
            Inst::Constant(_) | Inst::GetLocal(_) | Inst::Phi(_) => return Ok(()),
            Inst::NewArray
            | Inst::NewDict
            | Inst::GetField { .. }
            | Inst::Len(_)
            | Inst::ToFloat(_)
            | Inst::Sqrt(_)
            | Inst::In(..)
            | Inst::Format(_)
            | Inst::MakeClosure { .. }
            | Inst::NewInstance { .. }
            | Inst::IsInstance { .. }
            | Inst::IsRaised(_)
            | Inst::GetEntry(_)
            | Inst::RefBody(_)
            | Inst::BinOp { .. }
            | Inst::UnaryOp { .. }
                if !live =>
            {
                return Ok(());
            }
            _ => {}
        }
        match inst.clone() {
            Inst::SetLocal(l, v) => {
                let lx = l.index() as u32;
                if self.is_entry && self.entry_locals.contains(&lx) {
                    if self.cur.contains_key(&lx) {
                        let lk = self.lk(lx);
                        let val = self.get(v, lk)?;
                        self.cur.insert(lx, val);
                    }
                    let a = self.eaddr(lx);
                    let vv = self.get(v, K::Word)?;
                    self.store64(a, vv, 0);
                } else if self.cur.contains_key(&lx) {
                    let lk = self.lk(lx);
                    let val = self.get(v, lk)?;
                    self.cur.insert(lx, val);
                }
            }
            Inst::BinOp {
                left,
                op,
                right,
                kind,
            } => self.emit_bin(iid, left, op, right, kind)?,
            Inst::UnaryOp { op, right } => self.emit_unary(iid, op, right)?,
            Inst::NewArray => {
                let sz = self.k32(16);
                let hp = self.alloc(sz);
                let tag = self.k32(TAG_ARRAY as i32);
                self.store32(hp, tag, 0);
                let z = self.k32(0);
                self.store32(hp, z, 4);
                let z = self.k64(0);
                self.store64(hp, z, 8);
                let v = self.extend(hp);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::Push { array, value } => {
                let aw = self.get(array, K::Word)?;
                let hp = self.wrap(aw);
                let tb = self.load32(hp, 8); // len
                let cap = self.load32(hp, 12);
                let full = self.emit(WOp::I32Eq, &[tb, cap], &[WTy::I32]);
                let (t, f) = self.branch(full);
                {
                    self.w = t;
                    // newcap = cap == 0 ? 4 : cap*2
                    let cz = self.emit(WOp::I32Eqz, &[cap], &[WTy::I32]);
                    let four = self.k32(4);
                    let two = self.k32(2);
                    let cap2 = self.emit(WOp::I32Mul, &[cap, two], &[WTy::I32]);
                    let newcap = self.emit(WOp::Select, &[four, cap2, cz], &[WTy::I32]);
                    let eight = self.k32(8);
                    let szb = self.emit(WOp::I32Mul, &[newcap, eight], &[WTy::I32]);
                    let sz = self.alloc(szb);
                    let data = self.load32(hp, 4);
                    let nbytes = self.emit(WOp::I32Mul, &[tb, eight], &[WTy::I32]);
                    self.emit(
                        WOp::MemoryCopy {
                            dst_mem: Memory::new(0),
                            src_mem: Memory::new(0),
                        },
                        &[sz, data, nbytes],
                        &[],
                    );
                    self.store32(hp, sz, 4);
                    self.store32(hp, newcap, 12);
                }
                self.w = f;
                self.join(&[(t, vec![]), (f, vec![])], &[]);
                // data[len] = value; len += 1
                let data = self.load32(hp, 4);
                let eight = self.k32(8);
                let off = self.emit(WOp::I32Mul, &[tb, eight], &[WTy::I32]);
                let a = self.emit(WOp::I32Add, &[data, off], &[WTy::I32]);
                let vv = self.get(value, K::Word)?;
                self.store64(a, vv, 0);
                let one = self.k32(1);
                let nl = self.emit(WOp::I32Add, &[tb, one], &[WTy::I32]);
                self.store32(hp, nl, 8);
            }
            Inst::GetIndex { set, index, kind } => {
                self.direct(&kind)?;
                let sw = self.get(set, K::Word)?;
                let hp = self.wrap(sw);
                let tag = self.load32(hp, 0);
                let st = self.k32(TAG_STR as i32);
                let is_str = self.emit(WOp::I32Eq, &[tag, st], &[WTy::I32]);
                let (sw_blk, rest) = self.branch(is_str);
                self.w = sw_blk;
                let iv_w = self.get(index, K::Word)?;
                let ix32 = self.wrap(iv_w);
                let ch = self.call_h("__str_charat", &[hp, ix32], &[WTy::I32]);
                let sval = self.extend(ch);
                let s_end = self.w;
                self.w = rest;
                let dt = self.k32(TAG_DICT as i32);
                let is_dict = self.emit(WOp::I32Eq, &[tag, dt], &[WTy::I32]);
                let (dw, aw) = self.branch(is_dict);
                self.w = dw;
                let dval = if self.stored_k(index) == K::Int {
                    let iv_w = self.get(index, K::Word)?;
                    let ix32 = self.wrap(iv_w);
                    let e = self.call_h("__dict_entry", &[hp, ix32], &[WTy::I32]);
                    self.extend(e)
                } else {
                    let kv = self.get(index, K::Word)?;
                    let sz = self.call_h("__dict_find", &[hp, kv], &[WTy::I32]);
                    let miss = self.emit(WOp::I32Eqz, &[sz], &[WTy::I32]);
                    let val = self.load64(sz, 8);
                    let z = self.k64(0);
                    self.emit(WOp::Select, &[z, val, miss], &[WTy::I64])
                };
                let d_end = self.w;
                self.w = aw;
                let iv_w = self.get(index, K::Word)?;
                let tb = self.wrap(iv_w);
                let len = self.load32(hp, 8);
                let oob = self.emit(WOp::I32GeU, &[tb, len], &[WTy::I32]);
                self.trap_if(oob);
                let data = self.load32(hp, 4);
                let eight = self.k32(8);
                let off = self.emit(WOp::I32Mul, &[tb, eight], &[WTy::I32]);
                let a = self.emit(WOp::I32Add, &[data, off], &[WTy::I32]);
                let aval = self.load64(a, 0);
                let a_end = self.w;
                let outs = self.join(
                    &[
                        (s_end, vec![sval]),
                        (d_end, vec![dval]),
                        (a_end, vec![aval]),
                    ],
                    &[WTy::I64],
                );
                self.store_dst(iid, outs[0], K::Word)?;
            }
            Inst::SetIndex { set, index, value } => {
                let sw = self.get(set, K::Word)?;
                let hp = self.wrap(sw);
                let tag = self.load32(hp, 0);
                let dt = self.k32(TAG_DICT as i32);
                let is_dict = self.emit(WOp::I32Eq, &[tag, dt], &[WTy::I32]);
                let (dw, aw) = self.branch(is_dict);
                self.w = dw;
                let kv = self.get(index, K::Word)?;
                let vv = self.get(value, K::Word)?;
                self.call_h("__dict_set", &[hp, kv, vv], &[]);
                let d_end = self.w;
                self.w = aw;
                let iv_w = self.get(index, K::Word)?;
                let tb = self.wrap(iv_w);
                let len = self.load32(hp, 8);
                let oob = self.emit(WOp::I32GeU, &[tb, len], &[WTy::I32]);
                self.trap_if(oob);
                let data = self.load32(hp, 4);
                let eight = self.k32(8);
                let off = self.emit(WOp::I32Mul, &[tb, eight], &[WTy::I32]);
                let a = self.emit(WOp::I32Add, &[data, off], &[WTy::I32]);
                let vv = self.get(value, K::Word)?;
                self.store64(a, vv, 0);
                let a_end = self.w;
                self.join(&[(d_end, vec![]), (a_end, vec![])], &[]);
            }
            Inst::Len(src) => {
                let sw = self.get(src, K::Word)?;
                let hp = self.wrap(sw);
                let tag = self.load32(hp, 0);
                let dt = self.k32(TAG_DICT as i32);
                let is_dict = self.emit(WOp::I32Eq, &[tag, dt], &[WTy::I32]);
                let (dw, aw) = self.branch(is_dict);
                self.w = dw;
                let dv = self.load32(hp, 4);
                let d_end = self.w;
                self.w = aw;
                let av = self.load32(hp, 8);
                let a_end = self.w;
                let outs = self.join(&[(d_end, vec![dv]), (a_end, vec![av])], &[WTy::I32]);
                let v = self.extend(outs[0]);
                self.store_dst(iid, v, K::Int)?;
            }
            Inst::NewInstance { adt, fields } => {
                let n = fields.len() as i32;
                let sz = self.k32(8 + n * 8);
                let hp = self.alloc(sz);
                let tag = self.k32(TAG_INSTANCE as i32);
                self.store32(hp, tag, 0);
                let adt_i = self.k32(adt.index() as i32);
                self.store32(hp, adt_i, 4);
                for (fi, fld) in fields.iter().enumerate() {
                    let fv = self.get(*fld, K::Word)?;
                    self.store64(hp, fv, 8 + fi as u32 * 8);
                }
                let v = self.extend(hp);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::GetField { src, slot, kind } => {
                self.direct(&kind)?;
                let sw = self.get(src, K::Word)?;
                let hp = self.wrap(sw);
                let v = self.load64(hp, 8 + slot * 8);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::SetField {
                receiver,
                slot,
                value,
            } => {
                let rw = self.get(receiver, K::Word)?;
                let hp = self.wrap(rw);
                let vv = self.get(value, K::Word)?;
                self.store64(hp, vv, 8 + slot * 8);
            }
            Inst::IsInstance { src, adt } => {
                let sw = self.get(src, K::Word)?;
                let hp = self.wrap(sw);
                let isnull = self.emit(WOp::I32Eqz, &[hp], &[WTy::I32]);
                let (nw, cw) = self.branch(isnull);
                self.w = nw;
                let r0 = self.k32(0);
                let n_end = self.w;
                self.w = cw;
                let tag = self.load32(hp, 0);
                let it = self.k32(TAG_INSTANCE as i32);
                let isinst = self.emit(WOp::I32Eq, &[tag, it], &[WTy::I32]);
                let (iw, ow) = self.branch(isinst);
                self.w = iw;
                let a = self.load32(hp, 4);
                let adt_i = self.k32(adt.index() as i32);
                let res = self.emit(WOp::I32Eq, &[a, adt_i], &[WTy::I32]);
                let i_end = self.w;
                self.w = ow;
                let r1 = self.k32(0);
                let o_end = self.w;
                let inner = self.join(&[(i_end, vec![res]), (o_end, vec![r1])], &[WTy::I32]);
                let c_end = self.w;
                let outs = self.join(&[(n_end, vec![r0]), (c_end, vec![inner[0]])], &[WTy::I32]);
                self.store_dst(iid, outs[0], K::Bool)?;
            }
            Inst::In(needle, haystack, condition) => {
                let nw = self.get(needle, K::Word)?;
                let hw = self.get(haystack, K::Word)?;
                let sz = self.wrap(hw);
                let tag = self.load32(sz, 0);
                let dt = self.k32(TAG_DICT as i32);
                let is_dict = self.emit(WOp::I32Eq, &[tag, dt], &[WTy::I32]);
                let (dw, r1) = self.branch(is_dict);
                self.w = dw;
                let f = self.call_h("__dict_find", &[sz, nw], &[WTy::I32]);
                let z = self.k32(0);
                let rd = self.emit(WOp::I32Ne, &[f, z], &[WTy::I32]);
                let d_end = self.w;
                self.w = r1;
                let st = self.k32(TAG_STR as i32);
                let is_str = self.emit(WOp::I32Eq, &[tag, st], &[WTy::I32]);
                let (sw, aw) = self.branch(is_str);
                self.w = sw;
                let n32 = self.wrap(nw);
                let rs = self.call_h("__str_in", &[n32, sz], &[WTy::I32]);
                let s_end = self.w;
                self.w = aw;
                // array scan: for i in 0..len, key_eq(elem[i], needle)
                let tb = self.load32(sz, 8); // len
                let head = self.fb.add_block();
                let h_i = self.fb.add_blockparam(head, WTy::I32);
                let exit = self.fb.add_block();
                let e_found = self.fb.add_blockparam(exit, WTy::I32);
                let z0 = self.k32(0);
                self.fb.set_terminator(
                    aw,
                    Terminator::Br {
                        target: BlockTarget {
                            block: head,
                            args: vec![z0],
                        },
                    },
                );
                // head: if i >= len -> exit(0) else body
                let ge = self.op_at(head, WOp::I32GeU, &[h_i, tb], &[WTy::I32]);
                let (eb, bb) = (self.fb.add_block(), self.fb.add_block());
                self.fb.set_terminator(
                    head,
                    Terminator::CondBr {
                        cond: ge,
                        if_true: BlockTarget {
                            block: eb,
                            args: vec![],
                        },
                        if_false: BlockTarget {
                            block: bb,
                            args: vec![],
                        },
                    },
                );
                let zc = self.op_at(eb, WOp::I32Const { value: 0 }, &[], &[WTy::I32]);
                self.fb.set_terminator(
                    eb,
                    Terminator::Br {
                        target: BlockTarget {
                            block: exit,
                            args: vec![zc],
                        },
                    },
                );
                // body: elem = data[i]; if key_eq(elem, nw) -> exit(1) else i++
                let data = self.op_at(bb, WOp::I32Load { memory: marg(4, 2) }, &[sz], &[WTy::I32]);
                let three = self.op_at(bb, WOp::I32Const { value: 3 }, &[], &[WTy::I32]);
                let sh = self.op_at(bb, WOp::I32Shl, &[h_i, three], &[WTy::I32]);
                let ea = self.op_at(bb, WOp::I32Add, &[data, sh], &[WTy::I32]);
                let elem = self.op_at(bb, WOp::I64Load { memory: marg(0, 3) }, &[ea], &[WTy::I64]);
                let kf = self.helpers["__key_eq"];
                let eq = self.op_at(
                    bb,
                    WOp::Call { function_index: kf },
                    &[elem, nw],
                    &[WTy::I32],
                );
                let (hb, nb) = (self.fb.add_block(), self.fb.add_block());
                self.fb.set_terminator(
                    bb,
                    Terminator::CondBr {
                        cond: eq,
                        if_true: BlockTarget {
                            block: hb,
                            args: vec![],
                        },
                        if_false: BlockTarget {
                            block: nb,
                            args: vec![],
                        },
                    },
                );
                let one = self.op_at(hb, WOp::I32Const { value: 1 }, &[], &[WTy::I32]);
                self.fb.set_terminator(
                    hb,
                    Terminator::Br {
                        target: BlockTarget {
                            block: exit,
                            args: vec![one],
                        },
                    },
                );
                let one2 = self.op_at(nb, WOp::I32Const { value: 1 }, &[], &[WTy::I32]);
                let i2 = self.op_at(nb, WOp::I32Add, &[h_i, one2], &[WTy::I32]);
                self.fb.set_terminator(
                    nb,
                    Terminator::Br {
                        target: BlockTarget {
                            block: head,
                            args: vec![i2],
                        },
                    },
                );
                self.w = exit;
                let ra = e_found;
                let outs = self.join(
                    &[(d_end, vec![rd]), (s_end, vec![rs]), (exit, vec![ra])],
                    &[WTy::I32],
                );
                let mut r = outs[0];
                if !condition {
                    r = self.emit(WOp::I32Eqz, &[r], &[WTy::I32]);
                }
                self.store_dst(iid, r, K::Bool)?;
            }
            Inst::Unwrap(src) => {
                let v = self.get(src, K::Word)?;
                let z = self.k64(0);
                let bad = self.emit(WOp::I64LeS, &[v, z], &[WTy::I32]);
                self.trap_if(bad);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::UnwrapUnit(src) => {
                let v = self.get(src, K::Word)?;
                let z = self.k64(0);
                let bad = self.emit(WOp::I64LtS, &[v, z], &[WTy::I32]);
                self.trap_if(bad);
                let z = self.k64(0);
                self.store_dst(iid, z, K::Word)?;
            }
            Inst::UnwrapRaised(src) => {
                let v = self.get(src, K::Word)?;
                let m = self.k64(0x7fff_ffff_ffff_ffffu64 as i64);
                let r = self.emit(WOp::I64And, &[v, m], &[WTy::I64]);
                self.store_dst(iid, r, K::Word)?;
            }
            Inst::IsRaised(src) => {
                let v = self.get(src, K::Word)?;
                let z = self.k64(0);
                let r = self.emit(WOp::I64LtS, &[v, z], &[WTy::I32]);
                self.store_dst(iid, r, K::Bool)?;
            }
            Inst::Format(parts) => {
                let mut acc: Option<Value> = None;
                for p in &parts {
                    let pv = match p {
                        FormatPart::Literal(s) => {
                            let a = self
                                .ctx
                                .str_objs
                                .get(&(s.index() as u32))
                                .copied()
                                .ok_or("format literal not laid out")?;
                            self.k32(a as i32)
                        }
                        FormatPart::Value(v) => self.format_part(*v)?,
                    };
                    acc = Some(match acc {
                        None => pv,
                        Some(a) => self.call_h("__str_cat", &[a, pv], &[WTy::I32]),
                    });
                }
                let v = match acc {
                    Some(v) => v,
                    None => self.k32(self.ctx.obj_obj as i32),
                };
                let v = self.extend(v);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::NewDict => {
                let sz = self.k32(24);
                let hp = self.alloc(sz);
                let tag = self.k32(TAG_DICT as i32);
                self.store32(hp, tag, 0);
                let z = self.k64(0);
                self.store64(hp, z, 4);
                let z = self.k64(0);
                self.store64(hp, z, 12);
                let z = self.k32(0);
                self.store32(hp, z, 20);
                let v = self.extend(hp);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::Insert { dict, key, value } => {
                let dw = self.get(dict, K::Word)?;
                let hp = self.wrap(dw);
                let a = self
                    .ctx
                    .str_objs
                    .get(&(key.index() as u32))
                    .copied()
                    .ok_or("insert key not laid out")?;
                let k = self.k64(a as i64);
                let vv = self.get(value, K::Word)?;
                self.call_h("__dict_set", &[hp, k, vv], &[]);
            }
            Inst::RefBody(t) => {
                let b = t.index();
                let ti = *self.tramp.get(&b).ok_or("no trampoline for body")?;
                let sz = self.k32(16);
                let hp = self.alloc(sz);
                let tag = self.k32(TAG_CLOSURE as i32);
                self.store32(hp, tag, 0);
                let ti_v = self.k32(ti as i32);
                self.store32(hp, ti_v, 4);
                let z = self.k64(0);
                self.store64(hp, z, 8);
                let v = self.extend(hp);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::MakeClosure { body: t, captures } => {
                let b = t.index();
                let ti = *self.tramp.get(&b).ok_or("no trampoline for body")?;
                let sz = self.k32(16 + 8 * captures.len() as i32);
                let hp = self.alloc(sz);
                let tag = self.k32(TAG_CLOSURE as i32);
                self.store32(hp, tag, 0);
                let ti_v = self.k32(ti as i32);
                self.store32(hp, ti_v, 4);
                let nc = self.k64(captures.len() as i64);
                self.store64(hp, nc, 8);
                for (i, c) in captures.iter().enumerate() {
                    let cv = self.get(*c, K::Word)?;
                    self.store64(hp, cv, 16 + i as u32 * 8);
                }
                let v = self.extend(hp);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::GetEntry(l) => {
                let a = self.eaddr(l.index() as u32);
                let v = self.load64(a, 0);
                self.store_dst(iid, v, K::Word)?;
            }
            Inst::SetEntry(l, v) => {
                let a = self.eaddr(l.index() as u32);
                let vv = self.get(v, K::Word)?;
                self.store64(a, vv, 0);
            }
            Inst::ToFloat(v) => {
                let x = self.get(v, K::Int)?;
                let f = self.emit(WOp::F64ConvertI64S, &[x], &[WTy::F64]);
                self.store_dst(iid, f, K::Float)?;
            }
            Inst::Sqrt(v) => {
                let x = self.get(v, K::Float)?;
                let f = self.emit(WOp::F64Sqrt, &[x], &[WTy::F64]);
                self.store_dst(iid, f, K::Float)?;
            }
            Inst::CallDirect { body: b, args } => self.emit_call(iid, b.index(), &args)?,
            Inst::Call { callee, args } => {
                let direct = self
                    .ana
                    .callee
                    .get(&ix)
                    .map(|&b| {
                        self.sigs[b]
                            .as_ref()
                            .is_some_and(|s| s.params.len() == args.len())
                    })
                    .unwrap_or(false);
                if direct {
                    let b = self.ana.callee[&ix];
                    self.emit_call(iid, b, &args)?;
                } else {
                    let cw = self.get(callee, K::Word)?;
                    let hp = self.wrap(cw);
                    // null/tag gate
                    let nz = self.emit(WOp::I32Eqz, &[hp], &[WTy::I32]);
                    self.trap_if(nz);
                    let tag = self.load32(hp, 0);
                    let ct = self.k32(TAG_CLOSURE as i32);
                    let ne = self.emit(WOp::I32Ne, &[tag, ct], &[WTy::I32]);
                    self.trap_if(ne);
                    // marshal args to i64 words in call scratch
                    for (i, a) in args.iter().enumerate() {
                        let addr = self.k32((self.ctx.call_scratch + i as u32 * 8) as i32);
                        let av = self.get(*a, K::Word)?;
                        self.store64(addr, av, 0);
                    }
                    let sp = self.k32(self.ctx.call_scratch as i32);
                    let na = self.k32(args.len() as i32);
                    let tix = self.load32(hp, 4);
                    let r = self.emit(
                        WOp::CallIndirect {
                            sig_index: self.tramp_sig,
                            table_index: Table::new(0),
                        },
                        &[hp, sp, na, tix],
                        &[WTy::I64],
                    );
                    self.store_dst(iid, r, K::Word)?;
                }
            }
            Inst::CallNative { id, args } => self.emit_native(iid, id.index() as u32, &args)?,
            Inst::Return(_) | Inst::Raise(_) | Inst::Panic => {
                // terminators — consumed by the block loop
            }
            Inst::Jump { .. }
            | Inst::JumpIfFalse { .. }
            | Inst::ForNext { .. }
            | Inst::Switch { .. } => {
                // terminators — consumed by the block loop
            }
            Inst::Phi(_) | Inst::Constant(_) | Inst::GetLocal(_) => {}
        }
        Ok(())
    }

    fn emit_unary(&mut self, iid: InstId, op: UnaryOp, right: InstId) -> Result<(), Bail> {
        let ix = iid.index() as u32;
        match op {
            UnaryOp::Negative | UnaryOp::Positive => {
                let dk = self
                    .ana
                    .class
                    .get(&ix)
                    .or_else(|| self.ana.class.get(&(right.index() as u32)))
                    .copied()
                    .unwrap_or(K::Float);
                match (op, dk) {
                    (UnaryOp::Negative, K::Int) => {
                        let r = self.get(right, K::Int)?;
                        let z = self.k64(0);
                        let res = self.emit(WOp::I64Sub, &[z, r], &[WTy::I64]);
                        let m = self.k64(i64::MIN);
                        let r2 = self.get(right, K::Int)?;
                        let is_min = self.emit(WOp::I64Eq, &[r2, m], &[WTy::I32]);
                        self.trap_if(is_min);
                        self.store_dst(iid, res, K::Int)?;
                    }
                    (UnaryOp::Negative, K::Float) => {
                        let r = self.get(right, K::Float)?;
                        let n = self.emit(WOp::F64Neg, &[r], &[WTy::F64]);
                        self.store_dst(iid, n, K::Float)?;
                    }
                    (UnaryOp::Positive, K::Int) => {
                        let x = self.get(right, K::Int)?;
                        let m = self.k64(i64::MIN);
                        let is_min = self.emit(WOp::I64Eq, &[x, m], &[WTy::I32]);
                        self.trap_if(is_min);
                        // abs = x < 0 ? -x : x  (wasm select picks v1 when cond)
                        let z = self.k64(0);
                        let neg = self.emit(WOp::I64Sub, &[z, x], &[WTy::I64]);
                        let z = self.k64(0);
                        let negc = self.emit(WOp::I64LtS, &[x, z], &[WTy::I32]);
                        let abs = self.emit(WOp::Select, &[neg, x, negc], &[WTy::I64]);
                        self.store_dst(iid, abs, K::Int)?;
                    }
                    (UnaryOp::Positive, K::Float) => {
                        let r = self.get(right, K::Float)?;
                        let n = self.emit(WOp::F64Abs, &[r], &[WTy::F64]);
                        self.store_dst(iid, n, K::Float)?;
                    }
                    _ => bail!("unary {op:?} on {dk:?}"),
                }
            }
            UnaryOp::Not => {
                let r = self.get(right, K::Bool)?;
                let n = self.emit(WOp::I32Eqz, &[r], &[WTy::I32]);
                self.store_dst(iid, n, K::Bool)?;
            }
            UnaryOp::BitwiseNot => {
                let r = self.get(right, K::Int)?;
                let m = self.k64(-1);
                let n = self.emit(WOp::I64Xor, &[r, m], &[WTy::I64]);
                self.store_dst(iid, n, K::Int)?;
            }
        }
        Ok(())
    }

    fn format_part(&mut self, v: InstId) -> Result<Value, Bail> {
        Ok(match self.stored_k(v) {
            K::Int => {
                let x = self.get(v, K::Int)?;
                self.call_h("__i64_str", &[x], &[WTy::I32])
            }
            K::Float => {
                let x = self.get(v, K::Float)?;
                self.call_h("__f64_str", &[x], &[WTy::I32])
            }
            K::Bool => {
                let x = self.get(v, K::Bool)?;
                let t = self.k32(self.ctx.true_obj as i32);
                let f = self.k32(self.ctx.false_obj as i32);
                self.emit(WOp::Select, &[t, f, x], &[WTy::I32])
            }
            K::Word => {
                let x = self.get(v, K::Word)?;
                self.call_h("__str_or_obj", &[x], &[WTy::I32])
            }
        })
    }

    fn emit_call(&mut self, dst: InstId, b: usize, args: &[InstId]) -> Result<(), Bail> {
        let sig = self.sigs[b]
            .as_ref()
            .ok_or_else(|| format!("callee body {b} not emitted"))?;
        let ix = dst.index() as u32;
        if (self.ana.class.contains_key(&ix) || self.ana.wused.contains(&ix)) && sig.ret.is_none() {
            bail!("callee returns void but dst is read");
        }
        if sig.params.len() != args.len() {
            bail!("arity mismatch calling body {b}");
        }
        let mut argv = Vec::with_capacity(args.len());
        for (a, pk) in args.iter().zip(&sig.params) {
            argv.push(self.get(*a, *pk)?);
        }
        let f = *self
            .funcs
            .get(&b)
            .ok_or_else(|| format!("callee body {b} has no func slot"))?;
        if let Some(rk) = sig.ret {
            let r = self.emit(WOp::Call { function_index: f }, &argv, &[wty(rk)]);
            self.store_dst(dst, r, rk)?;
        } else {
            self.emit(WOp::Call { function_index: f }, &argv, &[]);
        }
        Ok(())
    }

    fn emit_native(&mut self, iid: InstId, ni: u32, args: &[InstId]) -> Result<(), Bail> {
        let ix = iid.index() as u32;
        let dk = self
            .ana
            .class
            .get(&ix)
            .copied()
            .or_else(|| self.ana.wused.contains(&ix).then_some(K::Word));
        // pure float natives inline
        if let Some(&mop) = self.math.get(&ni) {
            use crate::MathOp::*;
            if dk.is_some() && (mop == Identity || args.len() <= 2) {
                let k = dk.unwrap();
                let (v, pk) = match mop {
                    Identity => (self.get(args[0], k)?, k),
                    _ => {
                        let a = self.get(args[0], K::Float)?;
                        let mut r = a;
                        if args.len() > 1 {
                            let b = self.get(args[1], K::Float)?;
                            r = self.emit(
                                match mop {
                                    Min => WOp::F64Min,
                                    Max => WOp::F64Max,
                                    Abs | Floor | Identity => unreachable!(),
                                },
                                &[a, b],
                                &[WTy::F64],
                            );
                        } else {
                            r = self.emit(
                                match mop {
                                    Abs => WOp::F64Abs,
                                    Floor => WOp::F64Floor,
                                    _ => unreachable!(),
                                },
                                &[r],
                                &[WTy::F64],
                            );
                        }
                        (r, K::Float)
                    }
                };
                self.store_dst(iid, v, pk)?;
                return Ok(());
            }
        }
        // cov sink natives record into linear memory
        if let Some(&kind) = self.sink.get(&ni) {
            let Some(cov) = self.cov else {
                bail!("sink native {ni} without cov layout")
            };
            use crate::SinkKind::*;
            let w32 = |me: &mut Self, a: InstId| -> Result<Value, Bail> {
                let v = me.get(a, K::Word)?;
                Ok(me.wrap(v))
            };
            match kind {
                Point | PointPass => {
                    let base = self.k32(cov.ptmap_base as i32);
                    let p = w32(self, args[0])?;
                    let a = self.emit(WOp::I32Add, &[base, p], &[WTy::I32]);
                    let one = self.k32(1);
                    self.store8(a, one);
                }
                Begin => {
                    let d = w32(self, args[0])?;
                    self.emit(
                        WOp::Call {
                            function_index: Func::new(cov.h[1] as usize),
                        },
                        &[d],
                        &[],
                    );
                }
                Leaf => {
                    let a = args[0];
                    let cls = self.stored_k(a);
                    let num = matches!(cls, K::Int | K::Float);
                    let f = if cls == K::Float {
                        self.get(a, K::Float)?
                    } else if num {
                        let v = self.get(a, K::Word)?;
                        self.emit(WOp::F64ConvertI64S, &[v], &[WTy::F64])
                    } else {
                        self.kf(0.0)
                    };
                    let nf = self.k32(num as i32);
                    self.emit(
                        WOp::Call {
                            function_index: Func::new(cov.h[0] as usize),
                        },
                        &[f, nf],
                        &[],
                    );
                }
                Cond => {
                    let mut argv = Vec::new();
                    for &a in args.iter().take(3) {
                        argv.push(w32(self, a)?);
                    }
                    self.emit(
                        WOp::Call {
                            function_index: Func::new(cov.h[2] as usize),
                        },
                        &argv,
                        &[],
                    );
                }
                Cmp => {
                    let mut argv = Vec::new();
                    for &a in args.iter().take(4) {
                        argv.push(w32(self, a)?);
                    }
                    self.emit(
                        WOp::Call {
                            function_index: Func::new(cov.h[3] as usize),
                        },
                        &argv,
                        &[],
                    );
                }
                Dec => {
                    let ni32 = self.k32(ni as i32);
                    let mut argv = vec![ni32];
                    for &a in args.iter().take(2) {
                        argv.push(w32(self, a)?);
                    }
                    self.emit(
                        WOp::Call {
                            function_index: Func::new(cov.h[4] as usize),
                        },
                        &argv,
                        &[],
                    );
                }
                Hit | Passthru => {
                    bail!("generic Hit/Passthru sink kinds would corrupt the 16-byte dec ring")
                }
            }
            match kind {
                Point => {
                    let z = self.k64(0);
                    self.store_dst(iid, z, K::Word)?;
                }
                PointPass | Leaf | Cond | Cmp | Dec => {
                    let last = *args.last().unwrap();
                    let k = dk.unwrap_or(K::Int);
                    let v = self.get(last, k)?;
                    self.store_dst(iid, v, k)?;
                }
                Begin => {
                    let k = dk.unwrap_or(K::Int);
                    let v = match k {
                        K::Float => self.kf(1.0),
                        K::Bool => self.k32(1),
                        _ => self.k64(1),
                    };
                    self.store_dst(iid, v, k)?;
                }
                Hit | Passthru => unreachable!(),
            }
            return Ok(());
        }
        // plain import call — args at stored class, ret per call-site
        let mut params = Vec::with_capacity(args.len());
        let mut argv = Vec::with_capacity(args.len());
        for a in args {
            let k = self.stored_k(*a);
            params.push(k);
            argv.push(self.get(*a, k)?);
        }
        let ret = if let Some(&k) = self.ana.class.get(&ix) {
            k
        } else if self.ana.wused.contains(&ix) {
            K::Word
        } else {
            K::Int
        };
        let f = *self
            .natives
            .get(&(ni, params, Some(ret)))
            .ok_or("native import missing")?;
        let r = self.emit(WOp::Call { function_index: f }, &argv, &[wty(ret)]);
        self.store_dst(iid, r, ret)?;
        Ok(())
    }

    // ----- emit_bin transcription -----

    fn emit_bin(
        &mut self,
        dst: InstId,
        l: InstId,
        op: BinOp,
        r: InstId,
        kind: OperandKind,
    ) -> Result<(), Bail> {
        let mut nc = l;
        let mut op = op;
        let mut imm: Option<i64> = None;
        match op {
            BinOp::Add | BinOp::Mult => {
                if let Some(v) = self.imm(r, kind) {
                    imm = Some(v);
                } else if let Some(v) = self.imm(l, kind) {
                    imm = Some(v);
                    nc = r;
                }
            }
            BinOp::Sub | BinOp::Mod | BinOp::IDiv => {
                imm = self.imm(r, kind);
            }
            BinOp::LessThan
            | BinOp::LessEqual
            | BinOp::GreaterThan
            | BinOp::GreaterEqual
            | BinOp::Identity
            | BinOp::NotEqual => {
                if let Some(v) = self.imm(r, kind) {
                    imm = Some(v);
                } else if let Some(v) = self.imm(l, kind) {
                    imm = Some(v);
                    nc = r;
                    op = match op {
                        BinOp::LessThan => BinOp::GreaterThan,
                        BinOp::LessEqual => BinOp::GreaterEqual,
                        BinOp::GreaterThan => BinOp::LessThan,
                        BinOp::GreaterEqual => BinOp::LessEqual,
                        other => other,
                    };
                }
            }
            _ => {}
        }

        match kind {
            OperandKind::Int => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod | BinOp::IDiv => {
                    let v = if let Some(c) = imm {
                        self.checked_int_imm(nc, c, op)?
                    } else {
                        self.checked_int(l, r, op)?
                    };
                    self.store_dst(dst, v, K::Int)?;
                    return Ok(());
                }
                BinOp::Div => {
                    let a = self.get(l, K::Int)?;
                    let fa = self.emit(WOp::F64ConvertI64S, &[a], &[WTy::F64]);
                    let b = self.get(r, K::Int)?;
                    let fb = self.emit(WOp::F64ConvertI64S, &[b], &[WTy::F64]);
                    let d = self.emit(WOp::F64Div, &[fa, fb], &[WTy::F64]);
                    self.store_dst(dst, d, K::Float)?;
                    return Ok(());
                }
                BinOp::LessThan
                | BinOp::LessEqual
                | BinOp::GreaterThan
                | BinOp::GreaterEqual
                | BinOp::Identity
                | BinOp::NotEqual => {
                    let i = match op {
                        BinOp::LessThan => WOp::I64LtS,
                        BinOp::LessEqual => WOp::I64LeS,
                        BinOp::GreaterThan => WOp::I64GtS,
                        BinOp::GreaterEqual => WOp::I64GeS,
                        BinOp::Identity => WOp::I64Eq,
                        BinOp::NotEqual => WOp::I64Ne,
                        _ => unreachable!(),
                    };
                    let v = if let Some(c) = imm {
                        self.cmp_imm(nc, c, K::Int, K::Bool, i)?
                    } else {
                        self.cmp(l, r, K::Int, K::Bool, i)?
                    };
                    self.store_dst(dst, v, K::Bool)?;
                    return Ok(());
                }
                BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::BitShiftLeft
                | BinOp::BitShiftRight => {
                    let i = match op {
                        BinOp::BitAnd => WOp::I64And,
                        BinOp::BitOr => WOp::I64Or,
                        BinOp::BitXor => WOp::I64Xor,
                        BinOp::BitShiftLeft => WOp::I64Shl,
                        BinOp::BitShiftRight => WOp::I64ShrS,
                        _ => unreachable!(),
                    };
                    let v = self.cmp(l, r, K::Int, K::Int, i)?;
                    self.store_dst(dst, v, K::Int)?;
                    return Ok(());
                }
                _ => bail!("BinOp {op:?} on ints"),
            },
            OperandKind::Float => {
                let v = if let Some(c) = imm {
                    match op {
                        BinOp::Add => self.cmp_imm(nc, c, K::Float, K::Float, WOp::F64Add)?,
                        BinOp::Sub => self.cmp_imm(nc, c, K::Float, K::Float, WOp::F64Sub)?,
                        BinOp::Mult => self.cmp_imm(nc, c, K::Float, K::Float, WOp::F64Mul)?,
                        BinOp::Mod => {
                            let f = f64::from_bits(c as u64);
                            let a = self.get(nc, K::Float)?;
                            let a2 = self.get(nc, K::Float)?;
                            let cst = self.kf(f);
                            let q = self.emit(WOp::F64Div, &[a2, cst], &[WTy::F64]);
                            let t = self.emit(WOp::F64Trunc, &[q], &[WTy::F64]);
                            let cst2 = self.kf(f);
                            let m = self.emit(WOp::F64Mul, &[t, cst2], &[WTy::F64]);
                            self.emit(WOp::F64Sub, &[a, m], &[WTy::F64])
                        }
                        BinOp::LessThan => self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Lt)?,
                        BinOp::LessEqual => self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Le)?,
                        BinOp::GreaterThan => self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Gt)?,
                        BinOp::GreaterEqual => {
                            self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Ge)?
                        }
                        BinOp::Identity => self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Eq)?,
                        BinOp::NotEqual => self.cmp_imm(nc, c, K::Float, K::Bool, WOp::F64Ne)?,
                        _ => bail!("BinOp {op:?} on float imm"),
                    }
                } else {
                    match op {
                        BinOp::Add => self.cmp(l, r, K::Float, K::Float, WOp::F64Add)?,
                        BinOp::Sub => self.cmp(l, r, K::Float, K::Float, WOp::F64Sub)?,
                        BinOp::Mult => self.cmp(l, r, K::Float, K::Float, WOp::F64Mul)?,
                        BinOp::Div | BinOp::IDiv => {
                            self.cmp(l, r, K::Float, K::Float, WOp::F64Div)?
                        }
                        BinOp::Mod => {
                            let a = self.get(l, K::Float)?;
                            let a2 = self.get(l, K::Float)?;
                            let b = self.get(r, K::Float)?;
                            let q = self.emit(WOp::F64Div, &[a2, b], &[WTy::F64]);
                            let t = self.emit(WOp::F64Trunc, &[q], &[WTy::F64]);
                            let b2 = self.get(r, K::Float)?;
                            let m = self.emit(WOp::F64Mul, &[t, b2], &[WTy::F64]);
                            self.emit(WOp::F64Sub, &[a, m], &[WTy::F64])
                        }
                        BinOp::LessThan => self.cmp(l, r, K::Float, K::Bool, WOp::F64Lt)?,
                        BinOp::LessEqual => self.cmp(l, r, K::Float, K::Bool, WOp::F64Le)?,
                        BinOp::GreaterThan => self.cmp(l, r, K::Float, K::Bool, WOp::F64Gt)?,
                        BinOp::GreaterEqual => self.cmp(l, r, K::Float, K::Bool, WOp::F64Ge)?,
                        BinOp::Identity => self.cmp(l, r, K::Float, K::Bool, WOp::F64Eq)?,
                        BinOp::NotEqual => self.cmp(l, r, K::Float, K::Bool, WOp::F64Ne)?,
                        _ => bail!("BinOp {op:?} on floats"),
                    }
                };
                let produced = binop_prod(op, kind).unwrap_or(K::Word);
                self.store_dst(dst, v, produced)?;
                return Ok(());
            }
            OperandKind::Bool => {
                let v = match op {
                    BinOp::Identity => self.cmp(l, r, K::Bool, K::Bool, WOp::I32Eq)?,
                    BinOp::NotEqual | BinOp::Xor => self.cmp(l, r, K::Bool, K::Bool, WOp::I32Ne)?,
                    BinOp::And => self.cmp(l, r, K::Bool, K::Bool, WOp::I32And)?,
                    BinOp::Or => self.cmp(l, r, K::Bool, K::Bool, WOp::I32Or)?,
                    _ => bail!("BinOp {op:?} on bools"),
                };
                self.store_dst(dst, v, K::Bool)?;
                return Ok(());
            }
            OperandKind::Generic if matches!(op, BinOp::Identity | BinOp::NotEqual) => {
                let i = if op == BinOp::Identity {
                    WOp::I64Eq
                } else {
                    WOp::I64Ne
                };
                let v = self.cmp(l, r, K::Word, K::Bool, i)?;
                self.store_dst(dst, v, K::Bool)?;
                return Ok(());
            }
            OperandKind::Str => match op {
                BinOp::Add => {
                    let a = self.get(l, K::Word)?;
                    let a32 = self.wrap(a);
                    let b = self.get(r, K::Word)?;
                    let b32 = self.wrap(b);
                    let v = self.call_h("__str_cat", &[a32, b32], &[WTy::I32]);
                    let v = self.extend(v);
                    self.store_dst(dst, v, K::Word)?;
                    return Ok(());
                }
                BinOp::Identity | BinOp::NotEqual => {
                    let a = self.get(l, K::Word)?;
                    let a32 = self.wrap(a);
                    let b = self.get(r, K::Word)?;
                    let b32 = self.wrap(b);
                    let mut v = self.call_h("__str_eq", &[a32, b32], &[WTy::I32]);
                    if op == BinOp::NotEqual {
                        v = self.emit(WOp::I32Eqz, &[v], &[WTy::I32]);
                    }
                    self.store_dst(dst, v, K::Bool)?;
                    return Ok(());
                }
                BinOp::LessThan | BinOp::LessEqual | BinOp::GreaterThan | BinOp::GreaterEqual => {
                    let a = self.get(l, K::Word)?;
                    let a32 = self.wrap(a);
                    let b = self.get(r, K::Word)?;
                    let b32 = self.wrap(b);
                    let c = self.call_h("__str_cmp", &[a32, b32], &[WTy::I32]);
                    let z = self.k32(0);
                    let i = match op {
                        BinOp::LessThan => WOp::I32LtS,
                        BinOp::LessEqual => WOp::I32LeS,
                        BinOp::GreaterThan => WOp::I32GtS,
                        _ => WOp::I32GeS,
                    };
                    let v = self.emit(i, &[c, z], &[WTy::I32]);
                    self.store_dst(dst, v, K::Bool)?;
                    return Ok(());
                }
                _ => bail!("BinOp {op:?} on strs"),
            },
            OperandKind::Generic => {
                let g = self
                    .ana
                    .class
                    .get(&(dst.index() as u32))
                    .or_else(|| self.ana.class.get(&(l.index() as u32)))
                    .or_else(|| self.ana.class.get(&(r.index() as u32)))
                    .copied()
                    .unwrap_or(K::Float);
                let gk = match g {
                    K::Int => OperandKind::Int,
                    K::Float => OperandKind::Float,
                    K::Bool => OperandKind::Bool,
                    K::Word => unreachable!("guess is scalar"),
                };
                return self.emit_bin(dst, l, op, r, gk);
            }
        }
    }

    /// `l OP r` checked for i64 overflow; returns the result value.
    fn checked_int(&mut self, l: InstId, r: InstId, op: BinOp) -> Result<Value, Bail> {
        if matches!(op, BinOp::Mod | BinOp::IDiv) {
            let a = self.get(l, K::Int)?;
            let b = self.get(r, K::Int)?;
            return Ok(self.emit(
                if op == BinOp::Mod {
                    WOp::I64RemS
                } else {
                    WOp::I64DivS
                },
                &[a, b],
                &[WTy::I64],
            ));
        }
        let a = self.get(l, K::Int)?;
        let b = self.get(r, K::Int)?;
        let tmp = self.emit(
            match op {
                BinOp::Add => WOp::I64Add,
                BinOp::Sub => WOp::I64Sub,
                BinOp::Mult => WOp::I64Mul,
                _ => bail!("checked_int {op:?}"),
            },
            &[a, b],
            &[WTy::I64],
        );
        match op {
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (WOp::I64LtS, WOp::I64GtS)
                } else {
                    (WOp::I64GtS, WOp::I64LtS)
                };
                let z = self.k64(0);
                let bpos = self.emit(WOp::I64GtS, &[b, z], &[WTy::I32]);
                let a2 = self.get(l, K::Int)?;
                let m1 = self.emit(c1, &[tmp, a2], &[WTy::I32]);
                let p1 = self.emit(WOp::I32And, &[bpos, m1], &[WTy::I32]);
                let z = self.k64(0);
                let bneg = self.emit(WOp::I64LtS, &[b, z], &[WTy::I32]);
                let a3 = self.get(l, K::Int)?;
                let m2 = self.emit(c2, &[tmp, a3], &[WTy::I32]);
                let p2 = self.emit(WOp::I32And, &[bneg, m2], &[WTy::I32]);
                let bad = self.emit(WOp::I32Or, &[p1, p2], &[WTy::I32]);
                self.trap_if(bad);
            }
            BinOp::Mult => {
                let bz = self.emit(WOp::I64Eqz, &[b], &[WTy::I32]);
                let nonzero = self.emit(WOp::I32Eqz, &[bz], &[WTy::I32]);
                let (t, cont) = self.branch(nonzero);
                self.w = t;
                let b2 = self.get(r, K::Int)?;
                let q = self.emit(WOp::I64DivS, &[tmp, b2], &[WTy::I64]);
                let a2 = self.get(l, K::Int)?;
                let bad = self.emit(WOp::I64Ne, &[q, a2], &[WTy::I32]);
                self.trap_if(bad);
                let t_end = self.w;
                self.w = cont;
                self.join(&[(t_end, vec![]), (cont, vec![])], &[]);
            }
            _ => unreachable!(),
        }
        Ok(tmp)
    }

    fn checked_int_imm(&mut self, l: InstId, v: i64, op: BinOp) -> Result<Value, Bail> {
        match op {
            BinOp::Mod | BinOp::IDiv => {
                if v == 0 {
                    self.emit(WOp::Unreachable, &[], &[]);
                    // unreachable code still needs a value for the record —
                    // dead on every path that reaches it
                    return Ok(self.k64(0));
                }
                let a = self.get(l, K::Int)?;
                let c = self.k64(v);
                return Ok(self.emit(
                    if op == BinOp::Mod {
                        WOp::I64RemS
                    } else {
                        WOp::I64DivS
                    },
                    &[a, c],
                    &[WTy::I64],
                ));
            }
            BinOp::Mult if v == 0 => return Ok(self.k64(0)),
            _ => {}
        }
        let a = self.get(l, K::Int)?;
        let c = self.k64(v);
        let tmp = self.emit(
            match op {
                BinOp::Add => WOp::I64Add,
                BinOp::Sub => WOp::I64Sub,
                BinOp::Mult => WOp::I64Mul,
                _ => bail!("checked_int_imm {op:?}"),
            },
            &[a, c],
            &[WTy::I64],
        );
        match op {
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (WOp::I64LtS, WOp::I64GtS)
                } else {
                    (WOp::I64GtS, WOp::I64LtS)
                };
                let z = self.k64(0);
                let cv = self.k64(v);
                let bpos = self.emit(WOp::I64GtS, &[cv, z], &[WTy::I32]);
                let a2 = self.get(l, K::Int)?;
                let m1 = self.emit(c1, &[tmp, a2], &[WTy::I32]);
                let p1 = self.emit(WOp::I32And, &[bpos, m1], &[WTy::I32]);
                let z = self.k64(0);
                let cv = self.k64(v);
                let bneg = self.emit(WOp::I64LtS, &[cv, z], &[WTy::I32]);
                let a3 = self.get(l, K::Int)?;
                let m2 = self.emit(c2, &[tmp, a3], &[WTy::I32]);
                let p2 = self.emit(WOp::I32And, &[bneg, m2], &[WTy::I32]);
                let bad = self.emit(WOp::I32Or, &[p1, p2], &[WTy::I32]);
                self.trap_if(bad);
            }
            BinOp::Mult => {
                let cv = self.k64(v);
                let q = self.emit(WOp::I64DivS, &[tmp, cv], &[WTy::I64]);
                let a2 = self.get(l, K::Int)?;
                let bad = self.emit(WOp::I64Ne, &[q, a2], &[WTy::I32]);
                self.trap_if(bad);
            }
            _ => unreachable!(),
        }
        Ok(tmp)
    }

    /// `l i r` with operands at class `want`, result declared at `res`'s type.
    fn cmp(&mut self, l: InstId, r: InstId, want: K, res: K, i: WOp) -> Result<Value, Bail> {
        let a = self.get(l, want)?;
        let b = self.get(r, want)?;
        Ok(self.emit(i, &[a, b], &[wty(res)]))
    }

    fn cmp_imm(&mut self, l: InstId, v: i64, want: K, res: K, i: WOp) -> Result<Value, Bail> {
        let a = self.get(l, want)?;
        let b = match want {
            K::Int => self.k64(v),
            K::Float => self.kf(f64::from_bits(v as u64)),
            K::Bool => self.k32(v as i32),
            K::Word => unreachable!("word is never a comparison class"),
        };
        Ok(self.emit(i, &[a, b], &[wty(res)]))
    }
}

// ---------- body driver ----------

#[allow(clippy::too_many_arguments)]
fn wbody_full(
    module: &Module,
    sig: Signature,
    ir: &Ir,
    bid: usize,
    ana: &AnaI,
    sigs: &[Option<Sig>],
    funcs: &HashMap<usize, Func>,
    natives: &HashMap<(u32, Vec<K>, Option<K>), Func>,
    helpers: &HashMap<&'static str, Func>,
    tramp: &HashMap<usize, u32>,
    tramp_sig: Signature,
    entry_locals: &HashSet<u32>,
    entry_base: u32,
    ctx: &Statics,
    fuel_g: Option<Global>,
    pause_g: Option<Global>,
    cov_base: u32,
    sink: &crate::CovSink,
    math: &crate::MathNatives,
    cov: Option<&CovCtx>,
) -> Result<FunctionBody, Bail> {
    let body = &ir.bodies[compile::BodyId::from(bid as u32)];
    let mut fb = FunctionBody::new(module, sig);
    let entry = fb.entry;

    let mut wb: Vec<WBlock> = Vec::with_capacity(body.blocks.len());
    for (b, _) in body.blocks.iter() {
        wb.push(if b.index() == 0 {
            entry
        } else {
            fb.add_block()
        });
    }

    // blockparams: phis (stream order, at their stored class) then live locals
    let live: Vec<u32> = {
        let mut v: Vec<u32> = ana.lused.iter().copied().collect();
        v.sort();
        v
    };
    let mut phi_val: HashMap<u32, Value> = HashMap::new();
    let mut phis: Vec<Vec<InstId>> = vec![vec![]; body.blocks.len()];
    let mut local_bp: Vec<Vec<Value>> = vec![vec![]; body.blocks.len()];
    for (b, blk) in body.blocks.iter() {
        if b.index() == 0 {
            continue;
        }
        for &iid in &blk.stream {
            if let Inst::Phi(_) = &body.instructions[iid] {
                // dead phis get no blockparam — their sources needn't
                // materialize either
                let ix = iid.index() as u32;
                if !ana.class.contains_key(&ix) && !ana.wused.contains(&ix) {
                    continue;
                }
                phis[b.index()].push(iid);
                let pk = ana.class.get(&ix).copied().unwrap_or(K::Word);
                phi_val.insert(ix, fb.add_blockparam(wb[b.index()], wty(pk)));
            }
        }
        for &l in &live {
            let lk = ana.lclass.get(&l).copied().unwrap_or(K::Word);
            local_bp[b.index()].push(fb.add_blockparam(wb[b.index()], wty(lk)));
        }
    }

    let order = dfs_order(body);
    let pos: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(i, b)| (b.index(), i))
        .collect();
    let mut order_next: HashMap<usize, usize> = HashMap::new();
    for wpair in order.windows(2) {
        order_next.insert(wpair[0].index(), wpair[1].index());
    }

    let mut cx = WFx {
        body,
        ana,
        sigs,
        fb,
        wb,
        w: entry,
        phis,
        phi_val,
        live: live.clone(),
        local_bp,
        iv: HashMap::new(),
        cur: HashMap::new(),
        funcs,
        natives,
        helpers,
        tramp,
        tramp_sig,
        is_entry: bid == 0,
        entry_locals,
        entry_base,
        ctx,
        fuel_g,
        pause_g,
        cov_base,
        seq_i: 0,
        sink,
        math,
        cov,
    };

    for &b in &order {
        let bx = b.index();
        cx.w = cx.wb[bx];
        // seed cur: blockparams on non-entry; params/captures + zeros at entry
        cx.cur.clear();
        if bx == 0 {
            for (i, p) in body.captures.iter().enumerate() {
                cx.cur
                    .insert(p.index() as u32, cx.fb.blocks[entry].params[i].1);
            }
            let ncaps = body.captures.len();
            for (i, p) in body.params.iter().enumerate() {
                cx.cur
                    .insert(p.index() as u32, cx.fb.blocks[entry].params[ncaps + i].1);
            }
            for &l in &cx.live.clone() {
                if !cx.cur.contains_key(&l) {
                    let z = cx.zero(wty(ana.lclass.get(&l).copied().unwrap_or(K::Word)));
                    cx.cur.insert(l, z);
                }
            }
            // entry-shared params mirror into the memory region
            if cx.is_entry {
                for (i, p) in body.params.iter().enumerate() {
                    let px = p.index() as u32;
                    if cx.entry_locals.contains(&px) {
                        let pk = ana.lclass.get(&px).copied().unwrap_or(K::Word);
                        let v = cx.fb.blocks[entry].params[ncaps + i].1;
                        let wv = cx.coerce_val(v, pk, K::Word)?;
                        let a = cx.eaddr(px);
                        cx.store64(a, wv, 0);
                    }
                }
            }
        } else {
            for (i, &l) in cx.live.clone().iter().enumerate() {
                cx.cur.insert(l, cx.local_bp[bx][i]);
            }
        }

        // Phi insts materialize their blockparams
        for &piid in &cx.phis[bx].clone() {
            let pk = {
                let ix = piid.index() as u32;
                ana.class.get(&ix).copied().unwrap_or(K::Word)
            };
            let v = cx.phi_val[&(piid.index() as u32)];
            cx.iv.insert(piid.index() as u32, (v, pk));
        }

        let blk = &body.blocks[b];
        let mut jump: Option<BlockId> = None;
        let mut condbr: Option<(InstId, BlockId)> = None;
        let mut fornext: Option<(InstId, InstId, BlockId)> = None;
        let mut switch: Option<(InstId, u32, Vec<BlockId>, BlockId)> = None;
        let mut ret: Option<InstId> = None;
        let mut dead = false;

        for &iid in &blk.stream {
            cx.tick(cx.seq_i);
            cx.seq_i += 1;
            match &body.instructions[iid] {
                Inst::Jump { target } => jump = Some(*target),
                Inst::JumpIfFalse { condition, target } => condbr = Some((*condition, *target)),
                Inst::ForNext { idx, bound, target } => fornext = Some((*idx, *bound, *target)),
                Inst::Switch {
                    scrut,
                    base,
                    table,
                    default,
                } => switch = Some((*scrut, *base, table.clone(), *default)),
                Inst::Return(v) => {
                    ret = Some(*v);
                    break;
                }
                Inst::Raise(_) | Inst::Panic => {
                    cx.fb.set_terminator(cx.w, Terminator::Unreachable);
                    dead = true;
                    break;
                }
                _ => cx
                    .emit_inst(iid)
                    .map_err(|e| format!("{e} @emit_inst({})", iid.index()))?,
            }
        }
        if dead {
            continue;
        }

        // pause check when any branch target is a dfs-backward edge
        let backward = |t: BlockId| pos[&t.index()] <= pos[&bx];

        if let Some(v) = ret {
            match ana.ret {
                Some(k) => {
                    let val = cx.get(v, k)?;
                    cx.fb
                        .set_terminator(cx.w, Terminator::Return { values: vec![val] });
                }
                None => {
                    cx.fb
                        .set_terminator(cx.w, Terminator::Return { values: vec![] });
                }
            }
            continue;
        }
        if let Some((scrut, base, table, default)) = switch {
            let mut s = cx.get(scrut, K::Int)?;
            if base != 0 {
                let k = cx.k64(base as i64);
                s = cx.emit(WOp::I64Sub, &[s, k], &[WTy::I64]);
            }
            let s32 = cx.emit(WOp::I32WrapI64, &[s], &[WTy::I32]);
            let targets = table
                .iter()
                .map(|t| cx.tgt(bx, t.index()))
                .collect::<Result<Vec<_>, _>>()?;
            let d = cx.tgt(bx, default.index())?;
            cx.fb.set_terminator(
                cx.w,
                Terminator::Select {
                    value: s32,
                    targets,
                    default: d,
                },
            );
            continue;
        }
        if let Some((idx_iid, bound_iid, t)) = fornext {
            let Some(jt) = jump else {
                bail!("for_next without companion jump")
            };
            let Inst::GetLocal(l) = &body.instructions[idx_iid] else {
                bail!("for_next idx not a local read")
            };
            let lx = l.index() as u32;
            let cur = *cx.cur.get(&lx).ok_or("for_next idx local unclassed")?;
            let one = cx.k64(1);
            let i1 = cx.emit(WOp::I64Add, &[cur, one], &[WTy::I64]);
            cx.cur.insert(lx, i1);
            let bv = cx.get(bound_iid, K::Int)?;
            let c = cx.emit(WOp::I64LtS, &[i1, bv], &[WTy::I32]);
            if backward(t) || backward(jt) {
                cx.pause_check();
            }
            let tt = cx.tgt(bx, t.index())?;
            let ft = cx.tgt(bx, jt.index())?;
            cx.fb.set_terminator(
                cx.w,
                Terminator::CondBr {
                    cond: c,
                    if_true: tt,
                    if_false: ft,
                },
            );
            continue;
        }
        if let Some((cond, t)) = condbr {
            let Some(jt) = jump else {
                bail!("jump_if_false without companion jump")
            };
            let cv = cx.get(cond, K::Bool)?;
            if backward(t) || backward(jt) {
                cx.pause_check();
            }
            let tt = cx.tgt(bx, jt.index())?;
            let ft = cx.tgt(bx, t.index())?;
            cx.fb.set_terminator(
                cx.w,
                Terminator::CondBr {
                    cond: cv,
                    if_true: tt,
                    if_false: ft,
                },
            );
            continue;
        }
        if let Some(t) = jump {
            if backward(t) {
                cx.pause_check();
            }
            let tt = cx.tgt(bx, t.index())?;
            cx.fb.set_terminator(cx.w, Terminator::Br { target: tt });
            continue;
        }
        if let Some(&nx) = order_next.get(&bx) {
            let tt = cx.tgt(bx, nx)?;
            cx.fb.set_terminator(cx.w, Terminator::Br { target: tt });
        } else {
            // dead tail — unreachable blocks emit unreachable
            cx.fb.set_terminator(cx.w, Terminator::Unreachable);
        }
    }

    // unreachable blocks still need terminators
    let done: HashSet<usize> = order.iter().map(|b| b.index()).collect();
    for (b, _) in body.blocks.iter() {
        if done.contains(&b.index()) {
            continue;
        }
        cx.fb
            .set_terminator(cx.wb[b.index()], Terminator::Unreachable);
    }

    cx.fb.recompute_edges();
    Ok(cx.fb)
}
