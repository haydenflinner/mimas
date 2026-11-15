use api::{Intrinsic, NativeId};
use itertools::Itertools;
use parse::NodeId;
use shared::{IdVec, Location, StrId, StrInterner};
use solve::{
    Resolutions,
    components::{DecId, Ty},
};
use std::collections::HashMap;

use crate::{
    IrDisplay,
    ir::{Block, BlockId, BlockWriter},
};

use super::{Body, BodyId, Inst, InstId, Local};

pub struct Ir {
    pub bodies: IdVec<BodyId, Body>,
    pub str_interner: StrInterner,
    pub item_bodies: HashMap<DecId, BodyId>,
    /// Names of functions marked `#[test]`, in source order -- the inspector (and anything else
    /// that wants to run tests) looks these up in `Program::items` after compile.
    pub tests: Vec<String>,
    pub closure_bodies: HashMap<BodyId, NodeId>,
    pub(crate) resolutions: Resolutions,
    pub(crate) intrinsics: HashMap<NativeId, Intrinsic>,

    pub(crate) current_body: BodyId,
    body_stack: Vec<BodyId>,
    pub(crate) current_loc: Location,
    /// How many statements deep the lowering is: 1 for a statement written directly in a
    /// file's top level, more inside a block, loop or fn body. A `let` echoes only at 1.
    pub(crate) stmt_depth: u32,
    /// Bodies holding a test case (`check` lines, `#[tests]` exprs, `#[test]` fns): their
    /// statements echo at any depth, so a host that re-runs a failing check with `Echoes`
    /// on gets the tape of what it computed. The calls cost one no-op native entry while
    /// recording stays off. r[impl check.tape]
    pub(crate) test_bodies: std::collections::HashSet<BodyId>,
    /// The entry-frame slot of each stored `const`: a top-level const whose value is
    /// a list, a tuple, a struct or a record. Its value is built once, at the start
    /// of the program, and frozen. Every use reads the slot (`mimas.const.shared`).
    pub(crate) const_slots: HashMap<DecId, Local>,
}

impl Ir {
    pub fn new(resolutions: Resolutions, intrinsics: HashMap<NativeId, Intrinsic>) -> Self {
        let root = Body::new();
        let mut bodies = IdVec::new();
        bodies.push(root);
        Self {
            bodies,
            resolutions,
            intrinsics,
            str_interner: StrInterner::new(),
            current_body: BodyId::ZERO,
            body_stack: Vec::new(),
            item_bodies: HashMap::new(),
            tests: Vec::new(),
            closure_bodies: HashMap::new(),
            current_loc: Location::SYNTHETIC,
            stmt_depth: 0,
            test_bodies: std::collections::HashSet::new(),
            const_slots: HashMap::new(),
        }
    }

    pub(crate) fn with_loc<R>(&mut self, loc: Location, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = std::mem::replace(&mut self.current_loc, loc);
        let r = f(self);
        self.current_loc = prev;
        r
    }

    pub fn lower(&mut self, stmts: &[parse::Stmt]) {
        use super::Lower;
        use shared::Located;
        // r[impl mimas.const.computed]
        // a const the checker couldn't fold is computed first, in source order,
        // into the global its name reads: it is ready before any code uses it
        for stmt in stmts {
            if let parse::StmtKind::Item(item) = stmt.kind()
                && let parse::ItemKind::Const(con) = item.kind()
            {
                if self.resolutions.computed_consts.contains(&item.id()) {
                    let value = self.with_loc(con.right.location(), |ir| con.right.lower(ir));
                    if let Some(value) = value {
                        self.pattern(&con.left, Some(value));
                    }
                } else {
                    self.store_const(item.id(), con);
                }
            }
        }
        for stmt in stmts {
            self.stmt(stmt);
        }
        // the entry body implicitly returns null; emit it explicitly so its final block ends in a
        // terminator like every other body -- codegen no longer synthesizes one.
        if !self.is_terminated() {
            let null = self.current().constant(crate::Constant::Null);
            self.current().ret(null);
        }
    }

    // r[impl mimas.const.shared]
    /// A top-level `const` whose value is a list, a tuple, a struct or a record is
    /// built here, once, frozen, and kept in a slot of the entry frame. Every use of
    /// it reads that slot: one value for the whole program, which nothing can change.
    /// Without the `__freeze` native (no std) nothing is stored, and each use builds
    /// its own value as before.
    fn store_const(&mut self, item: parse::NodeId, con: &parse::Const) {
        use shared::Located;
        if self.resolutions.freeze_native.is_none() {
            return;
        }
        let Some(dec) = self.resolutions.node_decs.get(&item).copied() else {
            return; // a destructuring const: its names are plain values
        };
        let solve::ResolvedDeclKind::Constant(lit) = &self.resolutions.decs[dec].kind else {
            return;
        };
        if !matches!(
            lit,
            parse::Literal::Array(_)
                | parse::Literal::Tuple(_)
                | parse::Literal::Struct(_)
                | parse::Literal::Record(_)
        ) {
            return;
        }
        // with no slot yet, this builds the value and freezes it
        let value = self.with_loc(con.right.location(), |ir| {
            crate::Constant::emit_const_dec(ir, dec, con.right.id())
        });
        if let Some(value) = value {
            let local = self.local_in(dec, BodyId::ZERO);
            // the slot's name says what it is: a live reload carries a `let` over to
            // the edited program by its name, and must not carry a const (`rebind`)
            let name = format!("const {}", self.resolutions.decs[dec].name);
            self.bodies[BodyId::ZERO].artifacts.insert(local, name);
            self.current().set_local(local, value);
            self.const_slots.insert(dec, local);
        }
    }

    // pub fn ir_display(&self) -> String {
    //     self.functions
    //         .iter()
    //         .map(|f| f.ir_display(self))
    //         .collect::<Vec<_>>()
    //         .join("\n")
    // }

    pub(crate) fn is_terminated(&self) -> bool {
        let inst = self
            .current_body()
            .current_stream()
            .last()
            .map(|v| &self.current_body().instructions[v]);

        matches!(
            inst,
            Some(Inst::Jump { .. })
                | Some(Inst::JumpIfFalse { .. })
                | Some(Inst::Return { .. })
                | Some(Inst::Switch { .. })
        )
    }

    pub(crate) fn in_body<T>(&mut self, body: BodyId, emit: impl FnOnce(&mut Ir) -> T) -> T {
        self.body_stack.push(self.current_body);
        self.current_body = body;
        let out = emit(self);
        self.current_body = self.body_stack.pop().unwrap();
        out
    }

    pub(crate) fn push_block(&mut self, name: &str) -> BlockId {
        let block = Block::new();
        let id = self.current_body_mut().blocks.push(block);

        cfg_select! {
            feature = "logging" => {
                self.current_body_mut().block_names.insert(id, name.into());
            }

            _ => {
                let _name = name;
            }
        }

        id
    }

    pub(crate) fn target(&mut self, id: BlockId) {
        self.current_body_mut().current_block = id;
    }

    pub(crate) fn current_block_id(&self) -> BlockId {
        self.current_body().current_block
    }

    pub(crate) fn in_block<T>(
        &mut self,
        block: BlockId,
        emit: impl FnOnce(&mut BlockWriter<'_>) -> T,
    ) -> T {
        let old = self.current_block_id();
        self.target(block);
        let out = self.in_current(emit);
        self.target(old);
        out
    }

    pub(crate) fn in_current<T>(&mut self, emit: impl FnOnce(&mut BlockWriter<'_>) -> T) -> T {
        emit(&mut BlockWriter { ir: self })
    }

    pub(crate) fn current(&mut self) -> BlockWriter<'_> {
        BlockWriter { ir: self }
    }

    pub(super) fn loop_stack_mut(&mut self) -> &mut Vec<LoopCtx> {
        &mut self.current_body_mut().loop_stack
    }

    pub(super) fn push_inst(&mut self, inst: impl Into<Inst>) -> InstId {
        let loc = self.current_loc;
        let body = self.current_body_mut();
        let iid = body.instructions.push(inst.into());
        body.locs.push(loc);
        body.current_stream_mut().push(iid);
        iid
    }

    pub(crate) fn str(&self, id: StrId) -> &str {
        self.str_interner.get(id)
    }

    pub(super) fn intern_str(&mut self, s: &str) -> StrId {
        self.str_interner.intern(s)
    }

    pub(crate) fn try_node_dec(&self, node: NodeId) -> Option<DecId> {
        self.resolutions.node_decs.get(&node).copied()
    }

    pub(crate) fn node_dec(&self, node: NodeId) -> DecId {
        self.try_node_dec(node)
            .unwrap_or_else(|| panic!("no resolved DecId for node {node:?}"))
    }

    // struct fields are positional (declaration order, matching how instances are built);
    // resolve a named field on `receiver`'s type to its slot. unwraps an option layer if the
    // access was an optional `?.` -- the solver typed it that way but the slot lookup needs the
    // inner adt.
    pub(crate) fn field_index(&self, receiver: NodeId, field: &str) -> usize {
        let adt = match self.resolutions.node_tys.get(&receiver) {
            Some(Ty::Adt(adt, _)) | Some(Ty::Identity(adt, _)) => *adt,
            Some(Ty::Option(inner)) => match inner.as_ref() {
                Ty::Adt(adt, _) | Ty::Identity(adt, _) => *adt,
                other => panic!("field access on option of non-adt type: {other:?}"),
            },
            // axis names spell tuple slots: `pt.x`/`pt.y`/`pt.z`/`pt.t` are the
            // positions `.0`..`.3` index — the solver admits no other ident here.
            // r[impl mimas.tuple.axis]
            Some(Ty::Tuple(_)) => {
                return match field {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "t" => 3,
                    other => unreachable!("solver admitted `{other}` on a tuple"),
                };
            }
            other => panic!("field access on non-adt type: {other:?}"),
        };
        self.resolutions.adts[adt]
            .fields
            .iter()
            .position(|n| n == field)
            .unwrap_or_else(|| panic!("no field `{field}` on adt"))
    }

    // allocate-or-fetch this binding's slot within its home body: the entry body for
    // globals (a top-level `let` stores there no matter which body asks), the current body
    // for everything else.
    pub(crate) fn local_for(&mut self, dec: DecId) -> Local {
        self.local_in(dec, self.body_for_dec(dec))
    }

    /// The body that owns a dec's storage: globals live in the entry frame.
    pub(crate) fn body_for_dec(&self, dec: DecId) -> BodyId {
        match self.resolutions.decs[dec].kind {
            solve::ResolvedDeclKind::Global => BodyId::ZERO,
            _ => self.current_body,
        }
    }

    fn local_in(&mut self, dec: DecId, body_id: BodyId) -> Local {
        if let Some(&local) = self.bodies[body_id].dec_to_local.get(&dec) {
            return local;
        }
        let name = self.resolutions.decs[dec].name.to_string();
        let body = &mut self.bodies[body_id];
        let local = body.locals.push(dec);
        body.dec_to_local.insert(dec, local);
        body.artifacts.insert(local, name);
        local
    }

    /// Emits a read of `dec` in the current body. Globals reached from a non-entry body go
    /// through the entry frame's registers; everything else is an ordinary local read.
    pub(crate) fn read_binding(&mut self, dec: DecId) -> InstId {
        let local = self.local_for(dec);
        if self.body_for_dec(dec) != self.current_body {
            return self.current().get_entry(local);
        }
        self.current().get_local(local)
    }

    pub(crate) fn synthetic_local(&mut self, name: &str) -> Local {
        let body = &mut self.bodies[self.current_body];
        let local = body.locals.push(DecId::DANGLING);
        body.artifacts.insert(local, name.to_string());
        local
    }

    // allocate-or-fetch the body associated with this dec. A native has no lowered
    // body, but `ref_body` still needs one when the native is used as a value
    // (`let f = from_json`) -- synthesize a trampoline that forwards its params to
    // CallNative. Call sites don't go through this (they emit CallNative directly).
    pub(crate) fn item_body_for(&mut self, dec: DecId) -> BodyId {
        if let Some(&bid) = self.item_bodies.get(&dec) {
            return bid;
        }
        let bid = self.bodies.push(Body::new());
        self.item_bodies.insert(dec, bid);
        if let solve::ResolvedDeclKind::Item {
            native: Some(native),
            ..
        } = &self.resolutions.decs[dec].kind
        {
            let native = *native;
            let Ty::Fn(header) = &self.resolutions.decs[dec].ty else {
                panic!("a native dec's ty is always a Ty::Fn")
            };
            let arity = header.parameters.len();
            self.in_body(bid, |ir| {
                let params: Vec<Local> = (0..arity)
                    .map(|i| ir.synthetic_local(&format!("$trampoline_arg{i}")))
                    .collect();
                ir.current_body_mut().params = params.clone();
                let args: Vec<InstId> = params.iter().map(|&p| ir.current().get_local(p)).collect();
                let ret = ir.current().call_native(native, args);
                ir.current().ret(ret);
            });
        }
        bid
    }

    pub(super) fn current_body(&self) -> &Body {
        &self.bodies[self.current_body]
    }

    pub(super) fn current_body_mut(&mut self) -> &mut Body {
        &mut self.bodies[self.current_body]
    }
}

#[derive(Debug)]
pub struct LoopCtx {
    pub(crate) continuation: BlockId,
    pub(crate) exit: BlockId,
    pub(crate) breaks: Vec<(BlockId, InstId)>,
    pub(crate) continues: Vec<BlockId>,
    pub(crate) collection: Option<InstId>,
    pub(crate) collects: Vec<BlockId>,
}

impl LoopCtx {
    pub(super) fn new(continuation: BlockId, exit: BlockId, collection: Option<InstId>) -> Self {
        Self {
            continuation,
            exit,
            breaks: Vec::new(),
            continues: Vec::new(),
            collection,
            collects: Vec::new(),
        }
    }
}

#[mutants::skip]
impl std::fmt::Display for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // look upon my power
        write!(
            f,
            "{}",
            self.bodies
                .iter()
                .map(|(body_id, body)| {
                    std::iter::once(
                        (body_id != BodyId::ZERO).then(|| format!("body @{}", body_id.index())),
                    )
                    .flatten()
                    .chain(body.blocks.iter().map(|(block_id, block)| {
                        let instructions = block
                            .stream
                            .iter()
                            .map(|iid| {
                                format!(
                                    "    [ i{} ] {}",
                                    iid.index(),
                                    body.instructions[iid].ir_display(self),
                                )
                            })
                            .join("\n");

                        cfg_select! {
                            feature = "logging" => {
                                let name = body.block_names.get(&block_id).unwrap();
                            }
                            _ => {
                                let name = format!("b{}", block_id.index());
                            }
                        };

                        if instructions.is_empty() {
                            format!("{name}:")
                        } else {
                            format!("{name}:\n{instructions}")
                        }
                    }))
                    .join("\n")
                })
                .join("\n")
        )
    }
}

// impl IrDisplay for Body {
//     fn ir_display(&self, ir: &Ir) -> String {
//         let header = match self.name {
//             Some(n) => format!("fn @{} {}:", self.id.index(), ir.str(n)),
//             None if self.id.index() == 0 => String::new(), // for the root
//             None => format!("fn @{}:", self.id.index()),
//         };
//         let blocks = self
//             .blocks
//             .iter()
//             .map(|block| block.ir_display(ir))
//             .collect::<Vec<_>>()
//             .join("\n");
//         if header.is_empty() {
//             blocks
//         } else {
//             format!("{header}\n{blocks}")
//         }
//     }
// }
