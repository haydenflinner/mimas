use std::collections::HashMap;

use shared::{IdVec, Location};
use solve::components::{AdtId, DecId};

use crate::{BlockId, Inst, InstId, Local, OperandKind, ir::LoopCtx};

#[derive(Debug)]
pub struct Body {
    pub blocks: IdVec<BlockId, Block>,
    pub instructions: IdVec<InstId, Inst>,
    pub locals: IdVec<Local, DecId>,
    pub dec_to_local: HashMap<DecId, Local>,
    pub params: Vec<Local>,
    pub captures: Vec<Local>,
    pub locs: IdVec<InstId, Location>,
    pub artifacts: HashMap<Local, String>,
    /// The debugger's view of a named struct local the optimizer dissolved:
    /// a tree over the registers that now hold it. `Slot` is a local the
    /// program still writes (a slot SROA field) or a shadow local that only
    /// `Inst::Shadow` writes; `Struct` is a struct rebuilt from its fields.
    pub shadows: HashMap<Local, Shadow<Local>>,
    /// The solved type of each inst that lowered from an expr, by value class. Codegen
    /// allocates registers per class from this.
    pub kinds: HashMap<InstId, OperandKind>,

    pub(crate) current_block: BlockId,
    pub(crate) loop_stack: Vec<LoopCtx>,
    pub(crate) cleanup_scopes: Vec<InstId>,

    #[cfg(feature = "logging")]
    pub(crate) block_names: HashMap<BlockId, String>,
}

impl Body {
    pub(crate) fn new() -> Self {
        let mut blocks = IdVec::new();
        blocks.push(Block::new());
        Self {
            blocks,
            instructions: IdVec::new(),
            locals: IdVec::new(),
            locs: IdVec::new(),
            dec_to_local: HashMap::new(),
            params: Vec::new(),
            captures: Vec::new(),
            current_block: BlockId::ZERO,
            loop_stack: vec![],
            cleanup_scopes: vec![],
            artifacts: HashMap::new(),
            shadows: HashMap::new(),
            kinds: HashMap::new(),

            #[cfg(feature = "logging")]
            block_names: vec![(BlockId::ZERO, "root".to_string())]
                .into_iter()
                .collect(),
        }
    }

    pub(crate) fn current_stream(&self) -> &[InstId] {
        &self.blocks[self.current_block].stream
    }

    pub(crate) fn current_stream_mut(&mut self) -> &mut Vec<InstId> {
        &mut self.blocks[self.current_block].stream
    }
}

/// Where a dissolved struct local lives for the debugger: `T` is a `Local`
/// in the IR and a `Reg` in a chunk. A frame view reads the leaves and
/// rebuilds every `Struct` on the way up -- see `Vm::frames`.
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Shadow<T> {
    Slot(T),
    Struct { adt: AdtId, fields: Vec<Shadow<T>> },
}

impl<T> Shadow<T> {
    pub fn map<U>(&self, f: &impl Fn(&T) -> U) -> Shadow<U> {
        match self {
            Shadow::Slot(t) => Shadow::Slot(f(t)),
            Shadow::Struct { adt, fields } => Shadow::Struct {
                adt: *adt,
                fields: fields.iter().map(|x| x.map(f)).collect(),
            },
        }
    }

    pub fn leaves(&self) -> Vec<&T> {
        match self {
            Shadow::Slot(t) => vec![t],
            Shadow::Struct { fields, .. } => fields.iter().flat_map(|f| f.leaves()).collect(),
        }
    }

    /// The leaf `old`, replaced by `new` -- whether it was found.
    pub fn replace_leaf(&mut self, old: &T, new: Shadow<T>) -> bool
    where
        T: PartialEq + Clone,
    {
        match self {
            Shadow::Slot(t) if t == old => {
                *self = new;
                true
            }
            Shadow::Slot(_) => false,
            Shadow::Struct { fields, .. } => {
                for f in fields.iter_mut() {
                    if f.replace_leaf(old, new.clone()) {
                        return true;
                    }
                }
                false
            }
        }
    }
}

impl Body {
    /// The shadow tree whose leaf is `l`, if any: `(root local, tree)`.
    pub(crate) fn shadow_leaf_root(&self, l: Local) -> Option<Local> {
        self.shadows
            .iter()
            .find(|(_, sh)| sh.leaves().contains(&&l))
            .map(|(r, _)| *r)
    }
}

// todo, maybe this can die
#[derive(Debug, PartialEq)]
pub struct Block {
    pub stream: Vec<InstId>,
}

impl Block {
    pub(crate) fn new() -> Self {
        Self { stream: Vec::new() }
    }
}
