use std::collections::HashMap;

use shared::{IdVec, Location};
use solve::components::DecId;

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
    /// The solved type of each inst that lowered from an expr, by value class. Codegen
    /// allocates registers per class from this.
    pub kinds: HashMap<InstId, OperandKind>,

    pub(crate) current_block: BlockId,
    pub(crate) loop_stack: Vec<LoopCtx>,

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
            artifacts: HashMap::new(),
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
