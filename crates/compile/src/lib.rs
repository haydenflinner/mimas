mod ir {
    #[allow(clippy::module_inception)]
    mod ir;
    pub use ir::*;
    mod emit;
    pub(crate) use emit::is_generic_law;
    use emit::*;
    mod place;
    use place::*;
    mod body;
    mod constant;
    mod ids;
    mod inst;
    mod ops;
    pub use body::*;
    pub use constant::*;
    pub use ids::*;
    pub use inst::*;
    pub use ops::*;

    // rexport this
    pub use parse::AccessKind;
}
pub use ir::*;

#[macro_use]
mod codegen {
    mod bodycache;
    mod clean;
    mod compiler;
    pub(crate) mod opt;
    #[macro_use]
    mod op;
    mod program;
    mod render;
    pub use bodycache::body_cache_stats;
    pub use compiler::*;
    pub use op::*;
    pub use program::*;
}
pub use codegen::*;

mod dataflow;
pub use dataflow::*;

#[cfg(test)]
mod tests {
    #[macro_use]
    mod compile_test_utils;

    mod bytecode;
    mod ir_tests;
    mod opt_tests;
}
