//! Extracts the pure dataflow structure of a single function body -- its parameters, the chain
//! of operations between them and its return, and which operation's *value* feeds into which
//! other operation -- for rendering as a circuit/block diagram: boxes for operations, wires for
//! values, no notion of "this happens, then this happens".
//!
//! Deliberately scoped to function bodies without real control flow. Note that's *not* the same
//! as "a single IR block": even a one-expression function like `typst_box` compiles to a chain
//! of several blocks (one per lexical-scope boundary) joined end to end by plain, unconditional
//! `Jump`s -- structural bookkeeping, not a branch. This walks that chain as one straight-line
//! sequence, and only bails out ([`DataflowError::HasControlFlow`]) on an actual decision: a
//! conditional jump, a switch, or revisiting a block (a loop). `if`, `loop`, and pattern
//! matching all produce one of those. There's no dataflow-only way to draw "which branch runs"
//! or "how many times this loops" that isn't lying about what the program does, so those get a
//! real design (a clocked-register style feedback node, state made explicit) as a deliberate
//! follow-up, not an accidental gap here.

use std::collections::HashMap;

use crate::{BlockId, BodyId, FormatPart, Inst, InstId, Ir, Local};

/// One function's dataflow graph: [`DataflowNode`]s (boxes) connected by [`DataflowEdge`]s
/// (wires). Node indices are stable within a single `DataflowGraph` (0-based, dense) -- an
/// edge's `from`/`to` are indices into `nodes`.
#[derive(Debug)]
pub struct DataflowGraph {
    pub function_name: String,
    pub nodes: Vec<DataflowNode>,
    pub edges: Vec<DataflowEdge>,
}

#[derive(Debug)]
pub struct DataflowNode {
    pub label: String,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// A parameter -- where a value enters the function.
    In,
    /// The function's return -- where a value leaves. At most one per graph (a single-block
    /// body can only end one way).
    Out,
    /// Any other operation: a binary op, a field access, a call, a constant, ...
    Op,
}

#[derive(Debug)]
pub struct DataflowEdge {
    pub from: usize,
    pub to: usize,
    /// Which operand slot on `to` this feeds -- "left"/"right" for a `BinOp`, say -- shown as a
    /// small label near the wire's destination end. `None` for single-operand or positional-list
    /// operands (a call's arguments, say), where the wire's own left-to-right position already
    /// conveys the order.
    pub port: Option<&'static str>,
}

#[derive(Debug)]
pub enum DataflowError {
    /// No function (top-level item) with this name was found.
    UnknownFunction(String),
    /// The body has real control flow (more than one IR block) -- see the module docs.
    HasControlFlow,
}

impl std::fmt::Display for DataflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataflowError::UnknownFunction(name) => write!(f, "no function named `{name}`"),
            DataflowError::HasControlFlow => write!(
                f,
                "this function has branches or loops -- dataflow view is scoped to straight-line \
                 functions for now"
            ),
        }
    }
}

/// Extracts `function_name`'s dataflow graph from an already-lowered [`Ir`]. `function_name` is
/// looked up the same way `Compiler::compile` builds `Program::items` -- a top-level item's
/// declared name.
pub fn function_dataflow(ir: &Ir, function_name: &str) -> Result<DataflowGraph, DataflowError> {
    let body_id = ir
        .item_bodies
        .iter()
        .find(|&(&dec, _)| ir.resolutions.decs[dec].name == function_name)
        .map(|(_, &body)| body)
        .ok_or_else(|| DataflowError::UnknownFunction(function_name.to_string()))?;

    // for `CallDirect`'s label -- same inversion `Compiler::compile` does to build `items`.
    let body_names: HashMap<BodyId, &str> = ir
        .item_bodies
        .iter()
        .map(|(&dec, &body)| (body, ir.resolutions.decs[dec].name.as_str()))
        .collect();

    let body = &ir.bodies[body_id];

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    // an instruction's node, once it has one -- `GetLocal`/`SetLocal` never get their own node
    // (see below), so not every `InstId` ends up here.
    let mut inst_node: HashMap<InstId, usize> = HashMap::new();
    // each local's current producing node -- updated on `SetLocal`, read on `GetLocal`. Mutable
    // locals in a single block behave like a register being rewritten in place, not SSA, so this
    // is a plain last-write-wins map, valid precisely because there's no control-flow merge to
    // reconcile (that's exactly what `blocks.len() != 1` above rules out).
    let mut local_source: HashMap<Local, usize> = HashMap::new();

    for &param in &body.params {
        let name = local_name(ir, body_id, param);
        let idx = nodes.len();
        nodes.push(DataflowNode { label: name, kind: NodeKind::In });
        local_source.insert(param, idx);
    }

    // Walk the chain of blocks connected by plain `Jump`s as one straight-line sequence --
    // see the module docs for why that's not the same as requiring a single block. `visited`
    // catches a backward jump (a loop) even though nothing here looks at edge direction.
    let mut current = BlockId::ZERO;
    let mut visited = std::collections::HashSet::new();
    loop {
        if !visited.insert(current) {
            return Err(DataflowError::HasControlFlow);
        }
        let mut next_block = None;
        for &inst_id in &body.blocks[current].stream {
            let inst = &body.instructions[inst_id];
            match inst {
                // structural fallthrough to the next block in the chain -- not a decision.
                Inst::Jump { target } => next_block = Some(*target),
                // an actual decision: which block runs next depends on a runtime value. No
                // dataflow-only rendering of "which" is honest, so this is where it stops.
                Inst::JumpIfFalse { .. } | Inst::Switch { .. } | Inst::ForNext { .. } | Inst::Phi(_) => {
                    return Err(DataflowError::HasControlFlow);
                }
                // a read of a mutable local -- not an operation of its own, just a reference to
                // whichever node last wrote that local (or the param node, if never reassigned).
                Inst::GetLocal(local) => {
                    if let Some(&src) = local_source.get(local) {
                        inst_node.insert(inst_id, src);
                    }
                }
                // a write to a mutable local -- also not an operation of its own (`let x = ..`/
                // `x = ..` is naming a wire, not a gate); just repoints the local at whatever
                // node produced the assigned value.
                Inst::SetLocal(local, value) => {
                    if let Some(&src) = inst_node.get(value) {
                        local_source.insert(*local, src);
                    }
                }
                // the function's single exit -- becomes the one `Out` node instead of a regular
                // op. Can only be reached once: every path here has been checked branch-free.
                Inst::Return(value) => {
                    let idx = nodes.len();
                    nodes.push(DataflowNode { label: "out".to_string(), kind: NodeKind::Out });
                    if let Some(&src) = inst_node.get(value) {
                        edges.push(DataflowEdge { from: src, to: idx, port: None });
                    }
                    inst_node.insert(inst_id, idx);
                }
                other => {
                    let idx = nodes.len();
                    nodes.push(DataflowNode {
                        label: node_label(ir, other, &body_names),
                        kind: NodeKind::Op,
                    });
                    for (operand, port) in operands_of(other) {
                        if let Some(&src) = inst_node.get(&operand) {
                            edges.push(DataflowEdge { from: src, to: idx, port });
                        }
                    }
                    inst_node.insert(inst_id, idx);
                }
            }
        }
        match next_block {
            Some(target) => current = target,
            None => break,
        }
    }

    Ok(DataflowGraph { function_name: function_name.to_string(), nodes, edges })
}

fn local_name(ir: &Ir, body_id: BodyId, local: Local) -> String {
    let dec = ir.bodies[body_id].locals[local];
    ir.resolutions.decs[dec].name.clone()
}

/// Every `InstId`-valued operand of `inst`, paired with a port label where one operand could
/// otherwise be confused for another (`left`/`right` on a `BinOp`, say). Control-flow variants
/// and the ones handled specially by the caller (`GetLocal`/`SetLocal`/`Return`) return nothing
/// -- they're never passed here.
fn operands_of(inst: &Inst) -> Vec<(InstId, Option<&'static str>)> {
    match inst {
        Inst::Constant(_)
        | Inst::NewArray
        | Inst::NewDict
        | Inst::Panic
        | Inst::RefBody(_)
        | Inst::GetLocal(_)
        | Inst::SetLocal(..)
        | Inst::Return(_)
        | Inst::Jump { .. }
        | Inst::JumpIfFalse { .. }
        | Inst::Switch { .. }
        | Inst::Phi(_)
        | Inst::ForNext { .. } => vec![],
        Inst::BinOp { left, right, .. } => vec![(*left, Some("left")), (*right, Some("right"))],
        Inst::UnaryOp { right, .. } => vec![(*right, None)],
        Inst::SetIndex { set, index, value } => {
            vec![(*set, Some("set")), (*index, Some("index")), (*value, Some("value"))]
        }
        Inst::GetIndex { set, index, .. } => vec![(*set, Some("set")), (*index, Some("index"))],
        Inst::GetField { src, .. } => vec![(*src, None)],
        Inst::SetField { receiver, value, .. } => {
            vec![(*receiver, Some("self")), (*value, Some("value"))]
        }
        Inst::Push { array, value } => vec![(*array, Some("array")), (*value, Some("value"))],
        Inst::Insert { dict, value, .. } => vec![(*dict, Some("dict")), (*value, Some("value"))],
        Inst::Len(v)
        | Inst::ToFloat(v)
        | Inst::Sqrt(v)
        | Inst::Unwrap(v)
        | Inst::Raise(v)
        | Inst::IsRaised(v)
        | Inst::UnwrapRaised(v) => vec![(*v, None)],
        Inst::In(needle, haystack, _) => vec![(*needle, Some("needle")), (*haystack, Some("in"))],
        Inst::Format(parts) => parts
            .iter()
            .filter_map(|p| match p {
                FormatPart::Value(v) => Some((*v, None)),
                FormatPart::Literal(_) => None,
            })
            .collect(),
        Inst::MakeClosure { captures, .. } => {
            captures.iter().map(|c| (*c, Some("capture"))).collect()
        }
        Inst::Call { callee, args } => std::iter::once((*callee, Some("fn")))
            .chain(args.iter().map(|a| (*a, None)))
            .collect(),
        Inst::CallDirect { args, .. } | Inst::CallNative { args, .. } => {
            args.iter().map(|a| (*a, None)).collect()
        }
        Inst::NewInstance { fields, .. } => fields.iter().map(|f| (*f, None)).collect(),
        Inst::IsInstance { src, .. } => vec![(*src, None)],
    }
}

fn node_label(ir: &Ir, inst: &Inst, body_names: &HashMap<BodyId, &str>) -> String {
    match inst {
        Inst::Constant(c) => constant_label(ir, c),
        Inst::BinOp { op, .. } => op.to_string(),
        Inst::UnaryOp { op, .. } => op.to_string(),
        Inst::NewArray => "[ ]".to_string(),
        Inst::NewDict => "~{ }".to_string(),
        Inst::SetIndex { .. } => "[i] =".to_string(),
        Inst::GetIndex { .. } => "[i]".to_string(),
        Inst::GetField { slot, .. } => format!(".{slot}"),
        Inst::SetField { slot, .. } => format!(".{slot} ="),
        Inst::Push { .. } => "push".to_string(),
        Inst::Insert { key, .. } => format!("insert .{}", ir.str(*key)),
        Inst::Len(_) => "len".to_string(),
        Inst::ToFloat(_) => "to_float".to_string(),
        Inst::Sqrt(_) => "sqrt".to_string(),
        Inst::Unwrap(_) => "unwrap".to_string(),
        Inst::Raise(_) => "raise".to_string(),
        Inst::IsRaised(_) => "is_raised".to_string(),
        Inst::UnwrapRaised(_) => "unwrap_raised".to_string(),
        Inst::In(_, _, condition) => if *condition { "in" } else { "not in" }.to_string(),
        Inst::Format(_) => "format".to_string(),
        Inst::RefBody(body) => format!("fn {}", body_names.get(body).copied().unwrap_or("?")),
        Inst::MakeClosure { body, .. } => {
            format!("closure {}", body_names.get(body).copied().unwrap_or("?"))
        }
        Inst::Call { .. } => "call".to_string(),
        Inst::CallDirect { body, .. } => {
            body_names.get(body).copied().unwrap_or("?").to_string()
        }
        Inst::CallNative { id, .. } => format!("native #{}", id.index()),
        Inst::NewInstance { adt, .. } => {
            format!("new {}", ir.resolutions.adts[*adt].name)
        }
        Inst::IsInstance { adt, .. } => format!("is {}", ir.resolutions.adts[*adt].name),
        Inst::Panic => "panic".to_string(),
        // handled by the caller before this is ever reached.
        Inst::GetLocal(_)
        | Inst::SetLocal(..)
        | Inst::Return(_)
        | Inst::Jump { .. }
        | Inst::JumpIfFalse { .. }
        | Inst::Switch { .. }
        | Inst::Phi(_)
        | Inst::ForNext { .. } => unreachable!("control-flow/special inst reached node_label"),
    }
}

fn constant_label(ir: &Ir, c: &crate::Constant) -> String {
    match c {
        crate::Constant::Bool(b) => b.to_string(),
        crate::Constant::Int(i) => i.to_string(),
        crate::Constant::Float(f) => f.to_string(),
        crate::Constant::Str(s) => format!("{:?}", ir.str(*s)),
        crate::Constant::Null => "null".to_string(),
        crate::Constant::Array(items) => {
            format!("[{}]", items.iter().map(|i| constant_label(ir, i)).collect::<Vec<_>>().join(", "))
        }
    }
}
