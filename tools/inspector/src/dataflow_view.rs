//! Circuit/block-diagram view of a single function's dataflow -- boxes for operations, wires for
//! values, no "this happens, then this happens" -- as an alternative to the step debugger. See
//! `mimas::vm::compile::function_dataflow`'s docs for exactly what's extracted and why it's
//! scoped to straight-line (no branch/loop) bodies.

use bevy::color::Color;
use bevy::ecs::observer::On;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::Query;
use bevy::input::mouse::MouseScrollUnit;
use bevy::picking::events::{Pointer, Scroll};
use bevy::text::{Font, TextColor};
use bevy::ui::{
    AlignItems, BackgroundColor, BorderColor, BorderRadius, JustifyContent, Node, Overflow,
    OverflowAxis, PositionType, ScrollPosition, UiRect, Val,
};
use bevy::utils::default;
use bevy_immediate::Imm;
use bevy_immediate::ui::{CapsUi, text::ImmUiText};

use mimas::vm::{DataflowEdge, DataflowGraph, NodeKind};

use crate::theme;

/// Which function's graph is showing, and the last extraction failure (a bad name, or a body
/// with real control flow), kept on screen next to the still-open name field rather than just
/// blanking the canvas.
///
/// `graph`/`cache_key` memoize the extraction: re-parsing and re-solving on every single render
/// frame (mimas compiles in single-digit milliseconds, but that's still ~60 times a second for
/// no reason) would be wasted work whenever neither the source nor the requested function name
/// has actually changed since the last frame. `cache_key` is `(function_name,
/// Session::generation)` -- `generation` already exists specifically to mean "the running
/// program was replaced wholesale" (see `Session`), which is exactly when a cached graph could
/// be stale.
#[derive(Resource)]
pub(crate) struct DataflowView {
    pub(crate) active: bool,
    pub(crate) function_name: String,
    pub(crate) error: Option<String>,
    graph: Option<DataflowGraph>,
    cache_key: Option<(String, u64)>,
}

impl Default for DataflowView {
    fn default() -> Self {
        Self {
            active: false,
            function_name: "typst_box".to_string(),
            error: None,
            graph: None,
            cache_key: None,
        }
    }
}

impl DataflowView {
    /// Re-extracts `function_name`'s graph from `(display_source, TYPST_MODULE_SOURCE)` if the
    /// name or `generation` has changed since the last call; otherwise a no-op. Call once per
    /// frame before `render`, not inside it -- keeps the (potentially fallible, always
    /// `&mut`-needing) refresh separate from the (infallible, `&self`-only) drawing.
    pub(crate) fn refresh(&mut self, display_source: &str, generation: u64) {
        let key = (self.function_name.clone(), generation);
        if self.cache_key.as_ref() == Some(&key) {
            return;
        }
        self.cache_key = Some(key);
        match mimas::vm::Vm::function_dataflow(
            &[("main", display_source), ("typst", crate::TYPST_MODULE_SOURCE)],
            mimas::library::std,
            &self.function_name,
        ) {
            Ok(graph) => {
                self.graph = Some(graph);
                self.error = None;
            }
            Err(e) => {
                self.graph = None;
                self.error = Some(e.to_string());
            }
        }
    }

    pub(crate) fn graph(&self) -> Option<&DataflowGraph> {
        self.graph.as_ref()
    }
}

const COLUMN_WIDTH: f32 = 220.0;
const ROW_HEIGHT: f32 = 70.0;
const NODE_WIDTH: f32 = 170.0;
const NODE_HEIGHT: f32 = 44.0;
const WIRE_THICKNESS: f32 = 2.0;

#[derive(Clone, Copy)]
struct NodePos {
    x: f32,
    y: f32,
}

/// Positions every node by column (0 for a parameter; otherwise one more than the deepest
/// producer feeding it) and row (order within its column, stable against `graph.nodes`' own
/// order). Valid without a separate topological sort: an edge's `from` is always the node that
/// *produced* the value `to` consumes, and `function_dataflow` only ever pushes a node after
/// every node it references, so `from < to` always holds and one left-to-right pass suffices.
fn layout(graph: &DataflowGraph) -> Vec<NodePos> {
    let n = graph.nodes.len();
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in &graph.edges {
        incoming[e.to].push(e.from);
    }
    let mut depth = vec![0u32; n];
    for i in 0..n {
        depth[i] = incoming[i].iter().map(|&src| depth[src] + 1).max().unwrap_or(0);
    }
    let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for (i, &d) in depth.iter().enumerate() {
        columns[d as usize].push(i);
    }
    let mut positions = vec![NodePos { x: 0.0, y: 0.0 }; n];
    for (col, members) in columns.iter().enumerate() {
        // center the column vertically against the tallest one, so a single param/out node
        // doesn't sit pinned to the top next to a column of five.
        let max_rows = columns.iter().map(|c| c.len()).max().unwrap_or(1);
        let offset = (max_rows.saturating_sub(members.len())) as f32 * ROW_HEIGHT / 2.0;
        for (row, &node_idx) in members.iter().enumerate() {
            positions[node_idx] =
                NodePos { x: col as f32 * COLUMN_WIDTH, y: offset + row as f32 * ROW_HEIGHT };
        }
    }
    positions
}

fn node_colors(kind: NodeKind) -> (Color, Color) {
    match kind {
        NodeKind::In => (theme::green(), theme::surface0()),
        NodeKind::Out => (theme::red(), theme::surface0()),
        NodeKind::Op => (theme::overlay1(), theme::surface1()),
    }
}

fn canvas_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            position_type: PositionType::Relative,
            flex_grow: 1.0,
            min_height: Val::Px(0.),
            overflow: Overflow {
                x: OverflowAxis::Clip,
                y: OverflowAxis::Clip,
            },
            ..default()
        },
        BorderColor::all(theme::overlay0()),
        BackgroundColor(theme::surface0()),
    )
}

/// Mouse wheel over the canvas pans it in both axes (unlike the source panel's vertical-only
/// scroll -- a graph can be wide as easily as it is tall). Same "no wiring by default" gap and
/// same fix as `on_source_scroll`.
fn on_dataflow_scroll(trigger: On<Pointer<Scroll>>, mut positions: Query<&mut ScrollPosition>) {
    let event = trigger.event();
    let Ok(mut pos) = positions.get_mut(event.entity) else {
        return;
    };
    let (dx, dy) = match event.unit {
        MouseScrollUnit::Line => (event.x * 20.0, event.y * 20.0),
        MouseScrollUnit::Pixel => (event.x, event.y),
    };
    pos.0.x = (pos.0.x - dx).max(0.0);
    pos.0.y = (pos.0.y - dy).max(0.0);
}

/// Renders `graph` into `ui` as a pannable canvas of boxes and right-angle wires. `text_font`/
/// `text_color` style every node's label.
pub(crate) fn render(
    ui: &mut Imm<CapsUi>,
    font: bevy::asset::Handle<Font>,
    graph: &DataflowGraph,
) {
    let positions = layout(graph);

    ui.ch_id("dataflow_canvas")
        .on_spawn_insert(canvas_node)
        .on_spawn_observe(on_dataflow_scroll)
        .add(|ui| {
            // wires first so node boxes paint over the corner where a wire meets a port, not
            // the reverse.
            for (i, edge) in graph.edges.iter().enumerate() {
                draw_edge(ui, i, edge, &positions);
            }
            for (i, node) in graph.nodes.iter().enumerate() {
                let pos = positions[i];
                let (border, background) = node_colors(node.kind);
                let label_font = font.clone();
                ui.ch_id(("df_node", i))
                    .on_spawn_insert(move || {
                        (
                            Node {
                                position_type: PositionType::Absolute,
                                left: Val::Px(pos.x),
                                top: Val::Px(pos.y),
                                width: Val::Px(NODE_WIDTH),
                                height: Val::Px(NODE_HEIGHT),
                                border: UiRect::all(Val::Px(1.5)),
                                border_radius: BorderRadius::all(Val::Px(6.)),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                padding: UiRect::axes(Val::Px(6.), Val::Px(2.)),
                                ..default()
                            },
                            BorderColor::all(border),
                            BackgroundColor(background),
                        )
                    })
                    .add(|ui| {
                        ui.ch()
                            .on_spawn_insert(move || (TextColor(theme::text()), crate::text_font(label_font)))
                            .text(node.label.clone());
                    });
            }
        });
}

fn draw_edge(ui: &mut Imm<CapsUi>, edge_idx: usize, edge: &DataflowEdge, positions: &[NodePos]) {
    let from = positions[edge.from];
    let to = positions[edge.to];
    let from_x = from.x + NODE_WIDTH;
    let from_y = from.y + NODE_HEIGHT / 2.0;
    let to_x = to.x;
    let to_y = to.y + NODE_HEIGHT / 2.0;
    let mid_x = (from_x + to_x) / 2.0;

    let wire = theme::overlay1();
    let seg = |ui: &mut Imm<CapsUi>, id: (&'static str, usize, u8), x: f32, y: f32, w: f32, h: f32| {
        ui.ch_id(id).on_spawn_insert(move || {
            (
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    top: Val::Px(y),
                    width: Val::Px(w.max(WIRE_THICKNESS)),
                    height: Val::Px(h.max(WIRE_THICKNESS)),
                    ..default()
                },
                BackgroundColor(wire),
            )
        });
    };

    // three-segment right-angle route: out of `from`'s right edge, across to the horizontal
    // midpoint, up/down to `to`'s row, then in to `to`'s left edge.
    seg(ui, ("df_edge_h1", edge_idx, 0), from_x, from_y - WIRE_THICKNESS / 2.0, mid_x - from_x, WIRE_THICKNESS);
    seg(
        ui,
        ("df_edge_v", edge_idx, 1),
        mid_x - WIRE_THICKNESS / 2.0,
        from_y.min(to_y),
        WIRE_THICKNESS,
        (to_y - from_y).abs(),
    );
    seg(ui, ("df_edge_h2", edge_idx, 2), mid_x, to_y - WIRE_THICKNESS / 2.0, to_x - mid_x, WIRE_THICKNESS);
}
