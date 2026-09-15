//! A one-shot panel showing a `.darkly` file's layer tree, exercising `std::darkly::open`
//! (`crates/library/src/std_lib/darkly.rs`) the same way `dataflow_view` exercises
//! `Vm::function_dataflow`: run a small, self-contained mimas snippet and render its result,
//! rather than reaching into `mimas-library`'s (private) Rust types directly -- this crate only
//! ever sees mimas values, same as a real embedder would.
//!
//! The snippet itself walks the tree and builds one indented summary string (recursion, not a
//! mutable accumulator loop -- mimas's reassignment story wasn't worth the extra risk to check
//! for a one-off view like this), so the Rust side only needs `Vm::resolve_name_to_string`
//! (added earlier for exactly this kind of "read a value back out, unescaped" need) -- no new
//! `Inspect`-tree plumbing.

use bevy::ecs::resource::Resource;
use bevy::text::{Font, TextColor};
use bevy::ui::{
    BackgroundColor, BorderColor, BorderRadius, FlexDirection, Node, Overflow, OverflowAxis, UiRect,
    Val,
};
use bevy::utils::default;
use bevy_immediate::Imm;
use bevy_immediate::ui::{CapsUi, text::ImmUiText};

use crate::theme;

/// The example file this view always shows -- same hardcoding convention as
/// `default_script_path` for the debugged script itself.
const EXAMPLE_PATH: &str = "/Users/wow/code/dsa/darkly-p1.darkly";

const DARKLY_VIEW_SOURCE: &str = r#"
use std::darkly::*;

fn describe_children(children: [DarklyLayer], i: int, indent: str) -> str {
    if i >= children.len() {
        ""
    } else {
        describe(children[i], indent) + describe_children(children, i + 1, indent)
    }
}

fn describe(layer: DarklyLayer, indent: str) -> str {
    match layer {
        DarklyLayer::Group { id, name, visible, opacity, children } =>
            f"{indent}[group] {name} (visible={visible}, opacity={opacity})\n"
                + describe_children(children, 0, indent + "  "),
        DarklyLayer::Raster { id, name, visible, opacity, width, height } =>
            f"{indent}[raster] {name} {width}x{height} (visible={visible}, opacity={opacity})\n",
        DarklyLayer::Void { id, name, visible, void_type, transform, params_json } =>
            f"{indent}[void:{void_type}] {name} (visible={visible}, transform={transform})\n",
    }
}

let (doc, pixels) = open(PATH_PLACEHOLDER)!;
let summary = match doc {
    DarklyDocument::Document { name, width, height, root } =>
        f"{name} ({width}x{height})\n" + describe(root, ""),
};
"#;

/// `active` toggles the whole debugger body over to this panel (mirrors `DataflowView`). Loaded
/// lazily and cached forever (`loaded`) rather than per-frame like `DataflowView`'s cache-key
/// gating -- the example file and the snippet reading it are both fixed, so there's nothing to
/// ever invalidate the cache over.
#[derive(Resource, Default)]
pub(crate) struct DarklyView {
    pub(crate) active: bool,
    loaded: bool,
    text: Option<String>,
    error: Option<String>,
}

impl DarklyView {
    /// Compiles and runs `DARKLY_VIEW_SOURCE` once, caching either the resulting `summary`
    /// string or the compile/runtime error. Call once per frame before `render`, same convention
    /// as `DataflowView::refresh`.
    pub(crate) fn refresh(&mut self) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        let source = DARKLY_VIEW_SOURCE.replace("PATH_PLACEHOLDER", &format!("{EXAMPLE_PATH:?}"));
        match mimas::vm::Vm::execute(&source, mimas::library::std) {
            Ok(mut vm) => match vm.resolve_name_to_string("summary") {
                Some(Ok(text)) => self.text = Some(text),
                Some(Err(e)) => self.error = Some(e.to_string()),
                None => self.error = Some("summary was never bound".to_string()),
            },
            Err(e) => self.error = Some(e.to_string()),
        }
    }
}

fn panel_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(4.)),
            flex_grow: 1.0,
            min_height: Val::Px(0.),
            overflow: Overflow { x: OverflowAxis::Clip, y: OverflowAxis::Scroll },
            ..default()
        },
        BorderColor::all(theme::overlay0()),
        BackgroundColor(theme::surface0()),
    )
}

pub(crate) fn render(ui: &mut Imm<CapsUi>, font: bevy::asset::Handle<Font>, view: &DarklyView) {
    ui.ch_id("darkly_path")
        .on_spawn_insert({
            let font = font.clone();
            move || (TextColor(theme::subtext0()), crate::text_font(font))
        })
        .text(format!("std::darkly::open({EXAMPLE_PATH:?})"));

    ui.ch_id("darkly_panel").on_spawn_insert(panel_node).add(|ui| {
        if let Some(err) = &view.error {
            ui.ch_id("error")
                .on_spawn_insert({
                    let font = font.clone();
                    move || (TextColor(theme::red()), crate::text_font(font))
                })
                .text(err.clone());
            return;
        }
        let Some(text) = &view.text else {
            ui.ch_id("loading")
                .on_spawn_insert({
                    let font = font.clone();
                    move || (TextColor(theme::subtext0()), crate::text_font(font))
                })
                .text("(loading...)");
            return;
        };
        for (row, line) in text.lines().enumerate() {
            ui.ch_id(row)
                .on_spawn_insert({
                    let font = font.clone();
                    move || (TextColor(theme::text()), crate::text_font(font))
                })
                .text(line.to_string());
        }
    });
}
