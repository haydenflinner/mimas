//! A one-shot panel showing a `.darkly` file's layer tree plus its rendered composite picture,
//! exercising `std::darkly::open` (`crates/library/src/std_lib/darkly.rs`) the same way
//! `dataflow_view` exercises `Vm::function_dataflow`: run a small, self-contained mimas snippet
//! and render its result, rather than reaching into `mimas-library`'s (private) Rust types
//! directly -- this crate only ever sees mimas values, same as a real embedder would.
//!
//! The snippet itself walks the tree and builds one indented summary string (recursion, not a
//! mutable accumulator loop -- mimas's reassignment story wasn't worth the extra risk to check
//! for a one-off view like this), so the Rust side only needs `Vm::resolve_name_to_string`
//! (added earlier for exactly this kind of "read a value back out, unescaped" need) -- no new
//! `Inspect`-tree plumbing.
//!
//! The composite picture is a different story: `std::darkly` doesn't parse `composite.png` at
//! all (it's a top-level manifest field, not part of the layer tree, and a game loading layers
//! to drive gameplay has no use for a pre-flattened preview) -- so this module reads it straight
//! out of the same zip file itself, the same way `typst_preview.rs` writes a generated PNG under
//! `assets/preview/` for `AssetServer` to pick up. Kept deliberately separate from `std::darkly`:
//! this is an inspector-only convenience, not a reason to teach the mimas-facing module about
//! PNG decoding it otherwise has no need for.

use std::io::Read;

use bevy::asset::{AssetServer, Handle};
use bevy::ecs::resource::Resource;
use bevy::image::Image;
use bevy::text::{Font, TextColor};
use bevy::ui::widget::ImageNode;
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

/// Where the composite PNG gets copied to -- must be under Bevy's asset root (`assets/`) so
/// `AssetServer::load` can see it. Shares `typst_preview`'s `assets/preview/` directory (already
/// created by `TypstPreview::new` at startup) rather than a second asset subdirectory, with its
/// own filename so the two never collide.
const COMPOSITE_ASSET_PATH: &str = "preview/darkly_composite.png";

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
    /// `Some` once the composite PNG has been copied out and handed to `AssetServer` --
    /// independent of `text`/`error` (the layer tree can fail to parse while the composite still
    /// loads fine, or vice versa; each is shown or not on its own).
    composite: Option<Handle<Image>>,
}

impl DarklyView {
    /// Compiles and runs `DARKLY_VIEW_SOURCE` once (caching either the resulting `summary`
    /// string or the compile/runtime error), and separately copies `composite.png` out of the
    /// same file for `AssetServer` to load. Call once per frame before `render`, same convention
    /// as `DataflowView::refresh`.
    pub(crate) fn refresh(&mut self, asset_server: &AssetServer) {
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

        match extract_composite(EXAMPLE_PATH) {
            Ok(()) => self.composite = Some(asset_server.load(COMPOSITE_ASSET_PATH)),
            Err(e) => eprintln!("darkly view: couldn't extract composite.png: {e}"),
        }
    }
}

/// Copies the `composite.png` entry out of `path`'s zip container to `assets/`
/// + `COMPOSITE_ASSET_PATH`, creating the destination directory if needed. Plain zip + file I/O,
/// no mimas involved -- see the module doc for why this stays out of `std::darkly`.
fn extract_composite(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut entry = archive.by_name("composite.png").map_err(|e| e.to_string())?;
    let mut png_bytes = Vec::new();
    entry.read_to_end(&mut png_bytes).map_err(|e| e.to_string())?;
    drop(entry);

    let dest = std::path::Path::new("assets").join(COMPOSITE_ASSET_PATH);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&dest, &png_bytes).map_err(|e| e.to_string())
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
        if let Some(handle) = view.composite.clone() {
            // fixed width, auto height: preserves the composite's own aspect ratio (1920x1080
            // here) instead of showing it at native pixel size, which would blow well past the
            // panel.
            ui.ch_id("composite")
                .on_spawn_insert(|| Node {
                    width: Val::Px(480.),
                    height: Val::Auto,
                    margin: UiRect::bottom(Val::Px(10.)),
                    ..default()
                })
                .on_spawn_insert(move || ImageNode::new(handle.clone()));
        }

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
