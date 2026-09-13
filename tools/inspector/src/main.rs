//! First working slice of the mimas debugger/visualizer: single-steps a `Vm` and mirrors its
//! call stack + registers into a `bevy_immediate` UI. See `Vm::debug_step` / `Vm::frames` in
//! `mimas-vm` for the introspection this reads.

use std::path::PathBuf;

use bevy::DefaultPlugins;
use bevy::app::{App, PluginGroup, PostUpdate, PreUpdate, Startup, Update};
use bevy::asset::{AssetServer, Handle};
use bevy::color::Color;
use bevy::ecs::entity::Entity;
use bevy::ecs::observer::On;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Local, NonSendMut, Query, Res, ResMut};
use bevy::image::Image;
use bevy::input::ButtonInput;
use bevy::input::keyboard::KeyCode;
use bevy::input::mouse::MouseScrollUnit;
use bevy::math::Vec2;
use bevy::picking::events::{Pointer, Scroll};
use bevy::prelude::Camera2d;
use bevy::text::{
    EditableText, EditableTextGeneration, Font, FontSize, FontSource, TextColor, TextCursorStyle,
    TextEdit, TextFont, TextLayoutInfo,
};
use bevy::ui::widget::{ImageNode, TextScroll};
use bevy::ui::{
    AlignItems, BackgroundColor, BorderColor, BorderRadius, ComputedNode, FlexDirection,
    JustifyContent, Node, Overflow, OverflowAxis, ScrollPosition, UiRect, Val,
};
use bevy::ui_widgets::EditableTextInputPlugin;
use bevy::utils::default;
use bevy::window::{Window, WindowPlugin};
use bevy_immediate::{
    BevyImmediatePlugin, ImmCtx,
    ui::{CapsUi, clicked::ImmUiClicked, text::ImmUiText, text_input::ImmUiTextInput},
};

use mimas::vm::{Captured, Inspect, Vm};

mod autoshot;
mod theme;
mod typst_preview;

use typst_preview::TypstPreview;

/// Built-in `typst` module, compiled alongside every script as a second file (not a prefix of
/// it) so scripts pick it up with a plain `use typst::*;` -- see `~/code/dsa/scripts/main.mim`,
/// which implements `Typeset` for its own `Node` using this module's pact and helpers.
/// `impl Typeset for Node` itself has to stay in the script: this module can't know about
/// `Node`, and mimas has no prelude/auto-import (every cross-module item needs an explicit
/// `use`), so there's no way to fold the impl in here too.
const TYPST_MODULE_SOURCE: &str = r#"module typst;

pub pact Typeset {
    fn typeset(self) -> str;
}

pub fn typst_box(x: float, name: str, val: int) -> str {
    f"content(({x},0),name:\"{name}\",frame:\"rect\",[{val}]);"
}

pub fn typst_arrow(from: str, to: str, color: str) -> str {
    f"line(\"{from}\",\"{to}\",stroke:rgb(\"{color}\"),mark:(end:\"stealth\",fill:rgb(\"{color}\")));"
}
"#;

fn default_script_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME should be set");
    PathBuf::from(home).join("code/dsa/scripts/main.mim")
}

// Not `#[derive(Resource)]`: `mimas::vm::Vm` wraps a `gc_arena::Arena`, which holds raw pointers
// for its GC bookkeeping and is neither `Send` nor `Sync` -- a `Resource` must be both. Bevy's
// escape hatch for exactly this shape of type is a *non-send* resource, pinned to the main thread
// and accessed via `NonSend`/`NonSendMut` instead of `Res`/`ResMut`.
pub(crate) struct Session {
    vm: Vm,
    path: PathBuf,
    /// The file exactly as it is on disk -- what the source panel actually shows, and (since
    /// it's compiled unmodified as file 0, with `typst` a separate file alongside it) also
    /// exactly what every byte offset `Vm` reports is relative to, as long as it's for file 0.
    display_source: String,
    finished: bool,
    error: Option<String>,
    steps: u64,
    /// How many completed `step_line` calls brought us to the current point -- what the Up
    /// arrow replays back down to. Not a real rewind (see `rewind_one_line`): the only way back
    /// is forward again, from a fresh `Vm`.
    line_steps: u32,
    /// Coder-facing view is the default: function names, named locals, current source line.
    /// This reveals the VM-shaped view underneath (chunk ids, raw register windows).
    pub(crate) show_internals: bool,
    /// Bumped every time this `Session` is replaced wholesale (`reload`, `apply_edit`). `steps`
    /// alone isn't a safe key for "has anything changed" -- a fresh replay after Reset or Apply
    /// can land back on the exact same `steps` count the old (entirely different) `Vm` had, which
    /// left the Typst preview showing a stale picture from before the edit until the next real
    /// `Step` broke the coincidental tie. `ui_system` keys its preview cache off
    /// `(generation, steps)` instead, so a fresh `Vm` always counts as "changed".
    pub(crate) generation: u64,
}

impl Session {
    /// Compiles `display_source` as `main` (file 0, byte-for-byte -- no prefix, no offset
    /// translation needed to line up with what's on disk) alongside the built-in `typst`
    /// module. Shared by `load` (a bad file at startup is a real bug, so it panics) and
    /// `apply_edit` (a bad in-progress edit is normal, so it reports the error instead).
    fn compile(
        display_source: String,
        path: PathBuf,
        show_internals: bool,
        generation: u64,
    ) -> Result<Self, String> {
        let vm = mimas::compile_files(&[("main", &display_source), ("typst", TYPST_MODULE_SOURCE)])
            .map_err(|e| e.to_string())?;
        Ok(Session {
            vm,
            path,
            display_source,
            finished: false,
            error: None,
            steps: 0,
            line_steps: 0,
            show_internals,
            generation,
        })
    }

    pub(crate) fn load(path: PathBuf) -> Self {
        let display_source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        Self::compile(display_source, path, false, 0)
            .unwrap_or_else(|e| panic!("script should compile: {e}"))
    }

    /// Recompiles `new_source` and fast-forwards a fresh session back to where this one was, by
    /// replaying `step` `steps` times -- the same trick `rewind_one_line` uses `step_line` for.
    /// Raw op-steps, not `step_line`/`line_steps`: the "Step" button only advances `steps` (one
    /// op), so anyone stepping op-by-op rather than by line would have `line_steps` stuck at
    /// whatever it last was (often 0) -- replaying that many *lines* landed back near the very
    /// start regardless of how far `steps` actually was. `steps` is the one counter every
    /// stepping path (Step, Down-arrow, Run to end) advances consistently, so it's the only
    /// reliable "how far in" this session actually is.
    ///
    /// Simpler and safer than patching a live `Vm`'s bytecode/heap in place (no register layouts
    /// to reconcile, no stale `Gc` pointers to worry about), and it can absorb *any* edit -- a
    /// reordered declaration, a changed struct's fields, not just a tweaked function body --
    /// because it's not trying to reuse anything from the old compile. Cheap because this
    /// program runs end-to-end in microseconds; redoing that work against the edited source on
    /// every "Apply" is not something worth optimizing away.
    pub(crate) fn apply_edit(&mut self, new_source: String) -> Result<(), String> {
        let steps = self.steps;
        let line_steps = self.line_steps;
        let generation = self.generation.wrapping_add(1);
        let mut replayed =
            Self::compile(new_source, self.path.clone(), self.show_internals, generation)?;
        for _ in 0..steps {
            replayed.step();
        }
        // `step` alone never updates `line_steps` (only `step_line` does) -- carry the old
        // count across so the "N so far" display and the Up-arrow rewind stay consistent with
        // wherever the user actually was, rather than silently resetting to 0.
        replayed.line_steps = line_steps;
        *self = replayed;
        Ok(())
    }

    /// Re-reads `self.path` into a fresh `Vm` -- what "Reset" actually does, since there's no
    /// `Vm::reset`. Preserves the "see deeper" toggle across the reload; nothing else.
    pub(crate) fn reload(&mut self) {
        let show_internals = self.show_internals;
        let generation = self.generation.wrapping_add(1);
        let display_source = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", self.path.display()));
        *self = Self::compile(display_source, self.path.clone(), show_internals, generation)
            .unwrap_or_else(|e| panic!("script should compile: {e}"));
    }

    pub(crate) fn step(&mut self) {
        if self.finished || self.error.is_some() {
            return;
        }
        match self.vm.debug_step() {
            Ok(done) => {
                self.steps += 1;
                self.finished = done;
            }
            Err(err) => self.error = Some(format!("{err}")),
        }
    }

    pub(crate) fn run_to_end(&mut self) {
        // safety net against a runaway script -- this is a debugger, not a runtime.
        for _ in 0..1_000_000 {
            if self.finished || self.error.is_some() {
                break;
            }
            self.step();
        }
    }

    /// `(file, byte offset into `display_source`)` of the innermost frame's active op, or `None`
    /// when there's no frame, its location is synthetic (compiler-generated, no real source
    /// span), or it's not in `main` (file 0) at all -- e.g. mid-step inside the `typst` module,
    /// which has no counterpart in `display_source` to report an offset into.
    fn current_loc(&mut self) -> Option<(usize, usize)> {
        // `current_position`, not `frames()`: this runs once per op inside `step_line`'s inner
        // loop, and `frames()` would re-capture every live register (expensively, if one cycles
        // through `Gc` handles -- see `Vm::current_position`'s doc) on every single one of them.
        let (_, _, loc) = self.vm.current_position()?;
        if loc.is_synthetic() || loc.file_id != 0 {
            return None;
        }
        Some((loc.file_id, loc.span.start))
    }

    /// Runs ops until the innermost frame's active source line changes (or the program ends) --
    /// the Down-arrow "next line" step. The byte range of the starting line is computed once up
    /// front, not re-scanned on every op, so this stays cheap even stepping across a long loop
    /// body that keeps returning to the same line.
    pub(crate) fn step_line(&mut self) {
        if self.finished || self.error.is_some() {
            return;
        }
        let start = self.current_loc();
        let start_line = start
            .map(|(file, offset)| (file, line_byte_range(&self.display_source, offset)));
        for _ in 0..1_000_000 {
            self.step();
            if self.finished || self.error.is_some() {
                break;
            }
            let now = self.current_loc();
            let moved = match (&start_line, now) {
                (Some((file, range)), Some((now_file, now_offset))) => {
                    now_file != *file || !range.contains(&now_offset)
                }
                (None, Some(_)) | (Some(_), None) => true,
                (None, None) => false,
            };
            if moved {
                break;
            }
        }
        self.line_steps += 1;
    }

    /// The Up-arrow "back up one line" step. There's no real rewind (mimas's `Vm` doesn't
    /// support undoing an op), so this reloads from scratch and replays `step_line` back up to
    /// one step short of where it was -- same destination, gotten to the only way possible.
    pub(crate) fn rewind_one_line(&mut self) {
        if self.line_steps == 0 {
            return;
        }
        let target = self.line_steps - 1;
        self.reload();
        for _ in 0..target {
            self.step_line();
        }
    }
}

/// The `[start, end)` byte range of the line containing `offset` in `source`. Computed once per
/// `step_line` call (not per op) so repeated stepping through a loop body stays O(ops), not
/// O(ops * source length).
fn line_byte_range(source: &str, offset: usize) -> std::ops::Range<usize> {
    let offset = offset.min(source.len());
    let start = source[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = source[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(source.len());
    start..end
}

fn main() {
    let script_path = std::env::args()
        .skip_while(|a| a != "--script")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_script_path);

    // main.typ is expected alongside the script itself -- a plain file the user edits, not
    // something this app generates. PNG output goes under `assets/` (Bevy's asset root) rather
    // than next to the script, so `AssetServer::load` can actually see it.
    let typst_preview = TypstPreview::new(
        script_path.with_file_name("main.typ"),
        PathBuf::from("assets/preview"),
    );

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.build().set(WindowPlugin {
        primary_window: Some(Window {
            title: "mimas inspector".into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(BevyImmediatePlugin::<CapsUi>::new());
    if !app.is_plugin_added::<EditableTextInputPlugin>() {
        app.add_plugins(EditableTextInputPlugin);
    }
    app.insert_non_send(Session::load(script_path))
        .insert_resource(typst_preview)
        .init_resource::<AutoScrollState>()
        .init_resource::<CurrentPreview>()
        .init_resource::<Editor>()
        .init_resource::<ManualEditorScroll>()
        .add_systems(Startup, (setup_camera, setup_font))
        .add_systems(PreUpdate, keyboard_system)
        .add_systems(Update, ui_system)
        // must run after bevy_ui's own auto-scroll-to-cursor system -- see
        // `arbitrate_editor_scroll`'s doc comment for why.
        .add_systems(
            PostUpdate,
            arbitrate_editor_scroll.after(bevy::ui::widget::scroll_editable_text),
        );

    // `--screenshots <dir>`: drive the session through a fixed script, saving a PNG at each
    // checkpoint via Bevy's own screenshot API (reads the rendered frame back off the GPU, so
    // it works headless of window focus) and exiting on its own when done. See `autoshot`.
    autoshot::install(&mut app);

    app.run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// The one font every text row in the UI renders with -- see `text_style`/`dim_text_style`/
/// `line_row_style`, which all take a `Handle<Font>` for exactly this.
#[derive(bevy::ecs::resource::Resource)]
struct AppFont(Handle<Font>);

/// Loads from `assets/MapleMono-TTF/` (relative to the crate root, Bevy's default asset root)
/// -- a monospace font actually meant to be read as code, unlike the UI's built-in default.
fn setup_font(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("MapleMono-TTF/MapleMono-Regular.ttf");
    commands.insert_resource(AppFont(font));
}

/// Down arrow: run to the next source line. Up arrow: there's no real rewind, so this replays
/// from a fresh `Vm` back up to one line-step short of here -- see `Session::rewind_one_line`.
fn keyboard_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut session: NonSendMut<Session>,
    editor: Res<Editor>,
) {
    // the arrow keys double as text-editing keys (moving the cursor, extending a selection) --
    // while the source panel is an open `EditableText`, they belong to it, not to stepping.
    if editor.editing {
        return;
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        session.step_line();
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        session.rewind_one_line();
    }
}

fn root_node() -> (Node, BackgroundColor) {
    (
        Node {
            flex_direction: FlexDirection::Column,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            padding: UiRect::all(Val::Px(12.)),
            row_gap: Val::Px(10.),
            ..default()
        },
        BackgroundColor(theme::base()),
    )
}

fn row_node() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        column_gap: Val::Px(10.),
        align_items: AlignItems::Center,
        ..default()
    }
}

fn button_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            padding: UiRect::axes(Val::Px(10.), Val::Px(6.)),
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(4.)),
            ..default()
        },
        BorderColor::all(theme::overlay1()),
        BackgroundColor(theme::surface1()),
    )
}

/// Maple Mono renders noticeably larger than the UI's built-in default did at the same nominal
/// size (wider advance width, taller line box) -- this is smaller than `TextFont::default()`'s
/// 20px specifically to compensate, so buttons stop wrapping and `SOURCE_ROW_HEIGHT_PX` (tuned
/// against this value) stays roughly right.
const UI_FONT_SIZE: f32 = 15.0;

fn text_font(font: Handle<Font>) -> TextFont {
    TextFont {
        font: FontSource::Handle(font),
        font_size: FontSize::Px(UI_FONT_SIZE),
        ..default()
    }
}

fn text_style(font: Handle<Font>) -> (TextColor, TextFont) {
    (TextColor(theme::text()), text_font(font))
}

fn dim_text_style(font: Handle<Font>) -> (TextColor, TextFont) {
    (TextColor(theme::subtext0()), text_font(font))
}

fn error_text_style(font: Handle<Font>) -> (TextColor, TextFont) {
    (TextColor(theme::red()), text_font(font))
}

/// Source-line row style. Unified into one bundle type (rather than two differently-shaped
/// styles picked between) so it can go through `on_change_insert` every frame -- which line is
/// "current" changes as the program steps, on the same set of row entities.
fn line_row_style(
    is_current: bool,
    font: Handle<Font>,
) -> (TextColor, BackgroundColor, TextFont) {
    let text_font = text_font(font);
    if is_current {
        (
            TextColor(theme::text()),
            BackgroundColor(theme::yellow()),
            text_font,
        )
    } else {
        (
            TextColor(theme::subtext1()),
            BackgroundColor(Color::NONE),
            text_font,
        )
    }
}

fn source_panel_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(4.)),
            min_width: Val::Px(420.),
            // fill whatever space is left under the header/controls row, rather than growing to
            // fit the whole program -- `min_height: 0` is the flexbox gotcha that actually lets
            // it shrink below its content's natural size, which is what makes `Scroll` kick in
            // instead of just pushing the frames row off the bottom of the window.
            flex_grow: 1.0,
            min_height: Val::Px(0.),
            overflow: Overflow {
                x: OverflowAxis::Clip,
                y: OverflowAxis::Scroll,
            },
            ..default()
        },
        BorderColor::all(theme::overlay0()),
        BackgroundColor(theme::surface0()),
    )
}

/// Approximate row height for the source panel's text rows at `UI_FONT_SIZE`, used only to
/// scroll the current line into view (and to scale wheel-scroll deltas) -- not pixel-exact,
/// just close enough that auto-scroll and the wheel both track it.
const SOURCE_ROW_HEIGHT_PX: f32 = 18.0;
/// How many rows of context to keep above the current line when auto-scrolling to it.
const SOURCE_ROWS_ABOVE: f32 = 6.0;

/// The whole program, one row per line (1-based line number, line text, whether it contains
/// `current_offset`). Always the full source, not a window around the current line -- so the
/// highlight moves through a stable, always-visible piece of code instead of the display itself
/// jumping around under it.
fn all_lines(source: &str, current_offset: Option<usize>) -> Vec<(usize, String, bool)> {
    let mut pos = 0usize;
    source
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let end = pos + line.len() + 1;
            let is_current = current_offset.is_some_and(|o| o >= pos && o < end);
            pos = end;
            (i + 1, line.to_string(), is_current)
        })
        .collect()
}

/// Flattens an `Inspect` tree into indented, name-labeled display lines, depth-first -- one row
/// per line, the same convention `all_lines` uses for the source panel. This is the structural
/// inspector: unlike `Captured`'s flat positional dump (`{ 391, null, ∞ }`), every field is
/// labeled with its declared name and every instance with its struct name (`Node { val: 391,
/// .. }`), all the way down. `MAX_DEPTH` isn't the cycle guard's job -- `Inspect::Cycle` already
/// terminates a real cycle -- it just keeps one big-but-acyclic value from producing unbounded
/// rows.
fn inspect_lines(name: &str, value: &Inspect, depth: usize, out: &mut Vec<String>) {
    const MAX_DEPTH: usize = 8;
    let indent = "  ".repeat(depth);
    if depth > MAX_DEPTH {
        out.push(format!("{indent}{name} = .."));
        return;
    }
    match value {
        Inspect::Null => out.push(format!("{indent}{name} = null")),
        Inspect::Bool(b) => out.push(format!("{indent}{name} = {b}")),
        Inspect::Int(i) => out.push(format!("{indent}{name} = {i}")),
        Inspect::Float(f) => out.push(format!("{indent}{name} = {f}")),
        Inspect::Str(s) => out.push(format!("{indent}{name} = {s:?}")),
        Inspect::Fn(body) => out.push(format!("{indent}{name} = fn({})", body.index())),
        Inspect::Raised(s) => out.push(format!("{indent}{name} = raised({s:?})")),
        Inspect::Other => out.push(format!("{indent}{name} = <other>")),
        Inspect::Cycle => out.push(format!("{indent}{name} = \u{221e}")),
        Inspect::Array(items) if items.is_empty() => out.push(format!("{indent}{name} = []")),
        Inspect::Array(items) => {
            out.push(format!("{indent}{name} = ["));
            for (i, v) in items.iter().enumerate() {
                inspect_lines(&format!("[{i}]"), v, depth + 1, out);
            }
            out.push(format!("{indent}]"));
        }
        Inspect::Dict(entries) if entries.is_empty() => out.push(format!("{indent}{name} = ~{{}}")),
        Inspect::Dict(entries) => {
            out.push(format!("{indent}{name} = ~{{"));
            for (k, v) in entries {
                inspect_lines(k, v, depth + 1, out);
            }
            out.push(format!("{indent}}}"));
        }
        Inspect::Instance { type_name, fields } if fields.is_empty() => {
            out.push(format!("{indent}{name}: {type_name} {{}}"));
        }
        Inspect::Instance { type_name, fields } => {
            out.push(format!("{indent}{name}: {type_name} {{"));
            for (k, v) in fields {
                inspect_lines(k, v, depth + 1, out);
            }
            out.push(format!("{indent}}}"));
        }
    }
}

fn frame_panel_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            row_gap: Val::Px(4.),
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(4.)),
            overflow: Overflow::clip(),
            min_width: Val::Px(160.),
            ..default()
        },
        BorderColor::all(theme::overlay0()),
        BackgroundColor(theme::surface0()),
    )
}

/// The most recently compiled Typst preview image, if any -- `None` until the first frame that
/// finds something chain-shaped to show (see `TypstPreview::update`).
#[derive(bevy::ecs::resource::Resource, Default)]
struct CurrentPreview {
    handle: Option<Handle<Image>>,
    /// `(Session::generation, Session::steps)` as of the last frame we actually asked the
    /// program to typeset itself. Calling `call_method_on_first_instance` runs real mimas
    /// bytecode (a full injected call, recursing through the whole chain) -- worth doing on an
    /// actual step, not on every one of the ~60 render frames a step sits idle for. `generation`
    /// is part of the key (not just `steps`) so a fresh `Vm` from Reset/Apply always counts as
    /// changed, even when it happens to replay back to the same step count the old one had.
    last_seen: Option<(u64, u64)>,
}

fn body_row_node() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        column_gap: Val::Px(10.),
        flex_grow: 1.0,
        min_height: Val::Px(0.),
        ..default()
    }
}

fn main_column_node() -> Node {
    Node {
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(10.),
        flex_grow: 1.0,
        min_width: Val::Px(0.),
        min_height: Val::Px(0.),
        ..default()
    }
}

fn preview_panel_node() -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(4.)),
            width: Val::Px(380.),
            min_height: Val::Px(0.),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            overflow: Overflow::clip(),
            ..default()
        },
        BorderColor::all(theme::overlay0()),
        BackgroundColor(theme::surface0()),
    )
}

/// Live-editing state for the source panel. `editing` toggles the panel between its normal
/// read-only display and an `EditableText` bound to `buffer`; `buffer` only becomes real once
/// "Apply" recompiles it (see `Session::apply_edit`), so half-finished edits never touch the
/// running session. `error` holds the last compile failure so it stays on screen (next to the
/// still-open editor) until the user fixes it and retries, rather than flashing by for one frame.
#[derive(bevy::ecs::resource::Resource, Default)]
pub(crate) struct Editor {
    pub(crate) editing: bool,
    pub(crate) buffer: String,
    pub(crate) error: Option<String>,
    /// 0-indexed source line the debugger was on when "Edit" was clicked -- where the freshly
    /// spawned `EditableText` positions its cursor (see the `editor_text` bundle in `ui_system`),
    /// so opening the editor lands the view on the code you were just looking at instead of
    /// snapping to the top of the file.
    pub(crate) open_at_line: usize,
}

/// Tracks the last line the source panel auto-scrolled to, so `ui_system` only re-applies
/// `ScrollPosition` when that line actually changes -- otherwise re-inserting it every frame
/// (needed so the *initial* follow-the-highlight scroll works at all) would fight a manual
/// scroll back to a different line on every single frame.
#[derive(bevy::ecs::resource::Resource, Default)]
struct AutoScrollState {
    last_row: Option<usize>,
}

/// Mouse wheel over the source panel: bevy_ui's `Overflow::Scroll` + `ScrollPosition` don't come
/// wired to the wheel on their own (see `bevy_immediate`'s own scrollarea example, which hand-
/// rolls the same thing) -- this is that wiring, scoped to just the source panel entity via
/// `on_spawn_observe`.
fn on_source_scroll(trigger: On<Pointer<Scroll>>, mut positions: Query<&mut ScrollPosition>) {
    let event = trigger.event();
    let Ok(mut pos) = positions.get_mut(event.entity) else {
        return;
    };
    let delta = match event.unit {
        MouseScrollUnit::Line => event.y * SOURCE_ROW_HEIGHT_PX,
        MouseScrollUnit::Pixel => event.y,
    };
    pos.0.y = (pos.0.y - delta).max(0.0);
}

/// A wheel-driven scroll position for whichever `EditableText` entity last received wheel input
/// (see `on_editor_scroll`), tracked outside of `TextScroll` itself -- see
/// `arbitrate_editor_scroll` for why a direct write to `TextScroll` doesn't stick.
#[derive(bevy::ecs::resource::Resource, Default)]
pub(crate) struct ManualEditorScroll(pub(crate) Option<(Entity, f32)>);

/// `on_editor_scroll` (the wheel handler) only *proposes* a scroll position by updating
/// `ManualEditorScroll`; this is what actually applies it to `TextScroll`, and it has to run
/// after bevy_ui's own `scroll_editable_text` to win.
///
/// Why: `update_editable_text_layout` (bevy_ui, `PostLayout`) takes `&mut EditableText` and
/// `&mut EditableTextGeneration` on every `EditableText` entity every single frame -- needed for
/// cursor blinking, which has nothing to do with the text actually changing. That unconditional
/// `&mut` access satisfies `scroll_editable_text`'s "did anything change" gate every frame
/// regardless, so it recomputes and overwrites `TextScroll` back to wherever the cursor is on
/// every frame, permanently discarding a plain external write to `TextScroll` -- confirmed by
/// direct observation: a manual write was gone by the very next frame with no edits made at all.
///
/// The fix doesn't need to know anything about bevy_ui's scroll math: `EditableTextGeneration`'s
/// *value* (not its perpetually-true ECS change flag) only actually changes on a real edit or
/// cursor move, so comparing it frame-to-frame tells us whether `scroll_editable_text`'s output
/// this frame was a real, meaningful auto-follow (in which case it wins, and we resync our
/// tracked position to it) or just it re-asserting the same thing out of habit (in which case we
/// reassert our own last wheel-scrolled position over it instead).
fn arbitrate_editor_scroll(
    mut last_generation: Local<Option<(Entity, EditableTextGeneration)>>,
    mut manual_scroll: ResMut<ManualEditorScroll>,
    mut editors: Query<(Entity, &EditableTextGeneration, &mut TextScroll)>,
) {
    for (entity, &generation, mut scroll) in &mut editors {
        let real_change = *last_generation != Some((entity, generation));
        *last_generation = Some((entity, generation));
        if real_change {
            manual_scroll.0 = Some((entity, scroll.0.y));
        } else if let Some((e, y)) = manual_scroll.0
            && e == entity
        {
            scroll.0.y = y;
        }
    }
}

fn on_editor_scroll(
    trigger: On<Pointer<Scroll>>,
    mut manual_scroll: ResMut<ManualEditorScroll>,
    editors: Query<(&TextScroll, &ComputedNode, &TextLayoutInfo)>,
) {
    let event = trigger.event();
    let Ok((scroll, node, info)) = editors.get(event.entity) else {
        return;
    };
    let delta = match event.unit {
        MouseScrollUnit::Line => event.y * SOURCE_ROW_HEIGHT_PX,
        MouseScrollUnit::Pixel => event.y,
    };
    let view_height = node.content_box().size().y;
    let max_scroll_y = (info.size.y - view_height).max(0.0);
    let current = manual_scroll
        .0
        .filter(|&(e, _)| e == event.entity)
        .map_or(scroll.0.y, |(_, y)| y);
    manual_scroll.0 = Some((event.entity, (current - delta).clamp(0.0, max_scroll_y)));
}

fn ui_system(
    ctx: ImmCtx<CapsUi>,
    mut session: NonSendMut<Session>,
    mut auto_scroll: ResMut<AutoScrollState>,
    mut typst_preview: ResMut<TypstPreview>,
    mut current_preview: ResMut<CurrentPreview>,
    mut editor: ResMut<Editor>,
    app_font: Res<AppFont>,
    asset_server: Res<AssetServer>,
) {
    let font = app_font.0.clone();
    ctx.build_immediate_root("inspector_root")
        .ch()
        .on_spawn_insert(root_node)
        .add(|ui| {
            ui.ch()
                .on_spawn_insert(|| text_style(font.clone()))
                .text(format!("mimas inspector -- {}", session.path.display()));

            // controls row
            ui.ch().on_spawn_insert(row_node).add(|ui| {
                let mut step_btn = ui
                    .ch_id("step")
                    .on_spawn_insert(button_node)
                    .add(|ui| {
                        ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Step");
                    });
                if step_btn.clicked() {
                    session.step();
                }

                let mut run_btn = ui
                    .ch_id("run")
                    .on_spawn_insert(button_node)
                    .add(|ui| {
                        ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Run to end");
                    });
                if run_btn.clicked() {
                    session.run_to_end();
                }

                let mut reset_btn = ui
                    .ch_id("reset")
                    .on_spawn_insert(button_node)
                    .add(|ui| {
                        ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Reset");
                    });
                if reset_btn.clicked() {
                    session.reload();
                }

                let mut internals_btn = ui
                    .ch_id("internals")
                    .on_spawn_insert(button_node)
                    .add(|ui| {
                        let label = if session.show_internals {
                            "Hide internals"
                        } else {
                            "See deeper"
                        };
                        ui.ch().on_spawn_insert(|| text_style(font.clone())).text(label);
                    });
                if internals_btn.clicked() {
                    session.show_internals = !session.show_internals;
                }

                if editor.editing {
                    let mut apply_btn = ui
                        .ch_id("apply")
                        .on_spawn_insert(button_node)
                        .add(|ui| {
                            ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Apply");
                        });
                    if apply_btn.clicked() {
                        match session.apply_edit(editor.buffer.clone()) {
                            Ok(()) => {
                                editor.editing = false;
                                editor.error = None;
                            }
                            Err(e) => editor.error = Some(e),
                        }
                    }

                    let mut cancel_btn = ui
                        .ch_id("cancel")
                        .on_spawn_insert(button_node)
                        .add(|ui| {
                            ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Cancel");
                        });
                    if cancel_btn.clicked() {
                        editor.editing = false;
                        editor.error = None;
                    }
                } else {
                    let mut edit_btn = ui
                        .ch_id("edit")
                        .on_spawn_insert(button_node)
                        .add(|ui| {
                            ui.ch().on_spawn_insert(|| text_style(font.clone())).text("Edit");
                        });
                    if edit_btn.clicked() {
                        editor.buffer = session.display_source.clone();
                        editor.editing = true;
                        editor.error = None;
                        // Best-effort: land the view at the current execution line instead of
                        // the top of the file. Counts *source* lines (newlines before the
                        // current offset), not wrapped visual rows, so a long line that wraps
                        // earlier in the file can land this a few rows early -- the same
                        // approximation the read-only panel's own auto-scroll already makes.
                        editor.open_at_line = session
                            .current_loc()
                            .map(|(_, offset)| session.display_source[..offset].matches('\n').count())
                            .unwrap_or(0);
                    }
                }

                let status = status_text(&session);
                ui.ch_id("status")
                    .text(status)
                    .on_change_insert(true, || status_style(&session, font.clone()));

                ui.ch_id("keys")
                    .on_spawn_insert(|| dim_text_style(font.clone()))
                    .text(format!(
                        "  |  \u{2193} next line   \u{2191} back one line ({} so far)",
                        session.line_steps
                    ));
            });

            if let Some(err) = editor.error.clone() {
                ui.ch_id("edit_error")
                    .on_spawn_insert(|| error_text_style(font.clone()))
                    .text(err);
            }

            let frames = session.vm.frames();
            let mut image_changed = false;
            // ask the program itself how to typeset itself (see `Node`'s `impl Typeset` in
            // main.mim) -- this crate doesn't know what a `Node` is, or that there even is one.
            // Only on an actual step: this runs real mimas bytecode, not a free data read.
            if current_preview.last_seen != Some((session.generation, session.steps)) {
                current_preview.last_seen = Some((session.generation, session.steps));
                let scene = session
                    .vm
                    .call_method_on_first_instance("typeset")
                    .and_then(|c| match c {
                        Captured::Str(s) => Some(s),
                        _ => None,
                    });
                if let Some(scene) = scene
                    && let Some(path) = typst_preview.update(&scene)
                {
                    current_preview.handle = Some(asset_server.load(path));
                    image_changed = true;
                }
            }
            let show_internals = session.show_internals;
            // the innermost (currently executing) frame's location, as a byte offset into
            // `display_source` -- `None` when there's no real source span (before the first
            // real op runs, or after the program ends), or the frame is inside the `typst`
            // module rather than `main` (file 0), in which case the source is still shown in
            // full, just nothing highlighted. Mirrors `Session::current_loc`, which does the
            // same for `current_position()` instead of `frames()`.
            let current_offset = frames
                .last()
                .filter(|f| !f.loc.is_synthetic() && f.loc.file_id == 0)
                .map(|f| f.loc.span.start);

            // body: the existing debugger panels on the left, the Typst preview pane on the
            // right -- a vertically split pane inside the app itself, no separate browser tab.
            ui.ch_id("body").on_spawn_insert(body_row_node).add(|ui| {
                ui.ch_id("main").on_spawn_insert(main_column_node).add(|ui| {
                    // source panel: the whole program, current line highlighted. Always the
                    // full source (not a window around the current line) so the highlight
                    // moves through a stable view instead of the view itself jumping around.
                    // The panel scrolls (it's clipped to whatever space is left, see
                    // `source_panel_node`), and auto-scrolls to keep the current line in view.
                    let lines = all_lines(&session.display_source, current_offset);
                    let current_row = lines.iter().position(|&(_, _, is_current)| is_current);

                    let mut source_panel = ui
                        .ch_id("source")
                        .on_spawn_insert(source_panel_node)
                        .on_spawn_observe(on_source_scroll);
                    // only re-apply the auto-scroll when the highlighted line actually moved --
                    // not every frame, or it would fight a manual scroll to a different line.
                    let row_changed = current_row.is_some() && current_row != auto_scroll.last_row;
                    if let Some(row) = current_row {
                        let target_y =
                            ((row as f32 - SOURCE_ROWS_ABOVE) * SOURCE_ROW_HEIGHT_PX).max(0.0);
                        source_panel = source_panel.on_change_insert(row_changed, move || {
                            ScrollPosition(Vec2::new(0.0, target_y))
                        });
                    }
                    auto_scroll.last_row = current_row;
                    source_panel.add(|ui| {
                        if editor.editing {
                            // a raw editable buffer, not a per-line list -- there's no "current
                            // line" highlight to preserve while editing, and the underlying
                            // `EditableText` widget already handles multi-line text/newlines on
                            // its own.
                            ui.ch_id("editor_text")
                                .on_spawn_insert({
                                    let font = font.clone();
                                    let open_at_line = editor.open_at_line;
                                    move || {
                                        (
                                            Node {
                                                width: Val::Percent(100.0),
                                                flex_grow: 1.0,
                                                overflow: Overflow::clip(),
                                                ..default()
                                            },
                                            text_style(font),
                                            EditableText {
                                                // Queued rather than set directly on the
                                                // `PlainEditor`: `apply_text_edits` (bevy_text)
                                                // processes these *after* `input_text`'s own
                                                // initial `set_text`, so by the time these
                                                // `Down` moves run, there's real, laid-out
                                                // content for the cursor to move down through.
                                                pending_edits: vec![TextEdit::Down(false); open_at_line],
                                                ..default()
                                            },
                                            TextScroll::default(),
                                            TextCursorStyle {
                                                color: theme::text(),
                                                selection_color: theme::overlay0(),
                                                unfocused_selection_color: theme::overlay0(),
                                                selected_text_color: None,
                                            },
                                        )
                                    }
                                })
                                .on_spawn_observe(on_editor_scroll)
                                .input_text(&mut editor.buffer);
                        } else if lines.is_empty() {
                            ui.ch()
                                .on_spawn_insert(|| dim_text_style(font.clone()))
                                .text("(no source loaded)");
                        } else {
                            for (line_no, line_text, is_current) in lines {
                                ui.ch_id(line_no)
                                    .text(format!("{line_no:>4} | {line_text}"))
                                    .on_change_insert(true, || line_row_style(is_current, font.clone()));
                            }
                        }
                    });

                    // call stack, oldest frame first (matches ThreadState.frames order)
                    ui.ch_id("frames").on_spawn_insert(row_node).add(|ui| {
                        for (depth, frame) in frames.iter().enumerate() {
                            ui.ch_id(depth).on_spawn_insert(frame_panel_node).add(|ui| {
                                let title = if depth == 0 {
                                    "script".to_string()
                                } else {
                                    frame
                                        .function_name
                                        .clone()
                                        .unwrap_or_else(|| "<closure>".to_string())
                                };
                                ui.ch_id("header")
                                    .on_spawn_insert(|| text_style(font.clone()))
                                    .text(title);

                                // structural inspector: every local, name-labeled all the way
                                // down (struct name + field names, not `frame.locals`'s flat
                                // positional dump) -- what makes a value inspectable by default
                                // instead of only the ones a script bothers to implement a pact
                                // rendering for.
                                let mut inspect_rows = Vec::new();
                                for (name, val) in &frame.locals_inspect {
                                    inspect_lines(name, val, 0, &mut inspect_rows);
                                }
                                for (row, line) in inspect_rows.into_iter().enumerate() {
                                    ui.ch_id(("local", row))
                                        .on_spawn_insert(|| text_style(font.clone()))
                                        .text(line);
                                }

                                if show_internals {
                                    ui.ch_id("chunk_ip")
                                        .on_spawn_insert(|| dim_text_style(font.clone()))
                                        .text(format!(
                                            "chunk #{} @ip={}",
                                            frame.chunk.index(),
                                            frame.ip
                                        ));
                                    for (reg, val) in frame.registers.iter().enumerate() {
                                        ui.ch_id(("reg", reg))
                                            .on_spawn_insert(|| dim_text_style(font.clone()))
                                            .text(format!("r{reg} = {val}"));
                                    }
                                }
                            });
                        }
                    });
                });

                // Typst preview pane: a live cetz-rendered tree of the running program's linked
                // Node chain, updated whenever `typst_preview.update` produces a new image.
                ui.ch_id("preview").on_spawn_insert(preview_panel_node).add(|ui| {
                    match current_preview.handle.clone() {
                        Some(handle) => {
                            ui.ch_id("image")
                                .on_spawn_insert(|| Node {
                                    width: Val::Percent(100.0),
                                    ..default()
                                })
                                .on_change_insert(image_changed, move || ImageNode::new(handle.clone()));
                        }
                        None => {
                            ui.ch_id("empty")
                                .on_spawn_insert(|| dim_text_style(font.clone()))
                                .text("(no chain to show yet)");
                        }
                    }
                });
            });
        });
}

fn status_text(session: &Session) -> String {
    if let Some(err) = &session.error {
        format!("error: {err}")
    } else if session.finished {
        format!("finished ({} ops)", session.steps)
    } else {
        format!("running ({} ops so far)", session.steps)
    }
}

/// Color-codes `status_text`'s three states, so error/finished/running read at a glance instead
/// of only through the words. Re-applied every frame (`on_change_insert(true, ...)`, matching
/// `line_row_style`): which state we're in can change on the same status-text entity.
fn status_style(session: &Session, font: Handle<Font>) -> (TextColor, TextFont) {
    let color = if session.error.is_some() {
        theme::red()
    } else if session.finished {
        theme::green()
    } else {
        theme::text()
    };
    (TextColor(color), text_font(font))
}
