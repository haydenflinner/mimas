//! First working slice of the mimas debugger/visualizer: single-steps a `Vm` and mirrors its
//! call stack + registers into a `bevy_immediate` UI. See `Vm::debug_step` / `Vm::frames` in
//! `mimas-vm` for the introspection this reads.

use std::path::PathBuf;

use bevy::DefaultPlugins;
use bevy::app::{App, PluginGroup, PreUpdate, Startup, Update};
use bevy::asset::{AssetServer, Handle};
use bevy::color::Color;
use bevy::ecs::observer::On;
use bevy::ecs::system::{Commands, NonSendMut, Query, Res, ResMut};
use bevy::input::ButtonInput;
use bevy::input::keyboard::KeyCode;
use bevy::input::mouse::MouseScrollUnit;
use bevy::math::Vec2;
use bevy::picking::events::{Pointer, Scroll};
use bevy::prelude::Camera2d;
use bevy::text::{Font, FontSize, FontSource, TextColor, TextFont};
use bevy::ui::{
    AlignItems, BackgroundColor, BorderColor, BorderRadius, FlexDirection, Node, Overflow,
    OverflowAxis, ScrollPosition, UiRect, Val,
};
use bevy::utils::default;
use bevy::window::{Window, WindowPlugin};
use bevy_immediate::{
    BevyImmediatePlugin, ImmCtx,
    ui::{CapsUi, clicked::ImmUiClicked, text::ImmUiText},
};

use mimas::vm::Vm;

mod autoshot;
mod theme;

/// The single file every frame's source panel shows -- `Session::load` compiles exactly one
/// file named `"main"`, so `compile_files` always assigns it id 0.
const MAIN_FILE_ID: usize = 0;

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
}

impl Session {
    pub(crate) fn load(path: PathBuf) -> Self {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let vm = mimas::compile_files(&[("main", &source)]).expect("script should compile");
        Session {
            vm,
            path,
            finished: false,
            error: None,
            steps: 0,
            line_steps: 0,
            show_internals: false,
        }
    }

    /// Re-reads `self.path` into a fresh `Vm` -- what "Reset" actually does, since there's no
    /// `Vm::reset`. Preserves the "see deeper" toggle across the reload; nothing else.
    pub(crate) fn reload(&mut self) {
        let show_internals = self.show_internals;
        *self = Self::load(self.path.clone());
        self.show_internals = show_internals;
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

    /// `(file, byte offset)` of the innermost frame's active op, or `None` when there's no
    /// frame or its location is synthetic (compiler-generated, no real source span).
    fn current_loc(&mut self) -> Option<(usize, usize)> {
        // `current_position`, not `frames()`: this runs once per op inside `step_line`'s inner
        // loop, and `frames()` would re-capture every live register (expensively, if one cycles
        // through `Gc` handles -- see `Vm::current_position`'s doc) on every single one of them.
        let (_, _, loc) = self.vm.current_position()?;
        if loc.is_synthetic() {
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
        let start_line = start.and_then(|(file, offset)| {
            self.vm
                .source_text(file)
                .map(|text| (file, line_byte_range(&text, offset)))
        });
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

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.build().set(WindowPlugin {
        primary_window: Some(Window {
            title: "mimas inspector".into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(BevyImmediatePlugin::<CapsUi>::new())
    .insert_non_send(Session::load(script_path))
    .init_resource::<AutoScrollState>()
    .add_systems(Startup, (setup_camera, setup_font))
    .add_systems(PreUpdate, keyboard_system)
    .add_systems(Update, ui_system);

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
fn keyboard_system(keys: Res<ButtonInput<KeyCode>>, mut session: NonSendMut<Session>) {
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

fn ui_system(
    ctx: ImmCtx<CapsUi>,
    mut session: NonSendMut<Session>,
    mut auto_scroll: ResMut<AutoScrollState>,
    app_font: Res<AppFont>,
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

            let frames = session.vm.frames();
            let show_internals = session.show_internals;
            // the innermost (currently executing) frame's location -- `None` when there's no
            // real source span (before the first real op runs, or after the program ends), in
            // which case the source is still shown in full, just with nothing highlighted.
            let current_offset = frames
                .last()
                .filter(|f| !f.loc.is_synthetic())
                .map(|f| f.loc.span.start);

            // source panel: the whole program, current line highlighted. Always the full source
            // (not a window around the current line) so the highlight moves through a stable
            // view instead of the view itself jumping around. The panel scrolls (it's clipped to
            // whatever space is left in the window, see `source_panel_node`), and auto-scrolls
            // to keep the current line in view as the program steps.
            let text = session.vm.source_text(MAIN_FILE_ID);
            let lines = text
                .as_deref()
                .map(|t| all_lines(t, current_offset))
                .unwrap_or_default();
            let current_row = lines.iter().position(|&(_, _, is_current)| is_current);

            let mut source_panel = ui
                .ch_id("source")
                .on_spawn_insert(source_panel_node)
                .on_spawn_observe(on_source_scroll);
            // only re-apply the auto-scroll when the highlighted line actually moved -- not
            // every frame, or it would fight a manual scroll back to a different line.
            let row_changed = current_row.is_some() && current_row != auto_scroll.last_row;
            if let Some(row) = current_row {
                let target_y = ((row as f32 - SOURCE_ROWS_ABOVE) * SOURCE_ROW_HEIGHT_PX).max(0.0);
                source_panel = source_panel.on_change_insert(row_changed, move || {
                    ScrollPosition(Vec2::new(0.0, target_y))
                });
            }
            auto_scroll.last_row = current_row;
            source_panel.add(|ui| {
                if lines.is_empty() {
                    ui.ch()
                        .on_spawn_insert(|| dim_text_style(font.clone()))
                        .text("(no source loaded)");
                }
                for (line_no, line_text, is_current) in lines {
                    ui.ch_id(line_no)
                        .text(format!("{line_no:>4} | {line_text}"))
                        .on_change_insert(true, || line_row_style(is_current, font.clone()));
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
                        ui.ch_id("header").on_spawn_insert(|| text_style(font.clone())).text(title);

                        for (name, val) in &frame.locals {
                            ui.ch_id(name.as_str())
                                .on_spawn_insert(|| text_style(font.clone()))
                                .text(format!("{name} = {val}"));
                        }

                        if show_internals {
                            ui.ch_id("chunk_ip")
                                .on_spawn_insert(|| dim_text_style(font.clone()))
                                .text(format!("chunk #{} @ip={}", frame.chunk.index(), frame.ip));
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
