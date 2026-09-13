//! Unattended screenshot harness.
//!
//! Rather than capturing the host's screen (fragile, and liable to grab whatever else is on
//! screen), this drives the `Session` through a fixed script from inside the app itself and
//! saves each checkpoint via Bevy's own screenshot API, which reads the rendered frame back off
//! the GPU -- it works regardless of window focus. The app exits on its own once the script is
//! done, so this can run unattended: `cargo run -- --screenshots ./screenshots`.

use std::collections::VecDeque;
use std::path::PathBuf;

use bevy::app::{App, AppExit, PreUpdate};
use bevy::ecs::entity::Entity;
use bevy::ecs::message::MessageWriter;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::{Commands, NonSendMut, Query, ResMut};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::ui::widget::TextScroll;

use crate::{DataflowView, Editor, ManualEditorScroll, Session};

enum Action {
    Screenshot(&'static str),
    /// `Session::step_line`, called this many times.
    StepLineN(u32),
    /// `Session::step` (single-op, what the "Step" button calls), this many times -- unlike
    /// `StepLineN`, this never touches `line_steps`, reproducing how someone who only ever
    /// clicks "Step" (never the Down arrow) would leave `line_steps` stuck at 0.
    StepN(u32),
    /// `Session::rewind_one_line`, once.
    RewindLine,
    RunToEnd,
    Reset,
    ToggleInternals,
    /// Simulates clicking "Edit": opens the editor on the current `display_source`.
    StartEdit,
    /// Simulates typing into the open editor: a plain string replace against the buffer, so the
    /// script can target a substring instead of retyping the whole file.
    EditReplace(&'static str, &'static str),
    /// Simulates clicking "Apply" -- same call the button handler makes.
    ApplyEdit,
    /// Sets `ManualEditorScroll` to the open editor's entity at the given y -- what
    /// `on_editor_scroll` does on a real wheel event, minus computing the delta/entity from a
    /// `Pointer<Scroll>` this harness has no way to synthesize. Exercises the actual arbitration
    /// path (`arbitrate_editor_scroll`), not just a direct (and, it turns out, immediately
    /// discarded) `TextScroll` write -- see that system's doc comment for why the distinction
    /// matters.
    SetEditorScrollY(f32),
    /// Simulates clicking the "Dataflow"/"Debugger" toggle.
    ToggleDataflow,
    /// Simulates typing a new function name into the dataflow view's name field.
    SetDataflowFunction(&'static str),
    /// Skip this many frames before the next action -- bevy_ui needs a frame or two after a
    /// state change to re-measure text and settle layout, and a screenshot taken too eagerly
    /// would still show the stale frame.
    Wait(u32),
}

#[derive(Resource)]
struct AutoRunner {
    dir: PathBuf,
    actions: VecDeque<Action>,
    waiting: u32,
}

fn script() -> VecDeque<Action> {
    use Action::*;
    VecDeque::from([
        Screenshot("01_initial"),
        Wait(8),
        // Dataflow view checks: typst_box (default function) should render a small clean
        // circuit -- 3 params in, one `format` op, one `out`. Then switch to typst_arrow (also
        // straight-line, different arity) to confirm re-extraction on a name change works, then
        // to typeset (has a loop) to confirm the control-flow rejection surfaces as an error
        // instead of a blank or bogus diagram.
        ToggleDataflow,
        Wait(8),
        Screenshot("df_01_typst_box"),
        Wait(8),
        SetDataflowFunction("typst_arrow"),
        Wait(8),
        Screenshot("df_02_typst_arrow"),
        Wait(8),
        SetDataflowFunction("typeset"),
        Wait(8),
        Screenshot("df_03_typeset_rejected"),
        Wait(8),
        ToggleDataflow,
        Wait(8),
        Screenshot("df_04_back_to_debugger"),
        Wait(8),
        StepLineN(6),
        Wait(8),
        Screenshot("02_after_6_lines"),
        Wait(8),
        StepLineN(6),
        Wait(8),
        Screenshot("03_into_the_loop"),
        Wait(8),
        // Regression check for a real bug: apply a no-op edit here, mid-debug, with the preview
        // already showing a chain. The replay lands back on the exact same `steps` count the
        // live session had (nothing before this point changed), which used to fool the preview
        // cache into thinking nothing needed to be redrawn -- it kept showing the pre-edit
        // picture until the next real Step broke the coincidental tie. This should refresh
        // immediately, no extra Step, and the picture should look identical to 03 since the edit
        // only adds a comment.
        StartEdit,
        EditReplace("struct Node {", "struct Node { // a no-op edit, mid-debug"),
        ApplyEdit,
        Wait(8),
        Screenshot("03b_after_mid_debug_apply"),
        Wait(8),
        // Regression check for a second real bug: someone who only ever clicks "Step" (never
        // the Down arrow) leaves `line_steps` at 0 even though `steps` is well into the
        // program. `apply_edit` used to replay by `line_steps`, so Apply would silently discard
        // all that progress and land back at the very start. Reset first for a clean, larger
        // step count, then step by raw op only.
        Reset,
        StepN(40),
        Wait(8),
        Screenshot("03c_after_step_button_only"),
        Wait(8),
        // Bug 2 check, same setup: opening the editor here should land near the current
        // execution line, not snap to the top of the file.
        StartEdit,
        Wait(8),
        Screenshot("03d_editor_opened_at_current_line"),
        Wait(8),
        // Bug 1 check: apply a no-op edit and confirm the session is still ~40 ops in, not back
        // at 0 -- this is the actual scenario the user hit (Step-only, then Apply).
        EditReplace("struct Node {", "struct Node { // step-only apply check"),
        ApplyEdit,
        Wait(8),
        Screenshot("03e_after_apply_following_step_only"),
        Wait(8),
        RewindLine,
        Wait(8),
        Screenshot("04_after_rewind_one_line"),
        Wait(8),
        ToggleInternals,
        Wait(8),
        Screenshot("05_internals"),
        Wait(8),
        ToggleInternals,
        RunToEnd,
        Wait(8),
        Screenshot("06_finished"),
        Wait(8),
        Reset,
        Wait(8),
        Screenshot("07_reset"),
        Wait(8),
        StartEdit,
        Wait(8),
        Screenshot("08_editing"),
        Wait(8),
        // Scroll regression check: with the buffer's top visible (08_editing), force the
        // editor's internal scroll offset down and confirm the *rendered* text actually shifts
        // -- the render-side half of the mouse-wheel fix (`on_editor_scroll`), independent of
        // whether a real wheel event reaches it.
        SetEditorScrollY(300.0),
        Wait(8),
        Screenshot("08b_editor_scrolled"),
        Wait(8),
        EditReplace("struct Node {", "struct Node { // EDITED-MARKER"),
        Wait(8),
        Screenshot("09_editing_buffer_changed"),
        Wait(8),
        ApplyEdit,
        Wait(8),
        Screenshot("10_after_apply"),
        Wait(8),
        StartEdit,
        EditReplace("struct Node {", "struct Node ??totally broken syntax"),
        Wait(8),
        ApplyEdit,
        Wait(8),
        Screenshot("11_apply_error"),
        // give the last screenshot's async GPU readback + disk write time to finish before exit.
        Wait(15),
    ])
}

/// Reads `--screenshots <dir>` from argv; when present, registers the harness on `app` so it
/// drives itself through `script()` and exits when done instead of waiting on a human. `main`
/// should call this before `.run()` either way -- it's a no-op without the flag.
pub fn install(app: &mut App) {
    let Some(dir) = std::env::args()
        .skip_while(|a| a != "--screenshots")
        .nth(1)
    else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("--screenshots dir should be creatable");
    app.insert_resource(AutoRunner {
        dir,
        actions: script(),
        waiting: 0,
    });
    app.add_systems(PreUpdate, drive);
}

fn drive(
    mut commands: Commands,
    mut runner: ResMut<AutoRunner>,
    mut session: NonSendMut<Session>,
    mut editor: ResMut<Editor>,
    mut manual_scroll: ResMut<ManualEditorScroll>,
    mut dataflow_view: ResMut<DataflowView>,
    scrolls: Query<(Entity, &TextScroll)>,
    mut exit: MessageWriter<AppExit>,
) {
    if runner.waiting > 0 {
        runner.waiting -= 1;
        return;
    }
    let Some(action) = runner.actions.pop_front() else {
        exit.write(AppExit::Success);
        return;
    };
    match action {
        Action::Screenshot(name) => {
            let path = runner.dir.join(format!("{name}.png"));
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
        }
        Action::StepLineN(n) => {
            for _ in 0..n {
                session.step_line();
            }
        }
        Action::StepN(n) => {
            for _ in 0..n {
                session.step();
            }
        }
        Action::RewindLine => session.rewind_one_line(),
        Action::RunToEnd => session.run_to_end(),
        Action::Reset => session.reload(),
        Action::ToggleInternals => session.show_internals = !session.show_internals,
        Action::StartEdit => {
            editor.buffer = session.display_source.clone();
            editor.editing = true;
            editor.error = None;
            // Mirrors the real "Edit" button handler exactly, so this exercises the actual
            // open-at-current-line logic instead of a stand-in.
            editor.open_at_line = session
                .current_loc()
                .map(|(_, offset)| session.display_source[..offset].matches('\n').count())
                .unwrap_or(0);
        }
        Action::EditReplace(from, to) => {
            editor.buffer = editor.buffer.replace(from, to);
        }
        Action::ApplyEdit => match session.apply_edit(editor.buffer.clone()) {
            Ok(()) => {
                editor.editing = false;
                editor.error = None;
            }
            Err(e) => editor.error = Some(e),
        },
        Action::SetEditorScrollY(y) => {
            if let Some((entity, _)) = scrolls.iter().next() {
                manual_scroll.0 = Some((entity, y));
            }
        }
        Action::ToggleDataflow => dataflow_view.active = !dataflow_view.active,
        Action::SetDataflowFunction(name) => dataflow_view.function_name = name.to_string(),
        Action::Wait(n) => runner.waiting = n,
    }
}
