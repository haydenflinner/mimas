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
use bevy::ecs::message::MessageWriter;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::{Commands, NonSendMut, ResMut};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

use crate::Session;

enum Action {
    Screenshot(&'static str),
    /// `Session::step_line`, called this many times.
    StepLineN(u32),
    /// `Session::rewind_one_line`, once.
    RewindLine,
    RunToEnd,
    Reset,
    ToggleInternals,
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
        StepLineN(6),
        Wait(8),
        Screenshot("02_after_6_lines"),
        Wait(8),
        StepLineN(6),
        Wait(8),
        Screenshot("03_into_the_loop"),
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
        Action::RewindLine => session.rewind_one_line(),
        Action::RunToEnd => session.run_to_end(),
        Action::Reset => session.reload(),
        Action::ToggleInternals => session.show_internals = !session.show_internals,
        Action::Wait(n) => runner.waiting = n,
    }
}
