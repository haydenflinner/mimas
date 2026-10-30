//! Live Typst preview: renders the running program's own idea of how to typeset itself,
//! compiled to PNG and shown in a pane inside the app itself (see `preview_panel_node` /
//! `ui_system` in `main.rs`) -- no separate browser needed.
//!
//! Two pieces make this the mimas program's job, not this crate's:
//!
//! - **What to draw** comes from calling a live value's own pact method (see `Node`'s
//!   `impl Typeset for Node` in `main.mim`) via `Vm::call_method_on_first_instance` -- this
//!   module doesn't know what a `Node` is, or that there's a linked list at all. It gets back a
//!   Typst expression as a plain string (mimas building it with ordinary `f"..."` interpolation;
//!   see `Ctx`'s docs if that ever gets unwieldy enough to want native helpers instead).
//! - **How to draw it** is `main.typ`, alongside the script (`main.mim`) itself -- a plain file
//!   the user is meant to edit, not something this module generates. The mimas-produced string
//!   is handed in via `typst compile --input scene=<...>`, and `main.typ` picks it up as
//!   `sys.inputs.scene` and `eval`s it into the actual Typst value its template wants (a cetz
//!   tree.tree argument, in the shipped template -- but that's the template's business).
//!
//! Compiling shells out to the `typst` CLI (must be on PATH, and needs Typst >= 0.14 for the
//! cetz package `main.typ` uses) rather than embedding a Typst library, and runs synchronously
//! on whichever frame the rendered string actually changed on -- a compile is ~100ms warm, so
//! this only ever costs a frame stall on an actual step, never every render.

use std::path::PathBuf;
use std::process::Command;

#[derive(bevy::ecs::resource::Resource)]
pub struct TypstPreview {
    /// The root template, expected alongside the script (`main.mim`'s directory) -- see
    /// `TypstPreview::new`.
    typ_path: PathBuf,
    /// Where generated PNGs are written -- must be under Bevy's asset root (`assets/`) so
    /// `AssetServer::load` can see them.
    out_dir: PathBuf,
    /// The last scene string actually compiled -- skip re-running `typst` when it hasn't
    /// changed, which is most render frames.
    last_scene: Option<String>,
    /// Bumped on every successful compile so each version gets a fresh filename: `AssetServer`
    /// caches by path, and a fixed filename would mean a reload never picks up the new bytes
    /// without fighting that cache directly. The two most recent files are kept (the current
    /// one plus whatever `ui_system` might still be mid-load on), older ones deleted.
    generation: u64,
}

impl TypstPreview {
    /// `typ_path` is the root template to compile (expected to be `main.typ` next to the
    /// script). `out_dir` must be a subdirectory of the app's asset root (`assets/`) -- e.g.
    /// `assets/preview` -- so the paths `update` hands back are loadable via `AssetServer`.
    pub fn new(typ_path: PathBuf, out_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&out_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_dir.display()));
        Self {
            typ_path,
            out_dir,
            last_scene: None,
            generation: 0,
        }
    }

    /// Recompiles `main.typ` with `scene` as its `sys.inputs.scene` if `scene` changed since the
    /// last call, returning the new PNG's path (relative to the asset root, ready for
    /// `AssetServer::load`) on success. `None` otherwise -- including when `scene` is unchanged
    /// since last call (most render frames), or the compile failed (logged to stderr).
    pub fn update(&mut self, scene: &str) -> Option<String> {
        if self.last_scene.as_deref() == Some(scene) {
            return None;
        }
        if !self.typ_path.is_file() {
            eprintln!(
                "typst preview: {} doesn't exist -- nothing to compile",
                self.typ_path.display()
            );
            return None;
        }

        let generation = self.generation + 1;
        let png_name = format!("scene_{generation}.png");
        let png_path = self.out_dir.join(&png_name);
        let result = Command::new("typst")
            .args(["compile", "--input"])
            .arg(format!("scene={scene}"))
            // high PPI relative to the small pane cetz actually draws into: the preview panel
            // displays this at up to ~360px wide (see `preview_panel_node`), and a diagram this
            // simple renders under 100px natively at a typical PPI -- upscaling that blurs it.
            .args(["--format", "png", "--ppi", "600"])
            .arg(&self.typ_path)
            .arg(&png_path)
            .output();
        match result {
            Ok(output) if output.status.success() => {
                self.last_scene = Some(scene.to_string());
                self.generation = generation;
                // keep this one and the previous one (in case a load is still in flight on
                // it), delete anything older.
                if generation >= 2 {
                    let stale = self.out_dir.join(format!("scene_{}.png", generation - 2));
                    let _ = std::fs::remove_file(stale);
                }
                Some(format!("preview/{png_name}"))
            }
            Ok(output) => {
                eprintln!(
                    "typst preview: compile failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                None
            }
            Err(e) => {
                eprintln!(
                    "typst preview: couldn't run `typst` ({e}) -- is it installed and on PATH?"
                );
                None
            }
        }
    }
}
