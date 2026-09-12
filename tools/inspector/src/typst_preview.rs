//! Live Typst preview: renders the running program's linked-list-shaped data as a cetz tree,
//! compiled to PNG and shown in a pane inside the app itself (see `preview_panel_node` /
//! `ui_system` in `main.rs`) -- no separate browser needed.
//!
//! This is a first, deliberately narrow cut: it finds the first `Node`-shaped (3-field
//! instance) named local across the live frames and walks its field 0 ("next") as a chain,
//! labeling each stop with field 2 ("val"). It's hardcoded to `main.mim`'s
//! `Node { next, prev, val }` shape, not a general value-to-diagram renderer -- letting the
//! user author their own Typst template to control this is the deferred, harder problem this
//! is a first step toward.
//!
//! Compiling shells out to the `typst` CLI (must be on PATH, and needs Typst >= 0.14 for the
//! cetz package this uses) rather than embedding a Typst library, and runs synchronously on
//! whichever frame the visualized chain actually changed on -- a compile is ~100ms warm, so
//! this only ever costs a frame stall on an actual step, never every render.

use std::path::PathBuf;
use std::process::Command;

use mimas::vm::{Captured, FrameView};

#[derive(bevy::ecs::resource::Resource)]
pub struct TypstPreview {
    /// Where generated PNGs are written -- must be under Bevy's asset root (`assets/`) so
    /// `AssetServer::load` can see them; see `TypstPreview::new`.
    dir: PathBuf,
    /// The last `.typ` source actually compiled -- skip re-running `typst` when the visualized
    /// chain hasn't changed, which is most render frames.
    last_source: Option<String>,
    /// Bumped on every successful compile so each version gets a fresh filename: `AssetServer`
    /// caches by path, and a fixed filename would mean a reload never picks up the new bytes
    /// without fighting that cache directly. The two most recent files are kept (the current
    /// one plus whatever `ui_system` might still be mid-load on), older ones deleted.
    generation: u64,
}

impl TypstPreview {
    /// `dir` must be a subdirectory of the app's asset root (`assets/`) -- e.g.
    /// `assets/preview` -- so the paths `update` hands back are loadable via `AssetServer`.
    pub fn new(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
        Self {
            dir,
            last_source: None,
            generation: 0,
        }
    }

    /// Re-renders and recompiles if the visualized chain changed since the last call, returning
    /// the new PNG's path (relative to the asset root, ready for `AssetServer::load`) on success.
    /// `None` otherwise -- including when there's nothing chain-shaped to show, or the chain is
    /// unchanged since last call (most render frames), or the compile failed (logged to stderr).
    pub fn update(&mut self, frames: &[FrameView]) -> Option<String> {
        let source = build_scene(frames)?;
        if self.last_source.as_deref() == Some(source.as_str()) {
            return None;
        }

        let typ_path = self.dir.join("scene.typ");
        if let Err(e) = std::fs::write(&typ_path, &source) {
            eprintln!("typst preview: failed to write {}: {e}", typ_path.display());
            return None;
        }

        let generation = self.generation + 1;
        let png_name = format!("scene_{generation}.png");
        let png_path = self.dir.join(&png_name);
        let result = Command::new("typst")
            // high PPI relative to the small pane cetz actually draws into: the preview panel
            // displays this at up to ~360px wide (see `preview_panel_node`), and a diagram this
            // simple renders under 100px natively at a typical PPI -- upscaling that blurs it.
            .args(["compile", "--format", "png", "--ppi", "600"])
            .arg(&typ_path)
            .arg(&png_path)
            .output();
        match result {
            Ok(output) if output.status.success() => {
                self.last_source = Some(source);
                self.generation = generation;
                // keep this one and the previous one (in case a load is still in flight on
                // it), delete anything older.
                if generation >= 2 {
                    let stale = self.dir.join(format!("scene_{}.png", generation - 2));
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

/// Walks the first 3-field ("Node"-shaped) instance found among the live frames' named locals,
/// following field 0 as `next`, labeling each stop with field 2 as `val`. Stops at a non-node
/// (`null`, most commonly) or a `Captured::Cycle` (drawn as one extra leaf, not followed
/// further -- see `Val::capture`'s cycle detection in mimas-vm). Capped well short of that in
/// practice, but bounded regardless in case a chain is just very long.
fn build_scene(frames: &[FrameView]) -> Option<String> {
    let head = frames.iter().flat_map(|f| f.locals.iter()).find_map(|(_, v)| match v {
        Captured::Instance(fields) if fields.len() == 3 => Some(fields.clone()),
        _ => None,
    })?;

    let mut labels = Vec::new();
    let mut current = Captured::Instance(head);
    for _ in 0..256 {
        let Captured::Instance(fields) = &current else {
            break;
        };
        labels.push(fields.get(2).map(|v| v.to_string()).unwrap_or_default());
        match fields.first() {
            Some(next @ Captured::Instance(_)) => current = next.clone(),
            Some(Captured::Cycle) => {
                labels.push("\u{221e}".to_string());
                break;
            }
            _ => break,
        }
    }
    if labels.is_empty() {
        return None;
    }
    Some(render_typ(&labels))
}

fn render_typ(labels: &[String]) -> String {
    fn nest(labels: &[String]) -> String {
        match labels {
            [] => "[]".to_string(),
            [last] => format!("[{}]", typ_escape(last)),
            [first, rest @ ..] => format!("([{}], {})", typ_escape(first), nest(rest)),
        }
    }
    // `fill: none`: a transparent page, so the PNG blends into the preview pane's own
    // background instead of carrying its own white rectangle.
    format!(
        "#import \"@preview/cetz:0.5.2\": canvas, draw, tree\n\
         #set page(width: auto, height: auto, margin: .5cm, fill: none)\n\
         #canvas({{\n\
         \x20 import draw: *\n\
         \x20 set-style(content: (padding: 0.5em))\n\
         \x20 tree.tree({})\n\
         }})\n",
        nest(labels)
    )
}

/// Escapes the handful of characters that are special inside a Typst content block (`[...]`) --
/// values here are always `Captured::to_string()` output (numbers, quoted strings, `null`,
/// nested `{ .. }` instances), never arbitrary markup, but quoted strings can legitimately
/// contain `[`, `]`, or `\`.
fn typ_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('@', "\\@")
        .replace('#', "\\#")
}
