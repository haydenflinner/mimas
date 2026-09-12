//! Live Typst preview: renders the running program's linked-list-shaped data as a cetz tree,
//! compiled to SVG, and shown in a browser tab that polls the file for changes.
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

const INDEX_HTML: &str = r#"<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>mimas live preview</title>
<style>
  body { margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center; background: #f5efe6; }
  img { max-width: 95vw; max-height: 95vh; }
  #empty { font: 16px monospace; color: #576869; }
</style>
</head>
<body>
<img id="scene" src="scene.svg" onerror="this.style.display='none'; document.getElementById('empty').style.display='block';">
<div id="empty" style="display:none;">(nothing to show yet)</div>
<script>
setInterval(() => {
  const img = document.getElementById('scene');
  img.style.display = '';
  document.getElementById('empty').style.display = 'none';
  img.src = 'scene.svg?t=' + Date.now();
}, 400);
</script>
</body>
</html>
"#;

#[derive(bevy::ecs::resource::Resource)]
pub struct TypstPreview {
    dir: PathBuf,
    /// The last `.typ` source actually compiled -- skip re-running `typst` when the visualized
    /// chain hasn't changed, which is most render frames.
    last_source: Option<String>,
}

impl TypstPreview {
    pub fn new(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
        std::fs::write(dir.join("index.html"), INDEX_HTML)
            .expect("preview/index.html should be writable");
        Self {
            dir,
            last_source: None,
        }
    }

    pub fn index_html_path(&self) -> PathBuf {
        self.dir.join("index.html")
    }

    /// Re-renders and recompiles if the visualized chain changed since the last call. Cheap
    /// no-op otherwise (including when there's nothing chain-shaped to show).
    pub fn update(&mut self, frames: &[FrameView]) {
        let Some(source) = build_scene(frames) else {
            return;
        };
        if self.last_source.as_deref() == Some(source.as_str()) {
            return;
        }

        let typ_path = self.dir.join("scene.typ");
        if let Err(e) = std::fs::write(&typ_path, &source) {
            eprintln!("typst preview: failed to write {}: {e}", typ_path.display());
            return;
        }

        let svg_path = self.dir.join("scene.svg");
        match Command::new("typst")
            .args(["compile", "--format", "svg"])
            .arg(&typ_path)
            .arg(&svg_path)
            .output()
        {
            Ok(output) if output.status.success() => {
                self.last_source = Some(source);
            }
            Ok(output) => {
                eprintln!(
                    "typst preview: compile failed:\n{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => {
                eprintln!(
                    "typst preview: couldn't run `typst` ({e}) -- is it installed and on PATH?"
                );
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
    format!(
        "#import \"@preview/cetz:0.5.2\": canvas, draw, tree\n\
         #set page(width: auto, height: auto, margin: .5cm)\n\
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
