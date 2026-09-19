//! `std::darkly` -- reads a `.darkly` paint-app save file (a zip container: `manifest.json` plus
//! one raw pixel buffer per raster/mask layer) into a plain mimas layer tree, for a game to pull
//! assets and placement data out of ("this layer is the player sprite, this group is the
//! background"). File-only integration: this module never depends on the `darkly` crate itself
//! (a GPU paint *engine* -- wgpu/vello/kurbo/parley, no feature flag to opt out of any of it) or
//! talks to a running darkly process. `manifest.json` is plain, self-describing JSON with no
//! darkly Rust types serialized into it, and raster/mask pixel data is stored as raw, uncompressed
//! `width * height * channels` bytes with no encoding step -- both straightforward to read
//! independently.
//!
//! Not parsed yet, deliberately out of scope for this first pass: vector layers (their path
//! geometry serializes via `kurbo::BezPath`'s own format, not darkly's -- would need `kurbo`,
//! itself a reasonably light, GPU-free dependency, but not worth adding until something needs
//! it), masks/selections (referenced by id from a raster layer's `modifiers` list, materialized
//! separately in the manifest), and the `recording/` stroke-history directory (editor undo data,
//! never relevant at runtime). `"divider"` nodes (a pure editor concept -- the canvas/screen-space
//! boundary marker) are silently skipped rather than surfaced. Any other/future layer kind raises
//! rather than silently dropping content.

use std::{collections::HashMap, io::Read};

use macros::native;
use vm::{Ctx, MimasEnum, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_adt::<vm::conversion::DarklyImageTy>();
    api.add_adt::<DarklyLayer>();
    api.add_adt::<DarklyDocument>();
    {
        let mut m = api.module("std::darkly");
        m.add(open);
    }
    api.add_method(width);
    api.add_method(height);
}

#[native]
fn width<'gc>(image: vm::DarklyImage<'gc>) -> i64 {
    image.0.0.width as i64
}

#[native]
fn height<'gc>(image: vm::DarklyImage<'gc>) -> i64 {
    image.0.0.height as i64
}

#[derive(MimasEnum)]
enum DarklyDocument {
    Document {
        name: String,
        width: i64,
        height: i64,
        root: DarklyLayer,
    },
}

#[derive(MimasEnum)]
enum DarklyLayer {
    Group {
        id: i64,
        name: String,
        visible: bool,
        opacity: f64,
        children: Vec<DarklyLayer>,
    },
    Raster {
        /// Look this id up (stringified) in `open`'s second return value to get this layer's
        /// actual pixels. Not a field alongside it here: `#[derive(MimasEnum)]` can't be derived
        /// on a generic type (see `mimas-macros`' `derive.rs`), so a `DarklyLayer` value must be
        /// `'static` -- it can never hold a `Gc<'gc, ..>`-backed `DarklyImage` handle, at any
        /// field. Keeping pixels in a side table also means printing/inspecting/pattern-matching
        /// the tree never drags multi-megabyte buffers along uninvited.
        id: i64,
        name: String,
        visible: bool,
        opacity: f64,
        width: i64,
        height: i64,
    },
    Void {
        id: i64,
        name: String,
        visible: bool,
        void_type: String,
        /// The raw `[a, b, c, d, e, f]` 2D affine matrix -- `mode` (its own field on disk) is
        /// currently always `"Basic"`, so it isn't surfaced separately yet.
        transform: Vec<f64>,
        /// `void_type`-specific parameters, re-serialized as JSON text rather than modeled
        /// structurally (the shape varies per `void_type`) -- parse with `std::parse::from_json`
        /// if you need to look inside.
        params_json: String,
    },
}

/// Reads a `.darkly` file into `(document, pixels)`: `document` is the plain, printable layer
/// tree (names, kinds, visibility, opacity, void transforms/params); `pixels` maps each raster
/// layer's stringified `id` to its decoded `DarklyImage` -- separate because a `DarklyLayer`
/// value can't hold one directly (see the comment on `DarklyLayer::Raster`'s `id` field).
#[native]
fn open<'gc>(
    ctx: Ctx<'gc>,
    path: &str,
) -> Raisable<(DarklyDocument, HashMap<String, vm::DarklyImage<'gc>>)> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return Raisable::Raised(format!("darkly::open: {e}")),
    };
    let mut archive = match zip::ZipArchive::new(std::io::Cursor::new(bytes)) {
        Ok(a) => a,
        Err(e) => return Raisable::Raised(format!("darkly::open: not a valid .darkly file: {e}")),
    };
    let manifest: RawManifest = {
        let mut entry = match archive.by_name("manifest.json") {
            Ok(e) => e,
            Err(e) => {
                return Raisable::Raised(format!("darkly::open: missing manifest.json: {e}"));
            }
        };
        let mut text = String::new();
        if let Err(e) = entry.read_to_string(&mut text) {
            return Raisable::Raised(format!("darkly::open: reading manifest.json: {e}"));
        }
        drop(entry);
        match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => return Raisable::Raised(format!("darkly::open: parsing manifest.json: {e}")),
        }
    };

    let nodes: HashMap<i64, RawNode> = manifest.nodes.into_iter().map(|n| (n.id, n)).collect();
    let mut pixels = HashMap::new();
    let root = match build_layer(ctx, &mut archive, &nodes, manifest.root, &mut pixels) {
        Ok(Some(layer)) => layer,
        Ok(None) => {
            return Raisable::Raised("darkly::open: root node is a divider, not a layer".into());
        }
        Err(e) => return Raisable::Raised(e),
    };
    let doc = DarklyDocument::Document {
        name: manifest.name,
        width: manifest.canvas.width,
        height: manifest.canvas.height,
        root,
    };
    Raisable::Ok((doc, pixels))
}

/// Builds one layer (and, for a group, its whole subtree) from the node map, accumulating each
/// raster layer's decoded pixels into `pixels` (keyed by the layer's `id`, stringified -- mimas
/// dicts are always string-keyed) along the way. `Ok(None)` means "this id is a divider, skip
/// it" -- the caller filters those out of a group's children.
fn build_layer<'gc>(
    ctx: Ctx<'gc>,
    archive: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>,
    nodes: &HashMap<i64, RawNode>,
    id: i64,
    pixels: &mut HashMap<String, vm::DarklyImage<'gc>>,
) -> Result<Option<DarklyLayer>, String> {
    let Some(node) = nodes.get(&id) else {
        return Err(format!(
            "darkly::open: node {id} referenced but not defined"
        ));
    };
    match node.kind.as_str() {
        "divider" => Ok(None),
        "group" => {
            let body: GroupBody = serde_json::from_value(node.body.clone())
                .map_err(|e| format!("darkly::open: node {id} (group): {e}"))?;
            let mut children = Vec::with_capacity(body.children.len());
            for child_id in body.children {
                if let Some(layer) = build_layer(ctx, archive, nodes, child_id, pixels)? {
                    children.push(layer);
                }
            }
            Ok(Some(DarklyLayer::Group {
                id,
                name: body.name,
                visible: body.visible,
                opacity: body.opacity,
                children,
            }))
        }
        "raster" => {
            let body: RasterBody = serde_json::from_value(node.body.clone())
                .map_err(|e| format!("darkly::open: node {id} (raster): {e}"))?;
            let channels = channels_for_format(&body.pixels.format).ok_or_else(|| {
                format!(
                    "darkly::open: node {id}: unknown pixel format {:?}",
                    body.pixels.format
                )
            })?;
            let bytes = read_zip_entry(archive, &body.pixels.pixels)?;
            let expected = body.pixels.bounds.width as usize
                * body.pixels.bounds.height as usize
                * channels as usize;
            if bytes.len() != expected {
                return Err(format!(
                    "darkly::open: node {id}: pixel data is {} bytes, expected {expected} ({}x{}x{channels})",
                    bytes.len(),
                    body.pixels.bounds.width,
                    body.pixels.bounds.height,
                ));
            }
            let image = ctx.new_darkly_image(vm::RawImage {
                width: body.pixels.bounds.width as u32,
                height: body.pixels.bounds.height as u32,
                channels,
                bytes: bytes.into_boxed_slice(),
            });
            pixels.insert(id.to_string(), image);
            Ok(Some(DarklyLayer::Raster {
                id,
                name: body.name,
                visible: body.visible,
                opacity: body.opacity,
                width: body.pixels.bounds.width,
                height: body.pixels.bounds.height,
            }))
        }
        "void" => {
            let body: VoidBody = serde_json::from_value(node.body.clone())
                .map_err(|e| format!("darkly::open: node {id} (void): {e}"))?;
            Ok(Some(DarklyLayer::Void {
                id,
                name: body.name,
                visible: body.visible,
                void_type: body.void_type,
                transform: body.transform.data,
                params_json: serde_json::to_string(&body.params).unwrap_or_default(),
            }))
        }
        other => Err(format!(
            "darkly::open: node {id}: unsupported layer kind {other:?}"
        )),
    }
}

fn channels_for_format(format: &str) -> Option<u8> {
    match format {
        "rgba8unorm" => Some(4),
        "r8unorm" => Some(1),
        _ => None,
    }
}

fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>,
    name: &str,
) -> Result<Vec<u8>, String> {
    let mut entry = archive
        .by_name(name)
        .map_err(|e| format!("darkly::open: missing {name:?}: {e}"))?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .map_err(|e| format!("darkly::open: reading {name:?}: {e}"))?;
    Ok(bytes)
}

#[derive(serde::Deserialize)]
struct RawManifest {
    root: i64,
    nodes: Vec<RawNode>,
    canvas: RawCanvas,
    name: String,
}

#[derive(serde::Deserialize)]
struct RawCanvas {
    width: i64,
    height: i64,
}

#[derive(serde::Deserialize)]
struct RawNode {
    id: i64,
    #[serde(rename = "type")]
    kind: String,
    body: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct GroupBody {
    name: String,
    visible: bool,
    opacity: f64,
    children: Vec<i64>,
}

#[derive(serde::Deserialize)]
struct RasterBody {
    name: String,
    visible: bool,
    opacity: f64,
    pixels: PixelsRef,
}

#[derive(serde::Deserialize)]
struct PixelsRef {
    bounds: Bounds,
    format: String,
    pixels: String,
}

#[derive(serde::Deserialize)]
struct Bounds {
    width: i64,
    height: i64,
}

#[derive(serde::Deserialize)]
struct VoidBody {
    name: String,
    visible: bool,
    void_type: String,
    transform: Transform,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Transform {
    data: Vec<f64>,
}
