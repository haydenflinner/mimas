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
//! Masks and the saved selection come along too: a raster layer's `mask` is its greyscale
//! reveal/hide channel (255 keeps a pixel, 0 hides it, in-between softens the edge -- useful as
//! stencils, fog-of-war maps, hit regions, or per-pixel damage, all readable from plain mimas
//! data), and `Document`'s `selection` is the marching-ants region darkly saved the file with.
//! Both decode to `DarklyImage`s in the same `pixels` map the raster buffers land in.
//!
//! Not parsed yet, deliberately out of scope: vector layers (their path geometry serializes via
//! `kurbo::BezPath`'s own format, not darkly's -- would need `kurbo`, itself a reasonably light,
//! GPU-free dependency, but not worth adding until something needs it) and the `recording/`
//! stroke-history directory (editor undo data, never relevant at runtime). `"divider"` nodes (a
//! pure editor concept -- the canvas/screen-space boundary marker) are silently skipped rather
//! than surfaced. Any other/future layer or modifier kind raises rather than silently dropping
//! content.

use std::{collections::HashMap, io::Read};

use macros::native;
use vm::{Ctx, MimasEnum, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_adt::<vm::conversion::DarklyImageTy>();
    // Field types resolve at `add_adt` time -- `DarklyMask` before the
    // layers that may carry one, layers before the document that roots
    // them.
    api.add_adt::<DarklyMask>();
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
/// Returns the image's width in pixels. The image comes from `std::darkly::open`'s `pixels` map.
fn width<'gc>(image: vm::DarklyImage<'gc>) -> i64 {
    image.0.0.width as i64
}

#[native]
/// Returns the image's height in pixels. The image comes from `std::darkly::open`'s `pixels` map.
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
        /// When the file was saved with an active selection (the marching-ants region that
        /// confines edits), its id here -- `pixels[id]` (stringified) is that region's R8
        /// image. The selection isn't a layer: darkly roots it at the document, not on a host.
        selection: Option<i64>,
    },
}

/// A raster layer's greyscale reveal/hide channel: `pixels[mask.id]` (the `id`, stringified)
/// is an R8 image where 255 keeps a pixel, 0 hides it, and in-between softens the edge.
#[derive(MimasEnum)]
enum DarklyMask {
    Mask {
        id: i64,
        name: String,
        visible: bool,
        /// Whether the mask moves/scales along with its host layer's transform.
        linked_to_host: bool,
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
        /// The group's mask, when it has one -- masks attach to any layer
        /// kind in darkly, groups included (the whole group's composite
        /// gets the channel multiplied in).
        mask: Option<DarklyMask>,
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
        /// The layer's mask, when it has one -- darkly attaches at most one mask per layer
        /// today (see `mask` in `open`'s doc).
        mask: Option<DarklyMask>,
    },
    Void {
        id: i64,
        name: String,
        visible: bool,
        /// The void layer's mask, when it has one.
        mask: Option<DarklyMask>,
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
/// tree (names, kinds, visibility, opacity, void transforms/params, raster masks, the saved
/// selection); `pixels` maps each pixel-carrying node's stringified `id` to its decoded
/// `DarklyImage` -- a raster layer's own buffer, a `mask`'s R8 channel, the `selection`'s R8
/// region. Pixels live in a side map rather than on the tree because a `DarklyLayer` value
/// can't hold one directly (see the comment on `DarklyLayer::Raster`'s `id` field).
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

    // Filters (masks, the saved selection) serialize alongside the layer
    // nodes but in their own `modifiers` list -- merge the two so every
    // id lookup below sees one flat map.
    let nodes: HashMap<i64, RawNode> = manifest
        .nodes
        .into_iter()
        .chain(manifest.modifiers)
        .map(|n| (n.id, n))
        .collect();
    let mut pixels = HashMap::new();
    let root = match build_layer(ctx, &mut archive, &nodes, manifest.root, &mut pixels) {
        Ok(Some(layer)) => layer,
        Ok(None) => {
            return Raisable::Raised("darkly::open: root node is a divider, not a layer".into());
        }
        Err(e) => return Raisable::Raised(e),
    };
    let selection = match manifest.selection_id {
        None => None,
        Some(id) => {
            match read_filter_pixels(ctx, &mut archive, &nodes, id, "selection", &mut pixels) {
                Ok(()) => Some(id),
                Err(e) => return Raisable::Raised(e),
            }
        }
    };
    let doc = DarklyDocument::Document {
        name: manifest.name,
        width: manifest.canvas.width,
        height: manifest.canvas.height,
        root,
        selection,
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
                mask: build_mask(ctx, archive, nodes, id, &body.modifiers, pixels)?,
            }))
        }
        "raster" => {
            let body: RasterBody = serde_json::from_value(node.body.clone())
                .map_err(|e| format!("darkly::open: node {id} (raster): {e}"))?;
            let image = read_pixels(ctx, archive, &body.pixels, id)?;
            pixels.insert(id.to_string(), image);
            Ok(Some(DarklyLayer::Raster {
                id,
                name: body.name,
                visible: body.visible,
                opacity: body.opacity,
                width: body.pixels.bounds.width,
                height: body.pixels.bounds.height,
                mask: build_mask(ctx, archive, nodes, id, &body.modifiers, pixels)?,
            }))
        }
        "void" => {
            let body: VoidBody = serde_json::from_value(node.body.clone())
                .map_err(|e| format!("darkly::open: node {id} (void): {e}"))?;
            Ok(Some(DarklyLayer::Void {
                id,
                name: body.name,
                visible: body.visible,
                mask: build_mask(ctx, archive, nodes, id, &body.modifiers, pixels)?,
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

/// Reads a host layer's `modifiers` list into its `DarklyMask`. At most one mask per host
/// today, but the format allows a list, so this fails loudly rather than dropping a second
/// modifier on the floor.
fn build_mask<'gc>(
    ctx: Ctx<'gc>,
    archive: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>,
    nodes: &HashMap<i64, RawNode>,
    host_id: i64,
    modifiers: &[i64],
    pixels: &mut HashMap<String, vm::DarklyImage<'gc>>,
) -> Result<Option<DarklyMask>, String> {
    let mut mask = None;
    for &modifier_id in modifiers {
        let Some(modifier) = nodes.get(&modifier_id) else {
            return Err(format!(
                "darkly::open: node {host_id}: modifier {modifier_id} referenced but not defined"
            ));
        };
        match modifier.kind.as_str() {
            "mask" => {
                if mask.is_some() {
                    return Err(format!(
                        "darkly::open: node {host_id}: more than one mask (unsupported)"
                    ));
                }
                let body: MaskBody = serde_json::from_value(modifier.body.clone())
                    .map_err(|e| format!("darkly::open: node {modifier_id} (mask): {e}"))?;
                read_filter_pixels(ctx, archive, nodes, modifier_id, "mask", pixels)?;
                mask = Some(DarklyMask::Mask {
                    id: modifier_id,
                    name: body.name,
                    visible: body.visible,
                    linked_to_host: body.linked_to_host,
                });
            }
            other => {
                return Err(format!(
                    "darkly::open: node {host_id}: unsupported modifier kind {other:?}"
                ));
            }
        }
    }
    Ok(mask)
}

/// Decodes one `ManifestPixelRef`'s raw `w*h*channels` blob into a `DarklyImage`, keyed by the
/// owning node's `id` (a layer's or a filter's -- both are node ids in the manifest).
fn read_pixels<'gc>(
    ctx: Ctx<'gc>,
    archive: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>,
    spec: &PixelsRef,
    id: i64,
) -> Result<vm::DarklyImage<'gc>, String> {
    let channels = channels_for_format(&spec.format).ok_or_else(|| {
        format!(
            "darkly::open: node {id}: unknown pixel format {:?}",
            spec.format
        )
    })?;
    let bytes = read_zip_entry(archive, &spec.pixels)?;
    let expected = spec.bounds.width as usize * spec.bounds.height as usize * channels as usize;
    if bytes.len() != expected {
        return Err(format!(
            "darkly::open: node {id}: pixel data is {} bytes, expected {expected} ({}x{}x{channels})",
            bytes.len(),
            spec.bounds.width,
            spec.bounds.height,
        ));
    }
    Ok(ctx.new_darkly_image(vm::RawImage {
        width: spec.bounds.width as u32,
        height: spec.bounds.height as u32,
        channels,
        bytes: bytes.into_boxed_slice(),
    }))
}

/// Every filter body (mask, selection) carries the same `pixels` ref shape -- decode it and
/// file the image under the filter's own id, so `pixels` stays one flat id→image map whether
/// the owner is a layer or an attachment.
fn read_filter_pixels<'gc>(
    ctx: Ctx<'gc>,
    archive: &mut zip::ZipArchive<std::io::Cursor<Vec<u8>>>,
    nodes: &HashMap<i64, RawNode>,
    id: i64,
    kind: &str,
    pixels: &mut HashMap<String, vm::DarklyImage<'gc>>,
) -> Result<(), String> {
    let Some(node) = nodes.get(&id) else {
        return Err(format!(
            "darkly::open: {kind} node {id} referenced but not defined"
        ));
    };
    let body: FilterPixelsBody = serde_json::from_value(node.body.clone())
        .map_err(|e| format!("darkly::open: node {id} ({kind}): {e}"))?;
    let image = read_pixels(ctx, archive, &body.pixels, id)?;
    pixels.insert(id.to_string(), image);
    Ok(())
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
    /// Filter nodes (masks, the saved selection) -- same `{id, type, body}`
    /// envelope as `nodes`, listed separately on disk.
    #[serde(default)]
    modifiers: Vec<RawNode>,
    canvas: RawCanvas,
    name: String,
    #[serde(default)]
    selection_id: Option<i64>,
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
    #[serde(default)]
    modifiers: Vec<i64>,
}

#[derive(serde::Deserialize)]
struct RasterBody {
    name: String,
    visible: bool,
    opacity: f64,
    pixels: PixelsRef,
    /// Attached filter node ids -- today at most one mask; the document's
    /// root-anchored selection never appears here (it's `selection_id` on
    /// the manifest).
    #[serde(default)]
    modifiers: Vec<i64>,
}

/// A mask node body -- `name`/`visible`/`linked_to_host` ride the layer
/// tree; the pixel spec is read again through `FilterPixelsBody` (a second,
/// narrower view over the same JSON, so each struct stays honest).
#[derive(serde::Deserialize)]
struct MaskBody {
    name: String,
    visible: bool,
    linked_to_host: bool,
}

/// The part of a filter body (mask, selection) that carries pixels -- both
/// kinds serialize the same `ManifestPixelRef` shape.
#[derive(serde::Deserialize)]
struct FilterPixelsBody {
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
    #[serde(default)]
    modifiers: Vec<i64>,
    void_type: String,
    transform: Transform,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Transform {
    data: Vec<f64>,
}
