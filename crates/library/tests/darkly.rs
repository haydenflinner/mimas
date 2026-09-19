//! `std::darkly` -- reads a `.darkly` save file's layer tree + pixel data.
//! See `crates/library/src/std_lib/darkly.rs`.
//!
//! Unlike `dataframe.rs`'s tests, `open` takes a file *path*, and the path is only known at test
//! run time (a fresh temp file per test) -- so these are plain `#[test]` fns calling
//! `test_runner::render`/`render_display`/`try_execute` directly instead of going through the
//! `test_run!`/`test_run_display!` macros (which need a `const` preamble known at compile time).
#![cfg(feature = "darkly")]

#[macro_use]
mod test_runner;

use std::io::Write;

/// Writes a small, hand-built `.darkly` file (real shape, verified against an actual darkly
/// export -- see the module doc on `std_lib::darkly`): a root group with a 2x2 raster ("Sprite"),
/// a hidden "camera" void ("Spawn"), and a divider (which `open` must silently skip).
fn write_fixture(path: &std::path::Path) {
    let manifest = serde_json::json!({
        "format": "darkly",
        "container_version": 1,
        "name": "Test Doc",
        "canvas": { "width": 4, "height": 4, "origin_x": 0, "origin_y": 0 },
        "root": 1,
        "nodes": [
            {
                "id": 1,
                "type": "group",
                "body": {
                    "blend_mode": "normal",
                    "children": [2, 3, 99],
                    "collapsed": false,
                    "locked": false,
                    "modifiers": [],
                    "name": "Root",
                    "opacity": 1.0,
                    "passthrough": true,
                    "visible": true
                }
            },
            {
                "id": 2,
                "type": "raster",
                "body": {
                    "blend_mode": "normal",
                    "locked": false,
                    "modifiers": [],
                    "name": "Sprite",
                    "opacity": 1.0,
                    "pixels": {
                        "bounds": { "height": 2, "origin": { "x": 0, "y": 0 }, "width": 2 },
                        "format": "rgba8unorm",
                        "pixels": "layers/2.pixels"
                    },
                    "visible": true
                }
            },
            {
                "id": 3,
                "type": "void",
                "body": {
                    "blend_mode": "normal",
                    "locked": false,
                    "modifiers": [],
                    "name": "Spawn",
                    "opacity": 1.0,
                    "params": [true, 1],
                    "transform": { "data": [1.0, 0.0, 5.0, 0.0, 1.0, 10.0], "mode": "Basic" },
                    "visible": false,
                    "void_type": "camera"
                }
            },
            { "id": 99, "type": "divider", "body": {} }
        ]
    });

    let file = std::fs::File::create(path).expect("create fixture file");
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(manifest.to_string().as_bytes()).unwrap();
    zip.start_file("layers/2.pixels", options).unwrap();
    zip.write_all(&[255u8; 2 * 2 * 4]).unwrap(); // 2x2 rgba8, opaque white
    zip.finish().unwrap();
}

/// A unique temp path per test (parallel `cargo test` runs share the process's temp dir).
fn fixture_path(test_name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mimas-darkly-test-{test_name}-{}.darkly",
        std::process::id()
    ))
}

#[test]
fn open_reads_the_document_and_root_group() {
    let path = fixture_path("doc_and_root");
    write_fixture(&path);
    let preamble = format!(
        "use std::darkly::*;\nlet (doc, pixels) = open({:?})!;",
        path.display()
    );

    let root_kind_and_count = test_runner::render_display(
        &preamble,
        r#"match doc { DarklyDocument::Document { name, width, height, root } => f"{name} {width}x{height} " + (match root { DarklyLayer::Group { id, name, visible, opacity, children } => f"group '{name}' with {children.len()} children", _ => "not a group" }) }"#,
    );
    pretty_assertions::assert_eq!(
        root_kind_and_count,
        "Test Doc 4x4 group 'Root' with 2 children"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn open_skips_divider_nodes() {
    let path = fixture_path("divider_skip");
    write_fixture(&path);
    let preamble = format!(
        "use std::darkly::*;\nlet (doc, pixels) = open({:?})!;",
        path.display()
    );
    // children.len() == 2 (raster + void), not 3 -- the fixture's divider node is filtered out.
    let count = test_runner::render_display(
        &preamble,
        r#"match doc { DarklyDocument::Document { name, width, height, root } => match root { DarklyLayer::Group { id, name, visible, opacity, children } => f"{children.len()}", _ => "?" } }"#,
    );
    pretty_assertions::assert_eq!(count, "2");

    std::fs::remove_file(&path).ok();
}

#[test]
fn open_decodes_raster_pixels_with_matching_dimensions() {
    let path = fixture_path("raster_pixels");
    write_fixture(&path);
    let preamble = format!(
        "use std::darkly::*;\nlet (doc, pixels) = open({:?})!;
         let img = pixels[\"2\"]!;",
        path.display()
    );
    let dims = test_runner::render_display(&preamble, r#"f"{img.width()}x{img.height()}""#);
    pretty_assertions::assert_eq!(dims, "2x2");

    std::fs::remove_file(&path).ok();
}

#[test]
fn open_reads_void_transform_and_params() {
    let path = fixture_path("void_fields");
    write_fixture(&path);
    let preamble = format!(
        "use std::darkly::*;\nlet (doc, pixels) = open({:?})!;",
        path.display()
    );
    let void_summary = test_runner::render_display(
        &preamble,
        r#"match doc { DarklyDocument::Document { name, width, height, root } => match root { DarklyLayer::Group { id, name, visible, opacity, children } => match children[1] { DarklyLayer::Void { id, name, visible, void_type, transform, params_json } => f"{name} {void_type} {visible} {transform} {params_json}", _ => "not a void" }, _ => "not a group" } }"#,
    );
    pretty_assertions::assert_eq!(
        void_summary,
        "Spawn camera false [1, 0, 5, 0, 1, 10] [true,1]"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn open_missing_file_raises() {
    assert!(
        test_runner::try_execute(
            r#"use std::darkly::*; let d = open("/nonexistent/path/to/nothing.darkly")!;"#
        )
        .is_err(),
        "opening a nonexistent path should raise"
    );
}

#[test]
fn open_not_a_zip_raises() {
    let path = fixture_path("not_a_zip");
    std::fs::write(&path, b"this is not a zip file").unwrap();
    let src = format!(
        r#"use std::darkly::*; let d = open({:?})!;"#,
        path.display()
    );
    assert!(
        test_runner::try_execute(&src).is_err(),
        "opening a non-zip file should raise"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn open_unsupported_layer_kind_raises() {
    let manifest = serde_json::json!({
        "format": "darkly",
        "container_version": 1,
        "name": "Bad Doc",
        "canvas": { "width": 4, "height": 4, "origin_x": 0, "origin_y": 0 },
        "root": 1,
        "nodes": [
            {
                "id": 1,
                "type": "group",
                "body": {
                    "children": [2], "collapsed": false, "locked": false, "modifiers": [],
                    "name": "Root", "opacity": 1.0, "passthrough": true, "visible": true
                }
            },
            { "id": 2, "type": "vector", "body": {} }
        ]
    });
    let path = fixture_path("unsupported_kind");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(manifest.to_string().as_bytes()).unwrap();
    zip.finish().unwrap();

    let src = format!(
        r#"use std::darkly::*; let d = open({:?})!;"#,
        path.display()
    );
    assert!(
        test_runner::try_execute(&src).is_err(),
        "a vector (or other unrecognized) layer kind should raise, not be silently dropped"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn open_wrong_pixel_byte_count_raises() {
    let manifest = serde_json::json!({
        "format": "darkly",
        "container_version": 1,
        "name": "Bad Pixels",
        "canvas": { "width": 4, "height": 4, "origin_x": 0, "origin_y": 0 },
        "root": 1,
        "nodes": [
            {
                "id": 1,
                "type": "group",
                "body": {
                    "children": [2], "collapsed": false, "locked": false, "modifiers": [],
                    "name": "Root", "opacity": 1.0, "passthrough": true, "visible": true
                }
            },
            {
                "id": 2,
                "type": "raster",
                "body": {
                    "blend_mode": "normal", "locked": false, "modifiers": [], "name": "Sprite",
                    "opacity": 1.0, "visible": true,
                    "pixels": {
                        "bounds": { "height": 2, "origin": { "x": 0, "y": 0 }, "width": 2 },
                        "format": "rgba8unorm",
                        "pixels": "layers/2.pixels"
                    }
                }
            }
        ]
    });
    let path = fixture_path("wrong_byte_count");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("manifest.json", options).unwrap();
    zip.write_all(manifest.to_string().as_bytes()).unwrap();
    zip.start_file("layers/2.pixels", options).unwrap();
    zip.write_all(&[0u8; 3]).unwrap(); // way short of 2*2*4 = 16 bytes
    zip.finish().unwrap();

    let src = format!(
        r#"use std::darkly::*; let d = open({:?})!;"#,
        path.display()
    );
    assert!(
        test_runner::try_execute(&src).is_err(),
        "pixel byte count mismatching width*height*channels should raise"
    );
    std::fs::remove_file(&path).ok();
}
