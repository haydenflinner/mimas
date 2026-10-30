//! Native (bevy_ui, not Typst) renderer for `img::Image` values, shown as a small popup when the
//! user hovers an identifier in the source panel -- a mimas program hands back *data* (an
//! `Image` tree: shapes and how they're composed), not typeset text, so the debugger can draw it
//! with real, pickable entities instead of a static raster. Click-through interaction (routing a
//! click on the popup back into the VM) isn't wired up yet -- this is the rendering half that
//! has to exist first.
//!
//! Pipeline: `Inspect` (generic, name-labeled reflection of *any* mimas value -- see
//! `mimas::vm::Inspect`) is parsed into [`Shape`] (this module's own typed tree, understanding
//! only `img`'s specific variants) by matching on `Inspect::Instance`'s `type_name` -- enum
//! variants report qualified names ("Image::Circle", not "Circle"), confirmed via
//! `crates/vm/tests/img_inspect.rs`. From there it's a conventional two-pass layout: [`measure`]
//! computes each node's bounding box bottom-up, [`arrange`] walks back down assigning absolute
//! positions, matching Pyret's actual composition semantics (`overlay` centers, `beside`/`above`
//! concatenate and center-align the cross axis, `place-image` positions by center point, ..).
//!
//! Values whose type doesn't implement `img::Draw` still show *something*: `SceneView::refresh`
//! falls back to the value's own to-string form (`Vm::resolve_for_scene`'s `Fallback` case),
//! wrapped as a single [`Shape::Text`] so the popup renderer doesn't need a separate code path.

use bevy::asset::Handle;
use bevy::color::Color as BevyColor;
use bevy::ecs::resource::Resource;
use bevy::math::Vec2;
use bevy::text::{Font, LineBreak, TextColor, TextLayout};
use bevy::ui::{
    AlignItems, BackgroundColor, BorderColor, BorderRadius, JustifyContent, Node, Overflow,
    PositionType, UiRect, Val,
};
use bevy::utils::default;
use bevy_immediate::Imm;
use bevy_immediate::ui::{CapsUi, text::ImmUiText};

use mimas::vm::{Inspect, SceneResult};

use crate::theme;

const SHAPE_BORDER_WIDTH: f32 = 2.0;
/// Text color for the to-string fallback popup -- deliberately plain (no `img::Color` involved,
/// there's no script-chosen color to use), just legible against the popup's card background.
const FALLBACK_TEXT_COLOR: Rgba = Rgba { r: 30, g: 30, b: 30, a: 255 };

/// Which identifier the mouse is currently over in the source panel (`None` when it's over
/// nothing hoverable), the scene last extracted for it, and the last parse/extraction failure.
/// `last_seen` memoizes the same way `DataflowView` does, except keyed on the hovered identifier
/// too: unlike a dataflow graph (a static property of a function), a scene is meant to change as
/// the program runs, so it has to refresh on every step, not just when the whole program is
/// replaced -- and obviously also whenever the hovered identifier itself changes.
#[derive(Resource, Default)]
pub(crate) struct SceneView {
    /// Set by the source panel's hover observers; cleared when the pointer leaves every
    /// identifier span. `pos` is the pointer's last window-pixel location, used to anchor the
    /// popup near the cursor.
    pub(crate) hovered: Option<String>,
    pub(crate) pos: Vec2,
    error: Option<String>,
    shape: Option<Shape>,
    last_seen: Option<(u64, u64, String)>,
}

impl SceneView {
    /// Re-extracts the popup's contents for the currently hovered identifier (see
    /// [`SceneView::hovered`]), if `(generation, steps, identifier)` has changed since the last
    /// call; otherwise a no-op. Call once per frame before [`render_popup`].
    pub(crate) fn refresh(&mut self, vm: &mut mimas::vm::Vm, generation: u64, steps: u64) {
        let Some(name) = self.hovered.clone() else {
            self.shape = None;
            self.error = None;
            self.last_seen = None;
            return;
        };
        let key = (generation, steps, name.clone());
        if self.last_seen.as_ref() == Some(&key) {
            return;
        }
        self.last_seen = Some(key);
        match vm.resolve_for_scene(&name, "draw") {
            Some(SceneResult::Drawn(inspect)) => match parse_shape(&inspect) {
                Some(shape) => {
                    self.shape = Some(shape);
                    self.error = None;
                }
                None => {
                    self.shape = None;
                    self.error = Some(format!(
                        "draw() returned something that isn't an img::Image: {inspect:?}"
                    ));
                }
            },
            Some(SceneResult::Fallback(text)) => {
                self.shape =
                    Some(Shape::Text { value: text, size: 16.0, color: FALLBACK_TEXT_COLOR });
                self.error = None;
            }
            // `name` isn't a live top-level local right now (stale hover, or an identifier
            // that was never a binding -- a keyword, type name, or field name) -- show nothing
            // rather than an error; this fires on almost every token in the file.
            None => {
                self.shape = None;
                self.error = None;
            }
        }
    }

    pub(crate) fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }

    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Rgba {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl From<Rgba> for BevyColor {
    fn from(c: Rgba) -> Self {
        BevyColor::srgba(
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        )
    }
}

/// This module's own typed view of an `img::Image` tree -- everything `Inspect`'s generic,
/// name-labeled reflection could tell us about a value, narrowed down to exactly the shapes and
/// combinators `img` defines. Parsed once per `refresh` (not re-parsed on every render frame).
pub(crate) enum Shape {
    Circle { radius: f32, outline: bool, color: Rgba },
    Ellipse { width: f32, height: f32, outline: bool, color: Rgba },
    Rectangle { width: f32, height: f32, outline: bool, color: Rgba },
    // `outline` isn't read yet -- `Triangle`'s renderer is a placeholder (see `DrawKind::
    // Triangle`) until real polygon rendering exists to make use of it.
    Triangle {
        side: f32,
        #[allow(dead_code)]
        outline: bool,
        color: Rgba,
    },
    Text { value: String, size: f32, color: Rgba },
    Line { x: f32, y: f32, color: Rgba },
    Overlay { top: Box<Shape>, bottom: Box<Shape> },
    OverlayXy { top: Box<Shape>, dx: f32, dy: f32, bottom: Box<Shape> },
    Beside { left: Box<Shape>, right: Box<Shape> },
    Above { top: Box<Shape>, bottom: Box<Shape> },
    EmptyScene { width: f32, height: f32 },
    PlaceImage { pic: Box<Shape>, x: f32, y: f32, background: Box<Shape> },
}

fn field<'a>(fields: &'a [(String, Inspect)], name: &str) -> Option<&'a Inspect> {
    fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

fn field_f32(fields: &[(String, Inspect)], name: &str) -> f32 {
    match field(fields, name) {
        Some(Inspect::Float(f)) => *f as f32,
        Some(Inspect::Int(i)) => *i as f32,
        _ => 0.0,
    }
}

fn field_u8(fields: &[(String, Inspect)], name: &str) -> u8 {
    match field(fields, name) {
        Some(Inspect::Int(i)) => (*i).clamp(0, 255) as u8,
        _ => 0,
    }
}

fn field_str(fields: &[(String, Inspect)], name: &str) -> String {
    match field(fields, name) {
        Some(Inspect::Str(s)) => s.clone(),
        _ => String::new(),
    }
}

fn named_color(name: &str) -> Rgba {
    match name {
        "red" => Rgba { r: 214, g: 73, b: 73, a: 255 },
        "green" => Rgba { r: 91, g: 168, b: 110, a: 255 },
        "blue" => Rgba { r: 82, g: 126, b: 214, a: 255 },
        "yellow" => Rgba { r: 224, g: 196, b: 87, a: 255 },
        "orange" => Rgba { r: 224, g: 150, b: 74, a: 255 },
        "purple" => Rgba { r: 158, g: 101, b: 196, a: 255 },
        "black" => Rgba { r: 30, g: 30, b: 30, a: 255 },
        "white" => Rgba { r: 245, g: 245, b: 245, a: 255 },
        "gray" | "grey" => Rgba { r: 140, g: 140, b: 140, a: 255 },
        "transparent" => Rgba { r: 0, g: 0, b: 0, a: 0 },
        _ => Rgba { r: 110, g: 110, b: 110, a: 255 },
    }
}

fn parse_color(inspect: &Inspect) -> Rgba {
    let Inspect::Instance { type_name, fields } = inspect else {
        return named_color("");
    };
    match type_name.as_str() {
        "Color::Named" => named_color(&field_str(fields, "0")),
        "Color::Rgb" => Rgba {
            r: field_u8(fields, "0"),
            g: field_u8(fields, "1"),
            b: field_u8(fields, "2"),
            a: 255,
        },
        "Color::Rgba" => Rgba {
            r: field_u8(fields, "0"),
            g: field_u8(fields, "1"),
            b: field_u8(fields, "2"),
            a: field_u8(fields, "3"),
        },
        _ => named_color(""),
    }
}

fn parse_outline(inspect: Option<&Inspect>) -> bool {
    matches!(inspect, Some(Inspect::Instance { type_name, .. }) if type_name == "Mode::Outline")
}

fn parse_shape(inspect: &Inspect) -> Option<Shape> {
    let Inspect::Instance { type_name, fields } = inspect else {
        return None;
    };
    let color = || field(fields, "color").map(parse_color).unwrap_or_default();
    let outline = || parse_outline(field(fields, "mode"));
    Some(match type_name.as_str() {
        "Image::Circle" => {
            Shape::Circle { radius: field_f32(fields, "radius"), outline: outline(), color: color() }
        }
        "Image::Ellipse" => Shape::Ellipse {
            width: field_f32(fields, "width"),
            height: field_f32(fields, "height"),
            outline: outline(),
            color: color(),
        },
        "Image::Rectangle" => Shape::Rectangle {
            width: field_f32(fields, "width"),
            height: field_f32(fields, "height"),
            outline: outline(),
            color: color(),
        },
        "Image::Triangle" => {
            Shape::Triangle { side: field_f32(fields, "side"), outline: outline(), color: color() }
        }
        "Image::Text" => Shape::Text {
            value: field_str(fields, "value"),
            size: field_f32(fields, "size"),
            color: color(),
        },
        "Image::Line" => {
            Shape::Line { x: field_f32(fields, "x"), y: field_f32(fields, "y"), color: color() }
        }
        "Image::Overlay" => Shape::Overlay {
            top: Box::new(parse_shape(field(fields, "top")?)?),
            bottom: Box::new(parse_shape(field(fields, "bottom")?)?),
        },
        "Image::OverlayXy" => Shape::OverlayXy {
            top: Box::new(parse_shape(field(fields, "top")?)?),
            dx: field_f32(fields, "dx"),
            dy: field_f32(fields, "dy"),
            bottom: Box::new(parse_shape(field(fields, "bottom")?)?),
        },
        "Image::Beside" => Shape::Beside {
            left: Box::new(parse_shape(field(fields, "left")?)?),
            right: Box::new(parse_shape(field(fields, "right")?)?),
        },
        "Image::Above" => Shape::Above {
            top: Box::new(parse_shape(field(fields, "top")?)?),
            bottom: Box::new(parse_shape(field(fields, "bottom")?)?),
        },
        "Image::EmptyScene" => Shape::EmptyScene {
            width: field_f32(fields, "width"),
            height: field_f32(fields, "height"),
        },
        "Image::PlaceImage" => Shape::PlaceImage {
            pic: Box::new(parse_shape(field(fields, "pic")?)?),
            x: field_f32(fields, "x"),
            y: field_f32(fields, "y"),
            background: Box::new(parse_shape(field(fields, "background")?)?),
        },
        _ => return None,
    })
}

/// Bounding-box size, bottom-up. `Text`'s estimate is rough (real glyph metrics aren't known
/// until bevy_text lays it out) -- good enough to position siblings, not pixel-exact.
fn measure(shape: &Shape) -> (f32, f32) {
    match shape {
        Shape::Circle { radius, .. } => (radius * 2.0, radius * 2.0),
        Shape::Ellipse { width, height, .. } => (*width, *height),
        Shape::Rectangle { width, height, .. } => (*width, *height),
        Shape::Triangle { side, .. } => (*side, side * 0.866),
        Shape::Text { value, size, .. } => {
            ((value.chars().count().max(1) as f32) * size * 0.62, size * 1.3)
        }
        Shape::Line { x, y, .. } => (x.abs().max(2.0), y.abs().max(2.0)),
        Shape::Overlay { top, bottom } => {
            let (tw, th) = measure(top);
            let (bw, bh) = measure(bottom);
            (tw.max(bw), th.max(bh))
        }
        Shape::OverlayXy { top, dx, dy, bottom } => {
            let (tw, th) = measure(top);
            let (bw, bh) = measure(bottom);
            let min_x = dx.min(0.0);
            let min_y = dy.min(0.0);
            let max_x = (dx + bw).max(tw);
            let max_y = (dy + bh).max(th);
            (max_x - min_x, max_y - min_y)
        }
        Shape::Beside { left, right } => {
            let (lw, lh) = measure(left);
            let (rw, rh) = measure(right);
            (lw + rw, lh.max(rh))
        }
        Shape::Above { top, bottom } => {
            let (tw, th) = measure(top);
            let (bw, bh) = measure(bottom);
            (tw.max(bw), th + bh)
        }
        Shape::EmptyScene { width, height } => (*width, *height),
        Shape::PlaceImage { background, .. } => measure(background),
    }
}

/// One leaf shape, positioned -- `arrange`'s output, what `render` actually draws. `x`/`y` are
/// the top-left corner in the overall scene's coordinate space (origin at the root's top-left).
struct DrawCmd {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    kind: DrawKind,
}

enum DrawKind {
    Ellipse { outline: bool, color: Rgba },
    Rect { outline: bool, color: Rgba },
    /// `img` can construct a triangle, but bevy_ui has no arbitrary-polygon primitive to draw
    /// one with -- rendered as a dashed placeholder box with a `\u{25b3}` glyph rather than
    /// silently drawing the wrong shape. Real triangle rendering needs either a custom mesh or
    /// gizmos, deliberately deferred rather than faked here.
    Triangle { color: Rgba },
    Text { value: String, size: f32, color: Rgba },
    /// Same honesty policy as `Triangle`: an actually-diagonal line needs UI-node rotation this
    /// doesn't attempt yet, so this draws the line's bounding box, not the line.
    Line { color: Rgba },
}

fn arrange(shape: &Shape, x: f32, y: f32, out: &mut Vec<DrawCmd>) {
    match shape {
        Shape::Circle { radius, outline, color } => out.push(DrawCmd {
            x,
            y,
            w: radius * 2.0,
            h: radius * 2.0,
            kind: DrawKind::Ellipse { outline: *outline, color: *color },
        }),
        Shape::Ellipse { width, height, outline, color } => out.push(DrawCmd {
            x,
            y,
            w: *width,
            h: *height,
            kind: DrawKind::Ellipse { outline: *outline, color: *color },
        }),
        Shape::Rectangle { width, height, outline, color } => out.push(DrawCmd {
            x,
            y,
            w: *width,
            h: *height,
            kind: DrawKind::Rect { outline: *outline, color: *color },
        }),
        Shape::Triangle { color, .. } => {
            let (w, h) = measure(shape);
            out.push(DrawCmd { x, y, w, h, kind: DrawKind::Triangle { color: *color } });
        }
        Shape::Text { value, size, color } => {
            let (w, h) = measure(shape);
            out.push(DrawCmd {
                x,
                y,
                w,
                h,
                kind: DrawKind::Text { value: value.clone(), size: *size, color: *color },
            });
        }
        Shape::Line { color, .. } => {
            let (w, h) = measure(shape);
            out.push(DrawCmd { x, y, w, h, kind: DrawKind::Line { color: *color } });
        }
        Shape::Overlay { top, bottom } => {
            // `bottom` has to land in `out` (and so on screen -- later entries paint over
            // earlier ones) before `top`, or "overlay(top, bottom)" would render upside down
            // from what the name promises.
            let (w, h) = measure(shape);
            let (tw, th) = measure(top);
            let (bw, bh) = measure(bottom);
            arrange(bottom, x + (w - bw) / 2.0, y + (h - bh) / 2.0, out);
            arrange(top, x + (w - tw) / 2.0, y + (h - th) / 2.0, out);
        }
        Shape::OverlayXy { top, dx, dy, bottom } => {
            let (bw, bh) = measure(bottom);
            let _ = (bw, bh);
            let min_x = dx.min(0.0);
            let min_y = dy.min(0.0);
            arrange(bottom, x + dx - min_x, y + dy - min_y, out);
            arrange(top, x - min_x, y - min_y, out);
        }
        Shape::Beside { left, right } => {
            let (_, h) = measure(shape);
            let (lw, lh) = measure(left);
            let (_, rh) = measure(right);
            arrange(left, x, y + (h - lh) / 2.0, out);
            arrange(right, x + lw, y + (h - rh) / 2.0, out);
        }
        Shape::Above { top, bottom } => {
            let (w, _) = measure(shape);
            let (tw, th) = measure(top);
            let (bw, _) = measure(bottom);
            arrange(top, x + (w - tw) / 2.0, y, out);
            arrange(bottom, x + (w - bw) / 2.0, y + th, out);
        }
        Shape::EmptyScene { .. } => {}
        Shape::PlaceImage { pic, x: px, y: py, background } => {
            arrange(background, x, y, out);
            let (pw, ph) = measure(pic);
            arrange(pic, x + px - pw / 2.0, y + py - ph / 2.0, out);
        }
    }
}

/// A small floating card anchored just past the pointer's last hover position (in window-pixel
/// space -- close enough to the source panel's own coordinate space since `root_node` sits at
/// the window origin with only a small fixed padding, and being off by that padding is
/// imperceptible for a tooltip). Sized to the content, clamped so one huge scene can't cover the
/// whole window.
fn popup_node(pos: Vec2, w: f32, h: f32) -> (Node, BorderColor, BackgroundColor) {
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(pos.x + 16.0),
            top: Val::Px(pos.y + 16.0),
            width: Val::Px(w),
            height: Val::Px(h),
            padding: UiRect::all(Val::Px(8.0)),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            overflow: Overflow::clip(),
            ..default()
        },
        BorderColor::all(theme::overlay1()),
        BackgroundColor(theme::surface1()),
    )
}

/// Renders `root` as a popup card floating near `pos` (the hovered identifier's last known
/// pointer position) -- called once per frame while [`SceneView::hovered`] is `Some` and a shape
/// was successfully extracted.
pub(crate) fn render_popup(ui: &mut Imm<CapsUi>, font: Handle<Font>, root: &Shape, pos: Vec2) {
    let (content_w, content_h) = measure(root);
    let w = content_w.clamp(32.0, 480.0) + 16.0;
    let h = content_h.clamp(24.0, 360.0) + 16.0;
    let mut cmds = Vec::new();
    arrange(root, 0.0, 0.0, &mut cmds);

    ui.ch_id("hover_popup")
        // always reinsert (not just on spawn): `pos` tracks the pointer, which can keep moving
        // slightly while it's still over the same identifier's span.
        .on_change_insert(true, move || popup_node(pos, w, h))
        .add(|ui| {
            for (i, cmd) in cmds.iter().enumerate() {
                let node = Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(cmd.x),
                    top: Val::Px(cmd.y),
                    width: Val::Px(cmd.w.max(1.0)),
                    height: Val::Px(cmd.h.max(1.0)),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    // outline mode is a colored border over a transparent fill -- with no width
                    // set here, that border defaults to 0px, which made an outline shape (any
                    // `BorderColor` with no matching `border` width) completely invisible.
                    border: UiRect::all(Val::Px(SHAPE_BORDER_WIDTH)),
                    ..default()
                };
                // `on_change_insert(true, ..)`, not `on_spawn_insert`, throughout this loop:
                // unlike the dataflow graph or the old full-page scene view (long-lived, mostly
                // stable content), a hover popup's shape tree is replaced wholesale every time
                // the hovered identifier changes -- same `("scene_rect", i)`-shaped ids can end
                // up reused for a completely different shape one hover to the next. With
                // `on_spawn_insert`, only the *first* value ever written at that id would stick
                // (a stale color/position left over from whatever was hovered before); this
                // forces every bundle fresh each frame.
                match &cmd.kind {
                    DrawKind::Rect { outline, color } => {
                        let (border, background) = fill_or_outline(*outline, *color);
                        ui.ch_id(("scene_rect", i)).on_change_insert(true, move || {
                            (node.clone(), BorderColor::all(border), BackgroundColor(background))
                        });
                    }
                    DrawKind::Ellipse { outline, color } => {
                        let (border, background) = fill_or_outline(*outline, *color);
                        let radius = cmd.w.min(cmd.h) / 2.0;
                        ui.ch_id(("scene_ellipse", i)).on_change_insert(true, move || {
                            (
                                Node { border_radius: BorderRadius::all(Val::Px(radius)), ..node.clone() },
                                BorderColor::all(border),
                                BackgroundColor(background),
                            )
                        });
                    }
                    DrawKind::Triangle { color } => {
                        let c = *color;
                        ui.ch_id(("scene_triangle", i))
                            .on_change_insert(true, move || {
                                (node.clone(), BorderColor::all(BevyColor::from(c)), BackgroundColor(BevyColor::NONE))
                            })
                            .add(|ui| {
                                ui.ch()
                                    .on_change_insert(true, {
                                        let font = font.clone();
                                        move || {
                                            (
                                                TextColor(c.into()),
                                                crate::text_font(font),
                                                TextLayout::linebreak(LineBreak::NoWrap),
                                            )
                                        }
                                    })
                                    .text("\u{25b3}");
                            });
                    }
                    DrawKind::Text { value, size, color } => {
                        let color = *color;
                        let size = *size;
                        let font = font.clone();
                        let value = value.clone();
                        ui.ch_id(("scene_text", i)).on_change_insert(true, move || node.clone()).add(
                            move |ui| {
                                ui.ch()
                                    .on_change_insert(true, move || {
                                        let mut f = crate::text_font(font.clone());
                                        f.font_size = bevy::text::FontSize::Px(size);
                                        (
                                            TextColor(color.into()),
                                            f,
                                            TextLayout::linebreak(LineBreak::NoWrap),
                                        )
                                    })
                                    .text(value.clone());
                            },
                        );
                    }
                    DrawKind::Line { color } => {
                        let c = *color;
                        ui.ch_id(("scene_line", i)).on_change_insert(true, move || {
                            (node.clone(), BorderColor::all(BevyColor::from(c)), BackgroundColor(BevyColor::NONE))
                        });
                    }
                }
            }
        });
}

fn fill_or_outline(outline: bool, color: Rgba) -> (BevyColor, BevyColor) {
    if outline {
        (color.into(), BevyColor::NONE)
    } else {
        (color.into(), color.into())
    }
}
