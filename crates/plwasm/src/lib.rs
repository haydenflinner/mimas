//! wasm32 entry for a plgen-compiled program. `pliron-wasm-ld` merges the
//! generated `mimas_pliron` object into this crate's module: the `env.mj_*`
//! calls in `mimas_body_*` bind to `vm::bc::nat`'s exports, and the
//! `mimas_pliron_bodies`/`mimas_pliron_nbodies` globals and
//! `mimas_pliron_init` come in as externs here.
//!
//! On wasm there is no host `dlsym` to fill the body table —
//! `mimas_pliron_init` stores each `llvm.addressof @mimas_body_*`, which
//! the emit side reads back as a `GOT.func` table index. `install_bc`
//! then sees the same `Vec<Option<BodyFn>>` shape the native driver
//! builds; the "pointer" is a function-table index, which is exactly how
//! `BodyFn` calls dispatch under call_indirect.

use std::cell::RefCell;

use vm::{Captured, Vm};

// `rand` (through `mimas-library`) pulls `getrandom` 0.4, which has no
// wasm32-unknown-unknown backend without JS. The `custom` backend
// (selected via `--cfg getrandom_backend="custom"` in the build) calls
// this symbol; nothing in the fixture path wants real entropy.
#[unsafe(no_mangle)]
extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    let buf = unsafe { std::slice::from_raw_parts_mut(dest, len) };
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for b in buf.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = x as u8;
    }
    Ok(())
}

unsafe extern "C" {
    fn mimas_pliron_init();
    /// The `Library::fingerprint_install` of the install the module's
    /// baked `NativeId`s were numbered against — checked before
    /// `install_bc` so a shim/driver install drift fails loudly instead
    /// of misdispatching every typed-arg native.
    fn mimas_pliron_nsig() -> i64;
    /// First element of the `[usize]` body table — `&` gives the table
    /// base in linear memory. Elements step at `size_of::<usize>()`
    /// (4 on wasm32, matching plgen's `i32` entries); `BodyFn` on wasm
    /// is a function-table index.
    static mimas_pliron_bodies: usize;
    /// The `i64` body count global.
    static mimas_pliron_nbodies: i64;
}

thread_local! {
    /// `TEST_VALUE`'s Display text (or the run's error) for
    /// `mimas_wasm_result_ptr`/`mimas_wasm_result_len` — wasm extern fns
    /// can't return strings.
    static RESULT: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_result(s: String) {
    RESULT.with(|r| *r.borrow_mut() = s);
}

/// The fixture native table — `mimas-fixture-natives::install`, the one
/// copy the driver compiles against too. `compile_parts` numbers natives
/// by registration order: a second copy of the list here drifts silently
/// and every `CallNative` in generated code dispatches to the wrong fn
/// (that broke this lane once — `expected array, received <T>`).
use fixture_natives::install;

/// `(source_ptr, source_len) -> status`: compile `source`, fill the body
/// table, install the bodies, run the program. `0` when `TEST_VALUE`
/// resolved, `1` when the run errored (or the name is missing) — the
/// text is readable via `mimas_wasm_result_ptr`/`len` either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mimas_wasm_run(src: *const u8, src_len: usize) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(src, src_len) };
    let source = String::from_utf8_lossy(bytes).into_owned();
    let (program, sources) = match Vm::compile_parts(&[("main", &source)], install) {
        Ok(v) => v,
        Err(e) => {
            set_result(format!("ERR {e}"));
            return 1;
        }
    };
    unsafe { mimas_pliron_init() };
    let (want, debug) = {
        let mut probe = Vm::new();
        let lib = probe.install_library(install);
        (lib.fingerprint_install(), move || lib.install_debug())
    };
    let got = unsafe { mimas_pliron_nsig() } as u64;
    if got != want {
        set_result(format!(
            "ERR native-table drift: module nsig {got:#x} != install {want:#x}\n{}",
            debug()
        ));
        return 1;
    }
    let n = unsafe { mimas_pliron_nbodies } as usize;
    let tbl: *const usize = std::ptr::addr_of!(mimas_pliron_bodies);
    let bodies: Vec<Option<vm::bc::BodyFn>> = (0..n)
        .map(|i| {
            Some(unsafe { std::mem::transmute::<usize, vm::bc::BodyFn>(tbl.add(i).read() as usize) })
        })
        .collect();
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, install);
    vm.install_bc(bodies);
    let err = vm.run().err().map(|e| e.to_string());
    let test = vm.resolve_name("TEST_VALUE").map(|c: Captured| format!("{c}"));
    let text = match (test, &err) {
        (_, Some(e)) => format!("ERR {e}"),
        (Some(v), None) => v,
        (None, None) => "MISSING".to_string(),
    };
    set_result(text);
    i32::from(err.is_some())
}

thread_local! {
    /// `layout_pairs(jit::layout())` on *this* build — the wasm32
    /// `Layout` values the harness feeds `emit_module` so generated
    /// wasm gets wasm32 offsets, not the host's.
    static LAYOUT_PAIRS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Pointer to the wasm32 `Layout` pairs (`mimas_wasm_layout_len` u64s).
/// Run this module standalone as the probe; `mimas_wasm_run` doesn't
/// need it.
#[unsafe(no_mangle)]
pub extern "C" fn mimas_wasm_layout() -> *const u64 {
    LAYOUT_PAIRS.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_empty() {
            *p = vm::bc::jit::layout_pairs(&vm::bc::jit::layout());
        }
        p.as_ptr()
    })
}

/// Length (in u64s) of the `mimas_wasm_layout` buffer.
#[unsafe(no_mangle)]
pub extern "C" fn mimas_wasm_layout_len() -> usize {
    LAYOUT_PAIRS.with(|p| {
        let mut p = p.borrow_mut();
        if p.is_empty() {
            *p = vm::bc::jit::layout_pairs(&vm::bc::jit::layout());
        }
        p.len()
    })
}

/// Leaky bump for host→wasm buffers (`mimas_wasm_run`'s source input).
/// Freed never — the fixtures are one-shot per instance.
#[unsafe(no_mangle)]
pub extern "C" fn mimas_wasm_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Pointer to the last result text (valid until the next run).
#[unsafe(no_mangle)]
pub extern "C" fn mimas_wasm_result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}

/// Length of the last result text.
#[unsafe(no_mangle)]
pub extern "C" fn mimas_wasm_result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

