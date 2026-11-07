#![allow(dead_code)]

use vm::Vm;

pub fn render(preamble: &str, src: &str) -> String {
    let source = format!("{preamble}\nlet TEST_VALUE = {src};");
    let mut vm = Vm::execute(&source, library::std).expect("library test compiled");
    let captured = vm.resolve_name("TEST_VALUE").expect("TEST_VALUE was bound");
    format!("{captured}")
}

pub fn try_execute(src: &str) -> Result<(), vm::ExecuteError> {
    Vm::execute(src, library::std).map(|_| ())
}

/// Like [`render`], but through [`Vm::resolve_name_to_string`] instead of [`Vm::resolve_name`] --
/// the value's real, unescaped `Display` rendering (what `print` would show) rather than a
/// `Captured` debug snapshot. Use this when the expected output is multi-line or otherwise not
/// worth reading back out of `{:?}`-escaping (e.g. `Captured::Str`'s `Display` escapes newlines),
/// or when the value doesn't decompose into `Captured` at all (`Captured::Other`).
pub fn render_display(preamble: &str, src: &str) -> String {
    let source = format!("{preamble}\nlet TEST_VALUE = {src};");
    let mut vm = Vm::execute(&source, library::std).expect("library test compiled");
    vm.resolve_name_to_string("TEST_VALUE")
        .expect("TEST_VALUE was bound")
        .expect("TEST_VALUE displayed")
}

/// One case that came out wrong, for [`report`]: a `MISMATCH` line a
/// script can read back (`file`, source, got, want — tab-separated,
/// newlines escaped) plus the readable diff.
pub fn miss(file: &str, src: &str, actual: &str, expected: &str) -> String {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', "\\t");
    format!(
        "MISMATCH\t{file}\t{}\t{}\t{}\n  `{src}`\n    got  {actual}\n    want {expected}",
        esc(src),
        esc(actual),
        esc(expected)
    )
}

/// Every case in a `test_run!` runs; the test fails once, listing all
/// the misses, so a change that moves many expectations shows them in
/// one run instead of one per run.
pub fn report(misses: Vec<String>) {
    if !misses.is_empty() {
        panic!("{} case(s) came out wrong:\n{}", misses.len(), misses.join("\n"));
    }
}

#[macro_export]
macro_rules! test_fail {
    ($(#[$attr:meta])* $name:ident, $($src:expr),+ $(,)?) => {
        #[cfg(test)]
        #[test]
        $(#[$attr])*
        fn $name() {
            $(
                if $crate::test_runner::try_execute($src).is_ok() {
                    panic!("expected compilation/runtime failure for `{}`", $src);
                }
            )+
        }
    };
}

#[macro_export]
macro_rules! test_run {
    ($(#[$attr:meta])* $name:ident, $($src:expr => $expected:expr),* $(,)?) => {
        #[cfg(test)]
        #[test]
        $(#[$attr])*
        fn $name() {
            let mut misses = Vec::new();
            $({
                let actual = $crate::test_runner::render("", $src);
                if actual != $expected {
                    misses.push($crate::test_runner::miss(file!(), $src, &actual, $expected));
                }
            })*
            $crate::test_runner::report(misses);
        }
    };
    ($(#[$attr:meta])* $name:ident, $preamble:expr, $($src:expr => $expected:expr),* $(,)?) => {
        #[cfg(test)]
        #[test]
        $(#[$attr])*
        fn $name() {
            let mut misses = Vec::new();
            $({
                let actual = $crate::test_runner::render($preamble, $src);
                if actual != $expected {
                    misses.push($crate::test_runner::miss(file!(), $src, &actual, $expected));
                }
            })*
            $crate::test_runner::report(misses);
        }
    };
}

/// Like [`test_run!`], but compares against [`render_display`] -- the value's real, unescaped
/// `Display` output -- instead of [`render`]'s `Captured`-debug snapshot. Reach for this when the
/// expected value is multi-line or otherwise painful to hand-write escaped, or when the value is
/// opaque to `Captured` (`Captured::Other`).
#[macro_export]
macro_rules! test_run_display {
    ($(#[$attr:meta])* $name:ident, $($src:expr => $expected:expr),* $(,)?) => {
        #[cfg(test)]
        #[test]
        $(#[$attr])*
        fn $name() {
            let mut misses = Vec::new();
            $({
                let actual = $crate::test_runner::render_display("", $src);
                if actual != $expected {
                    misses.push($crate::test_runner::miss(file!(), $src, &actual, $expected));
                }
            })*
            $crate::test_runner::report(misses);
        }
    };
    ($(#[$attr:meta])* $name:ident, $preamble:expr, $($src:expr => $expected:expr),* $(,)?) => {
        #[cfg(test)]
        #[test]
        $(#[$attr])*
        fn $name() {
            let mut misses = Vec::new();
            $({
                let actual = $crate::test_runner::render_display($preamble, $src);
                if actual != $expected {
                    misses.push($crate::test_runner::miss(file!(), $src, &actual, $expected));
                }
            })*
            $crate::test_runner::report(misses);
        }
    };
}
