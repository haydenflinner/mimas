#[test]
fn nsig() {
    let mut probe = vm::Vm::new();
    let lib = probe.install_library(fixture_natives::install);
    eprintln!("fingerprint: {:#x}", lib.fingerprint());
}
