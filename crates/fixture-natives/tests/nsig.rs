/// `fingerprint_install` must be deterministic — the driver bakes it
/// into `mimas_pliron_nsig` and runtimes compare it exactly.
#[test]
fn nsig() {
    let mut probe = vm::Vm::new();
    let lib = probe.install_library(fixture_natives::install);
    let sig = lib.fingerprint_install();

    let mut probe2 = vm::Vm::new();
    let lib2 = probe2.install_library(fixture_natives::install);
    assert_eq!(sig, lib2.fingerprint_install());
    assert_eq!(lib.install_debug().len(), lib2.install_debug().len());
}
