// `mimas run` runs a program's checks after its top level, and a failing
// check fails the run. r[verify check.run.fails]
use std::process::Command;

fn run(src: &str, name: &str) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!("mimas-run-checks-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(name);
    std::fs::write(&file, src).unwrap();
    Command::new(env!("CARGO_BIN_EXE_mimas")).arg("run").arg(&file).output().unwrap()
}

#[test]
fn a_failing_check_fails_the_run() {
    let out = run("let w = 20px\nlet xs = [1, 2, 3]\ncheck xs.len() == 3\ncheck w == 30px\n", "fail.mimas");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("20px == 30px"), "{err}");
    assert!(err.contains("1 of 2 checks failed"), "{err}");
}

#[test]
fn passing_checks_pass_the_run() {
    let out = run("let xs = [1, 2, 3]\ncheck xs.len() == 3\ncheck xs[2] == 3\n", "pass.mimas");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}
