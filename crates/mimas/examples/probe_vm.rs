// End-to-end companion of `parse/examples/probe.rs`: runs the same `check`
// snippets through the whole pipeline (parse → solve → compile → VM) so a
// parse-level success that still breaks downstream shows up here.
fn main() {
    for src in [
        "fn add3(a: int, b: int, c: int) -> int { a + b + c }\ncheck 1 |> add3(10, 20, 30) == 61\n",
        "fn f(x: int) -> int { x + 1 }\ncheck f(2) == 3\n",
        "fn add3(a: int, b: int, c: int) -> int { a + b + c }\nlet x = 1 |> add3(10, 20, 30);\ncheck x == 61\n",
    ] {
        match mimas::Vm::compile(src, mimas::library::std) {
            Ok(mut vm) => {
                let results = vm.run_tests();
                let summary: Vec<String> = results
                    .iter()
                    .map(|r| match &r.error {
                        None => format!("PASS {}", r.name),
                        Some(e) => format!("FAIL {}: {e}", r.name),
                    })
                    .collect();
                eprintln!("{src:?} => {summary:?}");
            }
            Err(e) => eprintln!("{src:?} => compile error {e:?}"),
        }
    }
}
