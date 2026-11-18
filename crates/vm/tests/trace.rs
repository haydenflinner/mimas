// The sonification substrate: per-op hit deltas + call-edge deltas, and
// the pc→(fn, loc) table hosts resolve them against.
use vm::Vm;

#[test]
fn trace_counts_ops_and_call_edges() {
    // `add` is small enough to be inlined into `twice` (compile's
    // `codegen::opt`): the edge is still counted, by the `TraceEdge` op
    // the inliner leaves where the call was
    let mut vm = Vm::execute(
        "fn add(a: int, b: int) -> int { a + b }\n\
         fn twice() -> int { add(1, 2) + add(3, 4) }\n\
         let TEST_VALUE = twice();",
        |_| {},
    )
    .expect("test source compiled and ran");

    let (hits, calls) = vm.trace_take();
    assert_eq!(hits.len() % 2, 0, "hits are (ip, count) pairs");
    assert!(!hits.is_empty(), "ops were recorded");
    assert!(hits.iter().skip(1).step_by(2).all(|n| *n > 0));
    assert_eq!(
        calls.len() % 3,
        0,
        "calls are (caller, callee, count) triples"
    );

    let map = vm.trace_map();
    let body = |name: &str| map.iter().find(|c| c.name == name).unwrap().body as u32;
    let add = body("add");
    let twice = body("twice");
    let edge = calls.chunks(3).find(|t| t[0] == twice && t[1] == add);
    assert_eq!(
        edge.map(|t| t[2]),
        Some(2),
        "twice→add fired twice: {calls:?}"
    );

    // Every hit ip resolves inside some chunk's code range and through
    // its loc table to a span (or nothing before the first entry).
    for &ip in hits.iter().step_by(2) {
        let owner = map
            .iter()
            .rev()
            .find(|c| c.offset <= ip)
            .unwrap_or_else(|| panic!("no chunk owns ip {ip}: {map:?}"));
        let _loc = owner
            .locs
            .iter()
            .rev()
            .find(|(rel, _, _, _)| owner.offset + rel <= ip);
    }

    // Deltas, not cumulative: a second take is empty.
    let (hits2, calls2) = vm.trace_take();
    assert!(hits2.is_empty() && calls2.is_empty(), "drained: {hits2:?}");
}

// The ordered path is lossless: expanded back to one ip per op, it has
// exactly the per-ip counts `trace_take` saw, and a loop costs one
// marker however long it runs.
#[test]
fn trace_path_rebuilds_every_op_in_order() {
    let run = |n: usize| {
        let src = format!(
            "fn collatz(n: int) -> int {{\n\
               let steps = 0;\n\
               let x = n;\n\
               while x != 1 {{\n\
                 x = (x * 7 + 3) mod 101;\n\
                 if x mod 2 == 0 {{ steps += 1; }} else {{ steps += 2; }}\n\
               }}\n\
               steps\n\
             }}\n\
             let total = 0;\n\
             for i in 1..30 {{ total += collatz(i); }}\n\
             let spin = 0;\n\
             for i in 0..{n} {{ spin += 1; }}"
        );
        let mut vm = Vm::compile(&src, |_| {}).expect("compiled");
        vm.trace_path_enable(true);
        vm.trace_take();
        vm.run().expect("ran");
        let path = vm.trace_path_take();
        let (hits, _) = vm.trace_take();
        (vm.trace_path_expand(&path), path, hits)
    };
    let (ips, path, hits) = run(1000);
    let mut counts = std::collections::BTreeMap::new();
    for ip in &ips {
        *counts.entry(*ip).or_insert(0u32) += 1;
    }
    let want: std::collections::BTreeMap<u32, u32> =
        hits.chunks(2).map(|h| (h[0], h[1])).collect();
    assert_eq!(counts, want, "expanded path has the hit counts");
    assert!(path.len() < ips.len() / 2, "jumps only: {} u32s for {} ops", path.len(), ips.len());

    // 100× the spin loop: more ops, same path size.
    let (ips2, path2, _) = run(100_000);
    assert!(ips2.len() > ips.len() + 90_000);
    assert_eq!(path2.len(), path.len(), "a loop folds to one marker");

    // drained: a second take is empty
    let mut vm = Vm::compile("let x = 1;", |_| {}).unwrap();
    vm.trace_path_enable(true);
    vm.run().unwrap();
    assert!(!vm.trace_path_take().is_empty());
    assert!(vm.trace_path_take().is_empty());
}

// Timing probe, not a check: `cargo test --release -p mimas-vm --test
// trace -- --ignored --nocapture` prints the recorder's per-op cost.
#[test]
#[ignore]
fn trace_path_cost() {
    let src = "let x = 0;\nlet i = 0;\nwhile i < 20000000 { if i mod 3 == 0 { x += 1; } else { x += 2; }\n i += 1; }";
    for on in [false, true, false, true] {
        let mut vm = Vm::compile(src, |_| {}).unwrap();
        vm.trace_path_enable(on);
        let t = std::time::Instant::now();
        vm.run().unwrap();
        let dt = t.elapsed();
        let (hits, _) = vm.trace_take();
        let ops: u64 = hits.chunks(2).map(|h| h[1] as u64).sum();
        let path = vm.trace_path_take();
        eprintln!(
            "path {}: {ops} ops in {dt:?} = {:.2} ns/op, path {} u32s",
            if on { "on " } else { "off" },
            dt.as_nanos() as f64 / ops as f64,
            path.len()
        );
    }
}
