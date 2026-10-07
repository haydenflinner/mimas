// The sonification substrate: per-op hit deltas + call-edge deltas, and
// the pc→(fn, loc) table hosts resolve them against.
use vm::Vm;

#[test]
fn trace_counts_ops_and_call_edges() {
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
    assert_eq!(calls.len() % 3, 0, "calls are (caller, callee, count) triples");

    let map = vm.trace_map();
    let body = |name: &str| map.iter().find(|c| c.name == name).unwrap().body as u32;
    let add = body("add");
    let twice = body("twice");
    let edge = calls.chunks(3).find(|t| t[0] == twice && t[1] == add);
    assert_eq!(edge.map(|t| t[2]), Some(2), "twice→add fired twice: {calls:?}");

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
