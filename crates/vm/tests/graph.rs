//! `Vm::locals_graph`: shared/cyclic references keep their referent, and `index` fields carry
//! their unit -- what a host's boxes-and-arrows diagram needs that `Inspect` drops.

use vm::{Captured, Obj, Slot, Vm};

fn obj(_g: &vm::HeapGraph, s: &Slot) -> usize {
    match s {
        Slot::Ref(id) => *id as usize,
        other => panic!("expected a ref, got {other:?}"),
    }
}

#[test]
fn index_fields_are_tagged() {
    let src = r#"
struct Node { val: int, next: index? }
let nodes = [Node { val = 3, next = 1 }, Node { val = 5, next = null }];
let head: index = 0;
"#;
    let mut vm = Vm::compile(src, |_| {}).expect("compile");
    vm.run().expect("run");
    let g = vm.locals_graph(Some(&["nodes"]));
    assert_eq!(g.roots.len(), 1);
    let Obj::Array {
        items,
        more,
        gap_at,
    } = &g.objs[obj(&g, &g.roots[0].1)]
    else {
        panic!()
    };
    assert_eq!((items.len(), *more, *gap_at), (2, 0, 2));
    let Obj::Instance {
        type_name, fields, ..
    } = &g.objs[obj(&g, &items[0])]
    else {
        panic!()
    };
    assert_eq!(type_name, "Node");
    assert_eq!(fields[0].name, "val");
    assert!(!fields[0].is_index());
    assert_eq!(fields[1].name, "next");
    assert!(fields[1].is_index(), "{:?}", fields[1]);
}

#[test]
fn shared_and_cyclic_refs_keep_their_target() {
    let src = r#"
struct Node { val: int, next: Node?, prev: Node? }
let a = Node { val = 1, next = null, prev = null };
let b = Node { val = 2, next = null, prev = a };
a.next = b;
let alias = b;
"#;
    let mut vm = Vm::compile(src, |_| {}).expect("compile");
    vm.run().expect("run");
    let g = vm.locals_graph(None);
    let root = |n: &str| {
        g.roots
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, s)| obj(&g, s))
            .unwrap()
    };
    let (a, b) = (root("a"), root("b"));
    assert_eq!(root("alias"), b, "an alias is the same object, not a copy");
    let Obj::Instance { fields, .. } = &g.objs[a] else {
        panic!()
    };
    assert_eq!(obj(&g, &fields[1].slot), b);
    let Obj::Instance { fields, .. } = &g.objs[b] else {
        panic!()
    };
    assert_eq!(
        obj(&g, &fields[2].slot),
        a,
        "the back edge points at a, not <cycle>"
    );
    assert_eq!(g.objs.len(), 2);
}

/// Displaying a cyclic list prints a short back-reference instead of recursing
/// (or erroring): `a.next = a` renders the self-edge as `Node 1 ↺`.
// r[verify mimas.display.cycle]
#[test]
fn displaying_a_cycle_shows_a_back_reference() {
    let src = r#"
struct Node { val: int, next: Node? }
let a = Node { val = 1, next = null };
a.next = a;
let s = f"{a}";
"#;
    // 1 MiB: the old depth-1000 recursion overflowed this; the marker needs no stack at all
    let handle = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(move || {
            let mut vm = Vm::compile(src, |_| {}).expect("compile");
            vm.run().expect("run");
            vm.resolve_name("s")
        })
        .unwrap();
    let s = handle.join().expect("no stack overflow").expect("s bound");
    assert_eq!(s, Captured::Str("Node { 1, Node 1 ↺ }".to_string()));
}

#[test]
fn a_long_array_keeps_its_head_and_tail() {
    let nums: Vec<String> = (0..100).map(|i| i.to_string()).collect();
    let src = format!("let xs = [{}];", nums.join(", "));
    let src = src.as_str();
    let mut vm = Vm::compile(src, |_| {}).expect("compile");
    vm.run().expect("run");
    let g = vm.locals_graph(Some(&["xs"]));
    let Obj::Array {
        items,
        more,
        gap_at,
    } = &g.objs[obj(&g, &g.roots[0].1)]
    else {
        panic!()
    };
    assert_eq!(items.len() + more, 100);
    assert_eq!(items.len(), vm::GRAPH_MAX_ELEMS);
    let int = |s: &Slot| match s {
        Slot::Leaf(vm::Inspect::Int(n)) => *n,
        other => panic!("{other:?}"),
    };
    assert_eq!(int(&items[*gap_at - 1]), *gap_at as i64 - 1);
    assert_eq!(int(&items[*gap_at]), (*gap_at + more) as i64);
    assert_eq!(int(items.last().unwrap()), 99);
}

/// A struct the program never mutates is a value: its instances carry `value: true` so a
/// host can draw them inline. One with a mutating method, or mutated anywhere, is not.
#[test]
fn value_types_are_flagged_on_the_graph() {
    let mut vm = vm::Vm::compile(
        "struct V2 { x: float, y: float }
         struct Acc { total: float }
         impl Acc { fn add(self, d: float) { self.total = self.total + d; } }
         struct Seg { a: V2, b: V2, acc: Acc }
         let s = Seg { a = V2 { x = 1.0, y = 2.0 }, b = V2 { x = 3.0, y = 4.0 }, acc = Acc { total = 0.0 } };",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    let g = vm.locals_graph(Some(&["s"]));
    let flags: Vec<(String, bool)> = g
        .objs
        .iter()
        .filter_map(|o| match o {
            Obj::Instance {
                type_name, value, ..
            } => Some((type_name.clone(), *value)),
            _ => None,
        })
        .collect();
    assert!(flags.contains(&("V2".into(), true)), "{flags:?}");
    assert!(flags.contains(&("Acc".into(), false)), "{flags:?}");
    assert!(flags.contains(&("Seg".into(), true)), "{flags:?}");
}

/// A doubly linked list prints its chain once: `prev` edges back up the chain and the
/// list's `tail` (an alias of a node already printed) are references named by content,
/// `Node 2 ↺`, not a second copy of the chain. A shared plain value (`Point`) and a
/// repeated unit variant still print in full.
// r[verify mimas.display.cycle] r[verify mimas.display.cycle.full]
#[test]
fn display_names_revisited_nodes_by_content() {
    let src = r#"
struct Node { val: int, prev: Node?, next: Node? }
struct List { head: Node?, tail: Node?, len: int }
let a = Node { val = 1, prev = null, next = null };
let b = Node { val = 2, prev = a, next = null };
let c = Node { val = 3, prev = b, next = null };
a.next = b;
b.next = c;
let l = List { head = a, tail = c, len = 3 };
struct Point { x: int, y: int }
let p = Point { x = 1, y = 2 };
enum Sky { Sun, Moon }
let s_list = f"{l}";
let s_b = f"{b}";
let s_pts = f"{[p, p]}";
let s_sky = f"{[Sky::Moon, Sky::Moon]}";
"#;
    let mut vm = Vm::compile(src, |_| {}).expect("compile");
    vm.run().expect("run");
    let get = |vm: &mut Vm, n: &str| match vm.resolve_name(n) {
        Some(Captured::Str(s)) => s,
        other => panic!("{n}: {other:?}"),
    };
    assert_eq!(
        get(&mut vm, "s_list"),
        "List { Node { 1, null, Node { 2, Node 1 ↺, Node { 3, Node 2 ↺, null } } }, Node 3 ↺, 3 }"
    );
    assert_eq!(get(&mut vm, "s_b"), "Node { 2, Node { 1, null, Node 2 ↺ }, Node { 3, Node 2 ↺, null } }");
    assert_eq!(get(&mut vm, "s_pts"), "[Point { 1, 2 }, Point { 1, 2 }]");
    assert_eq!(get(&mut vm, "s_sky"), "[Sky::Moon, Sky::Moon]");
}
