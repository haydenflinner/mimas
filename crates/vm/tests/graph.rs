//! `Vm::locals_graph`: shared/cyclic references keep their referent, and `index` fields carry
//! their unit -- what a host's boxes-and-arrows diagram needs that `Inspect` drops.

use vm::{Obj, Slot, Vm};

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
    let Obj::Array { items, more } = &g.objs[obj(&g, &g.roots[0].1)] else { panic!() };
    assert_eq!((items.len(), *more), (2, 0));
    let Obj::Instance { type_name, fields } = &g.objs[obj(&g, &items[0])] else { panic!() };
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
    let root = |n: &str| g.roots.iter().find(|(k, _)| k == n).map(|(_, s)| obj(&g, s)).unwrap();
    let (a, b) = (root("a"), root("b"));
    assert_eq!(root("alias"), b, "an alias is the same object, not a copy");
    let Obj::Instance { fields, .. } = &g.objs[a] else { panic!() };
    assert_eq!(obj(&g, &fields[1].slot), b);
    let Obj::Instance { fields, .. } = &g.objs[b] else { panic!() };
    assert_eq!(obj(&g, &fields[2].slot), a, "the back edge points at a, not <cycle>");
    assert_eq!(g.objs.len(), 2);
}
