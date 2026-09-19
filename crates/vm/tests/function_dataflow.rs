use vm::{DataflowError, FunctionDataflowError, NodeKind, Vm};

#[test]
fn extracts_a_straight_line_function() {
    let source = r#"
        fn typst_box(x: float, name: str, val: int) -> str {
            f"content(({x},0),name:\"{name}\",frame:\"rect\",[{val}]);"
        }
        let unused = typst_box(1.0, "n0", 5);
    "#;
    let graph = Vm::function_dataflow(&[("main", source)], |_| {}, "typst_box")
        .expect("typst_box has no control flow, should extract fine");

    assert_eq!(graph.function_name, "typst_box");
    // 3 params (x, name, val) + 1 Format node + 1 Out node.
    assert_eq!(graph.nodes.len(), 5);
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::In)
            .count(),
        3
    );
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Out)
            .count(),
        1
    );
    let param_names: Vec<&str> = graph
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::In)
        .map(|n| n.label.as_str())
        .collect();
    assert_eq!(param_names, ["x", "name", "val"]);

    // every param feeds the Format node, which feeds Out: 4 edges total.
    assert_eq!(graph.edges.len(), 4);
    let format_idx = graph
        .nodes
        .iter()
        .position(|n| n.kind == NodeKind::Op && n.label == "format")
        .expect("should have a single `format` op node");
    let out_idx = graph
        .nodes
        .iter()
        .position(|n| n.kind == NodeKind::Out)
        .unwrap();
    for (i, _) in param_names.iter().enumerate() {
        assert!(
            graph
                .edges
                .iter()
                .any(|e| e.from == i && e.to == format_idx),
            "param {i} should wire into the format node"
        );
    }
    assert!(
        graph
            .edges
            .iter()
            .any(|e| e.from == format_idx && e.to == out_idx)
    );
}

#[test]
fn rejects_a_function_with_control_flow() {
    let source = r#"
        fn choose(a: int, b: int) -> int {
            if a > b {
                a
            } else {
                b
            }
        }
        let unused = choose(1, 2);
    "#;
    let err = Vm::function_dataflow(&[("main", source)], |_| {}, "choose")
        .expect_err("an `if` should be rejected as control flow, not silently misdrawn");
    assert!(matches!(
        err,
        FunctionDataflowError::Dataflow(DataflowError::HasControlFlow)
    ));
}

#[test]
fn reports_an_unknown_function_name() {
    let source = "let x = 1;";
    let err = Vm::function_dataflow(&[("main", source)], |_| {}, "does_not_exist")
        .expect_err("should fail to find a function that was never declared");
    assert!(matches!(
        err,
        FunctionDataflowError::Dataflow(DataflowError::UnknownFunction(name)) if name == "does_not_exist"
    ));
}

#[test]
fn collapses_a_reassigned_local_to_its_last_writer() {
    // `y` is written twice; a `GetLocal` after the second write should resolve to the *second*
    // write's producer, not the first -- exactly the "mutable local as a rewritten register"
    // behavior `function_dataflow`'s module docs describe.
    let source = r#"
        fn f(a: int) -> int {
            let y = a + 1;
            y = a + 2;
            y
        }
        let unused = f(1);
    "#;
    let graph = Vm::function_dataflow(&[("main", source)], |_| {}, "f")
        .expect("straight-line reassignment has no control flow");

    // 1 param (a) + 2 int-literal constants (1, 2) + 2 `add` op nodes + 1 Out. `y`/`GetLocal`/
    // `SetLocal` produce no nodes of their own -- see the module docs.
    assert_eq!(graph.nodes.len(), 6);
    let add_idxs: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Op && n.label == "add")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(add_idxs.len(), 2);
    let out_idx = graph
        .nodes
        .iter()
        .position(|n| n.kind == NodeKind::Out)
        .unwrap();
    // Out must be fed by the *second* add, not the first.
    assert!(
        graph
            .edges
            .iter()
            .any(|e| e.from == add_idxs[1] && e.to == out_idx)
    );
    assert!(
        !graph
            .edges
            .iter()
            .any(|e| e.from == add_idxs[0] && e.to == out_idx)
    );
}
