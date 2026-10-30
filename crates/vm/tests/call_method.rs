use vm::{Captured, Vm};

#[test]
fn call_method_on_first_instance_runs_a_user_pact_method() {
    let mut vm = Vm::compile(
        "pact Typeset {
            fn typeset(self) -> str;
         }
         struct Node { next: Node?, val: int }
         impl Typeset for Node {
             fn typeset(self) -> str {
                 if let n? = self.next {
                     f\"({self.val}, {n.typeset()})\"
                 } else {
                     f\"{self.val}\"
                 }
             }
         }
         let a = Node { next = null, val = 1 };
         let b = Node { next = null, val = 2 };
         a.next = b;",
        |_| {},
    )
    .expect("source should compile");

    // run to completion first: the method call must work on a value the top-level script has
    // already finished building, not just mid-construction.
    vm.run().expect("script should run to completion");

    let result = vm
        .call_method_on_first_instance("typeset")
        .expect("should find `a` (the first Node-typed local) and call typeset() on it");
    assert_eq!(result, Captured::Str("(1, 2)".to_string()));

    // calling again must still work and give the same answer -- the injected call shouldn't
    // have corrupted the program's own registers (the whole point of the fresh-return-slot
    // design over reusing register 0 like `call_fn` does).
    let result2 = vm
        .call_method_on_first_instance("typeset")
        .expect("second call should still find and call it");
    assert_eq!(result2, Captured::Str("(1, 2)".to_string()));
}

#[test]
fn call_method_on_first_instance_finds_nothing_when_no_type_implements_it() {
    let mut vm = Vm::compile(
        "struct Node { val: int }\nlet a = Node { val = 1 };",
        |_| {},
    )
    .expect("source should compile");
    vm.run().expect("script should run to completion");

    assert_eq!(vm.call_method_on_first_instance("typeset"), None);
}
