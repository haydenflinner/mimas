use vm::{Captured, DebugEvent, Vm};

#[test]
fn debug_step_events_report_calls_locals_and_returns() {
    let mut vm = Vm::compile(
        "fn add_one(n: int) -> int { let m = n + 1; m }\nlet result = add_one(4);",
        |_| {},
    )
    .expect("source should compile");

    let mut events = Vec::new();
    loop {
        let (done, step_events) = vm.debug_step_events().expect("step should not fault");
        events.extend(step_events);
        if done {
            break;
        }
    }

    // the call into `add_one` shows up as a new frame, named from the source...
    assert!(
        events.iter().any(|e| matches!(
            e,
            DebugEvent::Called { function_name: Some(name), .. } if name == "add_one"
        )),
        "expected a Called event for add_one, got: {events:#?}"
    );

    // ...its parameter binds as a named local as part of that same call...
    assert!(
        events.iter().any(|e| matches!(
            e,
            DebugEvent::LocalChanged { name, value: Captured::Int(4), was: None, .. }
                if name == "n"
        )),
        "expected n to bind to 4, got: {events:#?}"
    );

    // ...its own `let` shows up too...
    assert!(
        events.iter().any(|e| matches!(
            e,
            DebugEvent::LocalChanged { name, value: Captured::Int(5), .. } if name == "m"
        )),
        "expected m to bind to 5, got: {events:#?}"
    );

    // ...the frame disappears on return...
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DebugEvent::Returned { .. })),
        "expected a Returned event, got: {events:#?}"
    );

    // ...and the result flows back into the caller's own named local.
    assert!(
        events.iter().any(|e| matches!(
            e,
            DebugEvent::LocalChanged { name, value: Captured::Int(5), .. } if name == "result"
        )),
        "expected result to bind to 5, got: {events:#?}"
    );
}

// A `frames()`/`debug_step_events` regression: a doubly-linked list (`a.next = b; b.prev = a;
// b.next = c; c.prev = b;`) used to blow the native stack, the same failure mode `Ctx::display`
// was fixed against (see `types.rs`'s `displaying_a_reference_cycle_errors_instead_of_crashing`)
// -- but `Val::capture` had no equivalent guard, since nothing read live cyclic data through it
// before the debugger did.
//
// A plain recursion-depth cutoff (what this was first fixed with) isn't enough here: `b` has two
// live edges back into the cycle (`next` *and* `prev`), so each recursion step re-enters the
// cycle from a different field -- the depth limit still terminates it, but the *work* to get
// there is exponential in the limit, not just the call stack. This needs real cycle detection
// (`b` already being an ancestor of itself) to actually be cheap, not merely non-crashing.
#[test]
fn frames_do_not_crash_on_a_reference_cycle() {
    let mut vm = Vm::compile(
        "struct Node { next: Node?, prev: Node?, val: int }
         let a = Node { next = null, prev = null, val = 1 };
         let b = Node { next = null, prev = a, val = 2 };
         a.next = b;
         let c = Node { next = null, prev = b, val = 3 };
         b.next = c;",
        |_| {},
    )
    .expect("source should compile");

    loop {
        // every step re-captures every live register, `a`/`b` included once they reference
        // each other -- this must not stack-overflow.
        let (done, _events) = vm.debug_step_events().expect("step should not fault");
        if done {
            break;
        }
    }

    let frames = vm.frames();
    let cut = frames
        .iter()
        .flat_map(|f| f.locals.iter())
        .any(|(_, v)| contains_cycle(v));
    assert!(
        cut,
        "expected the cyclic instance to cut to Captured::Cycle somewhere, got: {frames:#?}"
    );
}

fn contains_cycle(v: &Captured) -> bool {
    match v {
        Captured::Cycle => true,
        Captured::Array(items) | Captured::Instance(items) => items.iter().any(contains_cycle),
        Captured::Dict(entries) => entries.iter().any(|(_, v)| contains_cycle(v)),
        _ => false,
    }
}
