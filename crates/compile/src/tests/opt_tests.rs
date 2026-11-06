// The IR optimizer's guarantees -- what inlining and scalar replacement leave
// behind -- which a result-only VM test cannot see (vm/tests/opt_semantics.rs
// pins that the results stay the same).
use super::compile_test_utils::optimized_ir;

const V2: &str = "struct V2 { x: float, y: float }
impl V2 {
    fn sub(self, o: V2) -> V2 { V2 { x = self.x - o.x, y = self.y - o.y } }
    fn dot(self, o: V2) -> float { self.x * o.x + self.y * o.y }
}";

/// The body of the last `fn` in the dump (bodies print in declaration order).
fn last_body(ir: &str) -> String {
    ir.rsplit("body @").next().unwrap().to_string()
}

#[test]
fn small_methods_inline_into_their_caller() {
    let ir = optimized_ir(&format!("{V2}\nfn dist2(a: V2, b: V2) -> float {{ let d = a.sub(b); d.dot(d) }}"));
    let body = last_body(&ir);
    assert!(!body.contains("call_direct"), "sub/dot should be inlined:\n{body}");
    assert!(body.contains("mult"), "dot's arithmetic should be in the caller:\n{body}");
}

#[test]
fn a_local_struct_becomes_its_fields() {
    let ir = optimized_ir("struct V2 { x: float, y: float }\nfn f(k: float) -> float { let d = V2 { x = k, y = 2.0 }; d.x = d.x + d.y; d.x * d.y }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field") && !body.contains("set_field"), "no field ops should survive:\n{body}");
    // the allocation is dead (dve drops it later); nothing must still read it
    assert!(!body.contains("make_instance") || !body.contains("get_field"), "{body}");
}

#[test]
fn an_inlined_temporary_is_scalarized_through_its_copies() {
    // `d` is the inlined `sub`'s return local copied into `d`, copied into
    // `dot`'s params: one object class, one set of field locals -- so no
    // `make_instance` result is stored anywhere any more (the params `a`
    // and `b` keep their objects: their field reads are the ones that stay)
    let ir = optimized_ir(&format!("{V2}\nfn dist2(a: V2, b: V2) -> float {{ let d = a.sub(b); d.dot(d) }}"));
    let body = last_body(&ir);
    for line in body.lines().filter(|l| l.contains("make_instance")) {
        let id = line.split("[ ").nth(1).unwrap().split(' ').next().unwrap();
        assert!(!body.lines().any(|l| l.contains(&format!("set l")) && l.ends_with(&format!(": {id}"))), "the struct is still stored: {line}\n{body}");
    }
    assert!(body.contains("mult"), "{body}");
}

#[test]
fn a_parameter_keeps_its_object() {
    // a param has a writer this pass can't see (the caller), so a copy of it
    // stays a copy and the mutation stays a set_field
    let ir = optimized_ir("struct V2 { x: float, y: float }\nfn f(p: V2) -> float { let q = p; q.x = 2.0; p.x }");
    let body = last_body(&ir);
    assert!(body.contains("set_field"), "mutation through an alias of a param must stay:\n{body}");
}

#[test]
fn an_escaping_struct_keeps_its_object() {
    let ir = optimized_ir("struct V2 { x: float, y: float }\nfn f() -> float { let v = V2 { x = 3.0, y = 0.0 }; let xs = [v]; v.x = 4.0; xs[0].x }");
    let body = last_body(&ir);
    assert!(body.contains("make_instance") && body.contains("set_field"), "a struct in an array keeps identity:\n{body}");
}

#[test]
fn a_rebound_local_keeps_its_objects() {
    // `a` is written twice; `b` aliases the first object, which must survive.
    // `V2` is mutated somewhere (`zero`), so it is a reference type and the
    // read stays a field read on a real object
    let ir = optimized_ir("struct V2 { x: float, y: float }\nimpl V2 { fn zero(self) { self.x = 0.0; } }\nfn f() -> float { let a = V2 { x = 1.0, y = 0.0 }; let b = a; a = V2 { x = 5.0, y = 0.0 }; b.x }");
    let body = last_body(&ir);
    assert!(body.matches("make_instance").count() == 2 && body.contains("get_field"), "{body}");
}

#[test]
fn a_rebound_value_local_reads_the_first_struct() {
    // the same with `V2` a value type: `b.x` is the first struct's operand
    let ir = optimized_ir("struct V2 { x: float, y: float }\nfn f() -> float { let a = V2 { x = 1.0, y = 0.0 }; let b = a; a = V2 { x = 5.0, y = 0.0 }; b.x }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field"), "{body}");
}

#[test]
fn the_entry_body_is_left_alone() {
    let ir = optimized_ir("struct V2 { x: float, y: float }\nlet v = V2 { x = 1.0, y = 2.0 };\nlet s = v.x + v.y;");
    let entry = ir.split("body @").next().unwrap();
    assert!(entry.contains("make_instance") && entry.contains("get_field"), "top-level bindings stay structs:\n{entry}");
}

#[test]
fn recursion_is_not_inlined() {
    let ir = optimized_ir("fn fact(n: int) -> int { if n <= 1 { 1 } else { n * fact(n - 1) } }\nfn g() -> int { fact(5) }");
    let fact = ir.split("body @1").nth(1).unwrap().split("body @").next().unwrap();
    assert!(fact.contains("call_direct @1"), "fact must still call itself:\n{fact}");
}

#[test]
fn a_loop_carried_struct_becomes_field_phis() {
    // `x` is a different struct every iteration; no struct is stored, read
    // or kept -- its field flows through a phi at the loop head
    let ir = optimized_ir("struct V2 { x: float, y: float }\nfn f(n: int) -> float { let v = V2 { x = 1.0, y = 0.0 }; for i in n { v = V2 { x = v.x * 2.0, y = v.y }; } v.x }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field"), "field reads should be gone:\n{body}");
    assert!(body.contains("phi"), "a loop-carried value needs a phi:\n{body}");
}

#[test]
fn a_named_scalar_local_keeps_its_slot() {
    // `t` holds ints: it is not promoted -- its reads are already free and
    // the slot is what the debugger's frame view shows -- so no phi is added
    // for it (the one the lowering makes for the loop's own value aside)
    let ir = optimized_ir("fn f(n: int) -> int { let t = 0; for i in n { t = t + i; } t }");
    let body = last_body(&ir);
    assert!(body.lines().any(|l| l.contains("set l")), "the named local's stores must stay:\n{body}");
    assert!(body.lines().any(|l| l.contains("get l")), "and its reads:\n{body}");
}

#[test]
fn a_value_struct_field_is_read_off_the_operand_even_when_it_escapes() {
    let ir = optimized_ir("struct V { x: float, y: float }\nfn keep(p: V) -> float { p.x }\nfn f(k: float) -> float { let d = V { x = k, y = 2.0 }; keep(d) + d.y }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field"), "field read should be the operand:\n{body}");
}

#[test]
fn a_loop_carried_value_struct_that_escapes_is_rebuilt_at_the_escape() {
    let ir = optimized_ir("struct V { x: float, y: float }\nfn keep(p: V) -> float { p.x }\nfn f(n: int) -> float { let v = V { x = 1.0, y = 0.0 }; for i in n { v = V { x = v.x * 2.0, y = v.y }; } keep(v) }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field"), "{body}");
    // one fresh struct for the escape, none for the loop
    assert!(body.contains("make_instance"), "the escape needs an object:\n{body}");
}

#[test]
fn a_mutated_adt_is_not_a_value_type() {
    // the struct escapes into an array (a call would be inlined away), and
    // `P` is mutated by `inc`, so the loop keeps real objects
    let ir = optimized_ir("struct P { x: float }\nstruct H { p: P, x: float }\nimpl P { fn inc(self) { self.x = self.x + 1.0; } }\nfn f(n: int) -> H { let p = P { x = 1.0 }; for i in n { p = P { x = p.x * 2.0 }; } H { p = p, x = p.x } }");
    let body = last_body(&ir);
    assert!(body.contains("get_field"), "a reference type keeps its object through the loop:\n{body}");
}

#[test]
fn a_value_type_escaping_into_an_array_is_rebuilt_there() {
    let ir = optimized_ir("struct V { x: float, y: float }\nstruct H { p: V, x: float }\nfn f(n: int) -> H { let p = V { x = 1.0, y = 0.0 }; for i in n { p = V { x = p.x * 2.0, y = p.y }; } H { p = p, x = p.x } }");
    let body = last_body(&ir);
    assert!(!body.contains("get_field"), "{body}");
}

#[test]
fn a_self_tail_call_becomes_a_jump() {
    let ir = optimized_ir("fn count(n: int, acc: int) -> int { if n == 0 { acc } else { count(n - 1, acc + n) } }");
    let body = last_body(&ir);
    assert!(!body.contains("call_direct"), "the tail call must be gone:\n{body}");
    assert!(body.contains("trace_edge"), "but its edge stays for the trace:\n{body}");
}

/// The debugger's shadows: a dissolved struct local's every store also lands in shadow
/// registers, one per field once the class is split, so the frame view can rebuild it.
#[test]
fn a_dissolved_struct_local_keeps_a_shadow_per_field() {
    let ir = optimized_ir("struct P { x: float, y: float }\nfn f(n: int) -> float { let p = P { x = 1.0, y = 0.0 }; for i in n { p = P { x = p.x * 2.0, y = p.y + 1.0 }; } p.x }");
    let body = last_body(&ir);
    // a shadow of each field at both stores, and no shadow of a whole struct
    // (the structs themselves are dead here; `clean` drops them after this)
    let made: Vec<String> = body
        .lines()
        .filter(|l| l.contains("make_instance"))
        .filter_map(|l| l.split("[ ").nth(1)?.split(" ]").next().map(str::to_string))
        .collect();
    let shadows: Vec<&str> = body.lines().filter(|l| l.contains("shadow ")).collect();
    assert_eq!(shadows.len(), 4, "{body}");
    for s in &shadows {
        let v = s.rsplit(": ").next().unwrap().trim();
        assert!(!made.contains(&v.to_string()), "a shadow of a whole struct: {s}\n{body}");
    }
}
