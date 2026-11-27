//! Atoms at run time: display, equality, lists, struct fields, raise /
//! absolve / match, and the fit into declared enums.

const HEAD: &str = "fn head(l: [int]) -> int! {
    if l.is_empty() { raise #EmptyList(\"the list was empty\") }
    l[0]
}";

// r[verify mimas.atom.display]
test_run_display!(
    atom_display,
    "#EmptyList" => "#EmptyList",
    "#ParseError(line: 3)" => "#ParseError(line: 3)",
    "#Pair(1, \"x\")" => "#Pair(1, \"x\")",
    "f\"{#Done}\"" => "#Done",
    "[#A, #B]" => "[#A, #B]",
);

// r[verify mimas.atom.equality]
test_run_display!(
    atom_equality,
    "#A == #A" => "true",
    "#A == #B" => "false",
    "#P(line: 3) == #P(line: 3)" => "true",
    "#P(line: 3) == #P(line: 4)" => "false",
    "[#A, #B, #A].contains(#B)" => "true",
);

// r[verify mimas.atom.raise]
test_run_display!(
    atom_raise_absolve_match,
    HEAD,
    "head([]) absolve |e| match e { #EmptyList => -1 }" => "-1",
    "head([7]) absolve |e| match e { #EmptyList => -1 }" => "7",
    "head([]) absolve |e| match e { #EmptyList(why) => why.len() }" => "18",
    "head([]) absolve |e| if e == #EmptyList(\"the list was empty\") { 1 } else { 2 }" => "1",
);

// r[verify mimas.atom.raise]
// `raise "text"` keeps working beside atoms
test_run_display!(
    string_raise_still_works,
    "fn s() -> int! { raise \"plain\" }",
    "s() absolve |e| match e { \"plain\" => 1, _ => 2 }" => "1",
);

// r[verify mimas.atom.type]
test_run_display!(
    atom_in_fields_and_branches,
    "struct Job { state: atom }",
    "Job { state: #Running }" => "Job { state: #Running }",
    "Job { state: #Running }.state == #Running" => "true",
    "match (if true { #Many } else { #Few }) { #Many => \"many\", #Few => \"few\" }" => "many",
);

// r[verify mimas.atom.const]
test_run_display!(atom_const, "const C = #Fixed", "C" => "#Fixed");

// r[verify mimas.atom.enum]
test_run_display!(
    atom_fits_enum,
    "enum Day { Mon, Tue, Wed }
    enum Shape { Circle { r: float }, Sq(float) }
    fn area(s: Shape) -> float { match s { #Circle { r } => r * r, #Sq(a) => a * a } }
    fn show(d: Day) -> str { f\"{d}\" }",
    "show(#Wed)" => "Day::Wed",
    "area(#Circle(r: 2.0)) + area(#Sq(3.0))" => "13.0",
    "{ let d: Day = #Tue; d == #Tue }" => "true",
);
