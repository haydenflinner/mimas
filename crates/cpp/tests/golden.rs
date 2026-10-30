//! Golden test: the emitted .mim for a representative program, so lowering
//! changes show up in diffs. Regenerate expectations by running `mimas cpp`.

#[test]
fn transpile_snapshot() {
    let src = r#"
#include <iostream>
#include <vector>
using namespace std;

struct Point {
    int x = 1;
    int y = 2;
    int mag2() { return x * x + y * y; }
};

int main() {
    vector<int> v = {3, 1, 2};
    Point p;
    cout << p.mag2() << " " << v[0] << endl;
    return 0;
}
"#;
    let out = cpp::transpile(src).unwrap();
    let expected = r#"// cpp: #include <iostream>
// cpp: #include <vector>
// cpp: using namespace std;
struct Point {
    x: int,
    y: int,
}
impl Point {
    fn mag2(self) -> int {
        return self.x * self.x + self.y * self.y;
    }
}
fn main() -> int {
    let v: [int] = [3, 1, 2];
    let p: Point = Point { x = 1, y = 2 };
    print(f"{p.mag2()} {v[0]}");
    return 0;
}
main();
"#;
    assert_eq!(out.source.trim_end(), expected.trim_end());
}

#[test]
fn unsupported_gets_diagnostic() {
    let src = "int main() { int x = 1; return *(int*)&x; }\n";
    let out = cpp::transpile(src).unwrap();
    assert!(out
        .diagnostics
        .iter()
        .any(|d| d.message.contains("dereference") || d.message.contains("address-of")));
}
