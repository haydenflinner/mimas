//! End-to-end: transpile C++ and run the emitted mimas through the real
//! pipeline (`compile_source` + `Vm::run`), capturing `print` output.

use std::cell::RefCell;
use std::rc::Rc;

/// Transpile `src`, run it, and return (emitted mimas, printed lines).
fn run_cpp(src: &str) -> (String, Vec<String>) {
    let out = cpp::transpile(src).expect("transpile failed");
    let lines = Rc::new(RefCell::new(Vec::new()));
    let sink = {
        let lines = lines.clone();
        move |line: &str| lines.borrow_mut().push(line.to_string())
    };
    let mut vm = mimas::compile_source(&out.source)
        .unwrap_or_else(|e| panic!("emitted mimas failed to compile:\n{}\n\n{e}", out.source));
    vm.fixture::<vm::fixtures::Out>().set(sink);
    vm.run()
        .unwrap_or_else(|e| panic!("emitted mimas failed at run time:\n{}\n\n{e}", out.source));
    (out.source, lines.borrow().clone())
}

fn assert_prints(src: &str, expected: &[&str]) {
    let (mim, got) = run_cpp(src);
    assert_eq!(
        got,
        expected,
        "wrong output for transpiled program:\n{mim}"
    );
}

#[test]
fn arithmetic_and_io() {
    assert_prints(
        r#"
int main() {
    std::cout << "hello " << 1 + 2 << std::endl;
    int d = 7 / 2;
    std::cout << "int div: " << d << std::endl;
    double q = 7 / 2.0;
    std::cout << "float div: " << q << std::endl;
    return 0;
}
"#,
        &["hello 3", "int div: 3", "float div: 3.5"],
    );
}

#[test]
fn loops_and_vector() {
    assert_prints(
        r#"
#include <vector>
using namespace std;
int main() {
    vector<int> v = {1, 2, 3};
    v.push_back(4);
    int sum = 0;
    for (int i = 0; i < v.size(); i++) sum += v[i];
    for (int x : v) sum += x;
    cout << sum << endl;
    return 0;
}
"#,
        &["20"],
    );
}

#[test]
fn functions_structs_and_methods() {
    assert_prints(
        r#"
#include <iostream>
using namespace std;

int twice(int n, int k = 2) { return n * k; }

struct Point {
    int x;
    int y = 5;
    Point() : x(1) {}
    int mag2() { return x * x + y * y; }
};

int main() {
    Point p;
    cout << p.mag2() << " " << twice(10) << " " << twice(10, 3) << endl;
    return 0;
}
"#,
        &["26 20 30"],
    );
}

#[test]
fn control_flow() {
    assert_prints(
        r#"
#include <iostream>
using namespace std;
int main() {
    int n = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 3 == 0) continue;
        if (i > 7) break;
        n += i;
    }
    int w = 0;
    while (n > 0) { w += n % 10; n /= 10; }
    int d = 2;
    do { d--; } while (d > 0);
    int s = 0;
    switch (w) {
        case 1: s = 100; break;
        case 8: s = 200; break;
        default: s = -1;
    }
    cout << w << " " << d << " " << s << endl;
    return 0;
}
"#,
        // n = 1+2+4+5+7 = 19; w = 9+1 = 10; d = 0; switch(10) → default → s = -1
        &["10 0 -1"],
    );
}

#[test]
fn maps_sets_strings() {
    assert_prints(
        r#"
#include <map>
#include <set>
#include <string>
#include <iostream>
using namespace std;
int main() {
    map<string, int> m;
    m["a"] = 1;
    m["b"] = m["a"] + 1;
    set<int> s;
    s.insert(3);
    string w = "hello";
    cout << m["b"] << " " << s.count(3) << " " << w.size() << " " << w.find("ll") << endl;
    return 0;
}
"#,
        &["2 1 5 2"],
    );
}

#[test]
fn out_params_and_refs() {
    assert_prints(
        r#"
#include <iostream>
using namespace std;
void minmax(int a, int b, int& lo, int& hi) {
    if (a < b) { lo = a; hi = b; } else { lo = b; hi = a; }
}
void bump(int& x) { x += 1; }
int main() {
    int lo, hi;
    minmax(9, 4, lo, hi);
    bump(lo);
    cout << lo << " " << hi << endl;
    return 0;
}
"#,
        &["5 9"],
    );
}

#[test]
fn algorithms() {
    assert_prints(
        r#"
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
    vector<int> v = {5, 1, 4, 2, 3};
    sort(v.begin(), v.end());
    cout << v[0] << v[4] << *min_element(v.begin(), v.end()) << endl;
    reverse(v.begin(), v.end());
    int c = count(v.begin(), v.end(), 4);
    int acc = accumulate(v.begin(), v.end(), 0);
    cout << v[0] << v[4] << " " << c << " " << acc << endl;
    return 0;
}
"#,
        // sorted [1..5] -> v0=1 v4=5 min=1; reversed -> v0=5 v4=1; count=1; sum=15
        &["151", "51 1 15"],
    );
}

#[test]
fn stack_queue_deque() {
    assert_prints(
        r#"
#include <stack>
#include <queue>
#include <iostream>
using namespace std;
int main() {
    stack<int> st;
    st.push(1); st.push(2);
    queue<int> q;
    q.push(7); q.push(8); q.pop();
    cout << st.top() << " " << q.front() << endl;
    return 0;
}
"#,
        &["2 8"],
    );
}
