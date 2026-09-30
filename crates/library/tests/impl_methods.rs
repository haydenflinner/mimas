use vm::{Ctx, mimas};

#[mimas]
struct Player {
    name: String,
    health: i64,
}

#[mimas]
impl Player {
    const MAX_HEALTH: i64 = 100;

    fn new(name: String) -> Self {
        Player {
            name,
            health: Self::MAX_HEALTH,
        }
    }

    fn greeting(&self) -> String {
        format!("hi, {}", self.name)
    }

    fn damage(&mut self, amount: i64) {
        self.health -= amount;
    }

    fn consume(self) -> i64 {
        self.health
    }

    /// a method may take ctx and auto-borrow containers, like any native
    fn drain_into(&mut self, _ctx: Ctx<'_>, sink: &mut Vec<vm::Val<'_>>) {
        sink.push(vm::Val::Int(self.health));
        self.health = 0;
    }
}

fn run(src: &str) -> String {
    let source = format!("{src}\n");
    let mut vm = vm::Vm::execute(&source, library::std).expect("test source compiled");
    let captured = vm.resolve_name("TEST_VALUE").expect("TEST_VALUE was bound");
    format!("{captured}")
}

#[test]
fn assoc_fn_constructs() {
    assert_eq!(
        run(r#"let p = Player::new("gabe"); let TEST_VALUE = p.health;"#),
        "100"
    );
}

#[test]
fn shared_receiver_reads() {
    assert_eq!(
        run(r#"let p = Player::new("gabe"); let TEST_VALUE = p.greeting();"#),
        "\"hi, gabe\""
    );
}

#[test]
fn mut_receiver_writes_back() {
    assert_eq!(
        run(r#"
            let p = Player::new("gabe");
            p.damage(30);
            p.damage(15);
            let TEST_VALUE = p.health;
        "#),
        "55"
    );
}

#[test]
fn mut_receiver_visible_through_alias() {
    assert_eq!(
        run(r#"
            let p = Player::new("gabe");
            let q = p;
            p.damage(30);
            let TEST_VALUE = q.health;
        "#),
        "70"
    );
}

#[test]
fn by_value_receiver() {
    assert_eq!(
        run(r#"let p = Player::new("gabe"); let TEST_VALUE = p.consume();"#),
        "100"
    );
}

#[test]
fn method_with_ctx_and_autoborrow() {
    assert_eq!(
        run(r#"
            let p = Player::new("gabe");
            let sink = [];
            p.drain_into(sink);
            let TEST_VALUE = [sink[0], p.health];
        "#),
        "[100, 0]"
    );
}

#[test]
fn assoc_const() {
    assert_eq!(run(r#"let TEST_VALUE = Player::MAX_HEALTH;"#), "100");
}

// `#[mimas]` items submit their declared parameter names through the `NativeMeta` inventory,
// which install joins onto the record -- `param_names` is parallel to `parameters`, with the
// receiver dropped for methods and the `Ctx` arg dropped for both shapes.
#[test]
fn param_names_come_from_the_signatures() {
    use api::ApiEntry;
    let library = vm::Vm::new().install_library(library::std);
    // the test's `#[mimas] struct Player` registers methods/assoc fns on an adt whose id we
    // read back off a known method rather than name-matching `adts()` (test modules in the
    // same binary may register their own)
    let damage = library
        .natives()
        .find_map(|(_, e)| match e {
            ApiEntry::Method(m) if m.name == "damage" => Some(m.recv_ty.clone()),
            _ => None,
        })
        .expect("damage registered");
    let names_of = |pred: &dyn Fn(&ApiEntry<()>) -> Option<&Vec<String>>| -> Vec<String> {
        library
            .natives()
            .find_map(|(_, e)| pred(e))
            .cloned()
            .expect("native was registered")
    };
    assert_eq!(
        names_of(&|e| match e {
            ApiEntry::Method(m) if m.name == "damage" && m.recv_ty == damage =>
                Some(&m.param_names),
            _ => None,
        }),
        ["amount"]
    );
    assert_eq!(
        names_of(&|e| match e {
            ApiEntry::Method(m) if m.name == "drain_into" && m.recv_ty == damage =>
                Some(&m.param_names),
            _ => None,
        }),
        ["sink"]
    );
    // assoc fns (`Player::new`) register as methods on the adt too, `takes_self: false`
    assert_eq!(
        names_of(&|e| match e {
            ApiEntry::Method(m) if m.name == "new" && m.recv_ty == damage => Some(&m.param_names),
            _ => None,
        }),
        ["name"]
    );
    // a `#[native]` std fn's names ride the same path
    assert_eq!(
        names_of(&|e| match e {
            ApiEntry::Function(f) if f.name == "write" && f.module == ["std", "fs"] =>
                Some(&f.param_names),
            _ => None,
        }),
        ["path", "contents"]
    );
}
