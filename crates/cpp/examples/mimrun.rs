fn main() {
    let src = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    match mimas::compile_source(&src) {
        Ok(mut vm) => {
            if let Err(e) = vm.run() { eprintln!("RUN ERR: {e}") }
        }
        Err(e) => eprintln!("COMPILE ERR: {e}"),
    }
}
