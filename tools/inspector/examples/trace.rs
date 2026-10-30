//! Fast, headless diagnostic: single-steps the configured script and prints each op's
//! `(chunk, ip, loc)` so hangs / runaway loops can be diagnosed without the GUI's overhead.
//! `cargo run --example trace -- [max_steps]` for a raw op trace, or
//! `cargo run --example trace -- step_line [n]` to replay `Session::step_line`'s algorithm.

use mimas::vm::Vm;

fn load() -> Vm {
    let home = std::env::var("HOME").unwrap();
    let path = std::path::PathBuf::from(home).join("code/dsa/scripts/main.mim");
    let source = std::fs::read_to_string(&path).unwrap();
    mimas::compile_files(&[("main", &source)]).expect("script should compile")
}

fn line_byte_range(source: &str, offset: usize) -> std::ops::Range<usize> {
    let offset = offset.min(source.len());
    let start = source[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = source[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(source.len());
    start..end
}

fn current_loc(vm: &mut Vm) -> Option<(usize, usize)> {
    let (_, _, loc) = vm.current_position()?;
    if loc.is_synthetic() {
        return None;
    }
    Some((loc.file_id, loc.span.start))
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("step_line") {
        let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
        let mut vm = load();
        for call in 0..n {
            let start = std::time::Instant::now();
            let start_pos = current_loc(&mut vm);
            let start_range = start_pos.and_then(|(file, offset)| {
                vm.source_text(file)
                    .map(|text| (file, line_byte_range(&text, offset)))
            });
            let mut ops = 0u32;
            let mut done = false;
            for _ in 0..1_000_000u32 {
                done = vm.debug_step().expect("step should not fault");
                ops += 1;
                if done {
                    break;
                }
                let now = current_loc(&mut vm);
                let moved = match (&start_range, now) {
                    (Some((file, range)), Some((now_file, now_offset))) => {
                        now_file != *file || !range.contains(&now_offset)
                    }
                    (None, Some(_)) | (Some(_), None) => true,
                    (None, None) => false,
                };
                if moved {
                    break;
                }
            }
            println!(
                "call {call:>3}: {ops:>7} ops, {:>8.3}ms, done={done}, start={start_pos:?} -> now={:?}",
                start.elapsed().as_secs_f64() * 1000.0,
                current_loc(&mut vm)
            );
            if done {
                break;
            }
        }
        return;
    }

    let max_steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut vm = load();

    let mut steps = 0u64;
    loop {
        let pos = vm.current_position();
        match pos {
            Some((chunk, ip, loc)) => {
                if loc.is_synthetic() {
                    println!("{steps:>6}  chunk#{} ip={ip}  <synthetic>", chunk.index());
                } else {
                    println!(
                        "{steps:>6}  chunk#{} ip={ip}  file={} span={}..{}",
                        chunk.index(),
                        loc.file_id,
                        loc.span.start,
                        loc.span.end
                    );
                }
            }
            None => println!("{steps:>6}  <no frame>"),
        }
        let done = vm.debug_step().expect("step should not fault");
        steps += 1;
        if done || steps >= max_steps {
            println!("stopped after {steps} steps, done={done}");
            break;
        }
    }
}
