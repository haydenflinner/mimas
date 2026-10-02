use clap::Parser;
use colored::Colorize;
use num_format::{Locale, ToFormattedString};
use std::{path::PathBuf, time::Duration};

mod build;
mod ice;
mod input;
mod render;
pub use input::*;

const ICE_EXIT_CODE: i32 = 101;

fn main() {
    ice::install_hook();
    let input = Cli::parse_from(massage_args(std::env::args().collect()));
    if input.color {
        unsafe {
            std::env::set_var("CLICOLOR_FORCE", "1");
        }
    }
    let status_code = match input.command {
        Some(Commands::Check { path }) => check(path, input.color),
        Some(Commands::Build { path }) => build(
            path,
            vec![],
            input.color,
            false,
            input.dump_bytes,
            input.dump_ir,
            input.time,
        ),
        Some(Commands::Hash { path, scoped, deps }) => hash(path, scoped || deps, deps),
        Some(Commands::Link { addr, blobs }) => link(&addr, blobs),
        Some(Commands::Run { path, script_args }) => build(
            path,
            script_args,
            input.color,
            true,
            input.dump_bytes,
            input.dump_ir,
            input.time,
        ),
        None => 0,
    };
    std::process::exit(status_code);
}

fn check(path: Option<PathBuf>, color: bool) -> i32 {
    let timer = std::time::Instant::now();
    let path = resolve_path(path);
    let (_vm, summary) = match ice::catch("check", || build::solve(&path)) {
        Ok(s) => s,
        Err(report) => {
            report.emit();
            return ICE_EXIT_CODE;
        }
    };
    let total_duration = timer.elapsed();

    emit_errors(&summary, color);
    report_meta_errors(&summary);

    let seperator = "-".repeat(50);
    println!("{seperator}");
    let count = summary.errors.len() + summary.io_errors.len();
    println!(
        "  {}",
        format!(
            "Found {} error{}.",
            count.to_string().bright_red().bold(),
            if count == 1 { "" } else { "s" },
        )
        .bold()
    );
    println!(
        "  {}",
        format!(
            "Ran on {} lines in {}.",
            summary.lines_parsed.to_formatted_string(&Locale::en),
            format_duration(total_duration),
        )
        .italic()
        .bright_black()
    );
    println!("{seperator}");

    i32::from(summary.had_errors())
}

/// `mimas hash`: content-hash the top-level fns of a `.mim` file (or every
/// `.mim` under a directory) and print `fn <name>  L<line>  <blake3>` per item.
/// `--scoped` prints the scoped hash instead — alpha-renamed locals, dep-hash
/// callee references — a different hash namespace than the token-level one.
/// `--deps` additionally indents each item's relocation sites beneath it:
/// `-> <name>  <dep hash>` per `@dep:` edge, `-> @self` for recursion.
fn hash(path: Option<PathBuf>, scoped: bool, deps: bool) -> i32 {
    let path = resolve_path(path);
    let (paths, io_errors) = if path.is_file() {
        (vec![path], vec![])
    } else {
        solve::mim_files(&path)
    };
    let mut io_errors = io_errors;
    let mut n_items = 0usize;
    for file_path in &paths {
        let source = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                io_errors.push(std::io::Error::new(
                    e.kind(),
                    format!("{}: {e}", file_path.display()),
                ));
                continue;
            }
        };
        let items = hash::extract(&source);
        // `use "name";` resolves to a sibling `<name>.mim` — the same
        // page-include BFS the lit host runs.
        let dir = file_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let globals = scoped.then(|| {
            hash::Globals::for_page(&source, |name| {
                if name.contains('/') || name.contains('\\') {
                    return None;
                }
                std::fs::read_to_string(dir.join(format!("{name}.mim"))).ok()
            })
        });
        let prefix = if paths.len() > 1 {
            format!("{}: ", file_path.display())
        } else {
            String::new()
        };
        for item in &items {
            let digest = match &globals {
                Some(globals) => item.scoped_hash(globals),
                None => item.hash(),
            };
            println!(
                "{prefix}fn {}  L{}  {}",
                item.name,
                item.line(&source),
                digest
            );
            if deps && let Some(globals) = &globals {
                for reloc in item.relocs(globals).unwrap_or_default() {
                    match reloc {
                        hash::Reloc::Dep { name, hash, .. } => {
                            println!("{prefix}  -> {name}  {hash}");
                        }
                        hash::Reloc::SelfRef { .. } => {
                            println!("{prefix}  -> @self");
                        }
                    }
                }
            }
            n_items += 1;
        }
    }
    if !io_errors.is_empty() {
        println!(
            "\n{}: The following errors occurred while trying to read files...",
            "error".bright_red().bold()
        );
        io_errors.iter().for_each(|error| {
            println!("{error}");
        });
        return 1;
    }
    if n_items == 0 {
        println!("no fn items");
    }
    0
}

/// `mimas link <addr> [--blobs <dir>]` — assemble the module a scoped
/// address names: fetch its blob, follow `@dep:` edges breadth-first,
/// patch the markers, print the self-contained source. `--blobs` names
/// the base dir (`<dir>-scoped` is read) or a scoped dir outright.
fn link(addr: &str, blobs: Option<PathBuf>) -> i32 {
    let dir = match scoped_dir(blobs) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let fetch = |key: &str| {
        let path = dir.join(key);
        std::fs::read_to_string(&path).ok()
    };
    match hash::link::link(addr, fetch) {
        Ok(linked) => {
            print!("{}", linked.source);
            if !linked.source.ends_with('\n') {
                println!();
            }
            0
        }
        Err(e) => {
            eprintln!("link {addr}: {e}");
            1
        }
    }
}

/// `--blobs` (or its default) → the scoped blob dir: `<dir>-scoped` when
/// the name isn't already scoped, `./blobs-scoped` when no flag is given.
fn scoped_dir(blobs: Option<PathBuf>) -> Result<PathBuf, String> {
    let base = blobs.unwrap_or_else(|| PathBuf::from("blobs"));
    let scoped = if base
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with("-scoped"))
    {
        base.clone()
    } else {
        let mut name = base.as_os_str().to_os_string();
        name.push("-scoped");
        PathBuf::from(name)
    };
    if scoped.is_dir() {
        Ok(scoped)
    } else {
        Err(format!(
            "no scoped blob dir at {} (looked next to {})",
            scoped.display(),
            base.display()
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    path: Option<PathBuf>,
    script_args: Vec<String>,
    color: bool,
    execute: bool,
    disasm: bool,
    dump_ir: bool,
    time: bool,
) -> i32 {
    let timer = std::time::Instant::now();
    let path = resolve_path(path);

    let (mut vm, summary) = match ice::catch("check", || build::solve(&path)) {
        Ok(s) => s,
        Err(report) => {
            report.emit();
            return ICE_EXIT_CODE;
        }
    };
    emit_errors(&summary, color);
    report_meta_errors(&summary);
    if summary.had_errors() {
        return 1;
    }

    let lines = summary.lines_parsed;
    let (stmts, solver, sources, intrinsics) = summary.into_compilation();
    let srcs: std::collections::HashMap<usize, std::sync::Arc<str>> = if disasm {
        sources
            .iter()
            .map(|(&id, ns)| (id, ns.inner().clone()))
            .collect()
    } else {
        std::collections::HashMap::new()
    };
    let program = match ice::catch("compilation", || {
        let resolutions = solve::Resolutions::from(solver);
        let mut ir = compile::Ir::new(resolutions, intrinsics);
        ir.lower(&stmts);
        if dump_ir {
            println!("{ir}");
        }
        compile::Compiler::new()
            .with_disasm(disasm)
            .with_sources(srcs)
            .compile(ir)
    }) {
        Ok(p) => p,
        Err(report) => {
            report.emit();
            return ICE_EXIT_CODE;
        }
    };
    if execute {
        let mut resolved_args = Vec::with_capacity(script_args.len() + 1);
        resolved_args.push(path.to_string_lossy().into_owned());
        resolved_args.extend(script_args);
        vm.fixture::<library::ScriptArgs>().set(resolved_args);

        let result = ice::catch("execution", move || {
            vm.load_program(program);
            vm.set_sources(sources);
            vm.run()
        });
        match result {
            Err(report) => {
                report.emit();
                return ICE_EXIT_CODE;
            }
            Ok(Err(report)) => {
                emit_runtime_error(&report, color);
                return 1;
            }
            Ok(Ok(())) => {
                if time {
                    eprintln!(
                        "{}",
                        format!("ran in {}", format_duration(timer.elapsed()))
                            .italic()
                            .bright_black()
                    );
                }
            }
        }
    }

    if !execute {
        let seperator = "-".repeat(50);
        println!("{seperator}");
        println!(
            "  {}",
            format!(
                "Compiled {} lines in {}.",
                lines.to_formatted_string(&Locale::en),
                format_duration(timer.elapsed()),
            )
            .italic()
            .bright_black()
        );
        println!("{seperator}");
    }

    0
}

// bare `mimas foo.mim` means `mimas run foo.mim`; inject `run` when the first
// positional isn't already a subcommand. `mimas` alone still falls through to help.
fn massage_args(mut args: Vec<String>) -> Vec<String> {
    const SUBCOMMANDS: [&str; 6] = ["check", "build", "hash", "run", "link", "help"];
    if let Some(idx) = args.iter().skip(1).position(|a| !a.starts_with('-')) {
        let idx = idx + 1;
        if !SUBCOMMANDS.contains(&args[idx].as_str()) {
            args.insert(idx, "run".to_string());
        }
    }
    args
}

fn resolve_path(path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(|| std::env::current_dir().expect("Cannot access the current directory!"))
}

fn emit_runtime_error(report: &miette::Report, color: bool) {
    render::emit(report.as_ref(), color);
}

fn emit_errors(summary: &build::SolveSummary, color: bool) {
    for error in &summary.errors {
        render::emit(error.as_ref(), color);
    }
}

fn report_meta_errors(summary: &build::SolveSummary) {
    if !summary.io_errors.is_empty() {
        println!(
            "\n{}: The following errors occurred while trying to read your project's files...",
            "error".bright_red().bold()
        );
        summary.io_errors.iter().for_each(|error| {
            println!("{error}");
        })
    }
}

fn format_duration(d: Duration) -> String {
    if d.as_micros() < 1000 {
        format!("{}µs", d.as_micros())
    } else if d.as_millis() < 1000 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{:.2}s", d.as_secs_f32())
    }
}
