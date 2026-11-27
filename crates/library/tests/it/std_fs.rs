test_run!(
    fs_read_missing_is_raised,
    r#"std::fs::read("/no/such/file/here") absolve |_| "fallback""# => r#""fallback""#
);

test_run!(
    fs_read_existing_unwraps_ok,
    r#""[package]" in std::fs::read("Cargo.toml")!"# => "true"
);

test_run!(
    fs_list_dir_missing_is_raised,
    r#"std::fs::list_dir("/no/such/dir/anywhere") absolve |_| ["fallback"]"#
        => r#"["fallback"]"#
);

test_run!(
    fs_write_missing_dir_is_raised,
    // `fs::write` answers `()!` (L8), so the handler answers `()` too
    r#"{ let raised = [false]; std::fs::write("/no/such/dir/file.txt", "x") absolve |_| { raised[0] = true; }; raised[0] }"# => "true"
);

test_fail!(
    fs_read_handler_type_must_match_inner,
    r#"let TEST_VALUE = std::fs::read("Cargo.toml") absolve |_| 0;"#
);
