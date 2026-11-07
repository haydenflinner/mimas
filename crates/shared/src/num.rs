//! Numbers as text. A float's text is a float literal that reads back
//! as the same value, so `print`, `to_str`, the `Captured` snapshot and
//! a host's inspector all agree, and a value written out can be read
//! in again — the primitive serde.

use std::fmt::Write;

// r[impl mimas.num.text]
/// The text of a float: the fewest digits that read back as the same
/// value, a whole number keeping its `.0` (`2.0`, never `2`), the
/// exponent form the lexer accepts where Rust's shortest form reaches
/// for it (`1e21`, `1e-7`), and the IEEE corners spelled as the
/// constants that name them (`float::INF`, `float::NEG_INF`,
/// `float::NAN`) rather than `inf`/`NaN`, which no literal spells.
pub fn write_float(out: &mut impl Write, f: f64) {
    if f.is_nan() {
        out.write_str("float::NAN")
    } else if f.is_infinite() {
        out.write_str(if f > 0.0 { "float::INF" } else { "float::NEG_INF" })
    } else {
        write!(out, "{f:?}")
    }
    .expect("writing to a string cannot fail");
}

/// [`write_float`] into a fresh string.
pub fn float_text(f: f64) -> String {
    let mut s = String::with_capacity(24);
    write_float(&mut s, f);
    s
}

#[cfg(test)]
mod tests {
    use super::float_text;

    // r[verify mimas.num.text]
    #[test]
    fn whole_floats_keep_their_point() {
        assert_eq!(float_text(2.0), "2.0");
        assert_eq!(float_text(0.0), "0.0");
        assert_eq!(float_text(-0.0), "-0.0");
        assert_eq!(float_text(360.0), "360.0");
    }

    #[test]
    fn fractions_use_the_shortest_digits_that_read_back() {
        assert_eq!(float_text(0.1), "0.1");
        assert_eq!(float_text(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float_text(1.5e-3), "0.0015");
    }

    #[test]
    fn the_corners_are_spelled_as_their_constants() {
        assert_eq!(float_text(f64::INFINITY), "float::INF");
        assert_eq!(float_text(f64::NEG_INFINITY), "float::NEG_INF");
        assert_eq!(float_text(f64::NAN), "float::NAN");
    }

    #[test]
    fn exponent_form_reads_back_through_rust_too() {
        for v in [1e21, 1e-7, 1e300, 5e-324, 123456789012345680000.0] {
            let text = float_text(v);
            assert!(text.contains('e'), "{text}");
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), v.to_bits(), "{text}");
        }
    }
}
