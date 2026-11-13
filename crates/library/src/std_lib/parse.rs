use std::collections::HashMap;

use macros::native;
use vm::{MimasEnum, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    {
        let mut m = api.module("std::parse");
        m.add_adt::<Value>();
        m.add(from_json);
        m.add(to_json);
    }
    api.add_method(Value::as_dict);
    api.add_method(Value::as_array);
    api.add_method(Value::as_str);
    api.add_method(Value::as_float);
    api.add_method(Value::get);
    api.add_method(Value::at);
    api.add_method(Value::str_or);
    api.add_method(Value::float_or);
    api.add_method(Value::int_or);
    api.add_method(Value::bool_or);
}

#[native]
/// Parses `text` as JSON and returns it as a `Value`. Raises if `text` isn't valid JSON, with a
/// message giving the line and column of the problem.
///
/// ```mimas
/// use std::parse;
///
/// let data = parse::from_json("{\"name\": \"ada\", \"hp\": 30}")!;
/// let name = data.as_dict()!["name"]!.as_str()!; // "ada"
/// ```
fn from_json(v: &str) -> Raisable<Value> {
    serde_json::from_str::<serde_json::Value>(v)
        .map(Into::into)
        .into()
}

#[native]
/// Returns `value` written as compact JSON, with no spaces or line breaks. Object keys are written
/// in alphabetical order, and a whole number has no decimal point (`3.0` is written as `3`). A
/// `NaN` or infinite number, which JSON can't represent, is written as `null`.
///
/// ```mimas
/// use std::parse;
/// use std::parse::Value;
///
/// let player = Value::Object({
///     name : Value::String("ada"),
///     level : Value::Number(3.0),
/// });
/// let text = parse::to_json(player)!; // "{\"level\":3,\"name\":\"ada\"}"
/// ```
fn to_json(v: Value) -> Raisable<String> {
    serde_json::to_string(&serde_json::Value::from(v)).into()
}

#[derive(MimasEnum)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    #[native]
    /// Returns the entries of an `Object`, or `null` for any other kind of value. The keys are in no
    /// particular order.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let data = parse::from_json("{\"hp\": 30}")!;
    /// let hp = data.as_dict()!["hp"]!.as_float()!; // 30.0
    /// ```
    fn as_dict(value: Value) -> Option<HashMap<String, Value>> {
        let Self::Object(dict) = value else {
            return None;
        };
        Some(dict)
    }

    #[native]
    /// Returns the elements of an `Array`, or `null` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let list = parse::from_json("[1, 2, 3]")!;
    /// let n = list.as_array()!.len(); // 3
    /// ```
    fn as_array(value: Value) -> Option<Vec<Value>> {
        let Self::Array(arr) = value else { return None };
        Some(arr)
    }

    #[native]
    /// Returns the text of a `String`, or `null` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("\"hi\"")!.as_str(); // "hi"
    /// let b = parse::from_json("12")!.as_str();     // null
    /// ```
    fn as_str(value: Value) -> Option<String> {
        let Self::String(s) = value else { return None };
        Some(s)
    }

    #[native]
    /// Returns the number in a `Number`, or `null` for any other kind of value. Every JSON number
    /// comes back as a `float`. Call `to_int` on the result when you need an `int`.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("12")!.as_float();           // 12.0
    /// let b = parse::from_json("12")!.as_float()!.to_int(); // 12
    /// ```
    fn as_float(value: Value) -> Option<f64> {
        let Self::Number(f) = value else { return None };
        Some(f)
    }

    #[native]
    /// Looks `key` up in an `Object`. A missing key, or any other kind of value, gives `Null`
    /// rather than raising, so a chain of lookups reads a nested document without a null check
    /// at every step — pair it with `str_or`/`float_or`/`int_or`/`bool_or` at the leaf.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let v = parse::from_json("{\"hero\": {\"name\": \"ada\", \"hp\": 30}}")!;
    /// let a = v.get("hero").get("name").str_or("?");  // "ada"
    /// let b = v.get("hero").get("mp").int_or(0);      // 0
    /// let c = v.get("nope").get("deeper").str_or(""); // ""
    /// ```
    fn get(value: Value, key: &str) -> Value {
        let Self::Object(mut dict) = value else {
            return Self::Null;
        };
        dict.remove(key).unwrap_or(Self::Null)
    }

    #[native]
    /// The `i`th element of an `Array`. Out of range, or any other kind of value, gives `Null`.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let v = parse::from_json("[[1, 2], [3, 4]]")!;
    /// let a = v.at(1).at(0).int_or(-1); // 3
    /// let b = v.at(5).int_or(-1);       // -1
    /// ```
    fn at(value: Value, i: i64) -> Value {
        let Self::Array(mut arr) = value else {
            return Self::Null;
        };
        if i < 0 || i as usize >= arr.len() {
            return Self::Null;
        }
        arr.swap_remove(i as usize)
    }

    #[native]
    /// The text of a `String`, or `default` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("\"hi\"")!.str_or("?"); // "hi"
    /// let b = parse::from_json("12")!.str_or("?");     // "?"
    /// ```
    fn str_or(value: Value, default: &str) -> String {
        match value {
            Self::String(s) => s,
            _ => default.to_string(),
        }
    }

    #[native]
    /// The number in a `Number`, or `default` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("1.5")!.float_or(0.0);  // 1.5
    /// let b = parse::from_json("null")!.float_or(0.0); // 0.0
    /// ```
    fn float_or(value: Value, default: f64) -> f64 {
        match value {
            Self::Number(f) => f,
            _ => default,
        }
    }

    #[native]
    /// The number in a `Number` truncated to an `int`, or `default` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("12")!.int_or(0);      // 12
    /// let b = parse::from_json("\"12\"")!.int_or(0);  // 0
    /// ```
    fn int_or(value: Value, default: i64) -> i64 {
        match value {
            Self::Number(f) => f as i64,
            _ => default,
        }
    }

    #[native]
    /// The `Bool`, or `default` for any other kind of value.
    ///
    /// ```mimas
    /// use std::parse;
    ///
    /// let a = parse::from_json("true")!.bool_or(false); // true
    /// let b = parse::from_json("1")!.bool_or(false);    // false
    /// ```
    fn bool_or(value: Value, default: bool) -> bool {
        match value {
            Self::Bool(b) => b,
            _ => default,
        }
    }
}

impl From<serde_json::Value> for Value {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(b) => Self::Bool(b),
            // saftey: without arbitrary_precision enabled, this is always Some
            serde_json::Value::Number(number) => Self::Number(number.as_f64().unwrap()),
            serde_json::Value::String(s) => Self::String(s),
            serde_json::Value::Array(values) => {
                Self::Array(values.into_iter().map(Into::into).collect())
            }

            serde_json::Value::Object(map) => {
                Self::Object(map.into_iter().map(|(k, v)| (k, v.into())).collect())
            }
        }
    }
}

impl From<Value> for serde_json::Value {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(b),
            // mimas has one number type (f64); emit whole values as ints so consumers
            // that expect integers (e.g. mdBook section numbers) round-trip.
            Value::Number(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.007e15 => {
                Self::Number((f as i64).into())
            }
            Value::Number(f) => serde_json::Number::from_f64(f).map_or(Self::Null, Self::Number),
            Value::String(s) => Self::String(s),
            Value::Array(a) => Self::Array(a.into_iter().map(Into::into).collect()),
            Value::Object(o) => Self::Object(o.into_iter().map(|(k, v)| (k, v.into())).collect()),
        }
    }
}
