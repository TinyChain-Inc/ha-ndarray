//! The OpenCL platform

use lazy_static::lazy_static;
use ocl::OclPrm;

use crate::access::{AccessBuf, AccessOp};
use crate::host::VEC_MIN_SIZE;
use crate::Float;

use programs::{ElementDual, ElementUnary};

pub use buffer::*;
pub use platform::{OpenCL, ACC_MIN_SIZE, GPU_MIN_SIZE};

mod buffer;
pub mod ops;
mod platform;
mod programs;

const TILE_SIZE: usize = 8;

const WG_SIZE: usize = 64;

fn real_bool(name: &'static str) -> String {
    format!("{name} != 0")
}

fn real_bool_cmp(op: &'static str) -> String {
    format!(
        "return ({lhs}) {op} ({rhs});",
        lhs = real_bool("lhs"),
        rhs = real_bool("rhs")
    )
}

fn real_cmp(op: &'static str) -> String {
    format!("return lhs {op} rhs;")
}

fn real_trig<T: Float>(name: &'static str) -> ElementUnary {
    debug_assert!(name.starts_with('_'));
    ElementUnary::new::<T, T, _>(name, format!("return {}(n);", &name[1..]))
}

#[cfg(feature = "complex")]
fn complex_bool(name: &'static str) -> String {
    format!("{name}.x != 0 || {name}.y != 0")
}

#[cfg(feature = "complex")]
fn complex_bool_cmp(op: &'static str) -> String {
    format!(
        "return ({lhs}) {op} ({rhs});",
        lhs = complex_bool("lhs"),
        rhs = complex_bool("rhs")
    )
}

#[cfg(feature = "complex")]
fn complex_cmp(cmp: &'static str, cond: &'static str) -> String {
    format!("return (lhs.x {cmp} rhs.x) {cond} (lhs.y {cmp} rhs.y);")
}

// TODO: can the `format!(...)` implementations be made static using const_format?
fn cast_body(input: &'static str, output: &'static str) -> String {
    fn scalar(input: &str, output: &str, value: &str) -> String {
        let is_float = |t: &str| matches!(t, "float" | "double");
        let is_signed = |t: &str| matches!(t, "char" | "short" | "int" | "long");
        let unsigned = |t| match t {
            "char" => "uchar",
            "short" => "ushort",
            "int" => "uint",
            "long" => "ulong",
            _ => t,
        };
        let signed = |t| match t {
            "uchar" => "short",
            "ushort" => "short",
            "uint" => "int",
            "ulong" => "long",
            _ => t,
        };
        if is_float(input) && !is_float(output) {
            let mid = match (input, is_signed(output)) {
                ("float", true) => "int",
                ("float", false) => "uint",
                (_, true) => "long",
                (_, false) => "ulong",
            };
            let converted = format!("(isnan({value}) ? ({mid})0 : convert_{mid}_sat_rtz({value}))");
            return format!("convert_{output}({converted})");
        }
        if !is_float(input) && is_float(output) {
            let mid = if matches!(input, "long" | "ulong") {
                "double"
            } else {
                "float"
            };
            return format!("convert_{output}_rte(convert_{mid}_rte({value}))");
        }
        if !is_float(input) && !is_float(output) && is_signed(input) != is_signed(output) {
            let mid = if is_signed(output) {
                signed(input)
            } else {
                unsigned(input)
            };
            return format!("convert_{output}(convert_{mid}({value}))");
        }
        if is_float(output) {
            format!("convert_{output}_rte({value})")
        } else {
            format!("convert_{output}({value})")
        }
    }
    let input_complex = input.ends_with('2');
    let output_complex = output.ends_with('2');
    let it = input.trim_end_matches('2');
    let ot = output.trim_end_matches('2');
    let re = scalar(it, ot, if input_complex { "n.x" } else { "n" });
    if output_complex {
        let im = if input_complex {
            scalar(it, ot, "n.y")
        } else {
            format!("({ot})0")
        };
        format!("return ({output})({re}, {im});")
    } else {
        format!("return {re};")
    }
}

pub trait CLElement: OclPrm {
    const REAL: bool;
    const TYPE: &'static str;

    // basic arithmetic (unary)
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return abs(n);")
    }

    fn cl_exp() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_exp", "return exp(n);")
    }

    fn cl_ln() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("ln", "return log(n);")
    }

    // basic arithmetic (dual)
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>("add", "return lhs + rhs;")
    }

    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "div",
            "if (rhs == 0) { return 0; } else { return lhs / rhs; }",
        )
    }

    fn cl_log() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_log", "return log(lhs) / log(rhs);")
    }

    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>("mul", "return lhs * rhs;")
    }

    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "return pow(lhs, rhs);")
    }

    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>("sub", "return lhs - rhs;")
    }

    // boolean logic (unary)
    fn cl_not() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("not", "return n == 0 ? 1 : 0;")
    }

    // boolean logic (dual)
    fn cl_and() -> ElementDual {
        ElementDual::new::<Self, u8, _>("and", real_bool_cmp("&&"))
    }

    fn cl_or() -> ElementDual {
        ElementDual::new::<Self, u8, _>("or", real_bool_cmp("||"))
    }

    fn cl_xor() -> ElementDual {
        ElementDual::new::<Self, u8, _>("xor", real_bool_cmp("^"))
    }

    // casting
    fn cl_cast<O: CLElement>() -> ElementUnary {
        ElementUnary::new::<Self, O, _>("_cast", cast_body(Self::TYPE, O::TYPE))
    }

    // comparison
    fn cl_eq() -> ElementDual {
        ElementDual::new::<Self, u8, _>("eq", real_cmp("=="))
    }

    fn cl_ne() -> ElementDual {
        ElementDual::new::<Self, u8, _>("ne", real_cmp("!="))
    }

    // floating-point identities
    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return false;")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return false;")
    }
}

pub trait CLElementComplex: CLElement {
    fn cl_angle() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("angle", "return atan2(n.y, n.x);")
    }

    fn cl_conj() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("conj", format!("return ({})(n.x, -n.y);", Self::TYPE))
    }

    fn cl_real() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("real", "return n.x;")
    }

    fn cl_imag() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("imag", "return n.y;")
    }
}

pub trait CLElementReal: CLElement {
    // basic arithmetic (dual)
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "return mod(lhs, rhs);")
    }

    // rounding
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return round(n);")
    }

    // comparison
    fn cl_ge() -> ElementDual {
        ElementDual::new::<Self, u8, _>("ge", real_cmp(">="))
    }

    fn cl_gt() -> ElementDual {
        ElementDual::new::<Self, u8, _>("gt", real_cmp(">"))
    }

    fn cl_le() -> ElementDual {
        ElementDual::new::<Self, u8, _>("le", real_cmp("<="))
    }

    fn cl_lt() -> ElementDual {
        ElementDual::new::<Self, u8, _>("lt", real_cmp("<"))
    }

    fn cl_max() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_max", "return max(lhs, rhs);")
    }

    fn cl_min() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_min", "return min(lhs, rhs);")
    }
}

pub trait CLElementTrig {
    // trigonometry
    fn cl_sin() -> ElementUnary;

    fn cl_asin() -> ElementUnary;

    fn cl_sinh() -> ElementUnary;

    fn cl_cos() -> ElementUnary;

    fn cl_acos() -> ElementUnary;

    fn cl_cosh() -> ElementUnary;

    fn cl_tan() -> ElementUnary;

    fn cl_atan() -> ElementUnary;

    fn cl_tanh() -> ElementUnary;
}

macro_rules! cl_trig_real {
    ($t:ty) => {
        impl CLElementTrig for $t {
            fn cl_sin() -> ElementUnary {
                real_trig::<Self>("_sin")
            }

            fn cl_asin() -> ElementUnary {
                real_trig::<Self>("_asin")
            }

            fn cl_sinh() -> ElementUnary {
                real_trig::<Self>("_sinh")
            }

            fn cl_cos() -> ElementUnary {
                real_trig::<Self>("_cos")
            }

            fn cl_acos() -> ElementUnary {
                real_trig::<Self>("_acos")
            }

            fn cl_cosh() -> ElementUnary {
                real_trig::<Self>("_cosh")
            }

            fn cl_tan() -> ElementUnary {
                real_trig::<Self>("_tan")
            }

            fn cl_atan() -> ElementUnary {
                real_trig::<Self>("_atan")
            }

            fn cl_tanh() -> ElementUnary {
                real_trig::<Self>("_tanh")
            }
        }
    };
}

impl CLElement for f32 {
    const REAL: bool = true;
    const TYPE: &'static str = "float";

    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return fabs(n);")
    }

    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "return lhs / rhs;")
    }

    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return isinf(n);")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return isnan(n);")
    }
}

impl CLElementReal for f32 {
    fn cl_max() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_max",
            "if (isnan(lhs) || isnan(rhs)) return NAN; if (lhs == 0 && rhs == 0) return signbit(lhs) && signbit(rhs) ? -0.0f : 0.0f; return fmax(lhs, rhs);")
    }
    fn cl_min() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_min",
            "if (isnan(lhs) || isnan(rhs)) return NAN; if (lhs == 0 && rhs == 0) return signbit(lhs) || signbit(rhs) ? -0.0f : 0.0f; return fmin(lhs, rhs);")
    }

    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "return fmod(lhs, rhs);")
    }
}

cl_trig_real!(f32);

impl CLElement for f64 {
    const REAL: bool = true;
    const TYPE: &'static str = "double";

    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return fabs(n);")
    }

    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "return lhs / rhs;")
    }

    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return isinf(n);")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return isnan(n);")
    }
}

impl CLElementReal for f64 {
    fn cl_max() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_max",
            "if (isnan(lhs) || isnan(rhs)) return NAN; if (lhs == 0 && rhs == 0) return signbit(lhs) && signbit(rhs) ? -0.0f : 0.0f; return fmax(lhs, rhs);")
    }
    fn cl_min() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_min",
            "if (isnan(lhs) || isnan(rhs)) return NAN; if (lhs == 0 && rhs == 0) return signbit(lhs) || signbit(rhs) ? -0.0f : 0.0f; return fmin(lhs, rhs);")
    }

    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "return fmod(lhs, rhs);")
    }
}

cl_trig_real!(f64);

impl CLElement for i8 {
    const REAL: bool = true;
    const TYPE: &'static str = "char";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return as_char((uchar)((ulong)(uchar)lhs + (ulong)(uchar)rhs));",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return as_char((uchar)((ulong)(uchar)lhs - (ulong)(uchar)rhs));",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return as_char((uchar)((ulong)(uchar)lhs * (ulong)(uchar)rhs));",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0; if (lhs == ((char)(((uchar)1) << 7)) && rhs == -1) return lhs; return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "if (rhs < 0) { if (lhs == 1) return 1; if (lhs == -1) return (rhs & 1) ? -1 : 1; return 0; } ulong b = (uchar)lhs; ulong e = (uchar)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return as_char((uchar)(r));")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>(
            "_abs",
            "return n < 0 ? as_char((uchar)(0UL - (ulong)(uchar)n)) : n;",
        )
    }
}
impl CLElementReal for i8 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0; if (lhs == ((char)(((uchar)1) << 7)) && rhs == -1) return 0; return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for i16 {
    const REAL: bool = true;
    const TYPE: &'static str = "short";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return as_short((ushort)((ulong)(ushort)lhs + (ulong)(ushort)rhs));",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return as_short((ushort)((ulong)(ushort)lhs - (ulong)(ushort)rhs));",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return as_short((ushort)((ulong)(ushort)lhs * (ulong)(ushort)rhs));",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0; if (lhs == ((short)(((ushort)1) << 15)) && rhs == -1) return lhs; return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "if (rhs < 0) { if (lhs == 1) return 1; if (lhs == -1) return (rhs & 1) ? -1 : 1; return 0; } ulong b = (ushort)lhs; ulong e = (ushort)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return as_short((ushort)(r));")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>(
            "_abs",
            "return n < 0 ? as_short((ushort)(0UL - (ulong)(ushort)n)) : n;",
        )
    }
}
impl CLElementReal for i16 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0; if (lhs == ((short)(((ushort)1) << 15)) && rhs == -1) return 0; return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for i32 {
    const REAL: bool = true;
    const TYPE: &'static str = "int";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return as_int((uint)((ulong)(uint)lhs + (ulong)(uint)rhs));",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return as_int((uint)((ulong)(uint)lhs - (ulong)(uint)rhs));",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return as_int((uint)((ulong)(uint)lhs * (ulong)(uint)rhs));",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0; if (lhs == ((int)(((uint)1) << 31)) && rhs == -1) return lhs; return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "if (rhs < 0) { if (lhs == 1) return 1; if (lhs == -1) return (rhs & 1) ? -1 : 1; return 0; } ulong b = (uint)lhs; ulong e = (uint)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return as_int((uint)(r));")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>(
            "_abs",
            "return n < 0 ? as_int((uint)(0UL - (ulong)(uint)n)) : n;",
        )
    }
}
impl CLElementReal for i32 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0; if (lhs == ((int)(((uint)1) << 31)) && rhs == -1) return 0; return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for i64 {
    const REAL: bool = true;
    const TYPE: &'static str = "long";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return as_long((ulong)((ulong)(ulong)lhs + (ulong)(ulong)rhs));",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return as_long((ulong)((ulong)(ulong)lhs - (ulong)(ulong)rhs));",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return as_long((ulong)((ulong)(ulong)lhs * (ulong)(ulong)rhs));",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0; if (lhs == ((long)(((ulong)1) << 63)) && rhs == -1) return lhs; return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "if (rhs < 0) { if (lhs == 1) return 1; if (lhs == -1) return (rhs & 1) ? -1 : 1; return 0; } ulong b = (ulong)lhs; ulong e = (ulong)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return as_long((ulong)(r));")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>(
            "_abs",
            "return n < 0 ? as_long((ulong)(0UL - (ulong)(ulong)n)) : n;",
        )
    }
}
impl CLElementReal for i64 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0; if (lhs == ((long)(((ulong)1) << 63)) && rhs == -1) return 0; return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for u8 {
    const REAL: bool = true;
    const TYPE: &'static str = "uchar";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return (uchar)((ulong)(uchar)lhs + (ulong)(uchar)rhs);",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return (uchar)((ulong)(uchar)lhs - (ulong)(uchar)rhs);",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return (uchar)((ulong)(uchar)lhs * (ulong)(uchar)rhs);",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0;  return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", " ulong b = (uchar)lhs; ulong e = (uchar)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return (uchar)(r);")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return n;")
    }
}
impl CLElementReal for u8 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0;  return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for u16 {
    const REAL: bool = true;
    const TYPE: &'static str = "ushort";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return (ushort)((ulong)(ushort)lhs + (ulong)(ushort)rhs);",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return (ushort)((ulong)(ushort)lhs - (ulong)(ushort)rhs);",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return (ushort)((ulong)(ushort)lhs * (ulong)(ushort)rhs);",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0;  return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", " ulong b = (ushort)lhs; ulong e = (ushort)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return (ushort)(r);")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return n;")
    }
}
impl CLElementReal for u16 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0;  return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for u32 {
    const REAL: bool = true;
    const TYPE: &'static str = "uint";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return (uint)((ulong)(uint)lhs + (ulong)(uint)rhs);",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return (uint)((ulong)(uint)lhs - (ulong)(uint)rhs);",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return (uint)((ulong)(uint)lhs * (ulong)(uint)rhs);",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0;  return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", " ulong b = (uint)lhs; ulong e = (uint)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return (uint)(r);")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return n;")
    }
}
impl CLElementReal for u32 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0;  return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}
impl CLElement for u64 {
    const REAL: bool = true;
    const TYPE: &'static str = "ulong";
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "add",
            "return (ulong)((ulong)(ulong)lhs + (ulong)(ulong)rhs);",
        )
    }
    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "sub",
            "return (ulong)((ulong)(ulong)lhs - (ulong)(ulong)rhs);",
        )
    }
    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "mul",
            "return (ulong)((ulong)(ulong)lhs * (ulong)(ulong)rhs);",
        )
    }
    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>("div", "if (rhs == 0) return 0;  return lhs / rhs;")
    }
    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", " ulong b = (ulong)lhs; ulong e = (ulong)rhs; ulong r = 1; while (e != 0) { if (e & 1) r *= b; e >>= 1; b *= b; } return (ulong)(r);")
    }
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return n;")
    }
}
impl CLElementReal for u64 {
    fn cl_rem() -> ElementDual {
        ElementDual::new::<Self, Self, _>("rem", "if (rhs == 0) return 0;  return lhs % rhs;")
    }
    fn cl_round() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_round", "return n;")
    }
}

#[cfg(feature = "complex")]
macro_rules! cl_complex {
    ($t:ty, $ct:expr) => {
        impl CLElement for num_complex::Complex<$t> {
            const REAL: bool = false;
            const TYPE: &'static str = $ct;

            fn cl_not() -> ElementUnary {
                ElementUnary::new::<Self, u8, _>("not", "return n.x == 0 && n.y == 0;")
            }
            fn cl_abs() -> ElementUnary {
                ElementUnary::new::<Self, $t, _>("_abs", "return hypot(n.x, n.y);")
            }
            // basic arithmetic (dual)
            fn cl_div() -> ElementDual {
                ElementDual::new::<Self, Self, _>(
                    "div",
                    format!(
                        "
                        if (rhs.x == 0.0f && rhs.y == 0.0f) {{
                            return ({c_type})(NAN, NAN);
                        }} else {{
                            {r_type} denom = (rhs.x * rhs.x) + (rhs.y * rhs.y);
                            {r_type} re = ((lhs.x * rhs.x) + (lhs.y * rhs.y)) / denom;
                            {r_type} im = ((lhs.y * rhs.x) - (lhs.x * rhs.y)) / denom;
                            return ({c_type})(re, im);
                        }}
                        ",
                        c_type = Self::TYPE,
                        r_type = <$t>::TYPE,
                    ),
                )
            }

            fn cl_mul() -> ElementDual {
                ElementDual::new::<Self, Self, _>(
                    "mul",
                    format!(
                        "
                        {r_type} re = ((lhs.x * rhs.x) - (lhs.y * rhs.y));
                        {r_type} im = ((lhs.x * rhs.y) + (lhs.y * rhs.x));
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$t>::TYPE,
                    ),
                )
            }

            fn cl_log() -> ElementDual {
                ElementDual::new::<Self, Self, _>("_log", format!(
                    "{r} a = log(hypot(lhs.x, lhs.y)); {r} b = atan2(lhs.y, lhs.x);
                     {r} c = log(hypot(rhs.x, rhs.y)); {r} d = atan2(rhs.y, rhs.x);
                     {r} denom = c*c + d*d;
                     return ({ct})((a*c+b*d)/denom, (b*c-a*d)/denom);",
                    r=<$t>::TYPE, ct=Self::TYPE
                ))
            }

            fn cl_pow() -> ElementDual {
                ElementDual::new::<Self, Self, _>(
                    "_pow",
                    format!(
                        "
                        if (rhs.x == 0 && rhs.y == 0) return ({c_type})(1, 0);
                        // log_lhs = log(lhs)
                        {r_type} norm = hypot(lhs.x, lhs.y);
                        {r_type} angle = atan2(lhs.y, lhs.x);
                        {c_type} log_lhs = ({c_type})(log(norm), angle);

                        // product = rhs * log(lhs)
                        {r_type} product_r = ((rhs.x * log_lhs.x) - (rhs.y * log_lhs.y));
                        {r_type} product_i = ((rhs.x * log_lhs.y) + (rhs.y * log_lhs.x));

                        if (isinf(product_r)) {{
                            if (product_r < 0 && !isfinite(product_i)) return ({c_type})(0, 0);
                            if (product_r > 0 && (product_i == 0 || !isfinite(product_i)))
                                return ({c_type})(product_r, isinf(product_i) ? NAN : product_i);
                        }} else if (isnan(product_r) && product_i == 0) return ({c_type})(product_r, product_i);
                        // return exp(product)
                        {r_type} r = exp(product_r);
                        {c_type} c = ({c_type})(cos(product_i), sin(product_i));

                        {r_type} re = r * c.x;
                        {r_type} im = r * c.y;
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$t>::TYPE,
                    ),
                )
            }

            // basic arithmetic (unary)
            fn cl_exp() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_exp",
                    format!(
                        "
if (isinf(n.x)) {{
                            if (n.x < 0 && !isfinite(n.y)) return ({c_type})(0, 0);
                            if (n.x > 0 && (n.y == 0 || !isfinite(n.y)))
                                return ({c_type})(n.x, isinf(n.y) ? NAN : n.y);
                        }} else if (isnan(n.x) && n.y == 0) return ({c_type})(n.x, n.y);
                        {r_type} lhs = exp(n.x);
                        {c_type} rhs = ({c_type})(cos(n.y), sin(n.y));

                        {r_type} re = lhs * rhs.x;
                        {r_type} im = lhs * rhs.y;
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$t>::TYPE,
                    ),
                )
            }

            fn cl_ln() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "ln",
                    format!(
                        "
                        {r_type} norm = hypot(n.x, n.y);
                        {r_type} angle = atan2(n.y, n.x);
                        return ({c_type})(log(norm), angle);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$t>::TYPE,
                    ),
                )
            }

            // boolean logic
            fn cl_and() -> ElementDual {
                ElementDual::new::<Self, u8, _>("and", complex_bool_cmp("&&"))
            }

            fn cl_or() -> ElementDual {
                ElementDual::new::<Self, u8, _>("or", complex_bool_cmp("||"))
            }

            fn cl_xor() -> ElementDual {
                ElementDual::new::<Self, u8, _>("xor", complex_bool_cmp("^"))
            }

            // comparison
            fn cl_eq() -> ElementDual {
                ElementDual::new::<Self, u8, _>("eq", complex_cmp("==", "&&"))
            }

            fn cl_ne() -> ElementDual {
                ElementDual::new::<Self, u8, _>("ne", complex_cmp("!=", "||"))
            }

            // special cases
            fn cl_inf() -> ElementUnary {
                ElementUnary::new::<Self, u8, _>("_isinf", "return isinf(n.x) || isinf(n.y);")
            }

            fn cl_nan() -> ElementUnary {
                ElementUnary::new::<Self, u8, _>("_isnan", "return isnan(n.x) || isnan(n.y);")
            }
        }

        impl CLElementComplex for num_complex::Complex<$t> {
            fn cl_angle() -> ElementUnary { ElementUnary::new::<Self, $t, _>("angle", "return atan2(n.y, n.x);") }
            fn cl_real() -> ElementUnary { ElementUnary::new::<Self, $t, _>("real", "return n.x;") }
            fn cl_imag() -> ElementUnary { ElementUnary::new::<Self, $t, _>("imag", "return n.y;") }
        }
    };
}

#[cfg(feature = "complex")]
cl_complex!(f32, "float2");
#[cfg(feature = "complex")]
cl_complex!(f64, "double2");

#[cfg(feature = "complex")]
macro_rules! cl_trig_complex {
    ($t:ty, $r:ty) => {
        impl CLElementTrig for $t {
            fn cl_sin() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_sin",
                    format!(
                        "
                        {r_type} re = (sin(n.x) * cosh(n.y));
                        {r_type} im = (cos(n.x) * sinh(n.y));
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_asin() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_asin",
                    format!(
                        "
                        // asin(z) = -i * log(i z + sqrt(1 - z^2))
                        {r_type} a = n.x;
                        {r_type} b = n.y;

                        // z^2
                        {r_type} z2_re = (a * a) - (b * b);
                        {r_type} z2_im = a * b + b * a;

                        // w = 1 - z^2
                        {r_type} w_re = ({r_type})1.0f - z2_re;
                        {r_type} w_im = ({r_type})0 - z2_im;

                        // sqrt(w)
                        {r_type} w_norm = hypot(w_re, w_im);
                        // Avoid cancellation in the small component of sqrt(w).
                        {r_type} t = sqrt((w_norm + fabs(w_re)) * ({r_type})0.5f);
                        {r_type} sqrt_re = w_re >= 0 ? t : (t == 0 ? 0 : fabs(w_im) / (2*t));
                        {r_type} sqrt_im = w_re >= 0 ? (t == 0 ? w_im : w_im / (2*t)) : copysign(t, w_im);

                        // i z
                        {r_type} iz_re = ({r_type})0 * a - b;
                        {r_type} iz_im = ({r_type})0 * b + a;

                        // i z + sqrt(1 - z^2)
                        {r_type} s_re = iz_re + sqrt_re;
                        {r_type} s_im = iz_im + sqrt_im;

                        // log(s)
                        {r_type} s_norm = hypot(s_re, s_im);
                        {r_type} log_re = log(s_norm);
                        {r_type} log_im = atan2(s_im, s_re);

                        // -i * log(s); -i has a negative-zero real component.
                        return ({c_type})(({r_type})(-0.0f) * log_re + log_im, ({r_type})(-0.0f) * log_im - log_re);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_sinh() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_sinh",
                    format!(
                        "
                        {r_type} re = (sinh(n.x) * cos(n.y));
                        {r_type} im = (cosh(n.x) * sin(n.y));
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_cos() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_cos",
                    format!(
                        "
                        {r_type} re = (cos(n.x) * cosh(n.y));
                        {r_type} im = -(sin(n.x) * sinh(n.y));
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_acos() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_acos",
                    format!(
                        "
                        // acos(z) = -i * log(z + i * sqrt(1 - z^2))
                        {r_type} a = n.x;
                        {r_type} b = n.y;

                        // z^2
                        {r_type} z2_re = (a * a) - (b * b);
                        {r_type} z2_im = a * b + b * a;

                        // w = 1 - z^2
                        {r_type} w_re = ({r_type})1.0f - z2_re;
                        {r_type} w_im = ({r_type})0 - z2_im;

                        // sqrt(w)
                        {r_type} w_norm = hypot(w_re, w_im);
                        // Avoid cancellation in the small component of sqrt(w).
                        {r_type} t = sqrt((w_norm + fabs(w_re)) * ({r_type})0.5f);
                        {r_type} sqrt_re = w_re >= 0 ? t : (t == 0 ? 0 : fabs(w_im) / (2*t));
                        {r_type} sqrt_im = w_re >= 0 ? (t == 0 ? w_im : w_im / (2*t)) : copysign(t, w_im);

                        // i * sqrt(w) = (-sqrt_im) + i * sqrt_re
                        {r_type} iz_re = ({r_type})0 * sqrt_re - sqrt_im;
                        {r_type} iz_im = ({r_type})0 * sqrt_im + sqrt_re;

                        // z + i * sqrt(1 - z^2)
                        {r_type} s_re = a + iz_re;
                        {r_type} s_im = b + iz_im;

                        // log(s)
                        {r_type} s_norm = hypot(s_re, s_im);
                        {r_type} log_re = log(s_norm);
                        {r_type} log_im = atan2(s_im, s_re);

                        // -i * log(s); -i has a negative-zero real component.
                        return ({c_type})(({r_type})(-0.0f) * log_re + log_im, ({r_type})(-0.0f) * log_im - log_re);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_cosh() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_cosh",
                    format!(
                        "
                        {r_type} re = (cosh(n.x) * cos(n.y));
                        {r_type} im = (sinh(n.x) * sin(n.y));
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_tan() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_tan",
                    format!(
                        "
                        // tan(a + i b) = (sin(2a) + i sinh(2b)) / (cos(2a) + cosh(2b))
                        {r_type} r2 = ({r_type})2.0f * n.x;
                        {r_type} i2 = ({r_type})2.0f * n.y;
                        {r_type} denom = cos(r2) + cosh(i2);

                        {r_type} re = sin(r2) / denom;
                        {r_type} im = sinh(i2) / denom;
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_atan() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_atan",
                    format!(
                        "
                        if (n.x == 0 && fabs(n.y) == 1) return ({c_type})(0, copysign(({r_type})INFINITY, n.y));
                        // atan(z) = (i / 2) * (log(1 - i z) - log(1 + i z))
                        {r_type} a = n.x;
                        {r_type} b = n.y;

                        // 1 - i z = (1 + b) - i a
                        {r_type} w1_re = ({r_type})1 - (({r_type})0 * a - b);
                        {r_type} w1_im = ({r_type})0 - (({r_type})0 * b + a);

                        // 1 + i z = (1 - b) + i a
                        {r_type} w2_re = ({r_type})1 + (({r_type})0 * a - b);
                        {r_type} w2_im = ({r_type})0 + (({r_type})0 * b + a);

                        {r_type} w1_norm = hypot(w1_re, w1_im);
                        {r_type} u1 = log(w1_norm);
                        {r_type} v1 = atan2(w1_im, w1_re);

                        {r_type} w2_norm = hypot(w2_re, w2_im);
                        {r_type} u2 = log(w2_norm);
                        {r_type} v2 = atan2(w2_im, w2_re);

                        // Divide log(1+iz)-log(1-iz) by 2i, retaining num-complex's
                        // zero products and exceptional-value conventions.
                        {r_type} p = u2 - u1;
                        {r_type} q = v2 - v1;
                        {r_type} re = (p * ({r_type})0 + q * ({r_type})2) / ({r_type})4;
                        {r_type} im = (q * ({r_type})0 - p * ({r_type})2) / ({r_type})4;
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }

            fn cl_tanh() -> ElementUnary {
                ElementUnary::new::<Self, Self, _>(
                    "_tanh",
                    format!(
                        "
                        // tanh(a + i b) = (sinh(2a) + i sin(2b)) / (cosh(2a) + cos(2b))
                        {r_type} r2 = ({r_type})2.0f * n.x;
                        {r_type} i2 = ({r_type})2.0f * n.y;
                        {r_type} denom = cosh(r2) + cos(i2);

                        {r_type} re = sinh(r2) / denom;
                        {r_type} im = sin(i2) / denom;
                        return ({c_type})(re, im);
                        ",
                        c_type = Self::TYPE,
                        r_type = <$r>::TYPE,
                    ),
                )
            }
        }
    };
}

#[cfg(feature = "complex")]
cl_trig_complex!(num_complex::Complex<f32>, f32);
#[cfg(feature = "complex")]
cl_trig_complex!(num_complex::Complex<f64>, f64);

lazy_static! {
    pub static ref CL_PLATFORM: platform::CLPlatform =
        platform::CLPlatform::default().expect("OpenCL platform");
}

pub type ArrayBuf<T> = crate::array::Array<T, AccessBuf<ocl::Buffer<T>>, OpenCL>;
pub type ArrayOp<T, O> = crate::array::Array<T, AccessOp<O, OpenCL>, OpenCL>;

const _: () = {
    assert!(VEC_MIN_SIZE < GPU_MIN_SIZE);
    assert!(GPU_MIN_SIZE < ACC_MIN_SIZE);
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "complex")]
    use crate::NDArrayTrig;
    use crate::{
        shape, slice, AxisRange, Error, MatrixDual, NDArray, NDArrayCompare, NDArrayMath,
        NDArrayRead, NDArrayReduceBoolean, NDArrayTransform, NDArrayWrite, Shape,
    };

    use super::*;

    #[cfg(feature = "complex")]
    fn assert_complex32_close(
        actual: &[num_complex::Complex32],
        expected: &[num_complex::Complex32],
    ) {
        const EPS: f32 = 1e-4;
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual.re - expected.re).abs() <= EPS,
                "expected re={}, got re={}",
                expected.re,
                actual.re
            );
            assert!(
                (actual.im - expected.im).abs() <= EPS,
                "expected im={}, got im={}",
                expected.im,
                actual.im
            );
        }
    }

    #[test]
    fn test_add() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let left = ArrayBuf::constant(0, shape.clone())?;
        let right = ArrayBuf::constant(0, shape.clone())?;
        let expected = ArrayBuf::constant(0, shape.clone())?;

        let actual = left.add(right)?;
        let eq = actual.eq(expected)?;

        assert!(eq.all()?);

        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_trig_complex32_basic() -> Result<(), Error> {
        let input: Vec<num_complex::Complex32> = vec![
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(0.125, -0.25),
            num_complex::Complex32::new(-0.5, 1.5),
            num_complex::Complex32::new(2.0, 0.75),
        ];

        let shape = shape![input.len()];
        let buf = OpenCL::copy_into_buffer::<num_complex::Complex32>(&input)?;
        let arr = ArrayBuf::new(buf, shape)?;

        let sin_expected = input.iter().map(|z| z.sin()).collect::<Vec<_>>();
        let sin_actual = arr.clone().sin()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&sin_actual, &sin_expected);

        let cos_expected = input.iter().map(|z| z.cos()).collect::<Vec<_>>();
        let cos_actual = arr.clone().cos()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&cos_actual, &cos_expected);

        let tan_expected = input.iter().map(|z| z.tan()).collect::<Vec<_>>();
        let tan_actual = arr.tan()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&tan_actual, &tan_expected);

        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_trig_complex32_hyperbolic() -> Result<(), Error> {
        let input: Vec<num_complex::Complex32> = vec![
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(0.25, 0.125),
            num_complex::Complex32::new(-1.0, 0.75),
            num_complex::Complex32::new(1.25, -0.5),
        ];

        let shape = shape![input.len()];
        let buf = OpenCL::copy_into_buffer::<num_complex::Complex32>(&input)?;
        let arr = ArrayBuf::new(buf, shape)?;

        let sinh_expected = input.iter().map(|z| z.sinh()).collect::<Vec<_>>();
        let sinh_actual = arr.clone().sinh()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&sinh_actual, &sinh_expected);

        let cosh_expected = input.iter().map(|z| z.cosh()).collect::<Vec<_>>();
        let cosh_actual = arr.clone().cosh()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&cosh_actual, &cosh_expected);

        let tanh_expected = input.iter().map(|z| z.tanh()).collect::<Vec<_>>();
        let tanh_actual = arr.tanh()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&tanh_actual, &tanh_expected);

        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_trig_complex32_inverse() -> Result<(), Error> {
        let input: Vec<num_complex::Complex32> = vec![
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(0.25, 0.125),
            num_complex::Complex32::new(-0.5, 0.25),
            num_complex::Complex32::new(0.9, -0.1),
        ];

        let shape = shape![input.len()];
        let buf = OpenCL::copy_into_buffer::<num_complex::Complex32>(&input)?;
        let arr = ArrayBuf::new(buf, shape)?;

        let asin_expected = input.iter().map(|z| z.asin()).collect::<Vec<_>>();
        let asin_actual = arr.clone().asin()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&asin_actual, &asin_expected);

        let acos_expected = input.iter().map(|z| z.acos()).collect::<Vec<_>>();
        let acos_actual = arr.clone().acos()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&acos_actual, &acos_expected);

        let atan_expected = input.iter().map(|z| z.atan()).collect::<Vec<_>>();
        let atan_actual = arr.atan()?.buffer()?.to_slice()?.to_vec();
        assert_complex32_close(&atan_actual, &atan_expected);

        Ok(())
    }

    #[test]
    fn test_matmul_2x2() -> Result<(), Error> {
        let l = ArrayOp::range(0, 4, shape![2, 2])?;
        let r = ArrayOp::range(0, 4, shape![2, 2])?;

        let actual = l.matmul(r)?;
        assert_eq!(actual.shape(), &[2, 2]);

        let expected = vec![2, 3, 6, 11];
        assert_eq!(actual.buffer()?.to_slice()?.to_vec(), expected);

        Ok(())
    }

    #[test]
    fn test_matmul_12x20() -> Result<(), Error> {
        let buf = OpenCL::copy_into_buffer::<i32>(&(0..12).collect::<Vec<_>>())?;
        let l = ArrayBuf::new(buf, shape![3, 4])?;

        let buf = OpenCL::copy_into_buffer::<i32>(&(0..20).collect::<Vec<_>>())?;
        let r = ArrayBuf::new(buf, shape![4, 5])?;

        let actual = l.matmul(r)?;
        assert_eq!(actual.shape(), &[3, 5]);

        let expected = vec![
            70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462,
        ];

        assert_eq!(actual.buffer()?.to_slice()?.to_vec(), expected);

        Ok(())
    }

    #[test]
    fn test_matmul_large() -> Result<(), Error> {
        let shapes: Vec<(Shape, Shape, Shape)> = vec![
            (shape![2, 3], shape![3, 4], shape![2, 4]),
            (shape![9, 7], shape![7, 12], shape![9, 12]),
            (shape![16, 8], shape![8, 24], shape![16, 24]),
            (shape![2, 9], shape![9, 1], shape![2, 1]),
            (shape![16, 8], shape![8, 32], shape![16, 32]),
            (shape![2, 15, 26], shape![2, 26, 37], shape![2, 15, 37]),
            (shape![3, 15, 26], shape![3, 26, 37], shape![3, 15, 37]),
            (shape![8, 44, 1], shape![8, 1, 98], shape![8, 44, 98]),
        ];

        let queue = OpenCL::queue(GPU_MIN_SIZE, &[])?;

        for (left_shape, right_shape, output_shape) in shapes {
            let left = ocl::Buffer::builder()
                .queue(queue.clone())
                .len(left_shape.iter().product::<usize>())
                .fill_val(1.)
                .build()?;

            let right = ocl::Buffer::builder()
                .queue(queue.clone())
                .len(right_shape.iter().product::<usize>())
                .fill_val(1.)
                .build()?;

            let left = ArrayBuf::new(left, left_shape)?;
            let right = ArrayBuf::new(right, right_shape)?;

            let expected = *left.shape().last().unwrap();

            let actual = left.matmul(right)?;
            assert_eq!(actual.shape(), output_shape.as_slice());

            let actual = actual.buffer()?.to_slice()?;
            assert!(
                actual.iter().copied().all(|n| n == expected as f32),
                "expected {expected} but found {actual:?}"
            );

            queue.flush()?;
        }

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let buffer = OpenCL::copy_into_buffer::<i32>(&[0, 1, 2, 3, 4, 5])?;
        let array = ArrayBuf::new(buffer, shape.clone())?;

        let actual = array.as_ref().sub(array.as_ref())?;

        assert!(!actual.any()?);

        Ok(())
    }

    #[test]
    fn test_slice() -> Result<(), Error> {
        let buf = OpenCL::copy_into_buffer::<u32>(&[0; 6])?;
        let array = ArrayBuf::new(buf, shape![2, 3])?;
        let mut slice = array.slice(slice![AxisRange::In(0, 2, 1), AxisRange::At(1)])?;

        let buf = OpenCL::copy_into_buffer::<u32>(&[0, 0])?;
        let zeros = ArrayBuf::new(buf, shape![2])?;

        let buf = OpenCL::copy_into_buffer::<u32>(&[0, 0])?;
        let ones = ArrayBuf::new(buf, shape![2])?;

        assert!(slice.as_ref().eq(zeros)?.all()?);

        slice.write(&ones)?;

        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_div_complex() -> Result<(), Error> {
        use num_complex::Complex64 as C64;

        let buf = OpenCL::copy_into_buffer(&[C64::new(0.5, 0.5)])?;
        let lhs = ArrayBuf::new(buf, shape![1])?;

        let buf = OpenCL::copy_into_buffer(&[C64::new(1., -1.)])?;
        let rhs = ArrayBuf::new(buf, shape![1])?;

        let actual = lhs.div(rhs)?;
        let actual = actual.buffer()?.to_slice()?;
        assert_eq!(actual.into_vec(), vec![C64::new(0., 0.5)]);
        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_mul_complex() -> Result<(), Error> {
        type C32 = num_complex::Complex<f32>;

        let buf = OpenCL::copy_into_buffer(&[C32::new(0.5, 0.5)])?;
        let lhs = ArrayBuf::new(buf, shape![1])?;

        let buf = OpenCL::copy_into_buffer(&[C32::new(1., -1.)])?;
        let rhs = ArrayBuf::new(buf, shape![1])?;

        let actual = lhs.mul(rhs)?;
        let actual = actual.buffer()?.to_slice()?;
        assert_eq!(actual.into_vec(), vec![C32::new(1., 0.)]);
        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_pow_complex() -> Result<(), Error> {
        use crate::{NDArrayAbs, NDArrayCompareScalar};

        type C32 = num_complex::Complex<f32>;

        let buf = OpenCL::copy_into_buffer(&[C32::new(0., 1.)])?;
        let lhs = ArrayBuf::new(buf, shape![1])?;

        let buf = OpenCL::copy_into_buffer(&[C32::new(-1., 0.)])?;
        let rhs = ArrayBuf::new(buf, shape![1])?;

        let actual = lhs.pow(rhs)?;

        let buf = OpenCL::copy_into_buffer(&[C32::new(0., -1.)])?;
        let expected = ArrayBuf::new(buf, shape![1])?;

        assert!(expected
            .sub(actual)?
            .abs()?
            .lt_scalar(f32::EPSILON)?
            .all()?);

        Ok(())
    }
}
