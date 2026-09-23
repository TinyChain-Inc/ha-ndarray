use std::fmt;

use crate::Error;

use super::{CLElement, OpenCL, TILE_SIZE, WG_SIZE};

pub mod constructors;
pub mod elementwise;
pub mod gather;
pub mod linalg;
pub mod reduce;
pub mod slice;
pub mod view;

#[derive(Clone, Eq, PartialEq, Hash, fmt::Debug)]
pub(super) enum CLExpr {
    Static(&'static str),
    String(String),
}

impl From<&'static str> for CLExpr {
    fn from(s: &'static str) -> Self {
        CLExpr::Static(s)
    }
}

impl From<String> for CLExpr {
    fn from(s: String) -> Self {
        CLExpr::String(s)
    }
}

impl fmt::Display for CLExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(s) => f.write_str(s),
            Self::String(s) => f.write_str(s),
        }
    }
}

pub trait Builder {
    fn build(self) -> String;
}

#[derive(Clone, Eq, PartialEq, Hash, fmt::Debug)]
pub struct ElementDual {
    pub(super) i_type: &'static str,
    pub(super) o_type: &'static str,
    pub(super) name: &'static str,
    pub(super) op: CLExpr,
}

impl ElementDual {
    pub(crate) fn into_reduction(mut self) -> Self {
        self.o_type = self.i_type;
        self
    }

    pub(super) fn new<I, O, Op>(name: &'static str, op: Op) -> Self
    where
        I: CLElement,
        O: CLElement,
        Op: Into<CLExpr>,
    {
        Self {
            i_type: I::TYPE,
            o_type: O::TYPE,
            name,
            op: op.into(),
        }
    }
}

impl Builder for ElementDual {
    fn build(self) -> String {
        format!(
            r#"
            inline {o_type} {name}(const {i_type} lhs, const {i_type} rhs) {{
                {op}
            }}
            "#,
            i_type = self.i_type,
            o_type = self.o_type,
            name = self.name,
            op = self.op
        )
    }
}

#[derive(Clone, Eq, PartialEq, Hash, fmt::Debug)]
pub struct ElementUnary {
    pub(super) i_type: &'static str,
    pub(super) o_type: &'static str,
    pub(super) name: &'static str,
    pub(super) op: CLExpr,
}

impl ElementUnary {
    pub(super) fn new<I, O, Op>(name: &'static str, op: Op) -> Self
    where
        I: CLElement,
        O: CLElement,
        Op: Into<CLExpr>,
    {
        Self {
            i_type: I::TYPE,
            o_type: O::TYPE,
            name,
            op: op.into(),
        }
    }
}

impl Builder for ElementUnary {
    fn build(self) -> String {
        format!(
            "
            inline {o_type} {name}(const {i_type} n) {{
                {op}
            }}
            ",
            i_type = self.i_type,
            o_type = self.o_type,
            name = self.name,
            op = self.op,
        )
    }
}

struct ArrayFormat<'a, T> {
    arr: &'a [T],
}

impl<'a, T> From<&'a [T]> for ArrayFormat<'a, T> {
    fn from(arr: &'a [T]) -> Self {
        Self { arr }
    }
}

impl<'a, T: fmt::Display> fmt::Display for ArrayFormat<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{ ")?;

        for item in self.arr {
            write!(f, "{item}, ")?;
        }

        f.write_str(" }")
    }
}

#[derive(Clone)]
pub(crate) struct Program {
    source: String,
    types: Vec<&'static str>,
    operation: &'static str,
}

impl Program {
    pub(crate) fn for_queue(&self, queue: &ocl::Queue) -> Result<ocl::Program, Error> {
        compile(
            self.source.clone(),
            self.types.clone(),
            self.operation,
            queue.device(),
        )
    }
}

fn build(src: &str, types: &[&'static str], operation: &'static str) -> Result<Program, Error> {
    Ok(Program {
        source: src.to_owned(),
        types: types.to_vec(),
        operation,
    })
}

#[memoize::memoize]
fn compile(
    source: String,
    types: Vec<&'static str>,
    operation: &'static str,
    device: ocl::Device,
) -> Result<ocl::Program, Error> {
    use ocl::core::{DeviceInfo, DeviceInfoResult};

    let device_name = device
        .name()
        .unwrap_or_else(|err| format!("{device:?} ({err})"));
    let fp32 = validate_capabilities(operation, &types, &device_name, |double| {
        let info = if double {
            DeviceInfo::DoubleFpConfig
        } else {
            DeviceInfo::SingleFpConfig
        };

        match device.info(info) {
            Ok(
                DeviceInfoResult::SingleFpConfig(flags) | DeviceInfoResult::DoubleFpConfig(flags),
            ) => Ok(flags),
            Ok(other) => Err(format!("unexpected capability response {other:?}")),
            Err(err) => Err(err.to_string()),
        }
    })?;
    let source = format!("#pragma OPENCL FP_CONTRACT OFF\n{source}");
    let mut builder = ocl::Program::builder();
    builder.source(source).devices(device);

    if fp32 {
        builder.cmplr_opt("-cl-fp32-correctly-rounded-divide-sqrt");
    }

    builder.build(OpenCL::context()).map_err(Error::from)
}

/// Validate requirements independently of querying a particular device. The caller
/// supplies the selected queue device's query results before compiling its program.
fn validate_capabilities(
    operation: &str,
    types: &[&str],
    device: &str,
    mut query: impl FnMut(bool) -> Result<ocl::core::DeviceFpConfig, String>,
) -> Result<bool, Error> {
    use ocl::core::DeviceFpConfig as F;

    let mut fp32 = false;

    for &dtype in types {
        let base = dtype.trim_end_matches('2');

        if base != "float" && base != "double" {
            continue;
        }

        fp32 |= base == "float";
        let flags = query(base == "double").map_err(|err| {
            Error::Unsupported(format!(
                "OpenCL {operation} for {dtype} on {device}: capability query failed: {err}"
            ))
        })?;
        let mut required = F::DENORM | F::INF_NAN | F::ROUND_TO_NEAREST;

        if base == "float" {
            required |= F::CORRECTLY_ROUNDED_DIVIDE_SQRT;
        }

        let missing = required & !flags;

        if !missing.is_empty() {
            return Err(Error::Unsupported(format!(
                "OpenCL {operation} for {dtype} on {device}: missing {missing:?} ({base} support); device reports {flags:?}"
            )));
        }
    }

    Ok(fp32)
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    use ocl::core::DeviceFpConfig as F;

    fn all() -> F {
        F::DENORM | F::INF_NAN | F::ROUND_TO_NEAREST | F::CORRECTLY_ROUNDED_DIVIDE_SQRT
    }

    fn unsupported(types: &[&str], query: impl FnMut(bool) -> Result<F, String>, missing: &str) {
        let err = validate_capabilities("fixture_op", types, "fixture_device", query).unwrap_err();

        assert!(matches!(err, Error::Unsupported(_)));
        let message = err.to_string();

        for part in ["fixture_op", "fixture_device", missing] {
            assert!(message.contains(part), "{message} lacks {part}");
        }

        assert!(types.iter().any(|dtype| message.contains(dtype)));
    }

    #[test]
    fn required_flags_and_queries_fail_closed() {
        for (flag, name) in [
            (F::DENORM, "DENORM"),
            (F::INF_NAN, "INF_NAN"),
            (F::ROUND_TO_NEAREST, "ROUND_TO_NEAREST"),
            (
                F::CORRECTLY_ROUNDED_DIVIDE_SQRT,
                "CORRECTLY_ROUNDED_DIVIDE_SQRT",
            ),
        ] {
            unsupported(&["float"], |_| Ok(all() & !flag), name);
        }

        unsupported(&["double"], |_| Ok(F::empty()), "double support");
        unsupported(
            &["float"],
            |_| Err("query unavailable".into()),
            "query unavailable",
        );
        unsupported(&["double2"], |_| Err("query unavailable".into()), "double2");

        assert!(
            validate_capabilities("op", &["float2", "double2"], "device", |_| Ok(all())).unwrap()
        );

        assert!(
            !validate_capabilities("op", &["int", "ulong"], "device", |_| panic!(
                "integer-only kernel queried floats"
            ))
            .unwrap()
        );
    }

    #[test]
    fn generated_casts_retain_all_precision_requirements() {
        for (input, output, intermediate) in [
            ("long", "float", "double"),
            ("ulong", "float2", "double"),
            ("int", "double", "float"),
            ("uint", "double2", "float"),
            ("double2", "int", "double2"),
            ("float2", "double2", "float2"),
        ] {
            let program = elementwise::cast(ElementUnary {
                i_type: input,
                o_type: output,
                name: "cast_fixture",
                op: "return n;".into(),
            })
            .unwrap();

            assert_eq!(program.types, vec![input, output, intermediate]);
            unsupported(&program.types, |_| Ok(F::empty()), "missing");
        }
    }

    #[test]
    fn input_output_and_intermediate_requirements() {
        for types in [
            &["double", "float"][..],
            &["float", "double"][..],
            &["long", "float", "double"][..],
            &["int", "double", "float"][..],
        ] {
            unsupported(
                types,
                |double| Ok(if double { F::empty() } else { all() }),
                "double",
            );
            unsupported(
                types,
                |double| Ok(if double { all() } else { F::empty() }),
                "float",
            );
        }

        for dtype in ["float2", "double2"] {
            unsupported(&[dtype], |_| Ok(F::empty()), dtype);
        }
    }
}
