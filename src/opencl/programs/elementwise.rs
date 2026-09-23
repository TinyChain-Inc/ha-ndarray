use super::Program;
use memoize::memoize;

use crate::Error;

use super::{build, Builder, ElementDual, ElementUnary};

#[memoize]
pub fn cast(op: ElementUnary) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void cast(
            __global const {i_type}* restrict input,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(input[offset]);
        }}
        "#,
    );

    // Integer-to-float casts pass through number-general's source-width float,
    // which can differ from the destination precision in either direction.
    let float = |t| matches!(t, "float" | "float2" | "double" | "double2");
    let intermediate = if !float(i_type) && float(o_type) {
        if matches!(i_type, "long" | "ulong") {
            "double"
        } else {
            "float"
        }
    } else {
        i_type
    };

    build(&src, &[i_type, o_type, intermediate], name)
}

#[memoize]
pub fn dual(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual(
            __global const {i_type}* restrict left,
            __global const {i_type}* restrict right,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right[offset]);
        }}
        "#,
    );

    build(&src, &[i_type, o_type], name)
}

#[memoize]
pub fn dual_scalar(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual_scalar(
            __global const {i_type}* restrict left,
            const {i_type} right,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right);
        }}
        "#,
    );

    build(&src, &[i_type, o_type], name)
}

pub fn unary(op: ElementUnary) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void unary(__global const {i_type}* input, __global {o_type}* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(input[offset]);
        }}
        "#,
    );

    build(&src, &[i_type, o_type], name)
}
