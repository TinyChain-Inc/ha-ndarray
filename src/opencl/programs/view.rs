use super::Program;
use memoize::memoize;

use crate::ops::{FlipSpec, ViewSpec};
use crate::Error;

use super::{build, ArrayFormat};

#[memoize(Capacity: 1024)]
pub fn flip(c_type: &'static str, spec: FlipSpec) -> Result<Program, Error> {
    let axis = spec.axis;
    let ndim = spec.shape.len();
    let shape = ArrayFormat::from(spec.shape.as_slice());
    let strides = ArrayFormat::from(spec.strides.as_slice());

    let src = format!(
        r#"
        const uint ndim = {ndim};

        const ulong strides[{ndim}] = {strides};
        const ulong dims[{ndim}] = {shape};

        __kernel void flip(
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            ulong offset_out = get_global_id(0);
            ulong offset_in = 0;

            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                uint stride = strides[x];

                uint i;
                if (stride == 0) {{
                    i = 0;
                }} else {{
                    i = (offset_out / stride) % dims[x];
                }}

                if (x == {axis}) {{
                    i = dims[x] - i - 1;
                }}

                offset_in += i * strides[x];
            }}

            output[offset_out] = input[offset_in];
        }}
        "#,
    );

    build(&src, &[c_type], "view")
}

// TODO: support SharedCache
#[memoize(Capacity: 1024)]
pub fn view(c_type: &'static str, spec: ViewSpec) -> Result<Program, Error> {
    let ndim_in = spec.source_strides.len();
    let ndim_out = spec.shape.len();

    let strides_in = ArrayFormat::from(spec.source_strides.as_slice());
    let strides_out = ArrayFormat::from(spec.strides.as_slice());
    let dims_out = ArrayFormat::from(spec.shape.as_slice());

    let src = format!(
        r#"
        const uint ndim_in = {ndim_in};
        const uint ndim_out = {ndim_out};

        const ulong strides_in[{ndim_in}] = {strides_in};
        const ulong strides_out[{ndim_out}] = {strides_out};
        const ulong dims[{ndim_out}] = {dims_out};

        __kernel void view(
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            ulong offset_out = get_global_id(0);
            ulong offset_in = 0;

            #pragma unroll
            for (uint x_in = 0; x_in < {ndim_in}; x_in++) {{
                uint x_out = {ndim_offset} + x_in;
                uint stride_out = strides_out[x_out];

                uint i;
                if (stride_out == 0) {{
                    i = 0;
                }} else {{
                    i = (offset_out / stride_out) % dims[x_out];
                }}

                offset_in += i * strides_in[x_in];
            }}

            output[offset_out] = input[offset_in];
        }}
        "#,
        ndim_offset = (ndim_out - ndim_in)
    );

    build(&src, &[c_type], "view")
}
