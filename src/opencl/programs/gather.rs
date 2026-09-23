use super::Program;
use memoize::memoize;

use crate::Error;

use super::build;

#[memoize]
pub fn gather_cond(c_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void gather_cond(
            __global const uchar* restrict cond,
            __global const {c_type}* restrict then,
            __global const {c_type}* restrict or_else,
            __global {c_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);

            if (cond[offset] != 0) {{
                output[offset] = then[offset];
            }} else {{
                output[offset] = or_else[offset];
            }}
        }}
        "#,
    );

    build(&src, &[c_type], "gather")
}
