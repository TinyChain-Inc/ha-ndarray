use super::Program;
use memoize::memoize;

use crate::Error;

use super::{build, Builder, ElementDual, ElementUnary};

const LIB: &str = r#"
const float pi = 3.14159265358979323846f;

// PCG hash by Melissa E. O'Neill: https://www.pcg-random.org/
uint pcg_hash(uint seed) {
    uint state = seed * 747796405 + 2891336453;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

// Xorshift algorithm by George Marsalia: https://www.jstatsoft.org/article/view/v008i14
uint xorshift(uint rng_state) {
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

float random(const ulong seed, const ulong offset) {
    // rotate the offset seed number of places through the range 0..2^32
    uint rng_state = (offset + seed) & UINT_MAX;

    rng_state = xorshift(rng_state);

    for (uint i = 32; i < log2((float) offset); ++i) {{
        rng_state = xorshift(rng_state);
    }}

    rng_state = pcg_hash(rng_state);

    return (rng_state >> 8) * 0x1.0p-24f;
}
"#;

#[memoize]
pub fn random_normal() -> Result<Program, Error> {
    let src = format!(
        r#"
        {LIB}

        __kernel void random_normal(
                const ulong seed,
                __global float* buffer,
                __local float* normal)
        {{
            const ulong global_offset = get_global_id(0);
            const uint local_offset = get_local_id(0);

            normal[local_offset] = random(seed, global_offset);
            barrier(CLK_LOCAL_MEM_FENCE);

            // Box-Muller algorithm
            if (local_offset % 2 == 0) {{
                float u1 = 1.0f - normal[local_offset];
                float u2 = normal[local_offset + 1];
                float r = sqrt(-2 * log(u1));
                float theta = 2 * pi * u2;
                buffer[global_offset] = r * cos(theta);
            }} else {{
                float u1 = 1.0f - normal[local_offset - 1];
                float u2 = normal[local_offset];
                float r = sqrt(-2 * log(u1));
                float theta = 2 * pi * u2;
                buffer[global_offset] = r * sin(theta);
            }}
        }}
        "#
    );

    build(&src, &["float"], "constructors")
}

#[memoize]
pub fn random_uniform() -> Result<Program, Error> {
    let src = format!(
        r#"
        {LIB}

        __kernel void random_uniform(const ulong seed, __global float* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = random(seed, offset);
        }}
        "#
    );

    build(&src, &["float"], "constructors")
}

#[memoize]
pub fn range(add: ElementDual, mul: ElementDual, cast: ElementUnary) -> Result<Program, Error> {
    let c_type = add.i_type;
    let add = add.build();
    let mul = mul.build();
    let cast = cast.build();
    let src = format!(
        r#"
        {add}
        {mul}
        {cast}

        __kernel void range(const {c_type} start, const {c_type} step, __global {c_type}* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = add(start, mul(_cast(offset), step));
        }}
        "#,
    );

    // Range offsets are u64 and use number-general's f64 intermediate.
    let intermediate = if matches!(c_type, "float" | "float2") {
        "double"
    } else {
        c_type
    };

    build(&src, &[c_type, intermediate], "range")
}
