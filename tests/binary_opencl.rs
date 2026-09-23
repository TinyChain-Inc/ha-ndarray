#![cfg(feature = "opencl")]

use ha_ndarray::{opencl::ArrayBuf, shape, NDArrayMath, NDArrayRead};

// Explicit OpenCL arrays never fall back to the CPU backend.
#[test]
fn binary_opencl_u8_wrapping_and_zero_divisors() {
    macro_rules! check {
        ($op:ident, $a:expr, $b:expr, $expected:expr) => {
            assert_eq!(
                ArrayBuf::constant($a, shape![1])
                    .unwrap()
                    .$op(ArrayBuf::constant($b, shape![1]).unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec(),
                vec![$expected]
            );
        };
    }
    check!(add, 255u8, 1u8, 0u8);
    check!(sub, 0u8, 1u8, 255u8);
    check!(mul, 255u8, 255u8, 1u8);
    check!(pow, 3u8, 6u8, 217u8);
    check!(pow, 255u8, 255u8, 255u8);
    check!(pow, 0u8, 0u8, 1u8);
    check!(rem, 5u8, 2u8, 1u8);
    check!(rem, 5u8, 0u8, 0u8);
    check!(div, 5u8, 0u8, 0u8);
}

#[test]
fn binary_opencl_float_edges() {
    macro_rules! check {
        ($t:ty) => {{
            for (a, b) in [(5.5 as $t, 2.0 as $t), (-5.5, 2.0), (-0.0, 2.0), (1.0, 0.0)] {
                let result = ArrayBuf::constant(a, shape![1])
                    .unwrap()
                    .rem(ArrayBuf::constant(b, shape![1]).unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec()[0];
                let expected = a % b;
                if expected.is_nan() {
                    assert!(result.is_nan());
                } else {
                    assert_eq!(result.to_bits(), expected.to_bits());
                }
            }
            for a in [0.0 as $t, 1.0, -1.0] {
                let result = ArrayBuf::constant(a, shape![1])
                    .unwrap()
                    .div(ArrayBuf::constant(0.0 as $t, shape![1]).unwrap())
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec()[0];
                if a == 0.0 {
                    assert!(result.is_nan());
                } else {
                    assert_eq!(result, a / 0.0);
                }
            }
        }};
    }
    check!(f32);
    check!(f64);
}
