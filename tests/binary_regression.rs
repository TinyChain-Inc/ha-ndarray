use ha_ndarray::{shape, AccessBuf, Array, NDArrayMath, NDArrayRead, Number};

fn input<T: Number>(values: Vec<T>) -> Array<T, AccessBuf<Vec<T>>> {
    let len = values.len();
    Array::new(values, shape![len]).unwrap()
}

#[test]
fn u8_binary_wrapping_and_zero_divisors() {
    macro_rules! check {
        ($op:ident, $a:expr, $b:expr, $expected:expr) => {
            assert_eq!(
                input($a)
                    .$op(input($b))
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec(),
                $expected
            );
        };
    }
    check!(add, vec![255u8, 127], vec![1, 255], vec![0, 126]);
    check!(sub, vec![0u8, 127], vec![1, 255], vec![255, 128]);
    check!(mul, vec![255u8, 128], vec![255, 2], vec![1, 0]);
    check!(
        pow,
        vec![2u8, 255, 0, 0, 3],
        vec![8, 255, 0, 1, 6],
        vec![0, 255, 1, 0, 217]
    );
    check!(div, vec![255u8, 5, 0], vec![0, 2, 0], vec![0, 2, 0]);
    check!(rem, vec![255u8, 5, 0], vec![0, 2, 0], vec![0, 1, 0]);
}

#[test]
fn floating_remainder_is_not_power() {
    macro_rules! check {
        ($t:ty) => {{
            let a: Vec<$t> = vec![5.5, -5.5, 5.5, -0.0, 1.0, <$t>::INFINITY, <$t>::NAN];
            let b: Vec<$t> = vec![2.0, 2.0, -2.0, 2.0, 0.0, 2.0, 2.0];
            let values = input(a.clone())
                .rem(input(b.clone()))
                .unwrap()
                .buffer()
                .unwrap()
                .to_slice()
                .unwrap()
                .into_vec();
            for ((actual, a), b) in values.into_iter().zip(a).zip(b) {
                let expected = a % b;
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }};
    }
    check!(f32);
    check!(f64);
}
