pub mod oracle;
#[macro_use]
mod aggregate;

pub fn close(actual: f64, expected: f64, bits: u32) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{actual} must be NaN");
        return;
    }
    if expected.is_infinite() || expected == 0.0 {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual} != {expected}"
        );
        return;
    }
    assert!(actual.is_finite(), "{actual} != {expected}");
    let (distance, limit) = if bits == 32 {
        (
            (actual as f32)
                .to_bits()
                .abs_diff((expected as f32).to_bits()) as u64,
            8,
        )
    } else {
        (actual.to_bits().abs_diff(expected.to_bits()), 8)
    };
    assert!(distance <= limit, "{actual} != {expected}: {distance} ULP");
}

macro_rules! conformance_suite {
    () => {
        use crate::conformance::close;
        use ha_ndarray::{
            NDArrayAbs, NDArrayBoolean, NDArrayCast, NDArrayCompare, NDArrayMath, NDArrayMathScalar,
            NDArrayNumeric, NDArrayRead, NDArrayReduceAll, NDArrayReduceBoolean, NDArrayTransform,
            NDArrayTrig, NDArrayUnary, NDArrayUnaryBoolean,
        };
        use safecast::CastFrom;

        #[test]
        fn exhaustive_byte_arithmetic() {
            macro_rules! test_type {
                ($t:ty) => {{
                    let a: Vec<$t> = (0..=255u16)
                        .flat_map(|a| std::iter::repeat_n(a as $t, 256))
                        .collect();
                    let b: Vec<$t> = (0..=255u16).cycle().take(65536).map(|b| b as $t).collect();
                    macro_rules! check {
                        ($op:ident, $expected:expr) => {{
                            let actual = input(a.clone())
                                .$op(input(b.clone()))
                                .unwrap()
                                .buffer()
                                .unwrap()
                                .to_slice()
                                .unwrap()
                                .into_vec();
                            for ((&a, &b), &actual) in a.iter().zip(&b).zip(&actual) {
                                let expected: $t = ($expected)(a, b);
                                assert_eq!(actual, expected, "{}({a}, {b})", stringify!($op));
                            }
                        }};
                    }
                    check!(add, |a: $t, b: $t| (a as i128 + b as i128) as $t);
                    check!(sub, |a: $t, b: $t| (a as i128 - b as i128) as $t);
                    check!(mul, |a: $t, b: $t| (a as i128 * b as i128) as $t);
                    check!(div, |a: $t, b: $t| if b == 0 {
                        0
                    } else {
                        (a as i128 / b as i128) as $t
                    });
                    check!(rem, |a: $t, b: $t| if b == 0 {
                        0
                    } else {
                        (a as i128 % b as i128) as $t
                    });
                    check!(pow, |a: $t, b: $t| {
                        if (b as i128) < 0 {
                            if a == 1 {
                                1
                            } else if a as i128 == -1 {
                                if b & 1 == 0 {
                                    1
                                } else {
                                    a
                                }
                            } else {
                                0
                            }
                        } else {
                            let mut r = 1i128;
                            for _ in 0..b as u32 {
                                r = (r * a as i128).rem_euclid(256);
                            }
                            r as $t
                        }
                    });
                    macro_rules! predicate {
                        ($op:ident, $expected:expr) => {{
                            let result = input(a.clone())
                                .$op(input(b.clone()))
                                .unwrap()
                                .buffer()
                                .unwrap()
                                .to_slice()
                                .unwrap()
                                .into_vec();
                            for ((&a, &b), v) in a.iter().zip(&b).zip(result) {
                                assert_eq!(v, u8::from(($expected)(a, b)));
                            }
                        }};
                    }
                    predicate!(eq, |a, b| a == b);
                    predicate!(ne, |a, b| a != b);
                    predicate!(gt, |a, b| a > b);
                    predicate!(ge, |a, b| a >= b);
                    predicate!(lt, |a, b| a < b);
                    predicate!(le, |a, b| a <= b);
                    predicate!(or, |a, b| a != 0 || b != 0);
                    predicate!(xor, |a, b| (a != 0) ^ (b != 0));
                    let all_equal = input(a.clone())
                        .eq(input(a.clone()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert!(all_equal.iter().all(|v| *v == 1));
                    let boolean = input(a.clone())
                        .and(input(b.clone()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    for ((a, b), result) in a.iter().zip(&b).zip(boolean) {
                        assert_eq!(result, u8::from(*a != 0 && *b != 0));
                    }
                }};
            }
            test_type!(u8);
            test_type!(i8);
        }

        #[test]
        fn wide_integer_boundaries() {
            macro_rules! check {
                ($t:ty) => {{
                    let a = vec![<$t>::MIN, <$t>::MAX, 0, 1, 2, 3];
                    let b = vec![(1 as $t).wrapping_neg(), 2, 0, <$t>::MAX, 0, 5];
                    let values = input(a.clone())
                        .div(input(b.clone()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    for ((a, b), v) in a.into_iter().zip(b).zip(values) {
                        assert_eq!(v, if b == 0 { 0 } else { a.wrapping_div(b) });
                    }
                    let high = <$t>::MAX;
                    let p = input(vec![1 as $t, 0, (1 as $t).wrapping_neg()])
                        .pow(input(vec![high; 3]))
                        .unwrap();
                    let values = p.buffer().unwrap().to_slice().unwrap().into_vec();
                    assert_eq!(values, vec![1, 0, (1 as $t).wrapping_neg()]);
                    let abs = input(vec![<$t>::MIN])
                        .abs()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert_eq!(abs[0], <$t>::MIN);
                }};
            }
            check!(i16);
            check!(i32);
            check!(i64);
            check!(u16);
            check!(u32);
            check!(u64);
            let p = input(vec![-1i64, -1, 2, 0])
                .pow(input(vec![i64::MIN, -3, -2, -1]))
                .unwrap();
            assert_eq!(
                p.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![1, -1, 0, 0]
            );
        }

        #[test]
        fn real_unary_oracle() {
            macro_rules! dtype {
                ($t:ty, $bits:expr) => {{
                    let values: Vec<$t> = vec![
                        -744.,
                        -100.,
                        -1.,
                        -0.5,
                        -0.,
                        0.,
                        0.2,
                        0.5,
                        1.,
                        2.,
                        10.,
                        1000.,
                        <$t>::from_bits(1),
                        <$t>::MIN_POSITIVE,
                        <$t>::MAX,
                        <$t>::NEG_INFINITY,
                        <$t>::INFINITY,
                        <$t>::NAN,
                    ];
                    macro_rules! check {
                        ($op:ident, $reference:ident) => {{
                            let expression = input(values.clone()).$op().unwrap();
                            let result = expression.buffer().unwrap().to_slice().unwrap().into_vec();
                            for (i, (&x, &actual)) in values.iter().zip(&result).enumerate() {
                                let expected = crate::conformance::oracle::real(x as f64, $bits, stringify!($reference)) as $t;
                                close(actual as f64, expected as f64, $bits);
                                close(
                                    expression.read_value(&[i]).unwrap() as f64,
                                    expected as f64,
                                    $bits,
                                );
                            }
                        }};
                    }
                    check!(exp, exp);
                    check!(ln, ln);
                    check!(sin, sin);
                    check!(cos, cos);
                    check!(tan, tan);
                    check!(asin, asin);
                    check!(acos, acos);
                    check!(atan, atan);
                    check!(sinh, sinh);
                    check!(cosh, cosh);
                    check!(tanh, tanh);
                    check!(round, round);
                    check!(abs, abs);
                }};
            }
            dtype!(f32, 32);
            dtype!(f64, 64);
        }

        #[test]
        fn exceptional_values_and_subnormals() {
            macro_rules! dtype {
                ($t:ty, $bits:expr) => {{
                    let a: Vec<$t> = vec![0., -0., 1., -1., <$t>::MAX, <$t>::MIN_POSITIVE, <$t>::from_bits(1), <$t>::INFINITY, <$t>::NAN];
                    let b: Vec<$t> = vec![0., 2., 0., 0., 0.5, 2., 1., 2., 2.];
                    macro_rules! check {
                        ($op:ident, $operator:tt) => {{
                            let expr = input(a.clone()).$op(input(b.clone())).unwrap();
                            let result = expr.buffer().unwrap().to_slice().unwrap().into_vec();
                            for ((a,b),actual) in a.iter().zip(&b).zip(result) {
                                let reference = crate::conformance::oracle::real_binary(*a as f64,*b as f64,$bits,stringify!($op));
                                let expected = reference as $t;
                                if expected.is_nan() { assert!(actual.is_nan()); }
                                else { assert_eq!(actual.to_bits(),expected.to_bits(), "{}({a},{b})", stringify!($op)); }
                            }
                        }};
                    }
                    check!(add, +); check!(sub, -); check!(mul, *); check!(div, /); check!(rem, %);
                    for size in [1, 63, 64, 129, 8193] {
                        assert_eq!(input(vec![<$t>::NEG_INFINITY;size]).max_all().unwrap(), <$t>::NEG_INFINITY);
                        assert_eq!(input(vec![<$t>::INFINITY;size]).min_all().unwrap(), <$t>::INFINITY);
                        let mut v=vec![1 as $t;size]; v[size-1]=<$t>::NAN;
                        assert!(input(v.clone()).min_all().unwrap().is_nan());
                        assert!(input(v).max_all().unwrap().is_nan());
                    }
                    assert_eq!(input(vec![-0. as $t,0.]).min_all().unwrap().to_bits(),(-0. as $t).to_bits());
                    assert_eq!(input(vec![-0. as $t,0.]).max_all().unwrap().to_bits(),(0. as $t).to_bits());
                    assert_eq!(input(a.clone()).not().unwrap().buffer().unwrap().to_slice().unwrap().into_vec(), vec![1,1,0,0,0,0,0,0,0]);
                    assert_eq!(input(a.clone()).is_nan().unwrap().buffer().unwrap().to_slice().unwrap().into_vec(), vec![0,0,0,0,0,0,0,0,1]);
                    assert_eq!(input(a).is_inf().unwrap().buffer().unwrap().to_slice().unwrap().into_vec(), vec![0,0,0,0,0,0,0,1,0]);
                }};
            }
            dtype!(f32, 32);
            dtype!(f64, 64);
        }

        #[test]
        fn cast_pipeline_all_real_pairs() {
            macro_rules! from {
                ($t:ty, $values:expr) => {{
                    let values: Vec<$t> = $values;
                    macro_rules! to {
                        ($o:ty) => {{
                            let expr = NDArrayCast::<$o>::cast(input(values.clone())).unwrap();
                            let result = expr.buffer().unwrap().to_slice().unwrap().into_vec();
                            for ((i, v), actual) in values.iter().enumerate().zip(result) {
                                let expected = <$o>::cast_from(number_general::Number::from(*v));
                                assert!(
                                    crate::conformance::same_number(actual, expected),
                                    "{} -> {}: {v} produced {actual}, expected {expected}",
                                    stringify!($t),
                                    stringify!($o)
                                );
                                assert!(crate::conformance::same_number(
                                    expr.read_value(&[i]).unwrap(),
                                    expected
                                ));
                            }
                        }};
                    }
                    to!(i8);
                    to!(i16);
                    to!(i32);
                    to!(i64);
                    to!(u8);
                    to!(u16);
                    to!(u32);
                    to!(u64);
                    to!(f32);
                    to!(f64);
                    #[cfg(feature = "complex")]
                    {
                        to!(ha_ndarray::complex::Complex32);
                        to!(ha_ndarray::complex::Complex64);
                    }
                }};
            }
            from!(i8, vec![i8::MIN, -1, 0, 1, i8::MAX]);
            from!(i16, vec![i16::MIN, -257, -1, 0, 1, 256, i16::MAX]);
            from!(i32, vec![i32::MIN, -1, 0, 1, 16_777_217, i32::MAX]);
            from!(
                i64,
                vec![i64::MIN, -1, 0, 1, 9_007_199_254_740_993, i64::MAX]
            );
            from!(u8, vec![0, 1, 127, 128, 255]);
            from!(u16, vec![0, 1, 32768, u16::MAX]);
            from!(u32, vec![0, 1, 16_777_217, u32::MAX]);
            from!(u64, vec![0, 1, 9_007_199_254_740_993, u64::MAX]);
            from!(
                f32,
                vec![
                    -f32::INFINITY,
                    f32::MIN,
                    -257.5,
                    -1.5,
                    -0.,
                    0.,
                    1.5,
                    255.9,
                    256.,
                    f32::MAX,
                    f32::INFINITY,
                    f32::NAN
                ]
            );
            from!(
                f64,
                vec![
                    -f64::INFINITY,
                    f64::MIN,
                    -257.5,
                    -1.5,
                    -0.,
                    0.,
                    1.5,
                    255.9,
                    256.,
                    f64::MAX,
                    f64::INFINITY,
                    f64::NAN
                ]
            );
            #[cfg(feature = "complex")]
            {
                from!(
                    ha_ndarray::complex::Complex32,
                    vec![
                        ha_ndarray::complex::Complex32::new(-257.5, 3.),
                        ha_ndarray::complex::Complex32::new(f32::NAN, f32::INFINITY),
                        ha_ndarray::complex::Complex32::new(0., -0.),
                    ]
                );
                from!(
                    ha_ndarray::complex::Complex64,
                    vec![
                        ha_ndarray::complex::Complex64::new(-257.5, 3.),
                        ha_ndarray::complex::Complex64::new(f64::NAN, f64::INFINITY),
                        ha_ndarray::complex::Complex64::new(0., -0.),
                    ]
                );
            }
        }

        #[test]
        fn cast_compatibility_fixtures() {
            let a = NDArrayCast::<u64>::cast(input(vec![-1i8, -128, 127])).unwrap();
            assert_eq!(
                a.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![255, 128, 127]
            );
            let a = NDArrayCast::<i64>::cast(input(vec![u32::MAX, 0])).unwrap();
            assert_eq!(
                a.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![-1, 0]
            );
            let a = NDArrayCast::<i8>::cast(input(vec![
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                256.,
            ]))
            .unwrap();
            assert_eq!(
                a.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![-1, 0, 0, 0]
            );
            let a = NDArrayCast::<f64>::cast(input(vec![16_777_217i32])).unwrap();
            assert_eq!(
                a.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![16_777_216.]
            );
        }

        #[test]
        fn real_power_and_logarithm() {
            macro_rules! dtype {
                ($t:ty,$bits:expr) => {{
                    let a: Vec<$t> = vec![0.25, 0.5, 1., 2., 10., 1.0001];
                    let b: Vec<$t> = vec![-2., 0.25, 3., 8., 2., 1.0002];
                    let actual = input(a.clone())
                        .pow(input(b.clone()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    let logs = input(a.clone())
                        .log(input(b.iter().map(|v| v.abs() + 0.5).collect()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    for (i, ((&a, &b), v)) in a.iter().zip(&b).zip(actual).enumerate() {
                        let pow =
                            crate::conformance::oracle::real_binary(a as f64, b as f64, $bits, "pow") as $t;
                        close(v as f64, pow as f64, $bits);
                        let base = (b.abs() + 0.5) as f64;
                        let log = crate::conformance::oracle::real_binary(a as f64, base, $bits, "log")
                            as $t;
                        close(logs[i] as f64, log as f64, $bits);
                    }
                    let a = input(vec![0. as $t, <$t>::NAN, 1., -1.])
                        .pow(input(vec![0. as $t, 0., <$t>::NAN, 0.5]))
                        .unwrap();
                    let v = a.buffer().unwrap().to_slice().unwrap().into_vec();
                    assert_eq!(&v[..3], &[1., 1., 1.]);
                    assert!(v[3].is_nan());
                }};
            }
            dtype!(f32, 32);
            dtype!(f64, 64);
        }
        #[test]
        fn scalar_array_and_point_parity() {
            macro_rules! dtype {
                ($t:ty) => {{
                    let values: Vec<$t> = vec![0 as $t, 1 as $t, 2 as $t, 3 as $t, <$t>::MAX];
                    macro_rules! check {
                        ($op:ident,$scalar:ident) => {{
                            for rhs in [0 as $t, 2 as $t, 3 as $t] {
                                let expression = input(values.clone()).$scalar(rhs).unwrap();
                                let scalar = expression.buffer().unwrap().to_slice().unwrap().into_vec();
                                let array = input(values.clone())
                                    .$op(input(vec![rhs; values.len()]))
                                    .unwrap()
                                    .buffer()
                                    .unwrap()
                                    .to_slice()
                                    .unwrap()
                                    .into_vec();
                                for (i, (s, a)) in scalar.into_iter().zip(array).enumerate() {
                                    assert!(
                                        crate::conformance::same_number(s, a),
                                        "{} {s} != {a}",
                                        stringify!($scalar)
                                    );
                                    assert!(crate::conformance::same_number(
                                        s,
                                        expression.read_value(&[i]).unwrap()
                                    ));
                                }
                            }
                        }};
                    }
                    check!(add, add_scalar);
                    check!(sub, sub_scalar);
                    check!(mul, mul_scalar);
                    check!(div, div_scalar);
                    check!(rem, rem_scalar);
                    check!(pow, pow_scalar);
                }};
            }
            dtype!(i8);
            dtype!(u8);
            dtype!(f32);
            dtype!(f64);
        }

        #[test]
        fn reductions_and_composition() {
            let expr = input(vec![0.2f64; 128])
                .round()
                .unwrap()
                .exp()
                .unwrap()
                .add_scalar(2.)
                .unwrap()
                .flip(0)
                .unwrap();
            assert!(expr
                .buffer()
                .unwrap()
                .to_slice()
                .unwrap()
                .iter()
                .all(|x| *x == 3.));
            assert_eq!(
                reduce_max(vec![f64::NEG_INFINITY; 128], 64),
                vec![f64::NEG_INFINITY; 2]
            );
            assert!(input(vec![1u8; 129]).all().unwrap());
            assert!(!input(vec![0u8; 129]).any().unwrap());
            assert_eq!(input(vec![127i8; 129]).sum_all().unwrap(), -1);
            assert_eq!(input(vec![3u8; 129]).product_all().unwrap(), 3);
            #[cfg(feature = "complex")]
            {
                use ha_ndarray::complex::Complex64;
                let values = vec![Complex64::new(0.25, 0.5); 129];
                let sum = input(values).sum_all().unwrap();
                assert_eq!(sum, Complex64::new(32.25, 64.5));
            }
        }
        #[cfg(feature = "complex")]
        #[test]
        fn complex_unary_oracle_and_predicates() {
            macro_rules! dtype {
                ($t:ty) => {{
                    use ha_ndarray::complex::Complex;
                    let values = vec![
                        Complex::<$t>::new(0.2, 0.3),
                        Complex::new(-0.5, 0.25),
                        Complex::new(1., 0.5),
                        Complex::new(-1., -0.),
                        Complex::new(-2., 0.001),
                        Complex::new(-2., -0.001),
                        Complex::new(2., 0.001),
                        Complex::new(2., -0.001),
                    ];
                    macro_rules! check {
                        ($op:ident) => {{
                            let expr = input(values.clone()).$op().unwrap();
                            let actual = expr.buffer().unwrap().to_slice().unwrap().into_vec();
                            for (&z, a) in values.iter().zip(actual) {
                                let expected = crate::conformance::oracle::complex(
                                    (z.re as f64, z.im as f64),
                                    None,
                                    if <$t>::MANTISSA_DIGITS == 24 { 32 } else { 64 },
                                    stringify!($op),
                                );
                                for (a, e) in [(a.re as f64, expected.0), (a.im as f64, expected.1)] {
                                    assert!(
                                        (a - e).abs() <= 16. * <$t>::EPSILON as f64 * e.abs().max(1.),
                                        "{}({z:?}): {a} != {e}",
                                        stringify!($op)
                                    );
                                }
                            }
                        }};
                    }
                    check!(exp);
                    check!(ln);
                    check!(sin);
                    check!(cos);
                    check!(tan);
                    check!(sinh);
                    check!(cosh);
                    check!(tanh);
                    check!(asin);
                    check!(acos);
                    check!(atan);
                    // Pin num-complex's exact-cut and exceptional conventions separately
                    // from the independent MPC finite-accuracy cases above.
                    let edges = vec![
                        Complex::<$t>::new(0., 0.),
                        Complex::new(-0., -0.),
                        Complex::new(2., 0.),
                        Complex::new(2., -0.),
                        Complex::new(-2., 0.),
                        Complex::new(-2., -0.),
                        Complex::new(0., 1.),
                        Complex::new(0., -1.),
                        Complex::new(<$t>::INFINITY, 0.),
                        Complex::new(<$t>::NEG_INFINITY, -0.),
                        Complex::new(<$t>::NAN, 0.),
                        Complex::new(0., <$t>::INFINITY),
                    ];
                    macro_rules! edge {
                        ($op:ident) => {{
                            let output = input(edges.clone())
                                .$op()
                                .unwrap()
                                .buffer()
                                .unwrap()
                                .to_slice()
                                .unwrap()
                                .into_vec();
                            for (&z, v) in edges.iter().zip(output) {
                                let expected = z.$op();
                                for (a, b) in [(v.re, expected.re), (v.im, expected.im)] {
                                    if b.is_nan() {
                                        assert!(a.is_nan(), "{}({z:?}): {a} != {b}", stringify!($op));
                                    } else if b == 0. || b.is_infinite() {
                                        assert_eq!(
                                            a.to_bits(),
                                            b.to_bits(),
                                            "{}({z:?}): {a} != {b}",
                                            stringify!($op)
                                        );
                                    } else {
                                        assert!(
                                            (a - b).abs() <= 16. * <$t>::EPSILON * b.abs().max(1.),
                                            "{}({z:?}): {a} != {b}",
                                            stringify!($op)
                                        );
                                    }
                                }
                            }
                        }};
                    }
                    edge!(exp);
                    edge!(ln);
                    edge!(sin);
                    edge!(cos);
                    edge!(tan);
                    edge!(sinh);
                    edge!(cosh);
                    edge!(tanh);
                    edge!(asin);
                    edge!(acos);
                    edge!(atan);
                    let special = vec![
                        Complex::<$t>::new(0., -0.),
                        Complex::new(1., 0.),
                        Complex::new(<$t>::NAN, 0.),
                        Complex::new(0., <$t>::INFINITY),
                    ];
                    assert_eq!(
                        input(special.clone())
                            .is_nan()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![0, 0, 1, 0]
                    );
                    assert_eq!(
                        input(special.clone())
                            .is_inf()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![0, 0, 0, 1]
                    );
                    assert_eq!(
                        input(special)
                            .not()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![1, 0, 0, 0]
                    );
                    use ha_ndarray::NDArrayComplex;
                    let components = vec![Complex::<$t>::new(3., 4.), Complex::new(-0., -1.)];
                    assert_eq!(
                        input(components.clone())
                            .re()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![3., -0.]
                    );
                    assert_eq!(
                        input(components.clone())
                            .im()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![4., -1.]
                    );
                    assert_eq!(
                        input(components.clone())
                            .conj()
                            .unwrap()
                            .buffer()
                            .unwrap()
                            .to_slice()
                            .unwrap()
                            .into_vec(),
                        vec![Complex::new(3., -4.), Complex::new(-0., 1.)]
                    );
                    let angles = input(components)
                        .angle()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert!((angles[0] as f64 - 0.9272952180016122).abs() <= 8. * <$t>::EPSILON as f64);
                    let abs = input(vec![Complex::<$t>::new(3., 4.)]).abs().unwrap();
                    assert_eq!(
                        abs.buffer().unwrap().to_slice().unwrap().into_vec(),
                        vec![5.]
                    );
                    let re: Vec<$t> = NDArrayCast::<$t>::cast(input(values.clone()))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert_eq!(re, values.iter().map(|z| z.re).collect::<Vec<_>>());
                }};
            }
            dtype!(f32);
            dtype!(f64);
        }

        #[cfg(feature = "complex")]
        #[test]
        fn complex_binary_oracle_and_branches() {
            macro_rules! dtype {
                ($t:ty) => {{
                    use ha_ndarray::complex::Complex;
                    let left = vec![
                        Complex::<$t>::new(0.25, 0.5),
                        Complex::new(-2., 0.125),
                        Complex::new(-2., -0.125),
                    ];
                    let right = vec![Complex::<$t>::new(0.5, -0.25); 3];
                    macro_rules! check {
                        ($op:ident) => {{
                            let expression = input(left.clone()).$op(input(right.clone())).unwrap();
                            let values = expression.buffer().unwrap().to_slice().unwrap().into_vec();
                            for (i, ((a, b), v)) in left.iter().zip(&right).zip(values).enumerate() {
                                let reference = crate::conformance::oracle::complex(
                                    (a.re as f64, a.im as f64),
                                    Some((b.re as f64, b.im as f64)),
                                    if <$t>::MANTISSA_DIGITS == 24 { 32 } else { 64 },
                                    stringify!($op),
                                );
                                for v in [v, expression.read_value(&[i]).unwrap()] {
                                    for (v, r) in [(v.re as f64, reference.0), (v.im as f64, reference.1)] {
                                        assert!(
                                            (v - r).abs() <= 16. * <$t>::EPSILON as f64 * r.abs().max(1.),
                                            "{}({a},{b}): {v} != {r}",
                                            stringify!($op)
                                        );
                                    }
                                }
                            }
                        }};
                    }
                    check!(add);
                    check!(sub);
                    check!(mul);
                    check!(div);
                    check!(pow);
                    check!(log);
                    let branch = input(vec![Complex::<$t>::new(-1., 0.), Complex::new(-1., -0.)])
                        .ln()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert!(branch[0].im > 0. && branch[1].im < 0.);
                    let zeros = input(vec![Complex::<$t>::new(0., 0.); 2])
                        .pow(input(vec![Complex::<$t>::new(0., 0.); 2]))
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();
                    assert_eq!(zeros, vec![Complex::new(1., 0.); 2]);
                }};
            }
            dtype!(f32);
            dtype!(f64);
        }

        #[test]
        fn random_bounds_and_moments() {
            for normal in [false, true] {
                let values = random(normal, 65536);
                assert!(values.iter().all(|x| x.is_finite()));
                let mean = values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64;
                let variance = values
                    .iter()
                    .map(|&x| (x as f64 - mean).powi(2))
                    .sum::<f64>()
                    / values.len() as f64;
                if normal {
                    assert!(mean.abs() < 0.05);
                    assert!((variance - 1.).abs() < 0.1);
                } else {
                    assert!(values.iter().all(|&x| (0.0..1.0).contains(&x)));
                    assert!((mean - 0.5).abs() < 0.02);
                    assert!((variance - 1. / 12.).abs() < 0.02);
                }
            }
        }

        #[test]
        fn batched_diagonal_points() {
            use ha_ndarray::MatrixUnary;
            let a = input((0..18).collect::<Vec<i32>>())
                .reshape(shape![2, 3, 3])
                .unwrap()
                .diag()
                .unwrap();
            assert_eq!(
                a.buffer().unwrap().to_slice().unwrap().into_vec(),
                vec![0, 4, 8, 9, 13, 17]
            );
            for batch in 0..2 {
                for i in 0..3 {
                    assert_eq!(
                        a.read_value(&[batch, i]).unwrap(),
                        (batch * 9 + i * 4) as i32
                    );
                }
            }
        }

        #[test]
        fn matrix_wrapping_and_float_accuracy() {
            use ha_ndarray::MatrixDual;
            let a = input(vec![127i8; 6]).reshape(shape![2, 3]).unwrap();
            let b = input(vec![3i8; 6]).reshape(shape![3, 2]).unwrap();
            assert_eq!(
                a.matmul(b)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec(),
                vec![119; 4]
            );
            let a = input(vec![0.25f64; 6]).reshape(shape![2, 3]).unwrap();
            let b = input(vec![0.5f64; 6]).reshape(shape![3, 2]).unwrap();
            assert_eq!(
                a.matmul(b)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .to_slice()
                    .unwrap()
                    .into_vec(),
                vec![0.375; 4]
            );
        }
        aggregate_suite!();
    };
}
pub fn same_number<T: ha_ndarray::Number>(a: T, b: T) -> bool {
    use safecast::CastFrom;
    let scalar = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    match a.into() {
        number_general::Number::Float(_) => {
            scalar(f64::cast_from(a.into()), f64::cast_from(b.into()))
        }
        #[cfg(feature = "complex")]
        number_general::Number::Complex(_) => {
            let a = ha_ndarray::complex::Complex64::cast_from(a.into());
            let b = ha_ndarray::complex::Complex64::cast_from(b.into());
            scalar(a.re, b.re) && scalar(a.im, b.im)
        }
        _ => a == b,
    }
}
