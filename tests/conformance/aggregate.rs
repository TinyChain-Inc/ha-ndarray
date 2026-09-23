macro_rules! aggregate_suite {
    () => {
        #[test]
        fn exact_aggregate_references() {
            use crate::conformance::oracle::{check_aggregate, ExactComplex};
            use ha_ndarray::{axes, ArrayAccess, MatrixDual, NDArray, NDArrayReduce};

            macro_rules! dtype {
                ($t:ty,$bits:expr,$complex:expr,$make:expr,$parts:expr) => {{
                    let make = $make;
                    let parts = $parts;
                    let check = |actual: $t, values: &[$t], product: bool| {
                        let terms: Vec<_> = values
                            .iter()
                            .map(|&v| {
                                let (re, im) = parts(v);
                                ExactComplex::new(re, im)
                            })
                            .collect();
                        let expected = terms.iter().fold(
                            ExactComplex::new(if product { 1. } else { 0. }, 0.),
                            |a, b| if product { a.mul(b) } else { a.add(b) },
                        );
                        check_aggregate(
                            parts(actual),
                            &expected,
                            &terms,
                            terms.len(),
                            $complex,
                            $bits,
                            product,
                        );
                    };

                    for n in [1, 7, 8, 9, 63, 64, 65, 129] {
                        for product in [false, true] {
                            let values: Vec<$t> = (0..2 * n * 3)
                                .map(|i| {
                                    let re = if product {
                                        1. + ((i % 7) as f64 - 3.) / 1024.
                                    } else {
                                        match i % 4 {
                                            0 => 1.,
                                            1 => -1.,
                                            2 => (i % 13) as f64 / 10.,
                                            _ => 1. / 65536.,
                                        }
                                    };

                                    let im = if $complex {
                                        ((i % 11) as f64 - 5.) / if product { 4096. } else { 17. }
                                    } else {
                                        0.
                                    };

                                    make(re, im)
                                })
                                .collect();
                            let small = values[..n].to_vec();
                            let actual = if product {
                                input(small.clone()).product_all().unwrap()
                            } else {
                                input(small.clone()).sum_all().unwrap()
                            };

                            check(actual, &small, product);

                            for multi in [false, true] {
                                for keep in [false, true] {
                                    // Original [batch, term, column] -> [column, reversed term, batch].
                                    let a = ArrayAccess::from(input(values.clone()))
                                        .reshape(shape![2, n, 3])
                                        .unwrap()
                                        .transpose(axes![2, 1, 0])
                                        .unwrap()
                                        .flip(1)
                                        .unwrap();
                                    let a = ArrayAccess::from(a);
                                    let axes = if multi { axes![0, 1] } else { axes![1] };
                                    // Pin kernel execution independently of automatic Array scheduling.
                                    let permutation = if multi { axes![2, 0, 1] } else { axes![0, 2, 1] };
                                    let explicit = reduce_axis(
                                        a.clone().transpose(permutation).unwrap().into_access(),
                                        if multi { 3 * n } else { n },
                                        product,
                                    );
                                    let result = if product {
                                        a.product(axes, keep).unwrap()
                                    } else {
                                        a.sum(axes, keep).unwrap()
                                    };

                                    let expected_shape: ha_ndarray::Shape = match (multi, keep) {
                                        (false, false) => shape![3, 2],
                                        (false, true) => shape![3, 1, 2],
                                        (true, false) => shape![2],
                                        (true, true) => shape![1, 1, 2],
                                    };

                                    assert_eq!(result.shape(), expected_shape.as_slice());
                                    let output = result.buffer().unwrap().to_slice().unwrap().into_vec();

                                    for (i, &value) in output.iter().enumerate() {
                                        let batch = i % 2;
                                        let columns: Vec<usize> =
                                            if multi { (0..3).collect() } else { vec![i / 2] };
                                        let group: Vec<_> = columns
                                            .into_iter()
                                            .flat_map(|c| {
                                                (0..n).rev().map(move |j| (batch * n + j) * 3 + c)
                                            })
                                            .map(|i| values[i])
                                            .collect();
                                        check(value, &group, product);
                                        check(explicit[i], &group, product);
                                    }
                                }
                            }
                        }
                    }

                    for (rows, inner, columns) in [(2, 3, 2), (8, 8, 8), (9, 17, 7)] {
                        let left: Vec<$t> = (0..3 * rows * inner)
                            .map(|i| {
                                make(
                                    ((i % 19) as f64 - 9.) / 10.,
                                    if $complex {
                                        ((i % 7) as f64 - 3.) / 13.
                                    } else {
                                        0.
                                    },
                                )
                            })
                            .collect();
                        // Transpose the stored right operand before multiplying.
                        let right: Vec<$t> = (0..3 * columns * inner)
                            .map(|i| {
                                make(
                                    ((i % 23) as f64 - 11.) / 17.,
                                    if $complex {
                                        ((i % 5) as f64 - 2.) / 7.
                                    } else {
                                        0.
                                    },
                                )
                            })
                            .collect();
                        let a = input(left.clone()).reshape(shape![3, rows, inner]).unwrap();
                        let b = input(right.clone())
                            .reshape(shape![3, columns, inner])
                            .unwrap()
                            .transpose(axes![0, 2, 1])
                            .unwrap();
                        let result = a.matmul(b).unwrap();

                        assert_eq!(result.shape(), &[3, rows, columns]);
                        // Point reads of matrix products remain unsupported.
                        assert!(result.read_value(&[0, 0, 0]).is_err());
                        let output = result.buffer().unwrap().to_slice().unwrap().into_vec();

                        for batch in 0..3 {
                            for row in 0..rows {
                                for col in 0..columns {
                                    let terms: Vec<_> = (0..inner)
                                        .map(|k| {
                                            let (ar, ai) = parts(left[(batch * rows + row) * inner + k]);
                                            let (br, bi) =
                                                parts(right[(batch * columns + col) * inner + k]);
                                            ExactComplex::new(ar, ai).mul(&ExactComplex::new(br, bi))
                                        })
                                        .collect();
                                    let expected = terms
                                        .iter()
                                        .fold(ExactComplex::new(0., 0.), |a, b| a.add(b));
                                    check_aggregate(
                                        parts(output[(batch * rows + row) * columns + col]),
                                        &expected,
                                        &terms,
                                        inner,
                                        $complex,
                                        $bits,
                                        false,
                                    );
                                }
                            }
                        }
                    }
                }};
            }

            dtype!(f32, 32, false, |re: f64, _im: f64| re as f32, |v: f32| (
                v as f64, 0.
            ));
            dtype!(f64, 64, false, |re: f64, _im: f64| re, |v: f64| (v, 0.));
            #[cfg(feature = "complex")]
            {
                use ha_ndarray::complex::{Complex32, Complex64};

                dtype!(
                    Complex32,
                    32,
                    true,
                    |re: f64, im: f64| Complex32::new(re as f32, im as f32),
                    |v: Complex32| (v.re as f64, v.im as f64)
                );
                dtype!(
                    Complex64,
                    64,
                    true,
                    |re: f64, im: f64| Complex64::new(re, im),
                    |v: Complex64| (v.re, v.im)
                );
            }
        }

        #[test]
        fn wider_integer_exact_wrapping() {
            use rug::Integer;

            macro_rules! dtype {
                ($t:ty,$bits:expr,$signed:expr) => {{
                    let values = vec![
                        <$t>::MIN,
                        <$t>::MAX,
                        0,
                        1,
                        2,
                        (1 as $t).wrapping_neg(),
                    ];
                    let left: Vec<$t> = values
                        .iter()
                        .copied()
                        .flat_map(|a| std::iter::repeat_n(a, values.len()))
                        .collect();
                    let right: Vec<$t> = values.iter().copied().cycle().take(left.len()).collect();
                    let wrap = |mut v: Integer| {
                        let modulus = Integer::from(1) << $bits;
                        v %= &modulus;

                        if v < 0 {
                            v += &modulus;
                        }

                        if $signed && v >= (Integer::from(1) << ($bits - 1)) {
                            v -= modulus;
                        }

                        v.to_i128().unwrap() as $t
                    };

                    macro_rules! operation {
                        ($method:ident,$reference:expr) => {{
                            let expr = input(left.clone())
                                .$method(input(right.clone()))
                                .unwrap();
                            let output = expr.buffer().unwrap().to_slice().unwrap().into_vec();

                            for (i, ((&a, &b), v)) in left.iter().zip(&right).zip(output).enumerate() {
                                let expected = wrap(($reference)(Integer::from(a), Integer::from(b)));

                                assert_eq!(v, expected, "{}({a},{b})", stringify!($method));
                                assert_eq!(expr.read_value(&[i]).unwrap(), expected);
                            }
                        }};
                    }

                    operation!(add, |a: Integer, b: Integer| a + b);
                    operation!(sub, |a: Integer, b: Integer| a - b);
                    operation!(mul, |a: Integer, b: Integer| a * b);
                    operation!(div, |a: Integer, b: Integer| if b == 0 {
                        Integer::from(0)
                    } else {
                        a / b
                    });
                    operation!(rem, |a: Integer, b: Integer| if b == 0 {
                        Integer::from(0)
                    } else {
                        a % b
                    });

                    for n in [7, 65, 129] {
                        let values: Vec<$t> = [<$t>::MAX, 2, 3]
                            .into_iter()
                            .cycle()
                            .take(n)
                            .collect();
                        let sum = values
                            .iter()
                            .fold(Integer::from(0), |a, &b| a + Integer::from(b));
                        let product = values
                            .iter()
                            .fold(Integer::from(1), |a, &b| a * Integer::from(b));

                        assert_eq!(input(values.clone()).sum_all().unwrap(), wrap(sum));
                        assert_eq!(input(values).product_all().unwrap(), wrap(product));
                    }
                }};
            }

            dtype!(i8, 8, true);
            dtype!(u8, 8, false);
            dtype!(i16, 16, true);
            dtype!(u16, 16, false);
            dtype!(i32, 32, true);
            dtype!(u32, 32, false);
            dtype!(i64, 64, true);
            dtype!(u64, 64, false);
        }
    };
}
