use ha_ndarray::statistics::{Mean, Norm, StandardDeviation, StatisticsElement};

fn partial(values: &[f64]) -> (Mean, Norm, StandardDeviation) {
    (
        f64::mean_batch(values).unwrap(),
        f64::norm_batch(values),
        f64::deviation_batch(values).unwrap(),
    )
}

#[cfg(feature = "complex")]
#[test]
fn complex_mean_accumulates_both_components() {
    use ha_ndarray::complex::Complex64 as C;

    let left = C::mean_batch(&[C::new(1e308, 1.)]).unwrap();
    let right = C::mean_batch(&[C::new(1e308, 2.)]).unwrap();
    let mean = C::finish_mean(C::merge_mean(left, right).unwrap());
    assert_eq!(mean, C::new(1e308, 1.5));
}

#[test]
fn integer_statistics_preserve_differences_before_conversion() {
    let signed = [i64::MIN, i64::MAX];
    let unsigned = [u64::MAX - 1, u64::MAX];
    for split in 0..=unsigned.len() {
        let left = i64::mean_batch(&signed[..split]).unwrap();
        let right = i64::mean_batch(&signed[split..]).unwrap();
        for mean in [i64::merge_mean(left, right), i64::merge_mean(right, left)] {
            assert_eq!(i64::finish_mean(mean.unwrap()), -0.5);
        }

        let left = u64::deviation_batch(&unsigned[..split]).unwrap();
        let right = u64::deviation_batch(&unsigned[split..]).unwrap();
        for deviation in [
            u64::merge_deviation(left, right),
            u64::merge_deviation(right, left),
        ] {
            assert_eq!(u64::finish_deviation(deviation.unwrap()), 0.5);
        }

        let left = i64::deviation_batch(&signed[..split]).unwrap();
        let right = i64::deviation_batch(&signed[split..]).unwrap();
        close(
            i64::finish_deviation(i64::merge_deviation(left, right).unwrap()),
            2f64.powi(63),
        );
    }

    let one = u64::mean_batch(&[u64::MAX]).unwrap();
    let mut mean = u64::mean_batch(&[]).unwrap();
    for _ in 0..64 {
        mean = u64::merge_mean(u64::merge_mean(mean, mean).unwrap(), one).unwrap();
    }

    assert_eq!(u64::finish_mean(mean), u64::MAX as f64);
    assert!(matches!(
        u64::merge_mean(mean, one),
        Err(ha_ndarray::Error::Bounds(_))
    ));
}

fn close(actual: f64, expected: f64) {
    if expected == 0. || !expected.is_finite() {
        assert!(
            actual == expected || actual.is_nan() && expected.is_nan(),
            "actual {actual:e}, expected {expected:e}"
        );
    } else {
        assert!(
            (actual / expected - 1.).abs() <= 2e-14 || actual == expected,
            "actual {actual:e}, expected {expected:e}"
        );
    }
}

#[test]
fn extremes_and_subnormals_survive_all_batch_splits() {
    let tiny = f64::from_bits(1);
    for (values, expected) in [
        (vec![1e200], [1e200, 1e200, 0.]),
        (vec![1e-200], [1e-200, 1e-200, 0.]),
        (vec![1e308, 1e308], [1e308, 2f64.sqrt() * 1e308, 0.]),
        (vec![-1e200, 1e200], [0., 2f64.sqrt() * 1e200, 1e200]),
        (vec![-1e-200, 1e-200], [0., 2f64.sqrt() * 1e-200, 1e-200]),
        (vec![-f64::MAX, f64::MAX], [0., f64::INFINITY, f64::MAX]),
        (vec![tiny, tiny], [tiny, tiny, 0.]),
        (vec![-tiny, tiny], [0., tiny, tiny]),
        (vec![-0., 0.], [0.; 3]),
    ] {
        for split in 0..=values.len() {
            let (lm, ln, ls) = partial(&values[..split]);
            let (rm, rn, rs) = partial(&values[split..]);
            for actual in [
                [
                    lm.merge(rm).unwrap().finish(),
                    ln.merge(rn).finish(),
                    ls.merge(rs).unwrap().finish(),
                ],
                [
                    rm.merge(lm).unwrap().finish(),
                    rn.merge(ln).finish(),
                    rs.merge(ls).unwrap().finish(),
                ],
            ] {
                for (actual, expected) in actual.into_iter().zip(expected) {
                    close(actual, expected);
                }
            }
        }
    }

    for count in [3, 7, 63, 129, 4097, 8193] {
        let (mean, _, deviation) = partial(&vec![f64::MAX; count]);
        close(mean.finish(), f64::MAX);
        assert_eq!(deviation.finish(), 0.);
    }
}

#[test]
fn empty_and_nonfinite_classifications_are_merge_independent() {
    let (mean, norm, deviation) = partial(&[]);
    assert!(mean.finish().is_nan());
    assert_eq!(norm.finish().to_bits(), 0f64.to_bits());
    assert!(deviation.finish().is_nan());

    for (values, expected_mean, expected_norm) in [
        (
            vec![f64::INFINITY, 1., f64::INFINITY],
            f64::INFINITY,
            f64::INFINITY,
        ),
        (
            vec![f64::NEG_INFINITY, 0.],
            f64::NEG_INFINITY,
            f64::INFINITY,
        ),
        (
            vec![f64::INFINITY, f64::NEG_INFINITY],
            f64::NAN,
            f64::INFINITY,
        ),
        (vec![f64::NAN, f64::INFINITY], f64::NAN, f64::NAN),
        (vec![f64::INFINITY, f64::NAN], f64::NAN, f64::NAN),
    ] {
        for split in 0..=values.len() {
            let (lm, ln, ls) = partial(&values[..split]);
            let (rm, rn, rs) = partial(&values[split..]);
            close(lm.merge(rm).unwrap().finish(), expected_mean);
            close(rm.merge(lm).unwrap().finish(), expected_mean);
            close(ln.merge(rn).finish(), expected_norm);
            close(rn.merge(ln).finish(), expected_norm);
            assert!(ls.merge(rs).unwrap().finish().is_nan());
            assert!(rs.merge(ls).unwrap().finish().is_nan());
        }
    }
}

#[test]
fn implicit_zero_multiplicity_uses_bounded_merge_state() {
    let (mut mean, mut norm, mut deviation) = partial(&[0.]);
    for _ in 0..63 {
        mean = mean.merge(mean).unwrap();
        norm = norm.merge(norm);
        deviation = deviation.merge(deviation).unwrap();
    }
    let (nonzero_mean, nonzero_norm, nonzero_deviation) = partial(&[1e200]);
    let count = 2f64.powi(63);
    close(mean.merge(nonzero_mean).unwrap().finish(), 1e200 / count);
    close(norm.merge(nonzero_norm).finish(), 1e200);
    close(
        deviation.merge(nonzero_deviation).unwrap().finish(),
        1e200 / count.sqrt(),
    );
}

#[cfg(feature = "complex")]
#[test]
fn component_deviations_combine_before_subnormal_rounding() {
    use ha_ndarray::complex::Complex64 as C;

    let deviation = |values: &[C]| C::finish_deviation(C::deviation_batch(values).unwrap());
    let tiny = f64::from_bits(1);
    let (_, _, component) = partial(&[0., tiny]);
    assert_eq!(component.finish(), 0.);
    assert_eq!(deviation(&[C::new(0., 0.), C::new(tiny, tiny)][..]), tiny);
    assert_eq!(deviation(&[C::new(1e308, 1.), C::new(1e308, 2.)][..]), 0.5);
    assert_eq!(deviation(&[C::new(1., 1e308), C::new(2., 1e308)][..]), 0.5);
    assert!(deviation(&[]).is_nan());
    assert!(deviation(&[C::new(1e308, f64::INFINITY)]).is_nan());
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn finite_statistics_match_exact_rational_moments_across_merge_orders() {
    use rug::{Float, Rational};

    for values in [
        vec![1., 2., 3., 4.],
        vec![2f64.powi(40) + 1., 2f64.powi(40) + 3., 2f64.powi(40) + 5.],
        vec![1e200, -1e200, 3e199, -5e199, 7e198],
        vec![1e-200, -1e-200, 3e-201, -5e-201, 7e-202],
        vec![1e16, 1., -1e16, 3., 0., -2.],
        (0..129).map(|i| (i % 11) as f64 - 5.).collect(),
    ] {
        let count = Rational::from(values.len());
        let sum: Rational = values.iter().map(|&v| Rational::from_f64(v).unwrap()).sum();
        let squares: Rational = values
            .iter()
            .map(|&v| {
                let value = Rational::from_f64(v).unwrap();
                value.clone() * value
            })
            .sum();
        let mean = sum / &count;
        let variance = squares.clone() / &count - mean.clone() * &mean;
        let expected = [
            mean.to_f64(),
            Float::with_val(256, &squares).sqrt().to_f64(),
            Float::with_val(256, &variance).sqrt().to_f64(),
        ];

        for split in 0..=values.len() {
            let (lm, ln, ls) = partial(&values[..split]);
            let (rm, rn, rs) = partial(&values[split..]);
            for actual in [
                [
                    lm.merge(rm).unwrap().finish(),
                    ln.merge(rn).finish(),
                    ls.merge(rs).unwrap().finish(),
                ],
                [
                    rm.merge(lm).unwrap().finish(),
                    rn.merge(ln).finish(),
                    rs.merge(ls).unwrap().finish(),
                ],
            ] {
                for (actual, expected) in actual.into_iter().zip(expected) {
                    close(actual, expected);
                }
            }
        }

        let (mut mean, mut norm, mut deviation) = partial(&[]);
        for &value in values.iter().rev() {
            let (m, n, s) = partial(&[value]);
            mean = mean.merge(m).unwrap();
            norm = norm.merge(n);
            deviation = deviation.merge(s).unwrap();
        }
        for (actual, expected) in [mean.finish(), norm.finish(), deviation.finish()]
            .into_iter()
            .zip(expected)
        {
            close(actual, expected);
        }
    }
}

#[test]
fn count_overflow_is_fallible_and_does_not_mutate_the_accumulator() {
    // Build cardinality u64::MAX using 64 merges, without allocating its values.
    let (one_mean, _, one_deviation) = partial(&[3.]);
    let (mut mean, _, mut deviation) = partial(&[]);
    for _ in 0..64 {
        mean = mean.merge(mean).unwrap().merge(one_mean).unwrap();
        deviation = deviation
            .merge(deviation)
            .unwrap()
            .merge(one_deviation)
            .unwrap();
    }
    assert_eq!(mean.finish(), 3.);
    assert_eq!(deviation.finish(), 0.);
    assert!(matches!(
        mean.merge(one_mean),
        Err(ha_ndarray::Error::Bounds(_))
    ));
    assert!(matches!(
        deviation.merge(one_deviation),
        Err(ha_ndarray::Error::Bounds(_))
    ));
    assert!(matches!(mean.push(0.), Err(ha_ndarray::Error::Bounds(_))));
    assert!(matches!(
        deviation.push(0.),
        Err(ha_ndarray::Error::Bounds(_))
    ));
    assert_eq!(mean.finish(), 3.);
    assert_eq!(deviation.finish(), 0.);
}
