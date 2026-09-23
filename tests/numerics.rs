//! The same numerical cases execute on explicit host and OpenCL buffers.
#[macro_use]
mod conformance;

mod host {
    use ha_ndarray::{host::ArrayBuf, shape, Number};

    fn input<T: Number>(values: Vec<T>) -> ArrayBuf<T> {
        let len = values.len();
        ArrayBuf::new(values.into(), shape![len]).unwrap()
    }

    fn reduce_axis<T: Number, A: ha_ndarray::Access<T>>(
        access: A,
        stride: usize,
        product: bool,
    ) -> Vec<T> {
        use ha_ndarray::PlatformInstance;
        use ha_ndarray::{ops::ReduceAxes, Access};

        let platform = ha_ndarray::host::Host::select(access.size());
        let result = if product {
            ReduceAxes::product(platform, access, stride)
        } else {
            ReduceAxes::sum(platform, access, stride)
        }
        .unwrap();
        result.read().unwrap().to_slice().unwrap().into_vec()
    }

    fn reduce_max(values: Vec<f64>, stride: usize) -> Vec<f64> {
        use ha_ndarray::{ops::ReduceAxes, Access};

        ReduceAxes::max(
            ha_ndarray::host::Host::Heap(ha_ndarray::host::Heap),
            input(values).into_access(),
            stride,
        )
        .unwrap()
        .read()
        .unwrap()
        .to_slice()
        .unwrap()
        .into_vec()
    }

    #[cfg(feature = "complex")]
    #[test]
    fn fft_batches_and_inverse() {
        macro_rules! dtype {
            ($t:ty,$bits:expr) => {{
                use crate::conformance::oracle::{check_dft, fft_bound, rational};
                use ha_ndarray::{complex::Complex, NDArrayFourier, NDArrayRead, NDArrayTransform};

                for n in [1, 3, 8, 17] {
                    let values: Vec<_> = (0..3 * n)
                        .map(|i| {
                            Complex::<$t>::new(
                                ((i % 13) as $t - 6.) / 7.,
                                ((i % 17) as $t - 8.) / 11.,
                            )
                        })
                        .collect();
                    let a = ha_ndarray::ArrayAccess::from(input(values.clone()).reshape(shape![3, n]).unwrap());
                    let forward = a.clone().fft().unwrap();

                    assert!(forward.read_value(&[0, 0]).is_err());
                    let actual = forward.buffer().unwrap().to_slice().unwrap().into_vec();
                    let inverse = a.ifft().unwrap();

                    assert!(inverse.read_value(&[0, 0]).is_err());
                    let backward = inverse.buffer().unwrap().to_slice().unwrap().into_vec();
                    let roundtrip = forward
                        .ifft()
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .to_slice()
                        .unwrap()
                        .into_vec();

                    for batch in 0..3 {
                        let original: Vec<_> = values[batch * n..(batch + 1) * n]
                            .iter()
                            .map(|v| (v.re as f64, v.im as f64))
                            .collect();
                        let transformed: Vec<_> = actual[batch * n..(batch + 1) * n]
                            .iter()
                            .map(|v| (v.re as f64, v.im as f64))
                            .collect();
                        // Each input component has forward error <= E. Each inverse
                        // component receives at most (|cos|+|sin|)E <= 2E per term.
                        let propagated = fft_bound(&original, $bits) * rug::Integer::from(2 * n);
                        let roundtrip_bound = propagated + fft_bound(&transformed, $bits);

                        for k in 0..n {
                            let f = actual[batch * n + k];
                            let b = backward[batch * n + k];
                            let r = roundtrip[batch * n + k];
                            check_dft(
                                (f.re as f64, f.im as f64),
                                &original,
                                k,
                                false,
                                $bits,
                            );
                            check_dft((b.re as f64, b.im as f64), &original, k, true, $bits);
                            check_dft(
                                (r.re as f64, r.im as f64),
                                &transformed,
                                k,
                                true,
                                $bits,
                            );

                            for (value, source) in [(r.re as f64, original[k].0), (r.im as f64, original[k].1)]
                            {
                                let error = (rational(value) - rational(source) * rug::Integer::from(n)).abs();

                                assert!(
                                    error <= roundtrip_bound,
                                    "roundtrip f{}, N={n}, batch={batch}, k={k}: {error} > {roundtrip_bound}",
                                    $bits
                                );
                            }
                        }
                    }
                }
            }};
        }

        dtype!(f32, 32);
        dtype!(f64, 64);
    }

    fn random(normal: bool, size: usize) -> Vec<f32> {
        use ha_ndarray::{ops::Random, Access};

        let platform = ha_ndarray::host::Host::Heap(ha_ndarray::host::Heap);

        if normal {
            platform
                .random_normal(size)
                .unwrap()
                .read()
                .unwrap()
                .to_slice()
                .unwrap()
                .into_vec()
        } else {
            platform
                .random_uniform(size)
                .unwrap()
                .read()
                .unwrap()
                .to_slice()
                .unwrap()
                .into_vec()
        }
    }

    conformance_suite!();
}

#[cfg(feature = "opencl")]
mod opencl {
    use ha_ndarray::{
        opencl::{ArrayBuf, OpenCL},
        shape, Number,
    };

    fn input<T: Number>(values: Vec<T>) -> ArrayBuf<T> {
        let len = values.len();
        ArrayBuf::new(OpenCL::copy_into_buffer(&values).unwrap(), shape![len]).unwrap()
    }

    #[cfg(feature = "complex")]
    #[test]
    fn fft_is_explicitly_unsupported() {
        use ha_ndarray::{complex::Complex32, ArrayAccess, Error, NDArrayFourier};

        let a = ArrayAccess::from(input(vec![Complex32::new(1., 2.); 3]));

        assert!(matches!(a.clone().fft(), Err(Error::Unsupported(_))));
        assert!(matches!(a.ifft(), Err(Error::Unsupported(_))));
    }

    fn reduce_axis<T: Number, A: ha_ndarray::Access<T>>(
        access: A,
        stride: usize,
        product: bool,
    ) -> Vec<T> {
        use ha_ndarray::{ops::ReduceAxes, Access};

        let platform = OpenCL;
        let result = if product {
            ReduceAxes::product(platform, access, stride)
        } else {
            ReduceAxes::sum(platform, access, stride)
        }
        .unwrap();
        result.read().unwrap().to_slice().unwrap().into_vec()
    }

    fn reduce_max(values: Vec<f64>, stride: usize) -> Vec<f64> {
        use ha_ndarray::{ops::ReduceAxes, Access};

        {
            let op = ReduceAxes::max(OpenCL, input(values).into_access(), stride).unwrap();
            let values = op.read().unwrap().to_slice().unwrap().into_vec();

            for (i, v) in values.iter().enumerate() {
                assert_eq!(op.read_value(i).unwrap(), *v);
            }

            values
        }
    }

    fn random(normal: bool, size: usize) -> Vec<f32> {
        use ha_ndarray::{ops::Random, Access};

        if normal {
            OpenCL
                .random_normal(size)
                .unwrap()
                .read()
                .unwrap()
                .to_slice()
                .unwrap()
                .into_vec()
        } else {
            OpenCL
                .random_uniform(size)
                .unwrap()
                .read()
                .unwrap()
                .to_slice()
                .unwrap()
                .into_vec()
        }
    }

    conformance_suite!();
}
