#![cfg(feature = "opencl")]

use ha_ndarray::opencl::ArrayBuf;
use ha_ndarray::{shape, NDArrayMath, NDArrayRead};

#[test]
fn range_parameters_are_checked_before_device_construction() -> Result<(), ha_ndarray::Error> {
    use ha_ndarray::opencl::ArrayOp;
    use ha_ndarray::Error;

    let signed = ArrayOp::range(-100i8, 100i8, shape![2])?;
    assert_eq!(&*signed.buffer()?.to_slice()?, &[-100, 0]);
    assert_eq!(signed.read_value(&[1])?, 0);
    let wide = ArrayOp::range(u64::MAX - 4, u64::MAX, shape![2])?;
    assert_eq!(&*wide.buffer()?.to_slice()?, &[u64::MAX - 4, u64::MAX - 2]);
    assert_eq!(wide.read_value(&[1])?, u64::MAX - 2);
    let signed_wide = ArrayOp::range(i64::MIN, i64::MAX, shape![3])?;
    assert_eq!(
        &*signed_wide.buffer()?.to_slice()?,
        &[i64::MIN, -3074457345618258603, 3074457345618258602]
    );
    assert_eq!(signed_wide.read_value(&[2])?, 3074457345618258602);
    let narrow = ArrayOp::range(-10i8, 10i8, shape![256])?;
    assert_eq!(&*narrow.buffer()?.to_slice()?, &vec![-10; 256]);
    let float = ArrayOp::range(-f64::MAX, f64::MAX, shape![2])?;
    assert_eq!(&*float.buffer()?.to_slice()?, &[-f64::MAX, 0.]);
    assert_eq!(float.read_value(&[0])?, -f64::MAX);
    assert!(matches!(
        ArrayOp::range(-f64::MAX, f64::MAX, shape![3]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        ArrayOp::range(0u8, 1u8, shape![0]),
        Err(Error::Unsupported(_))
    ));
    let singleton = ArrayOp::range(i64::MIN, i64::MAX, shape![1])?;
    assert_eq!(&*singleton.buffer()?.to_slice()?, &[i64::MIN]);
    Ok(())
}

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

#[test]
fn slice_boundaries_preserve_values_and_reject_unsupported_mutation(
) -> Result<(), ha_ndarray::Error> {
    use ha_ndarray::{
        range, AxisRange, NDArray, NDArrayMathScalar, NDArrayTransform, NDArrayWrite,
    };
    let mut source = ArrayBuf::new(
        ha_ndarray::opencl::OpenCL::copy_into_buffer(&[0i32, 1, 2, 3])?,
        shape![2, 1, 2],
    )?;
    let selection = range![(0..2).into(), (0..1).into(), (0..2).into()];
    let view = source.as_ref().slice(selection.clone())?;
    assert_eq!(&*view.buffer()?.to_slice()?, &[0, 1, 2, 3]);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(view.read_value(&[i, 0, j])?, (2 * i + j) as i32);
        }
    }
    drop(view);
    let replacement = ha_ndarray::ArrayBuf::new(vec![4i32, 5, 6, 7], shape![2, 1, 2])?;
    source.write(&replacement)?;
    assert_eq!(&*source.buffer()?.to_slice()?, &[4, 5, 6, 7]);
    source.write_value(8)?;
    assert_eq!(&*source.buffer()?.to_slice()?, &[8, 8, 8, 8]);
    let repeated = range![
        AxisRange::Of(vec![1, 1].into()),
        (0..1).into(),
        (0..2).into()
    ];
    let view = source.as_ref().slice(repeated.clone())?;
    assert_eq!(&*view.buffer()?.to_slice()?, &[8, 8, 8, 8]);
    drop(view);
    assert_eq!(&*source.buffer()?.to_slice()?, &[8, 8, 8, 8]);
    let scalar =
        source
            .as_ref()
            .slice(range![AxisRange::At(1), AxisRange::At(0), AxisRange::At(1)])?;
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.read_value(&[])?, 8);
    assert_eq!(&*scalar.buffer()?.to_slice()?, &[8]);
    let scalar = scalar.transpose(None)?.slice(range![])?;
    assert_eq!(&*scalar.buffer()?.to_slice()?, &[8]);
    let broadcast = scalar.broadcast(shape![2, 3])?.transpose(None)?.flip(1)?;
    assert_eq!(&*broadcast.buffer()?.to_slice()?, &[8; 6]);

    let mut copy = source.copy()?;
    copy.write_value(12)?;
    assert_eq!(&*copy.buffer()?.to_slice()?, &[12; 4]);
    assert_eq!(&*source.buffer()?.to_slice()?, &[8; 4]);
    let replacement = ArrayBuf::new(
        ha_ndarray::opencl::OpenCL::copy_into_buffer(&[1i32, 2, 3, 4])?,
        shape![2, 1, 2],
    )?;
    copy.write(&replacement)?;
    assert_eq!(&*copy.buffer()?.to_slice()?, &[1, 2, 3, 4]);
    assert_eq!(&*source.buffer()?.to_slice()?, &[8; 4]);
    for (coord, value) in [
        ([0, 0, 0], 9),
        ([1, 0, 0], 10),
        ([1, 0, 1], 11),
        ([1, 0, 1], 42),
    ] {
        copy.write_value_at(&coord, value)?;
        assert_eq!(copy.read_value(&coord)?, value);
    }
    assert_eq!(&*copy.buffer()?.to_slice()?, &[9, 2, 10, 42]);
    for coord in [&[2, 0, 0][..], &[0, 1, 0], &[0, 0, 2], &[0, 0], &[]] {
        assert!(copy.write_value_at(coord, -1).is_err());
        assert_eq!(&*copy.buffer()?.to_slice()?, &[9, 2, 10, 42]);
    }
    assert_eq!(&*source.buffer()?.to_slice()?, &[8; 4]);
    assert_eq!(
        &*copy.add_scalar(1)?.buffer()?.to_slice()?,
        &[10, 3, 11, 43]
    );

    Ok(())
}

#[test]
fn empty_matrix_preflight_rejects_before_device_work() -> Result<(), ha_ndarray::Error> {
    use ha_ndarray::{Error, MatrixDual, NDArrayTransform};
    let left = ArrayBuf::new(
        ha_ndarray::opencl::OpenCL::copy_into_buffer(&[1i32])?,
        shape![1, 1, 1],
    )?
    .slice(ha_ndarray::range![
        (0..0).into(),
        (0..1).into(),
        (0..1).into()
    ])?;
    let right = ArrayBuf::new(
        ha_ndarray::opencl::OpenCL::copy_into_buffer(&[1i32])?,
        shape![1, 1, 1],
    )?
    .slice(ha_ndarray::range![
        (0..0).into(),
        (0..1).into(),
        (0..1).into()
    ])?;
    assert!(matches!(left.matmul(right), Err(Error::Unsupported(_))));
    Ok(())
}

#[test]
fn scalar_construction_and_reduction_shapes() -> Result<(), ha_ndarray::Error> {
    use ha_ndarray::{axes, range, NDArray, NDArrayReduce, NDArrayTransform};
    let scalar = ArrayBuf::<i32>::constant(7, shape![])?;
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.read_value(&[])?, 7);
    assert_eq!(&*scalar.buffer()?.to_slice()?, &[7]);
    let converted = ArrayBuf::<i32>::convert(&[7i32][..], shape![])?;
    assert_eq!(&*converted.buffer()?.to_slice()?, &[7]);
    let view = scalar.as_ref().transpose(None)?.slice(range![])?;
    assert_eq!(&*view.buffer()?.to_slice()?, &[7]);
    let scalar_reduction = scalar.as_ref().sum(axes![], false)?;
    assert_eq!(scalar_reduction.shape(), &[]);
    assert_eq!(&*scalar_reduction.buffer()?.to_slice()?, &[7]);
    drop(scalar_reduction);
    drop(view);
    let broadcast = scalar.broadcast(shape![2, 3])?.transpose(None)?.flip(0)?;
    assert_eq!(broadcast.shape(), &[3, 2]);
    assert_eq!(&*broadcast.buffer()?.to_slice()?, &[7; 6]);
    for keepdims in [false, true] {
        let sum = broadcast.as_ref().sum(axes![0, 1], keepdims)?;
        assert_eq!(sum.shape(), if keepdims { &[1, 1][..] } else { &[][..] });
        assert_eq!(&*sum.buffer()?.to_slice()?, &[42]);
    }
    Ok(())
}

#[test]
fn random_arrays_are_stable_and_empty_construction_rejects() -> Result<(), ha_ndarray::Error> {
    use ha_ndarray::{ArrayAccess, Error, NDArray, NDArrayWrite};
    assert!(matches!(
        ArrayBuf::<f32>::random_normal(0),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        ArrayBuf::<f32>::random_uniform(0),
        Err(Error::Unsupported(_))
    ));
    for size in [1, 3, 64, 65] {
        for array in [
            ArrayBuf::<f32>::random_normal(size)?,
            ArrayBuf::<f32>::random_uniform(size)?,
        ] {
            assert_eq!(array.shape(), &[size]);
            let expected = array.buffer()?.to_slice()?.into_vec();
            assert!(expected.iter().all(|value| value.is_finite()));
            assert_eq!(&*array.buffer()?.to_slice()?, &expected);
            for (index, value) in expected.iter().enumerate() {
                assert_eq!(array.read_value(&[index])?.to_bits(), value.to_bits());
            }
            let borrowed = array.as_ref().into_read()?;
            assert_eq!(&*borrowed.buffer()?.to_slice()?, &expected);
            let mut copy = array.copy()?;
            copy.write_value(42.)?;
            assert_eq!(&*array.buffer()?.to_slice()?, &expected);
            let shared = ArrayAccess::from(array);
            let zero = shared.clone().sub(shared.clone())?;
            assert!(zero.buffer()?.to_slice()?.iter().all(|value| *value == 0.));
            let owned = shared.into_read()?;
            assert_eq!(&*owned.buffer()?.to_slice()?, &expected);
        }
    }
    Ok(())
}
