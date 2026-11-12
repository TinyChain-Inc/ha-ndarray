use ha_ndarray::{
    host, range, shape, Array, ArrayAccess, ArrayBuf, ArrayOp, Buffer, Error, NDArray,
    NDArrayCompare, NDArrayCompareScalar, NDArrayMath, NDArrayRead, NDArrayReduceAll,
    NDArrayReduceBoolean, NDArrayTransform, NDArrayWrite,
};

#[test]
fn test_concat() -> Result<(), Error> {
    let first = ArrayBuf::new(vec![1, 2, 3], shape![1, 3])?;
    let second = ArrayBuf::new(vec![4, 5, 6], shape![1, 3])?;

    let concatenated = Array::concat(vec![first, second])?;
    assert_eq!(
        concatenated.buffer()?.to_slice()?.into_vec(),
        vec![1, 2, 3, 4, 5, 6]
    );

    assert_eq!(concatenated.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn retained_host_buffer_size_includes_spare_allocation() {
    use get_size::GetSize;

    let mut values = Vec::<f64>::with_capacity(64);
    values.extend([1., 2., 3.]);
    let capacity = values.capacity();
    let buffer = Buffer::Host(host::Buffer::Heap(values));
    assert_eq!(
        buffer.get_size(),
        std::mem::size_of_val(&buffer) + capacity * std::mem::size_of::<f64>()
    );
    assert!(ArrayBuf::<u32, Buffer<u32>>::constant(0, shape![usize::MAX, 2]).is_err());
}

#[test]
fn test_transpose_concat() -> Result<(), Error> {
    let first = ArrayBuf::new(vec![1, 2, 3, 4, 5, 6], shape![2, 3])?;
    let second = ArrayBuf::new(vec![7, 8, 9, 10, 11, 12], shape![2, 3])?;
    let concatenated = Array::transpose_concat(vec![first, second], 1)?;

    assert_eq!(concatenated.shape(), &[2, 6]);

    assert_eq!(
        concatenated.buffer()?.to_slice()?.into_vec(),
        vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12],
    );

    Ok(())
}

#[test]
fn test_range() -> Result<(), Error> {
    use rayon::prelude::*;

    macro_rules! check {
        ($start:expr, $stop:expr, $size:expr, $expected:expr) => {{
            let expected = $expected;
            let range = ArrayOp::range($start, $stop, shape![$size])?;
            assert_eq!(&*range.buffer()?.to_slice()?, &expected);
            for (offset, value) in expected.iter().enumerate() {
                assert_eq!(&range.read_value(&[offset])?, value);
            }
        }};
    }

    check!(-100i8, 100i8, 2, [-100, 0]);
    check!(i8::MIN, i8::MAX, 2, [i8::MIN, -1]);
    check!(-10i8, 10i8, 256, vec![-10; 256]);
    check!(0u8, u8::MAX, 256, vec![0; 256]);
    check!(u64::MAX - 4, u64::MAX, 2, [u64::MAX - 4, u64::MAX - 2]);
    check!(
        i64::MIN,
        i64::MAX,
        3,
        [i64::MIN, -3074457345618258603, 3074457345618258602]
    );
    check!(-5i16, 5i16, 0, Vec::<i16>::new());
    check!(i64::MIN, i64::MAX, 1, [i64::MIN]);
    check!(7i32, 7i32, 3, [7, 7, 7]);
    check!(-f32::MAX, f32::MAX, 2, [-f32::MAX, 0.]);
    check!(-f64::MAX, f64::MAX, 2, [-f64::MAX, 0.]);
    check!(-f64::MAX, f64::MAX, 1, [-f64::MAX]);
    assert!(matches!(
        ArrayOp::range(-f64::MAX, f64::MAX, shape![3]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        ArrayOp::range(2i8, 1i8, shape![0]),
        Err(Error::Bounds(_))
    ));
    for (start, stop) in [(f64::NEG_INFINITY, 1.), (0., f64::INFINITY), (f64::NAN, 1.)] {
        assert!(ArrayOp::range(start, stop, shape![2]).is_err());
    }

    let size = 1_000_000;

    let expected = ArrayBuf::new(
        (0..size)
            .into_par_iter()
            .map(|n| n as f32 * 0.5)
            .collect::<Vec<f32>>(),
        shape![size],
    )?;

    let actual = ArrayOp::range(0f32, 500_000f32, shape![1_000_000])?;

    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_random_normal() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayBuf::<f32, Buffer<f32>>::random_normal(size)?;

    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    assert!(array.as_ref().gt_scalar(1.)?.any()?);
    assert!(array.as_ref().lt_scalar(-1.)?.any()?);
    check_stable_random(array)?;
    for size in [0, 1, 3, 63, 64, 65] {
        check_stable_random(ArrayBuf::<f32, Buffer<f32>>::random_normal(size)?)?;
    }

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayBuf::<f32, Buffer<f32>>::random_uniform(size)?;

    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    assert!(array.as_ref().ge_scalar(-1.)?.all()?);
    assert!(array.as_ref().le_scalar(1.)?.all()?);
    check_stable_random(array)?;
    for size in [0, 1, 3, 63, 64, 65] {
        check_stable_random(ArrayBuf::<f32, Buffer<f32>>::random_uniform(size)?)?;
    }

    Ok(())
}

#[test]
fn scalar_shapes_are_consistent() -> Result<(), Error> {
    let scalar = ArrayBuf::new(vec![7i32], shape![])?;
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.size(), 1);
    assert_eq!(scalar.read_value(&[])?, 7);
    let scalar = scalar.into_read()?;
    let converted = ArrayBuf::<i32, Buffer<i32>>::convert(&[7i32][..], shape![])?;
    let constant = ArrayBuf::<i32, Buffer<i32>>::constant(7, shape![])?;
    for array in [scalar, converted, constant] {
        assert_eq!(&*array.buffer()?.to_slice()?, &[7]);
        assert_eq!(
            array
                .as_ref()
                .reshape(shape![1])?
                .reshape(shape![])?
                .read_value(&[])?,
            7
        );
    }
    assert!(ArrayBuf::new(Vec::<i32>::new(), shape![]).is_err());
    assert!(ArrayBuf::new(vec![1i32, 2], shape![]).is_err());
    assert!(ArrayBuf::<i32, Buffer<i32>>::convert(&[1i32, 2][..], shape![]).is_err());

    let empty = ArrayBuf::new(Vec::<i32>::new(), shape![0])?;
    assert_eq!(empty.size(), 0);
    assert!(empty.reshape(shape![]).is_err());
    Ok(())
}

fn check_stable_random(array: ArrayBuf<f32, Buffer<f32>>) -> Result<(), Error> {
    let expected = array.buffer()?.to_slice()?.into_vec();
    assert!(expected.iter().all(|value| value.is_finite()));
    let shared = ArrayAccess::from(array);
    assert_eq!(&*shared.buffer()?.to_slice()?, &expected);
    if !expected.is_empty() {
        for index in [0, expected.len() / 2, expected.len() - 1] {
            assert_eq!(
                shared.read_value(&[index])?.to_bits(),
                expected[index].to_bits()
            );
        }
    }
    let view = shared
        .as_ref()
        .slice(range![(0..expected.len().min(3)).into()])?;
    assert_eq!(
        &*view.buffer()?.to_slice()?,
        &expected[..expected.len().min(3)]
    );
    drop(view);

    let difference = shared.clone().sub(shared.clone())?;
    assert!(difference
        .buffer()?
        .to_slice()?
        .iter()
        .all(|value| *value == 0.));
    let borrowed = shared.as_ref().into_read()?;
    assert_eq!(&*borrowed.buffer()?.to_slice()?, &expected);

    let mut copy = shared.copy()?;
    copy.write_value(42.)?;
    assert!(copy.buffer()?.to_slice()?.iter().all(|value| *value == 42.));
    assert_eq!(&*shared.buffer()?.to_slice()?, &expected);

    let owned = shared.into_read()?;
    assert_eq!(&*owned.buffer()?.to_slice()?, &expected);
    Ok(())
}
