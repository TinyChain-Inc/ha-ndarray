use ha_ndarray::{
    axes, range, shape, ArrayAccess, ArrayBuf, ArrayOp, AxisRange, Error, NDArray, NDArrayCompare,
    NDArrayCompareScalar, NDArrayRead, NDArrayReduce, NDArrayReduceAll, NDArrayReduceBoolean,
    NDArrayTransform,
};

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let array = ArrayBuf::constant(1, shape![10_usize.pow(x)]).map(ArrayAccess::from)?;
        assert_eq!(array.size() as i32, array.sum_all()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_range_axis() -> Result<(), Error> {
    let array = ArrayOp::range(0, 10, shape![1, 2, 5]).map(ArrayAccess::from)?;
    let actual = array.sum(axes![1], false)?;
    let expected = ArrayBuf::new(vec![5, 7, 9, 11, 13], shape![1, 5])?;

    println!("actual: {:?}", actual.buffer()?.to_slice()?);

    assert!(actual.eq(expected)?.all()?);
    Ok(())
}

#[test]
fn test_reduce_sum_axis() -> Result<(), Error> {
    let shapes = vec![
        shape![5],
        shape![2, 3, 4],
        shape![7],
        shape![2, 3, 129],
        shape![299],
        shape![51, 1, 13, 7, 64, 10, 2],
    ];

    for shape in shapes {
        let array = ArrayBuf::constant(1u32, shape.clone()).map(ArrayAccess::from)?;

        for x in 0..shape.len() {
            let expected = shape[x] as u32;
            let actual = array.clone().sum(axes![x], false)?;
            let eq = actual.eq_scalar(expected)?;
            assert!(eq.all()?);
        }
    }

    Ok(())
}

#[test]
fn empty_axis_reductions_return_errors() -> Result<(), Error> {
    let empty = ArrayBuf::new(Vec::<u32>::new(), shape![2, 0])?;
    let source = ArrayBuf::new(vec![1u32, 2, 3, 4], shape![2, 2])?;
    let sliced = source
        .as_ref()
        .slice(range![AxisRange::In(0, 2, 1), AxisRange::In(1, 1, 1)])?;

    for keepdims in [false, true] {
        for result in [
            empty.as_ref().sum(axes![1], keepdims),
            empty.as_ref().product(axes![1], keepdims),
            empty.as_ref().min(axes![1], keepdims),
            empty.as_ref().max(axes![1], keepdims),
        ] {
            assert!(matches!(result, Err(Error::Bounds(_))));
        }
        for result in [
            sliced.as_ref().sum(axes![1], keepdims),
            sliced.as_ref().product(axes![1], keepdims),
            sliced.as_ref().min(axes![1], keepdims),
            sliced.as_ref().max(axes![1], keepdims),
        ] {
            assert!(matches!(result, Err(Error::Bounds(_))));
        }

        let empty_output = ArrayBuf::new(Vec::<u32>::new(), shape![0, 2])?;
        for result in [
            empty_output.as_ref().sum(axes![1], keepdims),
            empty_output.as_ref().product(axes![1], keepdims),
            empty_output.as_ref().min(axes![1], keepdims),
            empty_output.as_ref().max(axes![1], keepdims),
        ] {
            let result = result?;
            assert_eq!(
                result.shape(),
                if keepdims { &[0, 1][..] } else { &[0][..] }
            );
            assert_eq!(result.buffer()?.len(), 0);
        }
    }

    let empty = ArrayBuf::new(Vec::<u32>::new(), shape![usize::MAX, 0, 2])?;
    assert!(matches!(
        empty.as_ref().sum(axes![0], false),
        Err(Error::Bounds(_))
    ));
    assert!(matches!(
        empty.sum(axes![0, 2], false),
        Err(Error::Bounds(_))
    ));
    Ok(())
}

#[test]
fn full_axis_reductions_preserve_scalar_shapes() -> Result<(), Error> {
    let source = ArrayBuf::new(vec![1i32, 2, 3, 4, 5, 6], shape![2, 3])?;
    for keepdims in [false, true] {
        for (result, expected) in [
            (source.as_ref().sum(axes![0, 1], keepdims)?, 21),
            (source.as_ref().product(axes![0, 1], keepdims)?, 720),
            (source.as_ref().min(axes![0, 1], keepdims)?, 1),
            (source.as_ref().max(axes![0, 1], keepdims)?, 6),
        ] {
            assert_eq!(result.shape(), if keepdims { &[1, 1][..] } else { &[][..] });
            assert_eq!(&*result.buffer()?.to_slice()?, &[expected]);
            assert_eq!(
                result.read_value(if keepdims { &[0, 0] } else { &[] })?,
                expected
            );
        }
        for result in [
            source.as_ref().sum(axes![], keepdims)?,
            source.as_ref().product(axes![], keepdims)?,
            source.as_ref().min(axes![], keepdims)?,
            source.as_ref().max(axes![], keepdims)?,
        ] {
            assert_eq!(result.shape(), &[2, 3]);
            assert_eq!(&*result.buffer()?.to_slice()?, &[1, 2, 3, 4, 5, 6]);
        }
        let scalar = ArrayBuf::new(vec![7i32], shape![])?;
        let result = scalar.sum(axes![], keepdims)?;
        assert_eq!(result.shape(), &[]);
        assert_eq!(result.read_value(&[])?, 7);
    }
    Ok(())
}
