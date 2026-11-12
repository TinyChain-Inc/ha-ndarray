#[cfg(feature = "complex")]
use ha_ndarray::fft;
use ha_ndarray::{
    axes, range, shape, stackvec, ArrayAccess, ArrayBuf, ArrayOp, AxisRange, Error, MatrixUnary,
    NDArray, NDArrayCompare, NDArrayMath, NDArrayRead, NDArrayReduce, NDArrayReduceAll,
    NDArrayReduceBoolean, NDArrayTransform, NDArrayWrite,
};

#[test]
fn test_broadcast_small() -> Result<(), Error> {
    let scalar = ArrayBuf::new(vec![7i32], shape![])?;
    let scalar_view = scalar.as_ref().transpose(None)?.slice(range![])?;
    assert_eq!(scalar_view.shape(), &[]);
    assert_eq!(scalar_view.read_value(&[])?, 7);
    assert_eq!(&*scalar_view.buffer()?.to_slice()?, &[7]);
    let broadcast = scalar.as_ref().broadcast(shape![2, 1, 3])?;
    assert_eq!(&*broadcast.buffer()?.to_slice()?, &[7; 6]);
    assert_eq!(broadcast.read_value(&[1, 0, 2])?, 7);
    let empty = scalar.broadcast(shape![0])?;
    assert_eq!(empty.buffer()?.len(), 0);

    let data = vec![5, 1];
    let strides = ArrayBuf::new(data.to_vec(), shape![2])?.broadcast(shape![1, 2])?;
    assert_eq!(&*strides.buffer()?.to_slice()?, &data);

    let source = ArrayBuf::new(vec![1u32, 2, 3, 4], shape![2, 1, 2])?;
    let expanded = source.broadcast(shape![2, 3, 2])?;
    let expected = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4];
    assert_eq!(&*expanded.buffer()?.to_slice()?, &expected);
    for (offset, &value) in expected.iter().enumerate() {
        assert_eq!(
            expanded.read_value(&[offset / 6, (offset / 2) % 3, offset % 2])?,
            value
        );
    }
    Ok(())
}

#[test]
fn test_broadcast_large() -> Result<(), Error> {
    let input = ArrayBuf::new(vec![1u32; 600], shape![300, 1, 2])?;
    let output = input.broadcast(shape![300, 250, 2])?;
    assert_eq!(output.shape(), &[300, 250, 2]);
    assert_eq!(output.sum_all()?, 150000);
    Ok(())
}

#[test]
fn test_flip() -> Result<(), Error> {
    let shape = shape![2, 3];
    let source = ArrayBuf::new(vec![0, 1, 2, 3, 4, 5], shape.clone())?;

    let flip0_expected = ArrayBuf::new(vec![3, 4, 5, 0, 1, 2], shape.clone())?;
    let flip0_actual = source.as_ref().flip(0)?;
    assert!(flip0_expected.eq(flip0_actual)?.all()?);

    let flip1_expected = ArrayBuf::new(vec![2, 1, 0, 5, 4, 3], shape)?;
    let flip1_actual = source.as_ref().flip(1)?;
    assert!(flip1_expected.eq(flip1_actual)?.all()?);

    Ok(())
}

#[test]
fn test_slice_1d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 4, shape![4])?;

    let expected = ArrayOp::range(1, 3, shape![2])?;

    let actual = input.slice(range![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_slice_2d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 12, shape![4, 3])?;
    let expected = ArrayOp::range(3, 9, shape![2, 3])?;

    let actual = input.slice(range![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(
        expected.as_ref().eq(actual.as_ref())?.all()?,
        "expected {:?} but found {:?}",
        expected.buffer()?.to_slice()?,
        actual.buffer()?.to_slice()?
    );

    let source = ArrayBuf::new((0..12).collect::<Vec<i32>>(), shape![4, 3])?;
    let scalar = source
        .as_ref()
        .slice(range![AxisRange::At(2), AxisRange::At(1)])?;
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.size(), 1);
    assert_eq!(scalar.read_value(&[])?, 7);
    assert_eq!(&*scalar.buffer()?.to_slice()?, &[7]);
    assert!(matches!(scalar.read_value(&[0]), Err(Error::Bounds(_))));
    Ok(())
}

#[test]
fn test_slice_and_write() -> Result<(), Error> {
    for (start, end) in [(0, 0), (1, 2), (128, 256)] {
        let source = ArrayBuf::new(vec![0; 512], shape![512])?;
        let view = source.as_ref().slice(range![(start..end).into()])?;
        assert_eq!(view.shape(), &[end - start]);
        assert_eq!(&*view.buffer()?.to_slice()?, &vec![0; end - start]);
        let mut copy = view.copy()?;
        copy.write_value(9)?;
        assert_eq!(&*copy.buffer()?.to_slice()?, &vec![9; end - start]);
        assert_eq!(&*source.buffer()?.to_slice()?, &vec![0; 512]);
    }

    let source = ArrayBuf::new(vec![0; 512], shape![512])?;
    let view = source.as_ref().slice(range![AxisRange::At(3)])?;
    let mut scalar = view.copy()?;
    scalar.write_value(7)?;
    assert_eq!(scalar.read_value(&[])?, 7);
    assert_eq!(source.read_value(&[3])?, 0);

    let matrix = ArrayBuf::new((0..12).collect::<Vec<i32>>(), shape![4, 3])?;
    let view = matrix
        .as_ref()
        .slice(range![AxisRange::At(2), AxisRange::At(1)])?;
    assert_eq!(view.read_value(&[])?, 7);
    let mut scalar = view.copy()?;
    scalar.write_value(99)?;
    assert_eq!(scalar.read_value(&[])?, 99);
    scalar.write_value_at(&[], 88)?;
    assert_eq!(scalar.read_value(&[])?, 88);
    assert!(matches!(
        scalar.write_value_at(&[0], 0),
        Err(Error::Bounds(_))
    ));
    assert_eq!(
        &*matrix.buffer()?.to_slice()?,
        &(0..12).collect::<Vec<i32>>()
    );

    Ok(())
}

#[test]
fn test_slice_3d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 24, shape![4, 3, 2])?;

    let expected = ArrayBuf::new(vec![8, 9, 10, 11], shape![2, 2])?;

    let actual = input.slice(range![1.into(), (1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_2d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 6, shape![2, 3])?;

    let expected = ArrayBuf::new(
        vec![
            0, 3, //
            1, 4, //
            2, 5, //
        ],
        shape![3, 2],
    )?;

    let actual = input.transpose(None)?;
    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_3d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 24, shape![2, 3, 4])?;

    let expected = ArrayBuf::new(
        vec![
            0, 4, 8, //
            12, 16, 20, //
            1, 5, 9, //
            13, 17, 21, //
            //
            2, 6, 10, //
            14, 18, 22, //
            3, 7, 11, //
            15, 19, 23, //
        ],
        shape![4, 2, 3],
    )?;

    let actual = input.transpose(Some(axes![2, 0, 1]))?;
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_offsets_to_coords() -> Result<(), Error> {
    let coords = ArrayBuf::new(stackvec![0, 1], shape![1, 2])?;
    let strides = ArrayBuf::new(vec![5, 1], shape![2])?.broadcast(coords.shape().into())?;
    let offsets = coords.mul(strides).map(ArrayAccess::from)?;
    let offsets = offsets.sum(axes![1], false)?;
    assert_eq!(offsets.buffer()?.to_slice()?.into_vec(), vec![1]);
    Ok(())
}

#[test]
fn invalid_transform_boundaries_and_stepped_cardinality() -> Result<(), Error> {
    let input = ArrayBuf::new(vec![0u32, 1, 2, 3, 4], shape![5])?;
    let stepped = input.as_ref().slice(range![AxisRange::In(0, 5, 2)])?;
    assert_eq!(stepped.shape(), &[3]);
    assert_eq!(&*stepped.buffer()?.to_slice()?, &[0, 2, 4]);
    for range in [
        range![AxisRange::In(0, 5, 0)],
        range![AxisRange::In(3, 2, 1)],
        range![0.into(), 0.into()],
    ] {
        assert!(input.as_ref().slice(range).is_err());
    }
    let empty = input.as_ref().slice(range![AxisRange::In(5, 5, 1)])?;
    assert_eq!(empty.buffer()?.len(), 0);
    assert!(input.as_ref().squeeze(axes![0]).is_err());
    assert!(input.as_ref().mt().is_err());
    let singleton = ArrayBuf::new(vec![1u32], shape![1, 1])?;
    assert!(singleton.squeeze(axes![0, 0]).is_err());
    assert!(input.reshape(shape![usize::MAX, 2]).is_err());
    assert!(ArrayBuf::<u32, Vec<u32>>::new(vec![1], shape![usize::MAX, 2]).is_err());

    let matrix = ArrayBuf::new(vec![0u32, 1, 2, 3, 4, 5], shape![2, 3])?;
    for shape in [shape![1, 3], shape![3], shape![2, 1], shape![1, 1, 3]] {
        assert!(matches!(
            matrix.as_ref().broadcast(shape),
            Err(Error::Bounds(_))
        ));
    }
    for permutation in [axes![0], axes![0, 0], axes![0, 2]] {
        assert!(matches!(
            matrix.as_ref().transpose(permutation),
            Err(Error::Bounds(_))
        ));
    }
    assert!(matches!(matrix.as_ref().flip(2), Err(Error::Bounds(_))));
    let empty = ArrayBuf::new(Vec::<u32>::new(), shape![usize::MAX, 0, 2])?;
    assert!(matches!(
        empty.transpose(axes![1, 2, 0]),
        Err(Error::Bounds(_))
    ));
    Ok(())
}

#[cfg(feature = "complex")]
#[test]
fn fft_shift_centers_even_and_odd_dimensions() -> Result<(), Error> {
    for (values, expected) in [
        (vec![0u32, 1, 2, 3], vec![2, 3, 0, 1]),
        (vec![0u32, 1, 2, 3, 4], vec![3, 4, 0, 1, 2]),
        (vec![7u32], vec![7]),
    ] {
        let len = values.len();
        let source = ArrayBuf::new(values, shape![len])?;
        let shifted = fft::shift(ArrayAccess::from(source), None)?;
        assert_eq!(&*shifted.buffer()?.to_slice()?, &expected);
    }
    Ok(())
}
