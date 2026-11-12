use ha_ndarray::*;

#[test]
fn test_diag() -> Result<(), Error> {
    let x = ArrayOp::range(0, 9, shape![3, 3])?;
    let diag = x.diag()?;
    assert_eq!(&*diag.buffer()?.to_slice()?, &[0, 4, 8]);
    for dims in [shape![0, 0], shape![3, 0, 0], shape![0, 3, 3]] {
        let x = ArrayBuf::new(Vec::<i32>::new(), dims.clone())?;
        let diag = x.diag()?;
        assert_eq!(diag.shape(), &dims[..dims.len() - 1]);
        assert!(diag.buffer()?.is_empty());
    }
    let x = ArrayBuf::new(Vec::<i32>::new(), shape![0, usize::MAX, 2, 0, 0])?;
    assert!(matches!(x.diag(), Err(Error::Bounds(_))));

    Ok(())
}

#[test]
fn test_matmul_12x20() -> Result<(), Error> {
    let l = ArrayBuf::new((0..12).collect::<Vec<_>>(), shape![3, 4])?;
    let r = ArrayBuf::new((0..20).collect::<Vec<_>>(), shape![4, 5])?;

    let actual = l.matmul(r)?;

    let expected = ArrayBuf::new(
        vec![
            70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462,
        ],
        shape![3, 5],
    )?;

    assert_eq!(actual.shape(), expected.shape());

    let eq = actual.eq(expected)?;
    assert!(eq.all()?);

    let empty_products: [(Shape, Shape, Shape); 3] = [
        (shape![0, 3], shape![3, 2], shape![0, 2]),
        (shape![2, 3], shape![3, 0], shape![2, 0]),
        (shape![0, 2, 3], shape![0, 3, 4], shape![0, 2, 4]),
    ];
    for (left, right, expected) in empty_products {
        let l = ArrayBuf::new(vec![1i32; left.iter().product()], left)?;
        let r = ArrayBuf::new(vec![1i32; right.iter().product()], right)?;
        let product = l.matmul(r)?;
        assert_eq!(product.shape(), &expected[..]);
        assert!(product.buffer()?.is_empty());
    }
    let l = ArrayBuf::new(Vec::<i32>::new(), shape![2, 0])?;
    let r = ArrayBuf::new(Vec::<i32>::new(), shape![0, 3])?;
    let product = l.matmul(r)?;
    assert_eq!(product.shape(), &[2, 3]);
    assert_eq!(&*product.buffer()?.to_slice()?, &[0; 6]);

    Ok(())
}

#[test]
fn test_dot_product() -> Result<(), Error> {
    let buffer = (0..16).collect::<Vec<_>>();
    let vectors = ArrayBuf::<i32, Buffer<i32>>::new(buffer.to_vec().into(), shape![4, 1, 4])?;

    let l: Array<i32, &AccessBuf<Buffer<i32>>> = vectors.as_ref();
    let r = l.clone().mt()?;

    let actual = l.matmul(r)?.buffer()?.to_slice()?.to_vec();

    let mut expected = Vec::with_capacity(4 * 4);
    for vector in buffer.chunks(4) {
        expected.push(vector.iter().zip(vector.iter()).map(|(l, r)| l * r).sum());
    }

    assert_eq!(actual, expected);

    let dims = shape![0, usize::MAX, 2, 0, 0];
    let l = ArrayBuf::new(Vec::<i32>::new(), dims.clone())?;
    let r = ArrayBuf::new(Vec::<i32>::new(), dims)?;
    assert!(matches!(l.matmul(r), Err(Error::Bounds(_))));
    let l = ArrayBuf::new(Vec::<i32>::new(), shape![usize::MAX, 0])?;
    let r = ArrayBuf::new(Vec::<i32>::new(), shape![0, 2])?;
    assert!(matches!(l.matmul(r), Err(Error::Bounds(_))));

    Ok(())
}
