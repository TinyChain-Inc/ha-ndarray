use ha_ndarray::*;

#[test]
fn slice_descriptor_validates_before_backend_construction() -> Result<(), Error> {
    // Direct backend construction consumes the same validated descriptor as Array.
    let spec = ops::SliceSpec::new(&[2, 3], range![AxisRange::At(1)])?;
    assert_eq!(spec.shape().as_slice(), &[3]);
    assert_eq!(spec.range().len(), 2);
    assert_eq!(spec.source_size(), 6);
    assert!(host::ops::Slice::<_, i32>::new(AccessBuf::from(vec![0; 5]), spec.clone()).is_err());
    let op = host::ops::Slice::<_, i32>::new(AccessBuf::from((0..6).collect::<Vec<_>>()), spec)?;
    assert_eq!(ops::ReadValue::<host::Host, i32>::read_value(&op, 0)?, 3);
    assert!(ops::SliceSpec::new(&[usize::MAX, 2], range![]).is_err());

    let view = ops::ViewSpec::transpose(shape![2, 3], axes![1, 0])?;
    assert!(host::ops::View::<_, i32>::new(AccessBuf::from(vec![0; 5]), view.clone()).is_err());
    assert_eq!(
        (0..6).map(|i| view.source_offset(i)).collect::<Vec<_>>(),
        [0, 3, 1, 4, 2, 5]
    );
    let flip = ops::FlipSpec::new(shape![2, 3], 1)?;
    assert!(host::ops::Flip::<_, i32>::new(AccessBuf::from(vec![0; 5]), flip.clone()).is_err());
    assert_eq!(
        (0..6).map(|i| flip.source_offset(i)).collect::<Vec<_>>(),
        [2, 1, 0, 5, 4, 3]
    );
    assert!(ops::ViewSpec::broadcast(shape![2, 3], shape![1, 3]).is_err());
    assert!(ops::ViewSpec::transpose(shape![2, 3], axes![1, 1]).is_err());
    assert!(ops::FlipSpec::new(shape![usize::MAX, 2], 0).is_err());
    if let Some(wide) = (u32::MAX as usize).checked_add(2) {
        let spec = ops::ViewSpec::transpose(shape![2, wide], axes![1, 0])?;
        assert_eq!(spec.source_offset(1), wide);
        let spec = ops::FlipSpec::new(shape![2, wide], 0)?;
        assert_eq!(spec.source_offset(0), wide);
    }
    Ok(())
}

#[test]
fn empty_stack_reductions_return_errors() {
    use ha_ndarray::host::Stack;
    use ha_ndarray::ops::ReduceAll;

    let empty = || AccessBuf::from(Vec::<i32>::new());
    assert!(Stack.max(empty()).is_err());
    assert!(Stack.min(empty()).is_err());
    assert!(Stack.product(empty()).is_err());
    assert!(Stack.sum(empty()).is_err());
}

#[test]
fn empty_host_matrix_kernels() -> Result<(), Error> {
    use ha_ndarray::host::ops::{MatDiag, MatMul};
    use ha_ndarray::host::{Heap, Stack};
    use ha_ndarray::ops::Enqueue;
    let empty = ArrayBuf::new(Vec::<i32>::new(), shape![0, 0])?;
    let op = MatDiag::new(empty.into_access(), 1, 0);
    assert!(Enqueue::<Heap, i32>::enqueue(&op)?.is_empty());
    assert!(Enqueue::<Stack, i32>::enqueue(&op)?.is_empty());
    let l = ArrayBuf::new(Vec::<i32>::new(), shape![2, 0])?;
    let r = ArrayBuf::new(Vec::<i32>::new(), shape![0, 3])?;
    let op = MatMul::new(l.into_access(), r.into_access(), [1, 2, 0, 3]);
    assert_eq!(&*Enqueue::<Heap, i32>::enqueue(&op)?, &[0; 6]);
    assert_eq!(&*Enqueue::<Stack, i32>::enqueue(&op)?, &[0; 6]);

    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_preflight_rejects_before_program_construction() -> Result<(), Error> {
    let descriptor = ha_ndarray::ops::SliceSpec::new(&[2, 2], range![])?;
    assert!(matches!(
        ha_ndarray::opencl::ops::Slice::<_, i32>::new(
            ha_ndarray::AccessBuf::from(vec![0; 3]),
            descriptor
        ),
        Err(ha_ndarray::Error::Bounds(_))
    ));
    let view = ops::ViewSpec::transpose(shape![2, 2], axes![1, 0])?;
    assert!(matches!(
        opencl::ops::View::<_, i32>::new(AccessBuf::from(vec![0; 3]), view),
        Err(Error::Bounds(_))
    ));
    let flip = ops::FlipSpec::new(shape![2, 2], 0)?;
    assert!(matches!(
        opencl::ops::Flip::<_, i32>::new(AccessBuf::from(vec![0; 3]), flip),
        Err(Error::Bounds(_))
    ));
    for dims in [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]] {
        let left = AccessBuf::from(Vec::<i32>::new());
        let right = AccessBuf::from(Vec::<i32>::new());
        assert!(matches!(
            ha_ndarray::opencl::ops::MatMul::<_, _, i32>::new(left, right, dims),
            Err(Error::Unsupported(_))
        ));
    }
    Ok(())
}

#[test]
fn singleton_reduction_preserves_source_errors_and_point_reads() -> Result<(), Error> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Source<'a>(&'a AtomicUsize);

    impl Access<i32> for Source<'_> {
        fn read(&self) -> Result<BufferConverter<'_, i32>, Error> {
            Err(Error::Unsupported("source read failed".into()))
        }

        fn read_value(&self, offset: usize) -> Result<i32, Error> {
            assert_eq!(offset, 0);
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(7)
        }

        fn size(&self) -> usize {
            1
        }
    }

    let reads = AtomicUsize::new(0);
    let op = host::ops::Reduce::sum(Source(&reads), 1);
    assert_eq!(ops::ReadValue::<host::Host, i32>::read_value(&op, 0)?, 7);
    assert!(ops::ReadValue::<host::Host, i32>::read_value(&op, usize::MAX).is_err());
    assert_eq!(reads.load(Ordering::Relaxed), 1);
    assert!(
        matches!(ops::Enqueue::<host::Host, i32>::enqueue(&op), Err(Error::Unsupported(message)) if message == "source read failed")
    );

    #[cfg(feature = "opencl")]
    {
        let op = opencl::ops::Reduce::sum(Source(&reads), 1)?;
        assert_eq!(
            ops::ReadValue::<opencl::OpenCL, i32>::read_value(&op, 0)?,
            7
        );
        assert!(ops::ReadValue::<opencl::OpenCL, i32>::read_value(&op, usize::MAX).is_err());
        assert_eq!(reads.load(Ordering::Relaxed), 2);
        assert!(
            matches!(ops::Enqueue::<opencl::OpenCL, i32>::enqueue(&op), Err(Error::Unsupported(message)) if message == "source read failed")
        );
    }
    Ok(())
}
