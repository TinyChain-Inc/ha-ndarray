use ha_ndarray::{
    shape, stackvec, Access, AccessBuf, Array, ArrayAccess, ArrayBuf, Buffer, BufferConverter,
    BufferInstance, Error, Float, NDArray, NDArrayMath, NDArrayMathScalar, NDArrayRead,
    NDArrayUnary, NDArrayWrite,
};

// Numerical accuracy belongs to the operation fixtures; this module checks ownership.
fn logit<A, T>(p: Array<T, A>) -> Result<Array<T, impl Access<T>>, Error>
where
    A: Access<T>,
    Array<T, A>: Clone,
    T: Float + std::ops::Neg<Output = T>,
{
    p.clone().div(p.add_scalar(-T::ONE)?)?.ln()
}

#[test]
fn test_as_ref() -> Result<(), Error> {
    let a = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.].into(), shape![1])?;
    let borrowed: Array<f32, &AccessBuf<Buffer<f32>>> = a.as_ref();
    logit(borrowed)?;
    let operation = a.as_ref().add_scalar(1.)?;
    logit(operation.as_ref())?;

    let values = vec![1., 2., 3.];
    let slice = ArrayBuf::new(values.as_slice(), shape![3])?;
    assert_eq!(slice.buffer()?.to_slice()?.as_ptr(), values.as_ptr());
    assert_eq!(slice.as_ref().add_scalar(2.)?.read_value(&[1])?, 4.);

    let shared = ArrayAccess::from(slice.as_ref());
    assert_eq!(shared.as_ref().read_value(&[2])?, 3.);
    assert_eq!(shared.clone().into_read()?.read_value(&[0])?, 1.);
    assert_eq!(shared.read_value(&[0])?, 1.);

    let mut mutable = ArrayBuf::new(vec![0, 1, 2], shape![3])?;
    {
        let view = &mut mutable;
        let read = view.buffer()?.to_slice()?;
        assert_eq!(&*read, &[0, 1, 2]);
        drop(read);
        view.write_value_at(&[1], 7)?;
        assert_eq!(view.as_ref().read_value(&[1])?, 7);
        let shared = ArrayAccess::from(view.as_ref());
        assert_eq!(shared.read_value(&[1])?, 7);
    }
    assert_eq!(mutable.read_value(&[1])?, 7);
    let mut independent = mutable.copy()?;
    independent.write_value(10)?;
    assert_eq!(&*independent.buffer()?.to_slice()?, &[10; 3]);
    assert_eq!(&*mutable.buffer()?.to_slice()?, &[0, 7, 2]);
    independent.write(&ArrayBuf::new(vec![3, 4, 5], shape![3])?)?;
    assert_eq!(&*independent.buffer()?.to_slice()?, &[3, 4, 5]);
    assert_eq!(&*mutable.buffer()?.to_slice()?, &[0, 7, 2]);
    independent.write_value_at(&[1], 42)?;
    assert_eq!(&*independent.buffer()?.to_slice()?, &[3, 42, 5]);
    assert_eq!(&*mutable.buffer()?.to_slice()?, &[0, 7, 2]);

    let retained = {
        let local = ArrayAccess::from(ArrayBuf::new(vec![11, 12], shape![2])?);
        local.clone()
    };
    assert_eq!(&*retained.buffer()?.to_slice()?, &[11, 12]);

    let mut values = [4, 5, 6];
    let mut slice = ArrayBuf::new(values.as_mut_slice(), shape![3])?;
    slice.write_value_at(&[2], 9)?;
    drop(slice);
    assert_eq!(values, [4, 5, 9]);
    Ok(())
}

#[test]
fn consuming_reads_preserve_owned_buffers() -> Result<(), Error> {
    let values: Vec<i32> = (0..64).collect();
    let pointer = values.as_ptr();
    let buffer = BufferInstance::into_buffer(values)?;
    let values = BufferConverter::from(buffer).to_slice()?.into_vec();
    assert_eq!(values.as_ptr(), pointer);

    let array = ArrayBuf::new(values, shape![8, 8])?;
    let copied = array.as_ref().into_read()?;
    assert_eq!(copied.shape(), &[8, 8]);
    assert_ne!(copied.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(array.read_value(&[0, 0])?, 0);

    let owned = array.into_read()?;
    assert_eq!(owned.shape(), &[8, 8]);
    assert_eq!(owned.buffer()?.to_slice()?.as_ptr(), pointer);
    let owned = owned.into_read()?;
    assert_eq!(owned.buffer()?.to_slice()?.as_ptr(), pointer);
    let buffer = owned.into_access().into_inner();
    let values = BufferConverter::from(buffer).to_slice()?.into_vec();
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values, (0..64).collect::<Vec<_>>());

    let host = ha_ndarray::host::ArrayBuf::new(values.into(), shape![64])?;
    let host = host.into_read()?.into_access().into_inner().into_vec();
    assert_eq!(host.as_ptr(), pointer);

    let stack = stackvec![1, 2, 3];
    let buffer = BufferInstance::into_buffer(stack)?;
    assert!(matches!(
        buffer,
        Buffer::Host(ha_ndarray::host::Buffer::Stack(_))
    ));
    assert_eq!(&*BufferConverter::from(buffer).to_slice()?, &[1, 2, 3]);

    let shared = ArrayAccess::from(ArrayBuf::new(host, shape![64])?);
    let copied = shared.clone().into_read()?;
    assert_ne!(copied.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(shared.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(shared.read_value(&[63])?, 63);
    Ok(())
}

#[cfg(feature = "complex")]
#[test]
fn consuming_fft_reuses_owned_workspace() -> Result<(), Error> {
    use ha_ndarray::complex::Complex64;
    use ha_ndarray::NDArrayFourier;

    let values: Vec<_> = (0..64)
        .map(|i| Complex64::new((i % 7) as f64, (i % 11) as f64))
        .collect();
    let original = values.clone();
    let pointer = values.as_ptr();
    let array = ha_ndarray::host::ArrayBuf::new(values.into(), shape![2, 32])?;
    let borrowed = array.as_ref().fft()?.into_read()?;
    assert_ne!(borrowed.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(&*array.buffer()?.to_slice()?, &original);

    let consumed = array.fft()?.into_read()?;
    assert_eq!(consumed.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(
        &*consumed.buffer()?.to_slice()?,
        &*borrowed.buffer()?.to_slice()?
    );

    let values = original.clone();
    let pointer = values.as_ptr();
    let dispatched = ArrayBuf::new(values, shape![2, 32])?.fft()?.into_read()?;
    assert_eq!(dispatched.buffer()?.to_slice()?.as_ptr(), pointer);
    assert_eq!(
        &*dispatched.buffer()?.to_slice()?,
        &*borrowed.buffer()?.to_slice()?
    );

    let shared = ArrayAccess::from(ArrayBuf::new(original.clone(), shape![2, 32])?);
    let shared_pointer = shared.buffer()?.to_slice()?.as_ptr();
    let transformed = shared.clone().fft()?.into_read()?;
    assert_ne!(transformed.buffer()?.to_slice()?.as_ptr(), shared_pointer);
    assert_eq!(&*shared.buffer()?.to_slice()?, &original);
    assert_eq!(
        &*transformed.buffer()?.to_slice()?,
        &*borrowed.buffer()?.to_slice()?
    );

    // Consuming a host array must retain its platform even above the general
    // platform's OpenCL threshold, where Fourier operations are unsupported.
    let host =
        ha_ndarray::host::ArrayBuf::new(vec![Complex64::new(1., 0.); 1024].into(), shape![1024])?;
    let owned = ArrayAccess::from(host).into_read()?;
    let transformed = owned.fft()?.into_read()?;
    let values = transformed.buffer()?.to_slice()?;
    assert_eq!(values[0], Complex64::new(1024., 0.));
    assert!(values[1..]
        .iter()
        .all(|value| *value == Complex64::new(0., 0.)));
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn consuming_device_buffers_preserves_the_allocation() -> Result<(), Error> {
    use ha_ndarray::opencl::{ArrayBuf as DeviceArray, OpenCL};

    let values: Vec<i32> = (0..128).collect();
    let buffer = OpenCL::copy_into_buffer(&values)?;
    let pointer = buffer.as_core().as_ptr();
    let Buffer::CL(buffer) = BufferInstance::into_buffer(buffer)? else {
        panic!("owned OpenCL buffer changed backend");
    };
    assert_eq!(buffer.as_core().as_ptr(), pointer);

    let array = DeviceArray::new(buffer, shape![8, 16])?;
    let copied = array.as_ref().into_read()?;
    assert_ne!(copied.access().inner().as_core().as_ptr(), pointer);
    assert_eq!(&*array.buffer()?.to_slice()?, &values);

    let consumed = array.into_read()?;
    assert_eq!(consumed.shape(), &[8, 16]);
    let buffer = consumed.into_access().into_inner();
    assert_eq!(buffer.as_core().as_ptr(), pointer);
    assert_eq!(BufferConverter::from(buffer).to_slice()?.into_vec(), values);
    Ok(())
}
