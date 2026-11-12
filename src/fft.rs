use std::iter;

use num_complex::Complex;

use crate::{
    Access, Array, ArrayAccess, Axes, AxisRange, Error, NDArray, NDArrayFourier, NDArrayTransform,
    Number, Range,
};

/// Fast Fourier Transform, an alias of [`NDArrayFourier::fft`]
pub fn fft<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    data.fft()
}

/// Inverse Fast Fourier Transform, an alias of [`NDArrayFourier::ifft`]
pub fn ifft<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    data.ifft()
}

/// Two-dimensional Fast Fourier Transform
pub fn fft2<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    if data.ndim() >= 2 {
        let mut permutation = (0..data.ndim()).collect::<Axes>();
        permutation.swap(data.ndim() - 1, data.ndim() - 2);

        data.fft()?
            .transpose(permutation.clone())?
            .fft()?
            .transpose(permutation)
    } else {
        Err(Error::bounds(format!(
            "{data:?} has less than two dimensions",
        )))
    }
}

/// Inverse two-dimensional Fast Fourier Transform
pub fn ifft2<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    if data.ndim() >= 2 {
        let mut permutation = (0..data.ndim()).collect::<Axes>();
        permutation.swap(data.ndim() - 1, data.ndim() - 2);

        data.transpose(permutation.clone())?
            .ifft()?
            .transpose(permutation)?
            .ifft()
    } else {
        Err(Error::bounds(format!(
            "{data:?} has less than two dimensions",
        )))
    }
}

/// Move the zero-frequency component to the center of the selected axis (FFT shift).
pub fn shift<'a, T, X>(
    data: ArrayAccess<'a, T>,
    axis: X,
) -> Result<Array<T, impl Access<T> + 'a>, Error>
where
    T: Number,
    X: Into<Option<usize>>,
{
    let axis = axis
        .into()
        .or_else(|| data.ndim().checked_sub(1))
        .ok_or_else(|| Error::bounds("FFT shift requires an axis".into()))?;

    if axis < data.ndim() {
        let dim = data.shape()[axis];
        let pivot = dim.div_ceil(2);

        let range = slice_range(data.shape(), axis, 0..pivot);
        let left = data.clone().slice(range)?;

        let range = slice_range(data.shape(), axis, pivot..dim);
        let right = data.clone().slice(range)?;

        Array::transpose_concat(vec![right, left], axis)
    } else {
        Err(Error::bounds(format!("{data:?} has no axis {axis}")))
    }
}

#[inline]
fn slice_range(shape: &[usize], axis: usize, range: std::ops::Range<usize>) -> Range {
    shape[..axis]
        .iter()
        .copied()
        .map(|dim| 0..dim)
        .map(AxisRange::from)
        .chain(iter::once(range.into()))
        .collect()
}
