use std::marker::PhantomData;

use rustfft::{FftDirection, FftNum, FftPlanner};

use crate::host::{Buffer, Host};
use crate::ops::{Enqueue, Op, ReadValue};
use crate::{Access, Complex, Error, Number};

pub struct FFT<A, T> {
    access: A,
    dim: usize,
    dir: FftDirection,
    dtype: PhantomData<T>,
}

impl<A: Access<T>, T: Number> FFT<A, T> {
    fn new(access: A, dim: usize, dir: FftDirection) -> Result<Self, Error> {
        let size = access.size();

        if dim != 0 && size != 0 && size % dim == 0 {
            Ok(Self {
                access,
                dim,
                dir,
                dtype: PhantomData,
            })
        } else {
            Err(Error::bounds(format!(
                "dimension {dim} is not a factor of size {size}"
            )))
        }
    }

    #[allow(clippy::self_named_constructors)]
    // This mirrors the public NDArray API (`fft`, `ifft`) and reads clearly at callsites.
    pub fn fft(access: A, dim: usize) -> Result<Self, Error> {
        Self::new(access, dim, FftDirection::Forward)
    }

    #[allow(clippy::self_named_constructors)]
    // This mirrors the public NDArray API (`fft`, `ifft`) and reads clearly at callsites.
    pub fn ifft(access: A, dim: usize) -> Result<Self, Error> {
        Self::new(access, dim, FftDirection::Inverse)
    }
}

impl<A: Access<T>, T: Number> Op for FFT<A, T> {
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, T> Enqueue<Host, num_complex::Complex<T>> for FFT<A, num_complex::Complex<T>>
where
    A: Access<num_complex::Complex<T>>,
    T: FftNum,
    num_complex::Complex<T>: Complex,
{
    type Buffer = Buffer<num_complex::Complex<T>>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut buffer = self.access.read()?.to_slice()?.into_buffer();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(self.dim, self.dir);
        for batch in buffer.as_mut().chunks_exact_mut(self.dim) {
            fft.process(batch);
        }

        Ok(buffer)
    }
}

impl<A: Access<T>, T: Number> ReadValue<Host, T> for FFT<A, T> {
    fn read_value(&self, _offset: usize) -> Result<T, Error> {
        Err(Error::unsupported(
            "read an individual value from a Fourier transform".into(),
        ))
    }
}
