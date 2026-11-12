use rustfft::FftNum;

use crate::access::AccessOp;
use crate::host::ops::complex as host;
use crate::host::Host;
use crate::ops::{Enqueue, Op, ReadOp, ReadValue};
use crate::{Access, Buffer, Complex, Error, Number, Platform, PlatformInstance};

pub trait ElementwiseUnaryComplex<A, T>: PlatformInstance
where
    A: Access<T>,
    T: Complex,
{
    type Real: ReadOp<Self, T::Real>;
    type Complex: ReadOp<Self, T>;

    fn angle(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error>;

    fn conj(self, access: A) -> Result<AccessOp<Self::Complex, Self>, Error>;

    fn re(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error>;

    fn im(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error>;
}

pub trait Fourier<A, T>: PlatformInstance
where
    A: Access<T>,
    T: Complex,
{
    type Op: ReadOp<Self, T>;

    fn fft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ifft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error>;
}

// TODO: add OpenCL support for FFT
pub enum Fft<A, T> {
    Host(host::Fft<A, T>),
}

impl<A: Access<T>, T: Number> Op for Fft<A, T> {
    fn size(&self) -> usize {
        match self {
            Self::Host(fft) => fft.size(),
        }
    }
}

impl<A, T> Enqueue<Platform, num_complex::Complex<T>> for Fft<A, num_complex::Complex<T>>
where
    A: Access<num_complex::Complex<T>>,
    T: FftNum,
    num_complex::Complex<T>: Complex,
{
    type Buffer = Buffer<num_complex::Complex<T>>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        match self {
            Self::Host(op) => Enqueue::<Host, _>::enqueue(op).map(Buffer::Host),
        }
    }

    fn into_enqueue(self) -> Result<Self::Buffer, Error> {
        match self {
            Self::Host(op) => Enqueue::<Host, _>::into_enqueue(op).map(Buffer::Host),
        }
    }
}

impl<A: Access<T>, T: Number> ReadValue<Platform, T> for Fft<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        match self {
            Self::Host(fft) => ReadValue::read_value(fft, offset),
        }
    }
}

impl<A, T> From<host::Fft<A, T>> for Fft<A, T> {
    fn from(op: host::Fft<A, T>) -> Self {
        Self::Host(op)
    }
}
