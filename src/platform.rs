use std::fmt;

use crate::access::{Access, AccessOp};
use crate::buffer::{Buffer, BufferConverter, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, Axes, Error, Float, Number, Range, Real, Shape};

/// A ha-ndarray platform
pub trait PlatformInstance: PartialEq + Eq + Clone + Copy + Send + Sync + fmt::Debug {
    /// Automatically select a sub-platform for the workload size within the
    /// platform type, enabled backends, and user-configured device constraints.
    /// Use the selection consistently for construction and returned array metadata.
    /// Selection does not authorize fallback after capability or execution failure.
    fn select(size_hint: usize) -> Self;
}

/// Constructor for a new buffer filled with a single value
pub trait Constant<T: Number>: PlatformInstance {
    /// The type of buffer use by this platform
    type Buffer: BufferInstance<T>;

    /// Construct a new buffer filled with a single value.
    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error>;
}

/// Converter to construct an owned, platform-specific buffer
pub trait Convert<T: Number>: PlatformInstance {
    /// The type of buffer use by this platform
    type Buffer: BufferInstance<T>;

    fn convert(&self, buffer: BufferConverter<T>) -> Result<Self::Buffer, Error>;
}

/// The global platform, responsible for delegating to specific hardware platforms
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Platform {
    #[cfg(feature = "opencl")]
    CL(opencl::OpenCL),
    Host(host::Host),
}

#[cfg(feature = "opencl")]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        if size_hint < opencl::GPU_MIN_SIZE {
            Self::Host(host::Host::select(size_hint))
        } else {
            Self::CL(opencl::OpenCL)
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        Self::Host(host::Host::select(size_hint))
    }
}

#[cfg(feature = "opencl")]
impl From<opencl::OpenCL> for Platform {
    fn from(opencl: opencl::OpenCL) -> Self {
        Self::CL(opencl)
    }
}

impl From<host::Host> for Platform {
    fn from(host: host::Host) -> Self {
        Self::Host(host)
    }
}

impl<T: Number> Convert<T> for Platform {
    type Buffer = Buffer<T>;

    fn convert<'a>(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.convert(buffer).map(Buffer::CL),
            Self::Host(host) => host.convert(buffer).map(Buffer::Host),
        }
    }
}

impl<T: Number> Constant<T> for Platform {
    type Buffer = Buffer<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.constant(value, size).map(Buffer::CL),
            Self::Host(host) => host.constant(value, size).map(Buffer::Host),
        }
    }
}

impl<A, T> ConstructConcat<A, T> for Platform
where
    A: Access<T>,
    T: Number,
{
    type Op = Concat<A, T>;

    fn concat(self, data: Vec<A>) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.concat(data).map(AccessOp::wrap),
            Self::Host(host) => host.concat(data).map(AccessOp::wrap),
        }
    }
}

impl<T: Number + PartialOrd> ConstructRange<T> for Platform {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.range(start, stop, size).map(AccessOp::wrap),
            Self::Host(host) => host.range(start, stop, size).map(AccessOp::wrap),
        }
    }
}

impl<A, T> ElementwiseAbs<A, T> for Platform
where
    A: Access<T>,
    T: Number,
{
    type Op = Unary<A, T, T::Abs>;

    fn abs(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.abs(access).map(AccessOp::wrap),
            Self::Host(host) => host.abs(access).map(AccessOp::wrap),
        }
    }
}

impl<L, R, T> ElementwiseBoolean<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.and(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.and(left, right).map(AccessOp::wrap),
        }
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.or(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.or(left, right).map(AccessOp::wrap),
        }
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.xor(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.xor(left, right).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> ElementwiseBooleanScalar<A, T> for Platform {
    type Op = Scalar<A, T, u8>;

    fn and_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.and_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.and_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn or_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.or_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.or_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn xor_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.xor_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.xor_scalar(left, right).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> ElementwiseCast<A, IT, OT> for Platform {
    type Op = Cast<A, IT, OT>;

    fn cast(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.cast(access).map(AccessOp::wrap),
            Self::Host(host) => host.cast(access).map(AccessOp::wrap),
        }
    }
}

impl<L, R, T> ElementwiseCompare<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.eq(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.eq(left, right).map(AccessOp::wrap),
        }
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.ge(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ge(left, right).map(AccessOp::wrap),
        }
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.gt(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.gt(left, right).map(AccessOp::wrap),
        }
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.le(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.le(left, right).map(AccessOp::wrap),
        }
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.lt(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.lt(left, right).map(AccessOp::wrap),
        }
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.ne(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ne(left, right).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> ElementwiseCompareScalar<A, T> for Platform {
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.eq_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.eq_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.ge_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ge_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.gt_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.gt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.le_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.le_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.lt_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.lt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.ne_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ne_scalar(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "complex")]
impl<A, T> complex::ElementwiseUnaryComplex<A, T> for Platform
where
    A: Access<T>,
    T: crate::Complex,
{
    type Real = Unary<A, T, T::Real>;
    type Complex = Unary<A, T, T>;

    fn angle(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.angle(access).map(AccessOp::wrap),
            Self::Host(host) => host.angle(access).map(AccessOp::wrap),
        }
    }

    fn conj(self, access: A) -> Result<AccessOp<Self::Complex, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.conj(access).map(AccessOp::wrap),
            Self::Host(host) => host.conj(access).map(AccessOp::wrap),
        }
    }

    fn re(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.re(access).map(AccessOp::wrap),
            Self::Host(host) => host.re(access).map(AccessOp::wrap),
        }
    }

    fn im(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.im(access).map(AccessOp::wrap),
            Self::Host(host) => host.im(access).map(AccessOp::wrap),
        }
    }
}

impl<L, R, T> ElementwiseDual<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.add(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.add(left, right).map(AccessOp::wrap),
        }
    }

    fn div(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.div(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.div(left, right).map(AccessOp::wrap),
        }
    }

    fn log(self, arg: L, base: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.log(arg, base).map(AccessOp::wrap),
            Self::Host(host) => host.log(arg, base).map(AccessOp::wrap),
        }
    }

    fn mul(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.mul(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.mul(left, right).map(AccessOp::wrap),
        }
    }

    fn pow(self, arg: L, exp: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.pow(arg, exp).map(AccessOp::wrap),
            Self::Host(host) => host.pow(arg, exp).map(AccessOp::wrap),
        }
    }

    fn rem(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.rem(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.rem(left, right).map(AccessOp::wrap),
        }
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.sub(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.sub(left, right).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> ElementwiseScalar<A, T> for Platform {
    type Op = Scalar<A, T, T>;

    fn add_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.add_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.add_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn div_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.div_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.div_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn log_scalar(self, arg: A, base: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.log_scalar(arg, base).map(AccessOp::wrap),
            Self::Host(host) => host.log_scalar(arg, base).map(AccessOp::wrap),
        }
    }

    fn mul_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.mul_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.mul_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn pow_scalar(self, arg: A, exp: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.pow_scalar(arg, exp).map(AccessOp::wrap),
            Self::Host(host) => host.pow_scalar(arg, exp).map(AccessOp::wrap),
        }
    }

    fn rem_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.rem_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.rem_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn sub_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.sub_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.sub_scalar(left, right).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Float> ElementwiseNumeric<A, T> for Platform {
    type Op = Unary<A, T, u8>;

    fn is_inf(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.is_inf(access).map(AccessOp::wrap),
            Self::Host(host) => host.is_inf(access).map(AccessOp::wrap),
        }
    }

    fn is_nan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.is_nan(access).map(AccessOp::wrap),
            Self::Host(host) => host.is_nan(access).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Float> ElementwiseTrig<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn sin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.sin(access).map(AccessOp::wrap),
            Self::Host(host) => host.sin(access).map(AccessOp::wrap),
        }
    }

    fn asin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.asin(access).map(AccessOp::wrap),
            Self::Host(host) => host.asin(access).map(AccessOp::wrap),
        }
    }

    fn sinh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.sinh(access).map(AccessOp::wrap),
            Self::Host(host) => host.sinh(access).map(AccessOp::wrap),
        }
    }

    fn cos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.cos(access).map(AccessOp::wrap),
            Self::Host(host) => host.cos(access).map(AccessOp::wrap),
        }
    }

    fn acos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.acos(access).map(AccessOp::wrap),
            Self::Host(host) => host.acos(access).map(AccessOp::wrap),
        }
    }

    fn cosh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.cosh(access).map(AccessOp::wrap),
            Self::Host(host) => host.cosh(access).map(AccessOp::wrap),
        }
    }

    fn tan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.tan(access).map(AccessOp::wrap),
            Self::Host(host) => host.tan(access).map(AccessOp::wrap),
        }
    }

    fn atan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.atan(access).map(AccessOp::wrap),
            Self::Host(host) => host.atan(access).map(AccessOp::wrap),
        }
    }

    fn tanh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.tanh(access).map(AccessOp::wrap),
            Self::Host(host) => host.tanh(access).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Float> ElementwiseUnary<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn exp(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.exp(access).map(AccessOp::wrap),
            Self::Host(host) => host.exp(access).map(AccessOp::wrap),
        }
    }

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.ln(access).map(AccessOp::wrap),
            Self::Host(host) => host.ln(access).map(AccessOp::wrap),
        }
    }

    fn round(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.round(access).map(AccessOp::wrap),
            Self::Host(host) => host.round(access).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> ElementwiseUnaryBoolean<A, T> for Platform {
    type Op = Unary<A, T, u8>;

    fn not(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.not(access).map(AccessOp::wrap),
            Self::Host(host) => host.not(access).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "complex")]
impl<A, T> complex::Fourier<A, num_complex::Complex<T>> for Platform
where
    A: Access<num_complex::Complex<T>>,
    T: rustfft::FftNum,
    num_complex::Complex<T>: crate::Complex,
{
    type Op = complex::FFT<A, num_complex::Complex<T>>;

    fn fft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        // TODO: support FFT on OpenCL
        let host = match self {
            #[cfg(feature = "opencl")]
            Self::CL(_) => return Err(Error::Unsupported(format!("OpenCL Fourier transform for {} is unavailable; select the host backend explicitly", std::any::type_name::<num_complex::Complex<T>>()))),
            Self::Host(host) => host,
        };

        host.fft(access, dim).map(AccessOp::wrap)
    }

    fn ifft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        // TODO: support IFFT on OpenCL
        let host = match self {
            #[cfg(feature = "opencl")]
            Self::CL(_) => return Err(Error::Unsupported(format!("OpenCL Fourier transform for {} is unavailable; select the host backend explicitly", std::any::type_name::<num_complex::Complex<T>>()))),
            Self::Host(host) => host,
        };

        host.ifft(access, dim).map(AccessOp::wrap)
    }
}

impl<A, L, R, T> GatherCond<A, L, R, T> for Platform
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Cond<A, L, R, T>;

    fn cond(self, cond: A, then: L, or_else: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.cond(cond, then, or_else).map(AccessOp::wrap),
            Self::Host(host) => host.cond(cond, then, or_else).map(AccessOp::wrap),
        }
    }
}

impl<L, R, T> LinAlgDual<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = MatMul<L, R, T>;

    fn matmul(
        self,
        left: L,
        right: R,
        dims: [usize; 4],
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.matmul(left, right, dims).map(AccessOp::wrap),
            Self::Host(host) => host.matmul(left, right, dims).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> LinAlgUnary<A, T> for Platform {
    type Op = MatDiag<A, T>;

    fn diag(
        self,
        access: A,
        batch_size: usize,
        dim: usize,
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.diag(access, batch_size, dim).map(AccessOp::wrap),
            Self::Host(host) => host.diag(access, batch_size, dim).map(AccessOp::wrap),
        }
    }
}

impl Random for Platform {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.random_normal(size).map(AccessOp::wrap),
            Self::Host(host) => host.random_normal(size).map(AccessOp::wrap),
        }
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.random_uniform(size).map(AccessOp::wrap),
            Self::Host(host) => host.random_uniform(size).map(AccessOp::wrap),
        }
    }
}

impl<A, T> ReduceAll<A, T> for Platform
where
    A: Access<T>,
    T: Number,
{
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.all(access),
            Self::Host(host) => host.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.any(access),
            Self::Host(host) => host.any(access),
        }
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAll::max(cl, access),
            Self::Host(host) => ReduceAll::max(host, access),
        }
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAll::min(cl, access),
            Self::Host(host) => ReduceAll::min(host, access),
        }
    }

    fn product(self, access: A) -> Result<T, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAll::product(cl, access),
            Self::Host(host) => ReduceAll::product(host, access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAll::sum(cl, access),
            Self::Host(host) => ReduceAll::sum(host, access),
        }
    }
}

impl<A: Access<T>, T: Number> ReduceAxes<A, T> for Platform {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAxes::max(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxes::max(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAxes::min(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxes::min(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAxes::product(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxes::product(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => ReduceAxes::sum(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxes::sum(host, access, stride).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: Number> Transform<A, T> for Platform {
    type Broadcast = View<A, T>;
    type Flip = Flip<A, T>;
    type Slice = Slice<A, T>;
    type Transpose = View<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.broadcast(access, shape, broadcast).map(AccessOp::wrap),
            Self::Host(host) => host.broadcast(access, shape, broadcast).map(AccessOp::wrap),
        }
    }

    fn flip(
        self,
        access: A,
        shape: Shape,
        axis: usize,
    ) -> Result<AccessOp<Self::Flip, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.flip(access, shape, axis).map(AccessOp::wrap),
            Self::Host(host) => host.flip(access, shape, axis).map(AccessOp::wrap),
        }
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.slice(access, shape, range).map(AccessOp::wrap),
            Self::Host(host) => host.slice(access, shape, range).map(AccessOp::wrap),
        }
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.transpose(access, shape, permutation).map(AccessOp::wrap),
            Self::Host(host) => host
                .transpose(access, shape, permutation)
                .map(AccessOp::wrap),
        }
    }
}
