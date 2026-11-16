use std::fmt;

use crate::access::{Access, AccessOp};
use crate::buffer::{Buffer, BufferConverter, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, Error, Float, Number, Real};

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
    /// The type of buffer used by this platform
    type Buffer: BufferInstance<T>;

    /// Construct a new buffer filled with a single value.
    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error>;
}

/// Converter to construct an owned, platform-specific buffer
pub trait Convert<T: Number>: PlatformInstance {
    /// The type of buffer used by this platform
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
        op_dispatch!(self, backend, backend.concat(data).map(AccessOp::wrap))
    }
}

impl<T: Number + PartialOrd> ConstructRange<T> for Platform {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.range(start, stop, size).map(AccessOp::wrap)
        )
    }
}

impl<A, T> ElementwiseAbs<A, T> for Platform
where
    A: Access<T>,
    T: Number,
{
    type Op = Unary<A, T, T::Abs>;

    fn abs(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.abs(access).map(AccessOp::wrap))
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
        op_dispatch!(self, backend, backend.and(left, right).map(AccessOp::wrap))
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.or(left, right).map(AccessOp::wrap))
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.xor(left, right).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Number> ElementwiseBooleanScalar<A, T> for Platform {
    type Op = Scalar<A, T, u8>;

    fn and_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.and_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn or_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.or_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn xor_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.xor_scalar(left, right).map(AccessOp::wrap)
        )
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> ElementwiseCast<A, IT, OT> for Platform {
    type Op = Cast<A, IT, OT>;

    fn cast(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.cast(access).map(AccessOp::wrap))
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
        op_dispatch!(self, backend, backend.eq(left, right).map(AccessOp::wrap))
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.ge(left, right).map(AccessOp::wrap))
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.gt(left, right).map(AccessOp::wrap))
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.le(left, right).map(AccessOp::wrap))
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.lt(left, right).map(AccessOp::wrap))
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.ne(left, right).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Number> ElementwiseCompareScalar<A, T> for Platform {
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.eq_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            backend.ge_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            backend.gt_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            backend.le_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            backend.lt_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.ne_scalar(left, right).map(AccessOp::wrap)
        )
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
        op_dispatch!(self, backend, backend.angle(access).map(AccessOp::wrap))
    }

    fn conj(self, access: A) -> Result<AccessOp<Self::Complex, Self>, Error> {
        op_dispatch!(self, backend, backend.conj(access).map(AccessOp::wrap))
    }

    fn re(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        op_dispatch!(self, backend, backend.re(access).map(AccessOp::wrap))
    }

    fn im(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        op_dispatch!(self, backend, backend.im(access).map(AccessOp::wrap))
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
        op_dispatch!(self, backend, backend.add(left, right).map(AccessOp::wrap))
    }

    fn div(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.div(left, right).map(AccessOp::wrap))
    }

    fn log(self, arg: L, base: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        op_dispatch!(self, backend, backend.log(arg, base).map(AccessOp::wrap))
    }

    fn mul(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.mul(left, right).map(AccessOp::wrap))
    }

    fn pow(self, arg: L, exp: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.pow(arg, exp).map(AccessOp::wrap))
    }

    fn rem(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.rem(left, right).map(AccessOp::wrap))
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.sub(left, right).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Number> ElementwiseScalar<A, T> for Platform {
    type Op = Scalar<A, T, T>;

    fn add_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.add_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn div_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.div_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn log_scalar(self, arg: A, base: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        op_dispatch!(
            self,
            backend,
            backend.log_scalar(arg, base).map(AccessOp::wrap)
        )
    }

    fn mul_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.mul_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn pow_scalar(self, arg: A, exp: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.pow_scalar(arg, exp).map(AccessOp::wrap)
        )
    }

    fn rem_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            backend.rem_scalar(left, right).map(AccessOp::wrap)
        )
    }

    fn sub_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.sub_scalar(left, right).map(AccessOp::wrap)
        )
    }
}

impl<A: Access<T>, T: Float> ElementwiseNumeric<A, T> for Platform {
    type Op = Unary<A, T, u8>;

    fn is_inf(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.is_inf(access).map(AccessOp::wrap))
    }

    fn is_nan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.is_nan(access).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Float> ElementwiseTrig<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn sin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.sin(access).map(AccessOp::wrap))
    }

    fn asin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.asin(access).map(AccessOp::wrap))
    }

    fn sinh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.sinh(access).map(AccessOp::wrap))
    }

    fn cos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.cos(access).map(AccessOp::wrap))
    }

    fn acos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.acos(access).map(AccessOp::wrap))
    }

    fn cosh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.cosh(access).map(AccessOp::wrap))
    }

    fn tan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.tan(access).map(AccessOp::wrap))
    }

    fn atan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.atan(access).map(AccessOp::wrap))
    }

    fn tanh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.tanh(access).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Float> ElementwiseUnary<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn exp(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.exp(access).map(AccessOp::wrap))
    }

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.ln(access).map(AccessOp::wrap))
    }

    fn round(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, backend.round(access).map(AccessOp::wrap))
    }
}

impl<A: Access<T>, T: Number> ElementwiseUnaryBoolean<A, T> for Platform {
    type Op = Unary<A, T, u8>;

    fn not(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(self, backend, backend.not(access).map(AccessOp::wrap))
    }
}

#[cfg(feature = "complex")]
impl<A, T> complex::Fourier<A, num_complex::Complex<T>> for Platform
where
    A: Access<num_complex::Complex<T>>,
    T: rustfft::FftNum,
    num_complex::Complex<T>: crate::Complex,
{
    type Op = complex::Fft<A, num_complex::Complex<T>>;

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
        op_dispatch!(
            self,
            backend,
            backend.cond(cond, then, or_else).map(AccessOp::wrap)
        )
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
        op_dispatch!(
            self,
            backend,
            backend.matmul(left, right, dims).map(AccessOp::wrap)
        )
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
        op_dispatch!(
            self,
            backend,
            backend.diag(access, batch_size, dim).map(AccessOp::wrap)
        )
    }
}

impl Random for Platform {
    fn random_normal(self, size: usize) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.random_normal(size).map(Buffer::CL),
            Self::Host(host) => host.random_normal(size).map(Buffer::Host),
        }
    }

    fn random_uniform(self, size: usize) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.random_uniform(size).map(Buffer::CL),
            Self::Host(host) => host.random_uniform(size).map(Buffer::Host),
        }
    }
}

impl<A, T> ReduceAll<A, T> for Platform
where
    A: Access<T>,
    T: Number,
{
    fn all(self, access: A) -> Result<bool, Error> {
        op_dispatch!(self, backend, backend.all(access))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        op_dispatch!(self, backend, backend.any(access))
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, ReduceAll::max(backend, access))
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        op_dispatch!(self, backend, ReduceAll::min(backend, access))
    }

    fn product(self, access: A) -> Result<T, Error> {
        op_dispatch!(self, backend, ReduceAll::product(backend, access))
    }

    fn sum(self, access: A) -> Result<T, Error> {
        op_dispatch!(self, backend, ReduceAll::sum(backend, access))
    }
}

impl<A: Access<T>, T: Number> ReduceAxes<A, T> for Platform {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            ReduceAxes::max(backend, access, stride).map(AccessOp::wrap)
        )
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        op_dispatch!(
            self,
            backend,
            ReduceAxes::min(backend, access, stride).map(AccessOp::wrap)
        )
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            ReduceAxes::product(backend, access, stride).map(AccessOp::wrap)
        )
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            ReduceAxes::sum(backend, access, stride).map(AccessOp::wrap)
        )
    }
}

impl<A: Access<T>, T: Number> Transform<A, T> for Platform {
    type View = View<A, T>;
    type Flip = Flip<A, T>;
    type Slice = Slice<A, T>;

    fn view(
        self,
        access: A,
        spec: crate::ops::ViewSpec,
    ) -> Result<AccessOp<Self::View, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.view(access, spec).map(AccessOp::wrap)
        )
    }

    fn flip(
        self,
        access: A,
        spec: crate::ops::FlipSpec,
    ) -> Result<AccessOp<Self::Flip, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.flip(access, spec).map(AccessOp::wrap)
        )
    }

    fn slice(
        self,
        access: A,
        spec: crate::ops::SliceSpec,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        op_dispatch!(
            self,
            backend,
            backend.slice(access, spec).map(AccessOp::wrap)
        )
    }
}
