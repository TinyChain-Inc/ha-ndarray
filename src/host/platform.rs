use number_general as ng;
use rayon::prelude::*;
use safecast::CastInto;

use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::host::StackVec;
use crate::ops::{
    Concat, ConstructConcat, ConstructRange, ElementwiseAbs, ElementwiseBoolean,
    ElementwiseBooleanScalar, ElementwiseCast, ElementwiseCompare, ElementwiseCompareScalar,
    ElementwiseDual, ElementwiseNumeric, ElementwiseScalar, ElementwiseTrig, ElementwiseUnary,
    ElementwiseUnaryBoolean, GatherCond, LinAlgDual, LinAlgUnary, Random, ReduceAll, ReduceAxes,
    Transform,
};
use crate::platform::{Convert, PlatformInstance};
use crate::{stackvec, Axes, Constant, Error, Float, Number, Range, Real, Shape};

use super::buffer::Buffer;
use super::ops::*;

pub const VEC_MIN_SIZE: usize = 64;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: Number> Constant<T> for Stack {
    type Buffer = StackVec<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        Ok(stackvec![value; size])
    }
}

impl<T: Number> Convert<T> for Stack {
    type Buffer = StackVec<T>;

    fn convert(&self, buffer: BufferConverter<T>) -> Result<Self::Buffer, Error> {
        buffer.to_slice().map(|buf| buf.into_stackvec())
    }
}

impl<A, T> ReduceAll<A, T> for Stack
where
    A: Access<T>,
    T: Number,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().any(|n| n != T::ZERO))
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::max).expect("max"))
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::min).expect("min"))
    }

    fn product(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::mul).expect("product"))
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.iter().copied().reduce(T::add).expect("sum"))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: Number> Constant<T> for Heap {
    type Buffer = Vec<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        Ok(vec![value; size])
    }
}

impl<T: Number> Convert<T> for Heap {
    type Buffer = Vec<T>;

    fn convert(&self, buffer: BufferConverter<T>) -> Result<Self::Buffer, Error> {
        buffer.to_slice().map(|buf| buf.into_vec())
    }
}

impl<A, T> ReduceAll<A, T> for Heap
where
    A: Access<T>,
    T: Number,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().any(|n| n != T::ZERO))
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            slice
                .into_par_iter()
                .copied()
                .reduce(|| crate::numeric::minimum::<T>(), T::max)
        })
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            slice
                .into_par_iter()
                .copied()
                .reduce(|| crate::numeric::maximum::<T>(), T::min)
        })
    }

    fn product(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().reduce(|| T::ONE, T::mul))
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.into_par_iter().copied().reduce(|| T::ZERO, T::add))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Host {
    Stack(Stack),
    Heap(Heap),
}

impl PlatformInstance for Host {
    fn select(size_hint: usize) -> Self {
        if size_hint < VEC_MIN_SIZE {
            Self::Stack(Stack)
        } else {
            Self::Heap(Heap)
        }
    }
}

impl<T: Number> Constant<T> for Host {
    type Buffer = Buffer<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        match self {
            Self::Heap(heap) => heap.constant(value, size).map(Buffer::Heap),
            Self::Stack(stack) => stack.constant(value, size).map(Buffer::Stack),
        }
    }
}

impl<T: Number> Convert<T> for Host {
    type Buffer = Buffer<T>;

    fn convert(&self, buffer: BufferConverter<T>) -> Result<Self::Buffer, Error> {
        match self {
            Self::Heap(heap) => heap.convert(buffer).map(Buffer::Heap),
            Self::Stack(stack) => stack.convert(buffer).map(Buffer::Stack),
        }
    }
}

impl From<Heap> for Host {
    fn from(heap: Heap) -> Self {
        Self::Heap(heap)
    }
}

impl From<Stack> for Host {
    fn from(stack: Stack) -> Self {
        Self::Stack(stack)
    }
}

impl<A, T> ConstructConcat<A, T> for Host
where
    A: Access<T>,
    T: Number,
{
    type Op = Concat<A, T>;

    fn concat(self, data: Vec<A>) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Concat::new(data).into())
    }
}

impl<T: Number + PartialOrd> ConstructRange<T> for Host {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        if start <= stop {
            let size_t = ng::Number::from(size as u64).cast_into();
            let step = T::div(T::sub(stop, start), size_t);
            Ok(Linear::new(start, step, size).into())
        } else {
            Err(Error::bounds(format!("invalid range: [{start}, {stop})")))
        }
    }
}

impl<A: Access<IT>, IT: Number, OT: Number> ElementwiseCast<A, IT, OT> for Host {
    type Op = Cast<A, IT, OT>;

    fn cast(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Cast::new(access).into())
    }
}

impl<A, L, R, T> GatherCond<A, L, R, T> for Host
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Cond<A, L, R, T>;

    fn cond(self, cond: A, then: L, or_else: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Cond::new(cond, then, or_else).into())
    }
}

impl<A: Access<T>, T: Number> ElementwiseAbs<A, T> for Host {
    type Op = Unary<A, T, T::Abs>;

    fn abs(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::abs(access).into())
    }
}

impl<L, R, T> ElementwiseBoolean<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::and(left, right).into())
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::or(left, right).into())
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::xor(left, right).into())
    }
}

impl<A: Access<T>, T: Number> ElementwiseBooleanScalar<A, T> for Host {
    type Op = Scalar<A, T, u8>;

    fn and_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::and(left, right).into())
    }

    fn or_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::or(left, right).into())
    }

    fn xor_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::xor(left, right).into())
    }
}

impl<L, R, T> ElementwiseCompare<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialEq,
    {
        Ok(Dual::eq(left, right).into())
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialOrd,
    {
        Ok(Dual::ge(left, right).into())
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialOrd,
    {
        Ok(Dual::gt(left, right).into())
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialOrd,
    {
        Ok(Dual::le(left, right).into())
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialOrd,
    {
        Ok(Dual::lt(left, right).into())
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: PartialEq,
    {
        Ok(Dual::ne(left, right).into())
    }
}

impl<A, T> ElementwiseCompareScalar<A, T> for Host
where
    A: Access<T>,
    T: Number,
{
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::eq(left, right).into())
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Scalar::ge(left, right).into())
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Scalar::gt(left, right).into())
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Scalar::le(left, right).into())
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Scalar::lt(left, right).into())
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::ne(left, right).into())
    }
}

#[cfg(feature = "complex")]
impl<A, T> crate::ops::complex::ElementwiseUnaryComplex<A, T> for Host
where
    A: Access<T>,
    T: crate::Complex,
{
    type Real = Unary<A, T, T::Real>;
    type Complex = Unary<A, T, T>;

    fn angle(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Ok(Unary::angle(access).into())
    }

    fn conj(self, access: A) -> Result<AccessOp<Self::Complex, Self>, Error> {
        Ok(Unary::conj(access).into())
    }

    fn re(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Ok(Unary::re(access).into())
    }

    fn im(self, access: A) -> Result<AccessOp<Self::Real, Self>, Error> {
        Ok(Unary::im(access).into())
    }
}

impl<L, R, T> ElementwiseDual<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: Number,
{
    type Op = Dual<L, R, T, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::add(left, right).into())
    }

    fn div(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::div(left, right).into())
    }

    fn log(self, arg: L, base: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        Ok(Dual::log(arg, base).into())
    }

    fn mul(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::mul(left, right).into())
    }

    fn pow(self, arg: L, exp: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::pow(arg, exp).into())
    }

    fn rem(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Dual::rem(left, right).into())
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::sub(left, right).into())
    }
}

impl<A: Access<T>, T: Number> ElementwiseScalar<A, T> for Host {
    type Op = Scalar<A, T, T>;

    fn add_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::add(left, right).into())
    }

    fn div_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::div(left, right).into())
    }

    fn log_scalar(self, arg: A, base: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Float,
    {
        Ok(Scalar::log(arg, base).into())
    }

    fn mul_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::mul(left, right).into())
    }

    fn pow_scalar(self, arg: A, exp: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::pow(arg, exp).into())
    }

    fn rem_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Scalar::rem(left, right).into())
    }

    fn sub_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Scalar::sub(left, right).into())
    }
}

impl<A: Access<T>, T: Float> ElementwiseNumeric<A, T> for Host {
    type Op = Unary<A, T, u8>;

    fn is_inf(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::inf(access).into())
    }

    fn is_nan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::nan(access).into())
    }
}

impl<A: Access<T>, T: Float> ElementwiseTrig<A, T> for Host {
    type Op = Unary<A, T, T>;

    fn sin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::sin(access).into())
    }

    fn asin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::asin(access).into())
    }

    fn sinh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::sinh(access).into())
    }

    fn cos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::cos(access).into())
    }

    fn acos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::acos(access).into())
    }

    fn cosh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::cosh(access).into())
    }

    fn tan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::tan(access).into())
    }

    fn atan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::atan(access).into())
    }

    fn tanh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::tanh(access).into())
    }
}

impl<A: Access<T>, T: Float> ElementwiseUnary<A, T> for Host {
    type Op = Unary<A, T, T>;

    fn exp(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::exp(access).into())
    }

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::ln(access).into())
    }

    fn round(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Unary::round(access).into())
    }
}

impl<A: Access<T>, T: Number> ElementwiseUnaryBoolean<A, T> for Host {
    type Op = Unary<A, T, u8>;

    fn not(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::not(access).into())
    }
}

#[cfg(feature = "complex")]
impl<A, T> crate::ops::complex::Fourier<A, num_complex::Complex<T>> for Host
where
    A: Access<num_complex::Complex<T>>,
    T: rustfft::FftNum,
    num_complex::Complex<T>: crate::Complex,
{
    type Op = complex::FFT<A, num_complex::Complex<T>>;

    fn fft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        complex::FFT::fft(access, dim).map(AccessOp::from)
    }

    fn ifft(self, access: A, dim: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        complex::FFT::ifft(access, dim).map(AccessOp::from)
    }
}

impl<L, R, T> LinAlgDual<L, R, T> for Host
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
        Ok(MatMul::new(left, right, dims).into())
    }
}

impl<A: Access<T>, T: Number> LinAlgUnary<A, T> for Host {
    type Op = MatDiag<A, T>;

    fn diag(
        self,
        access: A,
        batch_size: usize,
        dim: usize,
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(MatDiag::new(access, batch_size, dim).into())
    }
}

impl Random for Host {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        Ok(RandomNormal::new(size).into())
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        Ok(RandomUniform::new(size).into())
    }
}

impl<A: Access<T>, T: Number> ReduceAll<A, T> for Host {
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.all(access),
            Self::Stack(stack) => stack.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.any(access),
            Self::Stack(stack) => stack.any(access),
        }
    }

    fn max(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        match self {
            Self::Heap(heap) => heap.max(access),
            Self::Stack(stack) => stack.max(access),
        }
    }

    fn min(self, access: A) -> Result<T, Error>
    where
        T: Real,
    {
        match self {
            Self::Heap(heap) => heap.min(access),
            Self::Stack(stack) => stack.min(access),
        }
    }

    fn product(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.product(access),
            Self::Stack(stack) => stack.product(access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.sum(access),
            Self::Stack(stack) => stack.sum(access),
        }
    }
}

impl<A: Access<T>, T: Number> ReduceAxes<A, T> for Host {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Reduce::max(access, stride).into())
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>
    where
        T: Real,
    {
        Ok(Reduce::min(access, stride).into())
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::product(access, stride).into())
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Reduce::sum(access, stride).into())
    }
}

impl<A, T> Transform<A, T> for Host
where
    A: Access<T>,
    T: Number,
{
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
        Ok(View::broadcast(access, shape, broadcast).into())
    }

    fn flip(
        self,
        access: A,
        shape: Shape,
        axis: usize,
    ) -> Result<AccessOp<Self::Flip, Self>, Error> {
        Flip::new(access, shape, axis).map(AccessOp::from)
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Ok(Slice::new(access, shape, range).into())
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        Ok(View::transpose(access, shape, permutation).into())
    }
}
