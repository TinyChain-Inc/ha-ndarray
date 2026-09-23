#![doc = include_str!("../NUMERICS.md")]

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use number_general as ng;
use safecast::CastFrom;

#[cfg(feature = "complex")]
pub use num_complex as complex;
pub use smallvec::smallvec as axes;
pub use smallvec::smallvec as coord;
pub use smallvec::smallvec as range;
pub use smallvec::smallvec as slice;
pub use smallvec::smallvec as shape;
pub use smallvec::smallvec as stackvec;
use smallvec::SmallVec;

pub use access::*;
pub use array::{
    same_shape, MatrixDual, MatrixUnary, NDArray, NDArrayAbs, NDArrayBoolean, NDArrayBooleanScalar,
    NDArrayCast, NDArrayCompare, NDArrayCompareScalar, NDArrayMath, NDArrayMathScalar,
    NDArrayNumeric, NDArrayRead, NDArrayReduce, NDArrayReduceAll, NDArrayReduceBoolean,
    NDArrayTransform, NDArrayTrig, NDArrayUnary, NDArrayUnaryBoolean, NDArrayWhere, NDArrayWrite,
};
#[cfg(feature = "complex")]
pub use array::{MatrixUnaryComplex, NDArrayComplex, NDArrayFourier};
pub use buffer::{Buffer, BufferConverter, BufferInstance, BufferMut};
pub use host::StackVec;
pub use platform::*;

mod access;
mod array;
mod buffer;
#[cfg(feature = "complex")]
pub mod fft;
pub mod host;
mod numeric;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod ops;
mod platform;

fn id<T>(this: T) -> T {
    this
}

#[cfg(feature = "opencl")]
pub trait CLType:
    opencl::CLElement + PartialEq + Copy + Send + Sync + fmt::Display + fmt::Debug + 'static
{
}

#[cfg(not(feature = "opencl"))]
pub trait CLType: PartialEq + Copy + Send + Sync + fmt::Display + fmt::Debug + 'static {}

impl CLType for f32 {}
impl CLType for f64 {}
impl CLType for i8 {}
impl CLType for i16 {}
impl CLType for i32 {}
impl CLType for i64 {}
impl CLType for u8 {}
impl CLType for u16 {}
impl CLType for u32 {}
impl CLType for u64 {}
#[cfg(feature = "complex")]
impl CLType for complex::Complex<f32> {}
#[cfg(feature = "complex")]
impl CLType for complex::Complex<f64> {}

/// A numeric type supported by ha-ndarray
pub trait Number: CLType + Into<ng::Number> + CastFrom<ng::Number> + Default {
    /// The zero value of this data type.
    const ZERO: Self;

    /// The one value of this data type.
    const ONE: Self;

    /// The absolute value type of this [`Number`].
    type Abs: Number;

    // arithmetic

    /// Construct an instance of this type from a [`f64`].
    fn abs(self) -> Self::Abs;

    /// Add two instances of this type.
    fn add(self, other: Self) -> Self;

    /// Divide two instances of this type.
    fn div(self, other: Self) -> Self;

    /// Multiply two instances of this type.
    fn mul(self, other: Self) -> Self;

    /// Subtract two instances of this type.
    fn sub(self, other: Self) -> Self;

    /// Raise this value to the power of the given exponent.
    /// Integer powers wrap at full width; negative exponents follow the crate numerical contract.
    fn pow(self, exp: Self) -> Self;
}

macro_rules! number {
    ($t:ty, $abs_t:ty, $one:expr, $zero:expr, $abs:expr, $add:expr, $div:expr, $mul:expr, $sub:expr, $pow:expr) => {
        impl Number for $t {
            const ONE: Self = $one;

            const ZERO: Self = $zero;

            type Abs = $abs_t;

            fn abs(self) -> Self::Abs {
                $abs(self)
            }

            fn add(self, other: Self) -> Self {
                $add(self, other)
            }

            fn div(self, other: Self) -> Self {
                $div(self, other)
            }

            fn mul(self, other: Self) -> Self {
                $mul(self, other)
            }

            fn sub(self, other: Self) -> Self {
                $sub(self, other)
            }

            fn pow(self, exp: Self) -> Self {
                ($pow)(self, exp)
            }
        }
    };
}

#[cfg(feature = "complex")]
number!(
    complex::Complex32,
    f32,
    complex::Complex32::ONE,
    complex::Complex32::ZERO,
    complex::Complex32::norm,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    complex::Complex32::powc
);

#[cfg(feature = "complex")]
number!(
    complex::Complex64,
    f64,
    complex::Complex64::ONE,
    complex::Complex64::ZERO,
    complex::Complex64::norm,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    complex::Complex64::powc
);

number!(
    f32,
    Self,
    1.,
    0.,
    f32::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    f32::powf
);

number!(
    f64,
    Self,
    1.,
    0.,
    f64::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    f64::powf
);

// Exponentiation stays in the integer domain, including full-width exponents.
macro_rules! integer_number {
    ($t:ty, $signed:expr) => {
        number!(
            $t,
            Self,
            1,
            0,
            |n: Self| if $signed && (n as i128) < 0 {
                n.wrapping_neg()
            } else {
                n
            },
            Self::wrapping_add,
            |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
            Self::wrapping_mul,
            Self::wrapping_sub,
            |mut base: Self, mut exp: Self| {
                if $signed && (exp as i128) < 0 {
                    return if base == 1 {
                        1
                    } else if base == (1 as Self).wrapping_neg() {
                        if exp & 1 == 0 {
                            1
                        } else {
                            base
                        }
                    } else {
                        0
                    };
                }

                let mut result: Self = 1;

                while exp != 0 {
                    if exp & 1 != 0 {
                        result = result.wrapping_mul(base);
                    }

                    exp >>= 1;
                    base = base.wrapping_mul(base);
                }

                result
            }
        );
    };
}

integer_number!(i8, true);
integer_number!(i16, true);
integer_number!(i32, true);
integer_number!(i64, true);
integer_number!(u8, false);
integer_number!(u16, false);
integer_number!(u32, false);
integer_number!(u64, false);

#[cfg(not(feature = "opencl"))]
/// A real-valued [`Number`]
pub trait Real: Number + PartialOrd {
    /// The maximum value of this data type.
    const MAX: Self;

    /// The minimum value of this data type.
    const MIN: Self;

    /// Return the maximum; floating NaNs propagate and equal zeros select +0.
    fn max(l: Self, r: Self) -> Self;

    /// Return the minimum; floating NaNs propagate and equal zeros select -0.
    fn min(l: Self, r: Self) -> Self;

    /// Compute the remainder of `self.div(other)`.
    fn rem(self, other: Self) -> Self;

    /// Round to nearest integer with ties away from zero; preserve zero sign.
    fn round(self) -> Self;
}

#[cfg(feature = "opencl")]
/// A real-valued [`Number`]
pub trait Real: Number + PartialOrd + opencl::CLElementReal {
    /// The maximum value of this data type.
    const MAX: Self;

    /// The minimum value of this data type.
    const MIN: Self;

    /// Return the maximum; floating NaNs propagate and equal zeros select +0.
    fn max(l: Self, r: Self) -> Self;

    /// Return the minimum; floating NaNs propagate and equal zeros select -0.
    fn min(l: Self, r: Self) -> Self;

    /// Compute the remainder of `self.div(other)`.
    fn rem(self, other: Self) -> Self;

    /// Round to nearest integer with ties away from zero; preserve zero sign.
    fn round(self) -> Self;
}

macro_rules! real {
    ($t:ty, $rem:expr, $ord:expr, $round:expr) => {
        impl Real for $t {
            const MAX: Self = <$t>::MAX;

            const MIN: Self = <$t>::MIN;

            fn max(l: Self, r: Self) -> $t {
                match $ord(&l, &r) {
                    Ordering::Greater | Ordering::Equal => l,
                    Ordering::Less => r,
                }
            }

            fn min(l: Self, r: Self) -> $t {
                match $ord(&l, &r) {
                    Ordering::Less | Ordering::Equal => l,
                    Ordering::Greater => r,
                }
            }

            fn rem(self, other: Self) -> Self {
                $rem(self, other)
            }

            fn round(self) -> Self {
                $round(self)
            }
        }
    };
}

macro_rules! real_float {
    ($t:ty) => {
        impl Real for $t {
            const MAX: Self = <$t>::MAX;
            const MIN: Self = <$t>::MIN;

            fn max(l: Self, r: Self) -> Self {
                if l.is_nan() || r.is_nan() {
                    Self::NAN
                } else if l == 0.0 && r == 0.0 {
                    if l.is_sign_positive() || r.is_sign_positive() {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    l.max(r)
                }
            }

            fn min(l: Self, r: Self) -> Self {
                if l.is_nan() || r.is_nan() {
                    Self::NAN
                } else if l == 0.0 && r == 0.0 {
                    if l.is_sign_negative() || r.is_sign_negative() {
                        -0.0
                    } else {
                        0.0
                    }
                } else {
                    l.min(r)
                }
            }

            fn rem(self, rhs: Self) -> Self {
                self % rhs
            }

            fn round(self) -> Self {
                <$t>::round(self)
            }
        }
    };
}

real_float!(f32);
real_float!(f64);
real!(
    i8,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    i16,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    i32,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    i64,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    u8,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    u16,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    u32,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);
real!(
    u64,
    |l, r| if r == 0 { 0 } else { Self::wrapping_rem(l, r) },
    Ord::cmp,
    id
);

#[cfg(not(feature = "opencl"))]
/// A floating-point [`Number`]
pub trait Float: Number {
    // numeric methods
    /// Return `true` if this [`Float`] is infinite (positive or negative infinity).
    fn is_inf(self) -> bool;

    /// Return `true` if this [`Float`] is not a number (e.g. a float representation of `1.0 / 0.0`).
    fn is_nan(self) -> bool;

    // logarithms
    /// Exponentiate this number (equivalent to `consts::E.pow(self)`).
    fn exp(self) -> Self;

    /// Return the natural logarithm of this [`Float`].
    fn ln(self) -> Self;

    /// Calculate the logarithm of this [`Float`] w/r/t the given `base`.
    fn log(self, base: Self) -> Self;

    // trigonometry
    /// Return the sine of this [`Float`] (in radians).
    fn sin(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn asin(self) -> Self;

    /// Return the hyperbolic sine of this [`Float`] (in radians).
    fn sinh(self) -> Self;

    /// Return the cosine of this [`Float`] (in radians).
    fn cos(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn acos(self) -> Self;

    /// Return the hyperbolic cosine of this [`Float`] (in radians).
    fn cosh(self) -> Self;

    /// Return the tangent of this [`Float`] (in radians).
    fn tan(self) -> Self;

    /// Return the arctangent of this [`Float`] (in radians).
    fn atan(self) -> Self;

    /// Return the hyperbolic tangent of this [`Float`] (in radians).
    fn tanh(self) -> Self;
}

#[cfg(feature = "opencl")]
/// A floating-point [`Number`]
pub trait Float: Number + opencl::CLElementTrig {
    // numeric methods
    /// Return `true` if this [`Float`] is infinite (positive or negative infinity).
    fn is_inf(self) -> bool;

    /// Return `true` if this [`Float`] is not a number (e.g. a float representation of `1.0 / 0.0`).
    fn is_nan(self) -> bool;

    // logarithms
    /// Exponentiate this number (equivalent to `consts::E.pow(self)`).
    fn exp(self) -> Self;

    /// Return the natural logarithm of this [`Float`].
    fn ln(self) -> Self;

    /// Calculate the logarithm of this [`Float`] w/r/t the given `base`.
    fn log(self, base: Self) -> Self;

    // trigonometry
    /// Return the sine of this [`Float`] (in radians).
    fn sin(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn asin(self) -> Self;

    /// Return the hyperbolic sine of this [`Float`] (in radians).
    fn sinh(self) -> Self;

    /// Return the cosine of this [`Float`] (in radians).
    fn cos(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn acos(self) -> Self;

    /// Return the hyperbolic cosine of this [`Float`] (in radians).
    fn cosh(self) -> Self;

    /// Return the tangent of this [`Float`] (in radians).
    fn tan(self) -> Self;

    /// Return the arctangent of this [`Float`] (in radians).
    fn atan(self) -> Self;

    /// Return the hyperbolic tangent of this [`Float`] (in radians).
    fn tanh(self) -> Self;
}

macro_rules! float_type {
    ($t:ty, $inf:expr, $nan:expr) => {
        impl Float for $t {
            fn is_inf(self) -> bool {
                $inf(self)
            }

            fn is_nan(self) -> bool {
                $nan(self)
            }

            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            fn log(self, base: Self) -> Self {
                self.ln() / base.ln()
            }

            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            fn asin(self) -> Self {
                <$t>::asin(self)
            }

            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }

            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            fn acos(self) -> Self {
                <$t>::acos(self)
            }

            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }

            fn tan(self) -> Self {
                <$t>::tan(self)
            }

            fn atan(self) -> Self {
                <$t>::atan(self)
            }

            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
        }
    };
}

#[cfg(feature = "complex")]
float_type!(
    complex::Complex32,
    |n: complex::Complex32| n.re.is_infinite() || n.im.is_infinite(),
    |n: complex::Complex32| n.re.is_nan() || n.im.is_nan()
);
#[cfg(feature = "complex")]
float_type!(
    complex::Complex64,
    |n: complex::Complex64| n.re.is_infinite() || n.im.is_infinite(),
    |n: complex::Complex64| n.re.is_nan() || n.im.is_nan()
);
float_type!(f32, f32::is_infinite, f32::is_nan);
float_type!(f64, f64::is_infinite, f64::is_nan);

#[cfg(all(feature = "complex", not(feature = "opencl")))]
/// A complex [`Number`]
pub trait Complex: Float<Abs = Self::Real> {
    type Real: Float + Real;

    fn angle(self) -> Self::Real;

    fn conj(self) -> Self;

    fn im(self) -> Self::Real;

    fn re(self) -> Self::Real;
}

#[cfg(all(feature = "complex", feature = "opencl"))]
/// A complex [`Number`]
pub trait Complex: Float<Abs = Self::Real> + opencl::CLElementComplex {
    type Real: Float + Real;

    fn angle(self) -> Self::Real;

    fn conj(self) -> Self;

    fn im(self) -> Self::Real;

    fn re(self) -> Self::Real;
}

#[cfg(feature = "complex")]
macro_rules! complex_type {
    ($t:ty, $r:ty) => {
        impl Complex for $t {
            type Real = $r;

            fn angle(self) -> $r {
                Self::arg(self)
            }

            fn conj(self) -> Self {
                complex::Complex::<$r>::conj(&self)
            }

            fn im(self) -> $r {
                self.im
            }

            fn re(self) -> $r {
                self.re
            }
        }
    };
}

#[cfg(feature = "complex")]
complex_type!(complex::Complex32, f32);
#[cfg(feature = "complex")]
complex_type!(complex::Complex64, f64);

/// An array math error
pub enum Error {
    Bounds(String),
    Unsupported(String),
    #[cfg(feature = "opencl")]
    OCL(std::sync::Arc<ocl::Error>),
}

impl Error {
    pub fn bounds(msg: String) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("{}", msg);

        #[cfg(not(feature = "debug_crash"))]
        Self::Bounds(msg)
    }

    pub fn unsupported(msg: String) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("{}", msg);

        #[cfg(not(feature = "debug_crash"))]
        Self::Unsupported(msg)
    }
}

// Clone is required to support memoizing OpenCL programs
// since constructing an [`ocl::Program`] may return an error
impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Self::Bounds(msg) => Self::Bounds(msg.clone()),
            Self::Unsupported(msg) => Self::Unsupported(msg.clone()),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => Self::OCL(cause.clone()),
        }
    }
}

#[cfg(feature = "opencl")]
impl From<ocl::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("OpenCL error: {:?}", cause);

        #[cfg(not(feature = "debug_crash"))]
        Self::OCL(std::sync::Arc::new(cause))
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Unsupported(cause) => f.write_str(cause),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => cause.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Unsupported(cause) => f.write_str(cause),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

/// A list of n-dimensional array axes
pub type Axes = SmallVec<[usize; 8]>;

/// An n-dimensional selection range, used to slice an array
pub type Range = SmallVec<[AxisRange; 8]>;

/// The shape of an n-dimensional array
pub type Shape = SmallVec<[usize; 8]>;

/// The strides used to access an n-dimensional array
pub type Strides = SmallVec<[usize; 8]>;

/// An n-dimensional array on the top-level [`Platform`]
pub type Array<T, A> = array::Array<T, A, Platform>;

/// An n-dimensional array backed by a buffer on the top-level [`Platform`]
pub type ArrayBuf<T, B> = array::Array<T, AccessBuf<B>, Platform>;

/// The result of an n-dimensional array operation
pub type ArrayOp<T, Op> = array::Array<T, AccessOp<Op>, Platform>;

/// A general type of n-dimensional array used to elide recursive types
pub type ArrayAccess<'a, T> = array::Array<T, Accessor<'a, T>, Platform>;

/// An accessor for the result of an n-dimensional array operation on the top-level [`Platform`]
pub type AccessOp<Op> = access::AccessOp<Op, Platform>;

/// Bounds on an individual array axis
#[derive(Clone, Eq, PartialEq, Hash)]
pub enum AxisRange {
    At(usize),
    In(usize, usize, usize),
    Of(SmallVec<[usize; 8]>),
}

impl AxisRange {
    /// Return `true` if this is an index bound (i.e. not a slice)
    pub fn is_index(&self) -> bool {
        matches!(self, Self::At(_))
    }

    /// Return the number of elements contained within this bound.
    /// Returns `None` for an index bound.
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::At(_) => None,
            Self::In(start, stop, step) => Some((stop - start) / step),
            Self::Of(indices) => Some(indices.len()),
        }
    }
}

impl From<usize> for AxisRange {
    fn from(i: usize) -> Self {
        Self::At(i)
    }
}

impl From<std::ops::Range<usize>> for AxisRange {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::In(range.start, range.end, 1)
    }
}

impl From<SmallVec<[usize; 8]>> for AxisRange {
    fn from(indices: SmallVec<[usize; 8]>) -> Self {
        Self::Of(indices)
    }
}

impl fmt::Debug for AxisRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::At(i) => write!(f, "{}", i),
            Self::In(start, stop, 1) => write!(f, "{}:{}", start, stop),
            Self::In(start, stop, step) => write!(f, "{}:{}:{}", start, stop, step),
            Self::Of(indices) => write!(f, "{:?}", indices),
        }
    }
}

/// Compute the shape which results from broadcasting the `left` and `right` shapes, if possible.
#[inline]
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
    let ndim = usize::max(left.len(), right.len());
    let mut shape = Shape::with_capacity(ndim);

    let mut left = left.iter().rev().copied();
    let mut right = right.iter().rev().copied();

    while let Some(dim) = broadcast_dim(left.next(), right.next())? {
        shape.push(dim)
    }

    shape.reverse();

    Ok(shape)
}

/// Compute the shapes needed to multiply the `left` and `right` matrices, if possible.
#[inline]
pub fn broadcast_matmul_shape(left: &[usize], right: &[usize]) -> Result<(Shape, Shape), Error> {
    let (left_ndim, right_ndim) = (left.len(), right.len());
    let ndim = usize::max(left_ndim, right_ndim);

    let mut left = left.iter().rev().copied();
    let mut right = right.iter().rev().copied();

    let k = right.next().unwrap_or(1);
    let j = match (left.next(), right.next()) {
        (Some(jl), Some(jr)) => match (jl, jr) {
            (jl, jr) if jl == jr => Ok(jl),
            (1, jr) => Ok(jr),
            (jl, 1) => Ok(jl),
            _ => Err(Error::bounds(format!(
                "cannot matrix-multiply shapes {left:?} and {right:?}"
            ))),
        },
        (Some(jl), None) => Ok(jl),
        (None, Some(jr)) => Ok(jr),
        (None, None) => Ok(1),
    }?;
    let i = left.next().unwrap_or(1);

    let mut broadcast_shape = Shape::with_capacity(ndim);
    while let Some(dim) = broadcast_dim(left.next(), right.next())? {
        broadcast_shape.push(dim);
    }

    broadcast_shape.reverse();

    let left = broadcast_shape.iter().copied().chain([i, j]).collect();
    let right = broadcast_shape.into_iter().chain([j, k]).collect();
    Ok((left, right))
}

#[inline]
fn broadcast_dim(left: Option<usize>, right: Option<usize>) -> Result<Option<usize>, Error> {
    match (left, right) {
        (Some(l), Some(r)) if l == r => Ok(Some(l)),
        (Some(1), Some(r)) => Ok(Some(r)),
        (Some(l), Some(1)) => Ok(Some(l)),
        (None, Some(r)) => Ok(Some(r)),
        (Some(l), None) => Ok(Some(l)),
        (None, None) => Ok(None),
        (l, r) => Err(Error::bounds(format!(
            "cannot broadcast dimensions {l:?} and {r:?}"
        ))),
    }
}

#[inline]
fn range_shape(source_shape: &[usize], range: &[AxisRange]) -> Shape {
    debug_assert_eq!(source_shape.len(), range.len());
    range.iter().filter_map(|ar| ar.size()).collect()
}

/// Construct an iterator over the strides for the given shape and number of dimensions.
#[inline]
pub fn strides_for<'a>(shape: &'a [usize], ndim: usize) -> impl Iterator<Item = usize> + 'a {
    debug_assert!(ndim >= shape.len());

    let zeros = std::iter::repeat_n(0, ndim - shape.len());

    let strides = shape.iter().copied().enumerate().map(|(x, dim)| {
        if dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides)
}
