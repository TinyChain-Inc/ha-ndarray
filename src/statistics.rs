//! Bounded, mergeable statistical accumulators.
//!
//! Callers supply bounded blocks; accumulators own conversion and scaling.
//! They allocate no buffers. Counts retain exact u64 cardinality; conversion to
//! f64 rounds statistical weights above 2^53.

#![forbid(unsafe_code)]

use crate::host::Host;
use crate::{Error, Number, PlatformInstance};

/// A reduction over bounded blocks on platform `P` with element type `T`.
///
/// The caller supplies blocks in the platform's representation and controls
/// traversal and merge order. The built-in operations use borrowed host slices.
pub trait BlockReduce<P: PlatformInstance, T: Number>: Send + Sync {
    type Block: ?Sized;
    type Output: Number;

    /// Fixed-size state. Its default value represents empty input.
    type State: Copy + Default + Send + Sync;

    /// Reduce one block, retaining only its partial state.
    fn partial(&self, block: &Self::Block) -> Result<Self::State, Error>;

    /// Combine states representing disjoint inputs.
    fn merge(left: Self::State, right: Self::State) -> Result<Self::State, Error>;

    /// Produce the reduction result, including the operation's empty-input value.
    fn finish(state: Self::State) -> Self::Output;
}

/// Arithmetic mean, including both components of complex values.
#[derive(Clone, Copy, Debug)]
pub struct MeanOp;

/// Population standard deviation using squared complex magnitude.
#[derive(Clone, Copy, Debug)]
pub struct StandardDeviationOp;

/// Euclidean norm over real or complex values.
#[derive(Clone, Copy, Debug)]
pub struct NormOp;

fn count(left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right)
        .ok_or_else(|| Error::bounds("statistics input cardinality overflow".into()))
}

// A power-of-two scale keeps normalization exact except at the subnormal edge.
// Use the smallest normal power for subnormal values so division scales them up.
fn scale(value: f64) -> f64 {
    f64::from_bits(value.to_bits() & 0x7ff0_0000_0000_0000).max(f64::MIN_POSITIVE)
}

#[derive(Clone, Copy, Debug, Default)]
struct Sum {
    scale: f64,
    sum: f64,
    correction: f64,
}

impl Sum {
    fn rescale(&mut self, scale: f64) {
        if scale > self.scale {
            let ratio = self.scale / scale;
            self.sum *= ratio;
            self.correction *= ratio;
            self.scale = scale;
        }
    }

    fn add(&mut self, value: f64) {
        let sum = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - sum) + value
        } else {
            (value - sum) + self.sum
        };
        self.sum = sum;
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if !value.is_finite() || !self.sum.is_finite() {
            self.sum += value;
            self.correction = 0.;
        } else if value != 0. {
            self.rescale(scale(value));
            self.add(value / self.scale);
        }
    }

    fn merge(mut self, other: Self) -> Self {
        if !self.sum.is_finite() || !other.sum.is_finite() {
            self.sum += other.sum;
            self.correction = 0.;
        } else if other.scale != 0. {
            self.rescale(other.scale);
            let ratio = other.scale / self.scale;
            self.add(other.sum * ratio);
            self.add(other.correction * ratio);
        }
        self
    }
}

/// A compensated, scaled arithmetic mean. Empty input yields NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mean {
    count: u64,
    sum: Sum,
}

impl Mean {
    fn from_batch<T>(values: &[T], component: impl Fn(&T) -> f64) -> Self {
        let mut sum = Sum::default();
        for value in values {
            sum.push(component(value));
        }

        Self {
            count: values.len() as u64,
            sum,
        }
    }

    /// Include one value. Count overflow returns a bounds error without mutation.
    pub fn push(&mut self, value: f64) -> Result<(), Error> {
        self.count = count(self.count, 1)?;
        self.sum.push(value);
        Ok(())
    }

    /// Combine disjoint inputs, including empty inputs; reject count overflow.
    pub fn merge(self, other: Self) -> Result<Self, Error> {
        Ok(Self {
            count: count(self.count, other.count)?,
            sum: self.sum.merge(other.sum),
        })
    }

    /// Return the arithmetic mean; nonfinite inputs follow IEEE summation.
    pub fn finish(self) -> f64 {
        if self.sum.sum.is_finite() {
            ((self.sum.sum + self.sum.correction) / self.count as f64) * self.sum.scale
        } else {
            self.sum.sum
        }
    }
}

/// A scaled sum-of-squares Euclidean norm. Empty input yields positive zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct Norm {
    scale: f64,
    squares: f64,
}

impl Norm {
    /// Include one real component. NaN takes precedence over infinity.
    pub fn push(&mut self, value: f64) {
        if !value.is_finite() || !self.scale.is_finite() {
            self.scale = if value.is_nan() || self.scale.is_nan() {
                f64::NAN
            } else {
                f64::INFINITY
            };
        } else if value != 0. {
            let scale = scale(value);
            if scale > self.scale {
                let ratio = self.scale / scale;
                self.squares *= ratio * ratio;
                self.scale = scale;
            }
            self.squares += (value / self.scale).powi(2);
        }
    }

    /// Combine disjoint inputs, including empty inputs.
    pub fn merge(mut self, mut other: Self) -> Self {
        if !self.scale.is_finite() || !other.scale.is_finite() {
            self.push(other.scale);
        } else if other.scale != 0. {
            if self.scale < other.scale {
                std::mem::swap(&mut self, &mut other);
            }
            let ratio = other.scale / self.scale;
            self.squares += other.squares * ratio * ratio;
        }
        self
    }

    /// Return the Euclidean norm.
    pub fn finish(self) -> f64 {
        if self.scale.is_finite() {
            self.scale * self.squares.sqrt()
        } else {
            self.scale
        }
    }
}

/// Scaled, mergeable central moments for population standard deviation.
/// Empty input or any nonfinite input yields NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardDeviation {
    count: u64,
    scale: f64,
    mean: f64,
    squares: f64,
}

impl StandardDeviation {
    fn rescale(&mut self, scale: f64) {
        if scale > self.scale {
            let ratio = self.scale / scale;
            self.mean *= ratio;
            self.squares *= ratio * ratio;
            self.scale = scale;
        }
    }

    /// Include one value using scaled Welford central moments.
    /// Count overflow returns a bounds error without mutation.
    #[inline]
    pub fn push(&mut self, value: f64) -> Result<(), Error> {
        self.count = count(self.count, 1)?;
        if !value.is_finite() {
            self.mean = f64::NAN;
            self.squares = f64::NAN;
        } else {
            self.rescale(scale(value));
            let value = value / self.scale;
            let delta = value - self.mean;
            self.mean += delta / self.count as f64;
            self.squares += delta * (value - self.mean);
        }
        Ok(())
    }

    /// Combine disjoint inputs using scaled parallel central moments.
    /// Return a bounds error on count overflow.
    pub fn merge(self, other: Self) -> Result<Self, Error> {
        self.merge_shifted(other, 0.)
    }

    fn merge_shifted(mut self, mut other: Self, offset: f64) -> Result<Self, Error> {
        if self.count == 0 {
            return Ok(other);
        } else if other.count == 0 {
            return Ok(self);
        }

        let count = count(self.count, other.count)?;
        let mut normalization = self.scale.max(other.scale);
        if offset != 0. {
            normalization = normalization.max(scale(offset.abs()));
        }

        self.rescale(normalization);
        other.rescale(normalization);
        let mut delta = other.mean - self.mean;
        if offset != 0. {
            delta += offset / normalization;
        }

        let weight = other.count as f64 / count as f64;
        self.mean += delta * weight;
        self.squares += other.squares + delta * delta * self.count as f64 * weight;
        self.count = count;
        Ok(self)
    }

    /// Return the population standard deviation.
    pub fn finish(self) -> f64 {
        self.scale * (self.squares / self.count as f64).sqrt()
    }

    /// Return the norm of component deviations, combining scaled moments before
    /// final rounding. For complex inputs, supply real and imaginary components.
    #[cfg(feature = "complex")]
    fn finish_norm(components: [Self; 2]) -> f64 {
        let mut norm = Norm::default();
        for component in components {
            let squares = component.squares / component.count as f64;
            if squares.is_nan() {
                norm.push(f64::NAN);
            } else if squares != 0. {
                // A constant component's input scale must not hide variation
                // in another component.
                norm = norm.merge(Norm {
                    scale: component.scale,
                    squares,
                });
            }
        }
        norm.finish()
    }
}

/// Exact integer sum and count for a streaming mean.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegerMean<S> {
    count: u64,
    sum: S,
}

/// Central moments relative to an integer origin.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegerDeviation {
    origin: i128,
    moments: StandardDeviation,
}

macro_rules! real_norm {
    ($($ty:ty),+ $(,)?) => {$(
        impl BlockReduce<Host, $ty> for NormOp {
            type Block = [$ty];
            type Output = f64;
            type State = Norm;

            fn partial(&self, values: &Self::Block) -> Result<Norm, Error> {
                let mut norm = Norm::default();
                for &value in values {
                    norm.push(value as f64);
                }

                Ok(norm)
            }

            fn merge(left: Norm, right: Norm) -> Result<Norm, Error> {
                Ok(left.merge(right))
            }

            fn finish(state: Norm) -> f64 {
                state.finish()
            }
        }
    )+};
}

real_norm!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

macro_rules! integer_statistics {
    ($sum:ty; $($ty:ty),+ $(,)?) => {$(
        impl BlockReduce<Host, $ty> for MeanOp {
            type Block = [$ty];
            type Output = f64;
            type State = IntegerMean<$sum>;

            fn partial(&self, values: &Self::Block) -> Result<Self::State, Error> {
                let sum = values.iter().copied().map(<$sum>::from).sum();
                Ok(IntegerMean {
                    count: values.len() as u64,
                    sum,
                })
            }

            fn merge(left: Self::State, right: Self::State) -> Result<Self::State, Error> {
                // With a valid count, the sum of these 64-bit values fits the 128-bit state.
                Ok(IntegerMean {
                    count: count(left.count, right.count)?,
                    sum: left.sum + right.sum,
                })
            }

            fn finish(state: Self::State) -> f64 {
                state.sum as f64 / state.count as f64
            }
        }

        impl BlockReduce<Host, $ty> for StandardDeviationOp {
            type Block = [$ty];
            type Output = f64;
            type State = IntegerDeviation;

            fn partial(&self, values: &Self::Block) -> Result<IntegerDeviation, Error> {
                let origin = i128::from(values.first().copied().unwrap_or_default());
                let mut moments = StandardDeviation::default();
                for &value in values {
                    moments.push((i128::from(value) - origin) as f64)?;
                }

                Ok(IntegerDeviation { origin, moments })
            }

            fn merge(left: IntegerDeviation, right: IntegerDeviation) -> Result<IntegerDeviation, Error> {
                if left.moments.count == 0 {
                    return Ok(right);
                }

                let offset = (right.origin - left.origin) as f64;
                let moments = left.moments.merge_shifted(right.moments, offset)?;
                Ok(IntegerDeviation { origin: left.origin, moments })
            }

            fn finish(state: IntegerDeviation) -> f64 {
                state.moments.finish()
            }
        }
    )+};
}

integer_statistics!(i128; i8, i16, i32, i64);
integer_statistics!(u128; u8, u16, u32, u64);

macro_rules! float_statistics {
    ($($ty:ty),+ $(,)?) => {$(
        impl BlockReduce<Host, $ty> for MeanOp {
            type Block = [$ty];
            type Output = f64;
            type State = Mean;

            fn partial(&self, values: &Self::Block) -> Result<Mean, Error> {
                Ok(Mean::from_batch(values, |&value| f64::from(value)))
            }

            fn merge(left: Mean, right: Mean) -> Result<Mean, Error> {
                left.merge(right)
            }

            fn finish(state: Mean) -> f64 {
                state.finish()
            }
        }

        impl BlockReduce<Host, $ty> for StandardDeviationOp {
            type Block = [$ty];
            type Output = f64;
            type State = StandardDeviation;

            fn partial(&self, values: &Self::Block) -> Result<StandardDeviation, Error> {
                let mut moments = StandardDeviation::default();
                for &value in values {
                    moments.push(f64::from(value))?;
                }

                Ok(moments)
            }

            fn merge(left: Self::State, right: Self::State) -> Result<Self::State, Error> {
                left.merge(right)
            }

            fn finish(state: StandardDeviation) -> f64 {
                state.finish()
            }
        }
    )+};
}

float_statistics!(f32, f64);

#[cfg(feature = "complex")]
impl<F: Copy + Into<f64>> BlockReduce<Host, crate::complex::Complex<F>> for MeanOp
where
    crate::complex::Complex<F>: Number,
{
    type Block = [crate::complex::Complex<F>];
    type Output = crate::complex::Complex64;
    type State = [Mean; 2];

    // Keep the two compensated sums out of the caller's reduction loop.
    #[inline(never)]
    fn partial(&self, values: &Self::Block) -> Result<Self::State, Error> {
        Ok([
            Mean::from_batch(values, |value| value.re.into()),
            Mean::from_batch(values, |value| value.im.into()),
        ])
    }

    fn merge([re, im]: Self::State, [r, i]: Self::State) -> Result<Self::State, Error> {
        Ok([re.merge(r)?, im.merge(i)?])
    }

    fn finish([re, im]: Self::State) -> Self::Output {
        Self::Output::new(re.finish(), im.finish())
    }
}

#[cfg(feature = "complex")]
impl<F: Copy + Into<f64>> BlockReduce<Host, crate::complex::Complex<F>> for StandardDeviationOp
where
    crate::complex::Complex<F>: Number,
{
    type Block = [crate::complex::Complex<F>];
    type Output = f64;
    type State = [StandardDeviation; 2];

    fn partial(&self, values: &Self::Block) -> Result<Self::State, Error> {
        let mut re = StandardDeviation::default();
        let mut im = StandardDeviation::default();
        for value in values {
            re.push(value.re.into())?;
            im.push(value.im.into())?;
        }

        Ok([re, im])
    }

    fn merge([re, im]: Self::State, [r, i]: Self::State) -> Result<Self::State, Error> {
        Ok([re.merge(r)?, im.merge(i)?])
    }

    fn finish(state: Self::State) -> f64 {
        StandardDeviation::finish_norm(state)
    }
}

#[cfg(feature = "complex")]
impl<F: Copy + Into<f64>> BlockReduce<Host, crate::complex::Complex<F>> for NormOp
where
    crate::complex::Complex<F>: Number,
{
    type Block = [crate::complex::Complex<F>];
    type Output = f64;
    type State = Norm;

    fn partial(&self, values: &Self::Block) -> Result<Norm, Error> {
        let mut norm = Norm::default();
        for value in values {
            norm.push(value.re.into());
            norm.push(value.im.into());
        }

        Ok(norm)
    }

    fn merge(left: Norm, right: Norm) -> Result<Norm, Error> {
        Ok(left.merge(right))
    }

    fn finish(state: Norm) -> f64 {
        state.finish()
    }
}
