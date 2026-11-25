//! Bounded, mergeable statistical accumulators.
//!
//! Callers supply bounded batches; accumulators own conversion and scaling.
//! They allocate no buffers. Counts retain
//! exact u64 cardinality; conversion to f64 rounds statistical weights above 2^53.

#![forbid(unsafe_code)]

use crate::{Error, Number};

mod sealed {
    pub trait Sealed {}
}

/// Typed batch and merge operations for streaming statistics.
pub trait StatisticsElement: Number + sealed::Sealed {
    type Mean: Number;
    type MeanState: Copy + Default + Send + Sync;
    type DeviationState: Copy + Default + Send + Sync;

    fn mean_batch(values: &[Self]) -> Result<Self::MeanState, Error>;

    fn merge_mean(left: Self::MeanState, right: Self::MeanState) -> Result<Self::MeanState, Error>;

    fn finish_mean(state: Self::MeanState) -> Self::Mean;

    fn deviation_batch(values: &[Self]) -> Result<Self::DeviationState, Error>;

    fn merge_deviation(
        left: Self::DeviationState,
        right: Self::DeviationState,
    ) -> Result<Self::DeviationState, Error>;

    fn finish_deviation(state: Self::DeviationState) -> f64;

    fn norm_batch(values: &[Self]) -> Norm;
}

fn count(left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right)
        .ok_or_else(|| Error::bounds("statistics input cardinality overflow".into()))
}

fn batch_len(len: usize) -> Result<u64, Error> {
    u64::try_from(len).map_err(|_| Error::bounds("statistics input cardinality overflow".into()))
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
    fn from_batch<T>(values: &[T], component: impl Fn(&T) -> f64) -> Result<Self, Error> {
        let mut state = Self {
            count: batch_len(values.len())?,
            ..Self::default()
        };

        for value in values {
            state.sum.push(component(value));
        }

        Ok(state)
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
    () => {
        fn norm_batch(values: &[Self]) -> Norm {
            let mut norm = Norm::default();
            for &value in values {
                norm.push(value as f64);
            }

            norm
        }
    };
}

macro_rules! integer_statistics {
    ($sum:ty; $($ty:ty),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}

        impl StatisticsElement for $ty {
            type Mean = f64;
            type MeanState = IntegerMean<$sum>;
            type DeviationState = IntegerDeviation;

            fn mean_batch(values: &[Self]) -> Result<Self::MeanState, Error> {
                let count = batch_len(values.len())?;
                let sum = values.iter().copied().map(<$sum>::from).sum();
                Ok(IntegerMean { count, sum })
            }

            fn merge_mean(left: Self::MeanState, right: Self::MeanState) -> Result<Self::MeanState, Error> {
                // With a valid count, the sum of these 64-bit values fits the 128-bit state.
                Ok(IntegerMean {
                    count: count(left.count, right.count)?,
                    sum: left.sum + right.sum,
                })
            }

            fn finish_mean(state: Self::MeanState) -> f64 {
                state.sum as f64 / state.count as f64
            }

            fn deviation_batch(values: &[Self]) -> Result<Self::DeviationState, Error> {
                let origin = i128::from(values.first().copied().unwrap_or_default());
                let mut moments = StandardDeviation::default();
                for &value in values {
                    moments.push((i128::from(value) - origin) as f64)?;
                }

                Ok(IntegerDeviation { origin, moments })
            }

            fn merge_deviation(
                left: Self::DeviationState,
                right: Self::DeviationState,
            ) -> Result<Self::DeviationState, Error> {
                if left.moments.count == 0 {
                    return Ok(right);
                }

                let offset = (right.origin - left.origin) as f64;
                let moments = left.moments.merge_shifted(right.moments, offset)?;
                Ok(IntegerDeviation { origin: left.origin, moments })
            }

            fn finish_deviation(state: Self::DeviationState) -> f64 {
                state.moments.finish()
            }

            real_norm!();
        }
    )+};
}

integer_statistics!(i128; i8, i16, i32, i64);
integer_statistics!(u128; u8, u16, u32, u64);

macro_rules! float_statistics {
    ($($ty:ty),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}

        impl StatisticsElement for $ty {
            type Mean = f64;
            type MeanState = Mean;
            type DeviationState = StandardDeviation;

            fn mean_batch(values: &[Self]) -> Result<Self::MeanState, Error> {
                Mean::from_batch(values, |value| f64::from(*value))
            }

            fn merge_mean(left: Self::MeanState, right: Self::MeanState) -> Result<Self::MeanState, Error> {
                left.merge(right)
            }

            fn finish_mean(state: Self::MeanState) -> f64 {
                state.finish()
            }

            fn deviation_batch(values: &[Self]) -> Result<Self::DeviationState, Error> {
                let mut moments = StandardDeviation::default();
                for &value in values {
                    moments.push(f64::from(value))?;
                }

                Ok(moments)
            }

            fn merge_deviation(
                left: Self::DeviationState,
                right: Self::DeviationState,
            ) -> Result<Self::DeviationState, Error> {
                left.merge(right)
            }

            fn finish_deviation(state: Self::DeviationState) -> f64 {
                state.finish()
            }

            real_norm!();
        }
    )+};
}

float_statistics!(f32, f64);

#[cfg(feature = "complex")]
impl<F> sealed::Sealed for crate::complex::Complex<F> where Self: Number {}

#[cfg(feature = "complex")]
impl<F: Copy + Into<f64>> StatisticsElement for crate::complex::Complex<F>
where
    Self: Number,
{
    type Mean = crate::complex::Complex64;
    type MeanState = [Mean; 2];
    type DeviationState = [StandardDeviation; 2];

    fn mean_batch(values: &[Self]) -> Result<Self::MeanState, Error> {
        Ok([
            Mean::from_batch(values, |value| value.re.into())?,
            Mean::from_batch(values, |value| value.im.into())?,
        ])
    }

    fn merge_mean(
        [re, im]: Self::MeanState,
        [r, i]: Self::MeanState,
    ) -> Result<Self::MeanState, Error> {
        Ok([re.merge(r)?, im.merge(i)?])
    }

    fn finish_mean([re, im]: Self::MeanState) -> Self::Mean {
        Self::Mean::new(re.finish(), im.finish())
    }

    fn deviation_batch(values: &[Self]) -> Result<Self::DeviationState, Error> {
        let mut re = StandardDeviation::default();
        let mut im = StandardDeviation::default();
        for value in values {
            re.push(value.re.into())?;
            im.push(value.im.into())?;
        }

        Ok([re, im])
    }

    fn merge_deviation(
        [re, im]: Self::DeviationState,
        [r, i]: Self::DeviationState,
    ) -> Result<Self::DeviationState, Error> {
        Ok([re.merge(r)?, im.merge(i)?])
    }

    fn finish_deviation(state: Self::DeviationState) -> f64 {
        StandardDeviation::finish_norm(state)
    }

    fn norm_batch(values: &[Self]) -> Norm {
        let mut norm = Norm::default();
        for value in values {
            norm.push(value.re.into());
            norm.push(value.im.into());
        }

        norm
    }
}
