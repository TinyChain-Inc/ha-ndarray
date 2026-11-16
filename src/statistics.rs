//! Bounded, mergeable f64 statistical accumulators.
//!
//! Callers own traversal and conversion of integer/complex components. These
//! accumulators own numerical scaling; they allocate no buffers. Counts retain
//! exact u64 cardinality; conversion to f64 rounds statistical weights above 2^53.

#![forbid(unsafe_code)]

use crate::Error;

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

    // Borrowed batch projections can specialize constant zero components.
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
    /// Accumulate a borrowed batch with caller-owned component conversion.
    /// Validate cardinality once before evaluating any component.
    pub fn from_batch<T>(values: &[T], component: impl Fn(&T) -> f64) -> Result<Self, Error> {
        let count = u64::try_from(values.len())
            .map_err(|_| Error::bounds("statistics input cardinality overflow".into()))?;

        let mut state = Self {
            count,
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
    // Let adapters specialize the constant zero component of real inputs.
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
    pub fn merge(mut self, mut other: Self) -> Result<Self, Error> {
        if self.count == 0 {
            return Ok(other);
        } else if other.count == 0 {
            return Ok(self);
        }

        let count = count(self.count, other.count)?;
        let scale = self.scale.max(other.scale);
        self.rescale(scale);
        other.rescale(scale);
        let delta = other.mean - self.mean;
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
    pub fn finish_norm(components: impl IntoIterator<Item = Self>) -> f64 {
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
