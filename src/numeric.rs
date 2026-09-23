//! Internal reduction identities, without changing the public finite bounds.
use crate::Real;
use number_general::Number;

pub(crate) fn minimum<T: Real>() -> T {
    match T::ZERO.into() {
        Number::Float(_) => T::cast_from(f64::NEG_INFINITY.into()),
        _ => T::MIN,
    }
}

pub(crate) fn maximum<T: Real>() -> T {
    match T::ZERO.into() {
        Number::Float(_) => T::cast_from(f64::INFINITY.into()),
        _ => T::MAX,
    }
}
