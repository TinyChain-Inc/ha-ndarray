//! Test-only certified references. Bounds enclose every intermediate operation.
use rug::{float::Round, ops::*, Float, Integer, Rational};

#[derive(Clone, Debug)]
pub struct Interval {
    pub lo: Float,
    pub hi: Float,
}

impl Interval {
    pub fn exact(p: u32, value: &Rational) -> Self {
        Self {
            lo: Float::with_val_round(p, value, Round::Down).0,
            hi: Float::with_val_round(p, value, Round::Up).0,
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let p = self.lo.prec();
        Self {
            lo: Float::with_val_round(p, &self.lo + &rhs.lo, Round::Down).0,
            hi: Float::with_val_round(p, &self.hi + &rhs.hi, Round::Up).0,
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            lo: -self.hi.clone(),
            hi: -self.lo.clone(),
        }
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let p = self.lo.prec();
        let mut lo = Float::with_val(p, f64::INFINITY);
        let mut hi = Float::with_val(p, f64::NEG_INFINITY);
        for a in [&self.lo, &self.hi] {
            for b in [&rhs.lo, &rhs.hi] {
                let lower = Float::with_val_round(p, a * b, Round::Down).0;
                let upper = Float::with_val_round(p, a * b, Round::Up).0;
                if lower < lo {
                    lo = lower;
                }
                if upper > hi {
                    hi = upper;
                }
            }
        }
        Self { lo, hi }
    }

    pub fn div(&self, rhs: &Self) -> Self {
        assert!(
            rhs.lo > 0 || rhs.hi < 0,
            "reference division crosses zero: {rhs:?}"
        );
        let p = self.lo.prec();
        let reciprocal = Self {
            lo: Float::with_val_round(p, 1 / &rhs.hi, Round::Down).0,
            hi: Float::with_val_round(p, 1 / &rhs.lo, Round::Up).0,
        };
        self.mul(&reciprocal)
    }

    pub fn ln(&self) -> Self {
        let mut lo = self.lo.clone();
        let mut hi = self.hi.clone();
        lo.ln_round(Round::Down);
        hi.ln_round(Round::Up);
        Self { lo, hi }
    }

    /// Lipschitz enclosure: |sin(x)-sin(a)| and |cos(x)-cos(a)| <= |x-a|.
    /// This also handles intervals spanning stationary points without range heuristics.
    #[cfg(feature = "complex")]
    pub fn trig(&self, cosine: bool) -> Self {
        let p = self.lo.prec();
        let width = Float::with_val_round(p, &self.hi - &self.lo, Round::Up).0;
        let mut lo = self.lo.clone();
        let mut hi = self.lo.clone();
        if cosine {
            lo.cos_round(Round::Down);
            hi.cos_round(Round::Up);
        } else {
            lo.sin_round(Round::Down);
            hi.sin_round(Round::Up);
        }
        lo.sub_assign_round(&width, Round::Down);
        hi.add_assign_round(&width, Round::Up);
        Self { lo, hi }
    }
}

fn rounded(x: &Float, bits: u32) -> f64 {
    match bits {
        32 => x.to_f32_round(Round::Nearest) as f64,
        64 => x.to_f64_round(Round::Nearest),
        _ => panic!("invalid destination precision {bits}"),
    }
}

fn agreement(bounds: &Interval, bits: u32) -> Option<f64> {
    let lo = rounded(&bounds.lo, bits);
    let hi = rounded(&bounds.hi, bits);
    if lo.to_bits() == hi.to_bits() || (lo.is_nan() && hi.is_nan()) {
        Some(lo)
    } else {
        None
    }
}

pub fn certify(
    label: &str,
    bits: u32,
    mut bounds: impl FnMut(u32) -> Interval,
) -> Result<f64, String> {
    let mut p = 256;
    loop {
        let enclosure = bounds(p);
        if let Some(value) = agreement(&enclosure, bits) {
            return Ok(value);
        }
        if p == 4096 {
            return Err(format!(
                "ambiguous reference for {label}, f{bits}, at {p} bits: {enclosure:?}"
            ));
        }
        p *= 2;
    }
}

fn unary(mut x: Float, op: &str, round: Round) -> Float {
    match op {
        "abs" => x.abs_mut(),
        "round" => x.round_mut(),
        "exp" => {
            x.exp_round(round);
        }
        "ln" => {
            x.ln_round(round);
        }
        "sin" => {
            x.sin_round(round);
        }
        "cos" => {
            x.cos_round(round);
        }
        "tan" => {
            x.tan_round(round);
        }
        "asin" => {
            x.asin_round(round);
        }
        "acos" => {
            x.acos_round(round);
        }
        "atan" => {
            x.atan_round(round);
        }
        "sinh" => {
            x.sinh_round(round);
        }
        "cosh" => {
            x.cosh_round(round);
        }
        "tanh" => {
            x.tanh_round(round);
        }
        _ => panic!("unknown reference operation {op}"),
    }
    x
}

pub fn real(x: f64, bits: u32, op: &str) -> f64 {
    certify(&format!("{op}({x:?})"), bits, |p| Interval {
        lo: unary(Float::with_val(p, x), op, Round::Down),
        hi: unary(Float::with_val(p, x), op, Round::Up),
    })
    .unwrap()
}

fn binary(mut x: Float, y: &Float, op: &str, round: Round) -> Float {
    match op {
        "add" => {
            x.add_assign_round(y, round);
        }
        "sub" => {
            x.sub_assign_round(y, round);
        }
        "mul" => {
            x.mul_assign_round(y, round);
        }
        "div" => {
            x.div_assign_round(y, round);
        }
        "rem" => {
            x.rem_assign_round(y, round);
        }
        "pow" => {
            x.pow_assign_round(y, round);
        }
        _ => panic!("unknown reference operation {op}"),
    }
    x
}

pub fn real_binary(x: f64, y: f64, bits: u32, op: &str) -> f64 {
    certify(&format!("{op}({x:?},{y:?})"), bits, |p| {
        let a = Float::with_val(p, x);
        let b = Float::with_val(p, y);
        if op == "log" {
            return Interval {
                lo: a.clone(),
                hi: a,
            }
            .ln()
            .div(
                &Interval {
                    lo: b.clone(),
                    hi: b,
                }
                .ln(),
            );
        }
        let lo = binary(a.clone(), &b, op, Round::Down);
        let hi = binary(a.clone(), &b, op, Round::Up);
        // Exact cancellation's zero sign is specified by nearest rounding,
        // not by the directed-rounding endpoints used for finite error bounds.
        if lo.is_zero() && hi.is_zero() {
            let exact = binary(a, &b, op, Round::Nearest);
            Interval {
                lo: exact.clone(),
                hi: exact,
            }
        } else {
            Interval { lo, hi }
        }
    })
    .unwrap()
}

#[cfg(feature = "complex")]
fn complex_eval(mut a: rug::Complex, b: Option<&rug::Complex>, op: &str, r: Round) -> rug::Complex {
    let r = (r, r);
    if let Some(b) = b {
        match op {
            "add" => {
                a.add_assign_round(b, r);
            }
            "sub" => {
                a.sub_assign_round(b, r);
            }
            "mul" => {
                a.mul_assign_round(b, r);
            }
            "div" => {
                a.div_assign_round(b, r);
            }
            "pow" => {
                a.pow_assign_round(b, r);
            }
            _ => panic!("unknown complex operation {op}"),
        }
    } else {
        match op {
            "exp" => {
                a.exp_round(r);
            }
            "ln" => {
                a.ln_round(r);
            }
            "sin" => {
                a.sin_round(r);
            }
            "cos" => {
                a.cos_round(r);
            }
            "tan" => {
                a.tan_round(r);
            }
            "asin" => {
                a.asin_round(r);
            }
            "acos" => {
                a.acos_round(r);
            }
            "atan" => {
                a.atan_round(r);
            }
            "sinh" => {
                a.sinh_round(r);
            }
            "cosh" => {
                a.cosh_round(r);
            }
            "tanh" => {
                a.tanh_round(r);
            }
            _ => panic!("unknown complex operation {op}"),
        }
    }
    a
}

#[cfg(feature = "complex")]
fn complex_bounds(a: rug::Complex, b: Option<&rug::Complex>, op: &str) -> [Interval; 2] {
    let lo = complex_eval(a.clone(), b, op, Round::Down);
    let hi = complex_eval(a, b, op, Round::Up);
    [
        Interval {
            lo: lo.real().clone(),
            hi: hi.real().clone(),
        },
        Interval {
            lo: lo.imag().clone(),
            hi: hi.imag().clone(),
        },
    ]
}

#[cfg(feature = "complex")]
pub fn complex(a: (f64, f64), b: Option<(f64, f64)>, bits: u32, op: &str) -> (f64, f64) {
    let build = |p| {
        let a = rug::Complex::with_val(p, a);
        let b = b.map(|b| rug::Complex::with_val(p, b));
        if op == "log" {
            let [ar, ai] = complex_bounds(a, None, "ln");
            let [br, bi] = complex_bounds(b.unwrap(), None, "ln");
            let denominator = br.mul(&br).add(&bi.mul(&bi));
            [
                ar.mul(&br).add(&ai.mul(&bi)).div(&denominator),
                ai.mul(&br).sub(&ar.mul(&bi)).div(&denominator),
            ]
        } else {
            complex_bounds(a, b.as_ref(), op)
        }
    };
    let label = format!("complex {op}({a:?},{b:?})");
    (
        certify(&format!("{label}.re"), bits, |p| build(p)[0].clone()).unwrap(),
        certify(&format!("{label}.im"), bits, |p| build(p)[1].clone()).unwrap(),
    )
}

pub fn rational(x: f64) -> Rational {
    Rational::from_f64(x).expect("finite exact input")
}

/// Exact reduction reference, using dyadic source values without intermediate rounding.
#[derive(Clone, Debug)]
pub struct ExactComplex {
    pub re: Rational,
    pub im: Rational,
}
impl ExactComplex {
    pub fn new(re: f64, im: f64) -> Self {
        Self {
            re: rational(re),
            im: rational(im),
        }
    }
    pub fn add(&self, b: &Self) -> Self {
        Self {
            re: self.re.clone() + &b.re,
            im: self.im.clone() + &b.im,
        }
    }
    pub fn mul(&self, b: &Self) -> Self {
        Self {
            re: self.re.clone() * &b.re - self.im.clone() * &b.im,
            im: self.re.clone() * &b.im + self.im.clone() * &b.re,
        }
    }
    pub fn norm_lower(&self, p: u32) -> Float {
        let squared = self.re.clone() * &self.re + self.im.clone() * &self.im;
        let mut n = Float::with_val_round(p, squared, Round::Down).0;
        n.sqrt_round(Round::Down);
        n
    }
}

pub fn gamma(n: usize, complex: bool, bits: u32) -> Rational {
    let u = Rational::from((
        Integer::from(1),
        Integer::from(1) << if bits == 32 { 24 } else { 53 },
    ));
    let ku = u * Integer::from(n * if complex { 32 } else { 8 });
    assert!(ku < 1, "gamma is undefined for ku>=1");
    ku.clone() / (Rational::from(1) - ku)
}

/// The scale is a lower enclosure, so reference rounding cannot widen the contract.
pub fn check_aggregate(
    actual: (f64, f64),
    expected: &ExactComplex,
    terms: &[ExactComplex],
    n: usize,
    complex: bool,
    bits: u32,
    product: bool,
) {
    assert!(actual.0.is_finite() && actual.1.is_finite());
    let scale = if product {
        expected.norm_lower(256)
    } else {
        let mut scale = Float::with_val(256, 0);
        for term in terms {
            scale.add_assign_round(term.norm_lower(256), Round::Down);
        }
        scale
    };
    let bound = gamma(n, complex, bits) * scale.to_rational().unwrap();
    for (actual, expected) in [(actual.0, &expected.re), (actual.1, &expected.im)] {
        let error = (rational(actual) - expected).abs();
        assert!(
            error <= bound,
            "aggregate f{bits}, N={n}: {actual} differs from {expected} by {error}, bound {bound}"
        );
    }
}

#[cfg(feature = "complex")]
pub fn fft_bound(values: &[(f64, f64)], bits: u32) -> Rational {
    let mut scale = Float::with_val(256, 0);
    for &(re, im) in values {
        scale.add_assign_round(ExactComplex::new(re, im).norm_lower(256), Round::Down);
    }
    gamma(values.len(), true, bits) * scale.to_rational().unwrap()
}

/// Each angle, trigonometric result, product, and sum is enclosed separately.
#[cfg(feature = "complex")]
fn dft(values: &[(f64, f64)], k: usize, inverse: bool, p: u32) -> [Interval; 2] {
    let pi = Interval {
        lo: Float::with_val_round(p, rug::float::Constant::Pi, Round::Down).0,
        hi: Float::with_val_round(p, rug::float::Constant::Pi, Round::Up).0,
    };
    let mut re = Interval::exact(p, &Rational::from(0));
    let mut im = re.clone();
    for (j, &(ar, ai)) in values.iter().enumerate() {
        let factor = Rational::from((
            Integer::from(2 * k * j) * if inverse { 1 } else { -1 },
            Integer::from(values.len()),
        ));
        let angle = pi.mul(&Interval::exact(p, &factor));
        let cos = angle.trig(true);
        let sin = angle.trig(false);
        let ar = Interval::exact(p, &rational(ar));
        let ai = Interval::exact(p, &rational(ai));
        re = re.add(&ar.mul(&cos).sub(&ai.mul(&sin)));
        im = im.add(&ar.mul(&sin).add(&ai.mul(&cos)));
    }
    [re, im]
}

#[cfg(feature = "complex")]
pub fn check_dft(actual: (f64, f64), values: &[(f64, f64)], k: usize, inverse: bool, bits: u32) {
    let bound = fft_bound(values, bits);
    let mut p = 256;
    loop {
        let reference = dft(values, k, inverse, p);
        let mut accepted = true;
        for (value, range) in [actual.0, actual.1].into_iter().zip(reference) {
            let value = rational(value);
            let lo = range.lo.to_rational().unwrap();
            let hi = range.hi.to_rational().unwrap();
            let far = (value.clone() - &lo).abs().max((value.clone() - &hi).abs());
            if far <= bound {
                continue;
            }
            let near = if value < lo {
                lo - value
            } else if value > hi {
                value - hi
            } else {
                Rational::from(0)
            };
            assert!(
                near <= bound,
                "DFT f{bits}, N={}, k={k}, inverse={inverse}, actual={actual:?} exceeds {bound}",
                values.len()
            );
            accepted = false;
        }
        if accepted {
            return;
        }
        assert!(
            p < 4096,
            "unresolved DFT reference at {p} bits: values={values:?}, k={k}, inverse={inverse}"
        );
        p *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pow2(exp: usize) -> Rational {
        Rational::from((Integer::from(1), Integer::from(1) << exp))
    }
    #[test]
    fn halfway_direct_f32_and_subnormal_rounding() {
        let half = Rational::from(1) + pow2(53);
        assert_eq!(
            certify("f64 halfway", 64, |p| Interval::exact(p, &half)).unwrap(),
            1.
        );
        let above32 = Rational::from(1) + pow2(24) + pow2(80);
        assert_eq!(
            (certify("f32 above halfway", 32, |p| Interval::exact(p, &above32)).unwrap() as f32)
                .to_bits(),
            1f32.to_bits() + 1
        );
        let half_sub = pow2(1075);
        assert_eq!(
            certify("subnormal halfway", 64, |p| Interval::exact(p, &half_sub))
                .unwrap()
                .to_bits(),
            0
        );
        let above_sub = half_sub + pow2(1200);
        assert_eq!(
            certify("above subnormal halfway", 64, |p| Interval::exact(
                p, &above_sub
            ))
            .unwrap()
            .to_bits(),
            1
        );
    }
    #[test]
    fn precision_escalates_and_ambiguity_fails_closed() {
        let value = Rational::from(1) + pow2(53) + pow2(300);
        let mut precisions = Vec::new();
        let rounded = certify("requires 512 bits", 64, |p| {
            precisions.push(p);
            Interval::exact(p, &value)
        })
        .unwrap();
        assert_eq!(rounded.to_bits(), 1f64.to_bits() + 1);
        assert_eq!(precisions, vec![256, 512]);
        let error = certify("deliberately unresolved input", 64, |p| Interval {
            lo: Float::with_val(p, 1),
            hi: Float::with_val(p, 2),
        })
        .unwrap_err();
        assert!(error.contains("deliberately unresolved input") && error.contains("4096"));
    }
    #[test]
    fn composed_bounds_preserve_cancellation() {
        let value = certify("ln(2)/ln(2)", 64, |p| {
            let x = Interval::exact(p, &Rational::from(2)).ln();
            x.div(&x)
        })
        .unwrap();
        assert_eq!(value, 1.);
        let p = 256;
        let a = Interval::exact(p, &(Rational::from(1) + pow2(300)));
        let b = Interval::exact(p, &Rational::from(1));
        let diff = a.sub(&b);
        assert!(diff.lo.to_rational().unwrap() <= pow2(300));
        assert!(diff.hi.to_rational().unwrap() >= pow2(300));
    }
    #[cfg(feature = "complex")]
    #[test]
    fn complex_components_are_certified_independently() {
        let z = complex((0., 0.), None, 32, "exp");
        assert_eq!(z, (1., 0.));
        let z = complex((0.5, 0.25), Some((0.5, 0.25)), 64, "mul");
        assert_eq!(z, (0.1875, 0.25));
    }
}
