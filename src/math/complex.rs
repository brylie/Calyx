//! First-principles Cartesian and Polar complex number mathematics.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust.

use std::fmt;
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A complex number represented in Cartesian form ($z = \text{re} + i \cdot \text{im}$).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex<T = f64> {
    /// Real component
    pub re: T,
    /// Imaginary component
    pub im: T,
}

impl Complex<f64> {
    /// Additive identity $0 + 0i$.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Multiplicative identity $1 + 0i$.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// Imaginary unit $0 + 1i$.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct a new complex number from Cartesian real and imaginary components.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Return the additive identity $0 + 0i$.
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    /// Return the multiplicative identity $1 + 0i$.
    #[inline]
    pub const fn one() -> Self {
        Self::ONE
    }

    /// Return the imaginary unit $i$.
    #[inline]
    pub const fn i() -> Self {
        Self::I
    }

    /// Construct a complex number from polar coordinates $(r, \theta)$:
    /// $z = r (\cos \theta + i \sin \theta)$.
    ///
    /// Preserves exact zero when $\theta = 0.0$, avoiding $\infty \times 0.0 = \text{NaN}$
    /// when $r = \infty$.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        if theta == 0.0 {
            Self::new(r, 0.0)
        } else {
            let (sin, cos) = theta.sin_cos();
            let re = if cos == 0.0 { 0.0 } else { r * cos };
            let im = if sin == 0.0 { 0.0 } else { r * sin };
            Self::new(re, im)
        }
    }

    /// Compute the Euclidean norm (magnitude) $\|z\| = \sqrt{\text{re}^2 + \text{im}^2}$.
    ///
    /// Utilizes `hypot` to prevent intermediate floating-point overflow or underflow.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Compute the squared norm $\|z\|^2 = \text{re}^2 + \text{im}^2$.
    ///
    /// Avoids computing the square root, making it optimal for energy/power calculations.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Alias for `norm_sq`.
    #[inline]
    pub fn norm_sqr(&self) -> f64 {
        self.norm_sq()
    }

    /// Compute the phase angle (argument) $\theta = \text{atan2}(\text{im}, \text{re}) \in (-\pi, \pi]$.
    #[inline]
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Compute the complex conjugate $z^* = \text{re} - i \cdot \text{im}$.
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Compute the exponential $e^z = e^{\text{re}} (\cos(\text{im}) + i \sin(\text{im}))$.
    ///
    /// Preserves exact IEEE 754 / C99 Annex G boundary invariants:
    /// - Pure real inputs ($z = x + 0.0i$) yield strictly real results ($e^x + 0.0i$),
    ///   preventing spurious `NaN` in imaginary parts when $x \ge 709.8$ ($e^x = \infty$).
    /// - Preserves signed zero (`-0.0`) in the imaginary component.
    /// - Handles negative infinity real part ($e^{-\infty + iy} = 0$).
    /// - Avoids indeterminate $\infty \times 0.0 = \text{NaN}$ floating-point forms.
    #[inline]
    pub fn exp(&self) -> Self {
        // Purely real argument shortcut: e^(x + 0i) = e^x + 0i
        // Eliminates inf * 0.0 = NaN when self.re >= 709.7827
        if self.im == 0.0 {
            return Self::new(self.re.exp(), self.im);
        }

        // Real negative infinity: e^(-inf + iy) = 0 for any finite or infinite y
        if self.re == f64::NEG_INFINITY {
            return if self.im.is_finite() {
                Self::new(0.0 * self.im.cos(), 0.0 * self.im.sin())
            } else {
                Self::zero()
            };
        }

        let exp_re = self.re.exp();
        let (im_sin, im_cos) = self.im.sin_cos();

        let re = if im_cos == 0.0 {
            0.0 * im_cos
        } else {
            exp_re * im_cos
        };

        let im = if im_sin == 0.0 {
            0.0 * im_sin
        } else {
            exp_re * im_sin
        };

        Self::new(re, im)
    }

    /// Compute the principal natural logarithm $\ln(z) = \ln(|z|) + i \arg(z)$.
    ///
    /// Avoids premature overflow when $\|z\| > \text{f64::MAX}$ but individual coordinates are finite.
    #[inline]
    pub fn ln(&self) -> Self {
        let norm = self.norm();
        let ln_r = if norm.is_infinite() && self.re.is_finite() && self.im.is_finite() {
            let max = self.re.abs().max(self.im.abs());
            let min = self.re.abs().min(self.im.abs());
            max.ln() + (1.0f64).hypot(min / max).ln()
        } else {
            norm.ln()
        };
        Self::new(ln_r, self.arg())
    }

    /// Raise this complex number to a real floating-point power $p$: $z^p = r^p e^{i p \theta}$.
    #[inline]
    pub fn powf(&self, p: f64) -> Self {
        let r = self.norm();
        if r == 0.0 {
            if p == 0.0 {
                Self::one()
            } else if p > 0.0 {
                Self::zero()
            } else {
                Self::new(f64::INFINITY, f64::NAN)
            }
        } else if self.im == 0.0 && self.re > 0.0 {
            // Pure positive real base: preserve imaginary zero strictly
            Self::new(self.re.powf(p), 0.0)
        } else {
            let theta = self.arg();
            let r_pow = r.powf(p);
            Self::from_polar(r_pow, p * theta)
        }
    }

    /// Raise this complex number to a complex power: $z^w = e^{w \ln z}$.
    #[inline]
    pub fn powc(&self, p: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            if p.re == 0.0 && p.im == 0.0 {
                Self::one()
            } else if p.re > 0.0 {
                Self::zero()
            } else {
                Self::new(f64::INFINITY, f64::NAN)
            }
        } else {
            (p * self.ln()).exp()
        }
    }

    /// Compute the square root of this complex number: $\sqrt{z} = z^{0.5}$.
    #[inline]
    pub fn sqrt(&self) -> Self {
        self.powf(0.5)
    }

    /// Compute the multiplicative inverse $z^{-1} = 1 / z$.
    #[inline]
    pub fn inv(&self) -> Self {
        Self::one() / *self
    }
}

// =========================================================================
// Operator Overload Implementations
// =========================================================================

impl Neg for Complex<f64> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Neg for &Complex<f64> {
    type Output = Complex<f64>;
    #[inline]
    fn neg(self) -> Complex<f64> {
        Complex::new(-self.re, -self.im)
    }
}

// Add: Complex + Complex
impl Add for Complex<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

// Sub: Complex - Complex
impl Sub for Complex<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

// Mul: Complex * Complex
impl Mul for Complex<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

// Div: Complex / Complex (Smith's algorithm for overflow protection)
impl Div for Complex<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let c = rhs.re;
        let d = rhs.im;
        if c.abs() >= d.abs() {
            if c == 0.0 && d == 0.0 {
                return Self::new(f64::NAN, f64::NAN);
            }
            let ratio = d / c;
            let denom = c + d * ratio;
            Self::new(
                (self.re + self.im * ratio) / denom,
                (self.im - self.re * ratio) / denom,
            )
        } else {
            let ratio = c / d;
            let denom = c * ratio + d;
            Self::new(
                (self.re * ratio + self.im) / denom,
                (self.im * ratio - self.re) / denom,
            )
        }
    }
}

// Scalar Add
impl Add<f64> for Complex<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self::new(self.re + rhs, self.im)
    }
}

impl Add<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    #[inline]
    fn add(self, rhs: Complex<f64>) -> Complex<f64> {
        Complex::new(self + rhs.re, rhs.im)
    }
}

// Scalar Sub
impl Sub<f64> for Complex<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self::new(self.re - rhs, self.im)
    }
}

impl Sub<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    #[inline]
    fn sub(self, rhs: Complex<f64>) -> Complex<f64> {
        Complex::new(self - rhs.re, -rhs.im)
    }
}

// Scalar Mul
impl Mul<f64> for Complex<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    #[inline]
    fn mul(self, rhs: Complex<f64>) -> Complex<f64> {
        Complex::new(self * rhs.re, self * rhs.im)
    }
}

// Scalar Div
impl Div<f64> for Complex<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Div<Complex<f64>> for f64 {
    type Output = Complex<f64>;
    #[inline]
    fn div(self, rhs: Complex<f64>) -> Complex<f64> {
        Complex::new(self, 0.0) / rhs
    }
}

// Reference Binary Operator Helpers
macro_rules! impl_ref_binops {
    ($trait:ident, $method:ident) => {
        impl<'a> $trait<&'a Complex<f64>> for Complex<f64> {
            type Output = Complex<f64>;
            #[inline]
            fn $method(self, rhs: &'a Complex<f64>) -> Self::Output {
                $trait::$method(self, *rhs)
            }
        }
        impl<'a> $trait<Complex<f64>> for &'a Complex<f64> {
            type Output = Complex<f64>;
            #[inline]
            fn $method(self, rhs: Complex<f64>) -> Self::Output {
                $trait::$method(*self, rhs)
            }
        }
        impl<'a, 'b> $trait<&'b Complex<f64>> for &'a Complex<f64> {
            type Output = Complex<f64>;
            #[inline]
            fn $method(self, rhs: &'b Complex<f64>) -> Self::Output {
                $trait::$method(*self, *rhs)
            }
        }
        impl<'a> $trait<f64> for &'a Complex<f64> {
            type Output = Complex<f64>;
            #[inline]
            fn $method(self, rhs: f64) -> Self::Output {
                $trait::$method(*self, rhs)
            }
        }
        impl<'a> $trait<&'a Complex<f64>> for f64 {
            type Output = Complex<f64>;
            #[inline]
            fn $method(self, rhs: &'a Complex<f64>) -> Self::Output {
                $trait::$method(self, *rhs)
            }
        }
    };
}

impl_ref_binops!(Add, add);
impl_ref_binops!(Sub, sub);
impl_ref_binops!(Mul, mul);
impl_ref_binops!(Div, div);

// In-place Assignment Operators
impl AddAssign for Complex<f64> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl AddAssign<&Complex<f64>> for Complex<f64> {
    #[inline]
    fn add_assign(&mut self, rhs: &Complex<f64>) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl AddAssign<f64> for Complex<f64> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.re += rhs;
    }
}

impl SubAssign for Complex<f64> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl SubAssign<&Complex<f64>> for Complex<f64> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Complex<f64>) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl SubAssign<f64> for Complex<f64> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.re -= rhs;
    }
}

impl MulAssign for Complex<f64> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Complex<f64>> for Complex<f64> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Complex<f64>) {
        *self = *self * *rhs;
    }
}

impl MulAssign<f64> for Complex<f64> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl DivAssign for Complex<f64> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl DivAssign<&Complex<f64>> for Complex<f64> {
    #[inline]
    fn div_assign(&mut self, rhs: &Complex<f64>) {
        *self = *self / *rhs;
    }
}

impl DivAssign<f64> for Complex<f64> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

impl Sum for Complex<f64> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Complex<f64>> for Complex<f64> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + *x)
    }
}

impl fmt::Display for Complex<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{} + {}i", self.re, self.im)
        } else {
            write!(f, "{} - {}i", self.re, -self.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_constants_and_constructors() {
        assert_eq!(Complex::ZERO, Complex::new(0.0, 0.0));
        assert_eq!(Complex::ONE, Complex::new(1.0, 0.0));
        assert_eq!(Complex::I, Complex::new(0.0, 1.0));
        assert_eq!(Complex::zero(), Complex::ZERO);
        assert_eq!(Complex::one(), Complex::ONE);
        assert_eq!(Complex::i(), Complex::I);
    }

    #[test]
    fn test_polar_roundtrip() {
        let r = 5.0;
        let theta = PI / 3.0;
        let c = Complex::from_polar(r, theta);
        assert!((c.re - 2.5).abs() < EPSILON);
        assert!((c.im - 5.0 * (PI / 3.0).sin()).abs() < EPSILON);
        assert!((c.norm() - r).abs() < EPSILON);
        assert!((c.arg() - theta).abs() < EPSILON);
    }

    #[test]
    fn test_norm_hypot_overflow_resistance() {
        let large = 1e200;
        let c = Complex::new(large, large);
        // Direct squaring (1e400) would overflow to f64::INFINITY.
        // hypot handles this gracefully.
        let expected = large * std::f64::consts::SQRT_2;
        assert!((c.norm() - expected).abs() / expected < 1e-14);
    }

    #[test]
    fn test_norm_sq() {
        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.norm_sq(), 25.0);
        assert_eq!(c.norm_sqr(), 25.0);
    }

    #[test]
    fn test_arg_edge_cases() {
        assert_eq!(Complex::ZERO.arg(), 0.0);
        assert_eq!(Complex::new(1.0, 0.0).arg(), 0.0);
        assert!((Complex::new(0.0, 1.0).arg() - PI / 2.0).abs() < EPSILON);
        assert!((Complex::new(-1.0, 0.0).arg() - PI).abs() < EPSILON);
        assert!((Complex::new(0.0, -1.0).arg() - (-PI / 2.0)).abs() < EPSILON);
    }

    #[test]
    fn test_conj() {
        let c = Complex::new(3.0, -4.0);
        assert_eq!(c.conj(), Complex::new(3.0, 4.0));
        assert_eq!(c.conj().conj(), c);
        assert_eq!((c * c.conj()).im, 0.0);
        assert_eq!((c * c.conj()).re, c.norm_sq());
    }

    #[test]
    fn test_exp_euler_identity() {
        // e^(i * pi) + 1 = 0
        let c = Complex::new(0.0, PI).exp();
        assert!((c.re - (-1.0)).abs() < EPSILON);
        assert!(c.im.abs() < EPSILON);
        let res = c + 1.0;
        assert!(res.norm() < EPSILON);
    }

    #[test]
    fn test_powf() {
        let c = Complex::new(1.0, 1.0);
        // (1 + i)^3 = -2 + 2i
        let p3 = c.powf(3.0);
        assert!((p3.re - (-2.0)).abs() < EPSILON);
        assert!((p3.im - 2.0).abs() < EPSILON);

        // z^0 == 1
        assert_eq!(c.powf(0.0), Complex::ONE);
        // 0^0 == 1
        assert_eq!(Complex::ZERO.powf(0.0), Complex::ONE);
        // 0^2 == 0
        assert_eq!(Complex::ZERO.powf(2.0), Complex::ZERO);
        // 0^(-1) == Inf
        let p_neg = Complex::ZERO.powf(-1.0);
        assert!(p_neg.re.is_infinite());
    }

    #[test]
    fn test_powc_and_sqrt_and_inv() {
        let c = Complex::new(0.0, 4.0);
        let s = c.sqrt();
        // sqrt(4i) = sqrt(2) + sqrt(2)i
        let expected_re = std::f64::consts::SQRT_2;
        assert!((s.re - expected_re).abs() < EPSILON);
        assert!((s.im - expected_re).abs() < EPSILON);

        let inv = c.inv();
        assert!((inv.re - 0.0).abs() < EPSILON);
        assert!((inv.im - (-0.25)).abs() < EPSILON);

        // powc
        let p = Complex::new(2.0, 0.0);
        let c_pow = c.powc(p);
        assert!((c_pow.re - (-16.0)).abs() < EPSILON);
        assert!(c_pow.im.abs() < EPSILON);

        // 0^0 == 1
        assert_eq!(Complex::ZERO.powc(Complex::ZERO), Complex::ONE);
        // 0^2 == 0
        assert_eq!(Complex::ZERO.powc(Complex::new(2.0, 0.0)), Complex::ZERO);
    }

    #[test]
    fn test_arithmetic_operators() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);

        // Add
        assert_eq!(a + b, Complex::new(4.0, 2.0));
        assert_eq!(&a + &b, Complex::new(4.0, 2.0));
        assert_eq!(a + 2.0, Complex::new(5.0, 4.0));
        assert_eq!(2.0 + a, Complex::new(5.0, 4.0));

        // Sub
        assert_eq!(a - b, Complex::new(2.0, 6.0));
        assert_eq!(a - 1.0, Complex::new(2.0, 4.0));
        assert_eq!(5.0 - a, Complex::new(2.0, -4.0));

        // Mul
        // (3+4i)(1-2i) = 3 - 6i + 4i - 8i^2 = 11 - 2i
        assert_eq!(a * b, Complex::new(11.0, -2.0));
        assert_eq!(a * 2.0, Complex::new(6.0, 8.0));
        assert_eq!(2.0 * a, Complex::new(6.0, 8.0));

        // Div: (3+4i)/(1-2i) = (3+4i)(1+2i)/5 = (3 - 8 + 10i)/5 = -1 + 2i
        let div = a / b;
        assert!((div.re - (-1.0)).abs() < EPSILON);
        assert!((div.im - 2.0).abs() < EPSILON);

        // Div by scalar
        assert_eq!(a / 2.0, Complex::new(1.5, 2.0));
        let scalar_div = 1.0 / a;
        assert!((scalar_div - a.inv()).norm() < EPSILON);

        // Neg
        assert_eq!(-a, Complex::new(-3.0, -4.0));
        assert_eq!(-&a, Complex::new(-3.0, -4.0));

        // Div by zero gives NaN
        let div_zero = a / Complex::ZERO;
        assert!(div_zero.re.is_nan() && div_zero.im.is_nan());
    }

    #[test]
    fn test_in_place_assignment() {
        let mut c = Complex::new(1.0, 2.0);
        c += Complex::new(3.0, 4.0);
        assert_eq!(c, Complex::new(4.0, 6.0));
        c += &Complex::new(1.0, 1.0);
        assert_eq!(c, Complex::new(5.0, 7.0));
        c -= 1.0;
        assert_eq!(c, Complex::new(4.0, 7.0));
        c -= &Complex::new(1.0, 2.0);
        assert_eq!(c, Complex::new(3.0, 5.0));
        c *= 2.0;
        assert_eq!(c, Complex::new(6.0, 10.0));
        c *= &Complex::new(1.0, 0.0);
        assert_eq!(c, Complex::new(6.0, 10.0));
        c /= 2.0;
        assert_eq!(c, Complex::new(3.0, 5.0));
        c /= &Complex::new(1.0, 0.0);
        assert_eq!(c, Complex::new(3.0, 5.0));
    }

    #[test]
    fn test_sum_iterator() {
        let list = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let sum: Complex<f64> = list.iter().sum();
        assert_eq!(sum, Complex::new(9.0, 12.0));
        let sum2: Complex<f64> = list.into_iter().sum();
        assert_eq!(sum2, Complex::new(9.0, 12.0));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Complex::new(3.0, 4.0)), "3 + 4i");
        assert_eq!(format!("{}", Complex::new(3.0, -4.0)), "3 - 4i");
        assert_eq!(format!("{}", Complex::new(0.0, 0.0)), "0 + 0i");
    }

    #[test]
    fn test_exp_large_real_no_nan() {
        // Values exceeding ln(f64::MAX) ~= 709.7827
        for &re in &[710.0, 800.0, 1000.0, f64::INFINITY] {
            let z = Complex::new(re, 0.0).exp();
            assert!(z.re.is_infinite() && z.re > 0.0, "Real part must be +inf");
            assert_eq!(z.im, 0.0, "Imaginary part must be strictly 0.0");
            assert!(!z.im.is_nan(), "Imaginary part must NOT be NaN");
        }
    }

    #[test]
    fn test_exp_signed_zero_preservation() {
        let z = Complex::new(800.0, -0.0).exp();
        assert!(z.re.is_infinite() && z.re > 0.0);
        assert_eq!(z.im, 0.0);
        assert!(z.im.is_sign_negative(), "Imaginary part must preserve negative zero");
    }

    #[test]
    fn test_exp_negative_infinity_behavior() {
        // e^(-inf + 0i) == 0.0 + 0.0i
        assert_eq!(Complex::new(f64::NEG_INFINITY, 0.0).exp(), Complex::ZERO);
        // e^(-inf + 1i) == 0.0 + 0.0i
        assert_eq!(Complex::new(f64::NEG_INFINITY, 1.0).exp(), Complex::ZERO);
        // e^(-inf + inf i) == 0.0 + 0.0i (C99 Annex G)
        assert_eq!(Complex::new(f64::NEG_INFINITY, f64::INFINITY).exp(), Complex::ZERO);
        // e^(-inf + NaN i) == 0.0 + 0.0i (C99 Annex G)
        assert_eq!(Complex::new(f64::NEG_INFINITY, f64::NAN).exp(), Complex::ZERO);
    }

    #[test]
    fn test_from_polar_and_powf_overflow_no_nan() {
        // from_polar with infinite radius at theta = 0
        let p = Complex::from_polar(f64::INFINITY, 0.0);
        assert!(p.re.is_infinite());
        assert_eq!(p.im, 0.0);
        assert!(!p.im.is_nan());

        // powf on large positive real base
        let z = Complex::new(10.0, 0.0).powf(400.0);
        assert!(z.re.is_infinite() && z.re > 0.0);
        assert_eq!(z.im, 0.0);
        assert!(!z.im.is_nan());
    }

    #[test]
    fn test_ln_large_coordinates_no_premature_overflow() {
        let z = Complex::new(1.5e308, 1.5e308);
        let log_z = z.ln();
        assert!(log_z.re.is_finite(), "ln magnitude must not overflow to inf");
        assert!((log_z.re - 709.948247).abs() < 1e-4);
        assert!((log_z.im - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }
}
