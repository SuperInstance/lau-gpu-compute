//! SIMD-style vector operations — f32x4, f32x8 packing.

use serde::{Serialize, Deserialize};

/// SIMD-style f32x4 (128-bit packed)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct f32x4(pub [f32; 4]);

impl f32x4 {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self([a, b, c, d])
    }

    pub fn splat(val: f32) -> Self {
        Self([val, val, val, val])
    }

    pub fn zero() -> Self {
        Self([0.0; 4])
    }

    pub fn add(&self, other: &Self) -> Self {
        Self([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
            self.0[3] + other.0[3],
        ])
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
            self.0[3] - other.0[3],
        ])
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self([
            self.0[0] * other.0[0],
            self.0[1] * other.0[1],
            self.0[2] * other.0[2],
            self.0[3] * other.0[3],
        ])
    }

    pub fn div(&self, other: &Self) -> Self {
        Self([
            self.0[0] / other.0[0],
            self.0[1] / other.0[1],
            self.0[2] / other.0[2],
            self.0[3] / other.0[3],
        ])
    }

    /// Horizontal sum
    pub fn hsum(&self) -> f32 {
        self.0[0] + self.0[1] + self.0[2] + self.0[3]
    }

    /// Dot product (multiply + horizontal sum)
    pub fn dot(&self, other: &Self) -> f32 {
        self.mul(other).hsum()
    }

    /// Fused multiply-add: self * a + b
    pub fn fma(&self, a: &Self, b: &Self) -> Self {
        self.mul(a).add(b)
    }

    /// Max of each lane
    pub fn max(&self, other: &Self) -> Self {
        Self([
            self.0[0].max(other.0[0]),
            self.0[1].max(other.0[1]),
            self.0[2].max(other.0[2]),
            self.0[3].max(other.0[3]),
        ])
    }

    /// Min of each lane
    pub fn min(&self, other: &Self) -> Self {
        Self([
            self.0[0].min(other.0[0]),
            self.0[1].min(other.0[1]),
            self.0[2].min(other.0[2]),
            self.0[3].min(other.0[3]),
        ])
    }

    /// Pack from slice (takes 4 elements)
    pub fn load(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4);
        Self([slice[0], slice[1], slice[2], slice[3]])
    }

    /// Unpack to slice
    pub fn store(&self, slice: &mut [f32]) {
        assert!(slice.len() >= 4);
        slice[..4].copy_from_slice(&self.0);
    }
}

/// SIMD-style f32x8 (256-bit packed)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct f32x8(pub [f32; 8]);

impl f32x8 {
    pub fn new(vals: [f32; 8]) -> Self {
        Self(vals)
    }

    pub fn splat(val: f32) -> Self {
        Self([val; 8])
    }

    pub fn zero() -> Self {
        Self([0.0; 8])
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut r = [0.0f32; 8];
        for i in 0..8 { r[i] = self.0[i] + other.0[i]; }
        Self(r)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut r = [0.0f32; 8];
        for i in 0..8 { r[i] = self.0[i] * other.0[i]; }
        Self(r)
    }

    pub fn hsum(&self) -> f32 {
        self.0.iter().sum()
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.mul(other).hsum()
    }

    pub fn load(slice: &[f32]) -> Self {
        assert!(slice.len() >= 8);
        let mut vals = [0.0f32; 8];
        vals.copy_from_slice(&slice[..8]);
        Self(vals)
    }

    pub fn store(&self, slice: &mut [f32]) {
        assert!(slice.len() >= 8);
        slice[..8].copy_from_slice(&self.0);
    }
}

/// Trait for SIMD-style operations
pub trait SimdOps: Clone {
    fn splat(val: f32) -> Self;
    fn zero() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn hsum(&self) -> f32;
    fn dot(&self, other: &Self) -> f32 { self.mul(other).hsum() }
}

impl SimdOps for f32x4 {
    fn splat(val: f32) -> Self { f32x4::splat(val) }
    fn zero() -> Self { f32x4::zero() }
    fn add(&self, other: &Self) -> Self { self.add(other) }
    fn mul(&self, other: &Self) -> Self { self.mul(other) }
    fn hsum(&self) -> f32 { self.hsum() }
}

impl SimdOps for f32x8 {
    fn splat(val: f32) -> Self { f32x8::splat(val) }
    fn zero() -> Self { f32x8::zero() }
    fn add(&self, other: &Self) -> Self { self.add(other) }
    fn mul(&self, other: &Self) -> Self { self.mul(other) }
    fn hsum(&self) -> f32 { self.hsum() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32x4_basic() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(&b);
        assert_eq!(c.0, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_f32x4_mul() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::splat(2.0);
        let c = a.mul(&b);
        assert_eq!(c.0, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_f32x4_hsum() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.hsum(), 10.0);
    }

    #[test]
    fn test_f32x4_dot() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.dot(&b), 10.0);
    }

    #[test]
    fn test_f32x4_load_store() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = f32x4::load(&data);
        let mut out = [0.0f32; 6];
        v.store(&mut out);
        assert_eq!(out[..4], data[..4]);
    }

    #[test]
    fn test_f32x8_basic() {
        let a = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a.hsum(), 36.0);
    }

    #[test]
    fn test_f32x8_dot() {
        let a = f32x8::splat(2.0);
        let b = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a.dot(&b), 72.0);
    }

    #[test]
    fn test_f32x8_load_store_roundtrip() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = f32x8::load(&data);
        let mut out = [0.0f32; 9];
        v.store(&mut out);
        assert_eq!(&out[..8], &data[..8]);
    }

    #[test]
    fn test_f32x4_max_min() {
        let a = f32x4::new(1.0, 5.0, 3.0, 7.0);
        let b = f32x4::new(4.0, 2.0, 6.0, 1.0);
        let mx = a.max(&b);
        let mn = a.min(&b);
        assert_eq!(mx.0, [4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mn.0, [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn test_f32x4_fma() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::splat(2.0);
        let c = f32x4::splat(10.0);
        let result = a.fma(&b, &c);
        assert_eq!(result.0, [12.0, 14.0, 16.0, 18.0]);
    }
}
