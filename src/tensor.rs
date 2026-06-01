//! Tensor operations designed for parallelism.

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Mul, Div};

/// Multi-dimensional tensor with f32 data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Clone + Default> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Clone + Default> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "Data length mismatch");
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![T::default(); size],
        }
    }

    pub fn from_scalar(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; size],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            idx += indices[i] * stride;
            stride *= self.shape[i];
        }
        idx
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        &self.data[self.flat_index(indices)]
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.flat_index(indices);
        &mut self.data[idx]
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(new_size, self.data.len());
        Self {
            shape: new_shape,
            data: self.data.clone(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl Tensor<f32> {
    /// Element-wise map (parallel-ready pattern)
    pub fn elementwise<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Element-wise binary op (parallel-ready pattern)
    pub fn elementwise_binary<F: Fn(f32, f32) -> f32>(&self, other: &Self, f: F) -> Self {
        assert_eq!(self.shape, other.shape);
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect(),
        }
    }

    /// Reduce along an axis
    pub fn reduce_axis<F: Fn(f32, f32) -> f32>(&self, axis: usize, init: f32, f: F) -> Self {
        assert!(axis < self.shape.len());
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![init; out_size];

        // Iterate over all elements
        for (i, &val) in self.data.iter().enumerate() {
            // Convert flat index to multi-dim index
            let mut indices = Vec::with_capacity(self.shape.len());
            let mut rem = i;
            for d in 0..self.shape.len() {
                let stride: usize = self.shape[d + 1..].iter().product();
                indices.push(rem / stride);
                rem %= stride;
            }
            // Compute output index (collapse axis)
            let mut out_indices = indices.clone();
            out_indices[axis] = 0;
            let mut out_flat = 0;
            let mut stride = 1;
            for d in (0..out_shape.len()).rev() {
                out_flat += out_indices[d] * stride;
                stride *= out_shape[d];
            }
            result[out_flat] = f(result[out_flat], val);
        }

        Self { shape: out_shape, data: result }
    }

    /// Full reduce (sum all elements)
    pub fn reduce_sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Full reduce (max)
    pub fn reduce_max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Full reduce (min)
    pub fn reduce_min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Matrix multiply (2D tensors)
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.rank(), 2);
        assert_eq!(other.rank(), 2);
        assert_eq!(self.shape[1], other.shape[0]);
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Self { shape: vec![m, n], data }
    }
}

/// Tensor operation kinds for kernel dispatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorOp {
    ElementWise { op: ElementWiseOp },
    Reduce { axis: usize, op: ReduceOp },
    Scan { axis: usize, op: ScanOp },
    Sort { axis: usize, ascending: bool },
    MatMul,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementWiseOp {
    Add, Sub, Mul, Div,
    Relu, Sigmoid, Tanh,
    Exp, Log, Sqrt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum, Max, Min, Mean, Prod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScanOp {
    PrefixSum,
}

// Arithmetic impls for Tensor<f32>
impl Add for &Tensor<f32> {
    type Output = Tensor<f32>;
    fn add(self, other: &Tensor<f32>) -> Tensor<f32> {
        self.elementwise_binary(other, |a, b| a + b)
    }
}

impl Sub for &Tensor<f32> {
    type Output = Tensor<f32>;
    fn sub(self, other: &Tensor<f32>) -> Tensor<f32> {
        self.elementwise_binary(other, |a, b| a - b)
    }
}

impl Mul for &Tensor<f32> {
    type Output = Tensor<f32>;
    fn mul(self, other: &Tensor<f32>) -> Tensor<f32> {
        self.elementwise_binary(other, |a, b| a * b)
    }
}

impl Div for &Tensor<f32> {
    type Output = Tensor<f32>;
    fn div(self, other: &Tensor<f32>) -> Tensor<f32> {
        self.elementwise_binary(other, |a, b| a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<f32> = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.len(), 12);
        assert_eq!(t.shape, vec![3, 4]);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_elementwise() {
        let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let doubled = t.elementwise(|x| x * 2.0);
        assert_eq!(doubled.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_tensor_binary_ops() {
        let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]);
        let sum = &a + &b;
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_reduce_sum() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.reduce_sum(), 21.0);
    }

    #[test]
    fn test_tensor_reduce_max() {
        let t = Tensor::new(vec![4], vec![1.0, 5.0, 3.0, 2.0]);
        assert_eq!(t.reduce_max(), 5.0);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data[0], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(c.data[1], 28.0); // 1*2 + 2*4 + 3*6
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(vec![3, 2]);
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(*t.get(&[0, 1]), 2.0);
        *t.get_mut(&[1, 2]) = 99.0;
        assert_eq!(*t.get(&[1, 2]), 99.0);
    }

    #[test]
    fn test_tensor_serialization() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let json = serde_json::to_string(&t).unwrap();
        let t2: Tensor<f32> = serde_json::from_str(&json).unwrap();
        assert_eq!(t.data, t2.data);
    }
}
