//! Parallel primitives — prefix sum, histogram, radix sort, tiled matrix multiply, reduce.

use crate::tensor::Tensor;

/// Parallel reduce using tree reduction pattern
pub fn reduce(data: &[f32], op: &str) -> f32 {
    match op {
        "sum" => data.iter().sum(),
        "max" => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        "min" => data.iter().cloned().fold(f32::INFINITY, f32::min),
        "prod" => data.iter().product(),
        _ => panic!("Unknown reduce op: {}", op),
    }
}

/// Parallel-style reduce (simulated with tree reduction)
pub fn reduce_tree(data: &[f32]) -> f32 {
    if data.is_empty() { return 0.0; }
    if data.len() == 1 { return data[0]; }

    // Tree reduction: pairwise reduce
    let mut current = data.to_vec();
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        let mut i = 0;
        while i + 1 < current.len() {
            next.push(current[i] + current[i + 1]);
            i += 2;
        }
        if i < current.len() {
            next.push(current[i]);
        }
        current = next;
    }
    current[0]
}

/// Inclusive prefix sum (scan)
pub fn prefix_sum(data: &[f32]) -> Vec<f32> {
    let mut result = data.to_vec();
    let mut acc = 0.0f32;
    for val in result.iter_mut() {
        acc += *val;
        *val = acc;
    }
    result
}

/// Blelloch-style (exclusive) prefix sum — work-efficient parallel scan
pub fn prefix_sum_exclusive(data: &[f32]) -> Vec<f32> {
    let n = data.len();
    if n == 0 { return vec![]; }

    // Simple but correct exclusive scan
    let mut result = vec![0.0f32; n];
    for i in 1..n {
        result[i] = result[i - 1] + data[i - 1];
    }
    result
}

/// Histogram: count occurrences of binned values
pub fn histogram(data: &[f32], num_bins: usize, min: f32, max: f32) -> Vec<u32> {
    let mut bins = vec![0u32; num_bins];
    let range = max - min;
    if range <= 0.0 { return bins; }
    for &val in data {
        let bin = ((val - min) / range * num_bins as f32) as usize;
        let bin = bin.min(num_bins - 1);
        bins[bin] += 1;
    }
    bins
}

/// Radix sort (for u32 keys, parallel-ready pattern)
pub fn radix_sort(data: &mut [u32]) {
    let n = data.len();
    if n <= 1 { return; }

    const BITS: usize = 4; // radix = 16
    const RADIX: usize = 1 << BITS;
    let passes = (32 + BITS - 1) / BITS;

    let mut temp = vec![0u32; n];

    for pass in 0..passes {
        let shift = (pass * BITS) as u32;

        // Count
        let mut counts = vec![0usize; RADIX];
        for &val in data.iter() {
            let digit = ((val >> shift) & (RADIX as u32 - 1)) as usize;
            counts[digit] += 1;
        }

        // Prefix sum
        let mut total = 0;
        for count in counts.iter_mut() {
            let old = *count;
            *count = total;
            total += old;
        }

        // Scatter
        for &val in data.iter() {
            let digit = ((val >> shift) & (RADIX as u32 - 1)) as usize;
            temp[counts[digit]] = val;
            counts[digit] += 1;
        }

        data.copy_from_slice(&temp[..n]);
    }
}

/// Tiled matrix multiply — cache-friendly blocking pattern
pub fn tiled_matmul(a: &Tensor<f32>, b: &Tensor<f32>, tile_size: usize) -> Tensor<f32> {
    assert_eq!(a.rank(), 2);
    assert_eq!(b.rank(), 2);
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    assert_eq!(k, b.shape[0]);

    let mut c = vec![0.0f32; m * n];

    for i_tile in (0..m).step_by(tile_size) {
        for j_tile in (0..n).step_by(tile_size) {
            for p_tile in (0..k).step_by(tile_size) {
                // Process one tile
                let i_end = (i_tile + tile_size).min(m);
                let j_end = (j_tile + tile_size).min(n);
                let p_end = (p_tile + tile_size).min(k);

                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        let mut sum = 0.0;
                        for p in p_tile..p_end {
                            sum += a.data[i * k + p] * b.data[p * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    Tensor::new(vec![m, n], c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(reduce(&data, "sum"), 15.0);
    }

    #[test]
    fn test_reduce_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(reduce(&data, "max"), 5.0);
    }

    #[test]
    fn test_reduce_min() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(reduce(&data, "min"), 1.0);
    }

    #[test]
    fn test_reduce_tree() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(reduce_tree(&data), 36.0);
    }

    #[test]
    fn test_reduce_tree_odd() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(reduce_tree(&data), 6.0);
    }

    #[test]
    fn test_prefix_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = prefix_sum(&data);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_prefix_sum_empty() {
        let result = prefix_sum(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_prefix_sum_single() {
        let result = prefix_sum(&[42.0]);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_prefix_sum_exclusive() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = prefix_sum_exclusive(&data);
        assert_eq!(result, vec![0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bins = histogram(&data, 5, 0.0, 5.0);
        assert_eq!(bins.len(), 5);
        assert_eq!(bins.iter().sum::<u32>(), 5);
    }

    #[test]
    fn test_radix_sort() {
        let mut data = vec![5u32, 3, 1, 4, 2];
        radix_sort(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_radix_sort_duplicates() {
        let mut data = vec![3u32, 1, 4, 1, 5, 9, 2, 6, 5];
        radix_sort(&mut data);
        assert_eq!(data, vec![1, 1, 2, 3, 4, 5, 5, 6, 9]);
    }

    #[test]
    fn test_tiled_matmul_matches_naive() {
        let a = Tensor::new(vec![4, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        let b = Tensor::new(vec![3, 2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let naive = a.matmul(&b);
        let tiled = tiled_matmul(&a, &b, 2);
        assert_eq!(naive.shape, tiled.shape);
        for i in 0..naive.data.len() {
            assert!((naive.data[i] - tiled.data[i]).abs() < 1e-5,
                "Mismatch at {}: {} vs {}", i, naive.data[i], tiled.data[i]);
        }
    }

    #[test]
    fn test_tiled_matmul_identity() {
        let a = Tensor::new(vec![3, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let eye = Tensor::new(vec![3, 3], vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let result = tiled_matmul(&a, &eye, 2);
        assert_eq!(result.data, a.data);
    }
}
