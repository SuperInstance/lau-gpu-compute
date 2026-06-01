//! GPGPU kernel abstractions — work groups, thread IDs, barrier patterns.

use serde::{Serialize, Deserialize};
use crate::tensor::Tensor;

/// Thread ID within a work group / kernel dispatch
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThreadId {
    pub global: [u32; 3],
    pub local: [u32; 3],
    pub group: [u32; 3],
}

impl ThreadId {
    pub fn global_linear(&self, group_size: [u32; 3]) -> u32 {
        self.group[0] * group_size[0] + self.local[0]
            + (self.group[1] * group_size[1] + self.local[1]) * group_size[0]
            + (self.group[2] * group_size[2] + self.local[2]) * group_size[0] * group_size[1]
    }

    pub fn local_linear(&self) -> u32 {
        self.local[0] + self.local[1] + self.local[2]
    }
}

/// Work group configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorkGroup {
    pub size: [u32; 3],
    pub count: [u32; 3],
}

impl WorkGroup {
    pub fn new(size: [u32; 3], count: [u32; 3]) -> Self {
        Self { size, count }
    }

    pub fn total_threads(&self) -> u32 {
        self.size[0] * self.size[1] * self.size[2]
            * self.count[0] * self.count[1] * self.count[2]
    }

    pub fn total_local_threads(&self) -> u32 {
        self.size[0] * self.size[1] * self.size[2]
    }

    /// Decompose a flat work size into work groups
    pub fn decompose(total_items: u32, preferred_group_size: u32) -> Self {
        let group_size = preferred_group_size.min(total_items);
        let count = (total_items + group_size - 1) / group_size;
        Self {
            size: [group_size, 1, 1],
            count: [count, 1, 1],
        }
    }

    /// Iterate all thread IDs (for CPU simulation)
    pub fn iter_thread_ids(&self) -> impl Iterator<Item = ThreadId> + '_ {
        let sx = self.size[0];
        let sy = self.size[1];
        let sz = self.size[2];
        (0..self.count[2]).flat_map(move |gz| {
            (0..self.count[1]).flat_map(move |gy| {
                (0..self.count[0]).flat_map(move |gx| {
                    (0..sz).flat_map(move |lz| {
                        (0..sy).flat_map(move |_ly| {
                            (0..sx).map(move |lx| ThreadId {
                                global: [
                                    gx * sx + lx,
                                    gy * sy,
                                    gz * sz,
                                ],
                                local: [lx, 0, 0],
                                group: [gx, gy, gz],
                            })
                        })
                    })
                })
            })
        })
    }
}

/// Barrier synchronization type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Barrier {
    /// All threads in work group must reach before any proceed
    Local,
    /// All threads in dispatch must reach
    Global,
    /// Memory barrier only — ordering, not synchronization
    Memory,
}

/// A compute kernel with CPU execution fallback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    pub name: String,
    pub work_group: WorkGroup,
    pub op: KernelOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelOp {
    Map { func: MapFunc },
    Reduce { op: ReduceFunc },
    PrefixScan,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MapFunc {
    Relu, Sigmoid, Tanh, Exp, Square, Negate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReduceFunc {
    Sum, Max, Min, Prod,
}

impl Kernel {
    pub fn new(name: &str, work_group: WorkGroup, op: KernelOp) -> Self {
        Self { name: name.to_string(), work_group, op }
    }

    /// Execute on CPU (fallback)
    pub fn execute_cpu(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match &self.op {
            KernelOp::Map { func } => {
                input.elementwise(|x| match func {
                    MapFunc::Relu => x.max(0.0),
                    MapFunc::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    MapFunc::Tanh => x.tanh(),
                    MapFunc::Exp => x.exp(),
                    MapFunc::Square => x * x,
                    MapFunc::Negate => -x,
                })
            }
            KernelOp::Reduce { op } => {
                let val = match op {
                    ReduceFunc::Sum => input.data.iter().sum(),
                    ReduceFunc::Max => input.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ReduceFunc::Min => input.data.iter().cloned().fold(f32::INFINITY, f32::min),
                    ReduceFunc::Prod => input.data.iter().product(),
                };
                Tensor::new(vec![1], vec![val])
            }
            KernelOp::PrefixScan => {
                let mut data = input.data.clone();
                let mut acc = 0.0f32;
                for d in data.iter_mut() {
                    acc += *d;
                    *d = acc;
                }
                Tensor::new(input.shape.clone(), data)
            }
            KernelOp::Custom { .. } => input.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_group_decompose() {
        let wg = WorkGroup::decompose(1000, 256);
        assert_eq!(wg.size[0], 256);
        assert_eq!(wg.count[0], 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_work_group_total_threads() {
        let wg = WorkGroup::new([64, 1, 1], [10, 1, 1]);
        assert_eq!(wg.total_threads(), 640);
    }

    #[test]
    fn test_thread_id_global_linear() {
        let tid = ThreadId {
            global: [5, 0, 0],
            local: [5, 0, 0],
            group: [0, 0, 0],
        };
        assert_eq!(tid.global_linear([64, 1, 1]), 5);
    }

    #[test]
    fn test_kernel_map_relu() {
        let input = Tensor::new(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let wg = WorkGroup::decompose(5, 64);
        let kernel = Kernel::new("relu", wg, KernelOp::Map { func: MapFunc::Relu });
        let output = kernel.execute_cpu(&input);
        assert_eq!(output.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_kernel_reduce_sum() {
        let input = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let wg = WorkGroup::decompose(4, 64);
        let kernel = Kernel::new("sum", wg, KernelOp::Reduce { op: ReduceFunc::Sum });
        let output = kernel.execute_cpu(&input);
        assert_eq!(output.data[0], 10.0);
    }

    #[test]
    fn test_kernel_prefix_scan() {
        let input = Tensor::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let wg = WorkGroup::decompose(5, 64);
        let kernel = Kernel::new("prefix_sum", wg, KernelOp::PrefixScan);
        let output = kernel.execute_cpu(&input);
        assert_eq!(output.data, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn test_iter_thread_ids() {
        let wg = WorkGroup::new([2, 1, 1], [2, 1, 1]);
        let ids: Vec<_> = wg.iter_thread_ids().collect();
        assert_eq!(ids.len(), 4);
    }
}
