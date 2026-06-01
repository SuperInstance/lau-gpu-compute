//! # lau-gpu-compute
//!
//! GPU compute abstractions — tensor operations designed for parallel execution,
//! even without a physical GPU present. Provides CPU fallback implementations
//! and trait-based GPU device abstraction.

pub mod device;
pub mod tensor;
pub mod kernel;
pub mod memory;
pub mod primitives;
pub mod simd;
pub mod spirv;
pub mod cuda;
pub mod agent;

pub use device::{ComputeDevice, CpuDevice, GpuDevice};
pub use tensor::{Tensor, TensorOp};
pub use kernel::{WorkGroup, Kernel, ThreadId, Barrier};
pub use memory::{DeviceBuffer, MemoryTransfer};
pub use primitives::{prefix_sum, histogram, radix_sort, tiled_matmul, reduce};
pub use simd::{f32x4, f32x8, SimdOps};
pub use spirv::{SpirvBuilder, SpirvInstruction, SpirvType};
pub use cuda::{CudaGrid, CudaBlock, CudaThread, CudaKernelLayout};
pub use agent::{AgentBatch, AgentInferenceEngine};
