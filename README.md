# lau-gpu-compute

> GPU compute abstractions — tensor operations designed for parallel execution with CPU fallback.

Part of the **PLATO/LAU ecosystem** — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

---

## What This Does

`lau-gpu-compute` provides the foundational GPU compute layer for the `lau-*` crate family. It offers:

- **Multi-dimensional tensors** with element-wise ops, reductions, axis operations, and matrix multiplication
- **Device abstraction** — `GpuDevice` trait with a `CpuDevice` fallback; simulated GPU for testing
- **Kernel dispatch** — work groups, thread IDs, barriers, and CPU-executable kernel operations (map, reduce, prefix scan)
- **Device memory model** — host↔device transfers, device buffers, region copies, memory layouts (row/column major)
- **Parallel primitives** — inclusive/exclusive prefix sum, histogram, radix sort, tiled matrix multiply, tree reduction
- **SIMD-style vectors** — `f32x4` (128-bit) and `f32x8` (256-bit) with arithmetic, dot products, FMA, load/store
- **SPIR-V IR builder** — type system, instruction set, and disassembler for Vulkan compute shader generation
- **CUDA layout patterns** — grid/block/thread hierarchy, 1D/2D launchers, global thread ID computation
- **Agent batch inference** — parallel agent computation with tiled matmul, attention scores (softmax), and launch configs

Everything compiles without CUDA/Vulkan installed. GPU-specific code is abstracted behind traits; CPU fallbacks make it portable.

---

## The Key Idea

This crate separates **what** to compute from **where** to compute it. Every operation has a CPU implementation that matches the GPU execution pattern (same work decomposition, same memory access patterns). This means:

1. **Develop on CPU, deploy on GPU** — write and test kernels on CPU, then swap in a real GPU device
2. **Deterministic CPU simulation** — `WorkGroup::iter_thread_ids()` and `CudaGrid::iter_blocks_and_threads()` let you simulate GPU thread execution exactly
3. **Teaching-quality primitives** — tree reduction, Blelloch scan, radix sort, and tiled matmul are implemented with clear algorithmic structure

---

## Install

```bash
cargo add lau-gpu-compute
```

### Dependencies

| Crate | Version | Why |
|---|---|---|
| `serde` | 1 | Serialization of all public types |
| `nalgebra` | 0.33 | Matrix operations (available for integration) |
| `serde_json` | 1 | *(dev-only)* test serialization round-trips |

No GPU drivers or toolkits required.

---

## Quick Start

### Tensor operations

```rust
use lau_gpu_compute::Tensor;

let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let c = a.matmul(&b);
assert_eq!(c.shape, vec![2, 2]);

let relu = a.elementwise(|x| x.max(0.0));
let sum = &a + &b;  // element-wise via operator overloading
```

### Device abstraction

```rust
use lau_gpu_compute::{ComputeDevice, CpuDevice, GpuDevice};

let cpu = CpuDevice::new();
assert!(cpu.is_available());
println!("{} cores, {} bytes memory", cpu.compute_units(), cpu.memory_size());

let gpu = ComputeDevice::simulated_gpu("RTX 4090", 24 * 1024 * 1024 * 1024, 128);
```

### Kernel dispatch

```rust
use lau_gpu_compute::{Kernel, WorkGroup, KernelOp, MapFunc, Tensor};

let input = Tensor::new(vec![4], vec![-2.0, -1.0, 0.0, 3.0]);
let wg = WorkGroup::decompose(4, 64);
let kernel = Kernel::new("relu", wg, KernelOp::Map { func: MapFunc::Relu });
let output = kernel.execute_cpu(&input);
assert_eq!(output.data, vec![0.0, 0.0, 0.0, 3.0]);
```

### Parallel primitives

```rust
use lau_gpu_compute::{prefix_sum, radix_sort, tiled_matmul, histogram};

// Inclusive prefix sum
let scanned = prefix_sum(&[1.0, 2.0, 3.0, 4.0, 5.0]);
assert_eq!(scanned, vec![1.0, 3.0, 6.0, 10.0, 15.0]);

// Radix sort
let mut keys = vec![5u32, 3, 1, 4, 2];
radix_sort(&mut keys);
assert_eq!(keys, vec![1, 2, 3, 4, 5]);

// Histogram
let bins = histogram(&[1.0, 2.5, 3.0, 4.5, 5.0], 5, 0.0, 5.0);
```

### SIMD-style vectors

```rust
use lau_gpu_compute::{f32x4, SimdOps};

let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
let b = f32x4::splat(2.0);
let c = a.mul(&b);
assert_eq!(c.hsum(), 20.0);
assert_eq!(a.dot(&b), 20.0);
```

### CUDA grid layout

```rust
use lau_gpu_compute::{CudaGrid, CudaKernelLayout};

let grid = CudaGrid::launch_1d(10000, 256);
println!("{} blocks, {} threads total", grid.total_blocks(), grid.total_threads());

let layout = CudaKernelLayout::optimal_1d("vec_add", 100000)
    .with_shared_memory(4096);
```

### Agent batch inference

```rust
use lau_gpu_compute::{AgentBatch, AgentInferenceEngine, Tensor};

let mut batch = AgentBatch::new(8, 32);
let engine = AgentInferenceEngine::new(32);
let input = Tensor::from_scalar(vec![8, 32], 1.0);

engine.step(&mut batch, &input);

let scores = engine.attention_scores(&batch);
// scores is [8×8] softmax attention matrix
```

---

## API Reference

### Tensor

```rust
pub struct Tensor<T: Clone + Default> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}
impl<T: Clone + Default> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self;
    pub fn zeros(shape: Vec<usize>) -> Self;
    pub fn from_scalar(shape: Vec<usize>, value: T) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn flat_index(&self, indices: &[usize]) -> usize;
    pub fn get(&self, indices: &[usize]) -> &T;
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T;
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self;
    pub fn rank(&self) -> usize;
    pub fn size(&self) -> usize;
}
impl Tensor<f32> {
    pub fn elementwise<F: Fn(f32) -> f32>(&self, f: F) -> Self;
    pub fn elementwise_binary<F: Fn(f32, f32) -> f32>(&self, other: &Self, f: F) -> Self;
    pub fn reduce_axis<F: Fn(f32, f32) -> f32>(&self, axis: usize, init: f32, f: F) -> Self;
    pub fn reduce_sum(&self) -> f32;
    pub fn reduce_max(&self) -> f32;
    pub fn reduce_min(&self) -> f32;
    pub fn matmul(&self, other: &Self) -> Self;
}
// Operator overloads: &Tensor<f32> + &Tensor<f32>, -, *, /
```

### Device

```rust
pub trait GpuDevice: Debug {
    fn name(&self) -> &str;
    fn memory_size(&self) -> u64;
    fn compute_units(&self) -> u32;
    fn max_work_group_size(&self) -> u32;
    fn execute(&self, kernel: &Kernel, input: &Tensor<f32>) -> Tensor<f32>;
    fn is_available(&self) -> bool;
}

pub struct CpuDevice { /* ... */ }  // implements GpuDevice
pub enum ComputeDevice { Cpu(CpuDevice), SimulatedGpu { ... } }
```

### Kernel

```rust
pub struct WorkGroup { pub size: [u32; 3], pub count: [u32; 3] }
impl WorkGroup {
    pub fn new(size: [u32; 3], count: [u32; 3]) -> Self;
    pub fn total_threads(&self) -> u32;
    pub fn decompose(total_items: u32, preferred_group_size: u32) -> Self;
    pub fn iter_thread_ids(&self) -> impl Iterator<Item = ThreadId> + '_;
}

pub struct ThreadId { pub global: [u32; 3], pub local: [u32; 3], pub group: [u32; 3] }

pub enum Barrier { Local, Global, Memory }

pub struct Kernel { pub name: String, pub work_group: WorkGroup, pub op: KernelOp }
impl Kernel {
    pub fn new(name: &str, work_group: WorkGroup, op: KernelOp) -> Self;
    pub fn execute_cpu(&self, input: &Tensor<f32>) -> Tensor<f32>;
}
```

### Memory

```rust
pub struct DeviceBuffer<T: Clone + Default> { /* ... */ }
impl<T: Clone + Default> DeviceBuffer<T> {
    pub fn allocate(capacity: usize) -> Self;
    pub fn read(&self) -> Vec<T>;
    pub fn write(&mut self, data: &[T]);
    pub fn read_at(&self, index: usize) -> T;
    pub fn write_at(&mut self, index: usize, value: T);
    pub fn size_bytes(&self) -> usize;
}
impl DeviceBuffer<f32> {
    pub fn fill(&mut self, value: f32);
    pub fn copy_from(&mut self, src: &DeviceBuffer<f32>, src_offset: usize, dst_offset: usize, len: usize);
}

pub struct MemoryTransfer;
impl MemoryTransfer {
    pub fn host_to_device<T>(host_data: &[T]) -> DeviceBuffer<T>;
    pub fn device_to_host<T>(buffer: &DeviceBuffer<T>) -> Vec<T>;
    pub fn device_to_device<T>(src: &DeviceBuffer<T>) -> DeviceBuffer<T>;
}

pub enum MemoryLayout { RowMajor, ColumnMajor }
impl MemoryLayout {
    pub fn flat_index(&self, row: usize, col: usize, rows: usize, cols: usize) -> usize;
}
```

### Parallel Primitives

```rust
pub fn reduce(data: &[f32], op: &str) -> f32;               // "sum", "max", "min", "prod"
pub fn reduce_tree(data: &[f32]) -> f32;                      // pairwise tree reduction
pub fn prefix_sum(data: &[f32]) -> Vec<f32>;                  // inclusive scan
pub fn prefix_sum_exclusive(data: &[f32]) -> Vec<f32>;        // exclusive (Blelloch-style)
pub fn histogram(data: &[f32], num_bins: usize, min: f32, max: f32) -> Vec<u32>;
pub fn radix_sort(data: &mut [u32]);                           // LSD radix sort, radix-16
pub fn tiled_matmul(a: &Tensor<f32>, b: &Tensor<f32>, tile_size: usize) -> Tensor<f32>;
```

### SIMD

```rust
pub struct f32x4(pub [f32; 4]);
impl f32x4 {
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self;
    pub fn splat(val: f32) -> Self;
    pub fn zero() -> Self;
    pub fn add/sub/mul/div(&self, other: &Self) -> Self;
    pub fn hsum(&self) -> f32;
    pub fn dot(&self, other: &Self) -> f32;
    pub fn fma(&self, a: &Self, b: &Self) -> Self;  // self * a + b
    pub fn max/min(&self, other: &Self) -> Self;
    pub fn load(slice: &[f32]) -> Self;
    pub fn store(&self, slice: &mut [f32]);
}

pub struct f32x8(pub [f32; 8]);  // same pattern
pub trait SimdOps: Clone { fn splat/zero/add/mul/hsum/dot(...) }
```

### SPIR-V

```rust
pub enum SpirvType { Void, Float32, Int32, UInt32, Vector { scalar, len }, Array { element, len }, Struct { fields }, Pointer { target, storage } }
pub enum StorageClass { Uniform, StorageBuffer, PushConstant, Workgroup, Private, Function }
pub enum SpirvInstruction { FAdd/FSub/FMul/FDiv, Load/Store, AccessChain, Label/Branch/BranchConditional/Loop, Return/ReturnValue, GlobalInvocationId/LocalInvocationId, ControlBarrier/MemoryBarrier, CompositeConstruct/CompositeExtract }

pub struct SpirvBuilder { /* ... */ }
impl SpirvBuilder {
    pub fn new(name: &str) -> Self;
    pub fn next_id(&mut self) -> u32;
    pub fn emit(&mut self, inst: SpirvInstruction);
    pub fn instructions(&self) -> &[SpirvInstruction];
    pub fn build_vector_add(&mut self) -> u32;
    pub fn build_reduce_pattern(&mut self, workgroup_size: u32);
    pub fn disassemble(&self) -> String;
}
```

### CUDA

```rust
pub struct CudaThread { pub x: u32, pub y: u32, pub z: u32 }
pub struct CudaBlock { pub idx_x/y/z: u32, pub dim_x/y/z: u32 }
pub struct CudaGrid { pub grid_dim_x/y/z: u32, pub block_dim_x/y/z: u32 }
impl CudaGrid {
    pub fn new(grid: [u32;3], block: [u32;3]) -> Self;
    pub fn launch_1d(n: u32, tpb: u32) -> Self;
    pub fn launch_2d(w: u32, h: u32, bx: u32, by: u32) -> Self;
    pub fn total_blocks/total_threads/threads_per_block(&self) -> u32;
    pub fn global_thread_id_1d/2d(&self, block: &CudaBlock, thread: &CudaThread) -> _;
    pub fn iter_blocks_and_threads(&self) -> impl Iterator<Item = (CudaBlock, CudaThread)>;
}

pub struct CudaKernelLayout { pub name: String, pub grid: CudaGrid, pub shared_memory_bytes: u32 }
impl CudaKernelLayout {
    pub fn new(name: &str, grid: CudaGrid) -> Self;
    pub fn with_shared_memory(self, bytes: u32) -> Self;
    pub fn optimal_1d(name: &str, n: u32) -> Self;
}
```

### Agent

```rust
pub struct AgentState { pub id: u64, pub hidden: Vec<f32>, pub output: Vec<f32> }
pub struct AgentBatch { pub agents: Vec<AgentState>, pub hidden_dim: usize }
impl AgentBatch {
    pub fn new(batch_size: usize, hidden_dim: usize) -> Self;
    pub fn batch_size(&self) -> usize;
    pub fn stack_hidden(&self) -> Tensor<f32>;
    pub fn unstack_hidden/unstack_output(&mut self, tensor: &Tensor<f32>);
}

pub struct AgentInferenceEngine { pub weight_input: Tensor<f32>, pub weight_hidden: Tensor<f32>, pub hidden_dim: usize }
impl AgentInferenceEngine {
    pub fn new(hidden_dim: usize) -> Self;
    pub fn step(&self, batch: &mut AgentBatch, input: &Tensor<f32>);
    pub fn attention_scores(&self, batch: &AgentBatch) -> Vec<f32>;  // [bs×bs] softmax
    pub fn launch_config(&self, batch: &AgentBatch) -> CudaGrid;
}
```

---

## How It Works

### Architecture

```
┌──────────────┐
│  Tensor<T>   │  ← Multi-dimensional data (shape + flat data)
└──────┬───────┘
       │
┌──────▼───────┐     ┌───────────────┐
│   Kernel     │────▶│  GpuDevice    │  ← CPU fallback or real GPU
│ (work group) │     │  .execute()   │
└──────────────┘     └───────────────┘
       │
┌──────▼───────────────────────────┐
│  Parallel Primitives             │
│  prefix_sum, radix_sort, reduce  │
│  tiled_matmul, histogram         │
└──────────────────────────────────┘
       │
┌──────▼───────┐     ┌──────────────┐
│  SIMD types  │     │  Memory      │
│  f32x4/f32x8 │     │  DeviceBuffer│
└──────────────┘     └──────────────┘
       │
┌──────▼───────────────────────────┐
│  Shader/Kernel Generation        │
│  SPIR-V builder, CUDA layouts    │
└──────────────────────────────────┘
       │
┌──────▼───────────────────────────┐
│  Agent Batch Inference           │
│  tiled matmul, attention, launch │
└──────────────────────────────────┘
```

### CPU Simulation of GPU Execution

The crate simulates GPU execution patterns on CPU:

1. **WorkGroup** decomposes work into groups (like OpenCL/WebGPU)
2. **CudaGrid** decomposes into blocks × threads (like CUDA)
3. Both provide iterators that yield all (group, thread) pairs
4. **Kernel::execute_cpu** runs the operation sequentially but with the same logical decomposition

### Tiled Matrix Multiply

The tiled matmul algorithm blocks the computation into tile_size × tile_size submatrices:

```
for each tile (i_tile, j_tile):
    for each tile (p_tile):
        accumulate A[i_tile..][p_tile..] × B[p_tile..][j_tile..]
```

This matches GPU shared-memory tiling patterns and improves cache locality on CPU.

### Radix Sort

LSD radix sort with radix-16 (4 bits per pass):
1. Count digits → prefix sum of counts → scatter to temp buffer
2. Repeat for each 4-bit group (8 passes for u32)

### Tree Reduction

Pairwise reduction mimicking GPU parallel reduction:

```
[a, b, c, d, e, f, g, h] → [a+b, c+d, e+f, g+h] → [a+b+c+d, e+f+g+h] → [sum]
```

O(n) total work, O(log n) parallel depth.

### Agent Inference

Each agent has a hidden state vector. The engine runs:
1. **Input projection**: input @ W_input (tiled matmul)
2. **Hidden projection**: hidden @ W_hidden (tiled matmul)
3. **Residual + ReLU**: (input_proj + hidden_proj).max(0)
4. **Attention**: dot-product similarity across agents → row-wise softmax

---

## The Math

### Tiled Matrix Multiplication

For matrices A ∈ ℝ^{m×k} and B ∈ ℝ^{k×n}, the product C = AB is:

$$C_{ij} = \sum_{p=0}^{k-1} A_{ip} \cdot B_{pj}$$

The tiled version computes partial sums over tiles of size T:

$$C_{ij} = \sum_{\text{tile } t} \sum_{p \in \text{tile}} A_{ip} \cdot B_{pj}$$

This reduces cache misses from O(mnk) to O(mn·k/T) by keeping tiles in L1/L2 cache (or GPU shared memory).

### Prefix Sum (Scan)

**Inclusive scan**: out[i] = Σ_{j=0}^{i} in[j]

**Exclusive (Blelloch) scan**: out[i] = Σ_{j=0}^{i-1} in[j], out[0] = 0

The Blelloch algorithm is work-efficient: O(n) total work, O(log n) parallel depth. The implementation here uses the sequential formulation (correct, same API), while the GPU version would use the two-pass up-sweep/down-sweep tree.

### Radix Sort

LSD (Least Significant Digit) radix sort with radix R = 16:

For each digit position d (4 bits):
1. **Count**: histogram of digit values
2. **Prefix sum**: compute starting positions
3. **Scatter**: place elements at their computed positions

Stable, O(n · passes) where passes = ⌈32/log₂(R)⌉ = 8 for u32 with radix-16.

### Tree Reduction

Parallel reduction via pairwise summation:

$$\text{reduce}([x_1, ..., x_n]) = \text{reduce}([x_1 + x_2, x_3 + x_4, ...])$$

Total work: n−1 additions. Parallel depth: ⌈log₂(n)⌉. This is how GPU reduction kernels work (warp-level → block-level → global).

### Softmax Attention

For agent attention scores S ∈ ℝ^{B×B}:

$$S_{ij} = \text{softmax}_j\left(\mathbf{h}_i \cdot \mathbf{h}_j^T\right) = \frac{\exp(\mathbf{h}_i \cdot \mathbf{h}_j)}{\sum_k \exp(\mathbf{h}_i \cdot \mathbf{h}_k)}$$

Numerically stable with the max-subtraction trick: subtract max(S_i) before exp to prevent overflow.

---

## Testing

**69 tests** across 8 modules:

| Module | Tests | What's covered |
|---|---|---|
| `tensor` | 9 | Creation, elementwise, binary ops, reduce, matmul, reshape, indexing, serialization |
| `device` | 3 | CPU device, simulated GPU, serialization |
| `kernel` | 7 | Work group decomposition, thread IDs, map/reduce/prefix scan kernels |
| `memory` | 7 | Buffer allocation, host↔device transfer, copy regions, memory layouts |
| `primitives` | 15 | Reduce (sum/max/min/tree), prefix sum (inclusive/exclusive/edge cases), histogram, radix sort, tiled matmul |
| `simd` | 11 | f32x4 arithmetic, dot, FMA, max/min, load/store; f32x8 operations |
| `spirv` | 5 | Builder, type hierarchy, reduce pattern, disassembly, serialization |
| `cuda` | 7 | Grid launch (1D/2D), thread IDs, block/thread iteration, shared memory |
| `agent` | 6 | Batch creation, stack/unstack, inference step, attention softmax, launch config, serialization |

Run:

```bash
cargo test
```

---

## License

MIT
