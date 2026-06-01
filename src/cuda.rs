//! CUDA kernel layout patterns — grid, block, thread hierarchy.

use serde::{Serialize, Deserialize};

/// CUDA thread within a block
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaThread {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CudaThread {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn linear(&self) -> u32 {
        self.x + self.y + self.z
    }

    /// 1D linear index within a block
    pub fn linear_1d(&self, block_dims: &CudaBlock) -> u32 {
        self.x
    }

    /// Flatten to 1D index within block
    pub fn flat_index(&self, block: &CudaBlock) -> u32 {
        self.z * block.dim_x * block.dim_y + self.y * block.dim_x + self.x
    }
}

/// CUDA block within a grid
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaBlock {
    pub idx_x: u32,
    pub idx_y: u32,
    pub idx_z: u32,
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
}

impl CudaBlock {
    pub fn new(idx: [u32; 3], dims: [u32; 3]) -> Self {
        Self {
            idx_x: idx[0], idx_y: idx[1], idx_z: idx[2],
            dim_x: dims[0], dim_y: dims[1], dim_z: dims[2],
        }
    }

    pub fn threads_per_block(&self) -> u32 {
        self.dim_x * self.dim_y * self.dim_z
    }

    pub fn flat_block_index(&self) -> u32 {
        self.idx_z * self.dim_y * self.dim_x + self.idx_y * self.dim_x + self.idx_x
    }
}

/// CUDA grid (collection of blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaGrid {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
}

impl CudaGrid {
    pub fn new(grid_dims: [u32; 3], block_dims: [u32; 3]) -> Self {
        Self {
            grid_dim_x: grid_dims[0], grid_dim_y: grid_dims[1], grid_dim_z: grid_dims[2],
            block_dim_x: block_dims[0], block_dim_y: block_dims[1], block_dim_z: block_dims[2],
        }
    }

    /// 1D grid launcher convenience
    pub fn launch_1d(n: u32, threads_per_block: u32) -> Self {
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        Self::new([blocks, 1, 1], [threads_per_block, 1, 1])
    }

    /// 2D grid launcher
    pub fn launch_2d(width: u32, height: u32, block_x: u32, block_y: u32) -> Self {
        let bx = (width + block_x - 1) / block_x;
        let by = (height + block_y - 1) / block_y;
        Self::new([bx, by, 1], [block_x, block_y, 1])
    }

    pub fn total_blocks(&self) -> u32 {
        self.grid_dim_x * self.grid_dim_y * self.grid_dim_z
    }

    pub fn total_threads(&self) -> u32 {
        self.total_blocks() * self.threads_per_block()
    }

    pub fn threads_per_block(&self) -> u32 {
        self.block_dim_x * self.block_dim_y * self.block_dim_z
    }

    /// Compute global thread index for 1D layout
    pub fn global_thread_id_1d(&self, block: &CudaBlock, thread: &CudaThread) -> u32 {
        block.idx_x * self.block_dim_x + thread.x
    }

    /// Compute global thread index for 2D layout
    pub fn global_thread_id_2d(&self, block: &CudaBlock, thread: &CudaThread) -> (u32, u32) {
        let gx = block.idx_x * self.block_dim_x + thread.x;
        let gy = block.idx_y * self.block_dim_y + thread.y;
        (gx, gy)
    }

    /// Iterate all (block, thread) pairs (for CPU simulation)
    pub fn iter_blocks_and_threads(&self) -> impl Iterator<Item = (CudaBlock, CudaThread)> + '_ {
        let bx = self.block_dim_x;
        let by = self.block_dim_y;
        let bz = self.block_dim_z;
        (0..self.grid_dim_z).flat_map(move |_gz| {
            (0..self.grid_dim_y).flat_map(move |_gy| {
                (0..self.grid_dim_x).flat_map(move |gx| {
                    let block = CudaBlock::new(
                        [gx, _gy, _gz],
                        [bx, by, bz],
                    );
                    (0..bz).flat_map(move |_lz| {
                        (0..by).flat_map(move |_ly| {
                            (0..bx).map(move |lx| {
                                let thread = CudaThread::new(lx, _ly, _lz);
                                (block, thread)
                            })
                        })
                    })
                })
            })
        })
    }
}

/// Complete kernel layout specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaKernelLayout {
    pub name: String,
    pub grid: CudaGrid,
    pub shared_memory_bytes: u32,
}

impl CudaKernelLayout {
    pub fn new(name: &str, grid: CudaGrid) -> Self {
        Self {
            name: name.to_string(),
            grid,
            shared_memory_bytes: 0,
        }
    }

    pub fn with_shared_memory(mut self, bytes: u32) -> Self {
        self.shared_memory_bytes = bytes;
        self
    }

    /// Decompose a 1D work size into optimal CUDA launch config
    pub fn optimal_1d(name: &str, n: u32) -> Self {
        // Typical: 256 threads per block for compute
        let tpb = 256.min(n);
        Self::new(name, CudaGrid::launch_1d(n, tpb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_grid_1d() {
        let grid = CudaGrid::launch_1d(1000, 256);
        assert_eq!(grid.total_blocks(), 4); // ceil(1000/256) = 4
        assert_eq!(grid.threads_per_block(), 256);
        assert_eq!(grid.total_threads(), 1024);
    }

    #[test]
    fn test_cuda_grid_2d() {
        let grid = CudaGrid::launch_2d(32, 32, 16, 16);
        assert_eq!(grid.grid_dim_x, 2);
        assert_eq!(grid.grid_dim_y, 2);
        assert_eq!(grid.threads_per_block(), 256);
    }

    #[test]
    fn test_cuda_thread_flat_index() {
        let block = CudaBlock::new([0, 0, 0], [4, 4, 1]);
        let thread = CudaThread::new(2, 1, 0);
        assert_eq!(thread.flat_index(&block), 6); // 1*4 + 2
    }

    #[test]
    fn test_cuda_global_thread_id() {
        let grid = CudaGrid::launch_1d(512, 256);
        let block = CudaBlock::new([1, 0, 0], [256, 1, 1]);
        let thread = CudaThread::new(10, 0, 0);
        assert_eq!(grid.global_thread_id_1d(&block, &thread), 266); // 1*256 + 10
    }

    #[test]
    fn test_cuda_iter_blocks_threads() {
        let grid = CudaGrid::launch_1d(8, 4);
        let count = grid.iter_blocks_and_threads().count();
        assert_eq!(count, 8); // 2 blocks * 4 threads = 8
    }

    #[test]
    fn test_cuda_kernel_layout() {
        let layout = CudaKernelLayout::optimal_1d("vec_add", 10000);
        assert_eq!(layout.name, "vec_add");
        assert!(layout.grid.total_threads() >= 10000);
    }

    #[test]
    fn test_cuda_kernel_shared_memory() {
        let grid = CudaGrid::launch_1d(256, 256);
        let layout = CudaKernelLayout::new("reduce", grid).with_shared_memory(1024);
        assert_eq!(layout.shared_memory_bytes, 1024);
    }
}
