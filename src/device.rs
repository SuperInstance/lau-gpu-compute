//! Compute device abstraction — CPU fallback + GPU device trait.

use serde::{Serialize, Deserialize};
use crate::tensor::Tensor;
use crate::kernel::Kernel;

/// Trait for any compute device (GPU, CPU, etc.)
pub trait GpuDevice: std::fmt::Debug {
    /// Device name
    fn name(&self) -> &str;
    /// Total memory in bytes
    fn memory_size(&self) -> u64;
    /// Number of compute units
    fn compute_units(&self) -> u32;
    /// Max work group size
    fn max_work_group_size(&self) -> u32;
    /// Execute a kernel on this device
    fn execute(&self, kernel: &Kernel, input: &Tensor<f32>) -> Tensor<f32>;
    /// Check if device is available
    fn is_available(&self) -> bool;
}

/// CPU fallback device for compute operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuDevice {
    name: String,
    memory_size: u64,
    num_cores: u32,
}

impl CpuDevice {
    pub fn new() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        Self {
            name: "CPU Fallback".to_string(),
            memory_size: 8 * 1024 * 1024 * 1024, // 8 GB simulated
            num_cores: cores,
        }
    }

    pub fn with_cores(mut self, cores: u32) -> Self {
        self.num_cores = cores;
        self
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice for CpuDevice {
    fn name(&self) -> &str { &self.name }
    fn memory_size(&self) -> u64 { self.memory_size }
    fn compute_units(&self) -> u32 { self.num_cores }
    fn max_work_group_size(&self) -> u32 { 1024 }

    fn execute(&self, kernel: &Kernel, input: &Tensor<f32>) -> Tensor<f32> {
        kernel.execute_cpu(input)
    }

    fn is_available(&self) -> bool { true }
}

/// Generic compute device enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeDevice {
    Cpu(CpuDevice),
    SimulatedGpu { name: String, memory: u64, compute_units: u32 },
}

impl ComputeDevice {
    pub fn cpu() -> Self {
        ComputeDevice::Cpu(CpuDevice::new())
    }

    pub fn simulated_gpu(name: &str, memory: u64, compute_units: u32) -> Self {
        ComputeDevice::SimulatedGpu {
            name: name.to_string(),
            memory,
            compute_units,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ComputeDevice::Cpu(c) => c.name(),
            ComputeDevice::SimulatedGpu { name, .. } => name,
        }
    }

    pub fn compute_units(&self) -> u32 {
        match self {
            ComputeDevice::Cpu(c) => c.compute_units(),
            ComputeDevice::SimulatedGpu { compute_units, .. } => *compute_units,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_creation() {
        let cpu = CpuDevice::new();
        assert!(cpu.is_available());
        assert!(cpu.compute_units() >= 1);
        assert!(cpu.memory_size() > 0);
    }

    #[test]
    fn test_simulated_gpu() {
        let gpu = ComputeDevice::simulated_gpu("TestGPU", 4 * 1024 * 1024 * 1024, 32);
        assert_eq!(gpu.name(), "TestGPU");
        assert_eq!(gpu.compute_units(), 32);
    }

    #[test]
    fn test_compute_device_serialization() {
        let cpu = CpuDevice::new();
        let json = serde_json::to_string(&cpu).unwrap();
        let deserialized: CpuDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(cpu.name(), deserialized.name());
    }
}
