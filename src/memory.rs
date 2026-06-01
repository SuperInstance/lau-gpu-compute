//! GPU memory model — device buffers, host-device transfer simulation.

use serde::{Serialize, Deserialize};

/// Simulated device buffer with memory layout tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceBuffer<T: Clone + Default> {
    /// Data stored on "device"
    data: Vec<T>,
    /// Buffer size in elements
    capacity: usize,
    /// Whether buffer is currently on "device"
    on_device: bool,
    /// Buffer ID for tracking
    id: u64,
}

static mut BUFFER_COUNTER: u64 = 0;

fn next_buffer_id() -> u64 {
    unsafe {
        BUFFER_COUNTER += 1;
        BUFFER_COUNTER
    }
}

impl<T: Clone + Default> DeviceBuffer<T> {
    /// Allocate a new device buffer
    pub fn allocate(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            on_device: true,
            id: next_buffer_id(),
        }
    }

    /// Buffer ID
    pub fn id(&self) -> u64 { self.id }

    /// Capacity in elements
    pub fn capacity(&self) -> usize { self.capacity }

    /// Whether the buffer is on device
    pub fn is_on_device(&self) -> bool { self.on_device }

    /// Read data from device buffer
    pub fn read(&self) -> Vec<T> {
        assert!(self.on_device, "Buffer not on device");
        self.data.clone()
    }

    /// Write data to device buffer
    pub fn write(&mut self, data: &[T]) {
        assert!(self.on_device, "Buffer not on device");
        assert!(data.len() <= self.capacity, "Data exceeds buffer capacity");
        self.data[..data.len()].clone_from_slice(data);
    }

    /// Read a single element
    pub fn read_at(&self, index: usize) -> T {
        self.data[index].clone()
    }

    /// Write a single element
    pub fn write_at(&mut self, index: usize, value: T) {
        self.data[index] = value;
    }

    /// Size of buffer in bytes
    pub fn size_bytes(&self) -> usize {
        self.capacity * std::mem::size_of::<T>()
    }
}

impl DeviceBuffer<f32> {
    /// Fill with a value
    pub fn fill(&mut self, value: f32) {
        for d in &mut self.data {
            *d = value;
        }
    }

    /// Copy a region from another buffer
    pub fn copy_from(&mut self, src: &DeviceBuffer<f32>, src_offset: usize, dst_offset: usize, len: usize) {
        self.data[dst_offset..dst_offset + len]
            .copy_from_slice(&src.data[src_offset..src_offset + len]);
    }
}

/// Memory transfer operations (simulated)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTransfer;

impl MemoryTransfer {
    /// Host → Device transfer
    pub fn host_to_device<T: Clone + Default>(host_data: &[T]) -> DeviceBuffer<T> {
        let mut buf = DeviceBuffer::allocate(host_data.len());
        buf.write(host_data);
        buf
    }

    /// Device → Host transfer
    pub fn device_to_host<T: Clone + Default>(buffer: &DeviceBuffer<T>) -> Vec<T> {
        buffer.read()
    }

    /// Device → Device copy
    pub fn device_to_device<T: Clone + Default>(src: &DeviceBuffer<T>) -> DeviceBuffer<T> {
        let mut dst = DeviceBuffer::allocate(src.capacity);
        dst.write(&src.read());
        dst
    }
}

/// Memory layout for structured buffer access
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
}

impl MemoryLayout {
    /// Compute flat index for 2D coordinates
    pub fn flat_index(&self, row: usize, col: usize, rows: usize, cols: usize) -> usize {
        match self {
            MemoryLayout::RowMajor => row * cols + col,
            MemoryLayout::ColumnMajor => col * rows + row,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_buffer_allocate() {
        let buf: DeviceBuffer<f32> = DeviceBuffer::allocate(100);
        assert_eq!(buf.capacity(), 100);
        assert!(buf.is_on_device());
    }

    #[test]
    fn test_host_to_device_transfer() {
        let host_data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = MemoryTransfer::host_to_device(&host_data);
        let result = MemoryTransfer::device_to_host(&buf);
        assert_eq!(host_data, result);
    }

    #[test]
    fn test_device_to_device_copy() {
        let buf = MemoryTransfer::host_to_device(&[1.0f32, 2.0, 3.0]);
        let copy = MemoryTransfer::device_to_device(&buf);
        assert_eq!(buf.read(), copy.read());
    }

    #[test]
    fn test_buffer_read_write() {
        let mut buf: DeviceBuffer<f32> = DeviceBuffer::allocate(4);
        buf.write(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(buf.read(), vec![10.0, 20.0, 30.0, 40.0]);
        buf.write_at(2, 99.0);
        assert_eq!(buf.read_at(2), 99.0);
    }

    #[test]
    fn test_memory_layout_row_major() {
        let layout = MemoryLayout::RowMajor;
        // 3x2 matrix: [[a,b],[c,d],[e,f]]
        assert_eq!(layout.flat_index(0, 0, 3, 2), 0);
        assert_eq!(layout.flat_index(0, 1, 3, 2), 1);
        assert_eq!(layout.flat_index(1, 0, 3, 2), 2);
        assert_eq!(layout.flat_index(2, 1, 3, 2), 5);
    }

    #[test]
    fn test_memory_layout_column_major() {
        let layout = MemoryLayout::ColumnMajor;
        // 3x2 matrix
        assert_eq!(layout.flat_index(0, 0, 3, 2), 0);
        assert_eq!(layout.flat_index(1, 0, 3, 2), 1);
        assert_eq!(layout.flat_index(0, 1, 3, 2), 3);
        assert_eq!(layout.flat_index(2, 1, 3, 2), 5);
    }

    #[test]
    fn test_buffer_fill() {
        let mut buf: DeviceBuffer<f32> = DeviceBuffer::allocate(5);
        buf.fill(42.0);
        assert!(buf.read().iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_buffer_copy_region() {
        let src = MemoryTransfer::host_to_device(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let mut dst = DeviceBuffer::allocate(5);
        dst.copy_from(&src, 1, 0, 3); // copy elements 1,2,3 from src to dst[0..3]
        assert_eq!(dst.read_at(0), 2.0);
        assert_eq!(dst.read_at(1), 3.0);
        assert_eq!(dst.read_at(2), 4.0);
    }
}
