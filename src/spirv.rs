//! Vulkan compute shader generation basics — SPIR-V-like IR.

use serde::{Serialize, Deserialize};

/// SPIR-V-like intermediate representation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpirvType {
    Void,
    Float32,
    Int32,
    UInt32,
    Vector { scalar: Box<SpirvType>, len: u32 },
    Array { element: Box<SpirvType>, len: u32 },
    Struct { fields: Vec<(String, SpirvType)> },
    Pointer { target: Box<SpirvType>, storage: StorageClass },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StorageClass {
    Uniform,
    StorageBuffer,
    PushConstant,
    Workgroup,
    Private,
    Function,
}

/// SPIR-V-like instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpirvInstruction {
    // Arithmetic
    FAdd { result: u32, ty: SpirvType, a: u32, b: u32 },
    FSub { result: u32, ty: SpirvType, a: u32, b: u32 },
    FMul { result: u32, ty: SpirvType, a: u32, b: u32 },
    FDiv { result: u32, ty: SpirvType, a: u32, b: u32 },

    // Built-in
    Load { result: u32, ty: SpirvType, pointer: u32 },
    Store { pointer: u32, value: u32 },
    AccessChain { result: u32, base: u32, indices: Vec<u32> },

    // Control flow
    Label { id: u32 },
    Branch { target: u32 },
    BranchConditional { condition: u32, true_block: u32, false_block: u32 },
    Loop { merge: u32, continue_block: u32 },
    Return,
    ReturnValue { value: u32 },

    // Built-in variables
    GlobalInvocationId,
    LocalInvocationId,
    WorkgroupId,

    // Barrier
    ControlBarrier,
    MemoryBarrier { memory_scope: u32, semantics: u32 },

    // Composite
    CompositeConstruct { result: u32, ty: SpirvType, constituents: Vec<u32> },
    CompositeExtract { result: u32, composite: u32, indices: Vec<u32> },
}

/// SPIR-V-like module builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvBuilder {
    instructions: Vec<SpirvInstruction>,
    next_id: u32,
    name: String,
}

impl SpirvBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            instructions: Vec::new(),
            next_id: 1,
            name: name.to_string(),
        }
    }

    /// Allocate a new result ID
    pub fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Push an instruction
    pub fn emit(&mut self, inst: SpirvInstruction) {
        self.instructions.push(inst);
    }

    /// Get all instructions
    pub fn instructions(&self) -> &[SpirvInstruction] {
        &self.instructions
    }

    /// Number of instructions
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Build a simple vector add kernel
    pub fn build_vector_add(&mut self) -> u32 {
        let result_id = self.next_id();
        self.emit(SpirvInstruction::FAdd {
            result: result_id,
            ty: SpirvType::Float32,
            a: 1,
            b: 2,
        });
        result_id
    }

    /// Build a reduce kernel pattern
    pub fn build_reduce_pattern(&mut self, workgroup_size: u32) {
        let thread_id = self.next_id();
        self.emit(SpirvInstruction::Load {
            result: thread_id,
            ty: SpirvType::UInt32,
            pointer: 0, // placeholder
        });

        // Stride loop (tree reduction)
        let mut stride = 1u32;
        while stride < workgroup_size {
            let partner = self.next_id();
            self.emit(SpirvInstruction::FAdd {
                result: partner,
                ty: SpirvType::Float32,
                a: thread_id,
                b: stride,
            });
            self.emit(SpirvInstruction::ControlBarrier);
            stride *= 2;
        }
    }

    /// Module name
    pub fn name(&self) -> &str { &self.name }

    /// Generate a pseudo-assembly string
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("; SPIR-V module: {}\n", self.name));
        for (i, inst) in self.instructions.iter().enumerate() {
            out.push_str(&format!("{:04}: {:?}\n", i, inst));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spirv_builder_basic() {
        let mut builder = SpirvBuilder::new("test");
        let id = builder.build_vector_add();
        assert_eq!(id, 1);
        assert_eq!(builder.instruction_count(), 1);
    }

    #[test]
    fn test_spirv_type_hierarchy() {
        let vec4 = SpirvType::Vector {
            scalar: Box::new(SpirvType::Float32),
            len: 4,
        };
        if let SpirvType::Vector { len, .. } = vec4 {
            assert_eq!(len, 4);
        }
    }

    #[test]
    fn test_spirv_reduce_pattern() {
        let mut builder = SpirvBuilder::new("reduce");
        builder.build_reduce_pattern(256);
        assert!(builder.instruction_count() > 0);
    }

    #[test]
    fn test_spirv_disassemble() {
        let mut builder = SpirvBuilder::new("simple");
        builder.emit(SpirvInstruction::FAdd {
            result: 1, ty: SpirvType::Float32, a: 0, b: 0,
        });
        builder.emit(SpirvInstruction::Return);
        let asm = builder.disassemble();
        assert!(asm.contains("SPIR-V"));
        assert!(asm.contains("FAdd"));
    }

    #[test]
    fn test_spirv_serialization() {
        let mut builder = SpirvBuilder::new("serde_test");
        builder.build_vector_add();
        let json = serde_json::to_string(&builder).unwrap();
        let deserialized: SpirvBuilder = serde_json::from_str(&json).unwrap();
        assert_eq!(builder.instruction_count(), deserialized.instruction_count());
    }
}
