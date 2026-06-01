//! Agent parallel inference — batch agent computations via GPU patterns.

use serde::{Serialize, Deserialize};
use crate::tensor::Tensor;
use crate::primitives::{tiled_matmul, prefix_sum};
use crate::cuda::CudaGrid;

/// A single agent's state for batch inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub id: u64,
    pub hidden: Vec<f32>,
    pub output: Vec<f32>,
}

/// Batch of agent states for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatch {
    pub agents: Vec<AgentState>,
    pub hidden_dim: usize,
}

impl AgentBatch {
    pub fn new(batch_size: usize, hidden_dim: usize) -> Self {
        Self {
            agents: (0..batch_size)
                .map(|i| AgentState {
                    id: i as u64,
                    hidden: vec![0.0; hidden_dim],
                    output: vec![0.0; hidden_dim],
                })
                .collect(),
            hidden_dim,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.agents.len()
    }

    /// Stack all hidden states into a [batch, hidden] tensor
    pub fn stack_hidden(&self) -> Tensor<f32> {
        let bs = self.batch_size();
        let h = self.hidden_dim;
        let mut data = Vec::with_capacity(bs * h);
        for agent in &self.agents {
            data.extend_from_slice(&agent.hidden);
        }
        Tensor::new(vec![bs, h], data)
    }

    /// Write back hidden states from tensor
    pub fn unstack_hidden(&mut self, tensor: &Tensor<f32>) {
        let h = self.hidden_dim;
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.hidden.copy_from_slice(&tensor.data[i * h..(i + 1) * h]);
        }
    }

    /// Write back outputs from tensor
    pub fn unstack_output(&mut self, tensor: &Tensor<f32>) {
        let h = self.hidden_dim;
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.output.copy_from_slice(&tensor.data[i * h..(i + 1) * h]);
        }
    }
}

/// Simple agent inference engine using GPU compute patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInferenceEngine {
    pub weight_input: Tensor<f32>,
    pub weight_hidden: Tensor<f32>,
    pub hidden_dim: usize,
}

impl AgentInferenceEngine {
    pub fn new(hidden_dim: usize) -> Self {
        // Initialize weights (small random-ish values)
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let make_weights = |rows: usize, cols: usize| {
            let mut data = Vec::with_capacity(rows * cols);
            for i in 0..rows * cols {
                // Deterministic pseudo-random using sin
                let val = ((i as f32 * 0.123 + 0.456) * 1000.0).sin() * scale;
                data.push(val);
            }
            Tensor::new(vec![rows, cols], data)
        };

        Self {
            weight_input: make_weights(hidden_dim, hidden_dim),
            weight_hidden: make_weights(hidden_dim, hidden_dim),
            hidden_dim,
        }
    }

    /// Run one step of batched agent inference
    pub fn step(&self, batch: &mut AgentBatch, input: &Tensor<f32>) {
        let bs = batch.batch_size();
        let h = self.hidden_dim;

        // hidden_new = input @ W_input + hidden @ W_hidden
        let input_proj = tiled_matmul(input, &self.weight_input, 16);
        let hidden = batch.stack_hidden();
        let hidden_proj = tiled_matmul(&hidden, &self.weight_hidden, 16);

        // Add (residual) with ReLU activation
        let mut new_hidden = Vec::with_capacity(bs * h);
        for i in 0..bs * h {
            let val = input_proj.data[i] + hidden_proj.data[i];
            new_hidden.push(val.max(0.0)); // ReLU
        }

        let new_hidden_tensor = Tensor::new(vec![bs, h], new_hidden);
        batch.unstack_hidden(&new_hidden_tensor);
        batch.unstack_output(&new_hidden_tensor);
    }

    /// Compute attention-like scores across agents (softmax pattern)
    pub fn attention_scores(&self, batch: &AgentBatch) -> Vec<f32> {
        let hidden = batch.stack_hidden();
        let bs = batch.batch_size();
        let h = self.hidden_dim;

        // Dot product similarity: score[i][j] = hidden[i] . hidden[j]
        let mut scores = Vec::with_capacity(bs * bs);
        for i in 0..bs {
            for j in 0..bs {
                let mut dot = 0.0f32;
                for k in 0..h {
                    dot += hidden.data[i * h + k] * hidden.data[j * h + k];
                }
                scores.push(dot);
            }
        }

        // Softmax per row
        for i in 0..bs {
            let row_start = i * bs;
            let max_val = scores[row_start..row_start + bs]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0;
            for j in 0..bs {
                scores[row_start + j] = (scores[row_start + j] - max_val).exp();
                exp_sum += scores[row_start + j];
            }
            for j in 0..bs {
                scores[row_start + j] /= exp_sum;
            }
        }

        scores
    }

    /// Launch configuration for this batch
    pub fn launch_config(&self, batch: &AgentBatch) -> CudaGrid {
        CudaGrid::launch_1d(batch.batch_size() as u32 * self.hidden_dim as u32, 256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_batch_creation() {
        let batch = AgentBatch::new(4, 8);
        assert_eq!(batch.batch_size(), 4);
        assert_eq!(batch.hidden_dim, 8);
    }

    #[test]
    fn test_agent_batch_stack_unstack() {
        let mut batch = AgentBatch::new(2, 3);
        batch.agents[0].hidden = vec![1.0, 2.0, 3.0];
        batch.agents[1].hidden = vec![4.0, 5.0, 6.0];

        let stacked = batch.stack_hidden();
        assert_eq!(stacked.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(stacked.shape, vec![2, 3]);

        let mut batch2 = AgentBatch::new(2, 3);
        batch2.unstack_hidden(&stacked);
        assert_eq!(batch2.agents[0].hidden, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_agent_inference_step() {
        let mut batch = AgentBatch::new(4, 16);
        let engine = AgentInferenceEngine::new(16);

        let input = Tensor::new(vec![4, 16], vec![1.0; 64]);
        engine.step(&mut batch, &input);

        // After step, hidden should have changed from zeros
        let has_nonzero = batch.agents.iter()
            .any(|a| a.hidden.iter().any(|&v| v != 0.0));
        assert!(has_nonzero);
    }

    #[test]
    fn test_agent_attention_scores() {
        let mut batch = AgentBatch::new(3, 4);
        for (i, agent) in batch.agents.iter_mut().enumerate() {
            agent.hidden = vec![i as f32; 4];
        }

        let engine = AgentInferenceEngine::new(4);
        let scores = engine.attention_scores(&batch);

        assert_eq!(scores.len(), 9); // 3x3

        // Each row should sum to ~1.0 (softmax)
        for i in 0..3 {
            let row_sum: f32 = scores[i * 3..(i + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_agent_launch_config() {
        let batch = AgentBatch::new(8, 64);
        let engine = AgentInferenceEngine::new(64);
        let grid = engine.launch_config(&batch);
        assert!(grid.total_threads() >= 8 * 64);
    }

    #[test]
    fn test_agent_batch_serialization() {
        let batch = AgentBatch::new(2, 4);
        let json = serde_json::to_string(&batch).unwrap();
        let deserialized: AgentBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(batch.batch_size(), deserialized.batch_size());
    }
}
