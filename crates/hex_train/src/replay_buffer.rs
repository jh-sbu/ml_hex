use std::collections::VecDeque;

use hex_core::CELLS;

/// One training sample produced by self-play.
#[derive(Clone)]
pub struct Experience {
    /// Flat `[3 × 11 × 11 = 363]` f32 board encoding.
    pub state_tensor: Vec<f32>,
    /// Normalised visit count distribution over all `CELLS` positions.
    pub visit_dist: Vec<f32>,
    /// Game outcome from this position's player's perspective: +1.0 or -1.0.
    pub value_target: f32,
}

impl Experience {
    pub fn new(state_tensor: Vec<f32>, visit_dist: Vec<f32>, value_target: f32) -> Self {
        debug_assert_eq!(visit_dist.len(), CELLS);
        Self { state_tensor, visit_dist, value_target }
    }
}

/// A fixed-capacity circular replay buffer.
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an experience, evicting the oldest if the buffer is full.
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
    }

    /// Sample `size` experiences uniformly at random (with replacement).
    pub fn sample_batch(&self, size: usize) -> Vec<&Experience> {
        use rand::prelude::IndexedRandom as _;
        let mut rng = rand::rng();
        let slice: Vec<_> = self.buffer.iter().collect();
        (0..size)
            .map(|_| *slice.choose(&mut rng).expect("buffer non-empty"))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_eviction() {
        let mut buf = ReplayBuffer::new(2);
        let make = |v: f32| {
            Experience::new(vec![0.0; 363], vec![0.0; CELLS], v)
        };
        buf.push(make(1.0));
        buf.push(make(2.0));
        buf.push(make(3.0)); // evicts first
        assert_eq!(buf.len(), 2);
        // Remaining should be value 2.0 and 3.0
        let values: Vec<f32> = buf.buffer.iter().map(|e| e.value_target).collect();
        assert!(values.contains(&2.0));
        assert!(values.contains(&3.0));
        assert!(!values.contains(&1.0));
    }
}
