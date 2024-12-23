use std::collections::{BinaryHeap, HashSet};
use index::utils::*;
use distances::distances_aarch::*; 

#[derive(Clone)]
pub struct HNSWConfig {
    pub max_level: usize, 
    pub m: usize, 
    pub m0: usize, 
    pub ef_construction: usize, 
    pub ml_factor: f32
}


// TODO: create another way to do db config
impl Default for HNSWConfig {
    fn default() -> Self {
        HNSWConfig {
            max_level: 16,
            m: 16,
            m0: 32,
            ef_construction: 100,
            ml_factor: 1.0 / (16_f32.ln()),
        }
    }
}


struct NeighborStorage {
    neighbors: Vec<Vec<usize>>, 
    offsets: Vec<Vec<usize>>, 
    counts: Vec<Vec<usize>>,
}

impl NeighborStorage {
    fn new(max_level: usize) -> Self {
        NeighborStorage {
            neighbors: vec![Vec::new(); max_level+1],
            offsets: vec![Vec::new(); max_level+1],
            counts: vec![Vec::new(); max_level+1],
        }
    }

    fn add_node(&mut self, level: usize) {
        for l in 0..=level {
            self.offsets[l].push(self.neighbors[l].len());
            self.counts[l].push(0)
        }
    }

    #[inline]
    fn get_neighbors(&self, level: usize, node_id: usize) -> &[usize] {
        let start = self.offsets[level][node_id]; 
        let count = self.counts[level][node_id];
        &self.neighbors[level][start..start + count]
    }

    fn add_neighbor(&mut self, level: usize, node_id: usize, neighbor: usize) {
        let offset = self.offsets[level][node_id];
        let count = &mut self.counts[level][node_id]; 

        if offset + *count >= self.neighbors[level].len() {
            self.neighbors[level].push(neighbor);
        } else {
            self.neighbors[level][offset + *count] = neighbor;
        }
        *count += 1;
    }
}

// Actual HNSW index 
pub struct HNSW<T: Document> {
    vectors: VectorStorage, 
    neighbors: NeighborStorage,

    // node metadata 
    levels: Vec<usize>, 
    doc_ids: Vec<usize>, 
    documents: Vec<T>, 

    // index metadata
    entry_point: Option<usize>, 
    config: HNSWConfig, 
    rng: rand::rngs::SmallRng,
}

impl<T: Document> HNSW<T> {
    // TODO...
}