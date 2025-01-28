#![allow(unused)]
#![feature(stdsimd)]
use std::{
    alloc::Layout, cmp::Reverse, collections::BinaryHeap, fmt::Binary, mem, ptr, simd::{f32x16, SimdFloat}, sync::atomic::AtomicUsize
}; 

use super::distances::*;
use bitvec::prelude::*; 
use rand::{rngs::SmallRng, SeedableRng}; 
use wyrand::WyRnad; 



// --------------------------------
// | Cache-Optimized Data Layouts |
// --------------------------------

#[repr(align(64))]
struct AlignedVec {
    data: Vec<f32>, 
    dim: usize,
}

impl AlignedVec {
    #[inline]
    fn with_capacity(dim: usize, capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity * dim); 
        data.resize(capacity * dim, 0.0); 
        Self { data, dim }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> *const f32 {
        self.data.as_ptr().wrapping_add(idx * self.dim)
    }
}


// --------------------
// | HNSW Core Engine |
// --------------------

struct HNSWConfig {
    max_layers: usize, 
    m: usize,
    ef_construction: usize, 
    ml_factor: f32,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        HNSWConfig {
            max_layers: 12, 
            m: 24, 
            ef_construction: 120, 
            ml_factor: 1.0 / 16.0f32.ln()
        }
    }
}


struct HNSW<D: CosDistanceAARCH> {
    vectors: AlignedVec, 
    neighbors: Vec<Vec<Vec<u32>>>, // [layer][node][neighbors]
    entry_point: AtomicUsize,
    config: HNSWConfig, 
    rng: WyRand, 
    _marker: std::marker::PhantomData<D>
}

impl<D: CosDistanceAARCH> HNSW<D> {
    pub fn new(dim: usize) -> Self {
        let config = HNSWConfig::default(); 
        let mut neighbors = Vec::with_capacity(conig.max_layers); 
        for _ in 0..config.max_layers {
            neighbors.push(Vec::with_capacity(1024))
        }

        Self {
            vectors: AlignedVec(dim, 1024), 
            neighbors: neighbors, 
            entry_point: AtomicUsize::new(0), 
            config,
            rng: WyRand::new(), 
            _marker: std::marker::PhantomData
        }
    }

    #[inline]
    pub fn insert(&mut self, vector: &[f32]) {

    }

    #[inline(always)]
    fn insert_connections(&mut self, idx: usize, layer: usize, ep: usize) {

    }

    #[inline(always)]
    unsafe fn search_layer(
        &self, 
        ep: usize, 
        target: usize, 
        layer: usize,
        candidates: &mut BinaryHeap<(Reverse<f32>, usize)>, 
        visited: &mut BitVec
    ) {

    }

    #[inline]
    fn random_level(&mut self) -> usize {

    }
    
    #[inline]
    fn get_node_level(&self, node: usize) -> usize {

    }

    pub fn batch_insert(&mut self, vectors: &[&[f32]]) {

    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<usize> {

    }
}