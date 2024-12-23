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
    dist: L2DistanceAARCH,

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
    pub fn with_config(dim: usize, config: HNSWConfig) -> Self {

    }

    pub fn new(dim: usize, config: Option<HNSWConfig>) {
        Self::with_config(dim, config.unwrap_or(HNSWConfig::default()))
    }

    // search layer algorithm (adopted from Malkov, Yahunin)
    fn search_layer(
        &self, 
        entry_point: usize, 
        query: &[f32], 
        ef_construction: usize, 
        level: usize, 
        visited: &mut HashSet<PriorityElement>
    ) -> BinaryHeap<PriorityElement> {
        let mut candidates = BinaryHeap::new(); 
        let mut results = BinaryHeap::new(); 

        // TODO replace with distance metric
        let d = self.dist.distance(self.vectors.get_vector(entry_point), query);
        candidates.push(PriorityElement { distance: -d, node_id: entry_point });
        results.push(PriorityElement { distance: d, node_id: entry_point });
        visited.insert(entry_point);

        while !candidates.is_empty() {
            let PriorityElement { distance: cur_d, node_id: cur_id } = candidates.pop().unwrap(); 
            let furthest_d = results.peek().unwrap().distance; 

            if -cur_d > furthest_d {
                break;
            }

            for &neigh_id in self.neighbors.get_neighbors(level, current) {
                if visited.insert(neigh_id) {
                    let neigh_d = self.dist.distance(self.vectors.get_vector(neigh_id), query);

                    if results.len() < ef_construction || neigh_d < furthest_d {
                        candidates.push(PriorityElement { distance: -neigh_d, node_id: neigh_id });
                        results.push(PriorityElement { distance: neigh_d, node_id: neigh_id });

                        if results.len() > ef_construction {
                            results.pop();
                        }
                    }
                }
            }
        }
        results
    }

    // insert algorithm (adopted from Malkov, Yahunin)
    pub fn insert(&mut self, vector: Vec<f32>, document: T) -> usize {
        let node_id = self.vectors.add_vector(vector); 
        let doc_id = self.documents.len(); 
        self.documents.push(document); 

        let level = self.generate_random_level(); 
        self.levels.push(level); 
        self.doc_ids.push(doc_id); 
        self.neighbors.add_node(level);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id); 
            return node_id;
        }

        let mut cur_ep = self.entry_point.unwrap();
        let mut cur_d = self.dist.distance(
            self.vectors.get_vector(cur_ep), self.vectors.get_vector(node_id)
        );

        for lc in (1..=level).rev() {
            let mut changed = true; 
            while changed {
                changed = false; 

                for &neigh_id in self.neighbors.get_neighbors(lc, cur_ep) {
                    let d = self.dist.distance(
                        self.vectors.get_vector(neigh_id), self.vectors.get_vector(node_id)
                    ); 
                    if d < cur_d {
                        cur_d = d;
                        cur_ep = neigh_id; 
                        changed = true;
                    }
                }
            }
        }

        let mut visited = HashSet::new(); 
        for lc in (0..=level).rev() {
            let ef = if lc == 0 { self.config.m0 } else { self.config.m }; 
            let mut neighbors = self.search_layer(
                cur_ep, 
                self.vectors.get_vector(node_id), 
                self.config.ef_construction, 
                lc,
                &mut visited
            );

            while neighbors.len() > ef {
                neighbors.pop();
            }

            for neigh in neighbors {
                self.neighbors.add_neighbor(lc, node_id, neigh.node_id); 
                self.neighbors.add_neighbor(lc, neigh.node_id, node_id); 
            }
        }

        if level > self.levels[self.entry_point.unwrap()] {
            self.entry_point = Some(node_id);
        }
        node_id
    }

    pub fn batch_insert(&mut self, items: Vec<(Vec<f32>, T)>) {
        let entry = self.entry_point.map(|ep| self.vectors.get_vector(ep).to_vec()); 

        // pre-allocate space for batch insert 
        let n = items.len(); 
        self.vectors.data.reserve(n * self.vectors.dim); 
        self.levels.reserve(n);
        self.doc_ids.reserve(n);
        self.documents.reserve(n);

        // sort by proximity to entry point
        let mut sorted_items = items; 
        if let Some(entry_vector) = entry { 
            sorted_items.sort_by_cached_key(|(v, _)| {
                self.dist.distance(v, &entry_vector) as i64
            });
        }

        for (vector, document) in sorted_items {
            self.insert(vector, document);
        }
    }

    pub fn search(&self, query: &[f32], k: size) -> Vec<(T, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut cur_ep = self.entry_point.unwrap(); 
        let mut cur_d = self.dist.distance(self.vectors.get_vector(cur_ep), query);
        let ep_level = self.levels[cur_ep]; 

        for level in (1..=ep_level).rev() {
            let mut changed = true; 
            while changed {
                changed = false; 

                for &neigh_id in self.neighbors.get_neighbors(level, cur_ep) {
                    let d = self.dist.distance(self.vectors.get_vector(neigh_id), query); 
                    if d < cur_d {
                        cur_d = d; 
                        cur_ep = neigh_id; 
                        changed = true;
                    }
                }
            }
        }

        let mut visited = HashSet::new(); 
        let nearest = self.search_layer(cur_ep, query, k, 0, &mut visited); 

        nearest.into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|e| {
                let doc_id = self.doc_ids[e.node_id]; 
                (self.documents[doc_id].clone(), e.distance)
            })
            .collect()
    }

    #[inline]
    fn generate_random_level(&mut self) -> usize {
        // TODO
    }

    pub fn len(&self) -> usize {
        self.vectors.data.len() / self.vectors.dim
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.data.is_empty()
    }
}