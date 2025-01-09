use super::hnsw::HNSW;

// TODO: add errors



pub struct Partition {
    id: usize, 
    hnsw: HNSW, 
    centroid: Vec<f32>
}

pub struct PatritionIndex {
    partitions: Vec<Partition>, 
    num_partitions: usize, 
    dim: usize, 
}

impl PatritionIndex {
    pub fn new(num_partitions: usize, dim: usize) -> Self {
        Self {
            partitions: Vec::with_capacity(num_partitions), 
            num_partitions: num_partitions, 
            dim: dim
        }
    }

    pub fn init_partitions(&mut self, vectors: &[Vec<f32>]) -> Result<(), Error> {
        let centroids = self.select_initial_centroids(vectors)?; 
        for (id, centroid) in centroids.into_iter().enumerate() {
            self.partitions.push(
                Partition {
                    id, 
                    hnsw: HNSW::new(/*params*/), 
                    centroid
                }
            );
        }
        self.assign_vectors_to_partitions(vectors)?; 

        Ok(())
    }

    fn select_initial_centroids(&self, vector: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Error> {

    }

    fn assign_vectors_to_partitions(&mut self, vectors: &[Vec<f32>]) -> Result<(), Error> {

    }

    fn find_nearest_partition(&self, vector: &[f32]) -> Result<(usize, f32), Error> {

    }

}