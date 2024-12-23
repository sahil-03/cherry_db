use std::cmp::Ordering;



// Trait for storing Documents that are associated with respective vectors.
pub trait Document: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> Document for T {} 

pub struct DocumentStore<T: Document> {
    documents: Vec<T>
}

impl<T: Document> DocumentStore<T> {
    pub fn new() -> Self {
        DocumentStore {
            documents: Vec::new(),
        }
    }

    pub fn add_doc(&mut self, document: T) -> usize {
        let id = self.documents.len(); 
        self.documents.push(document);
        id
    }

    pub fn get_doc(&self, doc_id: usize) -> Option<&T> {
        self.documents.get(doc_id)
    }
}


// Trait for priority queue 
struct PriorityElement {
    distance: f32, 
    node_id: usize, 
}

impl Ord for PriorityElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for PriorityElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PriorityElement {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for PriorityElement {}


// Trait for vector storage 
// Cache aligned data storage for all vectors 
#[repr(align(64))]
struct VectorStorage {
    data: Vec<f32>, 
    dim: usize, 
}

impl VectorStorage {
    fn new(dim: usize) -> Self {
        VectorStorage {
            data: Vec::new(), 
            dim, 
        }
    }

    #[inline]
    fn add_vector(&mut self, v: Vec<f32>) -> usize {
        debug_assert_eq!(v.len(), self.dim); 

        let idx = self.data.len() / self.dim; 
        self.data.extend(v); 
        idx 
    }

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim; 
        &self.data[start..start + self.dim]
    }
}
