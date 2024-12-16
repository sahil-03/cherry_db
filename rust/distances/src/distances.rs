// Implement vector distance calculations 

pub trait VecDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32; 
}

pub struct L1Distance; 
pub struct L2Distance; 
pub struct CosDistance; 


// L1-distance (naive)
impl VecDistance for L1Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }
}


// L2-distance 
impl VecDistance for L2Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}


// Cosine distance
impl VecDistance for CosDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let mag_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        dot_product / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*; 

    #[test]
    fn test_l1_distance_serial() {
        let distance_impl = L1Distance; 
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let distance = distance_impl.distance(&v1, &v2);
        assert_eq!(distance, 9.0);
    }

}