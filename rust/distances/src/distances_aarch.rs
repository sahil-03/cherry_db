// SIMD distance calculation for aarch64 (i.e. m-series mac)

use std::arch::aarch64::*;
use super::distances::VecDistance;

pub struct L1DistanceAARCH;
pub struct L2DistanceAARCH;
pub struct CosDistanceAARCH;

// 64 bytes/cache line and 4 bytes/float --> 64/4 = 16
const CHUNK_LEN: usize = 16;

impl VecDistance for L1DistanceAARCH {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let n: usize = a.len(); 
        assert_eq!(n, b.len()); 

        let n_chunks = n / CHUNK_LEN;
        let mut sum = 0.0;

        unsafe {
            let mut vacc = vdupq_n_f32(0.0); 

            for chunk in 0..n_chunks {
                let idx = chunk * CHUNK_LEN; 

                // Load in NEON vectors 
                for i in 0..4 {
                    let offset = idx + i * 4; 
                    let va = vld1q_f32(a[offset..].as_ptr()); 
                    let vb = vld1q_f32(b[offset..].as_ptr());
                    let vdiff = vabdq_f32(va, vb); 

                    vacc = vaddq_f32(vacc, vdiff); 
                }

                let paired_sums = vpaddq_f32(vacc, vacc); 
                sum += vgetq_lane_f32(paired_sums, 0) + vgetq_lane_f32(paired_sums, 1);
            } 
        }

        let rem = n_chunks * CHUNK_LEN; 
        for i in rem..n {
            sum += (a[i] - b[i]).abs()
        }
        sum
    }
}


impl VecDistance for L2DistanceAARCH {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let n: usize = a.len(); 
        assert_eq!(n, b.len()); 

        let n_chunks = n / CHUNK_LEN;
        let mut sum = 0.0;

        unsafe {
            let mut vacc = vdupq_n_f32(0.0); 

            for chunk in 0..n_chunks {
                let idx = chunk * CHUNK_LEN; 

                // Load in NEON vectors 
                for i in 0..4 {
                    let offset = idx + i * 4; 
                    let va = vld1q_f32(a[offset..].as_ptr()); 
                    let vb = vld1q_f32(b[offset..].as_ptr());
                    let vdiff = vsubq_f32(va, vb); 
                    let vsquared = vmulq_f32(vdiff, vdiff); 

                    vacc = vaddq_f32(vacc, vsquared); 
                }

                let paired_sums = vpaddq_f32(vacc, vacc); 
                sum += vgetq_lane_f32(paired_sums, 0) + vgetq_lane_f32(paired_sums, 1);
            } 
        }

        let rem = n_chunks * CHUNK_LEN; 
        for i in rem..n {
            sum += (a[i] - b[i]).powi(2)
        }
        sum.sqrt()
    }
}


impl VecDistance for CosDistanceAARCH {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let n: usize = a.len(); 
        assert_eq!(n, b.len()); 

        let n_chunks = n / CHUNK_LEN;
        let mut dot_product = 0.0; 
        let mut mag_a = 0.0; 
        let mut mag_b = 0.0; 


        unsafe {
            let mut vacc_dot = vdupq_n_f32(0.0); 
            let mut vacc_a = vdupq_n_f32(0.0); 
            let mut vacc_b = vdupq_n_f32(0.0); 

            for chunk in 0..n_chunks {
                let idx = chunk * CHUNK_LEN; 

                // Load in NEON vectors 
                for i in 0..4 {
                    let offset = idx + i * 4; 
                    let va = vld1q_f32(a[offset..].as_ptr()); 
                    let vb = vld1q_f32(b[offset..].as_ptr());

                    vacc_dot = vmlaq_f32(vacc_dot, va, vb);
                    vacc_a = vmlaq_f32(vacc_a, va, va);
                    vacc_b = vmlaq_f32(vacc_b, vb, vb);
                }

                let paired_dot = vpaddq_f32(vacc_dot, vacc_dot); 
                dot_product += vgetq_lane_f32(paired_dot, 0) + vgetq_lane_f32(paired_dot, 1);

                let paired_a = vpaddq_f32(vacc_a, vacc_a); 
                mag_a += vgetq_lane_f32(paired_a, 0) + vgetq_lane_f32(paired_a, 1);

                let paired_b = vpaddq_f32(vacc_b, vacc_b); 
                mag_b += vgetq_lane_f32(paired_b, 0) + vgetq_lane_f32(paired_b, 1);
            } 
        }

        let rem = n_chunks * CHUNK_LEN; 
        for i in rem..n {
            dot_product += a[i] * b[i]; 
            mag_a += a[i] * a[i];
            mag_b += b[i] * b[i];
        }
        
        mag_a = mag_a.sqrt(); 
        mag_b = mag_b.sqrt(); 

        if mag_a == 0.0 || mag_b == 0.0 {
            return 1.0;
        }

        1.0 - (dot_product / (mag_a * mag_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*; 

    #[test]
    fn test_l1_distance_simd() {
        let distance_impl = L1DistanceAARCH; 
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let distance = distance_impl.distance(&v1, &v2);
        print!("{distance}");
        assert_eq!(distance, 9.0);
    }

}


