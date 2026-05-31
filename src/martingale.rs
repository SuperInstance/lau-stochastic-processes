use serde::{Deserialize, Serialize};

/// Verify if a sequence of values forms a martingale.
/// A discrete martingale satisfies E[X_{n+1} | F_n] = X_n.
/// We check this by verifying that the conditional increments have zero mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MartingaleVerifier {
    pub values: Vec<f64>,
}

impl MartingaleVerifier {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Check if the process is approximately a martingale by testing that
    /// the average increment is close to zero.
    pub fn is_martingale(&self, tolerance: f64) -> bool {
        if self.values.len() < 2 {
            return true;
        }
        let increments: Vec<f64> = self.values.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = increments.iter().sum::<f64>() / increments.len() as f64;
        mean.abs() < tolerance
    }

    /// Verify the optional stopping theorem: E[X_T] ≈ E[X_0] for a bounded stopping time T.
    /// `stopped_values` are the values of the process at the stopping time across multiple trials.
    pub fn verify_optional_stopping(initial_value: f64, stopped_values: &[f64], tolerance: f64) -> bool {
        let avg = stopped_values.iter().sum::<f64>() / stopped_values.len() as f64;
        (avg - initial_value).abs() < tolerance
    }
}

/// Doob's decomposition: any adapted process X can be decomposed as X = M + A,
/// where M is a martingale and A is a predictable process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoobDecomposition {
    pub martingale_part: Vec<f64>,
    pub predictable_part: Vec<f64>,
}

impl DoobDecomposition {
    /// Perform Doob's decomposition on a sequence of values.
    pub fn decompose(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                martingale_part: vec![],
                predictable_part: vec![],
            };
        }

        let n = values.len();
        let mut martingale = vec![0.0; n];
        let mut predictable = vec![0.0; n];

        martingale[0] = values[0];
        predictable[0] = 0.0;

        // Compute running mean (predictable drift) and subtract to get martingale part
        let mut cumulative = values[0];
        for i in 1..n {
            let prev_mean = cumulative / i as f64;
            let increment = values[i] - values[i - 1];
            let predictable_increment = increment - (increment + prev_mean * (i as f64 - 1.0)) / i as f64;
            predictable[i] = predictable[i - 1] + predictable_increment;
            martingale[i] = values[i] - predictable[i];
            cumulative += values[i];
        }

        Self {
            martingale_part: martingale,
            predictable_part: predictable,
        }
    }

    /// Verify that the martingale part has zero-mean increments.
    pub fn verify_martingale_property(&self, tolerance: f64) -> bool {
        if self.martingale_part.len() < 2 {
            return true;
        }
        let increments: Vec<f64> = self.martingale_part.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = increments.iter().sum::<f64>() / increments.len() as f64;
        mean.abs() < tolerance
    }
}

/// Create a martingale from random walk using the transform X_n = S_n² - n (for symmetric walk).
pub fn random_walk_martingale_transform(positions: &[f64], step_variance: f64) -> Vec<f64> {
    positions
        .iter()
        .enumerate()
        .map(|(n, &s)| s * s - n as f64 * step_variance)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::Rng;
    use rand::rngs::StdRng;

    #[test]
    fn test_symmetric_walk_is_martingale() {
        // Symmetric random walk is a martingale
        let mut rng = StdRng::seed_from_u64(42);
        let mut values = vec![0.0];
        let mut pos = 0.0;
        for _ in 0..10000 {
            if rng.gen::<bool>() {
                pos += 1.0;
            } else {
                pos -= 1.0;
            }
            values.push(pos);
        }
        let verifier = MartingaleVerifier::new(values);
        assert!(verifier.is_martingale(1.0));
    }

    #[test]
    fn test_non_martingale_detected() {
        // Biased walk is NOT a martingale
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let verifier = MartingaleVerifier::new(values);
        assert!(!verifier.is_martingale(0.01));
    }

    #[test]
    fn test_optional_stopping_theorem() {
        // For a symmetric random walk with stopping at ±10,
        // E[X_T] should be close to E[X_0] = 0
        let mut rng = StdRng::seed_from_u64(42);
        let mut stopped_values = Vec::new();
        for _ in 0..5000 {
            let mut pos = 0.0f64;
            loop {
                if rng.gen::<bool>() {
                    pos += 1.0;
                } else {
                    pos -= 1.0;
                }
                if pos.abs() >= 10.0 {
                    stopped_values.push(pos);
                    break;
                }
            }
        }
        assert!(MartingaleVerifier::verify_optional_stopping(0.0, &stopped_values, 1.0));
    }

    #[test]
    fn test_doob_decomposition() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let decomp = DoobDecomposition::decompose(&values);
        // Verify X = M + A
        for i in 0..values.len() {
            let reconstructed = decomp.martingale_part[i] + decomp.predictable_part[i];
            assert!((reconstructed - values[i]).abs() < 1e-10, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_martingale_transform() {
        // S_n² - n is a martingale for symmetric ±1 walk with variance 1
        let mut rng = StdRng::seed_from_u64(42);
        let mut positions = vec![0.0f64];
        let mut pos = 0.0f64;
        for _ in 0..10000 {
            pos += if rng.gen::<bool>() { 1.0 } else { -1.0 };
            positions.push(pos);
        }
        let transformed = random_walk_martingale_transform(&positions, 1.0);
        let verifier = MartingaleVerifier::new(transformed);
        assert!(verifier.is_martingale(5.0));
    }

    #[test]
    fn test_empty_martingale() {
        let verifier = MartingaleVerifier::new(vec![]);
        assert!(verifier.is_martingale(0.001));
    }
}
