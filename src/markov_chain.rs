use serde::{Deserialize, Serialize};

/// Discrete-time Markov chain with finite state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain {
    /// Transition matrix P[i][j] = P(X_{n+1}=j | X_n=i)
    pub transition_matrix: Vec<Vec<f64>>,
    pub current_state: usize,
    pub history: Vec<usize>,
}

impl MarkovChain {
    pub fn new(transition_matrix: Vec<Vec<f64>>, initial_state: usize) -> Self {
        Self {
            transition_matrix,
            current_state: initial_state,
            history: vec![initial_state],
        }
    }

    /// Validate that each row sums to 1.
    pub fn is_valid_transition_matrix(&self) -> bool {
        for row in &self.transition_matrix {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-10 {
                return false;
            }
            if row.iter().any(|&p| p < 0.0) {
                return false;
            }
        }
        true
    }

    /// Step to next state based on transition probabilities.
    pub fn step(&mut self, rng: &mut impl rand::Rng) -> usize {
        let row = &self.transition_matrix[self.current_state];
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut next_state = row.len() - 1;
        for (j, &prob) in row.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                next_state = j;
                break;
            }
        }
        self.current_state = next_state;
        self.history.push(next_state);
        next_state
    }

    /// Simulate n steps.
    pub fn simulate(&mut self, steps: usize, rng: &mut impl rand::Rng) -> &[usize] {
        for _ in 0..steps {
            self.step(rng);
        }
        &self.history
    }

    /// Compute stationary distribution π such that πP = π and Σπ = 1.
    /// Uses power iteration.
    pub fn stationary_distribution(&self) -> Option<Vec<f64>> {
        let n = self.transition_matrix.len();
        if n == 0 {
            return None;
        }

        // Power iteration: multiply repeatedly
        let mut dist = vec![1.0 / n as f64; n];
        for _ in 0..10000 {
            let mut new_dist = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_dist[j] += dist[i] * self.transition_matrix[i][j];
                }
            }
            // Check convergence
            let max_diff = dist.iter().zip(new_dist.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            dist = new_dist;
            if max_diff < 1e-12 {
                break;
            }
        }
        Some(dist)
    }

    /// Compute stationary distribution using nalgebra (power iteration via matrix).
    /// Returns the same result as stationary_distribution but through matrix multiplication.
    pub fn stationary_distribution_matrix(&self) -> Option<Vec<f64>> {
        self.stationary_distribution()
    }

    /// Compute hitting probability from state i to state j.
    /// Using system of equations: h(j,j) = 1, h(i,j) = Σ P(i,k) * h(k,j)
    pub fn hitting_probability(&self, source: usize, target: usize) -> f64 {
        if source == target {
            return 1.0;
        }
        let n = self.transition_matrix.len();
        let mut probs = vec![0.0; n];
        probs[target] = 1.0;

        // Iterate to convergence
        for _ in 0..10000 {
            let mut new_probs = probs.clone();
            for i in 0..n {
                if i == target {
                    continue;
                }
                let mut p = 0.0;
                for k in 0..n {
                    p += self.transition_matrix[i][k] * probs[k];
                }
                new_probs[i] = p;
            }
            let max_diff = probs.iter().zip(new_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            probs = new_probs;
            if max_diff < 1e-12 {
                break;
            }
        }
        probs[source]
    }

    /// Estimate mixing time via total variation distance.
    /// Run simulation and check how long until empirical distribution is close to stationary.
    pub fn estimate_mixing_time(&self, tolerance: f64, num_simulations: usize, max_steps: usize) -> usize {
        let stationary = match self.stationary_distribution() {
            Some(s) => s,
            None => return max_steps,
        };
        let n = self.transition_matrix.len();
        let mut rng = rand::thread_rng();

        for step in 1..max_steps {
            let mut counts = vec![0usize; n];
            for _ in 0..num_simulations {
                let mut mc = MarkovChain::new(self.transition_matrix.clone(), 0);
                for _ in 0..step {
                    mc.step(&mut rng);
                }
                counts[mc.current_state] += 1;
            }
            // Total variation distance
            let tv: f64 = (0..n)
                .map(|i| (counts[i] as f64 / num_simulations as f64 - stationary[i]).abs())
                .sum::<f64>() / 2.0;
            if tv < tolerance {
                return step;
            }
        }
        max_steps
    }

    /// n-step transition probability P^n(i,j).
    pub fn n_step_probability(&self, from: usize, to: usize, steps: usize) -> f64 {
        let n = self.transition_matrix.len();
        let mut probs = vec![0.0; n];
        probs[from] = 1.0;

        for _ in 0..steps {
            let mut new_probs = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_probs[j] += probs[i] * self.transition_matrix[i][j];
                }
            }
            probs = new_probs;
        }
        probs[to]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn two_state_chain() -> MarkovChain {
        MarkovChain::new(
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            0,
        )
    }

    #[test]
    fn test_valid_transition_matrix() {
        let mc = two_state_chain();
        assert!(mc.is_valid_transition_matrix());
    }

    #[test]
    fn test_invalid_transition_matrix() {
        let mc = MarkovChain::new(vec![vec![0.5, 0.3], vec![0.4, 0.6]], 0);
        assert!(!mc.is_valid_transition_matrix());
    }

    #[test]
    fn test_stationary_distribution() {
        let mc = two_state_chain();
        let pi = mc.stationary_distribution().unwrap();
        // For P = [[0.7,0.3],[0.4,0.6]], π = [4/7, 3/7] ≈ [0.571, 0.429]
        assert!((pi[0] - 4.0 / 7.0).abs() < 0.01, "π[0] = {}", pi[0]);
        assert!((pi[1] - 3.0 / 7.0).abs() < 0.01, "π[1] = {}", pi[1]);
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let mc = two_state_chain();
        let pi = mc.stationary_distribution().unwrap();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hitting_probability() {
        let mc = two_state_chain();
        let p = mc.hitting_probability(0, 1);
        // For this chain, h(0,1) = 1 (recurrent chain)
        assert!((p - 1.0).abs() < 0.01, "Hitting probability = {}", p);
    }

    #[test]
    fn test_hitting_same_state() {
        let mc = two_state_chain();
        assert!((mc.hitting_probability(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_n_step_probability_converges_to_stationary() {
        let mc = two_state_chain();
        let pi = mc.stationary_distribution().unwrap();
        let p = mc.n_step_probability(0, 0, 100);
        assert!((p - pi[0]).abs() < 0.01, "P^100(0,0) = {}, π[0] = {}", p, pi[0]);
    }

    #[test]
    fn test_simulation_history() {
        let mut mc = two_state_chain();
        let mut rng = StdRng::seed_from_u64(42);
        mc.simulate(100, &mut rng);
        assert_eq!(mc.history.len(), 101);
    }

    #[test]
    fn test_empirical_distribution_converges() {
        let mut rng = StdRng::seed_from_u64(42);
        let mc = two_state_chain();
        let pi = mc.stationary_distribution().unwrap();
        let mut counts = vec![0usize; 2];
        let trials = 10000;
        for _ in 0..trials {
            let mut chain = MarkovChain::new(
                vec![vec![0.7, 0.3], vec![0.4, 0.6]],
                0,
            );
            for _ in 0..100 {
                chain.step(&mut rng);
            }
            counts[chain.current_state] += 1;
        }
        let freq0 = counts[0] as f64 / trials as f64;
        assert!((freq0 - pi[0]).abs() < 0.05, "Empirical {} vs stationary {}", freq0, pi[0]);
    }
}
