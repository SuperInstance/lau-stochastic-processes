use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Wiener process (standard Brownian motion) simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrownianMotion {
    pub path: Vec<f64>,
    pub times: Vec<f64>,
    pub dt: f64,
}

impl BrownianMotion {
    pub fn new(dt: f64) -> Self {
        Self {
            path: vec![0.0],
            times: vec![0.0],
            dt,
        }
    }

    pub fn step(&mut self, rng: &mut impl Rng) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dW = normal.sample(rng) * self.dt.sqrt();
        let new_val = self.path.last().unwrap() + dW;
        let new_time = self.times.last().unwrap() + self.dt;
        self.path.push(new_val);
        self.times.push(new_time);
        new_val
    }

    pub fn simulate(&mut self, steps: usize, rng: &mut impl Rng) -> &[f64] {
        for _ in 0..steps {
            self.step(rng);
        }
        &self.path
    }

    /// Brownian bridge from (0, a) to (T, b).
    /// X(t) = a(1-t/T) + b(t/T) + W(t) - (t/T)W(T)
    pub fn bridge(
        dt: f64,
        total_time: f64,
        start: f64,
        end: f64,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, Vec<f64>) {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let steps = (total_time / dt) as usize;
        let mut bm = vec![0.0];
        let mut w = 0.0;
        for _ in 0..steps {
            w += normal.sample(rng) * dt.sqrt();
            bm.push(w);
        }
        let w_T = *bm.last().unwrap();
        let mut path = Vec::new();
        let mut times = Vec::new();
        for (i, &w_t) in bm.iter().enumerate() {
            let t = i as f64 * dt;
            let ratio = t / total_time;
            let x = start * (1.0 - ratio) + end * ratio + w_t - ratio * w_T;
            path.push(x);
            times.push(t);
        }
        (path, times)
    }

    /// Reflection principle: P(max_{0≤s≤t} W(s) ≥ a) = 2 * P(W(t) ≥ a)
    pub fn reflection_probability(t: f64, a: f64) -> f64 {
        // P(W(t) ≥ a) = 0.5 * erfc(a / sqrt(2t))
        let z = a / (2.0 * t).sqrt();
        1.0 - 0.5 * (1.0 + erf(z))
    }

    /// Hitting time distribution for Brownian motion.
    /// P(T_a ≤ t) = 2 * P(W(t) ≥ a) = 2 * (1 - Φ(a/√t))
    pub fn hitting_time_cdf(t: f64, a: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let z = a / t.sqrt();
        2.0 * (1.0 - phi(z))
    }

    /// Expected hitting time of level a: E[T_a] = ∞ for standard BM,
    /// but for BM with drift μ: E[T_a] = a/μ if μ > 0.
    pub fn expected_hitting_time_with_drift(a: f64, drift: f64) -> f64 {
        if drift > 0.0 { a / drift } else { f64::INFINITY }
    }

    /// Geometric Brownian motion: dS = μS dt + σS dW
    pub fn gbm(
        s0: f64,
        mu: f64,
        sigma: f64,
        dt: f64,
        steps: usize,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut path = vec![s0];
        let mut s = s0;
        for _ in 0..steps {
            let dW = normal.sample(rng) * dt.sqrt();
            s = s * ((mu - 0.5 * sigma * sigma) * dt + sigma * dW).exp();
            path.push(s);
        }
        path
    }
}

/// Standard normal CDF (approximation).
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_brownian_motion_starts_at_zero() {
        let bm = BrownianMotion::new(0.01);
        assert_eq!(bm.path[0], 0.0);
    }

    #[test]
    fn test_brownian_motion_scaling() {
        // W(t) ~ N(0, t), so Var(W(t)) ≈ t
        let dt = 0.01;
        let steps = 1000; // t = 10
        let num_trials = 2000;
        let mut rng = StdRng::seed_from_u64(42);
        let mut final_values = Vec::new();
        for _ in 0..num_trials {
            let mut bm = BrownianMotion::new(dt);
            bm.simulate(steps, &mut rng);
            final_values.push(*bm.path.last().unwrap());
        }
        let mean = final_values.iter().sum::<f64>() / num_trials as f64;
        let variance = final_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / num_trials as f64;
        let expected_time = dt * steps as f64;
        assert!(mean.abs() < 0.5, "Mean {} should be near 0", mean);
        assert!((variance - expected_time).abs() / expected_time < 0.3,
            "Variance {} should be near {}", variance, expected_time);
    }

    #[test]
    fn test_brownian_bridge_ends_at_target() {
        let mut rng = StdRng::seed_from_u64(42);
        let (path, _) = BrownianMotion::bridge(0.01, 1.0, 0.0, 5.0, &mut rng);
        assert!((path.last().unwrap() - 5.0).abs() < 1e-10);
        assert!((path.first().unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflection_probability() {
        // P(max W ≥ a) should be between 0 and 1
        let p = BrownianMotion::reflection_probability(1.0, 1.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_hitting_time_cdf_increasing() {
        let p1 = BrownianMotion::hitting_time_cdf(1.0, 1.0);
        let p2 = BrownianMotion::hitting_time_cdf(2.0, 1.0);
        assert!(p2 > p1, "CDF should be increasing");
    }

    #[test]
    fn test_hitting_time_drift() {
        let t = BrownianMotion::expected_hitting_time_with_drift(10.0, 2.0);
        assert!((t - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gbm_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        let path = BrownianMotion::gbm(100.0, 0.05, 0.2, 0.01, 1000, &mut rng);
        assert!(path.iter().all(|&s| s > 0.0), "GBM should stay positive");
    }

    #[test]
    fn test_bm_increments_independent() {
        // Test that increments have approximately the right variance
        let dt = 0.01;
        let mut bm = BrownianMotion::new(dt);
        let mut rng = StdRng::seed_from_u64(42);
        bm.simulate(5000, &mut rng);
        let increments: Vec<f64> = bm.path.windows(2).map(|w| w[1] - w[0]).collect();
        let var = increments.iter().map(|x| x * x).sum::<f64>() / increments.len() as f64;
        assert!((var - dt).abs() / dt < 0.2, "Increment variance {} should be near {}", var, dt);
    }
}
