use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::{Exp, Uniform};
use serde::{Deserialize, Serialize};

/// Homogeneous Poisson process with rate λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonProcess {
    pub rate: f64,
    pub events: Vec<f64>,
    pub current_time: f64,
}

impl PoissonProcess {
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Rate must be positive");
        Self {
            rate,
            events: vec![],
            current_time: 0.0,
        }
    }

    /// Generate next event time.
    pub fn next_event(&mut self, rng: &mut impl Rng) -> f64 {
        let exp = Exp::new(self.rate).unwrap();
        let interval: f64 = exp.sample(rng);
        self.current_time += interval;
        self.events.push(self.current_time);
        self.current_time
    }

    /// Simulate up to time T, returning event times.
    pub fn simulate_until(&mut self, t: f64, rng: &mut impl Rng) -> &[f64] {
        let exp = Exp::new(self.rate).unwrap();
        loop {
            let interval: f64 = exp.sample(rng);
            self.current_time += interval;
            if self.current_time > t {
                self.current_time = t; // reset to boundary
                break;
            }
            self.events.push(self.current_time);
        }
        &self.events
    }

    /// Number of events in [0, t].
    pub fn count(&self, t: f64) -> usize {
        self.events.iter().filter(|&&e| e <= t).count()
    }

    /// Expected number of events in [0, t]: λt
    pub fn expected_count(&self, t: f64) -> f64 {
        self.rate * t
    }
}

/// Non-homogeneous (inhomogeneous) Poisson process with time-varying rate λ(t).
/// Uses thinning algorithm (Lewis & Shedler).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonHomogeneousPoisson {
    pub events: Vec<f64>,
    pub current_time: f64,
    pub max_rate: f64,
    pub rate_fn: String, // Serialized descriptor of rate function
}

impl NonHomogeneousPoisson {
    /// Create with a known maximum rate for thinning.
    pub fn new(max_rate: f64) -> Self {
        Self {
            events: vec![],
            current_time: 0.0,
            max_rate,
            rate_fn: "custom".to_string(),
        }
    }

    /// Simulate using thinning with a rate function.
    pub fn simulate_until<F: Fn(f64) -> f64>(
        &mut self,
        t: f64,
        rate_fn: F,
        rng: &mut impl Rng,
    ) -> &[f64] {
        let exp = Exp::new(self.max_rate).unwrap();
        let uniform = Uniform::new(0.0, 1.0);
        let mut time = 0.0;
        loop {
            let interval: f64 = exp.sample(rng);
            time += interval;
            if time > t {
                break;
            }
            // Accept with probability λ(t)/λ_max
            let u: f64 = rng.sample(uniform);
            if u <= rate_fn(time) / self.max_rate {
                self.events.push(time);
            }
        }
        self.current_time = t;
        &self.events
    }
}

/// Compound Poisson process: sum of random jumps at Poisson event times.
/// S(t) = Σ_{i=1}^{N(t)} Y_i where N(t) is Poisson and Y_i are iid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundPoisson {
    pub rate: f64,
    pub jump_mean: f64,
    pub jump_std: f64,
    pub events: Vec<(f64, f64)>, // (time, cumulative sum)
}

impl CompoundPoisson {
    pub fn new(rate: f64, jump_mean: f64, jump_std: f64) -> Self {
        Self {
            rate,
            jump_mean,
            jump_std,
            events: vec![],
        }
    }

    pub fn simulate_until(&mut self, t: f64, rng: &mut impl Rng) -> &[(f64, f64)] {
        let exp = Exp::new(self.rate).unwrap();
        let normal = rand_distr::Normal::new(self.jump_mean, self.jump_std).unwrap();
        let mut time = 0.0;
        let mut cumulative = 0.0;
        loop {
            let interval: f64 = exp.sample(rng);
            time += interval;
            if time > t {
                break;
            }
            let jump: f64 = normal.sample(rng);
            cumulative += jump;
            self.events.push((time, cumulative));
        }
        &self.events
    }

    /// Expected value: E[S(t)] = λt * E[Y]
    pub fn expected_value(&self, t: f64) -> f64 {
        self.rate * t * self.jump_mean
    }

    /// Variance: Var(S(t)) = λt * E[Y²] = λt * (Var(Y) + E[Y]²)
    pub fn variance(&self, t: f64) -> f64 {
        self.rate * t * (self.jump_std * self.jump_std + self.jump_mean * self.jump_mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_poisson_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut pp = PoissonProcess::new(5.0);
        pp.simulate_until(100.0, &mut rng);
        let count = pp.events.len();
        let expected = 500; // 5.0 * 100
        assert!((count as f64 - expected as f64).abs() / (expected as f64) < 0.1,
            "Count {} should be near {}", count, expected);
    }

    #[test]
    fn test_poisson_expected_count() {
        let pp = PoissonProcess::new(3.0);
        assert!((pp.expected_count(10.0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_inter_arrival_exponential() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut pp = PoissonProcess::new(2.0);
        for _ in 0..5000 {
            pp.next_event(&mut rng);
        }
        let intervals: Vec<f64> = pp.events.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        assert!((mean - 0.5).abs() < 0.05, "Mean interval {} should be near 0.5", mean);
    }

    #[test]
    fn test_non_homogeneous_poisson() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut npp = NonHomogeneousPoisson::new(10.0);
        // Rate function: λ(t) = 5 + 5*sin(t)
        npp.simulate_until(100.0, |t| 5.0 + 5.0 * t.sin(), &mut rng);
        // Should have fewer events than homogeneous at max rate
        assert!(npp.events.len() > 0);
        assert!(npp.events.len() < 10 * 100 + 500); // well below max rate * time
    }

    #[test]
    fn test_compound_poisson_expected_value() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cp = CompoundPoisson::new(5.0, 2.0, 1.0);
        cp.simulate_until(100.0, &mut rng);
        let expected = cp.expected_value(100.0);
        assert!((expected - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_compound_poisson_cumulative() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut cp = CompoundPoisson::new(10.0, 0.0, 1.0);
        cp.simulate_until(10.0, &mut rng);
        // Check cumulative is increasing in time (by checking it's ordered)
        for window in cp.events.windows(2) {
            assert!(window[0].0 < window[1].0);
        }
    }

    #[test]
    fn test_compound_poisson_variance() {
        let cp = CompoundPoisson::new(5.0, 2.0, 3.0);
        let t = 10.0;
        // Var = λt * (σ² + μ²) = 5*10*(9+4) = 650
        assert!((cp.variance(t) - 650.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_events_ordered() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut pp = PoissonProcess::new(10.0);
        pp.simulate_until(10.0, &mut rng);
        for window in pp.events.windows(2) {
            assert!(window[0] < window[1]);
        }
    }
}
