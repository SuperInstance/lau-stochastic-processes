use rand::Rng;
use serde::{Deserialize, Serialize};

/// A 1D random walk with optional bias and barriers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomWalk1D {
    pub position: f64,
    pub step_size: f64,
    pub up_probability: f64,
    pub upper_barrier: Option<f64>,
    pub lower_barrier: Option<f64>,
    pub history: Vec<f64>,
}

impl RandomWalk1D {
    pub fn new(step_size: f64, up_probability: f64) -> Self {
        Self {
            position: 0.0,
            step_size,
            up_probability,
            upper_barrier: None,
            lower_barrier: None,
            history: vec![0.0],
        }
    }

    pub fn with_barriers(mut self, lower: f64, upper: f64) -> Self {
        self.lower_barrier = Some(lower);
        self.upper_barrier = Some(upper);
        self
    }

    pub fn step(&mut self, rng: &mut impl Rng) -> f64 {
        let r: f64 = rng.gen();
        if r < self.up_probability {
            self.position += self.step_size;
        } else {
            self.position -= self.step_size;
        }
        if let Some(upper) = self.upper_barrier {
            if self.position > upper {
                self.position = upper;
            }
        }
        if let Some(lower) = self.lower_barrier {
            if self.position < lower {
                self.position = lower;
            }
        }
        self.history.push(self.position);
        self.position
    }

    pub fn simulate(&mut self, steps: usize, rng: &mut impl Rng) -> &[f64] {
        for _ in 0..steps {
            self.step(rng);
        }
        &self.history
    }

    /// Expected hitting time for symmetric walk to reach ±b (discrete approximation).
    /// For a symmetric walk starting at 0, E[T] = b² for barrier at ±b (in steps).
    pub fn expected_hitting_time(barrier: f64, step_size: f64) -> f64 {
        let b = barrier / step_size;
        b * b
    }

    /// Run many simulations to estimate hitting time of a barrier.
    pub fn estimate_hitting_time(
        step_size: f64,
        up_probability: f64,
        target: f64,
        max_steps: usize,
        num_trials: usize,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let mut total = 0usize;
        let mut hits = 0usize;
        for _ in 0..num_trials {
            let mut pos = 0.0f64;
            for s in 0..max_steps {
                let r: f64 = rng.gen();
                if r < up_probability {
                    pos += step_size;
                } else {
                    pos -= step_size;
                }
                if pos >= target {
                    total += s + 1;
                    hits += 1;
                    break;
                }
            }
        }
        if hits == 0 {
            f64::NAN
        } else {
            total as f64 / hits as f64
        }
    }
}

/// A 2D random walk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomWalk2D {
    pub x: f64,
    pub y: f64,
    pub step_size: f64,
    pub history: Vec<(f64, f64)>,
}

impl RandomWalk2D {
    pub fn new(step_size: f64) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            step_size,
            history: vec![(0.0, 0.0)],
        }
    }

    pub fn step(&mut self, rng: &mut impl Rng) -> (f64, f64) {
        let direction: f64 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        self.x += self.step_size * direction.cos();
        self.y += self.step_size * direction.sin();
        self.history.push((self.x, self.y));
        (self.x, self.y)
    }

    pub fn simulate(&mut self, steps: usize, rng: &mut impl Rng) -> &[(f64, f64)] {
        for _ in 0..steps {
            self.step(rng);
        }
        &self.history
    }

    /// Euclidean distance from origin.
    pub fn distance_from_origin(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_symmetric_walk_zero_mean() {
        let mut rw = RandomWalk1D::new(1.0, 0.5);
        let mut rng = StdRng::seed_from_u64(42);
        rw.simulate(10000, &mut rng);
        // Check final position - should be relatively small compared to sqrt(10000)=100
        assert!(rw.position.abs() < 200.0, "Final position {} should be bounded", rw.position);
    }

    #[test]
    fn test_biased_walk_drifts() {
        let mut rw = RandomWalk1D::new(1.0, 0.7);
        let mut rng = StdRng::seed_from_u64(42);
        rw.simulate(10000, &mut rng);
        let final_pos = rw.position;
        assert!(final_pos > 500.0, "Biased walk should drift up, got {}", final_pos);
    }

    #[test]
    fn test_barriers_clamp() {
        let mut rw = RandomWalk1D::new(1.0, 0.5).with_barriers(-5.0, 5.0);
        let mut rng = StdRng::seed_from_u64(42);
        rw.simulate(10000, &mut rng);
        assert!(rw.history.iter().all(|&p| p >= -5.0 && p <= 5.0));
    }

    #[test]
    fn test_expected_hitting_time_symmetric() {
        // For symmetric walk, E[T] to reach ±b = (b/step)²
        let step = 1.0;
        let barrier = 10.0;
        let expected = RandomWalk1D::expected_hitting_time(barrier, step);
        assert!((expected - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_estimate_hitting_time() {
        let avg_steps = RandomWalk1D::estimate_hitting_time(1.0, 0.5, 5.0, 100000, 2000);
        // Should be roughly 25 (barrier²) for symmetric walk
        assert!(avg_steps > 5.0 && avg_steps < 2000.0, "Avg steps = {}", avg_steps);
    }

    #[test]
    fn test_walk_2d_stays_reasonable() {
        let mut rw = RandomWalk2D::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);
        rw.simulate(1000, &mut rng);
        // After 1000 steps, distance should be roughly sqrt(1000) ≈ 31.6
        let dist = rw.distance_from_origin();
        assert!(dist < 100.0, "2D walk distance {} seems too large", dist);
    }

    #[test]
    fn test_walk_history_length() {
        let mut rw = RandomWalk1D::new(1.0, 0.5);
        let mut rng = StdRng::seed_from_u64(1);
        rw.simulate(100, &mut rng);
        assert_eq!(rw.history.len(), 101); // initial + 100 steps
    }
}
