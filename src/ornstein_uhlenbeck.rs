use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σ dW
/// Mean-reverting process with long-term mean μ, speed θ, and volatility σ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrnsteinUhlenbeck {
    pub theta: f64,     // Mean reversion speed
    pub mu: f64,        // Long-term mean
    pub sigma: f64,     // Volatility
    pub x0: f64,        // Initial value
    pub path: Vec<f64>,
    pub times: Vec<f64>,
    pub dt: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, mu: f64, sigma: f64, x0: f64, dt: f64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            x0,
            path: vec![x0],
            times: vec![0.0],
            dt,
        }
    }

    /// Euler-Maruyama discretization step.
    pub fn step(&mut self, rng: &mut impl Rng) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = *self.path.last().unwrap();
        let dW = normal.sample(rng) * self.dt.sqrt();
        let dx = self.theta * (self.mu - x) * self.dt + self.sigma * dW;
        let new_x = x + dx;
        let new_t = self.times.last().unwrap() + self.dt;
        self.path.push(new_x);
        self.times.push(new_t);
        new_x
    }

    pub fn simulate(&mut self, steps: usize, rng: &mut impl Rng) -> &[f64] {
        for _ in 0..steps {
            self.step(rng);
        }
        &self.path
    }

    /// Exact transition: X(t) ~ N(x0*exp(-θt) + μ(1-exp(-θt)), σ²(1-exp(-2θt))/(2θ))
    pub fn exact_mean(&self, t: f64) -> f64 {
        self.x0 * (-self.theta * t).exp() + self.mu * (1.0 - (-self.theta * t).exp())
    }

    pub fn exact_variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma * (1.0 - (-2.0 * self.theta * t).exp()) / (2.0 * self.theta)
    }

    /// Autocorrelation function: ρ(s,t) = exp(-θ|t-s|)
    pub fn autocorrelation(&self, lag: f64) -> f64 {
        (-self.theta * lag.abs()).exp()
    }

    /// Stationary distribution mean (always μ).
    pub fn stationary_mean(&self) -> f64 {
        self.mu
    }

    /// Stationary distribution variance: σ²/(2θ).
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta)
    }

    /// Half-life: time for autocorrelation to drop to 0.5 = ln(2)/θ.
    pub fn half_life(&self) -> f64 {
        2.0f64.ln() / self.theta
    }

    /// Verify mean reversion by simulating from a displaced initial value
    /// and checking that the average converges toward μ.
    pub fn verify_mean_reversion(
        theta: f64,
        mu: f64,
        sigma: f64,
        x0: f64,
        dt: f64,
        steps: usize,
        num_trials: usize,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let mut final_values = Vec::new();
        for _ in 0..num_trials {
            let mut ou = OrnsteinUhlenbeck::new(theta, mu, sigma, x0, dt);
            ou.simulate(steps, &mut rng);
            final_values.push(*ou.path.last().unwrap());
        }
        final_values.iter().sum::<f64>() / final_values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_ou_mean_reversion() {
        // Start far from mean, should converge
        let avg = OrnsteinUhlenbeck::verify_mean_reversion(
            5.0, 10.0, 0.5, 0.0, 0.01, 5000, 2000,
        );
        assert!((avg - 10.0).abs() < 1.0, "Final average {} should be near 10", avg);
    }

    #[test]
    fn test_ou_exact_mean() {
        let ou = OrnsteinUhlenbeck::new(1.0, 5.0, 1.0, 0.0, 0.01);
        // At t=0, mean should be x0
        assert!((ou.exact_mean(0.0) - 0.0).abs() < 1e-10);
        // As t→∞, mean should approach μ
        assert!((ou.exact_mean(100.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_ou_exact_variance() {
        let ou = OrnsteinUhlenbeck::new(1.0, 5.0, 2.0, 0.0, 0.01);
        // At t=0, variance should be 0
        assert!((ou.exact_variance(0.0) - 0.0).abs() < 1e-10);
        // Stationary variance = σ²/(2θ) = 4/2 = 2
        assert!((ou.exact_variance(100.0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_ou_stationary_variance() {
        let ou = OrnsteinUhlenbeck::new(2.0, 0.0, 1.0, 0.0, 0.01);
        assert!((ou.stationary_variance() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_ou_half_life() {
        let ou = OrnsteinUhlenbeck::new(2.0, 0.0, 1.0, 0.0, 0.01);
        let hl = ou.half_life();
        assert!((hl - 2.0f64.ln() / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ou_autocorrelation() {
        let ou = OrnsteinUhlenbeck::new(1.0, 0.0, 1.0, 0.0, 0.01);
        assert!((ou.autocorrelation(0.0) - 1.0).abs() < 1e-10);
        assert!(ou.autocorrelation(1.0) < 1.0);
        assert!(ou.autocorrelation(1.0) > 0.0);
    }

    #[test]
    fn test_ou_simulation_variance() {
        let mut rng = StdRng::seed_from_u64(42);
        let theta = 2.0;
        let mu = 5.0;
        let sigma = 1.0;
        let dt = 0.01;
        let steps = 5000; // t=50

        let mut final_values = Vec::new();
        for seed in 0..1000u64 {
            let mut ou = OrnsteinUhlenbeck::new(theta, mu, sigma, mu, dt);
            let mut rng = StdRng::seed_from_u64(seed);
            ou.simulate(steps, &mut rng);
            final_values.push(*ou.path.last().unwrap());
        }
        let mean = final_values.iter().sum::<f64>() / final_values.len() as f64;
        let var = final_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / final_values.len() as f64;
        let expected_var = sigma * sigma / (2.0 * theta);
        assert!((mean - mu).abs() < 0.2, "Mean {} should be near {}", mean, mu);
        assert!((var - expected_var).abs() / expected_var < 0.3,
            "Variance {} should be near {}", var, expected_var);
    }
}
