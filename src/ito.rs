use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Itô integral approximation: I(t) = ∫₀ᵗ f(s) dW(s)
/// Approximated as Σ f(t_i) * (W_{i+1} - W_i)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItoIntegral {
    pub integrand_values: Vec<f64>,
    pub brownian_increments: Vec<f64>,
    pub integral_values: Vec<f64>,
    pub times: Vec<f64>,
    pub dt: f64,
}

impl ItoIntegral {
    pub fn new(dt: f64) -> Self {
        Self {
            integrand_values: vec![],
            brownian_increments: vec![],
            integral_values: vec![0.0],
            times: vec![0.0],
            dt,
        }
    }

    /// Approximate the Itô integral ∫₀ᵗ f(s) dW(s) for a deterministic function f.
    /// `f` is evaluated at the left endpoint (Itô convention).
    pub fn approximate<F: Fn(f64) -> f64>(
        dt: f64,
        total_time: f64,
        f: F,
        rng: &mut impl Rng,
    ) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let steps = (total_time / dt) as usize;
        let mut result = Self::new(dt);
        let mut integral = 0.0;
        let mut t = 0.0;

        for _ in 0..steps {
            let f_val = f(t);
            let dW = normal.sample(rng) * dt.sqrt();
            integral += f_val * dW;
            result.integrand_values.push(f_val);
            result.brownian_increments.push(dW);
            result.integral_values.push(integral);
            t += dt;
            result.times.push(t);
        }
        result
    }

    /// Itô isometry: E[(∫₀ᵗ f(s) dW(s))²] = ∫₀ᵗ f(s)² ds
    /// Verify this empirically across multiple trials.
    pub fn verify_ito_isometry<F: Fn(f64) -> f64>(
        dt: f64,
        total_time: f64,
        f: F,
        num_trials: usize,
        rng: &mut impl Rng,
    ) -> (f64, f64) {
        let steps = (total_time / dt) as usize;
        // Analytical: ∫₀ᵗ f(s)² ds
        let mut analytical = 0.0;
        for i in 0..steps {
            let t = i as f64 * dt;
            analytical += f(t) * f(t) * dt;
        }

        let mut sum_squares = 0.0;
        for _ in 0..num_trials {
            let result = Self::approximate(dt, total_time, &f, rng);
            let final_val = *result.integral_values.last().unwrap();
            sum_squares += final_val * final_val;
        }
        let empirical = sum_squares / num_trials as f64;
        (empirical, analytical)
    }
}

/// Itô's lemma for a function f of an Itô process X:
/// df(X) = f'(X)dX + (1/2)f''(X)(dX)²
/// For X = W (Brownian motion): df(W) = f'(W)dW + (1/2)f''(W)dt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItoLemma {
    pub process_values: Vec<f64>,
    pub transformed_values: Vec<f64>,
    pub times: Vec<f64>,
    pub dt: f64,
}

impl ItoLemma {
    /// Apply Itô's lemma: if X_t follows dX = μdt + σdW,
    /// then f(X_t) follows df = f'(X)(μdt + σdW) + (1/2)f''(X)σ²dt
    pub fn apply<F, Fp, Fpp>(
        x0: f64,
        mu: f64,
        sigma: f64,
        dt: f64,
        steps: usize,
        f: F,
        f_prime: Fp,
        f_double_prime: Fpp,
        rng: &mut impl Rng,
    ) -> Self
    where
        F: Fn(f64) -> f64,
        Fp: Fn(f64) -> f64,
        Fpp: Fn(f64) -> f64,
    {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x = x0;
        let mut process_values = vec![x0];
        let mut transformed_values = vec![f(x0)];
        let mut times = vec![0.0];
        let mut t = 0.0;

        for _ in 0..steps {
            let dW = normal.sample(rng) * dt.sqrt();
            // Itô's lemma: df = f'(X)(μdt + σdW) + (1/2)f''(X)σ²dt
            let df = f_prime(x) * (mu * dt + sigma * dW) + 0.5 * f_double_prime(x) * sigma * sigma * dt;
            let new_f = f(x) + df;
            // Also evolve the underlying process
            x += mu * dt + sigma * dW;
            t += dt;

            process_values.push(x);
            transformed_values.push(new_f);
            times.push(t);
        }

        Self {
            process_values,
            transformed_values,
            times,
            dt,
        }
    }

    /// Verify Itô's lemma by comparing: f(W_T) vs f(W_0) + integral of Itô's formula
    pub fn verify<F, Fp, Fpp>(
        dt: f64,
        total_time: f64,
        f: F,
        f_prime: Fp,
        f_double_prime: Fpp,
        num_trials: usize,
    ) -> bool
    where
        F: Fn(f64) -> f64,
        Fp: Fn(f64) -> f64,
        Fpp: Fn(f64) -> f64,
    {
        let steps = (total_time / dt) as usize;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let mut max_error = 0.0;

        for _ in 0..num_trials {
            let mut w = 0.0f64;
            let mut ito_sum = 0.0f64;
            for _ in 0..steps {
                let dW: f64 = normal.sample(&mut rng) * dt.sqrt();
                ito_sum += f_prime(w) * dW + 0.5 * f_double_prime(w) * dt;
                w += dW;
            }
            let f_actual = f(w);
            let f_approx = f(0.0) + ito_sum;
            let error = (f_actual - f_approx).abs();
            if error > max_error {
                max_error = error;
            }
        }
        max_error < 0.5 // Allow some numerical error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_ito_integral_zero_mean() {
        // E[∫ f(s) dW(s)] = 0
        let mut rng = StdRng::seed_from_u64(42);
        let mut final_values = Vec::new();
        for _ in 0..1000 {
            let result = ItoIntegral::approximate(0.01, 1.0, |t| t + 1.0, &mut rng);
            final_values.push(*result.integral_values.last().unwrap());
        }
        let mean = final_values.iter().sum::<f64>() / final_values.len() as f64;
        assert!(mean.abs() < 0.3, "Mean of Itô integral should be near 0, got {}", mean);
    }

    #[test]
    fn test_ito_isometry() {
        let mut rng = StdRng::seed_from_u64(42);
        let (empirical, analytical) = ItoIntegral::verify_ito_isometry(
            0.01, 1.0, |t| t + 1.0, 5000, &mut rng,
        );
        let relative_error = (empirical - analytical).abs() / analytical;
        assert!(relative_error < 0.15,
            "Itô isometry: empirical={}, analytical={}, relative_error={}",
            empirical, analytical, relative_error);
    }

    #[test]
    fn test_ito_lemma_w_squared() {
        // f(W) = W², then f'(W)=2W, f''(W)=2
        // Itô: d(W²) = 2W dW + dt
        let mut rng = StdRng::seed_from_u64(42);
        let result = ItoLemma::apply(
            0.0, 0.0, 1.0, 0.001, 1000,
            |x| x * x,
            |x| 2.0 * x,
            |_| 2.0,
            &mut rng,
        );
        // The Itô-transformed final value should approximate f(W_final)
        let w_final = *result.process_values.last().unwrap();
        let f_ito = *result.transformed_values.last().unwrap();
        let f_actual = w_final * w_final;
        assert!((f_ito - f_actual).abs() / f_actual.abs().max(1.0) < 0.05,
            "Itô f(W²)={}, actual f(W²)={}", f_ito, f_actual);
    }

    #[test]
    fn test_ito_lemma_verification() {
        // f(x) = x² for W
        let ok = ItoLemma::verify(
            0.01, 1.0,
            |x| x * x,
            |x| 2.0 * x,
            |_| 2.0,
            500,
        );
        assert!(ok, "Itô's lemma verification failed");
    }

    #[test]
    fn test_ito_lemma_exp_w() {
        // f(W) = exp(W), f'=exp(W), f''=exp(W)
        // Itô: d(exp(W)) = exp(W)dW + (1/2)exp(W)dt
        let mut rng = StdRng::seed_from_u64(42);
        let result = ItoLemma::apply(
            0.0, 0.0, 1.0, 0.001, 1000,
            |x| x.exp(),
            |x| x.exp(),
            |x| x.exp(),
            &mut rng,
        );
        let w_final = *result.process_values.last().unwrap();
        let f_ito = *result.transformed_values.last().unwrap();
        let f_actual = w_final.exp();
        let rel_error = (f_ito - f_actual).abs() / f_actual;
        assert!(rel_error < 0.05, "exp(W) Itô error: {}", rel_error);
    }
}
