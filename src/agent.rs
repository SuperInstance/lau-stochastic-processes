use rand::distributions::Distribution;
use serde::{Deserialize, Serialize};

/// Agent behavioral model: agents as stochastic processes with drift and volatility.
/// Combines continuous (diffusion) and jump components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub behavioral_state: AgentState,
    pub drift: f64,
    pub volatility: f64,
    pub mean_reversion_speed: f64,
    pub mean_reversion_level: f64,
    pub jump_intensity: f64,   // Poisson rate for behavioral jumps
    pub jump_mean: f64,
    pub jump_volatility: f64,
    pub history: Vec<AgentObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub sentiment: f64,      // Continuous behavioral state (-1 to 1 scale)
    pub activity: f64,       // Activity level
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObservation {
    pub state: AgentState,
    pub drift_component: f64,
    pub diffusion_component: f64,
    pub jump_component: f64,
}

impl Agent {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            behavioral_state: AgentState {
                sentiment: 0.0,
                activity: 1.0,
                timestamp: 0.0,
            },
            drift: 0.0,
            volatility: 0.5,
            mean_reversion_speed: 1.0,
            mean_reversion_level: 0.0,
            jump_intensity: 0.1,
            jump_mean: 0.0,
            jump_volatility: 0.5,
            history: vec![],
        }
    }

    pub fn with_drift(mut self, drift: f64) -> Self {
        self.drift = drift;
        self
    }

    pub fn with_volatility(mut self, vol: f64) -> Self {
        self.volatility = vol;
        self
    }

    pub fn with_mean_reversion(mut self, speed: f64, level: f64) -> Self {
        self.mean_reversion_speed = speed;
        self.mean_reversion_level = level;
        self
    }

    pub fn with_jumps(mut self, intensity: f64, mean: f64, vol: f64) -> Self {
        self.jump_intensity = intensity;
        self.jump_mean = mean;
        self.jump_volatility = vol;
        self
    }

    /// Simulate agent behavior over time using a jump-diffusion model:
    /// dX = [drift + θ(μ - X)]dt + σ dW + J dN
    /// Where J ~ N(jump_mean, jump_vol²) and N is a Poisson process.
    pub fn simulate(
        &mut self,
        dt: f64,
        steps: usize,
        rng: &mut impl rand::Rng,
    ) -> &[AgentObservation] {
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let jump_normal = rand_distr::Normal::new(self.jump_mean, self.jump_volatility).unwrap();
        let exp = rand_distr::Exp::new(self.jump_intensity).unwrap();

        let mut x = self.behavioral_state.sentiment;
        let mut t = 0.0;
        let mut next_jump_time = if self.jump_intensity > 0.0 {
            t + exp.sample(rng)
        } else {
            f64::INFINITY
        };

        for _ in 0..steps {
            let dW: f64 = normal.sample(rng) * dt.sqrt();

            // Diffusion component
            let diffusion = self.volatility * dW;

            // Drift + mean reversion
            let drift_component = (self.drift + self.mean_reversion_speed * (self.mean_reversion_level - x)) * dt;

            // Jump component
            let mut jump_component = 0.0;
            t += dt;
            while t >= next_jump_time {
                jump_component += jump_normal.sample(rng);
                next_jump_time = t + exp.sample(rng);
            }

            x += drift_component + diffusion + jump_component;

            self.history.push(AgentObservation {
                state: AgentState {
                    sentiment: x,
                    activity: 1.0 + x.abs() * 0.5,
                    timestamp: t,
                },
                drift_component,
                diffusion_component: diffusion,
                jump_component,
            });
        }

        self.behavioral_state.sentiment = x;
        self.behavioral_state.timestamp = t;
        &self.history
    }

    /// Compute the empirical drift of the agent's observed sentiment.
    pub fn empirical_drift(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let first = self.history.first().unwrap().state.sentiment;
        let last = self.history.last().unwrap().state.sentiment;
        let dt = self.history.last().unwrap().state.timestamp
            - self.history.first().unwrap().state.timestamp;
        if dt > 0.0 {
            (last - first) / dt
        } else {
            0.0
        }
    }

    /// Compute the empirical volatility (standard deviation of increments).
    pub fn empirical_volatility(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let sentiments: Vec<f64> = self.history.iter().map(|o| o.state.sentiment).collect();
        let increments: Vec<f64> = sentiments.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = increments.iter().sum::<f64>() / increments.len() as f64;
        let var = increments.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / increments.len() as f64;
        var.sqrt()
    }

    /// Check if the agent exhibits mean-reverting behavior.
    pub fn is_mean_reverting(&self, tolerance: f64) -> bool {
        if self.history.len() < 10 {
            return false;
        }
        // Test autocorrelation of increments: mean-reverting processes have negative autocorrelation
        let sentiments: Vec<f64> = self.history.iter().map(|o| o.state.sentiment).collect();
        let increments: Vec<f64> = sentiments.windows(2).map(|w| w[1] - w[0]).collect();
        if increments.len() < 2 {
            return false;
        }
        let n = increments.len();
        let mean = increments.iter().sum::<f64>() / n as f64;
        let var: f64 = increments.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-10 {
            return false;
        }
        let autocov: f64 = increments[..n - 1]
            .iter()
            .zip(&increments[1..])
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>() / (n - 1) as f64;
        let autocorr = autocov / var;
        autocorr < -tolerance
    }
}

/// Population of agents modeled as correlated stochastic processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPopulation {
    pub agents: Vec<Agent>,
    pub correlation: f64, // Cross-agent correlation
}

impl AgentPopulation {
    pub fn new(agents: Vec<Agent>, correlation: f64) -> Self {
        Self { agents, correlation }
    }

    /// Simulate all agents with correlated Brownian motions.
    pub fn simulate(&mut self, dt: f64, steps: usize, rng: &mut impl rand::Rng) {
        let n = self.agents.len();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        for step in 0..steps {
            // Generate correlated shocks
            let common: f64 = normal.sample(rng);
            let mut idiosyncratic = Vec::new();
            for _ in 0..n {
                idiosyncratic.push(normal.sample(rng));
            }

            for (i, agent) in self.agents.iter_mut().enumerate() {
                let rho = self.correlation;
                let shock = rho * common + (1.0 - rho * rho).sqrt() * idiosyncratic[i];
                let dW = shock * dt.sqrt();

                let x = agent.behavioral_state.sentiment;
                let drift = (agent.drift + agent.mean_reversion_speed * (agent.mean_reversion_level - x)) * dt;
                let diffusion = agent.volatility * dW;

                let new_x = x + drift + diffusion;

                let t = (step + 1) as f64 * dt;
                agent.history.push(AgentObservation {
                    state: AgentState {
                        sentiment: new_x,
                        activity: 1.0 + new_x.abs() * 0.5,
                        timestamp: t,
                    },
                    drift_component: drift,
                    diffusion_component: diffusion,
                    jump_component: 0.0,
                });
                agent.behavioral_state.sentiment = new_x;
                agent.behavioral_state.timestamp = t;
            }
        }
    }

    /// Compute average sentiment across the population.
    pub fn average_sentiment(&self) -> Vec<f64> {
        if self.agents.is_empty() {
            return vec![];
        }
        let len = self.agents[0].history.len();
        let mut avg = vec![0.0; len];
        for agent in &self.agents {
            for (i, obs) in agent.history.iter().enumerate() {
                if i < len {
                    avg[i] += obs.state.sentiment;
                }
            }
        }
        for a in &mut avg {
            *a /= self.agents.len() as f64;
        }
        avg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_agent_mean_reversion() {
        let mut agent = Agent::new("test")
            .with_mean_reversion(5.0, 0.0)
            .with_volatility(0.3)
            .with_jumps(0.0, 0.0, 0.0); // no jumps
        agent.behavioral_state.sentiment = 10.0;
        let mut rng = StdRng::seed_from_u64(42);
        agent.simulate(0.01, 5000, &mut rng);
        let final_sentiment = agent.history.last().unwrap().state.sentiment;
        assert!(final_sentiment.abs() < 3.0,
            "Agent should revert to 0, got {}", final_sentiment);
    }

    #[test]
    fn test_agent_drift() {
        let agent = Agent::new("drifty")
            .with_drift(2.0)
            .with_mean_reversion(0.0, 0.0) // no reversion
            .with_volatility(0.1)
            .with_jumps(0.0, 0.0, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        let mut a = agent;
        a.simulate(0.01, 1000, &mut rng);
        // After 10 seconds with drift 2.0, should have moved substantially
        let final_val = a.history.last().unwrap().state.sentiment;
        assert!(final_val > 10.0, "Agent with drift should move, got {}", final_val);
    }

    #[test]
    fn test_agent_population_correlation() {
        let agents: Vec<Agent> = (0..50)
            .map(|i| Agent::new(&format!("agent_{}", i))
                .with_volatility(1.0)
                .with_mean_reversion(0.1, 0.0))
            .collect();
        let mut pop = AgentPopulation::new(agents, 0.8);
        let mut rng = StdRng::seed_from_u64(42);
        pop.simulate(0.01, 1000, &mut rng);

        // Check that agents are correlated
        let sentiments: Vec<Vec<f64>> = pop.agents.iter()
            .map(|a| a.history.iter().map(|o| o.state.sentiment).collect())
            .collect();

        // Compute pairwise correlation between first two agents
        if sentiments.len() >= 2 && sentiments[0].len() > 10 {
            let s1 = &sentiments[0];
            let s2 = &sentiments[1];
            let n = s1.len().min(s2.len());
            let m1 = s1[..n].iter().sum::<f64>() / n as f64;
            let m2 = s2[..n].iter().sum::<f64>() / n as f64;
            let v1: f64 = s1[..n].iter().map(|x| (x - m1).powi(2)).sum::<f64>() / n as f64;
            let v2: f64 = s2[..n].iter().map(|x| (x - m2).powi(2)).sum::<f64>() / n as f64;
            let cov: f64 = s1[..n].iter().zip(&s2[..n])
                .map(|(a, b)| (a - m1) * (b - m2))
                .sum::<f64>() / n as f64;
            let corr = cov / (v1 * v2).sqrt();
            assert!(corr > 0.3, "Agents should be correlated, got {}", corr);
        }
    }

    #[test]
    fn test_agent_empirical_volatility() {
        let mut agent = Agent::new("volatile")
            .with_volatility(1.0)
            .with_mean_reversion(0.0, 0.0)
            .with_drift(0.0)
            .with_jumps(0.0, 0.0, 0.0);
        let mut rng = StdRng::seed_from_u64(42);
        agent.simulate(0.01, 5000, &mut rng);
        let emp_vol = agent.empirical_volatility();
        // dt = 0.01, so σ * sqrt(dt) ≈ 0.1 per step
        let expected = 1.0 * 0.01f64.sqrt();
        assert!((emp_vol - expected).abs() / expected < 0.3,
            "Empirical vol {} should be near {}", emp_vol, expected);
    }

    #[test]
    fn test_agent_mean_reverting_detection() {
        let mut agent = Agent::new("reverter")
            .with_mean_reversion(10.0, 0.0)
            .with_volatility(0.1)
            .with_jumps(0.0, 0.0, 0.0);
        agent.behavioral_state.sentiment = 5.0;
        let mut rng = StdRng::seed_from_u64(42);
        agent.simulate(0.01, 5000, &mut rng);
        // Verify it actually reverts toward 0
        let final_val = agent.history.last().unwrap().state.sentiment;
        assert!(final_val.abs() < 2.0, "Should revert toward 0, got {}", final_val);
    }

    #[test]
    fn test_agent_jumps() {
        let mut agent = Agent::new("jumpy")
            .with_volatility(0.1)
            .with_mean_reversion(1.0, 0.0)
            .with_jumps(5.0, 0.0, 2.0); // frequent large jumps
        let mut rng = StdRng::seed_from_u64(42);
        agent.simulate(0.01, 10000, &mut rng);
        // Should have some jumps
        let jumps: Vec<_> = agent.history.iter()
            .filter(|o| o.jump_component.abs() > 0.01)
            .collect();
        assert!(jumps.len() > 10, "Should have many jumps, got {}", jumps.len());
    }
}
