import numpy as np
import scipy.stats as stats

# Generate synthetic data from a Weibull distribution
np.random.seed(42)  # for reproducibility
true_shape = 2.0
true_scale = 3.0
true_location = 1.0  # example location parameter
sample_size = 100
data = np.random.weibull(true_shape, size=sample_size) * true_scale + true_location

# Prior distribution parameters (for simplicity, uniform priors are used here)
shape_prior = (0.1, 5.0)  # prior for shape parameter (uniform between 0.1 and 5.0)
scale_prior = (0.1, 10.0)  # prior for scale parameter (uniform between 0.1 and 10.0)
location_prior = (-5.0, 5.0)  # prior for location parameter (uniform between -5.0 and 5.0)

# Define likelihood function for the Weibull distribution
def likelihood(data, shape, scale, location):
    return np.prod(stats.weibull_min.pdf(data - location, c=shape, scale=scale))

# Simple Monte Carlo approach to estimate the posterior distribution
num_samples = 10000
shape_samples = np.random.uniform(*shape_prior, size=num_samples)
scale_samples = np.random.uniform(*scale_prior, size=num_samples)
location_samples = np.random.uniform(*location_prior, size=num_samples)
likelihoods = np.array([likelihood(data, shape, scale, location) for shape, scale, location in zip(shape_samples, scale_samples, location_samples)])

# Normalize likelihoods to obtain unnormalized posterior
unnormalized_posterior = likelihoods / np.sum(likelihoods)

# Sample from the unnormalized posterior
samples = np.random.choice(np.arange(num_samples), size=1000, replace=True, p=unnormalized_posterior)

# Extract samples of shape, scale, and location parameters
shape_samples_posterior = shape_samples[samples]
scale_samples_posterior = scale_samples[samples]
location_samples_posterior = location_samples[samples]

# Calculate credible intervals (e.g., 95% credible interval)
credible_interval_shape = np.percentile(shape_samples_posterior, [2.5, 97.5])
credible_interval_scale = np.percentile(scale_samples_posterior, [2.5, 97.5])
credible_interval_location = np.percentile(location_samples_posterior, [2.5, 97.5])

print("Credible interval for shape parameter:", credible_interval_shape)
print("Credible interval for scale parameter:", credible_interval_scale)
print("Credible interval for location parameter:", credible_interval_location)
