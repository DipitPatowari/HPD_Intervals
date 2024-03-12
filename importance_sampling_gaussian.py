import numpy as np
import scipy.stats as stats

# Generate synthetic data from a Gaussian Mixture Model
np.random.seed(42)  # for reproducibility
true_means = [0.0, 5.0]
true_std_devs = [1.0, 1.0]
true_weights = [0.6, 0.4]
sample_size = 100
component_samples = np.random.choice(len(true_means), size=sample_size, p=true_weights)
data = np.array([np.random.normal(true_means[i], true_std_devs[i]) for i in component_samples])

# Prior distribution parameters (for simplicity, uniform priors are used here)
mean_prior = (-5, 5)  # prior for means (uniform between -5 and 5)
std_dev_prior = (0.1, 2.0)  # prior for standard deviations (uniform between 0.1 and 2.0)
weights_prior = (0.01, 0.99)  # prior for weights (uniform between 0.01 and 0.99)

# Define likelihood function for the Gaussian Mixture Model
def likelihood(data, means, std_devs, weights):
    pdf_values = np.array([stats.norm.pdf(data, loc=mu, scale=std) for mu, std in zip(means, std_devs)])
    weighted_pdf = np.dot(weights, pdf_values)
    return np.prod(weighted_pdf)

# Simple Monte Carlo approach to estimate the posterior distribution
num_samples = 10000
mean_samples = np.random.uniform(*mean_prior, size=(num_samples, len(true_means)))
std_dev_samples = np.random.uniform(*std_dev_prior, size=(num_samples, len(true_std_devs)))
weights_samples = np.random.uniform(*weights_prior, size=(num_samples, len(true_weights)))
likelihoods = np.array([likelihood(data, means, std_devs, weights) for means, std_devs, weights in zip(mean_samples, std_dev_samples, weights_samples)])

# Normalize likelihoods to obtain unnormalized posterior
unnormalized_posterior = likelihoods / np.sum(likelihoods)

# Sample from the unnormalized posterior
samples = np.random.choice(np.arange(num_samples), size=1000, replace=True, p=unnormalized_posterior)

# Extract samples of means, standard deviations, and weights
mean_samples_posterior = mean_samples[samples]
std_dev_samples_posterior = std_dev_samples[samples]
weights_samples_posterior = weights_samples[samples]

# Calculate credible intervals (e.g., 95% credible interval)
credible_interval_means = np.percentile(mean_samples_posterior, [2.5, 97.5], axis=0)
credible_interval_std_devs = np.percentile(std_dev_samples_posterior, [2.5, 97.5], axis=0)
credible_interval_weights = np.percentile(weights_samples_posterior, [2.5, 97.5], axis=0)

print("Credible interval for means:", credible_interval_means)
print("Credible interval for standard deviations:", credible_interval_std_devs)
print("Credible interval for weights:", credible_interval_weights)
