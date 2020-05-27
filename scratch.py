import tensorflow_probability as tfp
tfd = tfp.distributions

# shape of a and b equals to the number of latent beta dists
a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 4, 5]

dist = tfd.Beta(a, b)

dist.sample()


# sampling 5
def sampling(args):
    # here we need to take two vectors, alpha and beta, as input
    alpha, beta = args
    beta_dist = tfd.Beta(alpha, beta)
    return beta_dist.sample(len(alpha))

