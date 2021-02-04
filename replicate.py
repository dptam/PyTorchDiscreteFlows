import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
from scipy.stats import multivariate_normal
import torch
import itertools

# importing all of the normalizing flows.
import discrete_flows.disc_utils
from discrete_flows.made import MADE
from discrete_flows.mlp import MLP
from discrete_flows.embed import EmbeddingLayer
from discrete_flows.disc_models import *

def sample_quantized_gaussian_mixture(batch_size):
    """Samples data from a 2D quantized mixture of Gaussians.
    This is a quantized version of the mixture of Gaussians experiment from the
    Unrolled GANS paper (Metz et al., 2017).
    Args:
        batch_size: The total number of observations.
    Returns:
        Tensor with shape `[batch_size, 2]`, where each entry is in
            `{0, 1, ..., max_quantized_value - 1}`, a rounded sample from a mixture
            of Gaussians.
    """
    clusters = np.array([[2., 0.], [np.sqrt(2), np.sqrt(2)],
                                             [0., 2.], [-np.sqrt(2), np.sqrt(2)],
                                             [-2., 0.], [-np.sqrt(2), -np.sqrt(2)],
                                             [0., -2.], [np.sqrt(2), -np.sqrt(2)]])
    assignments = torch.distributions.OneHotCategorical(
            logits=torch.zeros(8, dtype = torch.float32)).sample([batch_size])
    means = torch.matmul(assignments, torch.from_numpy(clusters).float())

    samples = torch.distributions.normal.Normal(loc=means, scale=0.1).sample()
    clipped_samples = torch.clamp(samples, -2.25, 2.25)
    quantized_samples = (torch.round(clipped_samples * 20) + 45).long()
    return quantized_samples

batch_size, sequence_length, vocab_size = 1024, 2, 91
data = sample_quantized_gaussian_mixture(batch_size)

num_flows = 1 # number of flow steps. This is different to the number of layers used inside each flow
temperature = 0.1 # used for the straight-through gradient estimator. Value taken from the paper
disc_layer_type = 'autoreg' #'autoreg' #'bipartite'

# This setting was previously used for the MLP and MADE networks.
nh = 8 # number of hidden units per layer
vector_length = sequence_length*vocab_size

flows = []
for i in range(num_flows):
    if disc_layer_type == 'autoreg':

        layer = EmbeddingLayer([batch_size, sequence_length, vocab_size], output_size=vocab_size)
        # MADE network is much more powerful.
        # layer = MADE([batch_size, sequence_length, vocab_size], vocab_size, [nh, nh, nh])

        disc_layer = DiscreteAutoregressiveFlow(layer, temperature,
                                                vocab_size)

    elif disc_layer_type == 'bipartite':
        # MLP will learn the factorized distribution and not perform well.
        # layer = MLP(vector_length//2, vector_length//2, nh)

        layer = torch.nn.Embedding(vector_length // 2, vector_length // 2)

        disc_layer = DiscreteBipartiteFlow(layer, i % 2, temperature,
                                           vocab_size, vector_length, embedding=True)
        # i%2 flips the parity of the masking. It splits the vector in half and alternates
        # each flow between changing the first half or the second.

    flows.append(disc_layer)

model = DiscreteAutoFlowModel(flows)

base_log_probs = torch.tensor(torch.randn(sequence_length, vocab_size), requires_grad = True)
base = torch.distributions.OneHotCategorical(logits = base_log_probs)

epochs = 1500
learning_rate = 0.01
print_loss_every = epochs // 10

losses = []
optimizer = torch.optim.Adam(
    [
        {'params': model.parameters(), 'lr': learning_rate},
        {'params': base_log_probs, 'lr': learning_rate}
    ])

model.train()
for e in range(epochs):
    x = sample_quantized_gaussian_mixture(batch_size)
    x = F.one_hot(x, num_classes=vocab_size).float()

    if disc_layer_type == 'bipartite':
        x = x.view(x.shape[0], -1)  # flattening vector

    optimizer.zero_grad()
    zs = model.forward(x)

    if disc_layer_type == 'bipartite':
        zs = zs.view(batch_size, sequence_length, -1)  # adding back in sequence dimension

    base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
    # print(zs.shape, base_log_probs_sm.shape)
    logprob = zs * base_log_probs_sm  # zs are onehot so zero out all other logprobs.
    loss = -torch.sum(logprob) / batch_size

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if e % print_loss_every == 0:
        print('epoch:', e, 'loss:', loss.item())
