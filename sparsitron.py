from gibbs_sampler import sigmoid
from tools import timeit
import numpy as np

### Questions I have about the paper
# why the weird renormalization
# why the 1 + in the general version...
# why not -2 in the sigmoid ?
# why the part with a,b not in Algorithm 1?


class Sparsitron:

    def __init__(self, gen, beta=0.96, sparsity=5):
        self.beta = beta
        self.gen = gen
        self.iter_gen = iter(gen)
        self.n = len(self.iter_gen.next())
        self.weights = np.ones(self.n - 1) / (self.n - 1)
        self.sparsity = sparsity
        self.proba = self.weights / self.weights.sum()

    def compute_predictions(self):
        """
        Computes the prediction scalar, assuming weights are up to date
        """
        self.predictions = self.sparsity * self.proba

    def compute_loss(self):
        """
        Computes the loss vector assuming that the predictions and spins are numpy array vectors
        """
        self.loss = 0.5 - 0.5 * (sigmoid(self.predictions.dot(self.spins)) - (1.0 - self.target_spin) / 2) * self.spins

    def update_weights(self):
        self.weights = self.weights * (self.beta ** self.loss)
        self.proba = self.weights / self.weights.sum()

    def update(self):
        self.compute_predictions()
        self.compute_loss()
        self.update_weights()

    def print_real_weights(self):
        G = self.gen.G
        real_weights = np.zeros(self.n - 1)
        for u in G[self.n - 1].items():
            real_weights[u[0]] = u[1]["weight"]
        print("Real weights: {}\n".format(real_weights))

    @timeit
    def run(self, nb=100):
        for i in range(nb):
            config = self.iter_gen.next()
            # Let us predict the last variable
            self.target_spin = config[-1]
            self.spins = config[:-1]
            self.update()

        self.print_real_weights()
        print(self.proba)

