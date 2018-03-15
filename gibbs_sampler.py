import numpy as np
import math

VERBOSE = False

####
# The tools to sample an Ising configuration on a graph.
####


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inner_product(G, i):
    """
    Computes the inner of weights x spins for edges incident to i
    """
    if VERBOSE:
        print("Computing w.x at {}".format(i))
    inner_product = 0
    for u in G[i].items():
        a = G.nodes[u[0]]["spin"]
        b = u[1]["weight"]
        if VERBOSE:
            print("a = {}, b = {}, adding a * b".format(a, b))
        inner_product += a * b
    if VERBOSE:
        print("Inner product at {} is {}".format(i, inner_product))
    return inner_product


def update(G, i):
    """
    Flips spin i according to Glauber dynamics
    """
    sigmo = sigmoid(2 * inner_product(G, i))
    G.nodes[i]["spin"] = 1 if (np.random.rand() < sigmo) else - 1
    if VERBOSE:
        print("Setting spin to 1 w.p. {}. Setting {} to {}". format(sigmo, i, G.nodes[i]["spin"]))
    return None


def print_config(G):
    print("Spin configuration : \n" + " -- ".join([str(G.nodes[u]["spin"]) for u in G.nodes]))


def sample(G, nb=100):
    """
    Iterates update nb times. If nb is big enough, should output something close to the
    stationary distribution...
    """
    n = len(G)
    for _ in range(nb):
        i = np.random.randint(n)
        update(G, i)

    if VERBOSE:
        print_config(G)
    return None


class Ising_Gen():
    def __init__(self, G, nb=100):
        self.G = G
        self.nb = nb

    def __iter__(self):
        while True:
            sample(self.G, nb=self.nb)
            yield np.array([self.G.nodes[u]["spin"] for u in self.G.nodes])

# t = Ising_Gen(G)
# iter(t).next()

# def ising_gen(G, nb=100):
#     while True:
#         sample(G, nb=nb)
#         yield np.array([G.nodes[u]["spin"] for u in G.nodes])
