# Creating a graph
# Here I work with the path of length 5
from gibbs_sampler import inner_product, update, sample, print_config, Ising_Gen
from sparsitron import Sparsitron
from tools import get_cor

import networkx as nx

# Building the graph. Here I consider a path with an extra edge
G = nx.Graph()
n = 5

[G.add_node(i, spin=1) for i in range(n + 1)]
[G.add_edge(i, i + 1, weight=0.3) for i in range(n)]
G.add_edge(2, 5, weight=1)

inner_product(G, 2)
update(G, 2)
sample(G)
print_config(G)
t = Ising_Gen(G, nb=300)
# get_cor(itt=iter(t))


# s = Sparsitron(t)


# s.run(4000)
# Sparsitron(t, beta=0.98, sparsity=0.7).run(1000)
Sparsitron(t, beta=0.98, sparsity=0.6).run(1000)
Sparsitron(t, beta=0.98, sparsity=0.2).run(1000)
Sparsitron(t, beta=0.98, sparsity=0.1).run(1000)
Sparsitron(t, beta=0.98, sparsity=1.3).run(1000)
Sparsitron(t, beta=0.98, sparsity=1.5).run(1000)
Sparsitron(t, beta=0.98, sparsity=2.5).run(1000)

# Sparsitron(t, beta=0.98, sparsity=0.8).run(1000)
# Sparsitron(t, beta=0.95, sparsity=0.9).run(500)
# Sparsitron(t, beta=0.9, sparsity=0.8).run(500)
# Sparsitron(t, sparsity=2).run(500)
# Sparsitron(t, sparsity=10).run(500)
# Sparsitron(t, sparsity=5).run(500)
# Sparsitron(t, sparsity=10).run(500)