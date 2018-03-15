# Creating a graph
# Here I work with the path of length 5
from gibbs_sampler import inner_product, update, sample, print_config, Ising_Gen
from Sparsitron import Sparsitron, get_cor
import networkx as nx

# Building the graph. Here I consider a path with an extra edge
G = nx.Graph()
n = 5

[G.add_node(i, spin=1) for i in range(n + 1)]
[G.add_edge(i, i + 1, weight=0.1) for i in range(n)]
G.add_edge(2, 5, weight=0.7)


inner_product(G, 2)
update(G, 2)
sample(G)
print_config(G)
t = Ising_Gen(G, nb=300)

s = Sparsitron(t)
s.run(100)

get_cor(itt=iter(t))
