# DeepTransport
## Stateful DeepWalk-like embedding through graph cellular sheaves

DeepWalk is a well-known milestone in graph representation learning capable of learning continuous embeddings of nodes in a graph through an online procedure yielding good reconstruction properties. DeepWalk and other shallow embeddings were subsumed by graph neural networks, capable of leveraging both nodes signals and network structure.

Our aim is to extend DeepWalk approach through graph cellular sheaves, a topological structure supporting higher dimensional data over a graph and allowing for a more expressive structuring of information. We merged DeepWalk algorithmic structure with the information encoded within the cellular sheaves in terms of the $\textit{agreement}$ of the adiacent 0-cochains on the shared edge stalk $||F_{u \triangleleft e}X_u - F_{v \triangleleft e}X_v||^2$, or the $\textit{similarity}$ between a set of cochains and its transported version $||X_u - F_{u \triangleleft e}^T F_{v \triangleleft e}X_v||^2$: these metrics allow to define a biased sampling strategy over the neighborhood of a node when sampling random walks, resulting in a stateful online learning algorithm. 
