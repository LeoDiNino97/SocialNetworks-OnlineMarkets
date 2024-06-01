import numpy as np
import networkx as nx 
from collections import deque 

import numpy as np

class Node:
    def __init__(self, 
                 d: int):

        self.children = {
            '0': None,
            '1': None
        }

        self.psi = np.random.rand(d)

class BinaryTree():
    def __init__(self, 
                 V: int, 
                 d: int):
        
        self.V = V
        self.d = d
        self.root = Node(d)
        self.H = len(bin(self.V).lstrip('0b'))
        self.populate()
    
    def insert(self, node):
        prev = self.root
        for i in node[:-1]:
            if prev.children[i]: 
                prev = prev.children[i]
            else:   
                N = Node(self.d)
                prev.children[i] = N
                prev = N

    def populate(self):
        seqs = [bin(v).lstrip('0b').rjust(self.H, '0') for v in range(self.V)]
        for seq in seqs:
            self.insert(seq)

    def traversal(self, node):

        node = bin(node).lstrip('0b').rjust(self.H, '0')
        prev = self.root
        path = [prev]
        for i in range(len(node[:-1])):
            prev = prev.children[node[i]]
            path.append(prev)
        return path, [node.psi for node in path]


class SkipGram():
    def __init__(self,
                 w: int,
                 d: int,
                 V: int,
                 LR = 0.025,
                 ):
        
        self.w = w
        self.d = d
        self.V = V
        self.LR = LR
        self.tree = BinaryTree(self.V, self.d)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def grads(self, u_k, Phi, v_j):

        path, psi = self.tree.traversal(u_k)
        u_k = bin(u_k).lstrip('0b').rjust(self.tree.H, '0')
        u_k = [float(u_k[i]) for i in range(len(u_k))]

        psi_grads = []
        phi_grad = np.zeros_like(Phi[v_j,:])

        for l in range(len(u_k)):
            S = self.sigmoid(np.dot(psi[l],Phi[v_j,:]))
            psi_grads.append( - ( Phi[v_j,:] * (u_k[l] - S) ) )
            phi_grad -= psi[l] * (u_k[l] - S)

        return path, psi_grads, phi_grad
    
    def step(self, u_k, Phi, v_j):

        path, psi_grads, phi_grad = self.grads(u_k, Phi, v_j)

        for i in range(len(path)):
            path[i].psi -= self.LR * psi_grads[i]

        Phi[v_j,:] -= self.LR * phi_grad

        #self.LR -= self.LR_schedule
        return Phi
    
    def window_step(self, W, Phi, v_j, j):

        for node in W[j - self.w : j + self.w]:
            Phi = self.step(node, Phi, v_j)

        return Phi
    
    def walk_step(self, W, Phi):

        for j in range(len(W)):
            v_j = W[j]
            Phi = self.window_step(W, Phi, v_j, j)
        return Phi

class DeepWalk():
    def __init__(self, 
                 graph: nx.Graph,
                 w: int,
                 d: int,
                 gamma: int,
                 t: int,
                 LR: float,
                 biased: bool,
                 nodes_signals: np.array,
                 stalk_dim: int, 
                 restriction_maps: dict,
                 mu: float
                 ):
        
        self.graph = graph
        self.nodes = np.array(self.graph.nodes)

        self.nodes_signals = nodes_signals
        self.stalk_dim = stalk_dim  
        self.restriction_maps = restriction_maps 

        self.w = w
        self.d = d 

        self.gamma = gamma
        self.t = t

        # Embedding initialization
        self.Phi = np.random.uniform(low=0, 
                                     high=1, 
                                     size = (len(self.nodes), d))

        # Learning rate initialization
        self.LR = LR

        # SkipGram initialization
        self.SkipGram = SkipGram(w, d, len(self.nodes), LR)

        # Sheaf initialization
        if self.nodes_signals:
            self.Sheaf = GraphSheaf(graph, stalk_dim, nodes_signals, mu)

        # Biased sampling attribute
        self.biased = biased
        
    def biased_sampling(self, node):
        
        neighbors = np.array([node for node in self.graph.neighbors(node)])
        probs = self.Sheaf.similarity_matrix[:,node]
        return np.random.choice(neighbors, p = probs)

    def random_walk(self, v):
        walk = [v]
        curr = v
        for _ in range(self.t):
            if not self.biased:
                w = np.random.choice(np.array([node for node in self.graph.neighbors(curr)]))
                walk.append(w)
                curr = w
            else:
                w = self.biased_sampling(v)
                walk.append(w)
                curr = w

        return walk

    def train(self):
        for _ in range(self.gamma):
            np.random.shuffle(self.nodes)

            for v in self.nodes:
                W = self.random_walk(v)
                self.Phi = self.SkipGram.walk_step(W, self.Phi)

class GraphSheaf():
    def __init__(self, 
                 G: nx.Graph, 
                 D: int,
                 node_signals: np.array,
                 mu: float):
        
        self.G = G
        self.nodes = np.array(list(G.nodes))
        self.edges = np.array(G.edges)

        self.D = D
        
        self.node_signals = node_signals

        self.maps = { 
            tuple(self.edges[i]):
                {self.edges[i,0]: None,
                 self.edges[i,1]: None}
            for i in range(self.edges.shape[0])
        }

        self.mu = mu
        self.sheaf_builder()
        self.similarity_matrix = self.similarity()

    def sheaf_builder(self):

        for i, _ in range(self.edges.shape[0]):
            e = self.edges[i]

            u = e[0]
            v = e[1]

            X_u = self.node_signals[u*self.D:(u+1)*self.D,:]
            X_v = self.node_signals[v*self.D:(v+1)*self.D,:]
            uu, uv, vv, vu = self.premultiplier(X_u, X_v)

            self.maps[e][u] = self.chi_u(uu, uv, vv, vu)
            self.maps[e][v] = self.chi_u(uu, uv, vv, vu)
            
            T += np.trace(self.maps[e][u]) + np.trace(self.maps[e][v])
        
        self.maps = { 
            tuple(self.edges[i]):
                {self.edges[i,0]: self.mu/T * self.maps[self.edges[i]][self.edges[i,0]],
                 self.edges[i,1]: self.mu/T * self.maps[self.edges[i]][self.edges[i,1]]}
            for i in range(self.edges.shape[0])
        }


    def premultiplier(self, Xu, Xv):
        uu = np.linalg.inv(Xu @ Xu.T)
        uv = Xu @ Xv.T
        vv = np.linalg.inv(Xv @ Xv.T)
        vu = Xv @ Xu.T

        return (uu, uv, vv, vu)

    def chi_u(self, uu, uv, vv, vu):

        return ((uu @ uv - np.eye(uu.shape[0])) @ vv @ np.linalg.inv(vu @ uu @ uv @ vv - np.eye(uu.shape[0])) @ vu - np.eye(uu.shape[0])) @ uu

    def chi_v(self, uu, uv, vv, vu):

        return (uu @ uv - np.eye(uu.shape[0])) @ vv @ np.linalg.inv(vu @ uu @ uv @ vv - np.eye(uu.shape[0]))
    
    def agreement(self, u, v):

        X_u = self.node_signals[u*self.stalk_dim:(u+1)*self.stalk_dim,:]
        X_v = self.node_signals[v*self.stalk_dim:(v+1)*self.stalk_dim,:]

        F_u = self.maps[u]
        F_v = self.maps[v]

        return np.linalg.norm(F_u @ X_u - F_v @ X_v, ord = 'fro')
    
    def similarity(self):
        similarity_matrix = np.zeros((self.nodes.shape[0],self.nodes.shape[0]))

        for i in range(self.nodes.shape[0]):
            for j in range(i, self.nodes.shape[0]):
                similarity_matrix[i,j] = self.agreement(self.nodes[i],self.nodes[j])
                similarity_matrix[j,i] = self.agreement(self.nodes[i],self.nodes[j])
        
        # Column stochastic matrix
        return similarity_matrix / np.sum(similarity_matrix, axis = 0)
        

