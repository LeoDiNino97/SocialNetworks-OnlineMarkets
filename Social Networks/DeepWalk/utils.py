import numpy as np
import networkx as nx 
from collections import deque 
from tqdm import tqdm

import numpy as np

# DeepWalk utils

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
                 LR_scheduler = 1e-6
                 ):
        
        self.w = w
        self.d = d
        self.V = V
        self.LR = LR
        self.LR_scheduler = LR_scheduler
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

        if self.LR > 1e-5:
            self.LR -= self.LR_scheduler
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
                 graph,
                 w: int,
                 d: int,
                 gamma: int,
                 t: int,
                 LR: float,
                 LR_scheduler: float,
                 biased: bool,
                 nodes_signals: np.array,
                 stalk_dim: int, 
                 mu: float
                 ):
        
        if type(graph) == nx.Graph:
            self.graph = graph
            self.nodes = np.array(self.graph.nodes)

            self.edges = np.zeros((2, len(self.graph.edges)), dtype='int32')
            for i, edge in enumerate(list(self.graph.edges)):
                self.edges[0,i] = edge[0]
                self.edges[1,i] = edge[1]

        elif type(graph) == dict:
            self.graph = graph
            self.nodes = np.array(self.graph['nodes'])

            self.graph['neighbors'] = {
                node: []
                for node in self.graph['nodes']
            }

            for edge in self.graph['edges']:
                self.graph['neighbors'][edge[0]].append(edge[1])
                self.graph['neighbors'][edge[1]].append(edge[0])

            self.edges = np.zeros((2, len(self.graph['edges'])), dtype='int32')
            for i, edge in enumerate(list(self.graph['edges'])):
                self.edges[0,i] = edge[0]
                self.edges[1,i] = edge[1]            
            
        
        else:
            raise('Unsupported data structure for underlying graph, please provide it as a nx.Graph or a dict containing nodes and edges list. ')


        self.nodes_signals = nodes_signals
        self.stalk_dim = stalk_dim  

        self.d = d 

        self.gamma = gamma
        self.t = t

        # Embedding initialization
        self.Phi = np.random.uniform(low=0, 
                                     high=1, 
                                     size = (len(self.nodes), d))

        # Learning rate initialization
        self.LR = LR
        self.LR_scheduler = LR_scheduler

        # SkipGram initialization
        self.SkipGram = SkipGram(w, 
                                 d, 
                                 len(self.nodes), 
                                 LR,
                                 LR_scheduler)

        # Sheaf initialization
        if self.nodes_signals is not None:
            self.Sheaf = GraphSheaf(self.nodes, 
                                    self.edges, 
                                    stalk_dim, 
                                    nodes_signals, 
                                    mu)

        # Biased sampling attribute
        self.biased = biased
        
    def biased_sampling(self, node):
        
        probs = self.Sheaf.similarity_matrix[:,node]
        return np.random.choice(self.nodes, p = probs)

    def random_walk(self, v):
        walk = [v]
        curr = v
        for _ in range(self.t):
            if not self.biased:
                if type(self.graph) == nx.Graph:
                    w = np.random.choice(np.array([node for node in self.graph.neighbors(curr)]))
                else:
                    w = np.random.choice(np.array([node for node in self.graph['neighbors'][curr]]))
                walk.append(w)
                curr = w
            else:
                w = self.biased_sampling(curr)
                walk.append(w)
                curr = w

        return walk

    def train(self):
        for _ in tqdm(range(self.gamma)):
            np.random.shuffle(self.nodes)

            for v in self.nodes:
                W = self.random_walk(v)
                self.Phi = self.SkipGram.walk_step(W, self.Phi)

class GraphSheaf():
    def __init__(self, 
                 nodes: np.array,
                 edges: np.array,
                 D: int,
                 node_signals: np.array,
                 mu: float):

        self.nodes = nodes
        self.edges = edges

        edges_tail = edges[0,:]
        edges_head = edges[1,:]

        self.edges_ = [(edges_tail[i].item(), 
                        edges_head[i].item()) for i in range(edges.shape[1])]
        self.D = D
        
        self.node_signals = node_signals

        self.maps = { 
            self.edges_[i]:
                {self.edges_[i][0]: None,
                 self.edges_[i][1]: None}
            for i in range(len(self.edges_))
        }

        self.mu = mu
        self.premultipliers = {
            e: {
                int(e[0]): 0,
                int(e[1]): 0,
                (int(e[0]),int(e[1])): 0,
                (int(e[1]),int(e[0])): 0,
            }
            for e in self.edges_
        }

    def builder(self):
        self.premultiplying()
        self.sheaf_builder()
        self.similarity_matrix = self.similarity()

    def premultiplying(self):
        def process_edge(edge):
            u = int(edge[0])
            v = int(edge[1])

            X_u = self.node_signals[u * self.D:(u + 1) * self.D, :]
            X_v = self.node_signals[v * self.D:(v + 1) * self.D, :]

            uu, uv, vv, vu = self.premultiplier(X_u, X_v)

            self.premultipliers[(u,v)][u] = uu
            self.premultipliers[(u,v)][v] = vv
            self.premultipliers[(u,v)][(u,v)] = uv
            self.premultipliers[(u,v)][(v,u)] = vu

        # Apply the process_edge function to each row in the edges array
        np.apply_along_axis(process_edge, 0, self.edges)

            
    def sheaf_builder(self):

        T = 0
        for i in tqdm(range(len(self.edges_))):
            e = self.edges_[i]

            u = e[0]
            v = e[1]

            uu, vv, uv, vu = list(self.premultipliers[e].values())

            self.maps[e][u] = self.chi_u(uu, uv, vv, vu)
            self.maps[e][v] = self.chi_u(uu, uv, vv, vu)
            
            T += np.trace(self.maps[e][u]) + np.trace(self.maps[e][v])
        
        self.maps = { 
            self.edges_[i]:
                {self.edges_[i][0]: self.mu/T * self.maps[self.edges_[i]][self.edges_[i][0]],
                 self.edges_[i][1]: self.mu/T * self.maps[self.edges_[i]][self.edges_[i][1]]}
            for i in range(len(self.edges_))
        }

    def premultiplier(self, Xu, Xv):
        uu = np.linalg.pinv(Xu @ Xu.T)
        uv = Xu @ Xv.T
        vv = np.linalg.pinv(Xv @ Xv.T)
        vu = Xv @ Xu.T

        return (uu, uv, vv, vu)

    def chi_u(self, uu, uv, vv, vu):

        return ((uu @ uv - np.eye(uu.shape[0])) @ vv @ np.linalg.inv(vu @ uu @ uv @ vv - np.eye(uu.shape[0])) @ vu - np.eye(uu.shape[0])) @ uu

    def chi_v(self, uu, uv, vv, vu):

        return (uu @ uv - np.eye(uu.shape[0])) @ vv @ np.linalg.inv(vu @ uu @ uv @ vv - np.eye(uu.shape[0]))
    
    def agreement(self, u, v):

        X_u = self.node_signals[u*self.D:(u+1)*self.D,:]
        X_v = self.node_signals[v*self.D:(v+1)*self.D,:]

        if (u,v) in self.edges_:
            F_u = self.maps[(u,v)][u]
            F_v = self.maps[(u,v)][v]

        elif (v,u) in self.edges_:
            F_u = self.maps[(v,u)][u]
            F_v = self.maps[(v,u)][v]

        else:
            return np.inf

        return np.linalg.norm(F_u @ X_u - F_v @ X_v, ord = 'fro')
    
    def similarity(self):
        similarity_matrix = np.zeros((self.nodes.shape[0],self.nodes.shape[0]))

        for i in range(self.nodes.shape[0]):
            for j in range(i, self.nodes.shape[0]):
                if (i,j) or (j,i) in self.edges_:
                    similarity_matrix[i,j] = np.exp(-self.agreement(self.nodes[i],self.nodes[j])/0.01)
                    similarity_matrix[j,i] = np.exp(-self.agreement(self.nodes[i],self.nodes[j])/0.01)
        
        # Column stochastic matrix
        return similarity_matrix / np.sum(similarity_matrix, axis = 0)