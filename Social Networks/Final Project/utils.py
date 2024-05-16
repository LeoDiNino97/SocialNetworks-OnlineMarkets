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
                 LR = 0.025):
        
        self.w = w
        self.d = d
        self.V = V
        self.LR = LR
        self.tree = BinaryTree(self.V, self.d)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def grads(self, u_k, Phi, v_j):

        walk, psi = self.tree.traversal(u_k)
        u_k = bin(u_k).lstrip('0b').rjust(self.tree.H, '0')
        u_k = [float(u_k[i]) for i in range(len(u_k))]

        psi_grads = []
        phi_grad = np.zeros_like(Phi[v_j,:])

        for j in range(len(u_k)):
            S = self.sigmoid(np.vdot(psi[j],Phi[v_j,:]))
            psi_grads.append(- ( Phi[v_j,:] * (u_k[j] - S) ))
            phi_grad -= psi[j] * (u_k[j] - 1) + Phi[v_j,:] * (1 - S)

        return walk, psi_grads, phi_grad
    
    def step(self, u_k, Phi, v_j):

        walk, psi_grads, phi_grad = self.grads(u_k, Phi, v_j)
        for i in range(len(walk)):
            walk[i].psi -= self.LR * psi_grads[i]
        Phi[v_j,:] -= self.LR * phi_grad

        return Phi
    
    def window_step(self, W, Phi, v_j):

        for node in W[v_j - self.w : v_j + self.w]:
            Phi = self.step(node, Phi, v_j)

        return Phi
    
    def walk_step(self, W, Phi):

        for node in W:
            Phi = self.window_step(W, Phi, node)

        return Phi

class DeepWalk():
    def __init__(self, 
                 graph: nx.Graph,
                 w: int,
                 d: int,
                 gamma: int,
                 t: int,
                 LR: float):
        
        self.graph = graph

        self.w = w
        self.d = d 

        self.gamma = gamma
        self.t = t

        # Embedding initialization
        self.Phi = np.random.uniform(low=0, 
                                     high=1, 
                                     size = (len(list(self.graph.nodes)), d))

        # Learning rate initialization
        self.LR = LR

        # SkipGram init
        self.SkipGram = SkipGram(w, d, len(list(self.graph.nodes)), LR)
        
    def random_walk(self, v):
        walk = [v]
        curr = v
        for _ in range(self.t):
            w = np.random.choice(np.array([node for node in self.graph.neighbors(curr)]))
            walk.append(w)
            curr = w

        return walk

    def main_loop(self):
        for _ in range(self.gamma):
            for v in self.graph.nodes:
                W = self.random_walk(v)
                self.Phi = self.SkipGram.walk_step(W, self.Phi)