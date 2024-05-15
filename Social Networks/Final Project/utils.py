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

class HCBinaryTree():
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
            if prev.children[i]: # Child Not Null
                prev = prev.children[i]
            else:   # child[i] is Null
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
        return path


class SkipGram():
    def __init__(self,
                 w: int,
                 d: int,
                 LR = 0.025):
        
        self.w = w
        self.d = d
        self.LR = LR

    def step(self, W):
        pass
    
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
        self.psi = np.random.uniform((len(list(self.graph.nodes)), d))

        # Learning rate initialization
        self.LR = LR

        # SkipGram init
        self.SkipGram = SkipGram()
        
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
                self.SkipGram.step(W)