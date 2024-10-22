import numpy as np 
import networkx as nx
from tqdm import tqdm as tqdm 
from matplotlib import pyplot as plt
from itertools import combinations

#######################
#### ENVIRONMENTS #####
#######################

class Environment:
    def __init__(self, N, d, C, mode):

        self.N = N
        self.mode = mode

        if self.mode == 'Uniform':

            # Ensure consistency for the dimension of the cube
            self.d = C * np.log(N)

        else:
            self.d = d

        self.generate()
        
    def generate(self):

        if self.mode == 'Uniform':
            self.env = np.random.choice([-1,1], 
                                        size = (int(self.N),int(self.d)))
        
        else:
            self.env = np.random.multivariate_normal(np.zeros(self.d),
                                                    np.eye(self.d),
                                                    self.N)
            
    def distanceBasedPermutation(self, 
                                 ord = 2):
    
        DBP = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                DBP[i,j] = np.linalg.norm(self.env[i] - self.env[j], ord)

        return np.argsort(DBP, 1)

    def randomizedConstruction(self,
                               ord = 2,
                               m = None,
                               scheme = 'paper'):

        A = np.zeros((self.N,self.N))
        DBP = self.distanceBasedPermutation(ord)

        if not m:
            m = int(np.floor(np.sqrt(3 * self.N * np.log(self.N))))

        for i in range(self.N):

            # Deterministic step
            A[i, DBP[i, 1:m]] = 1
            
            # Randomized step
            if scheme == 'paper':
                S = int(np.ceil(3 * self.N * np.log(self.N)/m))
            else:
                S = m
                
            R = np.random.choice(np.concatenate([np.arange(0,i), np.arange(i+1,self.N)]), S)

            A[i, R] = 1

        return A
    
    def randomizedPKConstruction(self,
                                 ord = 2,
                                 k = 10,
                                 m = None,
                                 scheme = 'Paper'):

        A = np.zeros((self.N,self.N))
        DBP = self.distanceBasedPermutation(ord)
        if not m:
            m = int(np.floor(np.sqrt(3 * self.N * np.log(self.N))))

        for i in range(self.N):

            # Deterministic step
            A[i, DBP[i, 1:m]] = 1
            
            # Randomized step
            if scheme == 'Paper':
                S = int(np.ceil(3 * self.N * np.log(n)/m))
            else:
                S = m
            
            for i in range(S):
                
                # Power of k construction
                R = np.random.choice(np.concatenate([np.arange(0,i), np.arange(i+1,self.N)]), k)

                in_degrees = np.sum(A, axis = 0)[R]

                A[i, R[np.argmin(in_degrees)]] = 1

        return A
    
########################
#### ROUTING ALGOS #####
########################

def greedyRouting(s: int,
                  t: int,
                  x: np.array,
                  A: np.array
                  ):
    
    done = False
    j = s

    steps = 0

    while not done:
        
        if np.all(A[j] == 0):
            done = True

        else:
            neighs = np.where(A[j,:] == 1)[0]
            X_ = np.copy(x[neighs])
            h = neighs[np.argmin(np.linalg.norm(X_ - x[t], axis = 1))]

            if np.linalg.norm(x[t] - x[h]) < np.linalg.norm(x[t] - x[j]):
                j = h
                steps += 1
                
            else:
                done = True

    return j, steps


def secondDegreedy(s: int,
                   t: int,
                   x: np.array,
                   A: np.array,
                   two_hops = False
                   ):
        
    done = False
    j = s

    steps = 0

    while not done:
        
        if np.all(A[j] == 0):
            done = True

        else:
            neighs = np.where(A[j,:] == 1)[0]
            X_ = np.copy(x[neighs])
            hs = neighs[np.argsort(np.linalg.norm(X_ - x[t], axis = 1))[0:2]]
            
            if t in hs:
                j = t
                steps += 1
                done = True
                break

            neighs = np.row_stack(np.where(A[hs,:] == 1))
            X_ = np.copy(x[neighs[1,:]])
            neighs = np.row_stack([neighs, np.linalg.norm(X_ - x[t], axis = 1)])

            h = int(hs[int(neighs[0,np.argmin(neighs[2,:])])])

            if two_hops:
                h = int(int(neighs[1,np.argmin(neighs[2,:])]))

            if np.linalg.norm(x[t] - x[h]) < np.linalg.norm(x[t] - x[j]):
                j = h
                steps += 1

            else:
                done = True


    return x[j], steps


def secondDegreedy_full(s: int,
                        t: int,
                        x: np.array,
                        A: np.array,
                        ):
        
    done = False
    j = s

    steps = 0

    while not done:
        
        if np.all(A[j] == 0):
            done = True

        else:
            neighs = np.where(A[j,:] == 1)[0]
            X_ = np.copy(x[neighs])
            
            if t in neighs:
                j = t
                steps += 1
                done = True
                break

            neighs = np.row_stack(np.where(A[neighs,:] == 1))
            X_ = np.copy(x[neighs[1,:]])
            neighs = np.row_stack([neighs, np.linalg.norm(X_ - x[t], axis = 1)])

            h = int(int(neighs[1,np.argmin(neighs[2,:])]))

            if np.linalg.norm(x[t] - x[h]) < np.linalg.norm(x[t] - x[j]):
                j = h
                steps += 1

            else:
                done = True


    return x[j], steps

# Greedy routing algorithm relaxed - we can visit further nodes

def greedyRoutingRelax(s: int,
                       t: int,
                       x: np.array,
                       A: np.array,
                       MAX_WORSE = 50
                       ):
    
    done = False
    j = s

    steps = 0
    worse = 0
    
    while not done and worse < MAX_WORSE:
        if np.all(A[j] == 0) or j == t:
            done = True
            break

        else:

            neighs = np.where(A[j,:] == 1)[0]
            X_ = np.copy(x[neighs])
            h = neighs[np.argmin(np.linalg.norm(X_ - x[t], axis = 1))]

            if np.linalg.norm(x[t] - x[h]) < np.linalg.norm(x[t] - x[j]):

                j = h
                steps += 1

                if j == t:

                    done = True
                    break
            
            else:

                # Visit a further node among the neighs
                j = np.random.choice(neighs)
                worse += 1
                steps += 1

  
    return j, steps, worse