# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import sys
import os
from numba import jit


def inverse(permutation): #takes in a permutation of 0,1,,,n as a list or numpy array 
    res = np.empty_like(permutation)
    res[permutation] = np.arange(len(permutation))
    return res #returns the inverse permutation as a numpy array. 
"""
A permutation of {0,1,..,n} is a bijective map from {0,1,..,n} to itself.  This
function takes in a permutation in the form of an array or list (v_{0},..,v_{n})
with v_{i} in {0,1,..,n} being the image of i.  Given a permutation g, 
the function returns the inverse permutation, (in array form).
"""

@jit #Don't know what this means
def cycles(permutation):
    N = len(permutation) 
    labels = np.empty(N, int)
    labels.fill(-1)
    label = 0
    k = 0
    order = np.empty(N, int)
    for i in range(N):
        if labels[i] != -1:
            continue
        while True:
            order[k] = i
            k = k + 1
            labels[i] = label
            i = permutation[i]
            if labels[i] != -1:
                break
        label += 1

    return order, labels

"""
Let g be a permutation and let gA be it's representation in array form.
Suppose we compute cycles(gA). 
 
Suppose that the index k appears r times in 'labels.' The, the slice 
order[k:k+r] is an r-cycle. 
When an index k appears once in labels at position i in the array.
Then, g(order[i])=order[i].

When r>1, let i be the smallest index at which k occurs in 'labels.' 
Then, we have g(order[i+j])=order[i+j+1] for
 j=0,..,r-2
 and g(order[i+r-1])=order[i].
 
 
e.g. 
Let g be the permutation such that gA=[0,1,5,3,2,4,6]. Evaluating
 cycles ([0,1,5,3,2,4,6]), returns [0,1,2,5,4,3,6] as 'order' and 
 [0,1,2,3,2,2,4] as 'labels.' 

The number 0 appears once in labels, at position 0. Order[0]=0 is a 1-cycle
 i.e. g(0)=0. 
 By similar arguments, order[1]=1 is a 1-cycle
The number  2 appears 3 times in labels, appearing first at position 2.  So
 order[2:5] is a 3-cycle.
 order[2:5]=[2,5,4].  By saying '[2,5,4]' is a cycle, we mean precisely that g
 sends 2 to 5, 5 to 4 and 4 to 2.
The number 3 appears once in 'labels,' at position 3.  order[3] is a 1-cycle,
i.e.g(order[3])=order[3]
Finally, 4 appears once in labels at position 6.  So order[6] is a 1-cycle.
"""

def cycle(permutation, idx): #input is a list or array, and an 'in range' index.
    res = []
    x = idx
    for _ in range(50000):
        res.append(x)
        x = permutation[x]
        if x == idx:
            return res
    os._exit(1)
    # raise Exception('cycle failed to terminate PILY')
"""Input is a permutation in list or array form and an index idx which is in 
range.   Returns the cycle starting at idx.
The returned list, res will be such that g(res[i])=res[i+1] for i=0,,len(res)-2
 and g(res[len(res)-1] = res[0]).
Example: Let g be the permutation such that its array representation is
gA=[0,1,5,3,2,4]. Calling cycle(gA,2) returns [2,5,4].
    Check: g(2)=5,g(5)=4 and g(4)=2.
"""



