# Fast inner loop for DBSCAN.
# Author: Lars Buitinck
# License: 3-clause BSD
#
# cython: boundscheck=False, wraparound=False

cimport cython
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np


# Work around Cython bug: C++ exceptions are not caught unless thrown within
# a cdef function with an "except +" declaration.
cdef inline void push(vector[np.npy_intp] &stack, np.npy_intp i) except +:
    stack.push_back(i)

def merge(np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
          np.npy_intp old_label,
          np.npy_intp new_label):
    cdef np.npy_intp i
    
    for i in range(labels.shape[0]):
        if labels[i] == old_label:
            labels[i] = new_label

def dbscan_inner(np.uint32_t start,
                 np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
                 np.ndarray[object, ndim=1] neighborhoods,
                 np.ndarray[np.npy_intp, ndim=1, mode='c'] labels):
    cdef np.npy_intp i, label_num, v, v_label
    cdef np.ndarray[np.npy_intp, ndim=1] neighb
    cdef vector[np.npy_intp] stack

    # Max(labels)
    label_num = -1
    for l in labels:
        if l > label_num:
            label_num = l
    label_num += 1

    for i in range(start, labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        v_label = labels[v]
                        if v_label == -1:
                            push(stack, v)
                        else:
                            merge(labels, v_label, label_num)


            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()

        label_num += 1
