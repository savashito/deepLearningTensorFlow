scores = [5.0, 4.0, 0.2] # this were output by linear classifier

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # pass  # TODO: Compute and return softmax(x)
    norm = np.sum(np.exp(x),axis=0)
    return np.exp(x)/norm

# print (softmax(scores))

# # [ 0.8360188   0.11314284  0.05083836]

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)

scores = np.vstack([x,np.ones_like(x),0.2*np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

# plot how probabilies vary when scores are ploted