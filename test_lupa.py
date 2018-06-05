import lutorpy

import numpy as np

require("tripletSelection")
require("torch")
embeddings = torch.DoubleTensor(12,3)
numImages = 10
numPerClass = [5,5]
print(numPerClass)
peoplePerBatch = 2
alpha = 1
embSize = 3
cuda = False
trip = triplets(embeddings, numImages, numPerClass, peoplePerBatch, alpha, embSize, cuda)

a = trip[0][0].asNumpyArray()
p = trip[0][1].asNumpyArray()
n = trip[0][2].asNumpyArray()
ids = trip[1].asNumpyArray()


print(a,p,n,ids)

