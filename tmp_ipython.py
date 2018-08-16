import numpy as np

img_in = np.random.rand(5,5) *2 -1
img_out = np.random.rand(5,5) *2 -1
mat_in = np.uint8((img_in +1) *127)
mat_out = np.uint8((img_out +1) *127)
mat_index = mat_in / 16
mat_feature = mat_out - mat_in



