import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img1 = Image.open('Flair/0001_090.png')
img1_mat = np.array(img1, dtype=np.float)
img2 = Image.open('T1/0001_090.png')
img2_mat = np.array(img2, dtype=np.float)

plt.imshow(img1_mat, cmap='gray')
plt.show()

img1_prob = img1_mat/ 127.0 -1
img2_prob = img2_mat/ 127.0 -1

mat_index = np.int32(img1_mat / cell_range)
mat_feature = img2_prob - img1_prob

cell_range = 16
bins_num = 256 / 16
bins_idx = np.zeros(bins_num)
for idx in range(bins_num):
    num = np.sum(mat_index == idx)
    if num >0:
        bins_idx[idx] = np.mean(mat_feature[mat_index == idx])

plt.plot(bins_idx)
plt.savefig('tmp.png')
