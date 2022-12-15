import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from skimage.draw import polygon

input_dir = "./samples/example4/"
output_dir = "./samples/example4/"

# read in the source image
im_name = "handwriting.jpg"
input_file = input_dir+im_name
im_s = skio.imread(input_file)
im_s = sk.img_as_float(im_s)

# select and draw the polygon mask
plt.imshow(im_s)
pts = np.array(plt.ginput(-1))
plt.close()
rr, cc = polygon(pts[:, 1], pts[:, 0])
mask = np.zeros(im_s.shape[:2])
mask[rr, cc] = 1
plt.imsave(output_dir+"mask.jpg", mask, cmap="gray")