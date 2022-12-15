import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio

input_dir = "./samples/example4/"
output_dir = "./samples/example4/"

# read in the source image
im_name = "handwriting.jpg"
input_file = input_dir+im_name
im_s = skio.imread(input_file)
im_s = sk.img_as_float(im_s)

# read in the target image
im_name = "wall.jpg"
input_file = input_dir+im_name
im_t = skio.imread(input_file)
im_t = sk.img_as_float(im_t)

# select the location where the source image should be blended
plt.imshow(im_s)
pts_s = np.array(plt.ginput(2))
plt.close()

plt.imshow(im_t)
pts_t = np.array(plt.ginput(2))
plt.close()

scale = np.linalg.norm(pts_t[1]-pts_t[0])/np.linalg.norm(pts_s[1]-pts_s[0])
translate = pts_t[0]-pts_s[0]*scale # first resize, then translate

f = open(output_dir+"transform.txt", 'w')
f.write(f"translate,{int(translate[0])},{int(translate[1])}\n")
f.write(f"scale,{scale}\n")
f.close()
