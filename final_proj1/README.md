# CS194-26 Final Project 1 - Gradient Domain Fushion

Name: Songwen Zhao

SID: 3038663175

Email: songwenzhao@berkeley.edu

Link to web page submission: https://inst.eecs.berkeley.edu/~cs194-26/fa22/upload/files/projFinalAssigned/cs194-26-aft/

## Project File

`samples/` directory should contain all the inputs including images, mask images and transformation data.

`output/` directory should contain all the output images in .jpg format.

`getMask.py` has the code for selecting a polygon region in the source image and outputing the mask. No need to run this code again.

`alignSource.py` has the code for aligning source region to the target image. It outputs the transformation parameters including translation and scaling to a .txt file. No need to run this code again.

`main.ipynb` has code that can execute all parts in full generating all the results.

`main.py` has the same code as in `main.ipynb`.