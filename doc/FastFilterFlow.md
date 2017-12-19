
FastFilterFlow 

Free and open source testbed solver for optical flow, stereo,  diffeomorphism, affine alignment problems in computer vision.

####Adding Fast Filter Flow algorithm for dealing with the pattern of apparent motion of image objects 
between two consecutive frames caused by the movemement of object or camera for OpenCV

Problem:
Given: A pair of images I1 , I2 and a transformation task (e.g., optical
flow, stereo, diffeomorphism) that we wish to estimate.

Objective: Design provably efficient algorithms that work for numerous computer vision problems 
with negligible adjustments. More importantly, a “one stop” testbed solver for such tasks.

Idea:
Idea 1. Reformulate. Exploit structure in the model. Avoid standard solver if possible.
Idea 2. Approximate solutions are enough for most problems.

Algorithm:
Randomized Block Coordinate – Conditional Gradient Method (see reference)

#### Citation
* Filter Flow made Practical: Massively Parallel and Lock-Free
Ravi, Sathya N., Xiong, Yunyang, Mukherjee, Lopamudra, and Singh, Vikas
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , July 2017
