Depth Map from Stereo Images {#tutorial_py_depthmap}
============================

Goal
----

We will learn to create a depth map from stereo images.

Basics
------

In the last session, we saw basic concepts like epipolar constraints and other related terms. We also
saw that if we have two images of same scene, we can get depth information from that in an intuitive
way. Below is an image and some simple mathematical formulas which prove that intuition.

![image](images/stereo_depth.jpg)

The above diagram contains equivalent triangles.

Writing their equivalent equations will yield us
following result:

\f[
disparity = x - x' = \frac{Bf}{Z}
\f]

$x$ and $x'$ are the distance between points in image plane corresponding to the scene point 3D and
their camera center. $B$ is the distance between two cameras (which we know) and $f$ is the focal
length of camera (measured in pixels!) So in short, the above equation says that the depth of a point in a
scene is inversely proportional to the difference in distance of corresponding image points and
their camera centers. So with this information, we can derive the depth of all pixels in an image.

So it finds corresponding matches between two images. We have already seen how epiline constraint
make this operation faster and accurate. Once it finds matches, it finds the disparity. Let's see
how we can do it with OpenCV.

Example
----

The following example is derived from the [Middlebury 2021 dataset](https://vision.middlebury.edu/stereo/data/scenes2021/)
The dataset comes with camera calibration data, which will be essential for computing a depth map:

```
cam0=[1758.23 0 872.36; 0 1758.23 552.32; 0 0 1]
cam1=[1758.23 0 872.36; 0 1758.23 552.32; 0 0 1]
doffs=0
baseline=124.86
width=1920
height=1080
ndisp=310
vmin=90
vmax=280
```

Below is a stereo view of the scene.

![image](images/stereo_chess_images.png)

Then, using the below code, we arrive at the following depth map.

![image](images/depth_map_comparison.png)

> **__Note:__**
> OpenCV offers two stereo corespondence objects: `StereoBM` and `StereoSGBM`, which implement a block matching algorithm and a modified H. Hirschmuller algorithm respectively. StereoBM depends heavily on block size: a smaller block size will result in more detail and more noise in the disparity map than a large block size. StereoSGBM takes less influence from block size because it strives to minimize global energy cost across the image, not across blocks, which produces smoother disparity maps. StereoSGBM is more computationally expensive but it can be hardware-optimized. StereoBM is suitable for real-time applications but at the cost of a certain level of detail. As you can see in the above image, one drawback to the BM algorithm is that it struggles to calculate disparity on certain surface textures, which results in well defined object edges but gaps in their interiors. The following code example uses StereoSGBM.

The code includes multiple parameters and post processing steps that are essential for creating a clear depth map with minimal noise. These parameters must be tuned to each scene to get the optimal depth map.

```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


#units must be pixels, not milimeters
focalLength = 1758.23
#units are meters, this will match the units of the depth map
baseline = 0.12486
nDisparities = 16*20
bSize = 7

imgL = cv.imread('path_to_left_image.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('path_to_right_image.png', cv.IMREAD_GRAYSCALE)


stereo = cv.StereoSGBM.create(
    minDisparity=0,
    numDisparities=nDisparities,
    blockSize=bSize,
    P1=8 * 1 * 7**2,
    P2=32 * 1 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=32
)


# normalized disparity map
disparity = stereo.compute(imgL,imgR).astype(np.float32) / 16.0

# masking out zeroes
depthMap = np.zeros_like(disparity)
valid = disparity > 0
depthMap[valid] = (focalLength * baseline) / disparity[valid]

# cropping image
depth_vis = depthMap.copy()
depth_vis = depth_vis[:, nDisparities:]


# Clip far depths
max_depth = np.nanpercentile(depth_vis, 85)
depth_vis = np.clip(depth_vis, 0, max_depth)

plt.imshow(depth_vis, cmap='gray')
plt.colorbar(label='Depth (m)')
plt.title('Depth Map')
plt.axis('off')
plt.show()
```

There are a number of parameters that you can pass to StereoSGBM that you can fine tune to get better results:

- Speckle range and size: Block-based matchers often produce "speckles" near the boundaries of objects, where the matching window catches the foreground on one side and the background on the other. In this scene it appears that the matcher is also finding small spurious matches in the projected texture on the table. To get rid of these artifacts we post-process the disparity image with a speckle filter controlled by the speckle_size and speckle_range parameters. speckle_size is the number of pixels below which a disparity blob is dismissed as "speckle." speckle_range controls how close in value disparities must be to be considered part of the same blob.
- Number of disparities: How many pixels to slide the window over. The larger it is, the larger the range of visible depths, but more computation is required.
- min_disparity: the offset from the x-position of the left pixel at which to begin searching.
- uniqueness_ratio: Another post-filtering step. If the best matching disparity is not sufficiently better than every other disparity in the search range, the pixel is filtered out. You can try tweaking this if texture_threshold and the speckle filtering are still letting through spurious matches.
- prefilter_size and prefilter_cap: The pre-filtering phase, which normalizes image brightness and enhances texture in preparation for block matching. Normally you should not need to adjust these.

Additional Resources
--------------------
- [Ros stereo img processing wiki page](http://wiki.ros.org/stereo_image_proc/Tutorials/ChoosingGoodStereoParameters)

Exercises
---------

-#  OpenCV samples contain an example of generating disparity map and its 3D reconstruction. Check
    stereo_match.py in OpenCV-Python samples.
