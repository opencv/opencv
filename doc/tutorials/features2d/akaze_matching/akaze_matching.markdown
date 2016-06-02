AKAZE local features matching {#tutorial_akaze_matching}
=============================

Introduction
------------

In this tutorial we will learn how to use AKAZE @cite ANB13 local features to detect and match keypoints on
two images.
We will find keypoints on a pair of images with given homography matrix, match them and count the

number of inliers (i. e. matches that fit in the given homography).

You can find expanded version of this example here:
<https://github.com/pablofdezalc/test_kaze_akaze_opencv>

Data
----

We are going to use images 1 and 3 from *Graffity* sequence of Oxford dataset.

![](images/graf.png)

Homography is given by a 3 by 3 matrix:
@code{.none}
7.6285898e-01  -2.9922929e-01   2.2567123e+02
3.3443473e-01   1.0143901e+00  -7.6999973e+01
3.4663091e-04  -1.4364524e-05   1.0000000e+00
@endcode
You can find the images (*graf1.png*, *graf3.png*) and homography (*H1to3p.xml*) in
*opencv/samples/cpp*.

### Source Code

@include cpp/tutorial_code/features2D/AKAZE_match.cpp

### Explanation

-#  **Load images and homography**
    @code{.cpp}
    Mat img1 = imread("graf1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("graf3.png", IMREAD_GRAYSCALE);

    Mat homography;
    FileStorage fs("H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;
    @endcode
    We are loading grayscale images here. Homography is stored in the xml created with FileStorage.

-#  **Detect keypoints and compute descriptors using AKAZE**
    @code{.cpp}
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    AKAZE akaze;
    akaze(img1, noArray(), kpts1, desc1);
    akaze(img2, noArray(), kpts2, desc2);
    @endcode
    We create AKAZE object and use it's *operator()* functionality. Since we don't need the *mask*
    parameter, *noArray()* is used.

-#  **Use brute-force matcher to find 2-nn matches**
    @code{.cpp}
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    @endcode
    We use Hamming distance, because AKAZE uses binary descriptor by default.

-#  **Use 2-nn matches to find correct keypoint matches**
    @code{.cpp}
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    @endcode
    If the closest match is *ratio* closer than the second closest one, then the match is correct.

-#  **Check if our matches fit in the homography model**
    @code{.cpp}
    for(int i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        float dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                           pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = inliers1.size();
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    @endcode
    If the distance from first keypoint's projection to the second keypoint is less than threshold,
    then it it fits in the homography.

    We create a new set of matches for the inliers, because it is required by the drawing function.

-#  **Output results**
    @code{.cpp}
    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("res.png", res);
    ...
    @endcode
    Here we save the resulting image and print some statistics.

### Results

Found matches
-------------

![](images/res.png)

A-KAZE Matching Results
-----------------------
@code{.none}
 Keypoints 1:   2943
 Keypoints 2:   3511
 Matches:       447
 Inliers:       308
 Inlier Ratio: 0.689038}
@endcode
