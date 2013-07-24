.. _detectionOfPlanarObjects:

Detection of planar objects
***************************

.. highlight:: cpp

The goal of this tutorial is to learn how to use *features2d* and *calib3d* modules for detecting known planar objects in scenes.

*Test data*: use images in your data folder, for instance, ``box.png`` and ``box_in_scene.png``.

#.
    Create a new console project. Read two input images. ::

        Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

#.
    Detect keypoints in both images. ::

        // detecting keypoints
        FastFeatureDetector detector(15);
        vector<KeyPoint> keypoints1;
        detector.detect(img1, keypoints1);

        ... // do the same for the second image

#.
    Compute descriptors for each of the keypoints. ::

        // computing descriptors
        SurfDescriptorExtractor extractor;
        Mat descriptors1;
        extractor.compute(img1, keypoints1, descriptors1);

        ... // process keypoints from the second image as well

#.
    Now, find the closest matches between descriptors from the first image to the second: ::

        // matching descriptors
        BruteForceMatcher<L2<float> > matcher;
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

#.
    Visualize the results: ::

        // drawing the results
        namedWindow("matches", 1);
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
        imshow("matches", img_matches);
        waitKey(0);

#.
    Find the homography transformation between two sets of points: ::

        vector<Point2f> points1, points2;
        // fill the arrays with the points
        ....
        Mat H = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);


#.
    Create a set of inlier matches and draw them. Use perspectiveTransform function to map points with homography:

        Mat points1Projected;
        perspectiveTransform(Mat(points1), points1Projected, H);

#.
    Use ``drawMatches`` for drawing inliers.
