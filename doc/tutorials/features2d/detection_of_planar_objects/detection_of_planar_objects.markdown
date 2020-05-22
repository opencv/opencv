Detection of planar objects {#tutorial_detection_of_planar_objects}
===========================

@prev_tutorial{tutorial_feature_homography}
@next_tutorial{tutorial_akaze_matching}


The goal of this tutorial is to learn how to use *features2d* and *calib3d* modules for detecting
known planar objects in scenes.

*Test data*: use images in your data folder, for instance, box.png and box_in_scene.png.

-   Create a new console project. Read two input images. :

        Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
        Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

-   Detect keypoints in both images and compute descriptors for each of the keypoints. :

        // detecting keypoints
        Ptr<Feature2D> surf = SURF::create();
        vector<KeyPoint> keypoints1;
        Mat descriptors1;
        surf->detectAndCompute(img1, Mat(), keypoints1, descriptors1);

        ... // do the same for the second image

-   Now, find the closest matches between descriptors from the first image to the second: :

        // matching descriptors
        BruteForceMatcher<L2<float> > matcher;
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

-   Visualize the results: :

        // drawing the results
        namedWindow("matches", 1);
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
        imshow("matches", img_matches);
        waitKey(0);

-   Find the homography transformation between two sets of points: :

        vector<Point2f> points1, points2;
        // fill the arrays with the points
        ....
        Mat H = findHomography(Mat(points1), Mat(points2), RANSAC, ransacReprojThreshold);

-   Create a set of inlier matches and draw them. Use perspectiveTransform function to map points
    with homography:

    Mat points1Projected; perspectiveTransform(Mat(points1), points1Projected, H);

-   Use drawMatches for drawing inliers.
