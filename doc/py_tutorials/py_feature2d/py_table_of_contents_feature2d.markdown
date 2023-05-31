Feature Detection and Description {#tutorial_py_table_of_contents_feature2d}
=================================

-   @subpage tutorial_py_features_meaning

    What are the main
    features in an image? How can finding those features be useful to us?

-   @subpage tutorial_py_features_harris

    Okay, Corners are good
    features? But how do we find them?

-   @subpage tutorial_py_shi_tomasi

    We will look into
    Shi-Tomasi corner detection

-   @subpage tutorial_py_sift_intro

    Harris corner detector
    is not good enough when scale of image changes. Lowe developed a breakthrough method to find
    scale-invariant features and it is called SIFT

-   @subpage tutorial_py_surf_intro

    SIFT is really good,
    but not fast enough, so people came up with a speeded-up version called SURF.

-   @subpage tutorial_py_fast

    All the above feature
    detection methods are good in some way. But they are not fast enough to work in real-time
    applications like SLAM. There comes the FAST algorithm, which is really "FAST".

-   @subpage tutorial_py_brief

    SIFT uses a feature
    descriptor with 128 floating point numbers. Consider thousands of such features. It takes lots of
    memory and more time for matching. We can compress it to make it faster. But still we have to
    calculate it first. There comes BRIEF which gives the shortcut to find binary descriptors with
    less memory, faster matching, still higher recognition rate.

-   @subpage tutorial_py_orb

    SIFT and SURF are good in what they do, but what if you have to pay a few dollars every year to use them in your applications? Yeah, they are patented!!! To solve that problem, OpenCV devs came up with a new "FREE" alternative to SIFT & SURF, and that is ORB.

-   @subpage tutorial_py_matcher

    We know a great deal about feature detectors and descriptors. It is time to learn how to match different descriptors. OpenCV provides two techniques, Brute-Force matcher and FLANN based matcher.

-   @subpage tutorial_py_feature_homography

    Now we know about feature matching. Let's mix it up with calib3d module to find objects in a complex image.
