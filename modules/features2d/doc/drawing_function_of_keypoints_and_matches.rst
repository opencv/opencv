Drawing Function of Keypoints and Matches
=========================================

.. highlight:: cpp

.. index:: drawMatches

cv::drawMatches
---------------
.. cfunction:: void drawMatches( const Mat\& img1, const vector<KeyPoint>\& keypoints1,          const Mat\& img2, const vector<KeyPoint>\& keypoints2,          const vector<DMatch>\& matches1to2, Mat\& outImg,          const Scalar\& matchColor=Scalar::all(-1),           const Scalar\& singlePointColor=Scalar::all(-1),          const vector<char>\& matchesMask=vector<char>(),          int flags=DrawMatchesFlags::DEFAULT )

    This function draws matches of keypints from two images on output image.
Match is a line connecting two keypoints (circles).

.. cfunction:: void drawMatches( const Mat\& img1, const vector<KeyPoint>\& keypoints1,           const Mat\& img2, const vector<KeyPoint>\& keypoints2,           const vector<vector<DMatch> >\& matches1to2, Mat\& outImg,           const Scalar\& matchColor=Scalar::all(-1),            const Scalar\& singlePointColor=Scalar::all(-1),           const vector<vector<char>>\& matchesMask=           vector<vector<char> >(),           int flags=DrawMatchesFlags::DEFAULT )

    :param img1: First source image.

    :param keypoints1: Keypoints from first source image.

    :param img2: Second source image.

    :param keypoints2: Keypoints from second source image.

    :param matches: Matches from first image to second one, i.e.  ``keypoints1[i]``                                         has corresponding point  ``keypoints2[matches[i]]`` .

    :param outImg: Output image. Its content depends on  ``flags``  value
                                   what is drawn in output image. See below possible  ``flags``  bit values.

    :param matchColor: Color of matches (lines and connected keypoints).
                                           If  ``matchColor==Scalar::all(-1)``  color will be generated randomly.

    :param singlePointColor: Color of single keypoints (circles), i.e. keypoints not having the matches.
                                                If  ``singlePointColor==Scalar::all(-1)``  color will be generated randomly.

    :param matchesMask: Mask determining which matches will be drawn. If mask is empty all matches will be drawn.

    :param flags: Each bit of  ``flags``  sets some feature of drawing.
                                  Possible  ``flags``  bit values is defined by  ``DrawMatchesFlags`` , see below. ::

    struct DrawMatchesFlags
    {
        enum{ DEFAULT = 0, // Output image matrix will be created (Mat::create),
                           // i.e. existing memory of output image may be reused.
                           // Two source image, matches and single keypoints
                           // will be drawn.
                           // For each keypoint only the center point will be
                           // drawn (without the circle around keypoint with
                           // keypoint size and orientation).
              DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
                           // created (Mat::create). Matches will be drawn
                           // on existing content of output image.
              NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
              DRAW_RICH_KEYPOINTS = 4 // For each keypoint the circle around
                           // keypoint with keypoint size and orientation will
                           // be drawn.
            };
    };
..

.. index:: drawKeypoints

cv::drawKeypoints
-----------------
.. cfunction:: void drawKeypoints( const Mat\& image,           const vector<KeyPoint>\& keypoints,           Mat\& outImg, const Scalar\& color=Scalar::all(-1),           int flags=DrawMatchesFlags::DEFAULT )

    Draw keypoints.

    :param image: Source image.

    :param keypoints: Keypoints from source image.

    :param outImg: Output image. Its content depends on  ``flags``  value
                                   what is drawn in output image. See possible  ``flags``  bit values.

    :param color: Color of keypoints

    .

    :param flags: Each bit of  ``flags``  sets some feature of drawing.
                                  Possible  ``flags``  bit values is defined by  ``DrawMatchesFlags`` ,
                                  see above in  :func:`drawMatches` .

