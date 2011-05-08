Drawing Function of Keypoints and Matches
=========================================

.. index:: drawMatches

drawMatches
---------------

.. c:function:: void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,          const Mat& img2, const vector<KeyPoint>& keypoints2,          const vector<DMatch>& matches1to2, Mat& outImg,          const Scalar& matchColor=Scalar::all(-1),           const Scalar& singlePointColor=Scalar::all(-1),          const vector<char>& matchesMask=vector<char>(),          int flags=DrawMatchesFlags::DEFAULT )

.. c:function:: void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,           const Mat& img2, const vector<KeyPoint>& keypoints2,           const vector<vector<DMatch> >& matches1to2, Mat& outImg,           const Scalar& matchColor=Scalar::all(-1),            const Scalar& singlePointColor=Scalar::all(-1),           const vector<vector<char>>& matchesMask=           vector<vector<char> >(),           int flags=DrawMatchesFlags::DEFAULT )

    :param img1: The first source image.

    :param keypoints1: Keypoints from the first source image.

    :param img2: The second source image.

    :param keypoints2: Keypoints from the second source image.

    :param matches: Matches from the first image to the second one, which means that  ``keypoints1[i]``  has a corresponding point in  ``keypoints2[matches[i]]`` .

    :param outImg: Output image. Its content depends on the ``flags``  value defining what is drawn in the output image. See possible  ``flags``  bit values below.

    :param matchColor: Color of matches (lines and connected keypoints). If  ``matchColor==Scalar::all(-1)`` , the color is generated randomly.

    :param singlePointColor: Color of single keypoints (circles), which means that keypoints do not have the matches. If  ``singlePointColor==Scalar::all(-1)`` , the color is generated randomly.

    :param matchesMask: Mask determining which matches are drawn. If the mask is empty, all matches are drawn.

    :param flags: Flags setting drawing features. Possible  ``flags``  bit values are defined by  ``DrawMatchesFlags``.
    
This function draws matches of keypints from two images in the output image. Match is a line connecting two keypoints (circles). The structure ``DrawMatchesFlags`` is defined as follows:

.. code-block:: cpp

    struct DrawMatchesFlags
    {
        enum
        {
            DEFAULT = 0, // Output image matrix will be created (Mat::create),
                         // i.e. existing memory of output image may be reused.
                         // Two source images, matches, and single keypoints
                         // will be drawn.
                         // For each keypoint, only the center point will be
                         // drawn (without a circle around the keypoint with the
                         // keypoint size and orientation).
            DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
                           // created (using Mat::create). Matches will be drawn
                           // on existing content of output image.
            NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
            DRAW_RICH_KEYPOINTS = 4 // For each keypoint, the circle around
                           // keypoint with keypoint size and orientation will
                           // be drawn.
        };
    };

..

.. index:: drawKeypoints

drawKeypoints
-----------------
.. c:function:: void drawKeypoints( const Mat& image,           const vector<KeyPoint>& keypoints,           Mat& outImg, const Scalar& color=Scalar::all(-1),           int flags=DrawMatchesFlags::DEFAULT )

    Draws keypoints.

    :param image: Source image.

    :param keypoints: Keypoints from the source image.

    :param outImg: Output image. Its content depends on  the ``flags``  value defining what is drawn in the output image. See possible  ``flags``  bit values below.

    :param color: Color of keypoints.

    :param flags: Flags setting drawing features. Possible  ``flags``  bit values are defined by  ``DrawMatchesFlags``. See details above in  :ref:`drawMatches` .

