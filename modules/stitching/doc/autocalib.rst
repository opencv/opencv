Autocalibration
===============

.. highlight:: cpp

detail::focalsFromHomography
----------------------------
Tries to estimate focal lengths from the given homography under the assumption that the camera undergoes rotations around its centre only.

.. ocv:function:: void focalsFromHomography(const Mat &H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)

    :param H: Homography.

    :param f0: Estimated focal length along X axis.

    :param f1: Estimated focal length along Y axis. 

    :param f0_ok: True, if f0 was estimated successfully, false otherwise.

    :param f1_ok: True, if f1 was estimated successfully, false otherwise.

detail::estimateFocal
---------------------
Estimates focal lengths for each given camera.

.. ocv:function:: void estimateFocal(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, std::vector<double> &focals)

    :param features: Features of images.

    :param pairwise_matches: Matches between all image pairs.

    :param focals: Estimated focal lengths for each camera.

