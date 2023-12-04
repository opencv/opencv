USAC: Improvement of Random Sample Consensus in OpenCV {#tutorial_usac}
==============================

@tableofcontents

@prev_tutorial{tutorial_interactive_calibration}

|    |    |
| -: | :- |
| Original author | Maksym Ivashechkin |
| Compatibility | OpenCV >= 4.0 |

This work was integrated as part of the Google Summer of Code (August 2020).

Contribution
------

The integrated part to OpenCV `calib3d` module is RANSAC-based universal
framework USAC (`namespace usac`) written in C++. The framework includes
different state-of-the-arts methods for sampling, verification or local
optimization. The main advantage of the framework is its independence to
any estimation problem and modular structure. Therefore, new solvers or
methods can be added/removed easily. So far it includes the following
components:

1.  Sampling method:

    1.  Uniform – standard RANSAC sampling proposed in @cite FischlerRANSAC which draw
        minimal subset independently uniformly at random. *The default
        option in proposed framework*.

    2.  PROSAC – method @cite ChumPROSAC that assumes input data points sorted by
        quality so sampling can start from the most promising points.
        Correspondences for this method can be sorted e.g., by ratio of
        descriptor distances of the best to second match obtained from
        SIFT detector. *This is method is recommended to use because it
        can find good model and terminate much earlier*.

    3.  NAPSAC – sampling method @cite MyattNAPSAC which takes initial point
        uniformly at random and the rest of points for minimal sample in
        the neighborhood of initial point. This is method can be
        potentially useful when models are localized. For example, for
        plane fitting. However, in practise struggles from degenerate
        issues and defining optimal neighborhood size.

    4.  Progressive-NAPSAC – sampler @cite barath2019progressive which is similar to NAPSAC,
        although it starts from local and gradually converges to
        global sampling. This method can be quite useful if local models
        are expected but distribution of data can be arbitrary. The
        implemented version assumes data points to be sorted by quality
        as in PROSAC.

2.  Score Method. USAC as well as standard RANSAC finds model which
    minimizes total loss. Loss can be represented by following
    functions:

    1.  RANSAC – binary 0 / 1 loss. 1 for outlier, 0 for inlier. *Good
        option if the goal is to find as many inliers as possible.*

    2.  MSAC – truncated squared error distance of point to model. *The
        default option in framework*. The model might not have as many
        inliers as using RANSAC score, however will be more accurate.

    3.  MAGSAC – threshold-free method @cite BarathMAGSAC to compute score. Using,
        although, maximum sigma (standard deviation of noise) level to
        marginalize residual of point over sigma. Score of the point
        represents likelihood of point being inlier. *Recommended option
        when image noise is unknown since method does not require
        threshold*. However, it is still recommended to provide at least
        approximated threshold, because termination itself is based on
        number of points which error is less than threshold. By giving 0
        threshold the method will output model after maximum number of
        iterations reached.

    4.  LMeds – the least median of squared error distances. In the
        framework finding median is efficiently implement with $O(n)$
        complexity using quick-sort algorithm. Note, LMeds does not have
        to work properly when inlier ratio is less than 50%, in other
        cases this method is robust and does not require threshold.

3.  Error metric which describes error distance of point to
    estimated model.

    1.  Re-projection distance – used for affine, homography and
        projection matrices. For homography also symmetric re-projection
        distance can be used.

    2.  Sampson distance – used for Fundamental matrix.

    3.  Symmetric Geometric distance – used for Essential matrix.

4.  Degeneracy:

    1.  DEGENSAC – method @cite ChumDominant which for Fundamental matrix estimation
        efficiently verifies and recovers model which has at least 5
        points in minimal sample lying on the dominant plane.

    2.  Collinearity test – for affine and homography matrix estimation
        checks if no 3 points lying on the line. For homography matrix
        since points are planar is applied test which checks if points
        in minimal sample lie on the same side w.r.t. to any line
        crossing any two points in sample (does not assume reflection).

    3.  Oriented epipolar constraint – method @cite ChumEpipolar for epipolar
        geometry which verifies model (fundamental and essential matrix)
        to have points visible in the front of the camera.

5.  SPRT verification – method @cite Matas2005RandomizedRW which verifies model by its
    evaluation on randomly shuffled points using statistical properties
    given by probability of inlier, relative time for estimation,
    average number of output models etc. Significantly speeding up
    framework, because bad model can be rejected very quickly without
    explicitly computing error for every point.

6.  Local Optimization:

    1.  Locally Optimized RANSAC – method @cite ChumLORANSAC that iteratively
        improves so-far-the-best model by non-minimal estimation. *The
        default option in framework. This procedure is the fastest and
        not worse than others local optimization methods.*

    2.  Graph-Cut RANSAC – method @cite BarathGCRANSAC that refine so-far-the-best
        model, however, it exploits spatial coherence of the
        data points. *This procedure is quite precise however
        computationally slower.*

    3.  Sigma Consensus – method @cite BarathMAGSAC which improves model by applying
        non-minimal weighted estimation, where weights are computed with
        the same logic as in MAGSAC score. This method is better to use
        together with MAGSAC score.

7.  Termination:

    1.  Standard – standard equation for independent and
        uniform sampling.

    2.  PROSAC – termination for PROSAC.

    3.  SPRT – termination for SPRT.

8.  Solver. In the framework there are minimal and non-minimal solvers.
    In minimal solver standard methods for estimation is applied. In
    non-minimal solver usually the covariance matrix is built and the
    model is found as the eigen vector corresponding to the highest
    eigen value.

    1.  Affine2D matrix

    2.  Homography matrix – for minimal solver is used RHO
        (Gaussian elimination) algorithm from OpenCV.

    3.  Fundamental matrix – for 7-points algorithm two null vectors are
        found using Gaussian elimination (eliminating to upper
        triangular matrix and back-substitution) instead of SVD and then
        solving 3-degrees polynomial. For 8-points solver Gaussian
        elimination is used too.

    4.  Essential matrix – 4 null vectors are found using
        Gaussian elimination. Then the solver based on Gröbner basis
        described in @cite SteweniusRecent is used. Essential matrix can be computed
        only if <span style="font-variant:small-caps;">LAPACK</span> or
        <span style="font-variant:small-caps;">Eigen</span> are
        installed as it requires eigen decomposition with complex
        eigen values.

    5.  Perspective-n-Point – the minimal solver is classical 3 points
        with up to 4 solutions. For RANSAC the low number of sample size
        plays significant role as it requires less iterations,
        furthermore in average P3P solver has around 1.39
        estimated models. Also, in new version of `solvePnPRansac(...)`
        with `UsacParams` there is an options to pass empty intrinsic
        matrix `InputOutputArray cameraMatrix`. If matrix is empty than
        using Direct Linear Transformation algorithm (PnP with 6 points)
        framework outputs not only rotation and translation vector but
        also calibration matrix.

Also, the framework can be run in parallel. The parallelization is done
in the way that multiple RANSACs are created and they share two atomic
variables `bool success` and `int num_hypothesis_tested` which
determines when all RANSACs must terminate. If one of RANSAC terminated
successfully then all other RANSAC will terminate as well. In the end
the best model is synchronized from all threads. If PROSAC sampler is
used then threads must share the same sampler since sampling is done
sequentially. However, using default options of framework parallel
RANSAC is not deterministic since it depends on how often each thread is
running. The easiest way to make it deterministic is using PROSAC
sampler without SPRT and Local Optimization and not for Fundamental
matrix, because they internally use random generators.

For NAPSAC, Progressive NAPSAC or Graph-Cut methods is required to build
a neighborhood graph. In framework there are 3 options to do it:

1.  NEIGH_FLANN_KNN – estimate neighborhood graph using OpenCV FLANN
    K nearest-neighbors. The default value for KNN is 7. KNN method may
    work good for sampling but not good for GC-RANSAC.

2.  `NEIGH_FLANN_RADIUS` – similarly as in previous case finds neighbor
    points which distance is less than 20 pixels.

3.  `NEIGH_GRID` – for finding points’ neighborhood tiles points in
    cells using hash-table. The method is described in @cite barath2019progressive. Less
    accurate than `NEIGH_FLANN_RADIUS`, although significantly faster.

Note, `NEIGH_FLANN_RADIUS` and `NEIGH_FLANN_RADIUS` are not able to PnP
solver, since there are 3D object points.

New flags:
------
1.  `USAC_DEFAULT` – has standard LO-RANSAC.

2.  `USAC_PARALLEL` – has LO-RANSAC and RANSACs run in parallel.

3.  `USAC_ACCURATE` – has GC-RANSAC.

4.  `USAC_FAST` – has LO-RANSAC with smaller number iterations in local
    optimization step. Uses RANSAC score to maximize number of inliers
    and terminate earlier.

5.  `USAC_PROSAC` – has PROSAC sampling. Note, points must be sorted.

6.  `USAC_FM_8PTS` – has LO-RANSAC. Only valid for Fundamental matrix
    with 8-points solver.

7.  `USAC_MAGSAC` – has MAGSAC++.

Every flag uses SPRT verification. And in the end the final
so-far-the-best model is polished by non minimal estimation of all found
inliers.

A few other important parameters:
------

1.  `randomGeneratorState` – since every USAC solver is deterministic in
    OpenCV (i.e., for the same points and parameters returns the
    same result) by providing new state it will output new model.

2.  `loIterations` – number of iterations for Local Optimization method.
    *The default value is 10*. By increasing `loIterations` the output
    model could be more accurate, however, the computationial time may
    also increase.

3.  `loSampleSize` – maximum sample number for Local Optimization. *The
    default value is 14*. Note, that by increasing `loSampleSize` the
    accuracy of model can increase as well as the computational time.
    However, it is recommended to keep value less than 100, because
    estimation on low number of points is faster and more robust.

Samples:
------

There are three new sample files in opencv/samples directory.

1.  `epipolar_lines.cpp` – input arguments of `main` function are two
    paths to images. Then correspondences are found using
    SIFT detector. Fundamental matrix is found using RANSAC from
    tentative correspondences and epipolar lines are plot.

2.  `essential_mat_reconstr.cpp` – input arguments are path to data file
    containing image names and single intrinsic matrix and directory
    where these images located. Correspondences are found using SIFT.
    The essential matrix is estimated using RANSAC and decomposed to
    rotation and translation. Then by building two relative poses with
    projection matrices image points are triangulated to object points.
    By running RANSAC with 3D plane fitting object points as well as
    correspondences are clustered into planes.

3.  `essential_mat_reconstr.py` – the same functionality as in .cpp
    file, however instead of clustering points to plane the 3D map of
    object points is plot.
