///////////// see LICENSE.txt in the OpenCV root directory //////////////

#ifndef __OPENCV_NONFREE_SURF_HPP__
#define __OPENCV_NONFREE_SURF_HPP__

namespace cv
{
//! Speeded up robust features, port from CUDA module.
////////////////////////////////// SURF //////////////////////////////////////////

class SURF_OCL
{
public:
    enum KeypointLayout
    {
        X_ROW = 0,
        Y_ROW,
        LAPLACIAN_ROW,
        OCTAVE_ROW,
        SIZE_ROW,
        ANGLE_ROW,
        HESSIAN_ROW,
        ROWS_COUNT
    };

    //! the full constructor taking all the necessary parameters
    SURF_OCL();

    bool init(const SURF* params);

    //! returns the descriptor size in float's (64 or 128)
    int descriptorSize() const { return params->extended ? 128 : 64; }

    void uploadKeypoints(const std::vector<KeyPoint> &keypoints, UMat &keypointsGPU);
    void downloadKeypoints(const UMat &keypointsGPU, std::vector<KeyPoint> &keypoints);

    //! finds the keypoints using fast hessian detector used in SURF
    //! supports CV_8UC1 images
    //! keypoints will have nFeature cols and 6 rows
    //! keypoints.ptr<float>(X_ROW)[i] will contain x coordinate of i'th feature
    //! keypoints.ptr<float>(Y_ROW)[i] will contain y coordinate of i'th feature
    //! keypoints.ptr<float>(LAPLACIAN_ROW)[i] will contain laplacian sign of i'th feature
    //! keypoints.ptr<float>(OCTAVE_ROW)[i] will contain octave of i'th feature
    //! keypoints.ptr<float>(SIZE_ROW)[i] will contain size of i'th feature
    //! keypoints.ptr<float>(ANGLE_ROW)[i] will contain orientation of i'th feature
    //! keypoints.ptr<float>(HESSIAN_ROW)[i] will contain response of i'th feature
    bool detect(InputArray img, InputArray mask, UMat& keypoints);
    //! finds the keypoints and computes their descriptors.
    //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
    bool detectAndCompute(InputArray img, InputArray mask, UMat& keypoints,
                          OutputArray descriptors, bool useProvidedKeypoints = false);

protected:
    bool setImage(InputArray img, InputArray mask);

    // kernel callers declarations
    bool calcLayerDetAndTrace(int octave, int layer_rows);

    bool findMaximaInLayer(int counterOffset, int octave, int layer_rows, int layer_cols);

    bool interpolateKeypoint(int maxCounter, UMat &keypoints, int octave, int layer_rows, int maxFeatures);

    bool calcOrientation(UMat &keypoints);

    bool setUpRight(UMat &keypoints);

    bool computeDescriptors(const UMat &keypoints, OutputArray descriptors);

    bool detectKeypoints(UMat &keypoints);

    const SURF* params;

    //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
    UMat sum, intBuffer;
    UMat det, trace;
    UMat maxPosBuffer;

    int img_cols, img_rows;

    int maxCandidates;
    int maxFeatures;

    UMat img, counters;

    // texture buffers
    ocl::Image2D imgTex, sumTex;
    bool haveImageSupport;
    String kerOpts;

    int status;
};

/*
template<typename _Tp> void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
    if(v.empty())
        um.release();
    else
        Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

template<typename _Tp> void copyUMatToVector(const UMat& um, std::vector<_Tp>& v)
{
    if(um.empty())
        v.clear();
    else
    {
        size_t sz = um.total()*um.elemSize();
        CV_Assert(um.isContinuous() && (sz % sizeof(_Tp) == 0));
        v.resize(sz/sizeof(_Tp));
        Mat m(um.size(), um.type(), &v[0]);
        um.copyTo(m);
    }
}*/

}

#endif
