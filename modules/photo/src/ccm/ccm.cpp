// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "opencv2/photo.hpp"
#include "linearize.hpp"
#include <cmath>
namespace cv {
namespace ccm {

class ColorCorrectionModel::Impl
{
public:
    Mat src;

    Color ref = Color();
    Mat dist;
    RGBBase_& cs;
    // Track initialization parameters for serialization
    ColorSpace csEnum;
    Mat mask;

    // RGBl of detected data and the reference
    Mat srcRgbl;
    Mat dstRgbl;

    // ccm type and shape
    CcmType ccmType;
    int shape;

    // linear method and distance
    std::shared_ptr<Linear> linear = std::make_shared<Linear>();
    DistanceType distance;
    LinearizationType linearizationType;

    Mat weights;
    Mat weightsList;
    Mat ccm;
    Mat ccm0;
    double gamma;
    int deg;
    std::vector<double> saturatedThreshold;
    InitialMethodType initialMethodType;
    double weightsCoeff;
    int maskedLen;
    double loss;
    int maxCount;
    double epsilon;
    bool rgb;
    Impl();

    /** @brief Make no change for CCM_LINEAR.
             convert cv::Mat A to [A, 1] in CCM_AFFINE.
        @param inp the input array, type of cv::Mat.
        @return the output array, type of cv::Mat
    */
    Mat prepare(const Mat& inp);

    /** @brief Calculate weights and mask.
        @param weightsList the input array, type of cv::Mat.
        @param weightsCoeff type of double.
        @param saturateMask the input array, type of cv::Mat.
    */
    void calWeightsMasks(const Mat& weightsList, double weightsCoeff, Mat saturateMask);

    /** @brief Fitting nonlinear - optimization initial value by white balance.
        @return the output array, type of Mat
    */
    void initialWhiteBalance(void);

    /** @brief Fitting nonlinear-optimization initial value by least square.
        @param fit if fit is True, return optimalization for rgbl distance function.
    */
    void initialLeastSquare(bool fit = false);

    double calcLoss_(Color color);
    double calcLoss(const Mat ccm_);

    /** @brief Fitting ccm if distance function is associated with CIE Lab color space.
             see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
            Set terminal criteria for solver is possible.
    */
    void fitting(void);

    void getColor(Mat& img_, bool islinear = false);
    void getColor(ColorCheckerType constColor);
    void getColor(Mat colors_, ColorSpace cs_, Mat colored_);
    void getColor(Mat colors_, ColorSpace refColorSpace_);

    /** @brief Loss function base on cv::MinProblemSolver::Function.
             see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
    */
    class LossFunction : public MinProblemSolver::Function
    {
    public:
        ColorCorrectionModel::Impl* ccmLoss;
        LossFunction(ColorCorrectionModel::Impl* ccm)
            : ccmLoss(ccm) {};

        /** @brief Reset dims to ccm->shape.
        */
        int getDims() const CV_OVERRIDE
        {
            return ccmLoss->shape;
        }

        /** @brief Reset calculation.
        */
        double calc(const double* x) const CV_OVERRIDE
        {
            Mat ccm_(ccmLoss->shape, 1, CV_64F);
            for (int i = 0; i < ccmLoss->shape; i++)
            {
                ccm_.at<double>(i, 0) = x[i];
            }
            ccm_ = ccm_.reshape(0, ccmLoss->shape / 3);
            return ccmLoss->calcLoss(ccm_);
        }
    };
};

ColorCorrectionModel::Impl::Impl()
    : cs(*GetCS::getInstance().getRgb(COLOR_SPACE_SRGB))
    , csEnum(COLOR_SPACE_SRGB)
    , ccmType(CCM_LINEAR)
    , distance(DISTANCE_CIE2000)
    , linearizationType(LINEARIZATION_GAMMA)
    , weights(Mat())
    , gamma(2.2)
    , deg(3)
    , saturatedThreshold({ 0, 0.98 })
    , initialMethodType(INITIAL_METHOD_LEAST_SQUARE)
    , weightsCoeff(0)
    , maxCount(5000)
    , epsilon(1.e-4)
    , rgb(true)
{}

Mat ColorCorrectionModel::Impl::prepare(const Mat& inp)
{
    switch (ccmType)
    {
    case cv::ccm::CCM_LINEAR:
        shape = 9;
        return inp;
    case cv::ccm::CCM_AFFINE:
    {
        shape = 12;
        Mat ones(inp.size(), CV_64F, Scalar(1));
        Mat out(inp.size(), CV_64FC4);
        const Mat srcs[] = { inp, ones };
        const int fromTo[] = { 0,0, 1,1, 2,2, 3,3 };         // inp[ch] → out[ch]
        mixChannels(srcs, 2, &out, 1, fromTo, 4);
        return out;
    }
    default:
        CV_Error(Error::StsBadArg, "Wrong ccmType!");
        break;
    }
}

void ColorCorrectionModel::Impl::calWeightsMasks(const Mat& weightsList_, double weightsCoeff_, Mat saturateMask)
{
    // weights
    if (!weightsList_.empty())
    {
        weights = weightsList_;
    }
    else if (weightsCoeff_ != 0)
    {
        pow(ref.toLuminant(cs.illumobserver), weightsCoeff_, weights);
    }

    // masks
    Mat weight_mask = Mat::ones(src.rows, 1, CV_8U);
    if (!weights.empty())
    {
        weight_mask = weights > 0;
    }
    this->mask = (weight_mask) & (saturateMask);

    // weights' mask
    if (!weights.empty())
    {
        Mat weights_masked = maskCopyTo(this->weights, this->mask);
        weights = weights_masked / mean(weights_masked)[0];
    }
    maskedLen = (int)sum(mask)[0];
}

void ColorCorrectionModel::Impl::initialWhiteBalance()
{
    // sum over all pixels – Scalar holds per-channel sums
    const cv::Scalar srcSum = cv::sum(srcRgbl);
    const cv::Scalar dstSum = cv::sum(dstRgbl);

    // channel-wise gain factors
    const double gR = dstSum[0] / srcSum[0];
    const double gG = dstSum[1] / srcSum[1];
    const double gB = dstSum[2] / srcSum[2];

    // shape == 9 for a 3×3 linear CCM, or 12 for a 3×4 affine CCM
    if (shape == 9) {
        // 3×3 diagonal matrix
        ccm0 = cv::Mat::zeros(3, 3, CV_64F);
        ccm0.at<double>(0, 0) = gR;
        ccm0.at<double>(1, 1) = gG;
        ccm0.at<double>(2, 2) = gB;
    }
    else {
        // 3×4 affine matrix (last column = zeros)
        ccm0 = cv::Mat::zeros(3, 4, CV_64F);
        ccm0.at<double>(0, 0) = gR;
        ccm0.at<double>(1, 1) = gG;
        ccm0.at<double>(2, 2) = gB;
    }
}


void ColorCorrectionModel::Impl::initialLeastSquare(bool fit)
{
    Mat A, B, w;
    if (weights.empty())
    {
        A = srcRgbl;
        B = dstRgbl;
    }
    else
    {
        pow(weights, 0.5, w);
        Mat w_;
        merge(std::vector<Mat> { w, w, w }, w_);
        A = w_.mul(srcRgbl);
        B = w_.mul(dstRgbl);
    }
    solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);

    // if fit is True, return optimalization for rgbl distance function.
    if (fit)
    {
        ccm = ccm0;
        Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
        Scalar s = residual.dot(residual);
        double sum = s[0];
        loss = sqrt(sum / maskedLen);
    }
}

double ColorCorrectionModel::Impl::calcLoss_(Color color)
{
    Mat distlist = color.diff(ref, distance);
    Color lab = color.to(COLOR_SPACE_LAB_D50_2);
    Mat dist_;
    pow(distlist, 2, dist_);
    if (!weights.empty())
    {
        dist_ = weights.mul(dist_);
    }
    Scalar ss = sum(dist_);
    return ss[0];
}

double ColorCorrectionModel::Impl::calcLoss(const Mat ccm_)
{
    Mat converted = srcRgbl.reshape(1, 0) * ccm_;
    Color color(converted.reshape(3, 0), *(cs.l));
    return calcLoss_(color);
}

void ColorCorrectionModel::Impl::fitting(void)
{
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
    solver->setFunction(ptr_F);
    Mat reshapeCcm = ccm0.clone().reshape(0, 1);
    Mat step = Mat::ones(reshapeCcm.size(), CV_64F);
    solver->setInitStep(step);
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxCount, epsilon);
    solver->setTermCriteria(termcrit);
    double res = solver->minimize(reshapeCcm);
    ccm = reshapeCcm.reshape(0, shape / 3);
    loss = sqrt(res / maskedLen);
}

ColorCorrectionModel::ColorCorrectionModel()
: p(std::make_shared<Impl>())
{}

void ColorCorrectionModel::correctImage(InputArray src, OutputArray ref, bool islinear)
{
    if (!p->ccm.data)
    {
        CV_Error(Error::StsBadArg, "No CCM values!" );
    }
    Mat img, normImg;
    if (p->rgb){
        cvtColor(src.getMat(), img, COLOR_BGR2RGB);
    } else {
        img = src.getMat();
    }

    double scale;
    int type = img.type();
    switch (type) {
        case CV_8UC3:
            scale = 1.0 / 255.0;
            break;
        case CV_16UC3:
            scale = 1.0 / 65535.0;
            break;
        case CV_32FC3:
            scale = 1.0;  // Already in [0,1] range
            break;
        default:
            CV_Error( cv::Error::StsUnsupportedFormat, "8-bit, 16-bit unsigned or 32-bit float 3-channel input images are supported");
    }

    img.convertTo(normImg, CV_64F, scale);
    Mat linearImg = (p->linear)->linearize(normImg);
    Mat ccm = p->ccm.reshape(0, p->shape / 3);
    Mat imgCcm = multiple(p->prepare(linearImg), ccm);
    if (islinear == true)
    {
        imgCcm.copyTo(ref);
    }
    Mat imgCorrected = p->cs.fromLFunc(imgCcm, linearImg);

    imgCorrected *= 1.0/scale;
    imgCorrected.convertTo(imgCorrected, type);

    if (p->rgb)
        cvtColor(imgCorrected, imgCorrected, COLOR_RGB2BGR);
    imgCorrected.copyTo(ref);
}

void ColorCorrectionModel::Impl::getColor(ColorCheckerType constColor)
{
    ref = GetColor().getColor(constColor);
}

void ColorCorrectionModel::Impl::getColor(Mat colors_, ColorSpace refColorSpace_)
{
    ref = Color(colors_, *GetCS::getInstance().getCS(refColorSpace_));
}

void ColorCorrectionModel::Impl::getColor(Mat colors_, ColorSpace cs_, Mat colored_)
{
    ref = Color(colors_, *GetCS::getInstance().getCS(cs_), colored_);
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, int constColor): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(static_cast<ColorCheckerType>(constColor));
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, InputArray colors_, ColorSpace refColorSpace_): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(colors_.getMat(), refColorSpace_);
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, InputArray colors_, ColorSpace cs_, InputArray coloredPatchesMask_): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(colors_.getMat(), cs_, coloredPatchesMask_.getMat());
}

void ColorCorrectionModel::setColorSpace(ColorSpace cs_)
{
    p->cs = *GetCS::getInstance().getRgb(cs_);
}
void ColorCorrectionModel::setCcmType(CcmType ccmType_)
{
    p->ccmType = ccmType_;
}
void ColorCorrectionModel::setDistance(DistanceType distance_)
{
    p->distance = distance_;
}
void ColorCorrectionModel::setLinearization(LinearizationType linearizationType)
{
    p->linearizationType = linearizationType;
}
void ColorCorrectionModel::setLinearizationGamma(double gamma)
{
    p->gamma = gamma;
}
void ColorCorrectionModel::setLinearizationDegree(int deg)
{
    p->deg = deg;
}
void ColorCorrectionModel::setSaturatedThreshold(double lower, double upper)
{  //std::vector<double> saturatedThreshold
    p->saturatedThreshold = { lower, upper };
}
void ColorCorrectionModel::setWeightsList(const Mat& weightsList)
{
    p->weightsList = weightsList;
}
void ColorCorrectionModel::setWeightCoeff(double weightsCoeff)
{
    p->weightsCoeff = weightsCoeff;
}
void ColorCorrectionModel::setInitialMethod(InitialMethodType initialMethodType)
{
    p->initialMethodType = initialMethodType;
}
void ColorCorrectionModel::setMaxCount(int maxCount_)
{
    p->maxCount = maxCount_;
}
void ColorCorrectionModel::setEpsilon(double epsilon_)
{
    p->epsilon = epsilon_;
}
void ColorCorrectionModel::setRGB(bool rgb_)
{
    p->rgb = rgb_;
}
Mat ColorCorrectionModel::compute()
{

    Mat saturateMask = saturate(p->src, p->saturatedThreshold[0], p->saturatedThreshold[1]);
    p->linear = getLinear(p->gamma, p->deg, p->src, p->ref, saturateMask, (p->cs), p->linearizationType);
    p->calWeightsMasks(p->weightsList, p->weightsCoeff, saturateMask);
    p->srcRgbl = p->linear->linearize(maskCopyTo(p->src, p->mask));
    p->ref.colors = maskCopyTo(p->ref.colors, p->mask);
    p->dstRgbl = p->ref.to(*(p->cs.l)).colors;

    // make no change for CCM_LINEAR, make change for CCM_AFFINE.
    p->srcRgbl = p->prepare(p->srcRgbl);

    // distance function may affect the loss function and the fitting function
    switch (p->distance)
    {
    case cv::ccm::DISTANCE_RGBL:
        p->initialLeastSquare(true);
        break;
    default:
        switch (p->initialMethodType)
        {
        case cv::ccm::INITIAL_METHOD_WHITE_BALANCE:
            p->initialWhiteBalance();
            break;
        case cv::ccm::INITIAL_METHOD_LEAST_SQUARE:
            p->initialLeastSquare();
            break;
        default:
            CV_Error(Error::StsBadArg, "Wrong initial_methoddistance_type!" );
            break;
        }
        break;
    }
    p->fitting();

    return p->ccm;
}
Mat ColorCorrectionModel::getColorCorrectionMatrix() const
{
    return p->ccm;
}
double ColorCorrectionModel::getLoss() const
{
    return p->loss;
}
Mat ColorCorrectionModel::getSrcLinearRGB() const{
    return p->srcRgbl;
}
Mat ColorCorrectionModel::getRefLinearRGB() const{
    return p->dstRgbl;
}
Mat ColorCorrectionModel::getMask() const{
    return p->mask;
}
Mat ColorCorrectionModel::getWeights() const{
    return p->weights;
}

void ColorCorrectionModel::write(FileStorage& fs) const
{
    fs << "ColorCorrectionModel" << "{"
       << "ccm" << p->ccm
       << "loss" << p->loss
       << "csEnum" << p->csEnum
       << "ccm_type" << p->ccmType
       << "shape" << p->shape
       << "linear" << *p->linear
       << "distance" << p->distance
       << "linear_type" << p->linearizationType
       << "gamma" << p->gamma
       << "deg" << p->deg
       << "saturated_threshold" << p->saturatedThreshold
       << "}";
}

void ColorCorrectionModel::read(const FileNode& node)
{
    node["ccm"] >> p->ccm;
    node["loss"] >> p->loss;
    node["ccm_type"] >> p->ccmType;
    node["shape"] >> p->shape;
    node["distance"] >> p->distance;
    node["gamma"] >> p->gamma;
    node["deg"] >> p->deg;
    node["saturated_threshold"] >> p->saturatedThreshold;

    ColorSpace csEnum;
    node["csEnum"] >> csEnum;
    setColorSpace(csEnum);

    node["linear_type"] >> p->linearizationType;
    switch (p->linearizationType) {
        case cv::ccm::LINEARIZATION_GAMMA:
            p->linear = std::shared_ptr<Linear>(new LinearGamma());
            break;
        case cv::ccm::LINEARIZATION_COLORPOLYFIT:
            p->linear = std::shared_ptr<Linear>(new LinearColor<Polyfit>());
            break;
        case cv::ccm::LINEARIZATION_IDENTITY:
            p->linear = std::shared_ptr<Linear>(new LinearIdentity());
            break;
        case cv::ccm::LINEARIZATION_COLORLOGPOLYFIT:
            p->linear = std::shared_ptr<Linear>(new LinearColor<LogPolyfit>());
            break;
        case cv::ccm::LINEARIZATION_GRAYPOLYFIT:
            p->linear = std::shared_ptr<Linear>(new LinearGray<Polyfit>());
            break;
        case cv::ccm::LINEARIZATION_GRAYLOGPOLYFIT:
            p->linear = std::shared_ptr<Linear>(new LinearGray<LogPolyfit>());
            break;
        default:
            CV_Error(Error::StsBadArg, "Wrong linear_type!");
            break;
    }
    node["linear"] >> *p->linear;
}

void write(FileStorage& fs, const std::string&, const cv::ccm::ColorCorrectionModel& ccm)
{
    ccm.write(fs);
}

void read(const cv::FileNode& node, cv::ccm::ColorCorrectionModel& ccm, const cv::ccm::ColorCorrectionModel& defaultValue)
{
    if (node.empty())
        ccm = defaultValue;
    else
        ccm.read(node);
}

}
}  // namespace cv::ccm
