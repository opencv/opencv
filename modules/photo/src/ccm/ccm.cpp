// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "opencv2/photo.hpp"
#include "linearize.hpp"
namespace cv {
namespace ccm {
class ColorCorrectionModel::Impl
{
public:
    Mat src;
    std::shared_ptr<Color> dst = std::make_shared<Color>();
    Mat dist;
    RGBBase_& cs;
    Mat mask;

    // RGBl of detected data and the reference
    Mat src_rgbl;
    Mat dst_rgbl;

    // ccm type and shape
    CCMType ccmType;
    int shape;

    // linear method and distance
    std::shared_ptr<Linear> linear = std::make_shared<Linear>();
    DistanceType distance;
    LinearType linearType;

    Mat weights;
    Mat weights_list;
    Mat ccm;
    Mat ccm0;
    double gamma;
    int deg;
    std::vector<double> saturatedThreshold;
    InitialMethodType initialMethodType;
    double weights_coeff;
    int masked_len;
    double loss;
    int max_count;
    double epsilon;
    Impl();

    /** @brief Make no change for CCM_LINEAR.
             convert cv::Mat A to [A, 1] in CCM_AFFINE.
        @param inp the input array, type of cv::Mat.
        @return the output array, type of cv::Mat
    */
    Mat prepare(const Mat& inp);

    /** @brief Calculate weights and mask.
        @param weights_list the input array, type of cv::Mat.
        @param weights_coeff type of double.
        @param saturate_mask the input array, type of cv::Mat.
    */
    void calWeightsMasks(const Mat& weights_list, double weights_coeff, Mat saturate_mask);

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
    void getColor(ColorCheckerType constcolor);
    void getColor(Mat colors_, ColorSpace cs_, Mat colored_);
    void getColor(Mat colors_, ColorSpace ref_cs_);

    /** @brief Loss function base on cv::MinProblemSolver::Function.
             see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
    */
    class LossFunction : public MinProblemSolver::Function
    {
    public:
        ColorCorrectionModel::Impl* ccm_loss;
        LossFunction(ColorCorrectionModel::Impl* ccm)
            : ccm_loss(ccm) {};

        /** @brief Reset dims to ccm->shape.
        */
        int getDims() const CV_OVERRIDE
        {
            return ccm_loss->shape;
        }

        /** @brief Reset calculation.
        */
        double calc(const double* x) const CV_OVERRIDE
        {
            Mat ccm_(ccm_loss->shape, 1, CV_64F);
            for (int i = 0; i < ccm_loss->shape; i++)
            {
                ccm_.at<double>(i, 0) = x[i];
            }
            ccm_ = ccm_.reshape(0, ccm_loss->shape / 3);
            return ccm_loss->calcLoss(ccm_);
        }
    };
};

ColorCorrectionModel::Impl::Impl()
    : cs(*GetCS::getInstance().getRgb(COLOR_SPACE_SRGB))
    , ccmType(CCM_LINEAR)
    , distance(DISTANCE_CIE2000)
    , linearType(LINEARIZATION_GAMMA)
    , weights(Mat())
    , gamma(2.2)
    , deg(3)
    , saturatedThreshold({ 0, 0.98 })
    , initialMethodType(INITIAL_METHOD_LEAST_SQUARE)
    , weights_coeff(0)
    , max_count(5000)
    , epsilon(1.e-4)
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
        Mat arr1 = Mat::ones(inp.size(), CV_64F);
        Mat arr_out(inp.size(), CV_64FC4);
        Mat arr_channels[3];
        split(inp, arr_channels);
        merge(std::vector<Mat> { arr_channels[0], arr_channels[1], arr_channels[2], arr1 }, arr_out);
        return arr_out;
    }
    default:
        CV_Error(Error::StsBadArg, "Wrong ccmType!");
        break;
    }
}

void ColorCorrectionModel::Impl::calWeightsMasks(const Mat& weights_list_, double weights_coeff_, Mat saturate_mask)
{
    // weights
    if (!weights_list_.empty())
    {
        weights = weights_list_;
    }
    else if (weights_coeff_ != 0)
    {
        pow(dst->toLuminant(cs.io), weights_coeff_, weights);
    }

    // masks
    Mat weight_mask = Mat::ones(src.rows, 1, CV_8U);
    if (!weights.empty())
    {
        weight_mask = weights > 0;
    }
    this->mask = (weight_mask) & (saturate_mask);

    // weights' mask
    if (!weights.empty())
    {
        Mat weights_masked = maskCopyTo(this->weights, this->mask);
        weights = weights_masked / mean(weights_masked)[0];
    }
    masked_len = (int)sum(mask)[0];
}

void ColorCorrectionModel::Impl::initialWhiteBalance(void)
{
    Mat schannels[4];
    split(src_rgbl, schannels);
    Mat dchannels[4];
    split(dst_rgbl, dchannels);
    std::vector<double> initial_vec = { sum(dchannels[0])[0] / sum(schannels[0])[0], 0, 0, 0,
        sum(dchannels[1])[0] / sum(schannels[1])[0], 0, 0, 0,
        sum(dchannels[2])[0] / sum(schannels[2])[0], 0, 0, 0 };
    std::vector<double> initial_vec_(initial_vec.begin(), initial_vec.begin() + shape);
    Mat initial_white_balance = Mat(initial_vec_, true).reshape(0, shape / 3);
    ccm0 = initial_white_balance;
}

void ColorCorrectionModel::Impl::initialLeastSquare(bool fit)
{
    Mat A, B, w;
    if (weights.empty())
    {
        A = src_rgbl;
        B = dst_rgbl;
    }
    else
    {
        pow(weights, 0.5, w);
        Mat w_;
        merge(std::vector<Mat> { w, w, w }, w_);
        A = w_.mul(src_rgbl);
        B = w_.mul(dst_rgbl);
    }
    solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);

    // if fit is True, return optimalization for rgbl distance function.
    if (fit)
    {
        ccm = ccm0;
        Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
        Scalar s = residual.dot(residual);
        double sum = s[0];
        loss = sqrt(sum / masked_len);
    }
}

double ColorCorrectionModel::Impl::calcLoss_(Color color)
{
    Mat distlist = color.diff(*dst, distance);
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
    Mat converted = src_rgbl.reshape(1, 0) * ccm_;
    Color color(converted.reshape(3, 0), *(cs.l));
    return calcLoss_(color);
}

void ColorCorrectionModel::Impl::fitting(void)
{
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
    solver->setFunction(ptr_F);
    Mat reshapeccm = ccm0.clone().reshape(0, 1);
    Mat step = Mat::ones(reshapeccm.size(), CV_64F);
    solver->setInitStep(step);
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_count, epsilon);
    solver->setTermCriteria(termcrit);
    double res = solver->minimize(reshapeccm);
    ccm = reshapeccm.reshape(0, shape / 3);
    loss = pow((res / masked_len), 0.5);
}

Mat ColorCorrectionModel::infer(const Mat& img, bool islinear)
{
    if (!p->ccm.data)
    {
        CV_Error(Error::StsBadArg, "No CCM values!" );
    }
    Mat img_lin = (p->linear)->linearize(img);
    Mat ccm = p->ccm.reshape(0, p->shape / 3);
    Mat img_ccm = multiple(p->prepare(img_lin), ccm);
    if (islinear == true)
    {
        return img_ccm;
    }
    return p->cs.fromLFunc(img_ccm, img_lin);
}

void ColorCorrectionModel::Impl::getColor(ColorCheckerType constcolor)
{
    dst = (GetColor::getColor(constcolor));
}

void ColorCorrectionModel::Impl::getColor(Mat colors_, ColorSpace ref_cs_)
{
    dst.reset(new Color(colors_, *GetCS::getInstance().getCS(ref_cs_)));
}

void ColorCorrectionModel::Impl::getColor(Mat colors_, ColorSpace cs_, Mat colored_)
{
    dst.reset(new Color(colors_, *GetCS::getInstance().getCS(cs_), colored_));
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, int constcolor): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(static_cast<ColorCheckerType>(constcolor));
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, InputArray colors_, ColorSpace ref_cs_): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(colors_.getMat(), ref_cs_);
}

ColorCorrectionModel::ColorCorrectionModel(InputArray src_, InputArray colors_, ColorSpace cs_, InputArray colored_): p(std::make_shared<Impl>())
{
    p->src = src_.getMat();
    p->getColor(colors_.getMat(), cs_, colored_.getMat());
}

void ColorCorrectionModel::setColorSpace(ColorSpace cs_)
{
    p->cs = *GetCS::getInstance().getRgb(cs_);
}
void ColorCorrectionModel::setCCMType(CCMType ccmType_)
{
    p->ccmType = ccmType_;
}
void ColorCorrectionModel::setDistance(DistanceType distance_)
{
    p->distance = distance_;
}
void ColorCorrectionModel::setLinear(LinearType linearType)
{
    p->linearType = linearType;
}
void ColorCorrectionModel::setLinearGamma(double gamma)
{
    p->gamma = gamma;
}
void ColorCorrectionModel::setLinearDegree(int deg)
{
    p->deg = deg;
}
void ColorCorrectionModel::setSaturatedThreshold(double lower, double upper)
{  //std::vector<double> saturatedThreshold
    p->saturatedThreshold = { lower, upper };
}
void ColorCorrectionModel::setWeightsList(const Mat& weights_list)
{
    p->weights_list = weights_list;
}
void ColorCorrectionModel::setWeightCoeff(double weights_coeff)
{
    p->weights_coeff = weights_coeff;
}
void ColorCorrectionModel::setInitialMethod(InitialMethodType initialMethodType)
{
    p->initialMethodType = initialMethodType;
}
void ColorCorrectionModel::setMaxCount(int max_count_)
{
    p->max_count = max_count_;
}
void ColorCorrectionModel::setEpsilon(double epsilon_)
{
    p->epsilon = epsilon_;
}
void ColorCorrectionModel::computeCCM()
{

    Mat saturate_mask = saturate(p->src, p->saturatedThreshold[0], p->saturatedThreshold[1]);
    p->linear = getLinear(p->gamma, p->deg, p->src, *(p->dst), saturate_mask, (p->cs), p->linearType);
    p->calWeightsMasks(p->weights_list, p->weights_coeff, saturate_mask);
    p->src_rgbl = p->linear->linearize(maskCopyTo(p->src, p->mask));
    p->dst->colors = maskCopyTo(p->dst->colors, p->mask);
    p->dst_rgbl = p->dst->to(*(p->cs.l)).colors;

    // make no change for CCM_LINEAR, make change for CCM_AFFINE.
    p->src_rgbl = p->prepare(p->src_rgbl);

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
}
Mat ColorCorrectionModel::getCCM() const
{
    return p->ccm;
}
double ColorCorrectionModel::getLoss() const
{
    return p->loss;
}
Mat ColorCorrectionModel::getSrcRgbl() const{
    return p->src_rgbl;
}
Mat ColorCorrectionModel::getDstRgbl() const{
    return p->dst_rgbl;
}
Mat ColorCorrectionModel::getMask() const{
    return p->mask;
}
Mat ColorCorrectionModel::getWeights() const{
    return p->weights;
}
}
}  // namespace cv::ccm
