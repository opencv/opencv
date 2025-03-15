// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                       License Agreement
//              For Open Source Computer Vision Library
//
// Copyright(C) 2020, Huawei Technologies Co.,Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//             http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "opencv2/ccm.hpp"
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
    CCM_TYPE ccm_type;
    int shape;

    // linear method and distance
    std::shared_ptr<Linear> linear = std::make_shared<Linear>();
    DISTANCE_TYPE distance;
    LINEAR_TYPE linear_type;

    Mat weights;
    Mat weights_list;
    Mat ccm;
    Mat ccm0;
    double gamma;
    int deg;
    std::vector<double> saturated_threshold;
    INITIAL_METHOD_TYPE initial_method_type;
    double weights_coeff;
    int masked_len;
    double loss;
    int max_count;
    double epsilon;
    Impl();

    /** @brief Make no change for CCM_3x3.
             convert cv::Mat A to [A, 1] in CCM_4x3.
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

    double calc_loss_(Color color);
    double calc_loss(const Mat ccm_);

    /** @brief Fitting ccm if distance function is associated with CIE Lab color space.
             see details in https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/optim.hpp
            Set terminal criteria for solver is possible.
    */
    void fitting(void);

    void getColor(Mat& img_, bool islinear = false);
    void getColor(CONST_COLOR constcolor);
    void getColor(Mat colors_, COLOR_SPACE cs_, Mat colored_);
    void getColor(Mat colors_, COLOR_SPACE ref_cs_);

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
            return ccm_loss->calc_loss(ccm_);
        }
    };
};

ColorCorrectionModel::Impl::Impl()
    : cs(*GetCS::getInstance().get_rgb(COLOR_SPACE_sRGB))
    , ccm_type(CCM_3x3)
    , distance(DISTANCE_CIE2000)
    , linear_type(LINEARIZATION_GAMMA)
    , weights(Mat())
    , gamma(2.2)
    , deg(3)
    , saturated_threshold({ 0, 0.98 })
    , initial_method_type(INITIAL_METHOD_LEAST_SQUARE)
    , weights_coeff(0)
    , max_count(5000)
    , epsilon(1.e-4)
{}

Mat ColorCorrectionModel::Impl::prepare(const Mat& inp)
{
    switch (ccm_type)
    {
    case cv::ccm::CCM_3x3:
        shape = 9;
        return inp;
    case cv::ccm::CCM_4x3:
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
        CV_Error(Error::StsBadArg, "Wrong ccm_type!");
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

double ColorCorrectionModel::Impl::calc_loss_(Color color)
{
    Mat distlist = color.diff(*dst, distance);
    Color lab = color.to(COLOR_SPACE_Lab_D50_2);
    Mat dist_;
    pow(distlist, 2, dist_);
    if (!weights.empty())
    {
        dist_ = weights.mul(dist_);
    }
    Scalar ss = sum(dist_);
    return ss[0];
}

double ColorCorrectionModel::Impl::calc_loss(const Mat ccm_)
{
    Mat converted = src_rgbl.reshape(1, 0) * ccm_;
    Color color(converted.reshape(3, 0), *(cs.l));
    return calc_loss_(color);
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

void ColorCorrectionModel::Impl::getColor(CONST_COLOR constcolor)
{
    dst = (GetColor::getColor(constcolor));
}
void ColorCorrectionModel::Impl::getColor(Mat colors_, COLOR_SPACE ref_cs_)
{
    dst.reset(new Color(colors_, *GetCS::getInstance().get_cs(ref_cs_)));
}
void ColorCorrectionModel::Impl::getColor(Mat colors_, COLOR_SPACE cs_, Mat colored_)
{
    dst.reset(new Color(colors_, *GetCS::getInstance().get_cs(cs_), colored_));
}
ColorCorrectionModel::ColorCorrectionModel(const Mat& src_, CONST_COLOR constcolor)
    : p(std::make_shared<Impl>())
{
    p->src = src_;
    p->getColor(constcolor);
}
ColorCorrectionModel::ColorCorrectionModel(const Mat& src_, Mat colors_, COLOR_SPACE ref_cs_)
    : p(std::make_shared<Impl>())
{
    p->src = src_;
    p->getColor(colors_, ref_cs_);
}
ColorCorrectionModel::ColorCorrectionModel(const Mat& src_, Mat colors_, COLOR_SPACE cs_, Mat colored_)
    : p(std::make_shared<Impl>())
{
    p->src = src_;
    p->getColor(colors_, cs_, colored_);
}

void ColorCorrectionModel::setColorSpace(COLOR_SPACE cs_)
{
    p->cs = *GetCS::getInstance().get_rgb(cs_);
}
void ColorCorrectionModel::setCCM_TYPE(CCM_TYPE ccm_type_)
{
    p->ccm_type = ccm_type_;
}
void ColorCorrectionModel::setDistance(DISTANCE_TYPE distance_)
{
    p->distance = distance_;
}
void ColorCorrectionModel::setLinear(LINEAR_TYPE linear_type)
{
    p->linear_type = linear_type;
}
void ColorCorrectionModel::setLinearGamma(const double& gamma)
{
    p->gamma = gamma;
}
void ColorCorrectionModel::setLinearDegree(const int& deg)
{
    p->deg = deg;
}
void ColorCorrectionModel::setSaturatedThreshold(const double& lower, const double& upper)
{  //std::vector<double> saturated_threshold
    p->saturated_threshold = { lower, upper };
}
void ColorCorrectionModel::setWeightsList(const Mat& weights_list)
{
    p->weights_list = weights_list;
}
void ColorCorrectionModel::setWeightCoeff(const double& weights_coeff)
{
    p->weights_coeff = weights_coeff;
}
void ColorCorrectionModel::setInitialMethod(INITIAL_METHOD_TYPE initial_method_type)
{
    p->initial_method_type = initial_method_type;
}
void ColorCorrectionModel::setMaxCount(const int& max_count_)
{
    p->max_count = max_count_;
}
void ColorCorrectionModel::setEpsilon(const double& epsilon_)
{
    p->epsilon = epsilon_;
}
void ColorCorrectionModel::run()
{

    Mat saturate_mask = saturate(p->src, p->saturated_threshold[0], p->saturated_threshold[1]);
    p->linear = getLinear(p->gamma, p->deg, p->src, *(p->dst), saturate_mask, (p->cs), p->linear_type);
    p->calWeightsMasks(p->weights_list, p->weights_coeff, saturate_mask);
    p->src_rgbl = p->linear->linearize(maskCopyTo(p->src, p->mask));
    p->dst->colors = maskCopyTo(p->dst->colors, p->mask);
    p->dst_rgbl = p->dst->to(*(p->cs.l)).colors;

    // make no change for CCM_3x3, make change for CCM_4x3.
    p->src_rgbl = p->prepare(p->src_rgbl);

    // distance function may affect the loss function and the fitting function
    switch (p->distance)
    {
    case cv::ccm::DISTANCE_RGBL:
        p->initialLeastSquare(true);
        break;
    default:
        switch (p->initial_method_type)
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
Mat ColorCorrectionModel::get_src_rgbl() const{
    return p->src_rgbl;
}
Mat ColorCorrectionModel::get_dst_rgbl() const{
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
