/**
 * @file 3d_sac_segmentation.cpp
 * @brief It demonstrates the usage of cv::SACSegmentation.
 *
 * It shows how to implement 3D point cloud plane segmentation
 * using the RANSAC algorithm via cv::SACSegmentation, and
 * the construction of cv::SACSegmentation::ModelConstraintFunction
 *
 * @author Yechun Ruan
 * @date December 2021
 */

#include <opencv2/core.hpp>
#include <opencv2/3d.hpp>

bool customFunc(const std::vector<double> &model_coefficients);

void usageExampleSacModelConstraintFunction();

int planeSegmentationUsingRANSAC(const cv::Mat &pt_cloud,
        std::vector<cv::Vec4d> &planes_coeffs, std::vector<char> &labels);

//! [usageExampleSacModelConstraintFunction]
bool customFunc(const std::vector<double> &model_coefficients)
{
    // check model_coefficients
    // The plane needs to pass through the origin, i.e. ax+by+cz+d=0 --> d==0
    return model_coefficients[3] == 0;
} // end of function customFunc()

void usageExampleSacModelConstraintFunction()
{
    using namespace cv;

    SACSegmentation::ModelConstraintFunction func_example1 = customFunc;

    SACSegmentation::ModelConstraintFunction func_example2 =
            [](const std::vector<double> &model_coefficients) {
                // check model_coefficients
                // The plane needs to pass through the origin, i.e. ax+by+cz+d=0 --> d==0
                return model_coefficients[3] == 0;
            };

    // Using local variables
    float x0 = 0.0, y0 = 0.0, z0 = 0.0;
    SACSegmentation::ModelConstraintFunction func_example3 =
            [x0, y0, z0](const std::vector<double> &model_coeffs) -> bool {
                // check model_coefficients
                // The plane needs to pass through the point (x0, y0, z0), i.e. ax0+by0+cz0+d == 0
                return model_coeffs[0] * x0 + model_coeffs[1] * y0 + model_coeffs[2] * z0
                       + model_coeffs[3] == 0;
            };

    // Next, use the constructed SACSegmentation::ModelConstraintFunction func_example1, 2, 3 ......

}
//! [usageExampleSacModelConstraintFunction]

//! [planeSegmentationUsingRANSAC]
int planeSegmentationUsingRANSAC(const cv::Mat &pt_cloud,
        std::vector<cv::Vec4d> &planes_coeffs, std::vector<char> &labels)
{
    using namespace cv;

    Ptr<SACSegmentation> sacSegmentation =
            SACSegmentation::create(SAC_MODEL_PLANE, SAC_METHOD_RANSAC);
    sacSegmentation->setDistanceThreshold(0.21);
    // The maximum number of iterations to attempt.(default 1000)
    sacSegmentation->setMaxIterations(1500);
    sacSegmentation->setNumberOfModelsExpected(2);

    Mat planes_coeffs_mat;
    // Number of final resultant models obtained by segmentation.
    int model_cnt = sacSegmentation->segment(pt_cloud,
            labels, planes_coeffs_mat);

    planes_coeffs.clear();
    for (int i = 0; i < model_cnt; ++i)
    {
        planes_coeffs.push_back(planes_coeffs_mat.row(i));
    }

    return model_cnt;
}
//! [planeSegmentationUsingRANSAC]

int main()
{

}