// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

int countNum(const vector<int> &m, int num)
{
    int t = 0;
    for (int a: m)
        if (a == num) t++;
    return t;
}

string getHeader(const SACSegmentation &s){
    string r;
    if(s.getNumberOfThreads() == -1)
        r += "One thread ";
    else
        r += std::to_string(s.getNumberOfThreads()) + "-thread ";

    if(s.getNumberOfModelsExpected() == 1)
        r += "single model segmentation ";
    else
        r += std::to_string(s.getNumberOfModelsExpected()) + " models segmentation ";

    if(s.getCustomModelConstraints() == nullptr)
        r += "without constraint:\n";
    else
        r += "with constraint:\n";

    r += "Confidence: " + std::to_string(s.getConfidence()) + "\n";
    r += "Max Iterations: " + std::to_string(s.getMaxIterations()) + "\n";
    r += "Expected Models Number: " + std::to_string(s.getNumberOfModelsExpected()) + "\n";
    r += "Distance Threshold: " + std::to_string(s.getDistanceThreshold());

    return r;
}

class SacSegmentationTest : public ::testing::Test
{
public:
    // Used to store the parameters of model generation
    vector<vector<float>> models, limits;
    vector<float> thrs;
    vector<int> pt_nums;

    int models_num = 0;
    // Used to store point cloud, generated plane and model
    Mat pt_cloud, generated_pts, segmented_models;
    vector<int> label;
    SACSegmentation sacSegmentation;
    SACSegmentation::ModelConstraintFunction model_constraint = nullptr;
    using CheckDiffFunction = std::function<bool(const Mat &, const Mat &)>;

    void singleModelSegmentation(int iter_num, const CheckDiffFunction &checkDiff, int idx)
    {
        sacSegmentation.setSacMethodType(SAC_METHOD_RANSAC);
        sacSegmentation.setConfidence(1);
        sacSegmentation.setMaxIterations(iter_num);
        sacSegmentation.setNumberOfModelsExpected(1);
        //A point with a distance equal to the threshold is not considered an inliner point
        sacSegmentation.setDistanceThreshold(thrs[idx] + 0.01);

        int num = sacSegmentation.segment(pt_cloud, label, segmented_models);

        string header = getHeader(sacSegmentation);

        ASSERT_EQ(1, num)
        << header << endl
        << "Model number should be equal to 1.";
        ASSERT_EQ(pt_cloud.rows, (int) (label.size()))
        << header << endl
        << "Label size should be equal to point number.";

        Mat ans_model, segmented_model;
        ans_model = Mat(1, (int) models[0].size(), CV_32F, models[idx].data());
        segmented_models.row(0).convertTo(segmented_model, CV_32F);

        ASSERT_TRUE(checkDiff(ans_model, segmented_model))
        << header << endl
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[idx], countNum(label, 1))
        << header << endl
        << "There are " << pt_nums[idx] << " points need to be marked.";

        int start_idx = 0;
        for (int i = 0; i < idx; i++) start_idx += pt_nums[i];

        for (int i = 0; i < pt_cloud.rows; i++)
        {
            if (i >= start_idx && i < start_idx + pt_nums[idx])
                ASSERT_EQ(1, label[i])
                << header << endl
                << "This index should be marked: " << i
                << ". This point is " << pt_cloud.row(i);
            else
                ASSERT_EQ(0, label[i])
                << header << endl
                << "This index should not be marked: "
                << i << ". This point is " << pt_cloud.row(i);
        }
    }

    void multiModelSegmentation(int iter_num, const CheckDiffFunction &checkDiff)
    {
        sacSegmentation.setSacMethodType(SAC_METHOD_RANSAC);
        sacSegmentation.setConfidence(1);
        sacSegmentation.setMaxIterations(iter_num);
        sacSegmentation.setNumberOfModelsExpected(models_num);
        sacSegmentation.setDistanceThreshold(thrs[models_num - 1] + 0.01);

        int num = sacSegmentation.segment(pt_cloud, label, segmented_models);

        string header = getHeader(sacSegmentation);

        ASSERT_EQ(models_num, num)
        << header << endl
        << "Model number should be equal to " << models_num << ".";
        ASSERT_EQ(pt_cloud.rows, (int) (label.size()))
        << header << endl
        << "Label size should be equal to point number.";

        int checked_num = 0;
        for (int i = 0; i < models_num; i++)
        {
            Mat ans_model, segmented_model;
            ans_model = Mat(1, (int) models[0].size(), CV_32F, models[models_num - 1 - i].data());
            segmented_models.row(i).convertTo(segmented_model, CV_32F);

            ASSERT_TRUE(checkDiff(ans_model, segmented_model))
            << header << endl
            << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
            << ". The difference in coefficients should not be too large.";
            ASSERT_EQ(pt_nums[models_num - 1 - i], countNum(label, i + 1))
            << header << endl
            << "There are " << pt_nums[i] << " points need to be marked.";

            for (int j = checked_num; j < pt_nums[i]; j++)
                ASSERT_EQ(models_num - i, label[j])
                << header << endl
                << "This index " << j << " should be marked as " << models_num - i
                << ". This point is " << pt_cloud.row(j);
            checked_num += pt_nums[i];
        }
    }

};

TEST_F(SacSegmentationTest, PlaneSacSegmentation)
{
    sacSegmentation.setSacModelType(SAC_MODEL_PLANE);
    models = {
            {0, 0, 1, 0},
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {1, 1, 1, -150},
    };
    thrs = {0.1f, 0.2f, 0.3f, 0.4f};
    pt_nums = {100, 200, 300, 400};
    limits = {
            {5,  55, 5,  55, 0, 0},
            {0,  0,  5,  55, 5, 55},
            {5,  55, 0,  0,  5, 55},
            {10, 50, 10, 50, 0, 0},
    };
    models_num = (int) models.size();

    /**
    * Used to generate a specific plane with random points
    * model: plane coefficient [a,b,c,d] means ax+by+cz+d=0
    * thr: generate the maximum distance from the point to the plane
    * limit: the range of xyz coordinates of the generated plane
    **/
    auto generatePlane = [](Mat &plane_pts, const vector<float> &model, float thr, int num,
            const vector<float> &limit) {
        plane_pts = Mat(num, 3, CV_32F);
        cv::RNG rng(0);
        auto *plane_pts_ptr = (float *) plane_pts.data;

        // Part of the points are generated for the specific model
        // The other part of the points are used to increase the thickness of the plane
        int std_num = (int) (num / 2);
        // Difference of maximum d between two parallel planes
        float d_thr = thr * sqrt(model[0] * model[0] + model[1] * model[1] + model[2] * model[2]);

        for (int i = 0; i < num; i++)
        {
            // Let d change then generate thickness
            float d = i < std_num ? model[3] : rng.uniform(model[3] - d_thr, model[3] + d_thr);
            float x, y, z;
            // c is 0 means the plane is vertical
            if (model[2] == 0)
            {
                z = rng.uniform(limit[4], limit[5]);
                if (model[0] == 0)
                {
                    x = rng.uniform(limit[0], limit[1]);
                    y = -d / model[1];
                }
                else if (model[1] == 0)
                {
                    x = -d / model[0];
                    y = rng.uniform(limit[2], limit[3]);
                }
                else
                {
                    x = rng.uniform(limit[0], limit[1]);
                    y = -(model[0] * x + d) / model[1];
                }
            }
                // c is not 0
            else
            {
                x = rng.uniform(limit[0], limit[1]);
                y = rng.uniform(limit[2], limit[3]);
                z = -(model[0] * x + model[1] * y + d) / model[2];
            }

            plane_pts_ptr[3 * i] = x;
            plane_pts_ptr[3 * i + 1] = y;
            plane_pts_ptr[3 * i + 2] = z;
        }
    };

    // 1 * 3.1415926f / 180
    float vector_radian_tolerance = 0.0174533f, ratio_tolerance = 0.1f;
    CheckDiffFunction planeCheckDiff = [vector_radian_tolerance, ratio_tolerance](const Mat &a,
            const Mat &b) -> bool {
        Mat m1, m2;
        a.convertTo(m1, CV_32F);
        b.convertTo(m2, CV_32F);
        auto p1 = (float *) m1.data, p2 = (float *) m2.data;
        Vec3f n1(p1[0], p1[1], p1[2]);
        Vec3f n2(p2[0], p2[1], p2[2]);
        float cos_theta_square = n1.dot(n2) * n1.dot(n2) / (n1.dot(n1) * n2.dot(n2));

        float r1 = p1[3] * p1[3] / n1.dot(n1);
        float r2 = p2[3] * p2[3] / n2.dot(n2);

        return cos_theta_square >= cos(vector_radian_tolerance) * cos(vector_radian_tolerance)
               && abs(r1 - r2) <= ratio_tolerance * ratio_tolerance;
    };

    // Single plane segmentation
    for (int i = 0; i < models_num; i++)
    {
        generatePlane(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
        singleModelSegmentation(1000, planeCheckDiff, i);
    }

    // Single plane segmentation with constraint
    for (int i = models_num / 2; i < models_num; i++)
    {
        vector<float> constraint_normal = {models[i][0], models[i][1], models[i][2]};
        // Normal vector constraint function
        model_constraint = [constraint_normal](const vector<double> &model) -> bool {
            // The angle between the model normals and the constraints must be less than 1 degree
            // 1 * 3.1415926f / 180
            float radian_thr = 0.0174533f;
            vector<float> model_normal = {(float) model[0], (float) model[1], (float) model[2]};
            float dot12 = constraint_normal[0] * model_normal[0] +
                          constraint_normal[1] * model_normal[1] +
                          constraint_normal[2] * model_normal[2];
            float m1m1 = constraint_normal[0] * constraint_normal[0] +
                         constraint_normal[1] * constraint_normal[1] +
                         constraint_normal[2] * constraint_normal[2];
            float m2m2 = model_normal[0] * model_normal[0] +
                         model_normal[1] * model_normal[1] +
                         model_normal[2] * model_normal[2];
            float square_cos_theta = dot12 * dot12 / (m1m1 * m2m2);

            return square_cos_theta >= cos(radian_thr) * cos(radian_thr);
        };
        sacSegmentation.setCustomModelConstraints(model_constraint);
        singleModelSegmentation(5000, planeCheckDiff, i);
    }

    pt_cloud.release();
    sacSegmentation.setCustomModelConstraints(nullptr);
    sacSegmentation.setNumberOfThreads(3);

    // Multi-plane segmentation
    for (int i = 0; i < models_num; i++)
    {
        generatePlane(generated_pts, models[i], thrs[models_num - 1], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }
    multiModelSegmentation(1000, planeCheckDiff);
}

TEST_F(SacSegmentationTest, SphereSacSegmentation)
{
    sacSegmentation.setSacModelType(cv::SAC_MODEL_SPHERE);
    models = {
            {15,  15,  30,  5},
            {-15, -15, -30, 8},
            {0,   0,   -35, 10},
            {0,   0,   0,   15},
            {0,   0,   0,   20},
    };
    thrs = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    pt_nums = {100, 200, 300, 400, 500};
    limits = {
            {0,  1, 0,  1, 0,  1},
            {-1, 0, -1, 0, -1, 0},
            {-1, 1, -1, 1, 0,  1},
            {-1, 1, -1, 1, -1, 1},
            {-1, 1, -1, 1, -1, 0},
    };
    models_num = (int) models.size();

    /**
    * Used to generate a specific sphere with random points
    * model: sphere coefficient [x,y,z,r] means x^2+y^2+z^2=r^2
    * thr: generate the maximum distance from the point to the surface of sphere
    * limit: the range of vector to make the generated sphere incomplete
    **/
    auto generateSphere = [](Mat &sphere_pts, const vector<float> &model, float thr, int num,
            const vector<float> &limit) {
        sphere_pts = cv::Mat(num, 3, CV_32F);
        cv::RNG rng(0);
        auto *sphere_pts_ptr = (float *) sphere_pts.data;

        // Part of the points are generated for the specific model
        // The other part of the points are used to increase the thickness of the sphere
        int sphere_num = (int) (num / 1.5);
        for (int i = 0; i < num; i++)
        {
            // Let r change then generate thickness
            float r = i < sphere_num ? model[3] : rng.uniform(model[3] - thr, model[3] + thr);
            // Generate a random vector and normalize it.
            Vec3f vec(rng.uniform(limit[0], limit[1]), rng.uniform(limit[2], limit[3]),
                    rng.uniform(limit[4], limit[5]));
            float l = sqrt(vec.dot(vec));
            // Normalizes it to have a magnitude of r
            vec /= l / r;

            sphere_pts_ptr[3 * i] = model[0] + vec[0];
            sphere_pts_ptr[3 * i + 1] = model[1] + vec[1];
            sphere_pts_ptr[3 * i + 2] = model[2] + vec[2];
        }
    };

    float distance_tolerance = 0.1f, radius_tolerance = 0.1f;
    CheckDiffFunction sphereCheckDiff = [distance_tolerance, radius_tolerance](const Mat &a,
            const Mat &b) -> bool {
        Mat d = a - b;
        auto d_ptr = (float *) d.data;
        // Distance square between sphere centers
        float d_square = d_ptr[0] * d_ptr[0] + d_ptr[1] * d_ptr[1] + d_ptr[2] * d_ptr[2];
        // Difference square between radius of two spheres
        float r_square = d_ptr[3] * d_ptr[3];

        return d_square <= distance_tolerance * distance_tolerance &&
               r_square <= radius_tolerance * radius_tolerance;
    };

    // Single sphere segmentation
    for (int i = 0; i < models_num; i++)
    {
        generateSphere(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
        singleModelSegmentation(3000, sphereCheckDiff, i);
    }

    // Single sphere segmentation with constraint
    for (int i = models_num / 2; i < models_num; i++)
    {
        float constraint_radius = models[i][3] + 0.5f;
        // Radius constraint function
        model_constraint = [constraint_radius](
                const vector<double> &model) -> bool {
            auto model_radius = (float) model[3];
            return model_radius <= constraint_radius;
        };
        sacSegmentation.setCustomModelConstraints(model_constraint);
        singleModelSegmentation(10000, sphereCheckDiff, i);
    }

    pt_cloud.release();
    sacSegmentation.setCustomModelConstraints(nullptr);
    sacSegmentation.setNumberOfThreads(3);

    // Multi-sphere segmentation
    for (int i = 0; i < models_num; i++)
    {
        generateSphere(generated_pts, models[i], thrs[models_num - 1], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }
    multiModelSegmentation(5000, sphereCheckDiff);
}

class CylinderSacSegmentationTest : public ::testing::Test
{
public:
    Mat generateCylinder(float *model, float thr, int num, float *limitVec, int seed = 0)
    {
        Mat pt_cloud;
        pt_cloud = cv::Mat(num, 3, CV_32F);
        cv::RNG rng(seed);
        auto *pt_cloud_ptr = (float *) pt_cloud.data;

        // Part of the points are generated for the specific model.
        int cylinder_num = (int) num / 3;

        Vec3f cen(model[0], model[1], model[2]);
        Vec3f dir(model[3], model[4], model[5]);
        for (int i = 0; i < num; i++)
        {
            float r = i < cylinder_num ? model[6] : rng.uniform(model[6] - thr, model[6] + thr);
            Vec3f vec(rng.uniform(limitVec[0], limitVec[1]), rng.uniform(limitVec[2], limitVec[3]),
                    rng.uniform(limitVec[4], limitVec[5]));
            // The height relative to the center point.
            float height = rng.uniform(0.0f, model[7]);
            // Place the point on the surface of the cylinder.
            vec = dir.cross(vec);
            float l = sqrt(vec.dot(vec));
            vec *= r / l;
            vec += height * dir;
            Vec3f p(vec + cen);

            pt_cloud_ptr[3 * i] = cen[0] + vec[0];
            pt_cloud_ptr[3 * i + 1] = cen[1] + vec[1];
            pt_cloud_ptr[3 * i + 2] = cen[2] + vec[2];
        }
        return pt_cloud;
    }
};

} // namespace
} // opencv_test
