// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {
class PlaneSacSegmentationTest : public ::testing::Test
{
public:
    vector<vector<float>> models = {
            {0, 0, 1, 0},
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {1, 1, 1, -150},
    };
    vector<float> thrs = {0.1f, 0.2f, 0.4f, 0.5f};
    vector<int> pt_nums = {100, 200, 300, 400};
    vector<vector<float>> limits = {
            {5,  55, 5,  55, 0, 0},
            {0,  0,  5,  55, 5, 55},
            {5,  55, 0,  0,  5, 55},
            {10, 50, 10, 50, 0, 0},
    };

    int models_num = (int) models.size();
    // Used to store point cloud, generated plane and model
    Mat pt_cloud, generated_pts, segmented_plane_models;
    vector<int> label;
    SACSegmentation sacSegmentation;

    void SetUp() override
    {
        // Set basic sac arguments.
        sacSegmentation.setSacModelType(SAC_MODEL_PLANE);
        sacSegmentation.setSacMethodType(SAC_METHOD_RANSAC);
        sacSegmentation.setConfidence(1);
    }

    /**
     * Used to generate a specific plane with random points
     * model: plane coefficient [a,b,c,d] means ax+by+cz+d=0
     * thr: generate the maximum distance from the point to the plane
     * limit: the range of xyz coordinates of the generated plane
     **/
    static void generatePlane(Mat &pt_cloud, const vector<float> &model, float thr, int num,
                              const vector<float> &limit)
    {
        pt_cloud = Mat(num, 3, CV_32F);
        cv::RNG rng(0);
        auto *pt_cloud_ptr = (float *) pt_cloud.data;

        // Part of the points are generated for the specific model
        // The other part of the points are used to increase the thickness of the plane
        int std_num = (int) num / 3;
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

            pt_cloud_ptr[3 * i] = x;
            pt_cloud_ptr[3 * i + 1] = y;
            pt_cloud_ptr[3 * i + 2] = z;
        }
    }
};

class SphereSacSegmentationTest : public ::testing::Test
{
public:
    vector<vector<float>> models = {
            {15,  15,  30,  5},
            {-15, -15, -30, 8},
            {0,   0,   -35, 10},
            {0,   0,   0,   15},
            {0,   0,   0,   20},
    };
    vector<float> thrs = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    vector<int> pt_nums = {100, 200, 300, 400, 500};
    vector<vector<float>> limits = {
            {0,  1, 0,  1, 0,  1},
            {-1, 0, -1, 0, -1, 0},
            {-1, 1, -1, 1, 0,  1},
            {-1, 1, -1, 1, -1, 1},
            {-1, 1, -1, 1, -1, 0},
    };

    int models_num = (int) models.size();
    // Used to store point cloud, generated plane and model
    Mat pt_cloud, generated_pts, segmented_sphere_models;
    vector<int> label;
    SACSegmentation sacSegmentation;

    void SetUp() override
    {
        // Set basic sac arguments.
        sacSegmentation.setSacModelType(SAC_MODEL_SPHERE);
        sacSegmentation.setSacMethodType(SAC_METHOD_RANSAC);
        sacSegmentation.setConfidence(1);
    }

    /**
    * Used to generate a specific sphere with random points
    * model: sphere coefficient [x,y,z,r] means x^2+y^2+z^2=r^2
    * thr: generate the maximum distance from the point to the surface of sphere
    * limit: the range of vector to make the generated sphere incomplete
    **/
    static void generateSphere(Mat &pt_cloud, const vector<float> &model, float thr, int num,
                               const vector<float> &limit)
    {
        pt_cloud = cv::Mat(num, 3, CV_32F);
        cv::RNG rng(0);
        auto *pt_cloud_ptr = (float *) pt_cloud.data;

        // Part of the points are generated for the specific model
        // The other part of the points are used to increase the thickness of the sphere
        int sphere_num = (int) num / 3;
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

            pt_cloud_ptr[3 * i] = model[0] + vec[0];
            pt_cloud_ptr[3 * i + 1] = model[1] + vec[1];
            pt_cloud_ptr[3 * i + 2] = model[2] + vec[2];
        }
    }
};

void normalizeMat(Mat &m)
{
    m.convertTo(m, CV_32F);
    float l = (float) sqrt(m.dot(m));
    m /= l;
}

float getDiff(const Mat &a, const Mat &b)
{
    // If the direction is the same
    Mat d1 = a - b;
    // If the direction is not the same
    Mat d2 = a + b;
    return (float) min(d1.dot(d1), d2.dot(d2));
}

int countNum(const vector<int> &m, int num)
{
    int t = 0;
    for (int a: m)
        if (a == num) t++;
    return t;
}

TEST_F(PlaneSacSegmentationTest, SinglePlaneSegmentation)
{
    sacSegmentation.setMaxIterations(1000);
    sacSegmentation.setNumberOfModelsExpected(1);

    // Single plane segmentation
    for (int i = 0; i < models_num; i++)
    {
        generatePlane(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
        //A point with a distance equal to the threshold is not considered an inliner point
        sacSegmentation.setDistanceThreshold(thrs[i] + 0.01);
        int num = sacSegmentation.segment(pt_cloud, label, segmented_plane_models);

        ASSERT_EQ(1, num)
        << "Model number should be equal to 1.";
        ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
        << "Label size should be equal to point number.";

        Mat ans_model(1, 4, CV_32F, models[i].data()), segmented_model(segmented_plane_models);
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[i], countNum(label, 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = 0; j < pt_cloud.rows; j++)
        {
            if (j < pt_cloud.rows - pt_nums[i])
                ASSERT_EQ(0, label[j])
                << "This index should not be marked: " << j << ". This point is "
                << pt_cloud.row(j);
            else
                ASSERT_EQ(1, label[j])
                << "This index should be marked: " << j << ". This point is "
                << pt_cloud.row(j);
        }
    }
}

TEST_F(PlaneSacSegmentationTest, MultiplePlaneSegmentation)
{
    sacSegmentation.setMaxIterations(1000);
    sacSegmentation.setDistanceThreshold(0.51);
    sacSegmentation.setNumberOfModelsExpected(models_num);

    for (int i = 0; i < models_num; i++)
    {
        generatePlane(generated_pts, models[i], 0.5, pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }

    // Multiple planes segmentation
    int num = sacSegmentation.segment(pt_cloud, label, segmented_plane_models);

    ASSERT_EQ(models_num, num)
    << "Model number should be equal to " << models_num << ".";
    ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
    << "Label size should be equal to point number.";

    int checked_num = 0;
    for (int i = 0; i < models_num; i++)
    {
        Mat ans_model(1, 4, CV_32F, models[models_num - 1 - i].data()), segmented_model(
                segmented_plane_models.row(i));
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[models_num - 1 - i], countNum(label, i + 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = checked_num; j < pt_nums[i]; j++)
            ASSERT_EQ(models_num - i, label[j])
            << "This index " << j << " should be marked as "<< models_num - i
            << ". This point is " << pt_cloud.row(j);
        checked_num += pt_nums[i];
    }
}

TEST_F(PlaneSacSegmentationTest, PlaneSegmentationWithConstraints)
{
    sacSegmentation.setNumberOfModelsExpected(1);
    // Just use 3 models
    models_num = 3;

    for (int i = 0; i < models_num; i++)
    {
        generatePlane(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }

    int checked_num = 0;
    for (int i = 0; i < models_num; i++)
    {
        sacSegmentation.setMaxIterations(3000 - i * 1000);
        sacSegmentation.setDistanceThreshold(thrs[i] + 0.01);

        // The angle between the model normals and the constraints must be less than 1 degree
        float radian_thr = 1 * 3.1415926f / 180;
        vector<float> constraint_normal = {models[i][0], models[i][1], models[i][2]};

        // Normal vector constraint function
        SACSegmentation::ModelConstraintFunction constraint = [constraint_normal, radian_thr](
                const vector<double> &model) -> bool {
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

        sacSegmentation.setCustomModelConstraints(constraint);

        // Single plane segmentation with constraint
        int num = sacSegmentation.segment(pt_cloud, label, segmented_plane_models);

        ASSERT_EQ(1, num)
        << "Model number should be equal to 1.";
        ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
        << "Label size should be equal to point number.";

        Mat ans_model(1, 4, CV_32F, models[i].data()), segmented_model(segmented_plane_models);
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[i], countNum(label, 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = 0; j < pt_cloud.rows; j++)
        {
            if (j >= checked_num && j < checked_num + pt_nums[i])
                ASSERT_EQ(1, label[j])
                << "This index should be marked: " << j << ". This point is "
                << pt_cloud.row(j);
            else
                ASSERT_EQ(0, label[j])
                << "This index should not be marked: " << j << ". This point is "
                << pt_cloud.row(j);
        }
        checked_num += pt_nums[i];
    }
}

TEST_F(SphereSacSegmentationTest, SingleSphereSegmentation)
{
    sacSegmentation.setMaxIterations(3000);
    sacSegmentation.setNumberOfModelsExpected(1);

    // Single sphere segmentation
    for (int i = 0; i < models_num; i++)
    {
        generateSphere(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
        //A point with a distance equal to the threshold is not considered an inliner point
        sacSegmentation.setDistanceThreshold(thrs[i] + 0.05);
        int num = sacSegmentation.segment(pt_cloud, label, segmented_sphere_models);

        ASSERT_EQ(1, num)
        << "Model number should be equal to 1.";
        ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
        << "Label size should be equal to point number.";

        Mat ans_model(1, 4, CV_32F, models[i].data()), segmented_model(segmented_sphere_models);
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[i], countNum(label, 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = 0; j < pt_cloud.rows; j++)
        {
            if (j < pt_cloud.rows - pt_nums[i])
                ASSERT_EQ(0, label[j])
                << "This index should not be marked: " << j << ". This point is "
                << pt_cloud.row(j);
            else
                ASSERT_EQ(1, label[j])
                << "This index should be marked: " << j << ". This point is "
                << pt_cloud.row(j);
        }
    }
}

TEST_F(SphereSacSegmentationTest, MultipleSphereSegmentation)
{
    sacSegmentation.setMaxIterations(3000);
    sacSegmentation.setDistanceThreshold(0.55);
    sacSegmentation.setNumberOfModelsExpected(models_num);

    for (int i = 0; i < models_num; i++)
    {
        generateSphere(generated_pts, models[i], 0.5, pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }

    // Multiple spheres segmentation
    int num = sacSegmentation.segment(pt_cloud, label, segmented_sphere_models);

    ASSERT_EQ(models_num, num)
    << "Model number should be equal to " << models_num << ".";
    ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
    << "Label size should be equal to point number.";

    int checked_num = 0;
    for (int i = 0; i < models_num; i++)
    {
        Mat ans_model(1, 4, CV_32F, models[models_num - 1 - i].data()), segmented_model(
                segmented_sphere_models.row(i));
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[models_num - 1 - i], countNum(label, i + 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = checked_num; j < pt_nums[i]; j++)
            ASSERT_EQ(models_num - i, label[j])
            << "This index " << j << " should be marked as "<< models_num - i
            << ". This point is " << pt_cloud.row(j);
        checked_num += pt_nums[i];
    }
}

TEST_F(SphereSacSegmentationTest, SphereSegmentationWithConstraints)
{
    sacSegmentation.setNumberOfModelsExpected(1);
    // Just use 3 models
    models_num = 3;

    for (int i = 0; i < models_num; i++)
    {
        generateSphere(generated_pts, models[i], thrs[i], pt_nums[i], limits[i]);
        pt_cloud.push_back(generated_pts);
    }

    int checked_num = 0;
    for (int i = 0; i < models_num; i++)
    {
        sacSegmentation.setMaxIterations(3000 - i * 1000);
        sacSegmentation.setDistanceThreshold(thrs[i] + 0.05);

        float constraint_radius = models[i][3] + 1;

        // Radius constraint function
        SACSegmentation::ModelConstraintFunction constraint = [constraint_radius](
                const vector<double> &model) -> bool {
            auto model_radius = (float) model[3];
            return model_radius <= constraint_radius;
        };

        sacSegmentation.setCustomModelConstraints(constraint);

        // Single sphere segmentation with constraint
        int num = sacSegmentation.segment(pt_cloud, label, segmented_sphere_models);

        ASSERT_EQ(1, num)
        << "Model number should be equal to 1.";
        ASSERT_EQ(pt_cloud.rows, (int)(label.size()))
        << "Label size should be equal to point number.";

        Mat ans_model(1, 4, CV_32F, models[i].data()), segmented_model(segmented_sphere_models);
        normalizeMat(ans_model);
        normalizeMat(segmented_model);

        EXPECT_LE(getDiff(ans_model, segmented_model), 0.0001)
        << "Initial model is " << ans_model << ". Segmented model is " << segmented_model
        << ". The difference in coefficients should not be too large.";
        ASSERT_EQ(pt_nums[i], countNum(label, 1))
        << "There are " << pt_nums[i] << " points need to be marked.";

        for (int j = 0; j < pt_cloud.rows; j++)
        {
            if (j >= checked_num && j < checked_num + pt_nums[i])
                ASSERT_EQ(1, label[j])
                << "This index should be marked: " << j << ". This point is "
                << pt_cloud.row(j);
            else
                ASSERT_EQ(0, label[j])
                << "This index should not be marked: " << j << ". This point is "
                << pt_cloud.row(j);
        }
        checked_num += pt_nums[i];
    }
}

} // namespace
} // opencv_test
