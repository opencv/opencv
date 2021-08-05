// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/3d.hpp>

namespace opencv_test { namespace {

#if 0
Point3f
rayPlaneIntersection(Point2f uv, const Mat& centroid, const Mat& normal, const Mat_<float>& Kinv)
{
    Matx33d dKinv(Kinv);
    Vec3d dNormal(normal);
    return rayPlaneIntersection(Vec3d(uv.x, uv.y, 1), centroid.dot(normal), dNormal, dKinv);
}
#endif

Vec3f rayPlaneIntersection(const Vec3d& uv1, double centroid_dot_normal, const Vec3d& normal, const Matx33d& Kinv)
{
    Matx31d L = Kinv * uv1; //a ray passing through camera optical center
    //and uv.
    L = L * (1.0 / cv::norm(L));
    double LdotNormal = L.dot(normal);
    double d;
    if (std::fabs(LdotNormal) > 1e-9)
    {
        d = centroid_dot_normal / LdotNormal;
    }
    else
    {
        d = 1.0;
        std::cout << "warning, LdotNormal nearly 0! " << LdotNormal << std::endl;
        std::cout << "contents of L, Normal: " << Mat(L) << ", " << Mat(normal) << std::endl;
    }
    Vec3f xyz((float)(d * L(0)), (float)(d * L(1)), (float)(d * L(2)));
    return xyz;
}

const int W = 640;
const int H = 480;
//int window_size = 5;
float focal_length = 525;
float cx = W / 2.f + 0.5f;
float cy = H / 2.f + 0.5f;

Mat K = (Mat_<double>(3, 3) << focal_length, 0, cx, 0, focal_length, cy, 0, 0, 1);
Mat Kinv = K.inv();

void points3dToDepth16U(const Mat_<Vec3f>& points3d, Mat& depthMap);

void points3dToDepth16U(const Mat_<Vec3f>& points3d, Mat& depthMap)
{
    std::vector<Point3f> points3dvec;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            points3dvec.push_back(Point3f(points3d(i, j)[0], points3d(i, j)[1], points3d(i, j)[2]));

    std::vector<Point2f> img_points;
    depthMap = Mat::zeros(H, W, CV_32F);
    Vec3f R(0.0, 0.0, 0.0);
    Vec3f T(0.0, 0.0, 0.0);
    cv::projectPoints(points3dvec, R, T, K, Mat(), img_points);

    int index = 0;
    for (int i = 0; i < H; i++)
    {

        for (int j = 0; j < W; j++)
        {
            float value = (points3d.at<Vec3f>(i, j))[2]; // value is the z
            depthMap.at<float>(cvRound(img_points[index].y), cvRound(img_points[index].x)) = value;
            index++;
        }
    }
    depthMap.convertTo(depthMap, CV_16U, 1000);
}

static RNG rng;
struct Plane
{
    Vec3d n, p;
    double p_dot_n;
    Plane()
    {
        n[0] = rng.uniform(-0.5, 0.5);
        n[1] = rng.uniform(-0.5, 0.5);
        n[2] = -0.3; //rng.uniform(-1.f, 0.5f);
        n = n / cv::norm(n);
        set_d((float)rng.uniform(-2.0, 0.6));
    }

    void
        set_d(float d)
    {
        p = Vec3d(0, 0, d / n[2]);
        p_dot_n = p.dot(n);
    }

    Vec3f
        intersection(float u, float v, const Matx33f& Kinv_in) const
    {
        return rayPlaneIntersection(Vec3d(u, v, 1), p_dot_n, n, Kinv_in);
    }
};

void gen_points_3d(std::vector<Plane>& planes_out, Mat_<unsigned char> &plane_mask, Mat& points3d, Mat& normals,
                   int n_planes)
{
    std::vector<Plane> planes;
    for (int i = 0; i < n_planes; i++)
    {
        Plane px;
        for (int j = 0; j < 1; j++)
        {
            px.set_d(rng.uniform(-3.f, -0.5f));
            planes.push_back(px);
        }
    }
    Mat_ < Vec3f > outp(H, W);
    Mat_ < Vec3f > outn(H, W);
    plane_mask.create(H, W);

    // n  ( r - r_0) = 0
    // n * r_0 = d
    //
    // r_0 = (0,0,0)
    // r[0]
    for (int v = 0; v < H; v++)
    {
        for (int u = 0; u < W; u++)
        {
            unsigned int plane_index = (unsigned int)((u / float(W)) * planes.size());
            Plane plane = planes[plane_index];
            outp(v, u) = plane.intersection((float)u, (float)v, Kinv);
            outn(v, u) = plane.n;
            plane_mask(v, u) = (uchar)plane_index;
        }
    }
    planes_out = planes;
    points3d = outp;
    normals = outn;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RgbdNormalsTest
{
public:
    RgbdNormalsTest() { }
    ~RgbdNormalsTest() { }

    void run()
    {
        Mat_<unsigned char> plane_mask;
        for (unsigned char i = 0; i < 3; ++i)
        {
            RgbdNormals::RGBD_NORMALS_METHOD method = RgbdNormals::RGBD_NORMALS_METHOD_FALS;;
            // inner vector: whether it's 1 plane or 3 planes
            // outer vector: float or double
            std::vector<std::vector<float> > errors(2);
            errors[0].resize(4);
            errors[1].resize(4);
            switch (i)
            {
            case 0:
                method = RgbdNormals::RGBD_NORMALS_METHOD_FALS;
                CV_LOG_INFO(NULL, "*** FALS");
                errors[0][0] = 0.006f;
                errors[0][1] = 0.03f;
                errors[1][0] = 0.0001f;
                errors[1][1] = 0.02f;
                break;
            case 1:
                method = RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD;
                CV_LOG_INFO(NULL, "*** LINEMOD");
                errors[0][0] = 0.04f;
                errors[0][1] = 0.07f;
                errors[0][2] = 0.04f; // depth 16U 1 plane
                errors[0][3] = 0.07f; // depth 16U 3 planes

                errors[1][0] = 0.05f;
                errors[1][1] = 0.08f;
                errors[1][2] = 0.05f; // depth 16U 1 plane
                errors[1][3] = 0.08f; // depth 16U 3 planes
                break;
            case 2:
                method = RgbdNormals::RGBD_NORMALS_METHOD_SRI;
                CV_LOG_INFO(NULL, "*** SRI");
                errors[0][0] = 0.02f;
                errors[0][1] = 0.04f;
                errors[1][0] = 0.02f;
                errors[1][1] = 0.04f;
                break;
            }

            for (unsigned char j = 0; j < 2; ++j)
            {
                int depth = (j % 2 == 0) ? CV_32F : CV_64F;
                if (depth == CV_32F)
                {
                    CV_LOG_INFO(NULL, " * float");
                }
                else
                {
                    CV_LOG_INFO(NULL, " * double");
                }

                Ptr<RgbdNormals> normals_computer = RgbdNormals::create(H, W, depth, K, 5, method);
                normals_computer->cache();

                std::vector<Plane> plane_params;
                Mat points3d, ground_normals;
                // 1 plane, continuous scene, very low error..
                CV_LOG_INFO(NULL, "1 plane - input 3d points");
                float err_mean = 0;
                for (int ii = 0; ii < 5; ++ii)
                {
                    gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 1);
                    err_mean += testit(points3d, ground_normals, normals_computer);
                }
                CV_LOG_INFO(NULL, "mean diff: " << (err_mean / 5));
                EXPECT_LE(err_mean / 5, errors[j][0]);

                // 3 discontinuities, more error expected.
                CV_LOG_INFO(NULL, "3 planes");
                err_mean = 0;
                for (int ii = 0; ii < 5; ++ii)
                {
                    gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 3);
                    err_mean += testit(points3d, ground_normals, normals_computer);
                }
                CV_LOG_INFO(NULL, "mean diff: " << (err_mean / 5));
                EXPECT_LE(err_mean / 5, errors[j][1]);

                if (method == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD)
                {
                    // depth 16U test
                    CV_LOG_INFO(NULL, "** depth 16U - 1 plane");
                    err_mean = 0;
                    for (int ii = 0; ii < 5; ++ii)
                    {
                        gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 1);
                        Mat depthMap;
                        points3dToDepth16U(points3d, depthMap);
                        err_mean += testit(depthMap, ground_normals, normals_computer);
                    }
                    CV_LOG_INFO(NULL, "mean diff: " << (err_mean / 5));
                    EXPECT_LE(err_mean / 5, errors[j][2]);

                    CV_LOG_INFO(NULL, "** depth 16U - 3 plane");
                    err_mean = 0;
                    for (int ii = 0; ii < 5; ++ii)
                    {
                        gen_points_3d(plane_params, plane_mask, points3d, ground_normals, 3);
                        Mat depthMap;
                        points3dToDepth16U(points3d, depthMap);
                        err_mean += testit(depthMap, ground_normals, normals_computer);
                    }
                    CV_LOG_INFO(NULL, "mean diff: " << (err_mean / 5));
                    EXPECT_LE(err_mean / 5, errors[j][3]);
                }
            }
        }

        //TODO test NaNs in data
    }

    float testit(const Mat& points3d, const Mat& in_ground_normals, const Ptr<RgbdNormals>& normals_computer)
    {
        TickMeter tm;
        tm.start();
        Mat in_normals;
        if (normals_computer->getMethod() == RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD && points3d.channels() == 3)
        {
            std::vector<Mat> channels;
            split(points3d, channels);
            normals_computer->apply(channels[2], in_normals);
        }
        else
            normals_computer->apply(points3d, in_normals);
        tm.stop();

        Mat_<Vec3f> normals, ground_normals;
        in_normals.convertTo(normals, CV_32FC3);
        in_ground_normals.convertTo(ground_normals, CV_32FC3);

        float err = 0;
        for (int y = 0; y < normals.rows; ++y)
            for (int x = 0; x < normals.cols; ++x)
            {
                Vec3f vec1 = normals(y, x), vec2 = ground_normals(y, x);
                vec1 = vec1 / cv::norm(vec1);
                vec2 = vec2 / cv::norm(vec2);

                float dot = vec1.dot(vec2);
                // Just for rounding errors
                if (std::abs(dot) < 1)
                    err += std::min(std::acos(dot), std::acos(-dot));
            }

        err /= normals.rows * normals.cols;
        CV_LOG_INFO(NULL, "Average error: " << err << " Speed: " << tm.getTimeMilli() << " ms");
        return err;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RgbdPlaneTest
{
public:
    RgbdPlaneTest() { }
    ~RgbdPlaneTest() { }

    void run()
    {
        std::vector<Plane> planes;
        Mat points3d, ground_normals;
        Mat_<unsigned char> plane_mask;
        gen_points_3d(planes, plane_mask, points3d, ground_normals, 1);
        testit(planes, plane_mask, points3d); // 1 plane, continuous scene, very low error..
        for (int ii = 0; ii < 10; ii++)
        {
            gen_points_3d(planes, plane_mask, points3d, ground_normals, 3); //three planes
            testit(planes, plane_mask, points3d); // 3 discontinuities, more error expected.
        }
    }

    void testit(const std::vector<Plane>& gt_planes, const Mat& gt_plane_mask, const Mat& points3d)
    {
        for (char i_test = 0; i_test < 2; ++i_test)
        {
            TickMeter tm1, tm2;
            Mat plane_mask;
            std::vector<Vec4f> plane_coefficients;

            if (i_test == 0)
            {
                tm1.start();
                // First, get the normals
                int depth = CV_32F;
                Ptr<RgbdNormals> normals_computer = RgbdNormals::create(H, W, depth, K, 5, RgbdNormals::RGBD_NORMALS_METHOD_FALS);
                Mat normals;
                normals_computer->apply(points3d, normals);
                tm1.stop();

                tm2.start();
                findPlanes(points3d, normals, plane_mask, plane_coefficients);
                tm2.stop();
            }
            else
            {
                tm2.start();
                findPlanes(points3d, noArray(), plane_mask, plane_coefficients);
                tm2.stop();
            }

            // Compare each found plane to each ground truth plane
            int n_planes = (int)plane_coefficients.size();
            int n_gt_planes = (int)gt_planes.size();
            Mat_<int> matching(n_gt_planes, n_planes);
            for (int j = 0; j < n_gt_planes; ++j)
            {
                Mat gt_mask = gt_plane_mask == j;
                int n_gt = countNonZero(gt_mask);
                int n_max = 0, i_max = 0;
                for (int i = 0; i < n_planes; ++i)
                {
                    Mat dst;
                    bitwise_and(gt_mask, plane_mask == i, dst);
                    matching(j, i) = countNonZero(dst);
                    if (matching(j, i) > n_max)
                    {
                        n_max = matching(j, i);
                        i_max = i;
                    }
                }
                // Get the best match
                ASSERT_LE(float(n_max - n_gt) / n_gt, 0.001);
                // Compare the normals
                Vec3d normal(plane_coefficients[i_max][0], plane_coefficients[i_max][1], plane_coefficients[i_max][2]);
                ASSERT_GE(std::abs(gt_planes[j].n.dot(normal)), 0.95);
            }

            CV_LOG_INFO(NULL, "Speed: ");
            if (i_test == 0)
                CV_LOG_INFO(NULL, "normals " << tm1.getTimeMilli() << " ms and ");
            CV_LOG_INFO(NULL, "plane " << tm2.getTimeMilli() << " ms");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(RGBD_Normals, compute)
{
  RgbdNormalsTest test;
  test.run();
}

TEST(RGBD_Plane, compute)
{
  RgbdPlaneTest test;
  test.run();
}

TEST(RGBD_Plane, regression_2309_valgrind_check)
{
    Mat points(640, 480, CV_32FC3, Scalar::all(0));
    // Note, 640%9 is 1 and 480%9 is 3
    int blockSize = 9;

    Mat mask;
    std::vector<cv::Vec4f> planes;
    // Will corrupt memory; valgrind gets triggered
    findPlanes(points, noArray(), mask, planes, blockSize);
}

}} // namespace
