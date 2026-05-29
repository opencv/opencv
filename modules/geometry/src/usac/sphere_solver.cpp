// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../ptcloud/ptcloud_wrapper.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {

class SphereModelMinimalSolverImpl : public SphereModelMinimalSolver, public PointCloudWrapper
{

public:
    explicit SphereModelMinimalSolverImpl(const Mat &points_)
            : PointCloudWrapper(points_)
    {
    }

    int getSampleSize() const override
    {
        return 4;
    }

    int getMaxNumberOfSolutions() const override
    {
        return 1;
    }

    /** [center_x, center_y, center_z, radius] <--> (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 = radius^2
     Fitting the sphere using Cramer's Rule.
    */
    int estimate(const std::vector<int> &sample, std::vector<Mat> &models) const override
    {
        models.clear();

        // Get point data
        const int p1_idx = sample[0], p2_idx = sample[1], p3_idx = sample[2], p4_idx = sample[3];
        float x1 = pts_ptr_x[p1_idx], y1 = pts_ptr_y[p1_idx], z1 = pts_ptr_z[p1_idx];
        float x2 = pts_ptr_x[p2_idx], y2 = pts_ptr_y[p2_idx], z2 = pts_ptr_z[p2_idx];
        float x3 = pts_ptr_x[p3_idx], y3 = pts_ptr_y[p3_idx], z3 = pts_ptr_z[p3_idx];
        float x4 = pts_ptr_x[p4_idx], y4 = pts_ptr_y[p4_idx], z4 = pts_ptr_z[p4_idx];

        double center_x, center_y, center_z, radius; // Cramer's Rule
        {
            double a11, a12, a13, a21, a22, a23, a31, a32, a33, t1, t2, t3, t4, d, d1, d2, d3;
            a11 = x2 - x1;
            a12 = y2 - y1;
            a13 = z2 - z1;
            a21 = x3 - x2;
            a22 = y3 - y2;
            a23 = z3 - z2;
            a31 = x4 - x3;
            a32 = y4 - y3;
            a33 = z4 - z3;
            t1 = x1 * x1 + y1 * y1 + z1 * z1;
            t2 = x2 * x2 + y2 * y2 + z2 * z2;
            t3 = x3 * x3 + y3 * y3 + z3 * z3;
            t4 = x4 * x4 + y4 * y4 + z4 * z4;

            double b1 = t2 - t1,
                    b2 = t3 - t2,
                    b3 = t4 - t3;

            //            d = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 -
            //                a12 * a21 * a33 - a13 * a22 * a31;
            //            d1 = b1 * a22 * a33 + a12 * a23 * b3 + a13 * b2 * a32 - b1 * a23 * a32 -
            //                 a12 * b2 * a33 - a13 * a22 * b3;
            //            d2 = a11 * b2 * a33 + b1 * a23 * a31 + a13 * a21 * b3 - a11 * a23 * b3 -
            //                 b1 * a21 * a33 - a13 * b2 * a31;
            //            d3 = a11 * a22 * b3 + a12 * b2 * a31 + b1 * a21 * a32 - a11 * b2 * a32 -
            //                 a12 * a21 * b3 - b1 * a22 * a31;

            double tmp1 = a22 * a33 - a23 * a32,
                    tmp2 = a23 * a31 - a21 * a33,
                    tmp3 = a21 * a32 - a22 * a31;
            d = a11 * tmp1 + a12 * tmp2 + a13 * tmp3;
            d1 = b1 * tmp1 + a12 * (a23 * b3 - b2 * a33) + a13 * (b2 * a32 - a22 * b3);
            d2 = a11 * (b2 * a33 - a23 * b3) + b1 * tmp2 + a13 * (a21 * b3 - b2 * a31);
            d3 = a11 * (a22 * b3 - b2 * a32) + a12 * (b2 * a31 - a21 * b3) + b1 * tmp3;

            if (d == 0) {
                return 0;
            }

            d = 0.5 / d;
            center_x = d1 * d;
            center_y = d2 * d;
            center_z = d3 * d;

            tmp1 = center_x - x1;
            tmp2 = center_y - y1;
            tmp3 = center_z - z1;
            radius = sqrt(tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3);
        }

        double sphere_coeff[4] = {center_x, center_y, center_z, radius};
        models.emplace_back(cv::Mat(1, 4, CV_64F, sphere_coeff).clone());

        return 1;
    }
};


Ptr <SphereModelMinimalSolver> SphereModelMinimalSolver::create(const Mat &points_)
{
    return makePtr<SphereModelMinimalSolverImpl>(points_);
}


class SphereModelNonMinimalSolverImpl : public SphereModelNonMinimalSolver, public PointCloudWrapper
{

public:
    explicit SphereModelNonMinimalSolverImpl(const Mat &points_)
            : PointCloudWrapper(points_)
    {
    }

    int getMinimumRequiredSampleSize() const override
    {
        return 4;
    }

    int getMaxNumberOfSolutions() const override
    {
        return 1;
    }

    /** [center_x, center_y, center_z, radius] <--> (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 = radius^2
     Fitting Sphere Using Differences of Squared Lengths and Squared Radius.
     Reference https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf section 5.2.
     */
    int estimate(const std::vector<int> &sample, int sample_size, std::vector<Mat> &models,
            const std::vector<double> &/*weights*/) const override
    {
        const double inv_sample_size = 1.0 / (double) sample_size;

        // Compute the average of the data points.
        //        Vec3d A(0, 0, 0);
        double A0 = 0, A1 = 0, A2 = 0;

        for (int i = 0; i < sample_size; ++i)
        {
            int sample_idx = sample[i];
            float x = pts_ptr_x[sample_idx];
            float y = pts_ptr_y[sample_idx];
            float z = pts_ptr_z[sample_idx];
            //            A += Vec3d(x, y, z);
            A0 += x;
            A1 += y;
            A2 += z;
        }

        //        A *= inv_sample_size;
        A0 *= inv_sample_size;
        A1 *= inv_sample_size;
        A2 *= inv_sample_size;

        // Compute the covariance matrix M of the Y[i] = X[i]-A and the
        // right-hand side R of the linear system M*(C-A) = R.
        double M00 = 0, M01 = 0, M02 = 0, M11 = 0, M12 = 0, M22 = 0;
        //        Vec3d R(0, 0, 0);
        double R0 = 0, R1 = 0, R2 = 0;
        for (int i = 0; i < sample_size; ++i)
        {
            int sample_idx = sample[i];
            float x = pts_ptr_x[sample_idx];
            float y = pts_ptr_y[sample_idx];
            float z = pts_ptr_z[sample_idx];

            //            Vec3d Y = Vec3d(x, y, z) - A;
            //            double Y0Y0 = Y[0] * Y[0];
            //            double Y0Y1 = Y[0] * Y[1];
            //            double Y0Y2 = Y[0] * Y[2];
            //            double Y1Y1 = Y[1] * Y[1];
            //            double Y1Y2 = Y[1] * Y[2];
            //            double Y2Y2 = Y[2] * Y[2];
            //            M00 += Y0Y0;
            //            M01 += Y0Y1;
            //            M02 += Y0Y2;
            //            M11 += Y1Y1;
            //            M12 += Y1Y2;
            //            M22 += Y2Y2;
            //            R += (Y0Y0 + Y1Y1 + Y2Y2) * Y;


            double Y0 = x - A0, Y1 = y - A1, Y2 = z - A2;
            double Y0Y0 = Y0 * Y0;
            double Y1Y1 = Y1 * Y1;
            double Y2Y2 = Y2 * Y2;
            M00 += Y0Y0;
            M01 += Y0 * Y1;
            M02 += Y0 * Y2;
            M11 += Y1Y1;
            M12 += Y1 * Y2;
            M22 += Y2Y2;
            double sum_diag = Y0Y0 + Y1Y1 + Y2Y2;
            R0 += sum_diag * Y0;
            R1 += sum_diag * Y1;
            R2 += sum_diag * Y2;
        }
        //        R *= 0.5;
        R0 *= 0.5;
        R1 *= 0.5;
        R2 *= 0.5;

        double center_x, center_y, center_z, radius;

        // Solve the linear system M*(C-A) = R for the center C.
        double cof00 = M11 * M22 - M12 * M12;
        double cof01 = M02 * M12 - M01 * M22;
        double cof02 = M01 * M12 - M02 * M11;
        double det = M00 * cof00 + M01 * cof01 + M02 * cof02;
        if (det != 0)
        {
            double cof11 = M00 * M22 - M02 * M02;
            double cof12 = M01 * M02 - M00 * M12;
            double cof22 = M00 * M11 - M01 * M01;
            //            center_x = A[0] + (cof00 * R[0] + cof01 * R[1] + cof02 * R[2]) / det;
            //            center_y = A[1] + (cof01 * R[0] + cof11 * R[1] + cof12 * R[2]) / det;
            //            center_z = A[2] + (cof02 * R[0] + cof12 * R[1] + cof22 * R[2]) / det;

            center_x = A0 + (cof00 * R0 + cof01 * R1 + cof02 * R2) / det;
            center_y = A1 + (cof01 * R0 + cof11 * R1 + cof12 * R2) / det;
            center_z = A2 + (cof02 * R0 + cof12 * R1 + cof22 * R2) / det;

            double rsqr = 0;
            for (int i = 0; i < sample_size; ++i)
            {
                int sample_idx = sample[i];
                float x = pts_ptr_x[sample_idx];
                float y = pts_ptr_y[sample_idx];
                float z = pts_ptr_z[sample_idx];

                double diff_x = x - center_x, diff_y = y - center_y, diff_z = z - center_z;
                rsqr += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
            }

            rsqr *= inv_sample_size;
            radius = std::sqrt(rsqr);

            double sphere_coeff[4] = {center_x, center_y, center_z, radius};
            models.emplace_back(cv::Mat(1, 4, CV_64F, sphere_coeff).clone());

            return 1;
        }
        else
        {
            return 0;
        }

    }
    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
                  const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}
};

Ptr <SphereModelNonMinimalSolver> SphereModelNonMinimalSolver::create(const Mat &points_)
{
    return makePtr<SphereModelNonMinimalSolverImpl>(points_);
}

}}