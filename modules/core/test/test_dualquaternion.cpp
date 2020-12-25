// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/dualquaternion.hpp>
#include <opencv2/ts/cuda_test.hpp>
using namespace cv;
namespace opencv_test{ namespace {
class DualQuatTest: public ::testing::Test {
protected:
    double scalar = 2.5;
    double angle = CV_PI;
    Vec<double, 3> axis{1, 1, 1};
    Vec<double, 3> unAxis{0, 0, 0};
    Vec<double, 3> unitAxis{1.0 / sqrt(3), 1.0 / sqrt(3), 1.0 / sqrt(3)};
    DualQuatd dq1{1, 2, 3, 4, 5, 6, 7, 8};
    Vec3d trans{0, 0, 5};
    double rotation_angle = 2.0 / 3 * CV_PI;
    DualQuatd dq2 = DualQuatd::createFromAngleAxisTrans(rotation_angle, axis, trans);
    DualQuatd dqAllOne{1, 1, 1, 1, 1, 1, 1, 1};
    DualQuatd dqAllZero{0, 0, 0, 0, 0, 0, 0, 0};
    DualQuatd dqIdentity{1, 0, 0, 0, 0, 0, 0, 0};
    DualQuatd dqTrans{1, 0, 0, 0, 0, 2, 3, 4};
    DualQuatd dqOnlyTrans{0, 0, 0, 0, 0, 2, 3, 4};
    DualQuatd dualNumber1{-3,0,0,0,-31.1,0,0,0};
    DualQuatd dualNumber2{4,0,0,0,5.1,0,0,0};
};

TEST_F(DualQuatTest, constructor){
    EXPECT_EQ(dq1, DualQuatd::createFromQuat(Quatd(1, 2, 3, 4), Quatd(5, 6, 7, 8)));
    EXPECT_EQ(dq2 * dq2.conjugate(), dqIdentity);
    EXPECT_NEAR(dq2.getRotation(QUAT_ASSUME_UNIT).norm(), 1, 1e-6);
    EXPECT_NEAR(dq2.getRealPart().dot(dq2.getDualPart()), 0, 1e-6);
    EXPECT_MAT_NEAR(dq2.getTranslation(QUAT_ASSUME_UNIT), trans, 1e-6);
    DualQuatd q_conj = DualQuatd::createFromQuat(dq2.getRealPart().conjugate(), -dq2.getDualPart().conjugate());
    DualQuatd q{1,0,0,0,0,3,0,0};
    EXPECT_EQ(dq2 * q * q_conj, DualQuatd(1,0,0,0,0,0,3,5));
    Matx44d R1 = dq2.toMat();
    DualQuatd dq3 = DualQuatd::createFromMat(R1);
    EXPECT_EQ(dq3, dq2);
    axis = axis / std::sqrt(axis.dot(axis));
    Vec3d moment = 1.0 / 2 * (trans.cross(axis) + axis.cross(trans.cross(axis)) *
                              std::cos(rotation_angle / 2) / std::sin(rotation_angle / 2));
    double d = trans.dot(axis);
    DualQuatd dq4 = DualQuatd::createFromPitch(rotation_angle, d, axis, moment);
    EXPECT_EQ(dq4, dq3);
    EXPECT_EQ(dq2, DualQuatd::createFromAffine3(dq2.toAffine3()));
    EXPECT_EQ(dq1.normalize(), DualQuatd::createFromAffine3(dq1.toAffine3()));
}

TEST_F(DualQuatTest, operator){
    DualQuatd dq_origin{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(dq1 - dqAllOne, DualQuatd(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_EQ(-dq1, DualQuatd(-1, -2, -3, -4, -5, -6, -7, -8));
    EXPECT_EQ(dq1 + dqAllOne, DualQuatd(2, 3, 4, 5, 6, 7, 8, 9));
    EXPECT_EQ(dq1 / dq1, dqIdentity);
    DualQuatd dq3{-4, 1, 3, 2, -15.5, 0, -3, 8.5};
    EXPECT_EQ(dq1 * dq2, dq3);
    EXPECT_EQ(dq3 / dq2, dq1);
    DualQuatd dq12{2, 4, 6, 8, 10, 12, 14, 16};
    EXPECT_EQ(dq1 * 2.0, dq12);
    EXPECT_EQ(2.0 * dq1, dq12);
    EXPECT_EQ(dq1 - 1.0, DualQuatd(0, 2, 3, 4, 5, 6, 7, 8));
    EXPECT_EQ(1.0 - dq1, DualQuatd(0, -2, -3, -4, -5, -6, -7, -8));
    EXPECT_EQ(dq1 + 1.0, DualQuatd(2, 2, 3, 4, 5, 6, 7, 8));
    EXPECT_EQ(1.0 + dq1, DualQuatd(2, 2, 3, 4, 5, 6, 7, 8));
    dq1 += dq2;
    EXPECT_EQ(dq1, dq_origin + dq2);
    dq1 -= dq2;
    EXPECT_EQ(dq1, dq_origin);
    dq1 *= dq2;
    EXPECT_EQ(dq1, dq_origin * dq2);
    dq1 /= dq2;
    EXPECT_EQ(dq1, dq_origin);
}

TEST_F(DualQuatTest, basic_ops){
    EXPECT_EQ(dq1.getRealPart(), Quatd(1, 2, 3, 4));
    EXPECT_EQ(dq1.getDualPart(), Quatd(5, 6, 7, 8));
    EXPECT_EQ(dq1.conjugate(), DualQuatd::createFromQuat(dq1.getRealPart().conjugate(), dq1.getDualPart().conjugate()));
    EXPECT_EQ((dq2 * dq1).conjugate(), dq1.conjugate() * dq2.conjugate());
    EXPECT_EQ(dq1.conjugate() * dq1, dq1.norm() * dq1.norm());
    EXPECT_EQ(dq1.conjugate() * dq1, dq1.norm().power(2.0));
    EXPECT_EQ(dualNumber2.power(2.0), DualQuatd(16, 0, 0, 0, 40.8, 0, 0, 0));
    EXPECT_EQ(dq1.power(2.0), (2.0 * dq1.log()).exp());
    EXPECT_EQ(dq2.power(3.0 / 2, QUAT_ASSUME_UNIT).power(4.0 / 3, QUAT_ASSUME_UNIT), dq2 * dq2);
    EXPECT_EQ(dq2.power(-0.5).power(2.0), dq2.inv());
    EXPECT_EQ((dq2.norm() * dq1).power(2.0), dq1.power(2.0) * dq2.norm().power(2.0));
    DualQuatd q1norm = dq1.normalize();
    EXPECT_EQ(dq2.norm(), dqIdentity);
    EXPECT_NEAR(q1norm.getRealPart().norm(), 1, 1e-6);
    EXPECT_NEAR(q1norm.getRealPart().dot(q1norm.getDualPart()), 0, 1e-6);
    EXPECT_NEAR(dq1.getRotation().norm(), 1, 1e-6);
    EXPECT_EQ(dq1.inv() * dq1, dqIdentity);
    EXPECT_EQ(dq2.inv(QUAT_ASSUME_UNIT) * dq2, dqIdentity);
    EXPECT_EQ(dq2.inv(), dq2.conjugate());
    EXPECT_EQ(dqIdentity.inv(), dqIdentity);
    EXPECT_ANY_THROW(dqAllZero.inv());
    EXPECT_EQ(dqAllZero.exp(), dqIdentity);
    EXPECT_EQ(dqIdentity.log(), dqAllZero);
    EXPECT_EQ(dualNumber1 * dualNumber2, dualNumber2 * dualNumber1);
    EXPECT_EQ(dualNumber2.exp().log(), dualNumber2);
    EXPECT_EQ(dq2.log(QUAT_ASSUME_UNIT).exp(), dq2);
    EXPECT_EQ(dqIdentity.log(QUAT_ASSUME_UNIT).exp(), dqIdentity);
    EXPECT_EQ(dq1.log().exp(), dq1);
    EXPECT_EQ(dqTrans.log().exp(), dqTrans);
    Matx44d R1 = dq2.toMat();
    Mat point = (Mat_<double>(4, 1) << 3, 0, 0, 1);
    Mat new_point = R1 * point;
    Mat after = (Mat_<double>(4, 1) << 0, 3, 5 ,1);
    EXPECT_MAT_NEAR(new_point,  after, 1e-6);
    Vec<double, 8> vec = dq1.toVec();
    EXPECT_EQ(DualQuatd(vec), dq1);
    Affine3d afd = dq2.toAffine3();
    EXPECT_MAT_NEAR(Mat(afd.translation()), Mat(dq2.getTranslation()), 1e-6);
    EXPECT_ANY_THROW(dqAllZero.toAffine3());
}

TEST_F(DualQuatTest, dqs) {
    std::vector<Vec3d> origin{{1, 1, 1},{2, 1, 3},{1, 2, 3}};
    std::vector<Vec3d> normal{{1, 1, 1},{2, 1, 3},{1, 2, 3}};
    std::vector<Vec<double, 3>> normal_out;
    std::vector<Vec<double, 3>> out;
    double angle1 = CV_PI / 2.0;
    double angle2 = CV_PI;
    Vec3d axis2(0, 0, 1);
    Vec3d axis1(1, 1, 1);
    Vec3d tran1(2, 3, 4);
    Vec3d tran2(0, 0, 0);
    DualQuatd dqs1 = DualQuatd::createFromAngleAxisTrans(angle1, axis1, tran1);
    DualQuatd dqs2 = DualQuatd::createFromAngleAxisTrans(angle2, axis2, tran2);
    DualQuatd dqs3 = DualQuatd::createFromAngleAxisTrans(angle1, axis2, tran1);
    std::vector<DualQuatd> blend{dqs1, dqs2, dqs3};
    std::vector< std::vector<double> > weights{std::vector<double>{0.1, 0.9}, std::vector<double>{}, std::vector<double>{0.5, 0.3, 0.2}};
    std::vector < std::vector<int> > id{std::vector<int>{0, 2}, std::vector<int>{}, std::vector<int>{2, 1, 0}};
    DualQuatd::dqs(origin, normal, out, normal_out, blend, weights, id);
    Mat normal_mat(normal_out), out_mat(out);
    normal_mat = normal_mat.reshape(1, 3);
    out_mat = out_mat.reshape(1, 3);
    Mat normal_result = (Mat_<double>(3, 3) << -0.83163416, 1.0395426, 1.1080325, 2, 1, 3, -1.4042857, 0.33521342, 3.4519);
    Mat out_result = (Mat_<double>(3, 3) << 1.1683658, 4.0395427, 5.1080332, 2, 1, 3, -0.63677007, 2.7560391, 6.5008297);
    EXPECT_MAT_NEAR(normal_result, normal_mat, 1e-6);
    EXPECT_MAT_NEAR(out_mat, out_result, 1e-6);

    std::vector<Vec3d> normal_out_anu;
    std::vector<Vec3d> out_anu;
    DualQuatd::dqs(origin, normal, out_anu, normal_out_anu, blend, weights, id, QUAT_ASSUME_NOT_UNIT);
    EXPECT_EQ(out_anu, out);
}

TEST_F(DualQuatTest, interpolation) {
    DualQuatd dq = DualQuatd::createFromAngleAxisTrans(8 * CV_PI / 5, Vec3d{0, 0, 1}, Vec3d{0, 0, 10});
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq, 0.5), DualQuatd::sclerp(-dqIdentity, dq, 0.5, false));
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq, 0), -dqIdentity);
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq2, 1), dq2);
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq2, 0.4, false, QUAT_ASSUME_UNIT), DualQuatd(0.91354546, 0.23482951, 0.23482951, 0.23482951, -0.23482951, -0.47824988, 0.69589767, 0.69589767));
    EXPECT_EQ(DualQuatd::dqblend(dqIdentity, dq1.normalize(), 0.2, QUAT_ASSUME_UNIT), DualQuatd::dqblend(dqIdentity, -dq1, 0.2));
    EXPECT_EQ(DualQuatd::dqblend(dqIdentity, dq2, 0.4), DualQuatd(0.91766294, 0.22941573, 0.22941573, 0.22941573, -0.21130397, -0.48298049, 0.66409818, 0.66409818));
    DualQuatd gdb = DualQuatd::gdqblend(std::vector<DualQuatd>{dqIdentity, dq, dq2}, std::vector<double>{0.4, 0, 0.6}, QUAT_ASSUME_UNIT);
    EXPECT_EQ(gdb, DualQuatd::dqblend(dqIdentity, dq2, 0.6));
    EXPECT_EQ(DualQuatd::gdqblend(std::vector<DualQuatd>{}, std::vector<double>{}), dqIdentity);
    EXPECT_EQ(gdb, DualQuatd::gdqblend(std::vector<DualQuatd>{dqIdentity, dq * dualNumber1, -dq2}, std::vector<double>{0.4, 0, 0.6}));
}

} // namespace

}// opencv_test
