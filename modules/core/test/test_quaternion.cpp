// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/quaternion.hpp>
#include <opencv2/ts/cuda_test.hpp>
using namespace cv;
namespace opencv_test{ namespace {
class QuatTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        q1 = {1,2,3,4};
        q2 = {2.5,-2,3.5,4};
        q1Unit = {1 / sqrt(30), sqrt(2) /sqrt(15), sqrt(3) / sqrt(10), 2 * sqrt(2) / sqrt(15)};
        q1Inv = {1.0 / 30, -1.0 / 15, -1.0 / 10, -2.0 / 15};
    }
    double scalar = 2.5;
    double angle = CV_PI;
    double qNorm2 = 2;
    Vec<double, 3> axis{1, 1, 1};
    Vec<double, 3> unAxis{0, 0, 0};
    Vec<double, 3> unitAxis{1.0 / sqrt(3), 1.0 / sqrt(3), 1.0 / sqrt(3)};
    Quatd q3 = Quatd::createFromAngleAxis(angle, axis);
    Quatd q3UnitAxis = Quatd::createFromAngleAxis(angle, unitAxis);
    Quat<double> q3Norm2 = q3 * qNorm2;

    Quat<double> q1Inv;
    Quat<double> q1;
    Quat<double> q2;
    Quat<double> q1Unit;

    Quatd qNull{0, 0, 0, 0};
    Quatd qIdentity{1, 0, 0, 0};
    QuatAssumeType assumeUnit = QUAT_ASSUME_UNIT;

};

TEST_F(QuatTest, constructor){
    Vec<double, 4> coeff{1, 2, 3, 4};
    EXPECT_EQ(Quat<double> (coeff), q1);
    EXPECT_EQ(q3, q3UnitAxis);
    EXPECT_ANY_THROW(Quatd::createFromAngleAxis(angle, unAxis));
    Matx33d R1{
        -1.0 / 3, 2.0 / 3 , 2.0 / 3,
        2.0 / 3 , -1.0 / 3, 2.0 / 3,
        2.0 / 3 , 2.0 / 3 , -1.0 / 3
    };
    Matx33d R2{
        -2.0 / 3, -2.0 / 3, -1.0 / 3,
        -2.0 / 3, 1.0 / 3, 2.0 / 3,
        -1.0 / 3, 2.0 / 3, -2.0 / 3
    };
    Matx33d R3{
        0.818181818181, 0.181818181818, 0.54545455454,
        0.545454545545, -0.54545454545, -0.6363636364,
        0.181818181818, 0.818181818182, -0.5454545455
    };
    Matx33d R4{
        0.818181818181, -0.181818181818, 0.54545455454,
        0.545454545545, 0.54545454545, -0.6363636364,
        -0.181818181818, 0.818181818182, 0.5454545455
    };
    Quatd qMat = Quatd::createFromRotMat(R1);
    Quatd qMat2 = Quatd::createFromRotMat(R2);
    Quatd qMat3 = Quatd::createFromRotMat(R3);
    Quatd qMat4 = Quatd::createFromRotMat(R4);
    EXPECT_EQ(qMat2, Quatd(0, -0.408248290463, 0.816496580927, 0.408248904638));
    EXPECT_EQ(qMat3, Quatd(-0.426401432711,-0.852802865422, -0.213200716355, -0.2132007163));
    EXPECT_EQ(qMat, q3);
    EXPECT_EQ(qMat4, -Quatd(0.852802865422, 0.426401432711221, 0.2132007163556, 0.2132007163));

    Vec3d rot{angle / sqrt(3),angle / sqrt(3), angle / sqrt(3)};
    Quatd rotQuad{0, 1.0 / sqrt(3), 1. / sqrt(3), 1. / sqrt(3)};
    Quatd qRot = Quatd::createFromRvec(rot);
    EXPECT_EQ(qRot, rotQuad);
    EXPECT_EQ(Quatd::createFromRvec(Vec3d(0, 0, 0)), qIdentity);
}

TEST_F(QuatTest, basicfuns){
    Quat<double> q1Conj{1, -2, -3, -4};
    EXPECT_EQ(q3Norm2.normalize(), q3);
    EXPECT_EQ(q1.norm(), sqrt(30));
    EXPECT_EQ(q1.normalize(), q1Unit);
    EXPECT_ANY_THROW(qNull.normalize());
    EXPECT_EQ(q1.conjugate(), q1Conj);
    EXPECT_EQ(q1.inv(), q1Inv);
    EXPECT_EQ(inv(q1), q1Inv);
    EXPECT_EQ(q3.inv(assumeUnit) * q3, qIdentity);
    EXPECT_EQ(q1.inv() * q1, qIdentity);
    EXPECT_ANY_THROW(inv(qNull));
    EXPECT_NO_THROW(q1.at(0));
    EXPECT_ANY_THROW(q1.at(4));

    Matx33d R{
        -2.0 / 3, 2.0 / 15 , 11.0 / 15,
        2.0 / 3 , -1.0 / 3 , 2.0 / 3  ,
        1.0 / 3 , 14.0 / 15, 2.0 / 15
    };
    Matx33d q1RotMat = q1.toRotMat3x3();
    EXPECT_MAT_NEAR(q1RotMat, R, 1e-6);
    Vec3d z_axis{0,0,1};
    Quatd q_unit1 = Quatd::createFromAngleAxis(angle, z_axis);
    Mat pointsA = (Mat_<double>(2, 3) << 1,0,0,1,0,1);
    pointsA = pointsA.t();
    Mat new_point = q_unit1.toRotMat3x3() * pointsA;
    Mat afterRo = (Mat_<double>(3, 2) << -1,-1,0,0,0,1);
    EXPECT_MAT_NEAR(afterRo, new_point, 1e-6);
    EXPECT_ANY_THROW(qNull.toRotVec());
    Vec3d rodVec{CV_PI/sqrt(3), CV_PI/sqrt(3), CV_PI/sqrt(3)};
    Vec3d q3Rod = q3.toRotVec();
    EXPECT_NEAR(q3Rod[0], rodVec[0], 1e-6);
    EXPECT_NEAR(q3Rod[1], rodVec[1], 1e-6);
    EXPECT_NEAR(q3Rod[2], rodVec[2], 1e-6);

    EXPECT_EQ(log(q1Unit, assumeUnit), log(q1Unit));
    EXPECT_EQ(log(qIdentity, assumeUnit), qNull);
    EXPECT_EQ(log(q3), Quatd(0, angle * unitAxis[0] / 2, angle * unitAxis[1] / 2, angle * unitAxis[2] / 2));
    EXPECT_ANY_THROW(log(qNull));
    EXPECT_EQ(log(Quatd(exp(1), 0, 0, 0)), qIdentity);

    EXPECT_EQ(exp(qIdentity), Quatd(exp(1), 0, 0, 0));
    EXPECT_EQ(exp(qNull), qIdentity);
    EXPECT_EQ(exp(Quatd(0, angle * unitAxis[0] / 2, angle * unitAxis[1] / 2, angle * unitAxis[2] / 2)), q3);

    EXPECT_EQ(power(q3, 2.0), Quatd::createFromAngleAxis(2*angle, axis));
    EXPECT_EQ(power(Quatd(0.5, 0.5, 0.5, 0.5), 2.0, assumeUnit), Quatd(-0.5,0.5,0.5,0.5));
    EXPECT_EQ(power(Quatd(0.5, 0.5, 0.5, 0.5), -2.0), Quatd(-0.5,-0.5,-0.5,-0.5));
    EXPECT_EQ(sqrt(q1), power(q1, 0.5));
    EXPECT_EQ(exp(q3 * log(q1)), power(q1, q3));
    EXPECT_EQ(exp(q1 * log(q3)), power(q3, q1, assumeUnit));
    EXPECT_EQ(crossProduct(q1, q3), (q1 * q3 - q3 * q1) / 2);
    EXPECT_EQ(sinh(qNull), qNull);
    EXPECT_EQ(sinh(q1), (exp(q1) - exp(-q1)) / 2);
    EXPECT_EQ(sinh(qIdentity), Quatd(sinh(1), 0, 0, 0));
    EXPECT_EQ(sinh(q1), Quatd(0.73233760604, -0.44820744998, -0.67231117497, -0.8964148999610843));
    EXPECT_EQ(cosh(qNull), qIdentity);
    EXPECT_EQ(cosh(q1), Quatd(0.961585117636, -0.34135217456, -0.51202826184, -0.682704349122));
    EXPECT_EQ(tanh(q1), sinh(q1) * inv(cosh(q1)));
    EXPECT_EQ(sin(qNull), qNull);
    EXPECT_EQ(sin(q1), Quatd(91.78371578403, 21.88648685303, 32.829730279543, 43.772973706058));
    EXPECT_EQ(cos(qNull), qIdentity);
    EXPECT_EQ(cos(q1), Quatd(58.9336461679, -34.0861836904, -51.12927553569, -68.17236738093));
    EXPECT_EQ(tan(q1), sin(q1)/cos(q1));
    EXPECT_EQ(sinh(asinh(q1)), q1);
    Quatd c1 = asinh(sinh(q1));
    EXPECT_EQ(sinh(c1), sinh(q1));
    EXPECT_EQ(cosh(acosh(q1)), q1);
    c1 = acosh(cosh(q1));
    EXPECT_EQ(cosh(c1), cosh(q1));
    EXPECT_EQ(tanh(atanh(q1)), q1);
    c1 = atanh(tanh(q1));
    EXPECT_EQ(tanh(q1), tanh(c1));
    EXPECT_EQ(asin(sin(q1)), q1);
    EXPECT_EQ(sin(asin(q1)), q1);
    EXPECT_EQ(acos(cos(q1)), q1);
    EXPECT_EQ(cos(acos(q1)), q1);
    EXPECT_EQ(atan(tan(q3)), q3);
    EXPECT_EQ(tan(atan(q1)), q1);
}

TEST_F(QuatTest, operator){
    Quatd minusQ{-1, -2, -3, -4};
    Quatd qAdd{3.5, 0, 6.5, 8};
    Quatd qMinus{-1.5, 4, -0.5, 0};
    Quatd qMultq{-20, 1, -5, 27};
    Quatd qMults{2.5, 5.0, 7.5, 10.0};
    Quatd qDvss{1.0 / 2.5, 2.0 / 2.5, 3.0 / 2.5, 4.0 / 2.5};
    Quatd qOrigin(q1);

    EXPECT_EQ(-q1, minusQ);
    EXPECT_EQ(q1 + q2, qAdd);
    EXPECT_EQ(q1 + scalar, Quatd(3.5, 2, 3, 4));
    EXPECT_EQ(scalar + q1, Quatd(3.5, 2, 3, 4));
    EXPECT_EQ(q1 + 2.0, Quatd(3, 2, 3, 4));
    EXPECT_EQ(2.0 + q1, Quatd(3, 2, 3, 4));
    EXPECT_EQ(q1 - q2, qMinus);
    EXPECT_EQ(q1 - scalar, Quatd(-1.5, 2, 3, 4));
    EXPECT_EQ(scalar - q1, Quatd(1.5, -2, -3, -4));
    EXPECT_EQ(q1 - 2.0, Quatd(-1, 2, 3, 4));
    EXPECT_EQ(2.0 - q1, Quatd(1, -2, -3, -4));
    EXPECT_EQ(q1 * q2, qMultq);
    EXPECT_EQ(q1 * scalar, qMults);
    EXPECT_EQ(scalar * q1, qMults);
    EXPECT_EQ(q1 / q1, qIdentity);
    EXPECT_EQ(q1 / scalar, qDvss);
    q1 += q2;
    EXPECT_EQ(q1, qAdd);
    q1 -= q2;
    EXPECT_EQ(q1, qOrigin);
    q1 *= q2;
    EXPECT_EQ(q1, qMultq);
    q1 /= q2;
    EXPECT_EQ(q1, qOrigin);
    q1 *= scalar;
    EXPECT_EQ(q1, qMults);
    q1 /= scalar;
    EXPECT_EQ(q1, qOrigin);
    EXPECT_NO_THROW(q1[0]);
    EXPECT_NO_THROW(q1.at(0));
    EXPECT_ANY_THROW(q1[4]);
    EXPECT_ANY_THROW(q1.at(4));
}

TEST_F(QuatTest, quatAttrs){
    double angleQ1 = 2 * acos(1.0 / sqrt(30));
    Vec3d axis1{0.3713906763541037, 0.557086014, 0.742781352};
    Vec<double, 3> q1axis1 = q1.getAxis();

    EXPECT_EQ(angleQ1, q1.getAngle());
    EXPECT_EQ(angleQ1, q1Unit.getAngle());
    EXPECT_EQ(angleQ1, q1Unit.getAngle(assumeUnit));
    EXPECT_EQ(0, qIdentity.getAngle());
    EXPECT_ANY_THROW(qNull.getAxis());
    EXPECT_NEAR(axis1[0], q1axis1[0], 1e-6);
    EXPECT_NEAR(axis1[1], q1axis1[1], 1e-6);
    EXPECT_NEAR(axis1[2], q1axis1[2], 1e-6);
    EXPECT_NEAR(q3Norm2.norm(), qNorm2, 1e-6);
    EXPECT_EQ(q3Norm2.getAngle(), angle);
    EXPECT_NEAR(axis1[0], axis1[0], 1e-6);
    EXPECT_NEAR(axis1[1], axis1[1], 1e-6);
    EXPECT_NEAR(axis1[2], axis1[2], 1e-6);
}

TEST_F(QuatTest, interpolation){
    Quatd qNoRot = Quatd::createFromAngleAxis(0, axis);
    Quatd qLerpInter(1.0 / 2, sqrt(3) / 6, sqrt(3) / 6, sqrt(3) / 6);
    EXPECT_EQ(Quatd::lerp(qNoRot, q3, 0), qNoRot);
    EXPECT_EQ(Quatd::lerp(qNoRot, q3, 1), q3);
    EXPECT_EQ(Quatd::lerp(qNoRot, q3, 0.5), qLerpInter);
    Quatd q3NrNn2 = qNoRot * qNorm2;
    EXPECT_EQ(Quatd::nlerp(q3NrNn2, q3Norm2, 0), qNoRot);
    EXPECT_EQ(Quatd::nlerp(q3NrNn2, q3Norm2, 1), q3);
    EXPECT_EQ(Quatd::nlerp(q3NrNn2, q3Norm2, 0.5), qLerpInter.normalize());
    EXPECT_EQ(Quatd::nlerp(qNoRot, q3, 0, assumeUnit), qNoRot);
    EXPECT_EQ(Quatd::nlerp(qNoRot, q3, 1, assumeUnit), q3);
    EXPECT_EQ(Quatd::nlerp(qNoRot, q3, 0.5, assumeUnit), qLerpInter.normalize());
    Quatd q3Minus(-q3);
    EXPECT_EQ(Quatd::nlerp(qNoRot, q3, 0.4), -Quatd::nlerp(qNoRot, q3Minus, 0.4));
    EXPECT_EQ(Quatd::slerp(qNoRot, q3, 0, assumeUnit), qNoRot);
    EXPECT_EQ(Quatd::slerp(qNoRot, q3, 1, assumeUnit), q3);
    EXPECT_EQ(Quatd::slerp(qNoRot, q3, 0.5, assumeUnit), -Quatd::nlerp(qNoRot, -q3, 0.5, assumeUnit));
    EXPECT_EQ(Quatd::slerp(qNoRot, q1, 0.5), Quatd(0.76895194, 0.2374325, 0.35614876, 0.47486501));
    EXPECT_EQ(Quatd::slerp(-qNoRot, q1, 0.5), Quatd(0.76895194, 0.2374325, 0.35614876, 0.47486501));
    EXPECT_EQ(Quatd::slerp(qNoRot, -q1, 0.5), -Quatd::slerp(-qNoRot, q1, 0.5));

    Quat<double> tr1 = Quatd::createFromAngleAxis(0, axis);
    Quat<double> tr2 = Quatd::createFromAngleAxis(angle / 2, axis);
    Quat<double> tr3 = Quatd::createFromAngleAxis(angle, axis);
    Quat<double> tr4 = Quatd::createFromAngleAxis(angle, Vec3d{-1/sqrt(2),0,1/(sqrt(2))});
    EXPECT_ANY_THROW(Quatd::spline(qNull, tr1, tr2, tr3, 0));
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr4, 0), tr2);
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr4, 1), tr3);
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr4, 0.6, assumeUnit), Quatd::spline(tr1, tr2, tr3, tr4, 0.6));
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr3, 0.5), Quatd::spline(tr1, -tr2, tr3, tr3, 0.5));
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr3, 0.5), -Quatd::spline(-tr1, -tr2, -tr3, tr3, 0.5));
    EXPECT_EQ(Quatd::spline(tr1, tr2, tr3, tr3, 0.5), Quatd(0.336889853392, 0.543600719487, 0.543600719487, 0.543600719487));
}

} // namespace

}// opencv_test
