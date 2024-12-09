// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp>  // EXPECT_MAT_NEAR

#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/dualquaternion.hpp>

namespace opencv_test{ namespace {

class QuatTest: public ::testing::Test
{
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

TEST_F(QuatTest, constructor)
{
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

TEST_F(QuatTest, basicfuns)
{
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

TEST_F(QuatTest, test_operator)
{
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

TEST_F(QuatTest, quatAttrs)
{
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

TEST_F(QuatTest, interpolation)
{
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

static const Quatd qEuler[24] = {
    Quatd(0.7233214, 0.3919013, 0.2005605, 0.5319728),  //INT_XYZ
    Quatd(0.8223654, 0.0222635, 0.3604221, 0.4396766),  //INT_XZY
    Quatd(0.822365, 0.439677, 0.0222635, 0.360422),     //INT_YXZ
    Quatd(0.723321, 0.531973, 0.391901, 0.20056),       //INT_YZX
    Quatd(0.723321, 0.20056, 0.531973, 0.391901),       //INT_ZXY
    Quatd(0.822365, 0.360422, 0.439677, 0.0222635),     //INT_ZYX
    Quatd(0.653285, 0.65328, 0.369641, -0.0990435),     //INT_XYX
    Quatd(0.653285, 0.65328, 0.0990435, 0.369641),      //INT_XZX
    Quatd(0.653285, 0.369641, 0.65328, 0.0990435),      //INT_YXY
    Quatd(0.653285, -0.0990435, 0.65328, 0.369641),     //INT_YZY
    Quatd(0.653285, 0.369641, -0.0990435, 0.65328),     //INT_ZXZ
    Quatd(0.653285, 0.0990435, 0.369641, 0.65328),      //INT_ZYZ

    Quatd(0.822365, 0.0222635, 0.439677, 0.360422),     //EXT_XYZ
    Quatd(0.723321, 0.391901, 0.531973, 0.20056),       //EXT_XZY
    Quatd(0.723321, 0.20056, 0.391901, 0.531973),       //EXT_YXZ
    Quatd(0.822365, 0.360422, 0.0222635, 0.439677),     //EXT_YZX
    Quatd(0.822365, 0.439677, 0.360422, 0.0222635),     //EXT_ZXY
    Quatd(0.723321, 0.531973, 0.20056, 0.391901),       //EXT_ZYX
    Quatd(0.653285, 0.65328, 0.369641, 0.0990435),      //EXT_XYX
    Quatd(0.653285, 0.65328, -0.0990435, 0.369641),     //EXT_XZX
    Quatd(0.653285, 0.369641, 0.65328, -0.0990435),     //EXT_YXY
    Quatd(0.653285, 0.0990435, 0.65328, 0.369641),      //EXT_YZY
    Quatd(0.653285, 0.369641, 0.0990435, 0.65328),      //EXT_ZXZ
    Quatd(0.653285, -0.0990435, 0.369641, 0.65328)      //EXT_ZYZ
};

TEST_F(QuatTest, EulerAngles)
{
    Vec3d test_angle = {0.523598, 0.78539, 1.04719};
    for (QuatEnum::EulerAnglesType i = QuatEnum::EulerAnglesType::INT_XYZ; i <= QuatEnum::EulerAnglesType::EXT_ZYZ; i = (QuatEnum::EulerAnglesType)(i + 1))
    {
        SCOPED_TRACE(cv::format("EulerAnglesType=%d", i));
        Quatd q = Quatd::createFromEulerAngles(test_angle, i);
        EXPECT_EQ(q, qEuler[i]);
        Vec3d Euler_Angles = q.toEulerAngles(i);
        EXPECT_NEAR(Euler_Angles[0], test_angle[0], 1e-6);
        EXPECT_NEAR(Euler_Angles[1], test_angle[1], 1e-6);
        EXPECT_NEAR(Euler_Angles[2], test_angle[2], 1e-6);
    }
    Quatd qEuler0 = {0, 0, 0, 0};
    EXPECT_ANY_THROW(qEuler0.toEulerAngles(QuatEnum::INT_XYZ));

    Quatd qEulerLock1 = {0.5612665, 0.43042, 0.5607083, 0.4304935};
    Vec3d test_angle_lock1 = {1.3089878, CV_PI * 0.5, 0};
    Vec3d Euler_Angles_solute_1 = qEulerLock1.toEulerAngles(QuatEnum::INT_XYZ);
    EXPECT_NEAR(Euler_Angles_solute_1[0], test_angle_lock1[0], 1e-6);
    EXPECT_NEAR(Euler_Angles_solute_1[1], test_angle_lock1[1], 1e-6);
    EXPECT_NEAR(Euler_Angles_solute_1[2], test_angle_lock1[2], 1e-6);

    Quatd qEulerLock2 = {0.7010574, 0.0922963, 0.7010573, -0.0922961};
    Vec3d test_angle_lock2 = {-0.2618, CV_PI * 0.5, 0};
    Vec3d Euler_Angles_solute_2 = qEulerLock2.toEulerAngles(QuatEnum::INT_ZYX);
    EXPECT_NEAR(Euler_Angles_solute_2[0], test_angle_lock2[0], 1e-6);
    EXPECT_NEAR(Euler_Angles_solute_2[1], test_angle_lock2[1], 1e-6);
    EXPECT_NEAR(Euler_Angles_solute_2[2], test_angle_lock2[2], 1e-6);

    Vec3d test_angle6 = {CV_PI * 0.25, CV_PI * 0.5, CV_PI * 0.25};
    Vec3d test_angle7 = {CV_PI * 0.5, CV_PI * 0.5, 0};
    EXPECT_EQ(Quatd::createFromEulerAngles(test_angle6, QuatEnum::INT_ZXY), Quatd::createFromEulerAngles(test_angle7, QuatEnum::INT_ZXY));
}



class DualQuatTest: public ::testing::Test
{
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

TEST_F(DualQuatTest, constructor)
{
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

TEST_F(DualQuatTest, test_operator)
{
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

TEST_F(DualQuatTest, basic_ops)
{
    EXPECT_EQ(dq1.getRealPart(), Quatd(1, 2, 3, 4));
    EXPECT_EQ(dq1.getDualPart(), Quatd(5, 6, 7, 8));
    EXPECT_EQ((dq1 * dq2).conjugate(), conjugate(dq1 * dq2));
    EXPECT_EQ(dq1.conjugate(), DualQuatd::createFromQuat(dq1.getRealPart().conjugate(), dq1.getDualPart().conjugate()));
    EXPECT_EQ((dq2 * dq1).conjugate(), dq1.conjugate() * dq2.conjugate());
    EXPECT_EQ(dq1.conjugate() * dq1, dq1.norm() * dq1.norm());
    EXPECT_EQ(dq1.conjugate() * dq1, dq1.norm().power(2.0));
    EXPECT_EQ(dualNumber2.power(2.0), DualQuatd(16, 0, 0, 0, 40.8, 0, 0, 0));
    EXPECT_EQ(dq1.power(2.0), (2.0 * dq1.log()).exp());
    EXPECT_EQ(power(dq1, 2.0), (exp(2.0 * log(dq1))));
    EXPECT_EQ(dq2.power(3.0 / 2, QUAT_ASSUME_UNIT).power(4.0 / 3, QUAT_ASSUME_UNIT), dq2 * dq2);
    EXPECT_EQ(dq2.power(-0.5).power(2.0), dq2.inv());
    EXPECT_EQ(power(dq1, dq2), exp(dq2 * log(dq1)));
    EXPECT_EQ(power(dq2, dq1, QUAT_ASSUME_UNIT), exp(dq1 * log(dq2)));
    EXPECT_EQ((dq2.norm() * dq1).power(2.0), dq1.power(2.0) * dq2.norm().power(2.0));
    DualQuatd q1norm = dq1.normalize();
    EXPECT_EQ(dq2.norm(), dqIdentity);
    EXPECT_NEAR(q1norm.getRealPart().norm(), 1, 1e-6);
    EXPECT_NEAR(q1norm.getRealPart().dot(q1norm.getDualPart()), 0, 1e-6);
    EXPECT_NEAR(dq1.getRotation().norm(), 1, 1e-6);
    EXPECT_NEAR(dq2.getRotation(QUAT_ASSUME_UNIT).norm(), 1, 1e-6);
    EXPECT_NEAR(dq2.getRotation(QUAT_ASSUME_UNIT).norm(), 1, 1e-6);
    EXPECT_MAT_NEAR(Mat(dq2.getTranslation()), Mat(trans), 1e-6);
    EXPECT_MAT_NEAR(Mat(q1norm.getTranslation(QUAT_ASSUME_UNIT)), Mat(dq1.getTranslation()), 1e-6);
    EXPECT_EQ(dq2.getTranslation(), dq2.getTranslation(QUAT_ASSUME_UNIT));
    EXPECT_EQ(dq1.inv() * dq1, dqIdentity);
    EXPECT_EQ(inv(dq1) * dq1, dqIdentity);
    EXPECT_EQ(dq2.inv(QUAT_ASSUME_UNIT) * dq2, dqIdentity);
    EXPECT_EQ(inv(dq2, QUAT_ASSUME_UNIT) * dq2, dqIdentity);
    EXPECT_EQ(dq2.inv(), dq2.conjugate());
    EXPECT_EQ(dqIdentity.inv(), dqIdentity);
    EXPECT_ANY_THROW(dqAllZero.inv());
    EXPECT_EQ(dqAllZero.exp(), dqIdentity);
    EXPECT_EQ(exp(dqAllZero), dqIdentity);
    EXPECT_ANY_THROW(log(dqAllZero));
    EXPECT_EQ(log(dqIdentity), dqAllZero);
    EXPECT_EQ(dqIdentity.log(), dqAllZero);
    EXPECT_EQ(dualNumber1 * dualNumber2, dualNumber2 * dualNumber1);
    EXPECT_EQ(dualNumber2.exp().log(), dualNumber2);
    EXPECT_EQ(dq2.log(QUAT_ASSUME_UNIT).exp(), dq2);
    EXPECT_EQ(exp(log(dq2, QUAT_ASSUME_UNIT)), dq2);
    EXPECT_EQ(dqIdentity.log(QUAT_ASSUME_UNIT).exp(), dqIdentity);
    EXPECT_EQ(dq1.log().exp(), dq1);
    EXPECT_EQ(dqTrans.log().exp(), dqTrans);
    EXPECT_MAT_NEAR(q1norm.toMat(QUAT_ASSUME_UNIT), dq1.toMat(), 1e-6);
    Matx44d R1 = dq2.toMat();
    Mat point = (Mat_<double>(4, 1) << 3, 0, 0, 1);
    Mat new_point = R1 * point;
    Mat after = (Mat_<double>(4, 1) << 0, 3, 5 ,1);
    EXPECT_MAT_NEAR(new_point,  after, 1e-6);
    Vec<double, 8> vec = dq1.toVec();
    EXPECT_EQ(DualQuatd(vec), dq1);
    Affine3d afd = q1norm.toAffine3(QUAT_ASSUME_UNIT);
    EXPECT_MAT_NEAR(Mat(afd.translation()), Mat(q1norm.getTranslation(QUAT_ASSUME_UNIT)), 1e-6);
    Affine3d dq1_afd = dq1.toAffine3();
    EXPECT_MAT_NEAR(dq1_afd.matrix, afd.matrix, 1e-6);
    EXPECT_ANY_THROW(dqAllZero.toAffine3());
}

TEST_F(DualQuatTest, interpolation)
{
    DualQuatd dq = DualQuatd::createFromAngleAxisTrans(8 * CV_PI / 5, Vec3d{0, 0, 1}, Vec3d{0, 0, 10});
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq, 0.5), DualQuatd::sclerp(-dqIdentity, dq, 0.5, false));
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq, 0), -dqIdentity);
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq2, 1), dq2);
    EXPECT_EQ(DualQuatd::sclerp(dqIdentity, dq2, 0.4, false, QUAT_ASSUME_UNIT), DualQuatd(0.91354546, 0.23482951, 0.23482951, 0.23482951, -0.23482951, -0.47824988, 0.69589767, 0.69589767));
    EXPECT_EQ(DualQuatd::dqblend(dqIdentity, dq1.normalize(), 0.2, QUAT_ASSUME_UNIT), DualQuatd::dqblend(dqIdentity, -dq1, 0.2));
    EXPECT_EQ(DualQuatd::dqblend(dqIdentity, dq2, 0.4), DualQuatd(0.91766294, 0.22941573, 0.22941573, 0.22941573, -0.21130397, -0.48298049, 0.66409818, 0.66409818));
    DualQuatd gdb = DualQuatd::gdqblend(Vec<DualQuatd, 3>{dqIdentity, dq, dq2}, Vec3d{0.4, 0, 0.6}, QUAT_ASSUME_UNIT);
    EXPECT_EQ(gdb, DualQuatd::dqblend(dqIdentity, dq2, 0.6));
    EXPECT_ANY_THROW(DualQuatd::gdqblend(Vec<DualQuatd, 1>{dq2}, Vec2d{0.5, 0.5}));
    Mat gdqb_d(1, 2, CV_64FC(7));
    gdqb_d.at<Vec<double, 7>>(0, 0) = Vec<double, 7>{1,2,3,4,5,6,7};
    gdqb_d.at<Vec<double, 7>>(0, 1) = Vec<double, 7>{1,2,3,4,5,6,7};
    EXPECT_ANY_THROW(DualQuatd::gdqblend(gdqb_d, Vec2d{0.5, 0.5}));
    Mat gdqb_f(1, 2, CV_32FC(8));
    gdqb_f.at<Vec<float, 8>>(0, 0) = Vec<float, 8>{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    gdqb_f.at<Vec<float, 8>>(0, 1) = Vec<float, 8>{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    EXPECT_ANY_THROW(DualQuatd::gdqblend(gdqb_f, Vec2d{0.5, 0.5}));
    EXPECT_ANY_THROW(DualQuatd::gdqblend(Vec<DualQuatd, 3>{dqIdentity, dq, dq2}, Vec3f{0.4f, 0.f, 0.6f}, QUAT_ASSUME_UNIT));
    EXPECT_EQ(gdb, DualQuatd::gdqblend(Vec<DualQuatd, 3>{dqIdentity, dq * dualNumber1, -dq2}, Vec3d{0.4, 0, 0.6}));
}


}} // namespace
