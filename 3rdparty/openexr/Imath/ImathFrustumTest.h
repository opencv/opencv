//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A viewing frustum class
//
// This file contains algorithms applied to or in conjunction with
// Frustum visibility testing (Imath::Frustum).
//
// Methods for frustum-based rejection of primitives are contained here.
//

#ifndef INCLUDED_IMATHFRUSTUMTEST_H
#define INCLUDED_IMATHFRUSTUMTEST_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathBox.h"
#include "ImathFrustum.h"
#include "ImathMatrix.h"
#include "ImathSphere.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// template class FrustumTest<T>
///
/// This is a helper class, designed to accelerate the case
/// where many tests are made against the same frustum.
/// That's a really common case.
///
/// The acceleration is achieved by pre-computing the planes of
/// the frustum, along with the ablsolute values of the plane normals.
///
/// How to use this
///
/// Given that you already have:
///    Imath::Frustum   myFrustum
///    Imath::Matrix44  myCameraWorldMatrix
///
/// First, make a frustum test object:
///    FrustumTest myFrustumTest(myFrustum, myCameraWorldMatrix)
///
/// Whenever the camera or frustum changes, call:
///    myFrustumTest.setFrustum(myFrustum, myCameraWorldMatrix)
///
/// For each object you want to test for visibility, call:
///    myFrustumTest.isVisible(myBox)
///    myFrustumTest.isVisible(mySphere)
///    myFrustumTest.isVisible(myVec3)
///    myFrustumTest.completelyContains(myBox)
///    myFrustumTest.completelyContains(mySphere)
///
/// Explanation of how it works
///
/// We store six world-space Frustum planes (nx, ny, nz, offset)
///
/// Points: To test a Vec3 for visibility, test it against each plane
///         using the normal (v dot n - offset) method. (the result is exact)
///
/// BBoxes: To test an axis-aligned bbox, test the center against each plane
///         using the normal (v dot n - offset) method, but offset by the
///         box extents dot the abs of the plane normal. (the result is NOT
///         exact, but will not return false-negatives.)
///
/// Spheres: To test a sphere, test the center against each plane
///         using the normal (v dot n - offset) method, but offset by the
///         sphere's radius. (the result is NOT exact, but will not return
///         false-negatives.)
///
///
/// SPECIAL NOTE: "Where are the dot products?"
///     Actual dot products are currently slow for most SIMD architectures.
///     In order to keep this code optimization-ready, the dot products
///     are all performed using vector adds and multipies.
///
///     In order to do this, the plane equations are stored in "transpose"
///     form, with the X components grouped into an X vector, etc.
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE FrustumTest
{
  public:
    /// @{
    /// @name Constructors

    /// Initialize camera matrix to identity
    FrustumTest() IMATH_NOEXCEPT
    {
        Frustum<T> frust;
        Matrix44<T> cameraMat;
        cameraMat.makeIdentity();
        setFrustum (frust, cameraMat);
    }

    /// Initialize to a given frustum and camera matrix.
    FrustumTest (const Frustum<T>& frustum, const Matrix44<T>& cameraMat) IMATH_NOEXCEPT
    {
        setFrustum (frustum, cameraMat);
    }

    /// @}

    /// @{
    /// @name Set Value
    
    /// Update the frustum test with a new frustum and matrix.
    /// This should usually be called just once per frame, or however
    /// often the camera moves.
    void setFrustum (const Frustum<T>& frustum, const Matrix44<T>& cameraMat) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Query
    
    /// Return true if any part of the sphere is inside the frustum.
    /// The result MAY return close false-positives, but not false-negatives.
    bool isVisible (const Sphere3<T>& sphere) const IMATH_NOEXCEPT;

    /// Return true if any part of the box is inside the frustum.
    /// The result MAY return close false-positives, but not false-negatives.
    bool isVisible (const Box<Vec3<T>>& box) const IMATH_NOEXCEPT;

    /// Return true if the point is inside the frustum.
    bool isVisible (const Vec3<T>& vec) const IMATH_NOEXCEPT;

    /// Return true if every part of the sphere is inside the frustum.
    /// The result MAY return close false-negatives, but not false-positives.
    bool completelyContains (const Sphere3<T>& sphere) const IMATH_NOEXCEPT;

    /// Return true if every part of the box is inside the frustum.
    /// The result MAY return close false-negatives, but not false-positives.
    bool completelyContains (const Box<Vec3<T>>& box) const IMATH_NOEXCEPT;

    /// Return the camera matrix (primarily for debugging)
    IMATH_INTERNAL_NAMESPACE::Matrix44<T> cameraMat() const IMATH_NOEXCEPT { return cameraMatrix; }

    /// Return the viewing frustum (primarily for debugging)
    IMATH_INTERNAL_NAMESPACE::Frustum<T> currentFrustum() const IMATH_NOEXCEPT { return currFrustum; }

    /// @}
    
  protected:

    // To understand why the planes are stored this way, see
    // the SPECIAL NOTE above.

    /// @cond Doxygen_Suppress

    Vec3<T> planeNormX[2]; // The X components from 6 plane equations
    Vec3<T> planeNormY[2]; // The Y components from 6 plane equations
    Vec3<T> planeNormZ[2]; // The Z components from 6 plane equations

    Vec3<T> planeOffsetVec[2]; // The distance offsets from 6 plane equations

    // The absolute values are stored to assist with bounding box tests.
    Vec3<T> planeNormAbsX[2]; // The abs(X) components from 6 plane equations
    Vec3<T> planeNormAbsY[2]; // The abs(X) components from 6 plane equations
    Vec3<T> planeNormAbsZ[2]; // The abs(X) components from 6 plane equations

    // These are kept primarily for debugging tools.
    Frustum<T> currFrustum;
    Matrix44<T> cameraMatrix;

    /// @endcond
};

template <class T>
void
FrustumTest<T>::setFrustum (const Frustum<T>& frustum, const Matrix44<T>& cameraMat) IMATH_NOEXCEPT
{
    Plane3<T> frustumPlanes[6];
    frustum.planes (frustumPlanes, cameraMat);

    // Here's where we effectively transpose the plane equations.
    // We stuff all six X's into the two planeNormX vectors, etc.
    for (int i = 0; i < 2; ++i)
    {
        int index = i * 3;

        planeNormX[i] = Vec3<T> (frustumPlanes[index + 0].normal.x,
                                 frustumPlanes[index + 1].normal.x,
                                 frustumPlanes[index + 2].normal.x);
        planeNormY[i] = Vec3<T> (frustumPlanes[index + 0].normal.y,
                                 frustumPlanes[index + 1].normal.y,
                                 frustumPlanes[index + 2].normal.y);
        planeNormZ[i] = Vec3<T> (frustumPlanes[index + 0].normal.z,
                                 frustumPlanes[index + 1].normal.z,
                                 frustumPlanes[index + 2].normal.z);

        planeNormAbsX[i] = Vec3<T> (std::abs (planeNormX[i].x),
                                    std::abs (planeNormX[i].y),
                                    std::abs (planeNormX[i].z));
        planeNormAbsY[i] = Vec3<T> (std::abs (planeNormY[i].x),
                                    std::abs (planeNormY[i].y),
                                    std::abs (planeNormY[i].z));
        planeNormAbsZ[i] = Vec3<T> (std::abs (planeNormZ[i].x),
                                    std::abs (planeNormZ[i].y),
                                    std::abs (planeNormZ[i].z));

        planeOffsetVec[i] = Vec3<T> (frustumPlanes[index + 0].distance,
                                     frustumPlanes[index + 1].distance,
                                     frustumPlanes[index + 2].distance);
    }
    currFrustum  = frustum;
    cameraMatrix = cameraMat;
}

template <typename T>
bool
FrustumTest<T>::isVisible (const Sphere3<T>& sphere) const IMATH_NOEXCEPT
{
    Vec3<T> center    = sphere.center;
    Vec3<T> radiusVec = Vec3<T> (sphere.radius, sphere.radius, sphere.radius);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0 = planeNormX[0] * center.x + planeNormY[0] * center.y + planeNormZ[0] * center.z -
                 radiusVec - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1 = planeNormX[1] * center.x + planeNormY[1] * center.y + planeNormZ[1] * center.z -
                 radiusVec - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

template <typename T>
bool
FrustumTest<T>::completelyContains (const Sphere3<T>& sphere) const IMATH_NOEXCEPT
{
    Vec3<T> center    = sphere.center;
    Vec3<T> radiusVec = Vec3<T> (sphere.radius, sphere.radius, sphere.radius);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0 = planeNormX[0] * center.x + planeNormY[0] * center.y + planeNormZ[0] * center.z +
                 radiusVec - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1 = planeNormX[1] * center.x + planeNormY[1] * center.y + planeNormZ[1] * center.z +
                 radiusVec - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

template <typename T>
bool
FrustumTest<T>::isVisible (const Box<Vec3<T>>& box) const IMATH_NOEXCEPT
{
    if (box.isEmpty())
        return false;

    Vec3<T> center = (box.min + box.max) / 2;
    Vec3<T> extent = (box.max - center);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0 = planeNormX[0] * center.x + planeNormY[0] * center.y + planeNormZ[0] * center.z -
                 planeNormAbsX[0] * extent.x - planeNormAbsY[0] * extent.y -
                 planeNormAbsZ[0] * extent.z - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1 = planeNormX[1] * center.x + planeNormY[1] * center.y + planeNormZ[1] * center.z -
                 planeNormAbsX[1] * extent.x - planeNormAbsY[1] * extent.y -
                 planeNormAbsZ[1] * extent.z - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

template <typename T>
bool
FrustumTest<T>::completelyContains (const Box<Vec3<T>>& box) const IMATH_NOEXCEPT
{
    if (box.isEmpty())
        return false;

    Vec3<T> center = (box.min + box.max) / 2;
    Vec3<T> extent = (box.max - center);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0 = planeNormX[0] * center.x + planeNormY[0] * center.y + planeNormZ[0] * center.z +
                 planeNormAbsX[0] * extent.x + planeNormAbsY[0] * extent.y +
                 planeNormAbsZ[0] * extent.z - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1 = planeNormX[1] * center.x + planeNormY[1] * center.y + planeNormZ[1] * center.z +
                 planeNormAbsX[1] * extent.x + planeNormAbsY[1] * extent.y +
                 planeNormAbsZ[1] * extent.z - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

template <typename T>
bool
FrustumTest<T>::isVisible (const Vec3<T>& vec) const IMATH_NOEXCEPT
{
    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0 = (planeNormX[0] * vec.x) + (planeNormY[0] * vec.y) + (planeNormZ[0] * vec.z) -
                 planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1 = (planeNormX[1] * vec.x) + (planeNormY[1] * vec.y) + (planeNormZ[1] * vec.z) -
                 planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

/// FrustymTest of type float
typedef FrustumTest<float> FrustumTestf;

/// FrustymTest of type double
typedef FrustumTest<double> FrustumTestd;

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHFRUSTUMTEST_H
