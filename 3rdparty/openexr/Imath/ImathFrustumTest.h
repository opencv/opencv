///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2011, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#ifndef INCLUDED_IMATHFRUSTUMTEST_H
#define INCLUDED_IMATHFRUSTUMTEST_H

//-------------------------------------------------------------------------
//
//      This file contains algorithms applied to or in conjunction with
//	Frustum visibility testing (Imath::Frustum).
//
//	Methods for frustum-based rejection of primitives are contained here.
//
//-------------------------------------------------------------------------

#include "ImathFrustum.h"
#include "ImathBox.h"
#include "ImathSphere.h"
#include "ImathMatrix.h"
#include "ImathVec.h"

namespace Imath {

/////////////////////////////////////////////////////////////////
// FrustumTest
//
//	template class FrustumTest<T>
//
// This is a helper class, designed to accelerate the case
// where many tests are made against the same frustum.
// That's a really common case.
//
// The acceleration is achieved by pre-computing the planes of
// the frustum, along with the ablsolute values of the plane normals.
//



//////////////////////////////////////////////////////////////////
// How to use this
//
// Given that you already have:
//    Imath::Frustum   myFrustum
//    IMath::Matrix44  myCameraWorldMatrix
//
// First, make a frustum test object:
//    FrustumTest myFrustumTest(myFrustum, myCameraWorldMatrix)
//
// Whenever the camera or frustum changes, call:
//    myFrustumTest.setFrustum(myFrustum, myCameraWorldMatrix)
//
// For each object you want to test for visibility, call:
//    myFrustumTest.isVisible(myBox)
//    myFrustumTest.isVisible(mySphere)
//    myFrustumTest.isVisible(myVec3)
//    myFrustumTest.completelyContains(myBox)
//    myFrustumTest.completelyContains(mySphere)
//




//////////////////////////////////////////////////////////////////
// Explanation of how it works
//
//
// We store six world-space Frustum planes (nx, ny, nz, offset)
//
// Points: To test a Vec3 for visibility, test it against each plane
//         using the normal (v dot n - offset) method. (the result is exact)
//
// BBoxes: To test an axis-aligned bbox, test the center against each plane
//         using the normal (v dot n - offset) method, but offset by the
//         box extents dot the abs of the plane normal. (the result is NOT
//         exact, but will not return false-negatives.)
//
// Spheres: To test a sphere, test the center against each plane
//         using the normal (v dot n - offset) method, but offset by the
//         sphere's radius. (the result is NOT exact, but will not return
//         false-negatives.)
//
//
// SPECIAL NOTE: "Where are the dot products?"
//     Actual dot products are currently slow for most SIMD architectures.
//     In order to keep this code optimization-ready, the dot products
//     are all performed using vector adds and multipies.
//
//     In order to do this, the plane equations are stored in "transpose"
//     form, with the X components grouped into an X vector, etc.
//


template <class T>
class FrustumTest
{
public:
    FrustumTest()
    {
        Frustum<T>  frust;
        Matrix44<T> cameraMat;
        cameraMat.makeIdentity();
        setFrustum(frust, cameraMat);
    }
    FrustumTest(Frustum<T> &frustum, const Matrix44<T> &cameraMat)
    {
        setFrustum(frustum, cameraMat);
    }

    ////////////////////////////////////////////////////////////////////
    // setFrustum()
    // This updates the frustum test with a new frustum and matrix.
    // This should usually be called just once per frame.
    void setFrustum(Frustum<T> &frustum, const Matrix44<T> &cameraMat);

    ////////////////////////////////////////////////////////////////////
    // isVisible()
    // Check to see if shapes are visible.
    bool isVisible(const Sphere3<T> &sphere) const;
    bool isVisible(const Box<Vec3<T> > &box) const;
    bool isVisible(const Vec3<T> &vec) const;
    
    ////////////////////////////////////////////////////////////////////
    // completelyContains()
    // Check to see if shapes are entirely contained.
    bool completelyContains(const Sphere3<T> &sphere) const;
    bool completelyContains(const Box<Vec3<T> > &box) const;
    
    // These next items are kept primarily for debugging tools.
    // It's useful for drawing the culling environment, and also
    // for getting an "outside view" of the culling frustum.
    Imath::Matrix44<T> cameraMat() const {return cameraMatrix;}
    Imath::Frustum<T> currentFrustum() const {return currFrustum;}

protected:
    // To understand why the planes are stored this way, see
    // the SPECIAL NOTE above.
    Vec3<T> planeNormX[2];  // The X compunents from 6 plane equations
    Vec3<T> planeNormY[2];  // The Y compunents from 6 plane equations
    Vec3<T> planeNormZ[2];  // The Z compunents from 6 plane equations

    Vec3<T> planeOffsetVec[2]; // The distance offsets from 6 plane equations

    // The absolute values are stored to assist with bounding box tests.
    Vec3<T> planeNormAbsX[2];  // The abs(X) compunents from 6 plane equations
    Vec3<T> planeNormAbsY[2];  // The abs(X) compunents from 6 plane equations
    Vec3<T> planeNormAbsZ[2];  // The abs(X) compunents from 6 plane equations

    // These are kept primarily for debugging tools.
    Frustum<T> currFrustum;
    Matrix44<T> cameraMatrix;
};


////////////////////////////////////////////////////////////////////
// setFrustum()
// This should usually only be called once per frame, or however
// often the camera moves.
template<class T>
void FrustumTest<T>::setFrustum(Frustum<T> &frustum,
                                const Matrix44<T> &cameraMat)
{
    Plane3<T> frustumPlanes[6];
    frustum.planes(frustumPlanes, cameraMat);

    // Here's where we effectively transpose the plane equations.
    // We stuff all six X's into the two planeNormX vectors, etc.
    for (int i = 0; i < 2; ++i)
    {
        int index = i * 3;

        planeNormX[i]     = Vec3<T>(frustumPlanes[index + 0].normal.x,
                                    frustumPlanes[index + 1].normal.x, 
                                    frustumPlanes[index + 2].normal.x);
        planeNormY[i]     = Vec3<T>(frustumPlanes[index + 0].normal.y,
                                    frustumPlanes[index + 1].normal.y,
                                    frustumPlanes[index + 2].normal.y);
        planeNormZ[i]     = Vec3<T>(frustumPlanes[index + 0].normal.z,
                                    frustumPlanes[index + 1].normal.z,
                                    frustumPlanes[index + 2].normal.z);

        planeNormAbsX[i]  = Vec3<T>(Imath::abs(planeNormX[i].x),
                                    Imath::abs(planeNormX[i].y), 
                                    Imath::abs(planeNormX[i].z));
        planeNormAbsY[i]  = Vec3<T>(Imath::abs(planeNormY[i].x), 
                                    Imath::abs(planeNormY[i].y),
                                    Imath::abs(planeNormY[i].z));
        planeNormAbsZ[i]  = Vec3<T>(Imath::abs(planeNormZ[i].x), 
                                    Imath::abs(planeNormZ[i].y),
                                    Imath::abs(planeNormZ[i].z));

        planeOffsetVec[i] = Vec3<T>(frustumPlanes[index + 0].distance,
                                    frustumPlanes[index + 1].distance,
                                    frustumPlanes[index + 2].distance);
    }
    currFrustum = frustum;
    cameraMatrix = cameraMat;
}


////////////////////////////////////////////////////////////////////
// isVisible(Sphere)
// Returns true if any part of the sphere is inside
// the frustum.
// The result MAY return close false-positives, but not false-negatives.
//
template<typename T>
bool FrustumTest<T>::isVisible(const Sphere3<T> &sphere) const
{
    Vec3<T> center = sphere.center;
    Vec3<T> radiusVec = Vec3<T>(sphere.radius, sphere.radius, sphere.radius);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0  = planeNormX[0] * center.x 
                + planeNormY[0] * center.y 
                + planeNormZ[0] * center.z 
                - radiusVec
                - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1  = planeNormX[1] * center.x 
                + planeNormY[1] * center.y 
                + planeNormZ[1] * center.z 
                - radiusVec
                - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

////////////////////////////////////////////////////////////////////
// completelyContains(Sphere)
// Returns true if every part of the sphere is inside
// the frustum.
// The result MAY return close false-negatives, but not false-positives.
//
template<typename T>
bool FrustumTest<T>::completelyContains(const Sphere3<T> &sphere) const
{
    Vec3<T> center = sphere.center;
    Vec3<T> radiusVec = Vec3<T>(sphere.radius, sphere.radius, sphere.radius);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0  = planeNormX[0] * center.x 
                + planeNormY[0] * center.y 
                + planeNormZ[0] * center.z 
                + radiusVec
                - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1  = planeNormX[1] * center.x 
                + planeNormY[1] * center.y 
                + planeNormZ[1] * center.z 
                + radiusVec
                - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

////////////////////////////////////////////////////////////////////
// isVisible(Box)
// Returns true if any part of the axis-aligned box
// is inside the frustum.
// The result MAY return close false-positives, but not false-negatives.
//
template<typename T>
bool FrustumTest<T>::isVisible(const Box<Vec3<T> > &box) const
{
    Vec3<T> center = (box.min + box.max) / 2;
    Vec3<T> extent = (box.max - center);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0  = planeNormX[0] * center.x 
                + planeNormY[0] * center.y 
                + planeNormZ[0] * center.z
                - planeNormAbsX[0] * extent.x 
                - planeNormAbsY[0] * extent.y 
                - planeNormAbsZ[0] * extent.z 
                - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1  = planeNormX[1] * center.x 
                + planeNormY[1] * center.y 
                + planeNormZ[1] * center.z
                - planeNormAbsX[1] * extent.x 
                - planeNormAbsY[1] * extent.y 
                - planeNormAbsZ[1] * extent.z 
                - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}

////////////////////////////////////////////////////////////////////
// completelyContains(Box)
// Returns true if every part of the axis-aligned box
// is inside the frustum.
// The result MAY return close false-negatives, but not false-positives.
//
template<typename T>
bool FrustumTest<T>::completelyContains(const Box<Vec3<T> > &box) const
{
    Vec3<T> center = (box.min + box.max) / 2;
    Vec3<T> extent = (box.max - center);

    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0  = planeNormX[0] * center.x 
                + planeNormY[0] * center.y 
                + planeNormZ[0] * center.z
                + planeNormAbsX[0] * extent.x 
                + planeNormAbsY[0] * extent.y 
                + planeNormAbsZ[0] * extent.z 
                - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1  = planeNormX[1] * center.x 
                + planeNormY[1] * center.y 
                + planeNormZ[1] * center.z
                + planeNormAbsX[1] * extent.x 
                + planeNormAbsY[1] * extent.y 
                + planeNormAbsZ[1] * extent.z 
                - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}


////////////////////////////////////////////////////////////////////
// isVisible(Vec3)
// Returns true if the point is inside the frustum.
//
template<typename T>
bool FrustumTest<T>::isVisible(const Vec3<T> &vec) const
{
    // This is a vertical dot-product on three vectors at once.
    Vec3<T> d0  = (planeNormX[0] * vec.x) 
                + (planeNormY[0] * vec.y) 
                + (planeNormZ[0] * vec.z) 
                - planeOffsetVec[0];

    if (d0.x >= 0 || d0.y >= 0 || d0.z >= 0)
        return false;

    Vec3<T> d1  = (planeNormX[1] * vec.x) 
                + (planeNormY[1] * vec.y) 
                + (planeNormZ[1] * vec.z) 
                - planeOffsetVec[1];

    if (d1.x >= 0 || d1.y >= 0 || d1.z >= 0)
        return false;

    return true;
}


typedef FrustumTest<float>	FrustumTestf;
typedef FrustumTest<double> FrustumTestd;

} //namespace Imath

#endif
