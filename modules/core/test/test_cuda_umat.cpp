// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>

namespace opencv_test { namespace {

static void skipIfNoCudaDevice()
{
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw SkipTestException("No CUDA device available");
}

static UMat makeCudaUMat()
{
    UMat u;
    u.allocator = cv::cuda::getCudaAllocator();
    return u;
}

TEST(Core_CudaUMat, roundtrip_contiguous)
{
    skipIfNoCudaDevice();
    ASSERT_TRUE(cv::cuda::getCudaAllocator() != NULL);

    Mat src(64, 48, CV_32FC1);
    randu(src, 0, 1);

    UMat u = makeCudaUMat();
    src.copyTo(u);
    ASSERT_TRUE(u.u != NULL && u.u->handle != NULL);

    Mat dst;
    u.copyTo(dst);

    EXPECT_EQ(src.size(), dst.size());
    EXPECT_EQ(src.type(), dst.type());
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

TEST(Core_CudaUMat, roundtrip_multichannel)
{
    skipIfNoCudaDevice();

    Mat src(37, 53, CV_8UC3);
    randu(src, 0, 255);

    UMat u = makeCudaUMat();
    src.copyTo(u);

    Mat dst;
    u.copyTo(dst);
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

TEST(Core_CudaUMat, roundtrip_roi_source)
{
    skipIfNoCudaDevice();

    Mat full(100, 100, CV_32FC1);
    randu(full, 0, 1);
    Mat roi = full(Rect(10, 12, 50, 40));
    ASSERT_FALSE(roi.isContinuous());

    UMat u = makeCudaUMat();
    roi.copyTo(u);

    Mat dst;
    u.copyTo(dst);
    EXPECT_EQ(roi.size(), dst.size());
    EXPECT_EQ(0.0, cvtest::norm(roi, dst, NORM_INF));
}

TEST(Core_CudaUMat, download_into_roi)
{
    skipIfNoCudaDevice();

    Mat src(40, 50, CV_32FC1);
    randu(src, 0, 1);

    UMat u = makeCudaUMat();
    src.copyTo(u);

    Mat canvas(80, 90, CV_32FC1, Scalar(0));
    Mat dstRoi = canvas(Rect(5, 7, 50, 40));
    ASSERT_FALSE(dstRoi.isContinuous());
    u.copyTo(dstRoi);

    EXPECT_EQ(0.0, cvtest::norm(src, dstRoi, NORM_INF));
}

TEST(Core_CudaUMat, device_to_device_copy)
{
    skipIfNoCudaDevice();

    Mat src(70, 90, CV_32FC1);
    randu(src, 0, 1);

    UMat a = makeCudaUMat();
    src.copyTo(a);

    UMat b = makeCudaUMat();
    a.copyTo(b);

    Mat dst;
    b.copyTo(dst);
    EXPECT_EQ(0.0, cvtest::norm(src, dst, NORM_INF));
}

static ::testing::AssertionResult expectCudaOwned(const UMat& u)
{
    if (u.u == NULL)
        return ::testing::AssertionFailure() << "UMat has null UMatData";
    if (u.u->currAllocator != cv::cuda::getCudaAllocator())
        return ::testing::AssertionFailure() << "UMat fell back to non-CUDA allocator";
    if (u.u->size > 0 && u.u->handle == NULL)
        return ::testing::AssertionFailure() << "Non-empty CUDA UMat has null device handle";
    return ::testing::AssertionSuccess();
}

TEST(Core_CudaUMat, getmat_view_coherence)
{
    skipIfNoCudaDevice();
    MatAllocator* cudaAlloc = cv::cuda::getCudaAllocator();

    Mat init(32, 48, CV_32FC1), mid(32, 48, CV_32FC1), after(32, 48, CV_32FC1);
    randu(init, -5.f, 5.f);
    randu(mid, 10.f, 20.f);
    randu(after, 30.f, 40.f);

    UMat src = makeCudaUMat();
    init.copyTo(src);
    ASSERT_TRUE(expectCudaOwned(src));

    // 1. Keep a writable getMat() view alive across copyTo(UMat), ROI copyTo(UMat),
    //    ROI download(), full copyTo(Mat), and a premature unmap(), then write again
    //    through the same view before releasing it.
    Mat view = src.getMat(ACCESS_RW);
    Mat view2 = src.getMat(ACCESS_RW); // concurrent second view (refcount == 2)
    mid.copyTo(view);

    // Releasing one of two concurrent views must not unmap/clear deviceCopyObsolete.
    view2.release();
    EXPECT_TRUE(src.u->deviceCopyObsolete());

    // Direct unmap() while refcount > 0 must leave deviceCopyObsolete intact.
    cudaAlloc->unmap(src.u);
    EXPECT_TRUE(src.u->deviceCopyObsolete());

    UMat dst = makeCudaUMat();
    src.copyTo(dst);
    ASSERT_TRUE(expectCudaOwned(dst));

    Rect srcRoi(4, 5, 16, 12), dstRoi(8, 6, 16, 12);
    UMat dstRoiUMat = makeCudaUMat();
    init.copyTo(dstRoiUMat);
    ASSERT_TRUE(expectCudaOwned(dstRoiUMat));
    src(srcRoi).copyTo(dstRoiUMat(dstRoi));

    // Full and ROI-to-ROI download() while view is live.
    Mat snap;
    src.copyTo(snap);
    EXPECT_EQ(0.0, cvtest::norm(mid, snap, NORM_INF));

    Mat roiCanvas(32, 48, CV_32FC1, Scalar::all(-1.f));
    Mat expectedCanvas = roiCanvas.clone();
    src(srcRoi).copyTo(roiCanvas(dstRoi));
    mid(srcRoi).copyTo(expectedCanvas(dstRoi));
    EXPECT_EQ(0.0, cvtest::norm(expectedCanvas, roiCanvas, NORM_INF));

    // More writes through the still-alive view must still reach the device on release.
    after.copyTo(view);
    view.release();
    EXPECT_FALSE(src.u->deviceCopyObsolete());

    Mat back;
    src.copyTo(back);
    EXPECT_EQ(0.0, cvtest::norm(after, back, NORM_INF));

    // ROI-to-ROI download() after release (device-resident path).
    Mat postCanvas(32, 48, CV_32FC1, Scalar::all(-2.f));
    Mat expectedPostCanvas = postCanvas.clone();
    src(srcRoi).copyTo(postCanvas(dstRoi));
    after(srcRoi).copyTo(expectedPostCanvas(dstRoi));
    EXPECT_EQ(0.0, cvtest::norm(expectedPostCanvas, postCanvas, NORM_INF));

    UMat dstAfter = makeCudaUMat();
    src.copyTo(dstAfter);
    ASSERT_TRUE(expectCudaOwned(dstAfter));
    Mat backFromCopy;
    dstAfter.copyTo(backFromCopy);
    EXPECT_EQ(0.0, cvtest::norm(after, backFromCopy, NORM_INF));

    Mat gotDst;
    dst.copyTo(gotDst);
    EXPECT_EQ(0.0, cvtest::norm(mid, gotDst, NORM_INF));

    Mat expectedRoi = init.clone();
    mid(srcRoi).copyTo(expectedRoi(dstRoi));
    Mat gotRoi;
    dstRoiUMat.copyTo(gotRoi);
    EXPECT_EQ(0.0, cvtest::norm(expectedRoi, gotRoi, NORM_INF));

    // 2. Destination with a live getMat() view: copyTo() from device UMat, upload() from Mat,
    //    and copyTo() from a live-viewed UMat all update the live view and keep deviceCopyObsolete.
    UMat dstMapped = makeCudaUMat();
    init.copyTo(dstMapped);
    ASSERT_TRUE(expectCudaOwned(dstMapped));
    Mat dstView = dstMapped.getMat(ACCESS_RW);
    Mat expectedMapped = init.clone();

    // Device-resident src -> live-viewed dstMapped
    src(srcRoi).copyTo(dstMapped(dstRoi));
    after(srcRoi).copyTo(expectedMapped(dstRoi));
    EXPECT_TRUE(dstMapped.u->deviceCopyObsolete());
    EXPECT_EQ(0.0, cvtest::norm(expectedMapped, dstView, NORM_INF));

    // Host Mat upload -> live-viewed dstMapped
    Rect patchRoi(28, 18, 12, 10);
    Mat patch(patchRoi.size(), CV_32FC1);
    randu(patch, 50.f, 60.f);
    patch.copyTo(dstMapped(patchRoi));
    patch.copyTo(expectedMapped(patchRoi));
    EXPECT_TRUE(dstMapped.u->deviceCopyObsolete());
    EXPECT_EQ(0.0, cvtest::norm(expectedMapped, dstView, NORM_INF));

    // Live-viewed UMat -> live-viewed dstMapped (distinct src/dst ROIs)
    Mat src2Data(32, 48, CV_32FC1);
    randu(src2Data, -40.f, -30.f);
    UMat src2 = makeCudaUMat();
    init.copyTo(src2);
    ASSERT_TRUE(expectCudaOwned(src2));
    Mat src2View = src2.getMat(ACCESS_RW);
    src2Data.copyTo(src2View);
    Rect src2Roi(2, 20, 10, 8), dst2Roi(14, 20, 10, 8);
    src2(src2Roi).copyTo(dstMapped(dst2Roi));
    src2Data(src2Roi).copyTo(expectedMapped(dst2Roi));
    src2View.release();
    EXPECT_TRUE(dstMapped.u->deviceCopyObsolete());
    EXPECT_EQ(0.0, cvtest::norm(expectedMapped, dstView, NORM_INF));

    // Further writes through dstView must still flush to device on release
    Rect tailRoi(30, 2, 10, 6);
    Mat tail(tailRoi.size(), CV_32FC1);
    randu(tail, 70.f, 80.f);
    tail.copyTo(dstView(tailRoi));
    tail.copyTo(expectedMapped(tailRoi));
    dstView.release();

    Mat finalDst;
    dstMapped.copyTo(finalDst);
    EXPECT_EQ(0.0, cvtest::norm(expectedMapped, finalDst, NORM_INF));

    // 3. Subsequent upload() and copy() on a UMat that already holds a cached host buffer
    //    must markHostCopyObsolete(true) so a later getMat(ACCESS_READ) sees fresh device data.
    Mat freshUpload(32, 48, CV_32FC1), freshCopy(32, 48, CV_32FC1);
    randu(freshUpload, -80.f, -70.f);
    randu(freshCopy, 80.f, 90.f);

    freshUpload.copyTo(dstMapped);
    {
        Mat readAfterUpload = dstMapped.getMat(ACCESS_READ);
        EXPECT_EQ(0.0, cvtest::norm(freshUpload, readAfterUpload, NORM_INF));
    }

    UMat uFreshSrc = makeCudaUMat();
    freshCopy.copyTo(uFreshSrc);
    ASSERT_TRUE(expectCudaOwned(uFreshSrc));
    uFreshSrc.copyTo(dstMapped);
    {
        Mat readAfterCopy = dstMapped.getMat(ACCESS_READ);
        EXPECT_EQ(0.0, cvtest::norm(freshCopy, readAfterCopy, NORM_INF));
    }

    // 4. Overlapping ROI copies within one UMat, without and with a live getMat() view.
    Mat overlapBase(32, 32, CV_32FC1);
    randu(overlapBase, -25.f, 25.f);
    Rect ovSrc(0, 0, 24, 24), ovDst(4, 4, 24, 24);
    Mat expectedOverlap = overlapBase.clone();
    overlapBase(ovSrc).clone().copyTo(expectedOverlap(ovDst));

    UMat uDevOv = makeCudaUMat();
    overlapBase.copyTo(uDevOv);
    ASSERT_TRUE(expectCudaOwned(uDevOv));
    uDevOv(ovSrc).copyTo(uDevOv(ovDst));
    Mat gotDevOv;
    uDevOv.copyTo(gotDevOv);
    EXPECT_EQ(0.0, cvtest::norm(expectedOverlap, gotDevOv, NORM_INF));

    UMat uHostOv = makeCudaUMat();
    overlapBase.copyTo(uHostOv);
    ASSERT_TRUE(expectCudaOwned(uHostOv));
    {
        Mat ovView = uHostOv.getMat(ACCESS_RW);
        uHostOv(ovSrc).copyTo(uHostOv(ovDst));
        EXPECT_EQ(0.0, cvtest::norm(expectedOverlap, ovView, NORM_INF));
    }
    Mat gotHostOv;
    uHostOv.copyTo(gotHostOv);
    EXPECT_EQ(0.0, cvtest::norm(expectedOverlap, gotHostOv, NORM_INF));
}

static UMat wrapHostDataWithCudaAlloc(Mat& host, MatAllocator* cudaAlloc)
{
    const int sizes[2] = { host.rows, host.cols };
    size_t steps[2] = { host.step[0], host.step[1] };
    UMat u;
    u.allocator = cudaAlloc;
    u.flags = UMat::MAGIC_VAL | (host.isContinuous() ? UMat::CONTINUOUS_FLAG : 0) | host.type();
    u.dims = 2;
    u.size = MatShape(2, sizes);
    u.rows = host.rows;
    u.cols = host.cols;
    u.u = cudaAlloc->allocate(2, sizes, host.type(), host.data, steps, ACCESS_RW, USAGE_DEFAULT);
    u.step[0] = steps[0];
    u.step[1] = steps[1];
    u.addref();
    return u;
}

TEST(Core_CudaUMat, user_host_data_coherence)
{
    skipIfNoCudaDevice();
    MatAllocator* cudaAlloc = cv::cuda::getCudaAllocator();
    ASSERT_TRUE(cudaAlloc != NULL);

    Mat host(32, 48, CV_32FC1);
    randu(host, -10.f, 10.f);

    UMat u = wrapHostDataWithCudaAlloc(host, cudaAlloc);
    ASSERT_TRUE(u.u != NULL);
    EXPECT_TRUE(u.u->copyOnMap());

    // 1. download() while deviceCopyObsolete() is true must read data0, not uninitialized VRAM
    Mat downloaded;
    u.copyTo(downloaded);
    EXPECT_EQ(0.0, cvtest::norm(host, downloaded, NORM_INF));

    // 2. Same-allocator copy() while src->deviceCopyObsolete() is true must read data0 directly
    UMat uCopy = makeCudaUMat();
    u.copyTo(uCopy);
    ASSERT_TRUE(expectCudaOwned(uCopy));
    Mat fromCopy;
    uCopy.copyTo(fromCopy);
    EXPECT_EQ(0.0, cvtest::norm(host, fromCopy, NORM_INF));
    EXPECT_TRUE(u.u->deviceCopyObsolete());

    // 2b. UMat::handle() on a data0-backed CUDA UMat must flush host->device via COPY_ON_MAP
    void* devPtr = u.handle(ACCESS_READ);
    ASSERT_TRUE(devPtr != NULL);
    EXPECT_FALSE(u.u->deviceCopyObsolete());
    Mat afterHandle;
    u.copyTo(afterHandle);
    EXPECT_EQ(0.0, cvtest::norm(host, afterHandle, NORM_INF));

    // 3. Partial-ROI upload() onto a data0-backed UMat (non-zero offset, (0,0) top-half rows,
    //    and (0,0) sub-width rect) must preserve unwritten pixels outside ROI and markHostCopyObsolete
    Rect patchRects[3] = { Rect(4, 5, 12, 10), Rect(0, 0, host.cols, host.rows / 2), Rect(0, 0, 12, 10) };
    for (int k = 0; k < 3; k++)
    {
        Mat hostK = host.clone();
        UMat uPartial = wrapHostDataWithCudaAlloc(hostK, cudaAlloc);
        ASSERT_TRUE(uPartial.u != NULL);

        Mat patch(patchRects[k].size(), CV_32FC1);
        randu(patch, 50.f, 60.f);
        Mat expected = hostK.clone();
        patch.copyTo(expected(patchRects[k]));
        patch.copyTo(uPartial(patchRects[k]));

        Mat afterPartial;
        uPartial.copyTo(afterPartial);
        EXPECT_EQ(0.0, cvtest::norm(expected, afterPartial, NORM_INF));
        EXPECT_EQ(0.0, cvtest::norm(expected, uPartial.getMat(ACCESS_READ), NORM_INF));
    }

    // 3b. Full-matrix upload() onto a data0-backed UMat
    {
        Mat hostFull = host.clone(), replacement(host.size(), CV_32FC1);
        randu(replacement, 100.f, 110.f);
        UMat uFull = wrapHostDataWithCudaAlloc(hostFull, cudaAlloc);
        ASSERT_TRUE(uFull.u != NULL);
        replacement.copyTo(uFull);
        EXPECT_FALSE(uFull.u->deviceCopyObsolete());
        Mat gotFull;
        uFull.copyTo(gotFull);
        EXPECT_EQ(0.0, cvtest::norm(replacement, gotFull, NORM_INF));
    }

    // 4. Partial-ROI copy() onto a data0-backed destination UMat (refcount == 0, deviceCopyObsolete == true)
    {
        Mat hostDst = host.clone();
        UMat uDst = wrapHostDataWithCudaAlloc(hostDst, cudaAlloc);
        ASSERT_TRUE(uDst.u != NULL);

        Mat patchData(10, 12, CV_32FC1);
        randu(patchData, -60.f, -50.f);
        UMat uPatchDev = makeCudaUMat();
        patchData.copyTo(uPatchDev);
        ASSERT_TRUE(expectCudaOwned(uPatchDev));

        Rect dstPatch(6, 7, 12, 10);
        uPatchDev.copyTo(uDst(dstPatch));
        EXPECT_FALSE(uDst.u->deviceCopyObsolete());

        Mat expectedDst = hostDst.clone();
        patchData.copyTo(expectedDst(dstPatch));
        Mat gotDst;
        uDst.copyTo(gotDst);
        EXPECT_EQ(0.0, cvtest::norm(expectedDst, gotDst, NORM_INF));
    }

    // 4b. Overlapping same-buffer ROI copy on a data0-backed UMat
    {
        Mat hostOv(32, 32, CV_32FC1);
        randu(hostOv, -15.f, 15.f);
        UMat uOv = wrapHostDataWithCudaAlloc(hostOv, cudaAlloc);
        ASSERT_TRUE(uOv.u != NULL);
        Rect ovSrc(0, 0, 20, 20), ovDst(5, 3, 20, 20);
        Mat expectedOv = hostOv.clone();
        hostOv(ovSrc).clone().copyTo(expectedOv(ovDst));
        uOv(ovSrc).copyTo(uOv(ovDst));
        Mat gotOv;
        uOv.copyTo(gotOv);
        EXPECT_EQ(0.0, cvtest::norm(expectedOv, gotOv, NORM_INF));
    }

    // 5. Caller-provided step with data0 must be preserved (host download and device flush/D2H);
    //    without data0, custom step is ignored; undersized step with data0 must throw.
    const int customSizes[2] = { 8, 10 };
    const size_t customRowStep = 16 * sizeof(float);
    Mat paddedBuf(8, 16, CV_32FC1);
    randu(paddedBuf, -5.f, 5.f);
    Mat paddedView = paddedBuf(Rect(0, 0, 10, 8));
    UMat uCustom = wrapHostDataWithCudaAlloc(paddedView, cudaAlloc);
    ASSERT_TRUE(uCustom.u != NULL);
    EXPECT_EQ(customRowStep, uCustom.step[0]);
    EXPECT_EQ(sizeof(float), uCustom.step[1]);
    EXPECT_EQ(8 * customRowStep, uCustom.u->size);
    Mat customBack;
    uCustom.copyTo(customBack);
    EXPECT_EQ(0.0, cvtest::norm(paddedView, customBack, NORM_INF));

    // Flush custom-stepped data0 to VRAM and verify strided D2H download
    ASSERT_TRUE(uCustom.handle(ACCESS_READ) != NULL);
    EXPECT_FALSE(uCustom.u->deviceCopyObsolete());
    Mat customBackDev;
    uCustom.copyTo(customBackDev);
    EXPECT_EQ(0.0, cvtest::norm(paddedView, customBackDev, NORM_INF));

    size_t tightSteps[2] = { customRowStep, CV_AUTOSTEP };
    UMatData* uTight = cudaAlloc->allocate(2, customSizes, CV_32FC1, NULL, tightSteps, ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(uTight != NULL);
    EXPECT_EQ(10 * sizeof(float), tightSteps[0]);
    EXPECT_EQ(sizeof(float), tightSteps[1]);
    cudaAlloc->deallocate(uTight);

    size_t badSteps[2] = { 4 * sizeof(float), sizeof(float) };
    EXPECT_THROW(cudaAlloc->allocate(2, customSizes, CV_32FC1, paddedBuf.data, badSteps, ACCESS_RW, USAGE_DEFAULT),
                 cv::Exception);
}

struct CountingCustomAllocator CV_FINAL : public MatAllocator
{
    mutable int deallocateCalls = 0;

    UMatData* make(size_t total, uchar* backing) const
    {
        UMatData* u = new UMatData(this);
        u->prevAllocator = NULL;
        u->size = total;
        u->data = u->origdata = backing;
        return u;
    }
    UMatData* allocate(int, const int*, int, void*, size_t*, AccessFlag, UMatUsageFlags) const CV_OVERRIDE
    { return NULL; }
    bool allocate(UMatData* u, AccessFlag, UMatUsageFlags) const CV_OVERRIDE
    { return u != NULL; }
    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        ASSERT_TRUE(u != NULL);
        EXPECT_TRUE(u->handle == NULL);
        EXPECT_EQ(this, u->currAllocator);
        deallocateCalls++;
        delete u;
    }
};

TEST(Core_CudaUMat, zero_dim_and_decline_foreign_umatdata)
{
    skipIfNoCudaDevice();
    MatAllocator* cudaAlloc = cv::cuda::getCudaAllocator();
    ASSERT_TRUE(cudaAlloc != NULL);

    // 0-D allocation must not underflow step[-1] or modify caller guard words
    size_t sentinel[3] = { 0xDEADBEEFu, 0xCAFEBABEu, 0x12345678u };
    UMatData* u0 = cudaAlloc->allocate(0, NULL, CV_32FC1, NULL, &sentinel[1], ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(u0 != NULL);
    EXPECT_EQ(size_t(0xDEADBEEFu), sentinel[0]);
    EXPECT_EQ(size_t(0xCAFEBABEu), sentinel[1]);
    EXPECT_EQ(size_t(0x12345678u), sentinel[2]);
    EXPECT_EQ(sizeof(float), u0->size);
    EXPECT_TRUE(cudaAlloc->allocate(u0, ACCESS_RW, USAGE_DEFAULT));
    cudaAlloc->deallocate(u0);

    // Zero-element shape must allocate 0 bytes and leave handle == NULL
    const int zeroSz[2] = { 0, 10 };
    size_t zeroSt[2] = { CV_AUTOSTEP, CV_AUTOSTEP };
    UMatData* uZero = cudaAlloc->allocate(2, zeroSz, CV_32FC1, NULL, zeroSt, ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(uZero != NULL);
    EXPECT_EQ(size_t(0), uZero->size);
    EXPECT_TRUE(uZero->handle == NULL);
    cudaAlloc->deallocate(uZero);

    // NULL and invalid/overflowing shapes must fail cleanly
    EXPECT_FALSE(cudaAlloc->allocate((UMatData*)NULL, ACCESS_RW, USAGE_DEFAULT));
    const int negSz[2] = { -1, -1 };
    EXPECT_THROW(cudaAlloc->allocate(2, negSz, CV_32FC1, NULL, NULL, ACCESS_RW, USAGE_DEFAULT), cv::Exception);
    const int ovfSz[3] = { 16, 536903681, 536838145 };
    EXPECT_THROW(cudaAlloc->allocate(3, ovfSz, CV_32FC1, NULL, NULL, ACCESS_RW, USAGE_DEFAULT), cv::Exception);

    // allocate(UMatData*) must decline UMatData owned by another allocator without allocating VRAM;
    // download()/upload()/copy() must reject foreign UMatData
    const int sz[2] = { 8, 12 };
    size_t st[2] = { CV_AUTOSTEP, CV_AUTOSTEP };
    UMatData* uh = Mat::getStdAllocator()->allocate(2, sz, CV_32FC1, NULL, st, ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(uh != NULL);
    EXPECT_FALSE(cudaAlloc->allocate(uh, ACCESS_RW, USAGE_DEFAULT));
    EXPECT_EQ(Mat::getStdAllocator(), uh->currAllocator);
    EXPECT_TRUE(uh->handle == NULL);
    const size_t copySz[2] = { 8, 12 * sizeof(float) }, copyStep[2] = { 12 * sizeof(float), sizeof(float) };
    Mat dummy(8, 12, CV_32FC1, Scalar(1.f));
    EXPECT_THROW(cudaAlloc->download(uh, dummy.data, 2, copySz, NULL, copyStep, copyStep), cv::Exception);
    EXPECT_THROW(cudaAlloc->upload(uh, dummy.data, 2, copySz, NULL, copyStep, copyStep), cv::Exception);
    UMatData* uOwn = cudaAlloc->allocate(2, sz, CV_32FC1, NULL, st, ACCESS_RW, USAGE_DEFAULT);
    ASSERT_TRUE(uOwn != NULL);
    EXPECT_THROW(cudaAlloc->copy(uh, uOwn, 2, copySz, NULL, copyStep, NULL, copyStep, false), cv::Exception);
    EXPECT_THROW(cudaAlloc->copy(uOwn, uh, 2, copySz, NULL, copyStep, NULL, copyStep, false), cv::Exception);
    cudaAlloc->deallocate(uOwn);
    uh->currAllocator->deallocate(uh);

    CountingCustomAllocator customAlloc;
    std::vector<float> buf(8 * 12, 1.f);
    UMatData* uc = customAlloc.make(buf.size() * sizeof(float), reinterpret_cast<uchar*>(buf.data()));
    EXPECT_FALSE(cudaAlloc->allocate(uc, ACCESS_RW, USAGE_DEFAULT));
    EXPECT_EQ(&customAlloc, uc->currAllocator);
    EXPECT_TRUE(uc->handle == NULL);
    uc->currAllocator->deallocate(uc);
    EXPECT_EQ(1, customAlloc.deallocateCalls);

    // Owned handle-less UMatData: allocate(u), download() with handle == NULL, and lazy upload()/copy()
    {
        Mat ref(8, 12, CV_32FC1), got(8, 12, CV_32FC1);
        randu(ref, -3.f, 3.f);

        UMatData* uDataOnly = new UMatData(cudaAlloc);
        uDataOnly->size = 8 * 12 * sizeof(float);
        uDataOnly->data = uDataOnly->origdata = ref.data;
        uDataOnly->flags |= UMatData::USER_ALLOCATED;
        uDataOnly->markDeviceCopyObsolete(true);
        cudaAlloc->download(uDataOnly, got.data, 2, copySz, NULL, copyStep, copyStep);
        EXPECT_EQ(0.0, cvtest::norm(ref, got, NORM_INF));
        ASSERT_TRUE(cudaAlloc->allocate(uDataOnly, ACCESS_RW, USAGE_DEFAULT));
        EXPECT_TRUE(uDataOnly->handle != NULL);
        EXPECT_TRUE(uDataOnly->deviceCopyObsolete());
        cudaAlloc->deallocate(uDataOnly);

        UMatData* uLazyUp = new UMatData(cudaAlloc);
        uLazyUp->size = 8 * 12 * sizeof(float);
        cudaAlloc->upload(uLazyUp, ref.data, 2, copySz, NULL, copyStep, copyStep);
        EXPECT_TRUE(uLazyUp->handle != NULL);
        cudaAlloc->download(uLazyUp, got.data, 2, copySz, NULL, copyStep, copyStep);
        EXPECT_EQ(0.0, cvtest::norm(ref, got, NORM_INF));

        UMatData* uLazyDst = new UMatData(cudaAlloc);
        uLazyDst->size = 8 * 12 * sizeof(float);
        cudaAlloc->copy(uLazyUp, uLazyDst, 2, copySz, NULL, copyStep, NULL, copyStep, true);
        EXPECT_TRUE(uLazyDst->handle != NULL);
        got.setTo(0);
        cudaAlloc->download(uLazyDst, got.data, 2, copySz, NULL, copyStep, copyStep);
        EXPECT_EQ(0.0, cvtest::norm(ref, got, NORM_INF));

        cudaAlloc->deallocate(uLazyUp);
        cudaAlloc->deallocate(uLazyDst);
    }

    // unmap() must deallocate when urefcount == 0 && refcount == 0
    {
        UMatData* uMatOnly = cudaAlloc->allocate(2, sz, CV_32FC1, NULL, st, ACCESS_RW, USAGE_DEFAULT);
        ASSERT_TRUE(uMatOnly != NULL);
        std::shared_ptr<int> sentinelObj = std::make_shared<int>(1);
        std::weak_ptr<int> weakSentinel = sentinelObj;
        uMatOnly->allocatorContext = sentinelObj;
        sentinelObj.reset();
        cudaAlloc->map(uMatOnly, ACCESS_RW);
        cudaAlloc->unmap(uMatOnly);
        EXPECT_TRUE(weakSentinel.expired());
    }
}

TEST(Core_CudaUMat, roi_to_roi_2d_and_nd_copy)
{
    skipIfNoCudaDevice();

    // 2-D strided ROI-to-ROI copy (exercises cudaMemcpy2D path)
    Mat srcFull(96, 112, CV_32FC3), dstFull(96, 112, CV_32FC3);
    randu(srcFull, -5.f, 5.f);
    randu(dstFull, 10.f, 20.f);

    UMat uSrc = makeCudaUMat(), uDst = makeCudaUMat();
    srcFull.copyTo(uSrc);
    dstFull.copyTo(uDst);
    ASSERT_TRUE(expectCudaOwned(uSrc));
    ASSERT_TRUE(expectCudaOwned(uDst));

    Rect srcRoi(7, 9, 45, 37), dstRoi(13, 19, 45, 37);
    uSrc(srcRoi).copyTo(uDst(dstRoi));

    Mat expected = dstFull.clone();
    srcFull(srcRoi).copyTo(expected(dstRoi));

    // Single-row strided 2-D ROI (exercises height == 1 fast path)
    Rect rowSrcRoi(5, 2, 40, 1), rowDstRoi(11, 3, 40, 1);
    uSrc(rowSrcRoi).copyTo(uDst(rowDstRoi));
    srcFull(rowSrcRoi).copyTo(expected(rowDstRoi));

    Mat got;
    uDst.copyTo(got);
    EXPECT_EQ(0.0, cvtest::norm(expected, got, NORM_INF));

    // 1-D full and non-zero-offset subrange copy (exercises dims == 1 direct cudaMemcpy fast path)
    const int sz1d[1] = { 128 };
    Mat src1d(1, sz1d, CV_32FC1), dst1d;
    randu(src1d, -10.f, 10.f);
    UMat u1a = makeCudaUMat(), u1b = makeCudaUMat();
    src1d.copyTo(u1a);
    ASSERT_TRUE(expectCudaOwned(u1a));
    u1a.copyTo(u1b);
    ASSERT_TRUE(expectCudaOwned(u1b));
    Range subSrc1d[1] = { Range(16, 80) }, subDst1d[1] = { Range(32, 96) };
    u1a(subSrc1d).copyTo(u1b(subDst1d));
    Mat expected1d = src1d.clone();
    src1d(subSrc1d).copyTo(expected1d(subDst1d));
    u1b.copyTo(dst1d);
    EXPECT_EQ(0.0, cvtest::norm(expected1d, dst1d, NORM_INF));

    // 3-D sub-volume ROI copy (exercises NAryMatIterator fallback in copyPlanes)
    const int sz3d[3] = { 6, 10, 12 };
    Mat src3d(3, sz3d, CV_32FC1), dst3d(3, sz3d, CV_32FC1);
    randu(src3d, -8.f, 8.f);
    randu(dst3d, 20.f, 30.f);
    UMat u3a = makeCudaUMat(), u3b = makeCudaUMat();
    src3d.copyTo(u3a);
    dst3d.copyTo(u3b);
    ASSERT_TRUE(expectCudaOwned(u3a));
    ASSERT_TRUE(expectCudaOwned(u3b));

    Range srcRanges[3] = { Range(1, 4), Range(2, 7), Range(3, 9) };
    Range dstRanges[3] = { Range(2, 5), Range(1, 6), Range(4, 10) };
    u3a(srcRanges).copyTo(u3b(dstRanges));

    Mat expected3d = dst3d.clone();
    src3d(srcRanges).copyTo(expected3d(dstRanges));
    Mat got3d;
    u3b.copyTo(got3d);
    EXPECT_EQ(0.0, cvtest::norm(expected3d, got3d, NORM_INF));
}

// Proves the allocator's handle is a real, kernel-usable CUDA buffer via GpuMat interop.
TEST(Core_CudaUMat, kernel_interop_via_gpumat)
{
    skipIfNoCudaDevice();

    Mat src(32, 40, CV_32FC1, Scalar(1.0f));

    UMat u = makeCudaUMat();
    src.copyTo(u);
    ASSERT_TRUE(u.u != NULL && u.u->handle != NULL);

    cv::cuda::GpuMat g(u.rows, u.cols, u.type(), u.u->handle,
                       (size_t)u.cols * CV_ELEM_SIZE(u.type()));

    Mat viaGpu;
    g.download(viaGpu);
    EXPECT_EQ(0.0, cvtest::norm(src, viaGpu, NORM_INF));

    g.setTo(Scalar(7.0f));
    u.u->markHostCopyObsolete(true);          // device was mutated behind UMat's back
    Mat afterKernel;
    u.copyTo(afterKernel);
    EXPECT_EQ(0.0, cvtest::norm(Mat(src.size(), src.type(), Scalar(7.0f)), afterKernel, NORM_INF));
}

// P1 gives a UMat storage on CUDA, not GPU execution. With OpenCL on, cv:: ops feed the
// CUDA pointer to the driver as a cl_mem and segfault; the tests below force OpenCL off so
// ops resolve through the host getMat() fallback, and pin why the OpenCL path is unusable.

struct NoOpenCL
{
    bool prev;
    NoOpenCL() : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(false); }
    ~NoOpenCL() { cv::ocl::setUseOpenCL(prev); }
};

// A CUDA UMat fails the OpenCL kernel path's isCLBuffer precondition (carries the CUDA allocator).
TEST(Core_CudaUMat, is_not_opencl_buffer)
{
    skipIfNoCudaDevice();
    UMat u = makeCudaUMat();
    Mat(8, 8, CV_32FC1, Scalar(1)).copyTo(u);
    ASSERT_TRUE(u.u != NULL);
    EXPECT_EQ(cv::cuda::getCudaAllocator(), u.u->currAllocator);
#ifdef HAVE_OPENCL
    EXPECT_NE((MatAllocator*)cv::ocl::getOpenCLAllocator(), u.u->currAllocator);
#endif
}

TEST(Core_CudaUMat, op_arithmetic_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(50, 40, CV_32FC1), b(50, 40, CV_32FC1);
    randu(a, 0, 10); randu(b, 0, 10);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);

    UMat usum = makeCudaUMat(), udiff = makeCudaUMat(), uprod = makeCudaUMat();
    cv::add(ua, ub, usum);
    cv::subtract(ua, ub, udiff);
    cv::multiply(ua, ub, uprod);

    Mat sum, diff, prod;
    usum.copyTo(sum); udiff.copyTo(diff); uprod.copyTo(prod);
    EXPECT_LE(cvtest::norm(sum,  a + b,       NORM_INF), 1e-5);
    EXPECT_LE(cvtest::norm(diff, a - b,       NORM_INF), 1e-5);
    EXPECT_LE(cvtest::norm(prod, a.mul(b),    NORM_INF), 1e-4);
}

TEST(Core_CudaUMat, op_convertTo_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(30, 45, CV_32FC1);
    randu(a, 0, 100);

    UMat ua = makeCudaUMat();
    a.copyTo(ua);
    UMat u8 = makeCudaUMat();
    ua.convertTo(u8, CV_8U, 2.0, 1.0);

    Mat got, expected;
    u8.copyTo(got);
    a.convertTo(expected, CV_8U, 2.0, 1.0);
    EXPECT_EQ(0.0, cvtest::norm(got, expected, NORM_INF));
}

TEST(Core_CudaUMat, op_addWeighted_host_fallback)
{
    skipIfNoCudaDevice();
    NoOpenCL guard;

    Mat a(40, 40, CV_32FC1), b(40, 40, CV_32FC1);
    randu(a, 0, 5); randu(b, 0, 5);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);

    UMat ur = makeCudaUMat();
    cv::addWeighted(ua, 2.0, ub, 1.0, 0.0, ur);

    Mat got, expected;
    ur.copyTo(got);
    cv::addWeighted(a, 2.0, b, 1.0, 0.0, expected);
    EXPECT_LE(cvtest::norm(got, expected, NORM_INF), 1e-4);
}

#ifdef HAVE_OPENCL
// turns OpenCL on for one test and restores the previous state afterwards
struct WithOpenCL
{
    bool prev;
    WithOpenCL() : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(true); }
    ~WithOpenCL() { cv::ocl::setUseOpenCL(prev); }
};

// with OpenCL on, the OpenCL kernel path declines CUDA UMats and the op finishes on the CPU
TEST(Core_CudaUMat, op_opencl_on_falls_back)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat a(50, 40, CV_32FC1), b(50, 40, CV_32FC1);
    randu(a, 0, 10); randu(b, 0, 10);

    UMat ua = makeCudaUMat(), ub = makeCudaUMat(), usum = makeCudaUMat();
    a.copyTo(ua); b.copyTo(ub);
    cv::add(ua, ub, usum);

    Mat sum;
    usum.copyTo(sum);
    EXPECT_LE(cvtest::norm(sum, a + b, NORM_INF), 1e-5);
}

// device to device across allocators (CUDA -> OpenCL -> CUDA) goes through a host copy
TEST(Core_CudaUMat, copy_cuda_opencl_cuda)
{
    skipIfNoCudaDevice();
    WithOpenCL guard;
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    Mat src(30, 20, CV_32FC1);
    randu(src, 0, 10);

    UMat cu = makeCudaUMat(), cl, cu2 = makeCudaUMat();
    src.copyTo(cu);
    cu.copyTo(cl);
    EXPECT_EQ(cv::ocl::getOpenCLAllocator(), cl.u->currAllocator);
    cl.copyTo(cu2);

    Mat back;
    cu2.copyTo(back);
    EXPECT_EQ(0.0, cvtest::norm(src, back, NORM_INF));
}
#endif // HAVE_OPENCL

}} // namespace

#endif // HAVE_CUDA
