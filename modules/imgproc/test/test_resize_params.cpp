// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat makeTestImage(int type, Size sz, int seed)
{
    Mat img(sz, type);
    RNG rng(seed);
    rng.fill(img, RNG::UNIFORM, 0, 255);
    return img;
}

TEST(Resize_Params, BackwardCompat)
{
    const int interpolations[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4 };
    const Size srcSize(173, 121), dstSize(64, 96);

    for (int interp : interpolations)
    {
        Mat src = makeTestImage(CV_8UC3, srcSize, 12345 + interp);

        Mat expected, actual;
        resize(src, expected, dstSize, 0, 0, interp);

        ResizeParams params(dstSize, 0, 0, interp);
        resize(src, actual, params);

        EXPECT_EQ(0, cvtest::norm(expected, actual, NORM_INF))
            << "interpolation=" << interp;
    }

    {
        const Size bitExactSrcSize(342, 256), bitExactDstSize(171, 128);
        Mat src = makeTestImage(CV_8UC3, bitExactSrcSize, 777);

        for (int interp : { INTER_LINEAR_EXACT, INTER_NEAREST_EXACT })
        {
            Mat expected, actual;
            resize(src, expected, bitExactDstSize, 0, 0, interp);
            resize(src, actual, ResizeParams(bitExactDstSize, 0, 0, interp));
            EXPECT_EQ(0, cvtest::norm(expected, actual, NORM_INF)) << "interpolation=" << interp;
        }
    }

    {
        // The batch kinds accept the classic argument list too, and both spellings must agree.
        const int N = 2, C = 3, inH = 20, inW = 16, outH = 8, outW = 12;
        int srcSizes[] = { N, C, inH, inW };
        Mat ndSrc(4, srcSizes, CV_32F);
        RNG rng(99);
        rng.fill(ndSrc, RNG::UNIFORM, 0, 255);

        Mat viaClassic, viaParams;
        resize(ndSrc, viaClassic, Size(outW, outH), 0, 0, INTER_LINEAR);
        resize(ndSrc, viaParams, ResizeParams(Size(outW, outH)));
        ASSERT_EQ(viaClassic.dims, 4);
        EXPECT_EQ(0, cvtest::norm(viaClassic, viaParams, NORM_INF));

        const Size vecDstSize(48, 64);
        std::vector<Mat> srcs = {
            makeTestImage(CV_8UC3, Size(100, 100), 4),
            makeTestImage(CV_8UC3, Size(64, 200), 5),
        };
        std::vector<Mat> viaClassicVec, viaParamsVec;
        resize(srcs, viaClassicVec, vecDstSize, 0, 0, INTER_LINEAR);
        resize(srcs, viaParamsVec, ResizeParams(vecDstSize));
        ASSERT_EQ(srcs.size(), viaClassicVec.size());
        ASSERT_EQ(srcs.size(), viaParamsVec.size());
        for (size_t i = 0; i < srcs.size(); i++)
            EXPECT_EQ(0, cvtest::norm(viaClassicVec[i], viaParamsVec[i], NORM_INF)) << "index=" << i;
    }
}

TEST(Resize_Params, RejectsInvalidInputs)
{
    Mat src = makeTestImage(CV_8UC3, Size(64, 64), 1), dst;

    // The bit-exact kernels have no coordinate-mode parameter, so they only serve PIXEL_CENTER.
    for (int interp : { INTER_LINEAR_EXACT, INTER_NEAREST_EXACT })
    {
        ResizeParams exactCoordMode(Size(32, 32), 0, 0, interp);
        exactCoordMode.coordMode = ResizeCoord::HALF_PIXEL;
        EXPECT_THROW(resize(src, dst, exactCoordMode), cv::Exception) << "interpolation=" << interp;
    }

    ResizeParams coordModeArea(Size(32, 32), 0, 0, INTER_AREA);
    coordModeArea.coordMode = ResizeCoord::HALF_PIXEL;
    EXPECT_THROW(resize(src, dst, coordModeArea), cv::Exception);

    // The separable kernels only exist for the depths classic resize supports, and a coordMode
    // does not add any: CV_8S and CV_32S have no INTER_LINEAR/CUBIC/LANCZOS4 kernel either way.
    for (int type : { CV_8SC1, CV_32SC1 })
    {
        Mat narrow = makeTestImage(type, Size(64, 64), 2), narrowDst;
        for (int interp : { INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4 })
        {
            ResizeParams params(Size(32, 32), 0, 0, interp);
            params.coordMode = ResizeCoord::HALF_PIXEL;
            EXPECT_THROW(resize(narrow, narrowDst, params), cv::Exception)
                << typeToString(type) << " interpolation=" << interp;
        }
        // INTER_NEAREST only copies pixels, so it takes any depth.
        ResizeParams nearest(Size(32, 32), 0, 0, INTER_NEAREST);
        nearest.coordMode = ResizeCoord::HALF_PIXEL;
        EXPECT_NO_THROW(resize(narrow, narrowDst, nearest)) << typeToString(type);
    }

    ResizeParams antialias(Size(32, 32));
    antialias.antialias = true;
    EXPECT_THROW(resize(src, dst, antialias), cv::Exception);

    ResizeParams antialiasWithCoordMode(Size(32, 32));
    antialiasWithCoordMode.coordMode = ResizeCoord::HALF_PIXEL;
    antialiasWithCoordMode.antialias = true;
    EXPECT_THROW(resize(src, dst, antialiasWithCoordMode), cv::Exception);

    int ndSizes[] = { 2, 3, 10, 10 };
    Mat ndSrc(4, ndSizes, CV_32F), ndDst;
    EXPECT_THROW(resize(ndSrc, ndDst, ResizeParams()), cv::Exception);   // neither dsize nor fx/fy

    std::vector<Mat> mismatched = {
        makeTestImage(CV_8UC3, Size(64, 64), 1),
        makeTestImage(CV_8UC1, Size(64, 64), 2),
    };
    std::vector<Mat> vecDst;
    EXPECT_THROW(resize(mismatched, vecDst, ResizeParams(Size(32, 32))), cv::Exception);

    std::vector<Mat> uniform = { makeTestImage(CV_8UC3, Size(64, 64), 3) };
    EXPECT_THROW(resize(uniform, vecDst, ResizeParams()), cv::Exception);
}

// ---- reference: the ONNX Resize definitions, spelled out independently of resize.cpp ----

// One axis of a resize.
struct AxisGeom
{
    int inLen, outLen;
    double scale;       // source units per destination unit
    double trueOutLen;  // inLen*fx when an ONNX mode was given fx (the unfloored length), else outLen
};

// The ResizeParams rules: an empty dsize is round(size*f), and a given dsize sets the scale, except
// that the ONNX modes keep a given fx/fy, since their dsize is usually floor(size*f).
static void referenceGeometry(const ResizeParams& p, Size ssize, Size& dsize, AxisGeom& gx, AxisGeom& gy)
{
    dsize = p.dsize.empty() ? Size(saturate_cast<int>(ssize.width*p.fx), saturate_cast<int>(ssize.height*p.fy))
                            : p.dsize;
    const bool onnxScale = p.coordMode != ResizeCoord::PIXEL_CENTER && p.fx > 0 && p.fy > 0;
    const bool fromScale = p.dsize.empty() || onnxScale;

    gx.inLen = ssize.width;  gx.outLen = dsize.width;
    gy.inLen = ssize.height; gy.outLen = dsize.height;
    gx.scale = fromScale ? 1./p.fx : (double)ssize.width/dsize.width;
    gy.scale = fromScale ? 1./p.fy : (double)ssize.height/dsize.height;
    gx.trueOutLen = onnxScale ? ssize.width*p.fx : dsize.width;
    gy.trueOutLen = onnxScale ? ssize.height*p.fy : dsize.height;
}

// ONNX coordinate_transformation_mode; PIXEL_CENTER is OpenCV's half-pixel formula.
static double referenceSrcCoord(ResizeCoord mode, const AxisGeom& g, int dst)
{
    switch (mode)
    {
    case ResizeCoord::ASYMMETRIC:
        return dst*g.scale;
    case ResizeCoord::ALIGN_CORNERS:
        return g.trueOutLen > 1 ? dst*(g.inLen - 1)/(g.trueOutLen - 1) : 0.0;
    case ResizeCoord::TF_HALF_PIXEL_FOR_NN:
        return (dst + 0.5)*g.scale;
    case ResizeCoord::PYTORCH_HALF_PIXEL:
        return g.outLen > 1 ? (dst + 0.5)*g.scale - 0.5 : 0.0;
    case ResizeCoord::HALF_PIXEL_SYMMETRIC:
        return g.inLen*0.5 - g.outLen*g.scale*0.5 + (dst + 0.5)*g.scale - 0.5;
    default:
        return (dst + 0.5)*g.scale - 0.5;
    }
}

// ONNX nearest_mode; an exact .5 (within 1e-6, as in the ONNX layer) goes the way the mode says.
static int referenceNearestIndex(double src, int inLen, ResizeNearest mode)
{
    const int last = inLen - 1;
    switch (mode)
    {
    case ResizeNearest::FLOOR: return (int)std::floor(std::min(std::max(src, 0.0), (double)last));
    case ResizeNearest::CEIL:  return (int)std::ceil(std::min(std::max(src, 0.0), (double)last));
    default:
        {
            const int f = (int)std::floor(src);
            const int tieUp = mode == ResizeNearest::ROUND_PREFER_CEIL ? 1 : 0;
            const int idx = std::abs(src - f - 0.5) <= 1e-6 ? f + tieUp : cvRound(src);
            return std::min(std::max(idx, 0), last);
        }
    }
}

static void referenceCubicWeights(float x, float A, float w[4])
{
    w[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    w[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    w[3] = 1.f - w[0] - w[1] - w[2];
}

// Mirrors resize.cpp's truncated-Lanczos weights: unlike Keys cubic they do not reproduce a
// linear ramp exactly, so the reference cannot shortcut them.
static void referenceLanczos4Weights(float x, float w[8])
{
    const double s45 = 0.70710678118654752440084436210485;
    const double cs[][2] = {{1,0},{-s45,-s45},{0,1},{s45,-s45},{-1,0},{s45,s45},{0,-1},{-s45,s45}};

    const double y0 = -(x + 3)*CV_PI*0.25, s0 = std::sin(y0), c0 = std::cos(y0);
    float sum = 0.f;
    for (int i = 0; i < 8; i++)
    {
        const float yi = x + 3 - i;
        if (std::fabs(yi) >= 1e-6f)
        {
            const double y = -yi*CV_PI*0.25;
            w[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        }
        else
            w[i] = 1e30f;
        sum += w[i];
    }
    sum = 1.f/sum;
    for (int i = 0; i < 8; i++)
        w[i] *= sum;
}

// Taps of one axis of a separable kernel: ksize source indices and weights per output sample.
// Linear clamps the coordinate; the wider kernels replicate the edge sample, or with
// excludeOutside drop the taps that fall outside and renormalise the rest.
static int referenceTaps(int interp, ResizeCoord mode, const AxisGeom& g, float A, bool excludeOutside,
                         std::vector<int>& ofs, std::vector<double>& w)
{
    const int ksize = interp == INTER_LINEAR ? 2 : interp == INTER_CUBIC ? 4 : 8;
    ofs.assign((size_t)g.outLen*ksize, 0);
    w.assign((size_t)g.outLen*ksize, 0.);
    for (int i = 0; i < g.outLen; i++)
    {
        double c = referenceSrcCoord(mode, g, i);
        int* o = &ofs[(size_t)i*ksize];
        double* wt = &w[(size_t)i*ksize];
        if (interp == INTER_LINEAR)
        {
            c = std::min(std::max(c, 0.0), g.inLen - 1.0);
            const int i0 = (int)std::floor(c);
            o[0] = i0;
            o[1] = std::min(i0 + 1, g.inLen - 1);
            wt[1] = c - i0;
            wt[0] = 1. - wt[1];
            continue;
        }
        const int i0 = (int)std::floor(c);
        float wf[8];
        if (interp == INTER_CUBIC)
            referenceCubicWeights((float)(c - i0), A, wf);
        else
            referenceLanczos4Weights((float)(c - i0), wf);
        double sum = 0.;
        for (int k = 0; k < ksize; k++)
        {
            const int idx = i0 - ksize/2 + 1 + k;
            const bool inside = 0 <= idx && idx < g.inLen;
            wt[k] = excludeOutside && !inside ? 0. : (double)wf[k];
            o[k] = std::min(std::max(idx, 0), g.inLen - 1);
            sum += wt[k];
        }
        if (excludeOutside && sum != 0.)
            for (int k = 0; k < ksize; k++)
                wt[k] /= sum;
    }
    return ksize;
}

// Separable resize in double, rounded to the source depth at the end.
static Mat referenceSeparable(const Mat& src, const ResizeParams& p, Size dsize,
                              const AxisGeom& gx, const AxisGeom& gy)
{
    std::vector<int> xo, yo;
    std::vector<double> xw, yw;
    const int kx = referenceTaps(p.interpolation, p.coordMode, gx, p.cubicCoeffA, p.excludeOutside, xo, xw);
    const int ky = referenceTaps(p.interpolation, p.coordMode, gy, p.cubicCoeffA, p.excludeOutside, yo, yw);

    const int cn = src.channels();
    Mat src64;
    src.convertTo(src64, CV_64F);
    Mat dst64(dsize, CV_64FC(cn));
    for (int y = 0; y < dsize.height; y++)
        for (int x = 0; x < dsize.width; x++)
            for (int c = 0; c < cn; c++)
            {
                double acc = 0.;
                for (int j = 0; j < ky; j++)
                {
                    const double* row = src64.ptr<double>(yo[(size_t)y*ky + j]);
                    double hsum = 0.;
                    for (int i = 0; i < kx; i++)
                        hsum += xw[(size_t)x*kx + i]*row[xo[(size_t)x*kx + i]*cn + c];
                    acc += yw[(size_t)y*ky + j]*hsum;
                }
                dst64.ptr<double>(y)[x*cn + c] = acc;
            }
    Mat dst;
    dst64.convertTo(dst, src.type());
    return dst;
}

// The separable kernels round their coordinates to float and, for 8U, their weights to 11-bit fixed
// point; the 8U bound grows with the number of taps those rounded weights add up.
static double referenceTolerance(int depth, int interp)
{
    if (depth == CV_32F || depth == CV_64F)
        return 1e-2;   // samples span [-100, 100)
    if (depth == CV_8U)
        return interp == INTER_LANCZOS4 ? 3 : interp == INTER_CUBIC ? 2 : 1;
    return 1;
}

// INTER_NEAREST copies samples, so it must match exactly. When a coordinate sits on a rounding
// boundary, either neighbour is accepted: the 1e-9 window only absorbs how the two sides evaluate the
// same formula.
static void checkNearest(const Mat& src, const Mat& dst, const ResizeParams& p,
                         const AxisGeom& gx, const AxisGeom& gy)
{
    const bool classic = p.coordMode == ResizeCoord::PIXEL_CENTER;
    const ResizeCoord mode = classic ? ResizeCoord::ASYMMETRIC : p.coordMode;
    const ResizeNearest rounding = classic ? ResizeNearest::FLOOR : p.nearestMode;

    std::vector<int> xlo(gx.outLen), xhi(gx.outLen), ylo(gy.outLen), yhi(gy.outLen);
    for (int x = 0; x < gx.outLen; x++)
    {
        const double c = referenceSrcCoord(mode, gx, x);
        xlo[x] = referenceNearestIndex(c - 1e-9, gx.inLen, rounding);
        xhi[x] = referenceNearestIndex(c + 1e-9, gx.inLen, rounding);
    }
    for (int y = 0; y < gy.outLen; y++)
    {
        const double c = referenceSrcCoord(mode, gy, y);
        ylo[y] = referenceNearestIndex(c - 1e-9, gy.inLen, rounding);
        yhi[y] = referenceNearestIndex(c + 1e-9, gy.inLen, rounding);
    }

    const size_t esz = src.elemSize();
    for (int y = 0; y < gy.outLen; y++)
        for (int x = 0; x < gx.outLen; x++)
        {
            const uchar* d = dst.ptr(y) + x*esz;
            const bool found =
                memcmp(d, src.ptr(ylo[y]) + xlo[x]*esz, esz) == 0 ||
                memcmp(d, src.ptr(ylo[y]) + xhi[x]*esz, esz) == 0 ||
                memcmp(d, src.ptr(yhi[y]) + xlo[x]*esz, esz) == 0 ||
                memcmp(d, src.ptr(yhi[y]) + xhi[x]*esz, esz) == 0;
            ASSERT_TRUE(found) << "dst(" << y << ", " << x << ") is not source pixel ("
                               << ylo[y] << ", " << xlo[x] << ")";
        }
}

static void checkAgainstReference(const Mat& src, const Mat& dst, const ResizeParams& p)
{
    Size dsize;
    AxisGeom gx, gy;
    referenceGeometry(p, src.size(), dsize, gx, gy);
    ASSERT_EQ(dsize, dst.size());
    ASSERT_EQ(src.type(), dst.type());

    if (p.interpolation == INTER_NEAREST)
    {
        checkNearest(src, dst, p, gx, gy);
        return;
    }

    Mat expected;
    if (p.interpolation == INTER_AREA)
        cv::resize(src, expected, p.dsize, p.fx, p.fy, INTER_AREA);   // classic resize is the reference
    else
        expected = referenceSeparable(src, p, dsize, gx, gy);
    EXPECT_LE(cvtest::norm(expected, dst, NORM_INF), referenceTolerance(src.depth(), p.interpolation));
}

// Resizes one image through a one-element batch, which always runs OpenCV's own kernels.
static Mat resizeAlone(const Mat& src, const ResizeParams& p)
{
    std::vector<Mat> one(1, src), out;
    resize(one, out, p);
    return out[0];
}

// ---- random configurations ----

enum InputKind { KIND_IMAGE, KIND_TENSOR, KIND_VECTOR };

struct RandomCase
{
    ResizeParams params;
    int type;
    InputKind kind;
    std::vector<Size> sizes;        // a size per image; they differ only in a ragged vector
    std::vector<int> leadingDims;   // KIND_TENSOR: the batch axes in front of height and width
    bool roi;                       // KIND_IMAGE and KIND_VECTOR: images are views into padded Mats

    std::string describe() const
    {
        std::ostringstream os;
        os << typeToString(type) << " interpolation=" << params.interpolation
           << " coordMode=" << (int)params.coordMode << " nearestMode=" << (int)params.nearestMode
           << " cubicCoeffA=" << params.cubicCoeffA << " excludeOutside=" << params.excludeOutside
           << " dsize=" << params.dsize << " fx=" << params.fx << " fy=" << params.fy
           << " kind=" << (int)kind << " roi=" << roi << " sizes=";
        for (const Size& sz : sizes)
            os << sz << ' ';
        for (int d : leadingDims)
            os << "lead=" << d << ' ';
        return os.str();
    }
};

// Exact ratios half the time: the integer ones are where the kernels take shortcuts.
static double drawScale(RNG& rng)
{
    static const double exact[] = { 0.25, 0.5, 1.0, 2.0, 3.0 };
    if (rng.uniform(0, 2) == 0)
        return exact[rng.uniform(0, 5)];
    return std::exp(rng.uniform(std::log(0.2), std::log(4.0)));
}

static RandomCase drawCase(RNG& rng)
{
    static const int interps[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA };
    static const int anyDepth[] = { CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F };
    static const int filterDepth[] = { CV_8U, CV_16U, CV_16S, CV_32F, CV_64F };

    RandomCase c;
    ResizeParams& p = c.params;
    p.interpolation = interps[rng.uniform(0, 5)];
    // INTER_AREA has no coordinate mode; INTER_NEAREST copies samples, so it takes any depth.
    p.coordMode = p.interpolation == INTER_AREA ? ResizeCoord::PIXEL_CENTER : (ResizeCoord)rng.uniform(0, 7);
    p.nearestMode = (ResizeNearest)rng.uniform(0, 4);
    p.cubicCoeffA = rng.uniform(0, 2) ? -0.75f : rng.uniform(-1.f, -0.5f);
    p.excludeOutside = rng.uniform(0, 2) == 1;

    const int depth = p.interpolation == INTER_NEAREST ? anyDepth[rng.uniform(0, 7)] : filterDepth[rng.uniform(0, 5)];
    c.type = CV_MAKETYPE(depth, rng.uniform(1, 5));
    c.kind = (InputKind)rng.uniform(0, 3);
    c.roi = c.kind != KIND_TENSOR && rng.uniform(0, 2) == 1;

    const int count = c.kind == KIND_VECTOR ? rng.uniform(1, 4) : 1;
    const bool ragged = c.kind == KIND_VECTOR && rng.uniform(0, 2) == 1;
    c.sizes.push_back(Size(rng.uniform(1, 41), rng.uniform(1, 41)));
    for (int i = 1; i < count; i++)
        c.sizes.push_back(ragged ? Size(rng.uniform(1, 41), rng.uniform(1, 41)) : c.sizes[0]);
    if (c.kind == KIND_TENSOR)
        for (int i = rng.uniform(1, 3); i > 0; i--)
            c.leadingDims.push_back(rng.uniform(1, 4));

    // Half the cases scale both axes alike, as most callers do.
    double sx = drawScale(rng), sy = rng.uniform(0, 2) ? sx : drawScale(rng);
    if (rng.uniform(0, 20) == 0)
        sx = sy = 1.0;
    int minW = INT_MAX, minH = INT_MAX;
    for (const Size& sz : c.sizes)
        minW = std::min(minW, sz.width), minH = std::min(minH, sz.height);

    // The geometry comes as dsize alone, as fx/fy alone, or as both, the way the ONNX layer passes
    // it: the true scale plus the floored size.
    const Size first = c.sizes[0];
    switch (rng.uniform(0, 3))
    {
    case 0:
        p.dsize = Size(std::max(1, cvRound(first.width*sx)), std::max(1, cvRound(first.height*sy)));
        break;
    case 1:
        p.fx = std::max(sx, 1./minW);
        p.fy = std::max(sy, 1./minH);
        break;
    default:
        p.fx = std::max(sx, 1./minW);
        p.fy = std::max(sy, 1./minH);
        p.dsize = Size(std::max(1, cvFloor(first.width*p.fx)), std::max(1, cvFloor(first.height*p.fy)));
        break;
    }
    return c;
}

static void fillRandom(Mat& m, RNG& rng)
{
    switch (m.depth())
    {
    case CV_8U:  rng.fill(m, RNG::UNIFORM, 0, 256); break;
    case CV_8S:  rng.fill(m, RNG::UNIFORM, -128, 128); break;
    case CV_16U: rng.fill(m, RNG::UNIFORM, 0, 1024); break;
    case CV_16S: rng.fill(m, RNG::UNIFORM, -512, 512); break;
    case CV_32S: rng.fill(m, RNG::UNIFORM, -1000000, 1000000); break;
    default:     rng.fill(m, RNG::UNIFORM, -100., 100.); break;
    }
}

// An image of the given size, possibly a view into a larger Mat so that its step is not width*esz.
static Mat makeImage(int type, Size sz, bool roi, RNG& rng)
{
    if (!roi)
    {
        Mat m(sz, type);
        fillRandom(m, rng);
        return m;
    }
    Mat padded(sz.height + 3, sz.width + 5, type);
    fillRandom(padded, rng);
    return padded(Rect(2, 1, sz.width, sz.height));
}

// Builds the case's input, resizes it, and returns every source plane with its result. A tensor's
// planes are views that do not own their data, so the tensors go to keepAlive.
static void runCase(const RandomCase& c, RNG& rng, std::vector<Mat>& srcPlanes, std::vector<Mat>& dstPlanes,
                    std::vector<Mat>& keepAlive)
{
    if (c.kind == KIND_IMAGE)
    {
        Mat src = makeImage(c.type, c.sizes[0], c.roi, rng), dst;
        resize(src, dst, c.params);
        srcPlanes.push_back(src);
        dstPlanes.push_back(dst);
        return;
    }
    if (c.kind == KIND_VECTOR)
    {
        std::vector<Mat> srcs, dsts;
        for (const Size& sz : c.sizes)
            srcs.push_back(makeImage(c.type, sz, c.roi, rng));
        resize(srcs, dsts, c.params);
        ASSERT_EQ(srcs.size(), dsts.size());
        srcPlanes = srcs;
        dstPlanes = dsts;
        return;
    }

    const Size sz = c.sizes[0];
    std::vector<int> shape = c.leadingDims;
    shape.push_back(sz.height);
    shape.push_back(sz.width);
    Mat src((int)shape.size(), shape.data(), c.type);
    fillRandom(src, rng);

    Mat dst;
    resize(src, dst, c.params);
    ASSERT_EQ(src.dims, dst.dims);
    for (int i = 0; i < src.dims - 2; i++)
        ASSERT_EQ(src.size[i], dst.size[i]);

    keepAlive.push_back(src);
    keepAlive.push_back(dst);
    const Size dsize(dst.size[dst.dims - 1], dst.size[dst.dims - 2]);
    const size_t planes = src.total()/sz.area();
    for (size_t i = 0; i < planes; i++)
    {
        srcPlanes.push_back(Mat(sz, c.type, src.ptr() + i*sz.area()*src.elemSize()));
        dstPlanes.push_back(Mat(dsize, c.type, dst.ptr() + i*dsize.area()*dst.elemSize()));
    }
}

// A fixed number of cases, each drawing every parameter at once, against an independent reference.
// A batch must also give each image exactly what it gets alone, and HALF_PIXEL, whose formula is
// PIXEL_CENTER's, must match it bit for bit: the coordinate modes share every kernel.
TEST(Resize_Params, RandomConfigs)
{
    RNG rng(0x7e5a1e);
    const int ncases = 200;
    for (int iter = 0; iter < ncases; iter++)
    {
        const RandomCase c = drawCase(rng);
        SCOPED_TRACE(cv::format("case %d: ", iter) + c.describe());

        std::vector<Mat> srcPlanes, dstPlanes, keepAlive;
        ASSERT_NO_FATAL_FAILURE(runCase(c, rng, srcPlanes, dstPlanes, keepAlive));
        ASSERT_EQ(srcPlanes.size(), dstPlanes.size());

        for (size_t i = 0; i < srcPlanes.size(); i++)
        {
            SCOPED_TRACE(cv::format("plane %d", (int)i));
            const Mat& src = srcPlanes[i];
            const Mat& dst = dstPlanes[i];
            ASSERT_NO_FATAL_FAILURE(checkAgainstReference(src, dst, c.params));

            if (c.kind == KIND_IMAGE)
                continue;
            EXPECT_EQ(0, cvtest::norm(resizeAlone(src, c.params), dst, NORM_INF)) << "batch differs from a lone image";

            const int interp = c.params.interpolation;
            if (c.params.coordMode == ResizeCoord::HALF_PIXEL && c.params.fx == 0 && dst.size() != src.size() &&
                (interp == INTER_LINEAR || interp == INTER_CUBIC || interp == INTER_LANCZOS4))
            {
                ResizeParams pixelCenter = c.params;
                pixelCenter.coordMode = ResizeCoord::PIXEL_CENTER;
                EXPECT_EQ(0, cvtest::norm(resizeAlone(src, pixelCenter), dst, NORM_INF)) << "HALF_PIXEL differs from PIXEL_CENTER";
            }
        }
        if (HasFailure())
            return;
    }
}

// Values worked out by hand from the ONNX Resize reference, with the scale given as fx/fy: dsize is
// floor(size*scale), so the scale cannot be recovered from it.
TEST(Resize_Params, KnownValues)
{
    {
        // ONNX test_resize_downsample_scales_linear_align_corners
        Mat src = Mat_<float>({2, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}), dst;
        ResizeParams params(Size(2, 1), 0.6, 0.6);
        params.coordMode = ResizeCoord::ALIGN_CORNERS;
        resize(src, dst, params);
        ASSERT_EQ(Size(2, 1), dst.size());
        EXPECT_NEAR(1.f, dst.at<float>(0, 0), 1e-5);
        EXPECT_NEAR(3.142857f, dst.at<float>(0, 1), 1e-5);
    }
    {
        // ONNX test_resize_downsample_scales_linear_half_pixel_symmetric
        Mat src = Mat_<float>({1, 4}, {1.f, 2.f, 3.f, 4.f}), dst;
        ResizeParams params(Size(2, 1), 0.6, 1.0);
        params.coordMode = ResizeCoord::HALF_PIXEL_SYMMETRIC;
        resize(src, dst, params);
        ASSERT_EQ(Size(2, 1), dst.size());
        EXPECT_NEAR(1.6666666f, dst.at<float>(0, 0), 1e-5);
        EXPECT_NEAR(3.3333333f, dst.at<float>(0, 1), 1e-5);
    }
    {
        // An exact 2x linear downscale averages 2x2 blocks only when sampling at pixel centres;
        // asymmetric samples land on src[2x] itself.
        Mat src = Mat_<float>({2, 8}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
                                       0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}), dst;
        ResizeParams params(Size(4, 1));
        params.coordMode = ResizeCoord::ASYMMETRIC;
        resize(src, dst, params);
        EXPECT_EQ(0, cvtest::norm(dst, Mat_<float>({1, 4}, {0.f, 2.f, 4.f, 6.f}), NORM_INF));
        params.coordMode = ResizeCoord::HALF_PIXEL;
        resize(src, dst, params);
        EXPECT_EQ(0, cvtest::norm(dst, Mat_<float>({1, 4}, {0.5f, 2.5f, 4.5f, 6.5f}), NORM_INF));

        // fx/fy pin the scale, so dsize may ask for more than the source covers. The samples past
        // the edge repeat it, at the exact 2x too, where there are no 2x2 blocks left to average.
        Mat ramp4 = repeat(Mat_<float>({1, 4}, {0.f, 1.f, 2.f, 3.f}), 4, 1);
        ResizeParams pinned(Size(4, 4), 0.5, 0.5);
        pinned.coordMode = ResizeCoord::HALF_PIXEL;
        resize(ramp4, dst, pinned);
        EXPECT_EQ(0, cvtest::norm(dst, repeat(Mat_<float>({1, 4}, {0.5f, 2.5f, 3.f, 3.f}), 4, 1), NORM_INF));
    }
    {
        // A lone PIXEL_CENTER image keeps cubicCoeffA and excludeOutside, which classic resize,
        // and the HAL behind it, would drop.
        Mat src = makeTestImage(CV_8UC3, Size(37, 29), 5), single;
        ResizeParams params(Size(23, 17), 0, 0, INTER_CUBIC);
        params.cubicCoeffA = -0.5f;
        params.excludeOutside = true;
        resize(src, single, params);
        EXPECT_EQ(0, cvtest::norm(single, resizeAlone(src, params), NORM_INF));
    }
    {
        // CV_64F accumulates in double, as classic cv::resize does, though its weights are float:
        // detail far below float resolution survives, the level is only float-weight accurate.
        Mat src(1, 9, CV_64FC1), dst;
        for (int x = 0; x < src.cols; x++)
            src.at<double>(0, x) = 1.0 + x * 1e-12;
        for (int interp : { INTER_LINEAR, INTER_CUBIC })
        {
            ResizeParams params(Size(5, 1), 0, 0, interp);
            params.coordMode = ResizeCoord::HALF_PIXEL;
            resize(src, dst, params);
            ASSERT_EQ(CV_64F, dst.depth());
            double lo, hi;
            minMaxLoc(dst, &lo, &hi);
            EXPECT_GT(hi - lo, 1e-13) << "interpolation=" << interp << ": detail lost to float";
            EXPECT_LT(std::abs(hi - 1.0), 1e-6) << "interpolation=" << interp;
            EXPECT_LT(std::abs(lo - 1.0), 1e-6) << "interpolation=" << interp;
        }
    }
}

// A vector of UMat goes image by image through the OpenCL path.
TEST(Resize_Params, UMatVector)
{
    const Size dstSize(48, 64);
    std::vector<UMat> srcs(2);
    makeTestImage(CV_8UC3, Size(120, 90), 11).copyTo(srcs[0]);
    makeTestImage(CV_8UC3, Size(90, 120), 22).copyTo(srcs[1]);

    std::vector<UMat> actual;
    resize(srcs, actual, ResizeParams(dstSize));

    ASSERT_EQ(srcs.size(), actual.size());
    for (size_t i = 0; i < srcs.size(); i++)
    {
        UMat expected;
        resize(srcs[i], expected, dstSize, 0, 0, INTER_LINEAR);
        EXPECT_EQ(0, cvtest::norm(expected.getMat(ACCESS_READ), actual[i].getMat(ACCESS_READ), NORM_INF)) << "index=" << i;
    }
}

}} // namespace
