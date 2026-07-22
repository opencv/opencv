// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// SVM::predict is dominated by the per-feature kernel reduction over the support
// vectors: calc_non_rbf_base (LINEAR/POLY/SIGMOID dot product), calc_rbf
// (squared distance), calc_intersec (min-sum) and calc_chi2. Train data is kept
// non-negative so the INTER and CHI2 kernels stay in their valid domain.
typedef TestBaseWithParam< tuple<int, int, int> > SVMPredict;  // (samples, dims, kernelType)

PERF_TEST_P(SVMPredict, kernels, testing::Combine(
            testing::Values(1500),
            testing::Values(512),
            testing::Values((int)SVM::RBF, (int)SVM::POLY, (int)SVM::INTER, (int)SVM::CHI2)))
{
    const int nsamples = get<0>(GetParam());
    const int dims     = get<1>(GetParam());
    const int kernel   = get<2>(GetParam());
    const int nquery   = 1000;

    Mat train(nsamples, dims, CV_32F), query(nquery, dims, CV_32F);
    Mat responses(nsamples, 1, CV_32S);
    RNG& rng = theRNG();
    rng.fill(train, RNG::UNIFORM, 0.f, 1.f);
    rng.fill(query, RNG::UNIFORM, 0.f, 1.f);
    for (int i = 0; i < nsamples; i++)
        responses.at<int>(i) = i & 1;

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(kernel);
    svm->setC(1);
    svm->setGamma(0.1);
    svm->setDegree(3);   // POLY
    svm->setCoef0(1);    // POLY
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-3));
    svm->train(train, ROW_SAMPLE, responses);

    Mat results;
    TEST_CYCLE() svm->predict(query, results);

    SANITY_CHECK_NOTHING();
}

}} // namespace
