// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../odometry/vo_impl.hpp"

namespace cv {
namespace slam {

// project a VLAD vector into a compact binary code via random projections
static std::vector<uint8_t> lshHash(const Mat& vlad, const Mat& proj)
{
    if (proj.empty() || vlad.empty()) return {};
    Mat p = proj * vlad.t();
    const int B = proj.rows;
    std::vector<uint8_t> code((B + 7) / 8, 0u);
    for (int i = 0; i < B; ++i)
        if (p.at<float>(i, 0) >= 0.f)
            code[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    return code;
}

// k-means on subsampled keyframe descriptors to build VLAD codebook + LSH projection matrix
void VisualOdometryImpl::buildVocabulary()
{
    int descDim = 0;
    for (KeyFrame* kf : map.keyframes())
        if (!kf->descriptors.empty()) { descDim = kf->descriptors.cols; break; }
    if (descDim < 1) return;

    int totalRows = 0;
    for (KeyFrame* kf : map.keyframes())
        if (kf->descriptors.cols == descDim) totalRows += kf->descriptors.rows;
    if (totalRows < 2) return;

    const int maxTrain = std::max(2, params.loopVladMaxTrain);
    const int stride   = std::max(1, totalRows / maxTrain);

    Mat samples(0, descDim, CV_32F);
    samples.reserve(std::min(totalRows, maxTrain) + 1);
    for (KeyFrame* kf : map.keyframes())
    {
        const Mat& dsrc = kf->descriptors;
        if (dsrc.cols != descDim || dsrc.rows < 1) continue;
        Mat d = dsrc;
        if (d.type() != CV_32F) d.convertTo(d, CV_32F);

        for (int r = 0; r < d.rows && samples.rows < maxTrain; r += stride)
            samples.push_back(d.row(r));
    }
    if (samples.rows < 2) return;

    const int Kv = std::min(params.loopVladK, samples.rows);
    if (Kv < 2) return;

    Mat labels, centers;
    TermCriteria crit(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 1e-4);
    kmeans(samples, Kv, labels, crit, 3, KMEANS_PP_CENTERS, centers);
    if (centers.empty()) return;

    vocab = centers;
    vocabReady = true;
    CV_LOG_INFO(NULL, "slam loop: built VLAD vocabulary K=" << Kv
                      << " D=" << descDim << " from " << samples.rows << " descriptors");

    const int B = (params.loopHashBits / 8) * 8;
    if (B > 0)
    {
        const int dim = Kv * vocab.cols;
        Mat W(B, dim, CV_32F);
        cv::randn(W, 0.f, 1.f);
        for (int i = 0; i < B; ++i)
        {
            double n = norm(W.row(i), NORM_L2);
            if (n > 1e-12) W.row(i) /= (float)n;
        }
        hashProj = W;
        CV_LOG_INFO(NULL, "slam loop: built " << B << "-bit LSH projection (" << dim << "D VLAD)");
    }
}

// encode a keyframe's descriptors as one fixed-size unit VLAD vector
Mat VisualOdometryImpl::computeVlad(const Mat& descriptorsIn) const
{
    if (vocab.empty() || descriptorsIn.empty()) return Mat();

    Mat desc = descriptorsIn;
    if (desc.type() != CV_32F) desc.convertTo(desc, CV_32F);
    if (desc.cols != vocab.cols) return Mat();

    const int Kv = vocab.rows, D = vocab.cols, M = desc.rows;

    // argmax(desc[m]·vocab[k] - 0.5||vocab[k]||²) == argmin squared distance to nearest word
    Mat cNorm2(1, Kv, CV_32F);
    for (int k = 0; k < Kv; ++k)
        cNorm2.at<float>(0, k) = 0.5f * (float)vocab.row(k).dot(vocab.row(k));

    Mat dots = desc * vocab.t();
    for (int k = 0; k < Kv; ++k)
        dots.col(k) -= cNorm2.at<float>(0, k);

    Mat v = Mat::zeros(Kv, D, CV_32F);
    for (int m = 0; m < M; ++m)
    {
        int best = 0; float bestScore = -FLT_MAX;
        const float* row = dots.ptr<float>(m);
        for (int k = 0; k < Kv; ++k)
            if (row[k] > bestScore) { bestScore = row[k]; best = k; }

        v.row(best) += desc.row(m) - vocab.row(best);
    }

    Mat vf = v.reshape(1, 1);
    for (int i = 0; i < vf.cols; ++i)
    {
        float& x = vf.at<float>(0, i);
        x = (x >= 0.f ? 1.f : -1.f) * std::sqrt(std::abs(x));
    }
    double n = norm(vf, NORM_L2);
    if (n > 1e-12) vf /= n;
    return vf.clone();
}

// Essential matrix RANSAC — returns inlier count for a candidate match pair
int VisualOdometryImpl::geometricVerify(const KeyFrame* q, const KeyFrame* c,
                                        const std::vector<DMatch>& matches) const
{
    std::vector<Point2f> pq, pc;
    pq.reserve(matches.size());
    pc.reserve(matches.size());
    for (const DMatch& m : matches)
    {
        if ((size_t)m.queryIdx >= q->undistKpts.size()) continue;
        if ((size_t)m.trainIdx >= c->undistKpts.size()) continue;
        pq.push_back(q->undistKpts[m.queryIdx]);
        pc.push_back(c->undistKpts[m.trainIdx]);
    }
    if ((int)pq.size() < 5) return 0;

    Mat mask;
    Mat E = findEssentialMat(pq, pc, K, RANSAC,
                             params.essentialRansacConfidence,
                             params.essentialRansacThresh, 1000, mask);
    if (E.empty() || mask.empty()) return 0;
    return countNonZero(mask);
}

void VisualOdometryImpl::detectLoop(KeyFrame* query)
{
    if (!params.loopEnable || !query) return;

    if (map.numKeyframes() <= params.loopMinDbSize) return;

    // build vocabulary once the map is large enough
    if (!vocabReady)
    {
        buildVocabulary();
        if (!vocabReady) return;
    }

    // encode any keyframe that doesn't have a VLAD vector or hash yet
    for (KeyFrame* kf : map.keyframes())
    {
        if (kf->globalDesc.empty() && !kf->descriptors.empty())
            kf->globalDesc = computeVlad(kf->descriptors);
        if (!hashProj.empty() && kf->globalHash.empty() && !kf->globalDesc.empty())
            kf->globalHash = lshHash(kf->globalDesc, hashProj);
    }

    if (query->globalDesc.empty()) return;

    // only match against keyframes old enough to be a genuine loop, not recent neighbours
    const int maxEligibleId = query->id - params.loopRecentGap;
    std::vector<std::pair<float, KeyFrame*>> scored;

    // coarse Hamming filter on binary codes, then cosine rerank on full VLAD vectors
    if (!hashProj.empty() && !query->globalHash.empty())
    {
        const int nbytes   = (int)query->globalHash.size();
        const int coarseK  = std::max(params.loopTopK, params.loopCoarseTopk);
        const Mat queryCode(1, nbytes, CV_8U, (void*)query->globalHash.data());
        std::vector<std::pair<int, KeyFrame*>> ham;
        ham.reserve(256);
        for (KeyFrame* kf : map.keyframes())
        {
            if (kf->id > maxEligibleId || kf->globalDesc.empty() || kf->bad) continue;
            if ((int)kf->globalHash.size() != nbytes) continue;
            const Mat candidateCode(1, nbytes, CV_8U, (void*)kf->globalHash.data());
            ham.emplace_back((int)norm(queryCode, candidateCode, NORM_HAMMING), kf);
        }
        std::sort(ham.begin(), ham.end(),
                  [](const std::pair<int, KeyFrame*>& a,
                     const std::pair<int, KeyFrame*>& b) { return a.first < b.first; });

        const int n = std::min((int)ham.size(), coarseK);
        for (int i = 0; i < n; ++i)
        {
            KeyFrame* kf = ham[i].second;
            float sim = (float)query->globalDesc.dot(kf->globalDesc);
            if (sim >= (float)params.loopMinSimilarity)
                scored.emplace_back(sim, kf);
        }
    }
    else
    {
        // No hash projection available — fall back to brute-force cosine on all KFs
        for (KeyFrame* kf : map.keyframes())
        {
            if (kf->id > maxEligibleId || kf->globalDesc.empty() || kf->bad) continue;
            float sim = (float)query->globalDesc.dot(kf->globalDesc);
            if (sim >= (float)params.loopMinSimilarity)
                scored.emplace_back(sim, kf);
        }
    }
    if (scored.empty()) return;

    std::sort(scored.begin(), scored.end(),
              [](const std::pair<float, KeyFrame*>& a,
                 const std::pair<float, KeyFrame*>& b) { return a.first > b.first; });
    const int topK = std::min((int)scored.size(), std::max(1, params.loopTopK));

    // descriptor match + geometric verify on the top candidates, keep the best
    KeyFrame* bestKf = nullptr;
    int bestInliers = 0, bestRaw = 0;
    float bestSim = 0.f;
    std::vector<DMatch> bestMatches;
    for (int i = 0; i < topK; ++i)
    {
        KeyFrame* cand = scored[i].second;

        std::vector<DMatch> matches;
        matchFrames(query->keypoints, query->descriptors, query->imageSize,
                    cand->keypoints, cand->descriptors, cand->imageSize, matches);
        const int raw = (int)matches.size();
        if (raw < params.loopMinRawMatches) continue;

        const int inliers = geometricVerify(query, cand, matches);
        const double ratio = raw > 0 ? (double)inliers / (double)raw : 0.0;

        if (inliers >= params.loopMinInliers &&
            ratio   >= params.loopMinInlierRatio &&
            inliers >  bestInliers)
        {
            bestKf = cand; bestInliers = inliers; bestRaw = raw;
            bestSim = scored[i].first;
            bestMatches = matches;
        }
    }

    if (!bestKf)
    {
        loopStreak    = 0;
        loopLastCand  = nullptr;
        return;
    }

    // require N consecutive detections pointing to the same region before acting
    const bool sameRegion =
        (loopLastCand == nullptr) ||
        (bestKf == loopLastCand) ||
        (loopLastCand->covisibility.count(bestKf) > 0) ||
        (bestKf->covisibility.count(loopLastCand) > 0);
    if (!sameRegion) loopStreak = 0;
    ++loopStreak;
    loopLastCand = bestKf;

    if (loopStreak < std::max(1, params.loopNConsistent))
        return;

    String ev = format("loop: kf=%d matches kf=%d (sim=%.3f inliers=%d/%d ratio=%.2f) n=%d",
                       query->id, bestKf->id, bestSim,
                       bestInliers, bestRaw, (double)bestInliers / (double)bestRaw,
                       loopStreak);
    lastEvent = lastEvent.empty() ? ev : (lastEvent + " | " + ev);
    CV_LOG_INFO(NULL, "slam " << ev);

    // cooldown — don't trigger another closure too soon after the last one
    if (lastClosedKfId >= 0 &&
        query->id - lastClosedKfId < params.loopCloseCooldown)
        return;

    if (closeLoop(query, bestKf, bestMatches))
    {
        loopStreak   = 0;
        loopLastCand = nullptr;
    }
}

}} // namespace cv::slam
