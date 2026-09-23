// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Detection-driven multi-object tracker. Motion is a constant-velocity cv::KalmanFilter per track,
// association is the two-pass scheme from ByteTrack solved with cv::linearAssignment, and the
// optional appearance term follows DeepSORT/StrongSORT.

#include "../precomp.hpp"

namespace cv {
inline namespace tracking {
namespace impl {

// State is [cx, cy, a, h, vcx, vcy, va, vh] with a = width / height, measurement is [cx, cy, a, h].
static const int STATE_DIM = 8;
static const int MEAS_DIM = 4;

// DeepSORT's noise weights. Both are multiplied by the track height, so a distant object is
// allowed to move less in pixels than a near one.
static const float STD_POSITION = 1.0f / 20.0f;
static const float STD_VELOCITY = 1.0f / 160.0f;

static void boxToMeasurement(const Rect2d& box, Mat& z)
{
    const double h = std::max(box.height, 1e-6);
    z.create(MEAS_DIM, 1, CV_32F);
    z.at<float>(0) = (float)(box.x + box.width * 0.5);
    z.at<float>(1) = (float)(box.y + box.height * 0.5);
    z.at<float>(2) = (float)(box.width / h);
    z.at<float>(3) = (float)h;
}

static Rect2d stateToBox(const Mat& state)
{
    const double cx = state.at<float>(0);
    const double cy = state.at<float>(1);
    const double h = std::max((double)state.at<float>(3), 0.0);
    const double w = std::max((double)state.at<float>(2) * h, 0.0);
    return Rect2d(cx - w * 0.5, cy - h * 0.5, w, h);
}

enum TrackState { TENTATIVE = 0, CONFIRMED = 1, LOST = 2 };

// Held only through a Ptr: KalmanFilter copies share their Mats, so two copies of a Track would
// share one filter state.
struct Track
{
    Track(int id_, const Rect2d& box, int classId_, float score_)
        : id(id_), classId(classId_), score(score_), state(TENTATIVE), hits(1), age(0)
    {
        kf.init(STATE_DIM, MEAS_DIM, 0, CV_32F);

        setIdentity(kf.transitionMatrix);
        for (int i = 0; i < MEAS_DIM; i++)
            kf.transitionMatrix.at<float>(i, i + MEAS_DIM) = 1.0f;   // dt = 1 frame

        kf.measurementMatrix = Mat::zeros(MEAS_DIM, STATE_DIM, CV_32F);
        for (int i = 0; i < MEAS_DIM; i++)
            kf.measurementMatrix.at<float>(i, i) = 1.0f;

        Mat z;
        boxToMeasurement(box, z);
        kf.statePost = Mat::zeros(STATE_DIM, 1, CV_32F);
        z.copyTo(kf.statePost(Rect(0, 0, 1, MEAS_DIM)));

        // init() leaves errorCovPost at zero, which would make the first correct() over-confident.
        const float h = z.at<float>(3);
        kf.errorCovPost = Mat::zeros(STATE_DIM, STATE_DIM, CV_32F);
        const float p[STATE_DIM] = {
            2 * STD_POSITION * h, 2 * STD_POSITION * h, 1e-2f, 2 * STD_POSITION * h,
            10 * STD_VELOCITY * h, 10 * STD_VELOCITY * h, 1e-5f, 10 * STD_VELOCITY * h
        };
        for (int i = 0; i < STATE_DIM; i++)
            kf.errorCovPost.at<float>(i, i) = p[i] * p[i];
    }

    // The noise depends on the current size, so both covariances are refreshed every frame.
    void setProcessNoise()
    {
        const float h = std::max(kf.statePost.at<float>(3), 1e-6f);
        kf.processNoiseCov = Mat::zeros(STATE_DIM, STATE_DIM, CV_32F);
        const float q[STATE_DIM] = {
            STD_POSITION * h, STD_POSITION * h, 1e-2f, STD_POSITION * h,
            STD_VELOCITY * h, STD_VELOCITY * h, 1e-5f, STD_VELOCITY * h
        };
        for (int i = 0; i < STATE_DIM; i++)
            kf.processNoiseCov.at<float>(i, i) = q[i] * q[i];
    }

    void setMeasurementNoise()
    {
        const float h = std::max(kf.statePre.at<float>(3), 1e-6f);
        kf.measurementNoiseCov = Mat::zeros(MEAS_DIM, MEAS_DIM, CV_32F);
        const float r[MEAS_DIM] = { STD_POSITION * h, STD_POSITION * h, 1e-1f, STD_POSITION * h };
        for (int i = 0; i < MEAS_DIM; i++)
            kf.measurementNoiseCov.at<float>(i, i) = r[i] * r[i];
    }

    // predict() copies statePre into statePost, so a track with no measurement keeps coasting on
    // its own prediction. That is what lets a lost track be picked up again when it reappears.
    void predict()
    {
        setProcessNoise();
        kf.predict();
        setMeasurementNoise();
        predictedBox = stateToBox(kf.statePre);
    }

    void correct(const Rect2d& box, int classId_, float score_)
    {
        Mat z;
        boxToMeasurement(box, z);
        kf.correct(z);
        classId = classId_;
        score = score_;
        hits++;
        age = 0;
    }

    Rect2d box() const { return stateToBox(kf.statePost); }

    // Innovation covariance S = H P' H^T + R, the spread the prediction allows a measurement.
    Mat innovationCovariance() const
    {
        const Mat& H = kf.measurementMatrix;
        return H * kf.errorCovPre * H.t() + kf.measurementNoiseCov;
    }

    KalmanFilter kf;
    Mat feature;            // running appearance descriptor, empty until one is supplied
    Rect2d predictedBox;
    int id;
    int classId;
    float score;
    int state;
    int hits;               // frames matched, only read while the track is still tentative
    int age;                // consecutive frames unmatched
};

class MultiTrackerImpl CV_FINAL : public MultiTracker
{
public:
    explicit MultiTrackerImpl(const MultiTracker::Params& parameters)
        : params(parameters), nextId(1)
    {
        CV_Assert(params.lowDetectionThreshold <= params.highDetectionThreshold);
        CV_Assert(params.minHits >= 1 && params.maxAge >= 0);
        CV_Assert(params.embeddingWeight >= 0.0f && params.embeddingWeight <= 1.0f);
    }

    void update(const std::vector<Rect2d>& detBoxes,
                const std::vector<float>& detScores,
                const std::vector<int>& detClassIds,
                std::vector<int>& trackIds,
                std::vector<Rect2d>& trackBoxes,
                std::vector<int>& trackClassIds) CV_OVERRIDE
    {
        run(detBoxes, detScores, detClassIds, Mat(), trackIds, trackBoxes, trackClassIds);
    }

    void update(const std::vector<Rect2d>& detBoxes,
                const std::vector<float>& detScores,
                const std::vector<int>& detClassIds,
                InputArray detEmbeddings,
                std::vector<int>& trackIds,
                std::vector<Rect2d>& trackBoxes,
                std::vector<int>& trackClassIds) CV_OVERRIDE
    {
        run(detBoxes, detScores, detClassIds, detEmbeddings.getMat(),
            trackIds, trackBoxes, trackClassIds);
    }

    void reset() CV_OVERRIDE
    {
        tracks.clear();
        nextId = 1;
    }

private:
    void run(const std::vector<Rect2d>& detBoxes,
             const std::vector<float>& detScores,
             const std::vector<int>& detClassIds,
             const Mat& embeddings,
             std::vector<int>& trackIds,
             std::vector<Rect2d>& trackBoxes,
             std::vector<int>& trackClassIds);

    // Accepts one pairing: corrects the filter and folds in the new appearance.
    void applyMatch(size_t trackIdx, int det,
                    const std::vector<Rect2d>& detBoxes,
                    const std::vector<float>& detScores,
                    const std::vector<int>& detClassIds,
                    const Mat& embeddings);

    // Blends a new descriptor into a track's running one, as StrongSORT does.
    void setFeature(const Ptr<Track>& tr, const Mat& f, bool replace) const;

    // Builds the cost matrix for one pass and solves it. On return matchedDet[t] is the index into
    // `dets` matched to candidate track `cands[t]`, or -1.
    void associate(const std::vector<size_t>& cands,
                   const std::vector<int>& dets,
                   const std::vector<Rect2d>& detBoxes,
                   const std::vector<int>& detClassIds,
                   const Mat& embeddings,
                   bool useAppearance,
                   bool useGate,
                   std::vector<int>& matchedDet) const;

    MultiTracker::Params params;
    std::vector<Ptr<Track> > tracks;
    int nextId;
};

void MultiTrackerImpl::associate(const std::vector<size_t>& cands,
                                 const std::vector<int>& dets,
                                 const std::vector<Rect2d>& detBoxes,
                                 const std::vector<int>& detClassIds,
                                 const Mat& embeddings,
                                 bool useAppearance,
                                 bool useGate,
                                 std::vector<int>& matchedDet) const
{
    matchedDet.assign(cands.size(), -1);
    if (cands.empty() || dets.empty())
        return;

    const int nr = (int)cands.size();
    const int nc = (int)dets.size();

    // The solver refuses anything above the threshold, so a rejected pair needs no inf sentinel.
    const double reject = (double)params.iouThreshold + 1.0;

    const bool gate = useGate && params.gatingThreshold > 0.0f;
    std::vector<Mat> detMeas;
    if (gate)
    {
        detMeas.resize((size_t)nc);
        for (int j = 0; j < nc; j++)
            boxToMeasurement(detBoxes[dets[j]], detMeas[(size_t)j]);
    }

    Mat cost(nr, nc, CV_64F);

    // Each row is written by one track and reads nothing another row writes, so the rows split
    // cleanly. Below a few dozen tracks the dispatch costs more than the loop, so stay serial.
    const auto fillRows = [&](const Range& range)
    {
    for (int i = range.start; i < range.end; i++)
    {
        const Ptr<Track>& tr = tracks[cands[i]];
        double* row = cost.ptr<double>(i);

        Mat sInv, predMeas;
        if (gate)
        {
            // S is a covariance, so Cholesky applies and reports a collapsed variance itself.
            if (invert(tr->innovationCovariance(), sInv, DECOMP_CHOLESKY) == 0.0)
                sInv.release();
            else
                predMeas = tr->kf.measurementMatrix * tr->kf.statePre;
        }

        for (int j = 0; j < nc; j++)
        {
            const int d = dets[j];

            if (params.classAware && tr->classId != detClassIds[d])
            {
                row[j] = reject;
                continue;
            }

            double c = jaccardDistance(tr->predictedBox, detBoxes[d]);
            const double iouDist = c;

            if (useAppearance && !tr->feature.empty() && !embeddings.empty())
            {
                const Mat f = embeddings.row(d);
                const double cosDist = 1.0 - tr->feature.dot(f);
                if (cosDist > (double)params.embeddingThreshold)
                {
                    row[j] = reject;
                    continue;
                }
                c = params.embeddingWeight * cosDist + (1.0 - params.embeddingWeight) * c;
            }

            if (!sInv.empty())
            {
                const double m = Mahalanobis(detMeas[(size_t)j], predMeas, sInv);
                if (m * m > (double)params.gatingThreshold)
                {
                    row[j] = reject;
                    continue;
                }
            }
            else if (iouDist > (double)params.iouThreshold)
            {
                row[j] = reject;   // ungated, and a good appearance score can hide zero overlap
                continue;
            }

            row[j] = c;
        }
    }
    };

    if (nr >= 64)
        parallel_for_(Range(0, nr), fillRows);
    else
        fillRows(Range(0, nr));

    std::vector<int> assignment;
    linearAssignment(cost, assignment, (double)params.iouThreshold);
    for (int i = 0; i < nr; i++)
        matchedDet[i] = assignment[i] < 0 ? -1 : dets[assignment[i]];
}

void MultiTrackerImpl::setFeature(const Ptr<Track>& tr, const Mat& f, bool replace) const
{
    Mat row;
    f.copyTo(row);                            // f views the caller's matrix, normalize is in place
    normalize(row, row, 1.0, 0.0, NORM_L2);   // the contract says normalised, but check anyway

    if (replace || tr->feature.empty())
    {
        tr->feature = row;
        return;
    }

    // Keeping only the newest descriptor makes a track forget itself the moment one frame is
    // blurred or half covered, so carry most of the old one forward.
    const float m = params.embeddingMomentum;
    Mat blended = m * tr->feature + (1.0f - m) * row;
    normalize(blended, blended, 1.0, 0.0, NORM_L2);
    tr->feature = blended;
}

void MultiTrackerImpl::applyMatch(size_t trackIdx, int det,
                                  const std::vector<Rect2d>& detBoxes,
                                  const std::vector<float>& detScores,
                                  const std::vector<int>& detClassIds,
                                  const Mat& embeddings)
{
    const Ptr<Track>& tr = tracks[trackIdx];
    tr->correct(detBoxes[det], detClassIds[det], detScores[det]);
    if (!embeddings.empty())
        setFeature(tr, embeddings.row(det), false);
}

void MultiTrackerImpl::run(const std::vector<Rect2d>& detBoxes,
                           const std::vector<float>& detScores,
                           const std::vector<int>& detClassIds,
                           const Mat& embeddings,
                           std::vector<int>& trackIds,
                           std::vector<Rect2d>& trackBoxes,
                           std::vector<int>& trackClassIds)
{
    CV_Assert(detBoxes.size() == detScores.size());
    CV_Assert(detBoxes.size() == detClassIds.size());

    const bool useAppearance = !embeddings.empty() && params.embeddingWeight > 0.0f;
    if (!embeddings.empty())
    {
        CV_CheckTypeEQ(embeddings.type(), CV_32FC1, "embeddings must be CV_32FC1");
        CV_Assert(embeddings.rows == (int)detBoxes.size());
    }

    // Split the detections. Confident ones drive the first pass; the weak ones are only offered to
    // tracks that found nothing, which is how a half-occluded object keeps its id.
    std::vector<int> highDets, lowDets;
    for (size_t i = 0; i < detBoxes.size(); i++)
    {
        if (detScores[i] >= params.highDetectionThreshold)
            highDets.push_back((int)i);
        else if (detScores[i] >= params.lowDetectionThreshold)
            lowDets.push_back((int)i);
    }

    for (size_t i = 0; i < tracks.size(); i++)
        tracks[i]->predict();

    std::vector<bool> detTaken(detBoxes.size(), false);
    std::vector<bool> trackMatched(tracks.size(), false);

    // Pass 1 -- established tracks against the confident detections.
    std::vector<size_t> established;
    for (size_t i = 0; i < tracks.size(); i++)
        if (tracks[i]->state != TENTATIVE)
            established.push_back(i);

    std::vector<int> matched;
    associate(established, highDets, detBoxes, detClassIds, embeddings, useAppearance, true, matched);
    for (size_t t = 0; t < established.size(); t++)
    {
        const int d = matched[t];
        if (d < 0)
            continue;
        applyMatch(established[t], d, detBoxes, detScores, detClassIds, embeddings);
        detTaken[d] = true;
        trackMatched[established[t]] = true;
    }

    // Pass 2 -- whatever is still unmatched against the weak detections. Motion only: a detection
    // this faint is usually a partial view, so its appearance is not worth trusting.
    std::vector<size_t> stillFree;
    for (size_t i = 0; i < established.size(); i++)
        if (!trackMatched[established[i]])
            stillFree.push_back(established[i]);

    associate(stillFree, lowDets, detBoxes, detClassIds, embeddings, false, true, matched);
    for (size_t t = 0; t < stillFree.size(); t++)
    {
        const int d = matched[t];
        if (d < 0)
            continue;
        applyMatch(stillFree[t], d, detBoxes, detScores, detClassIds, embeddings);
        detTaken[d] = true;
        trackMatched[stillFree[t]] = true;
    }

    // Pass 3 -- unconfirmed tracks against the confident detections nobody claimed.
    std::vector<size_t> tentative;
    for (size_t i = 0; i < tracks.size(); i++)
        if (tracks[i]->state == TENTATIVE)
            tentative.push_back(i);

    std::vector<int> freeHigh;
    for (size_t i = 0; i < highDets.size(); i++)
        if (!detTaken[highDets[i]])
            freeHigh.push_back(highDets[i]);

    associate(tentative, freeHigh, detBoxes, detClassIds, embeddings, useAppearance, false, matched);
    for (size_t t = 0; t < tentative.size(); t++)
    {
        const int d = matched[t];
        if (d < 0)
            continue;
        applyMatch(tentative[t], d, detBoxes, detScores, detClassIds, embeddings);
        detTaken[d] = true;
        trackMatched[tentative[t]] = true;
    }

    // Age the tracks that found nothing, and retire the ones that have waited too long.
    std::vector<Ptr<Track> > survivors;
    survivors.reserve(tracks.size());
    for (size_t i = 0; i < tracks.size(); i++)
    {
        Ptr<Track>& tr = tracks[i];
        if (trackMatched[i])
        {
            if (tr->state == TENTATIVE && tr->hits >= params.minHits)
                tr->state = CONFIRMED;
            else if (tr->state == LOST)
                tr->state = CONFIRMED;
            survivors.push_back(tr);
            continue;
        }

        tr->age++;
        if (tr->state == TENTATIVE)
            continue;                       // an unconfirmed track that misses once is dropped
        if (tr->age > params.maxAge)
            continue;                       // waited too long, the id retires with it
        tr->state = LOST;
        survivors.push_back(tr);
    }
    tracks.swap(survivors);

    // Every confident detection nobody claimed starts a new track.
    for (size_t i = 0; i < highDets.size(); i++)
    {
        const int d = highDets[i];
        if (detTaken[d])
            continue;
        Ptr<Track> tr = makePtr<Track>(nextId++, detBoxes[d], detClassIds[d], detScores[d]);
        // New tracks miss the promotion check above, so minHits == 1 has to be honoured here.
        if (tr->hits >= params.minHits)
            tr->state = CONFIRMED;
        if (!embeddings.empty())
            setFeature(tr, embeddings.row(d), true);
        tracks.push_back(tr);
    }

    trackIds.clear();
    trackBoxes.clear();
    trackClassIds.clear();
    for (size_t i = 0; i < tracks.size(); i++)
    {
        const Ptr<Track>& tr = tracks[i];
        if (tr->state == TENTATIVE)
            continue;                       // not reported until it has been seen minHits times
        trackIds.push_back(tr->id);
        trackBoxes.push_back(tr->box());
        trackClassIds.push_back(tr->classId);
    }
}

}}  // namespace tracking::impl

MultiTracker::Params::Params()
{
    highDetectionThreshold = 0.5f;
    lowDetectionThreshold = 0.1f;
    iouThreshold = 0.8f;
    gatingThreshold = 9.4877f;   // chi-square, 0.95 quantile, 4 degrees of freedom
    embeddingWeight = 0.0f;
    embeddingThreshold = 0.4f;
    embeddingMomentum = 0.9f;
    minHits = 3;
    maxAge = 30;
    classAware = true;
}

MultiTracker::MultiTracker()
{
    // nothing
}

MultiTracker::~MultiTracker()
{
    // nothing
}

Ptr<MultiTracker> MultiTracker::create(const MultiTracker::Params& parameters)
{
    return makePtr<tracking::impl::MultiTrackerImpl>(parameters);
}

}  // namespace cv
