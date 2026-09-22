// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Everything here is synthetic and deterministic, so no test data is needed.

struct Frame
{
    std::vector<Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    void add(const Rect2d& b, float s = 0.9f, int c = 0)
    {
        boxes.push_back(b);
        scores.push_back(s);
        classIds.push_back(c);
    }
};

struct Out
{
    std::vector<int> ids;
    std::vector<Rect2d> boxes;
    std::vector<int> classIds;

    int idAt(size_t i) const { return ids[i]; }
    bool has(int id) const { return std::find(ids.begin(), ids.end(), id) != ids.end(); }
};

static Out step(const Ptr<MultiTracker>& t, const Frame& f)
{
    Out o;
    t->update(f.boxes, f.scores, f.classIds, o.ids, o.boxes, o.classIds);
    return o;
}

static MultiTracker::Params fastParams()
{
    MultiTracker::Params p;
    p.minHits = 3;
    p.maxAge = 5;
    return p;
}

// A new object is withheld until it has been seen minHits times, then survives exactly maxAge
// frames of silence before its id retires.
TEST(Video_MultiTracker, lifecycle)
{
    MultiTracker::Params p = fastParams();
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame f;
    f.add(Rect2d(100, 100, 40, 80));

    EXPECT_EQ(0u, step(tracker, f).ids.size()) << "frame 1: too early to report";
    EXPECT_EQ(0u, step(tracker, f).ids.size()) << "frame 2: still too early";

    Out o = step(tracker, f);
    ASSERT_EQ(1u, o.ids.size()) << "frame 3: minHits reached, should be reported";
    const int id = o.idAt(0);

    const Frame empty;
    for (int i = 1; i <= p.maxAge; i++)
    {
        o = step(tracker, empty);
        EXPECT_TRUE(o.has(id)) << "coasting frame " << i << " of " << p.maxAge;
    }
    EXPECT_FALSE(step(tracker, empty).has(id)) << "should have retired after maxAge";
}

// minHits == 1 means one sighting is enough, so the track has to be reported on the very frame it
// appears. New tracks are created after the ageing pass, so they miss the promotion check and this
// boundary is easy to get wrong by one frame.
TEST(Video_MultiTracker, min_hits_one_reports_immediately)
{
    MultiTracker::Params p = fastParams();
    p.minHits = 1;
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame f;
    f.add(Rect2d(100, 100, 40, 80));
    Out o = step(tracker, f);
    ASSERT_EQ(1u, o.ids.size()) << "one sighting was asked for, so it should already be reported";

    const int id = o.idAt(0);
    EXPECT_EQ(id, step(tracker, f).idAt(0));
}

// An unconfirmed track that misses even once is discarded, so a one-frame false positive never
// becomes a track.
TEST(Video_MultiTracker, tentative_dropped_immediately)
{
    Ptr<MultiTracker> tracker = MultiTracker::create(fastParams());

    Frame f;
    f.add(Rect2d(10, 10, 20, 20));
    step(tracker, f);

    const Frame empty;
    for (int i = 0; i < 10; i++)
        EXPECT_EQ(0u, step(tracker, empty).ids.size());
}

// Two objects moving steadily apart keep their own ids.
TEST(Video_MultiTracker, stable_ids_two_objects)
{
    Ptr<MultiTracker> tracker = MultiTracker::create(fastParams());

    int idA = -1, idB = -1;
    for (int t = 0; t < 20; t++)
    {
        Frame f;
        f.add(Rect2d(50 + 4.0 * t, 100, 40, 80));    // moving right
        f.add(Rect2d(400 - 4.0 * t, 100, 40, 80));   // moving left
        Out o = step(tracker, f);

        if (t < 2)
            continue;
        ASSERT_EQ(2u, o.ids.size()) << "frame " << t;

        // left-most first, so the comparison does not depend on report order
        size_t left = o.boxes[0].x <= o.boxes[1].x ? 0 : 1;
        size_t right = 1 - left;
        if (idA < 0)
        {
            idA = o.ids[left];
            idB = o.ids[right];
        }
        EXPECT_EQ(idA, o.ids[left]) << "frame " << t << ": left object changed id";
        EXPECT_EQ(idB, o.ids[right]) << "frame " << t << ": right object changed id";
    }
    EXPECT_NE(idA, idB);
}

// An id belonging to a retired track is never handed to a later object.
TEST(Video_MultiTracker, no_id_reuse)
{
    MultiTracker::Params p = fastParams();
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame first;
    first.add(Rect2d(100, 100, 40, 80));
    for (int i = 0; i < 3; i++)
        step(tracker, first);
    const int oldId = step(tracker, first).idAt(0);

    const Frame empty;
    for (int i = 0; i < p.maxAge + 2; i++)
        step(tracker, empty);

    Frame second;
    second.add(Rect2d(300, 300, 40, 80));
    Out o;
    for (int i = 0; i < 4; i++)
        o = step(tracker, second);

    ASSERT_EQ(1u, o.ids.size());
    EXPECT_NE(oldId, o.idAt(0)) << "a retired id was handed out again";
}

// With classAware on, a detection may not be taken by a track of a different class.
TEST(Video_MultiTracker, class_aware)
{
    MultiTracker::Params p = fastParams();
    p.classAware = true;
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame f;
    f.add(Rect2d(100, 100, 40, 80), 0.9f, /*class*/ 1);
    for (int i = 0; i < 3; i++)
        step(tracker, f);
    const int id = step(tracker, f).idAt(0);

    // same place, different class: the old track must not absorb it
    Frame other;
    other.add(Rect2d(100, 100, 40, 80), 0.9f, /*class*/ 2);
    Out o = step(tracker, other);
    for (size_t i = 0; i < o.ids.size(); i++)
    {
        if (o.ids[i] == id)
        {
            EXPECT_EQ(1, o.classIds[i]) << "track changed class through a mismatched detection";
        }
    }
}

// A weak detection has to genuinely re-match the track, not merely let it coast. The two look
// identical for a few frames, so the run is stretched past maxAge: a coasting track dies there,
// a re-matched one keeps going.
TEST(Video_MultiTracker, low_score_detection_keeps_id)
{
    MultiTracker::Params p = fastParams();
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame strong;
    strong.add(Rect2d(100, 100, 40, 80), 0.9f);
    for (int i = 0; i < 3; i++)
        step(tracker, strong);
    const int id = step(tracker, strong).idAt(0);

    Frame weakDet;
    weakDet.add(Rect2d(100, 100, 40, 80), 0.2f);   // below high, above low
    for (int i = 0; i < p.maxAge + 5; i++)
    {
        Out o = step(tracker, weakDet);
        ASSERT_TRUE(o.has(id))
            << "lost the id on weak frame " << i << " of " << (p.maxAge + 5)
            << " -- past maxAge this can only survive if the second pass matched it";
    }
}

TEST(Video_MultiTracker, mahalanobis_gate_rejects_implausible_shape)
{
    // 40x80 track against an 80x40 detection on the same centre: IoU distance is 0.667, inside the
    // 0.8 threshold, but the aspect ratio and height both jump far outside the predicted spread.
    const Rect2d established(100, 100, 40, 80);
    const Rect2d wrongShape(80, 120, 80, 40);

    double heightWithGate = 0.0, heightWithoutGate = 0.0;
    for (int pass = 0; pass < 2; pass++)
    {
        MultiTracker::Params p = fastParams();
        p.gatingThreshold = pass == 0 ? 9.4877f : 0.0f;   // second pass disables the gate
        Ptr<MultiTracker> tracker = MultiTracker::create(p);

        Frame f;
        f.add(established);
        for (int i = 0; i < 3; i++)
            step(tracker, f);
        Out o = step(tracker, f);
        ASSERT_EQ(1u, o.ids.size());
        const int id = o.idAt(0);

        Frame odd;
        odd.add(wrongShape);
        o = step(tracker, odd);

        ASSERT_TRUE(o.has(id)) << "the original track should still be reported either way";
        for (size_t i = 0; i < o.ids.size(); i++)
        {
            if (o.ids[i] == id)
            {
                (pass == 0 ? heightWithGate : heightWithoutGate) = o.boxes[i].height;
            }
        }
    }

    // Gated, the track coasts on its prediction and keeps its height. Ungated, it is dragged
    // towards the 40-high detection.
    EXPECT_GT(heightWithGate, 70.0) << "gate did not reject the implausible detection";
    EXPECT_LT(heightWithoutGate, heightWithGate)
        << "with the gate off the same detection should have been absorbed";
}

// Builds a matrix of L2-normalised descriptors, one row per detection.
static Mat embeddingRows(const std::vector<std::vector<float> >& rows)
{
    Mat m((int)rows.size(), (int)rows[0].size(), CV_32F);
    for (size_t i = 0; i < rows.size(); i++)
    {
        Mat r = m.row((int)i);
        for (size_t j = 0; j < rows[i].size(); j++)
            r.at<float>((int)j) = rows[i][j];
        r /= cv::norm(r);
    }
    return m;
}

// Appearance has to be able to overrule position: the nearer detection is the wrong object.
// The gate's measurement noise scales with the predicted height, so a track that has never been
// corrected must be gated with that value and not the identity KalmanFilter::init leaves behind.
// On a 10x4 box a 2.5 px shift is 20.7 squared Mahalanobis units against the real noise and only
// 5.0 against the identity, so an unrefreshed gate accepts what the real one rejects.
TEST(Video_MultiTracker, gate_uses_height_scaled_noise_on_first_association)
{
    MultiTracker::Params p = fastParams();
    p.minHits = 1;                  // confirmed on the frame it is created, so never corrected
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame first;
    first.add(Rect2d(100, 100, 10, 4));
    Out o = step(tracker, first);
    ASSERT_EQ(1u, o.ids.size());
    const int id = o.idAt(0);

    Frame moved;
    moved.add(Rect2d(102.5, 100, 10, 4));
    o = step(tracker, moved);

    EXPECT_TRUE(o.has(id)) << "the gated track should still be coasting";
    EXPECT_EQ(2u, o.ids.size()) << "the detection was outside the gate and should have started "
                                   "its own track";
}

TEST(Video_MultiTracker, embedding_outweighs_closer_box)
{
    const std::vector<float> A = {1, 0, 0, 0};
    const std::vector<float> B = {0, 1, 0, 0};

    double cxWithout = 0.0, cxWith = 0.0;
    for (int pass = 0; pass < 2; pass++)
    {
        MultiTracker::Params p = fastParams();
        p.embeddingWeight = pass == 0 ? 0.0f : 0.9f;
        Ptr<MultiTracker> tracker = MultiTracker::create(p);

        Frame f;
        f.add(Rect2d(100, 100, 40, 80));
        Mat one = embeddingRows(std::vector<std::vector<float> >(1, A));
        std::vector<int> ids; std::vector<Rect2d> boxes; std::vector<int> classes;
        for (int i = 0; i < 4; i++)
            tracker->update(f.boxes, f.scores, f.classIds, one, ids, boxes, classes);
        ASSERT_EQ(1u, ids.size());
        const int id = ids[0];

        // Two candidates: the closer one looks wrong, the further one looks right.
        Frame two;
        two.add(Rect2d(102, 100, 40, 80));   // IoU distance 0.095, descriptor B
        two.add(Rect2d(120, 100, 40, 80));   // IoU distance 0.667, descriptor A
        std::vector<std::vector<float> > rows;
        rows.push_back(B);
        rows.push_back(A);
        tracker->update(two.boxes, two.scores, two.classIds, embeddingRows(rows),
                        ids, boxes, classes);

        for (size_t i = 0; i < ids.size(); i++)
        {
            if (ids[i] == id)
            {
                (pass == 0 ? cxWithout : cxWith) = boxes[i].x + boxes[i].width * 0.5;
            }
        }
    }

    EXPECT_GT(cxWith, cxWithout + 5.0)
        << "with appearance weighted the track should have followed the matching descriptor "
        << "rather than the nearer box (with = " << cxWith << ", without = " << cxWithout << ")";
}

// The per-track descriptor is a running average, so one odd-looking frame must not rewrite a
// track's identity. Checked by outcome: a track that still remembers A refuses a pure-B detection,
// which then has to start a track of its own. If the memory were overwritten, B would simply be
// absorbed and no second id would ever appear.
TEST(Video_MultiTracker, embedding_ema_keeps_memory)
{
    const std::vector<float> A = {1, 0, 0, 0};
    const std::vector<float> B = {0, 1, 0, 0};
    std::vector<float> mixed;
    mixed.push_back(1); mixed.push_back(1); mixed.push_back(0); mixed.push_back(0);

    MultiTracker::Params p = fastParams();
    p.embeddingWeight = 0.9f;
    Ptr<MultiTracker> tracker = MultiTracker::create(p);

    Frame f;
    f.add(Rect2d(100, 100, 40, 80));
    std::vector<int> ids; std::vector<Rect2d> boxes; std::vector<int> classes;

    const Mat rowA = embeddingRows(std::vector<std::vector<float> >(1, A));
    for (int i = 0; i < 4; i++)
        tracker->update(f.boxes, f.scores, f.classIds, rowA, ids, boxes, classes);
    ASSERT_EQ(1u, ids.size());
    const int id = ids[0];

    // One frame that looks halfway to B. Close enough to be accepted, and it shifts the running
    // descriptor only slightly because most of the old one is carried forward.
    const Mat rowMixed = embeddingRows(std::vector<std::vector<float> >(1, mixed));
    tracker->update(f.boxes, f.scores, f.classIds, rowMixed, ids, boxes, classes);
    ASSERT_TRUE(std::find(ids.begin(), ids.end(), id) != ids.end())
        << "a halfway descriptor should still match";

    // Now pure B, in the same place. Position alone would match it, so only the retained
    // appearance can refuse it -- and a refused detection becomes a track of its own.
    const Mat rowB = embeddingRows(std::vector<std::vector<float> >(1, B));
    bool sawNewId = false;
    for (int i = 0; i < p.minHits + 1; i++)
    {
        tracker->update(f.boxes, f.scores, f.classIds, rowB, ids, boxes, classes);
        for (size_t k = 0; k < ids.size(); k++)
        {
            if (ids[k] != id)
                sawNewId = true;
        }
    }
    EXPECT_TRUE(sawNewId)
        << "the B detection was absorbed by the A track, so the running descriptor was not kept";
}

TEST(Video_MultiTracker, reset_clears_everything)
{
    Ptr<MultiTracker> tracker = MultiTracker::create(fastParams());

    Frame f;
    f.add(Rect2d(100, 100, 40, 80));
    for (int i = 0; i < 4; i++)
        step(tracker, f);
    ASSERT_EQ(1u, step(tracker, f).ids.size());

    tracker->reset();
    EXPECT_EQ(0u, step(tracker, Frame()).ids.size());
}

// ---------------------------------------------------------------------------------------------
// A scripted sequence with occlusions, weak detections, false positives and measurement noise,
// scored the CLEAR MOT way against exact ground truth.
// ---------------------------------------------------------------------------------------------

namespace {

const int MOT_FRAMES = 60;
const int MOT_OBJECTS = 4;

// Ground truth for one object at one frame. Objects move steadily right on two rows.
static Rect2d truthBox(int obj, int frame)
{
    const double x = 40.0 + obj * 130.0 + 3.0 * frame;
    const double y = 90.0 + (obj % 2) * 70.0;
    return Rect2d(x, y, 40, 80);
}

// Detections are withheld for object 1 in the middle of the run, as if it passed behind something.
static bool occluded(int obj, int frame)
{
    return obj == 1 && frame >= 20 && frame <= 27;
}

// Object 2 is only partly visible for a while, so its detector score drops into the weak band.
static bool weak(int obj, int frame)
{
    return obj == 2 && frame >= 18 && frame <= 53;   // longer than maxAge, so coasting alone cannot carry it
}

struct MotScore
{
    int idSwitches, falsePositives, misses, groundTruth;
    double mota() const
    {
        return 1.0 - (double)(misses + falsePositives + idSwitches) / (double)groundTruth;
    }
};

// Greedy IoU matching at the usual 0.5 overlap. The ground-truth objects never overlap each other
// here, so greedy and optimal agree and the scorer stays independent of the code under test.
static void scoreFrame(const std::vector<Rect2d>& gtBoxes, const std::vector<int>& gtObjs,
                       const std::vector<Rect2d>& trkBoxes, const std::vector<int>& trkIds,
                       std::map<int, int>& lastIdForObject, MotScore& acc)
{
    std::vector<bool> trkUsed(trkBoxes.size(), false);

    for (size_t g = 0; g < gtBoxes.size(); g++)
    {
        int best = -1;
        double bestIou = 0.5;                       // must beat the overlap threshold to count
        for (size_t t = 0; t < trkBoxes.size(); t++)
        {
            if (trkUsed[t])
                continue;
            const double iou = 1.0 - jaccardDistance(gtBoxes[g], trkBoxes[t]);
            if (iou > bestIou)
            {
                bestIou = iou;
                best = (int)t;
            }
        }

        if (best < 0)
        {
            acc.misses++;
            continue;
        }

        trkUsed[best] = true;
        const int obj = gtObjs[g];
        const int id = trkIds[best];
        std::map<int, int>::iterator prev = lastIdForObject.find(obj);
        if (prev != lastIdForObject.end() && prev->second != id)
            acc.idSwitches++;
        lastIdForObject[obj] = id;
    }

    for (size_t t = 0; t < trkBoxes.size(); t++)
    {
        if (!trkUsed[t])
            acc.falsePositives++;
    }
}

static MotScore runSyntheticSequence(const MultiTracker::Params& params, bool verbose)
{
    Ptr<MultiTracker> tracker = MultiTracker::create(params);
    RNG rng(0x5A17ED);                              // fixed, so the sequence is reproducible

    MotScore acc;
    acc.idSwitches = acc.falsePositives = acc.misses = acc.groundTruth = 0;
    std::map<int, int> lastIdForObject;

    for (int frame = 0; frame < MOT_FRAMES; frame++)
    {
        Frame det;
        for (int obj = 0; obj < MOT_OBJECTS; obj++)
        {
            if (occluded(obj, frame))
                continue;
            Rect2d b = truthBox(obj, frame);
            b.x += rng.gaussian(1.0);               // detector jitter
            b.y += rng.gaussian(1.0);
            b.width += rng.gaussian(0.5);
            b.height += rng.gaussian(0.5);
            det.add(b, weak(obj, frame) ? 0.30f : 0.90f, 0);
        }

        // a spurious detection every so often, well away from every real object
        if (frame % 15 == 10)
            det.add(Rect2d(560, 300, 40, 80), 0.92f, 0);

        Out o = step(tracker, det);

        std::vector<Rect2d> gtBoxes;
        std::vector<int> gtObjs;
        for (int obj = 0; obj < MOT_OBJECTS; obj++)
        {
            gtBoxes.push_back(truthBox(obj, frame));   // an occluded object still exists
            gtObjs.push_back(obj);
        }
        acc.groundTruth += MOT_OBJECTS;

        scoreFrame(gtBoxes, gtObjs, o.boxes, o.ids, lastIdForObject, acc);
    }

    if (verbose)
    {
        std::cout << "[ MOT      ] ground truth  " << acc.groundTruth << "\n"
                  << "[ MOT      ] id switches   " << acc.idSwitches << "\n"
                  << "[ MOT      ] false pos     " << acc.falsePositives << "\n"
                  << "[ MOT      ] misses        " << acc.misses << "\n"
                  << "[ MOT      ] MOTA          " << acc.mota() << std::endl;
    }
    return acc;
}

}  // namespace

TEST(Video_MultiTracker, accuracy_synthetic_mot)
{
    MultiTracker::Params p;                        // shipped defaults, deliberately
    const MotScore s = runSyntheticSequence(p, true);

    ASSERT_EQ(240, s.groundTruth);

    // The sequence is deterministic, so the discrete counts are exact rather than bounded. Each
    // was measured, and each is explained:
    //
    //   id switches 0  -- the occlusion (8 frames, under maxAge) and the weak spell (36 frames,
    //                     over maxAge, so only the second pass can carry it) are both survived
    //   false pos   0  -- the spurious detections never confirm, because an unmatched tentative
    //                     track is dropped on the spot
    //   misses      8  -- 4 objects x the 2 frames each waits to reach minHits. Nothing else.
    //
    // Removing the coasting gives 1 / 0 / 18, removing the second pass gives 1 / 0 / 16, so these
    // numbers do fail when the behaviour they describe is taken away.
    EXPECT_EQ(0, s.idSwitches);
    EXPECT_EQ(0, s.falsePositives);
    EXPECT_EQ(8, s.misses);

    // MOTA is the same information as a single number, so it gets a small margin instead: it is
    // the one value that could move slightly on a different architecture without anything being
    // wrong. Both regressions above land near 0.92, well outside the band.
    const double baseline = 0.9667;
    EXPECT_GE(s.mota(), baseline - 0.02);
}

TEST(Video_MultiTracker, mismatched_input_sizes_throw)
{
    Ptr<MultiTracker> tracker = MultiTracker::create(MultiTracker::Params());

    std::vector<Rect2d> boxes(2, Rect2d(0, 0, 10, 10));
    std::vector<float> scores(1, 0.9f);          // wrong length
    std::vector<int> classIds(2, 0);
    std::vector<int> ids;
    std::vector<Rect2d> outBoxes;
    std::vector<int> outClasses;

    EXPECT_ANY_THROW(tracker->update(boxes, scores, classIds, ids, outBoxes, outClasses));
}

}} // namespace opencv_test::<anonymous>
