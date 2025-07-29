// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include <opencv2/3d/detail/optimizer.hpp>

#include <opencv2/core/dualquaternion.hpp>

namespace opencv_test { namespace {

using namespace cv;

#ifdef HAVE_EIGEN

static Affine3d readAffine(std::istream& input)
{
    Vec3d p;
    Vec4d q;
    input >> p[0] >> p[1] >> p[2];
    input >> q[1] >> q[2] >> q[3] >> q[0];
    // Normalize the quaternion to account for precision loss due to
    // serialization.
    return Affine3d(Quatd(q).toRotMat3x3(), p);
};

// Rewritten from Ceres pose graph demo: https://ceres-solver.org/
static Ptr<detail::PoseGraph> readG2OFile(const std::string& g2oFileName)
{
    Ptr<detail::PoseGraph> pg = detail::PoseGraph::create();

    // for debugging purposes
    size_t minId = 0, maxId = 1 << 30;

    std::ifstream infile(g2oFileName.c_str());
    if (!infile)
    {
        CV_Error(cv::Error::StsError, "failed to open file");
    }

    while (infile.good())
    {
        std::string data_type;
        // Read whether the type is a node or a constraint
        infile >> data_type;
        if (data_type == "VERTEX_SE3:QUAT")
        {
            size_t id;
            infile >> id;
            Affine3d pose = readAffine(infile);

            if (id < minId || id >= maxId)
                continue;

            bool fixed = (id == minId);

            // Ensure we don't have duplicate poses
            if (pg->isNodeExist(id))
            {
                CV_LOG_INFO(NULL, "duplicated node, id=" << id);
            }
            pg->addNode(id, pose, fixed);
        }
        else if (data_type == "EDGE_SE3:QUAT")
        {
            size_t startId, endId;
            infile >> startId >> endId;
            Affine3d pose = readAffine(infile);

            Matx66d info;
            for (int i = 0; i < 6 && infile.good(); ++i)
            {
                for (int j = i; j < 6 && infile.good(); ++j)
                {
                    infile >> info(i, j);
                    if (i != j)
                    {
                        info(j, i) = info(i, j);
                    }
                }
            }

            if ((startId >= minId && startId < maxId) && (endId >= minId && endId < maxId))
            {
                pg->addEdge(startId, endId, pose, info);
            }
        }
        else
        {
            CV_Error(cv::Error::StsError, "unknown tag");
        }

        // Clear any trailing whitespace from the line
        infile >> std::ws;
    }

    return pg;
}


TEST(PoseGraph, sphereG2O)
{
    // Test takes 15+ sec in Release mode and 400+ sec in Debug mode
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG);

    // The dataset was taken from here: https://lucacarlone.mit.edu/datasets/
    // Connected paper:
    // L.Carlone, R.Tron, K.Daniilidis, and F.Dellaert.
    // Initialization Techniques for 3D SLAM : a Survey on Rotation Estimation and its Use in Pose Graph Optimization.
    // In IEEE Intl.Conf.on Robotics and Automation(ICRA), pages 4597 - 4604, 2015.

    std::string filename = cvtest::TS::ptr()->get_data_path() + "/cv/rgbd/sphere_bignoise_vertex3.g2o";

    Ptr<detail::PoseGraph> pg = readG2OFile(filename);

    // You may change logging level to view detailed optimization report
    // For example, set env. variable like this: OPENCV_LOG_LEVEL=INFO

    // geoScale=1 is experimental, not guaranteed to work on other problems
    // the rest are default params
    pg->createOptimizer(LevMarq::Settings().setGeoScale(1.0)
                        .setMaxIterations(100)
                        .setCheckRelEnergyChange(true)
                        .setRelEnergyDeltaTolerance(1e-6)
                        .setGeodesic(true));

    auto r = pg->optimize();

    EXPECT_TRUE(r.found);
    EXPECT_LE(r.iters, 20); // should converge in 31 iterations

    EXPECT_LE(r.energy, 1.47723e+06); // should converge to 1.47722e+06 or less

    // Add the "--test_debug" to arguments to see resulting pose graph nodes positions
    if (cvtest::debugLevel > 0)
    {
        // Write edge-only model of how nodes are located in space
        std::string fname = "pgout.obj";
        std::fstream of(fname, std::fstream::out);
        std::vector<size_t> ids = pg->getNodesIds();
        for (const size_t& id : ids)
        {
            Point3d d = pg->getNodePose(id).translation();
            of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
        }

        size_t esz = pg->getNumEdges();
        for (size_t i = 0; i < esz; i++)
        {
            size_t sid = pg->getEdgeStart(i), tid = pg->getEdgeEnd(i);
            of << "l " << sid + 1 << " " << tid + 1 << std::endl;
        }

        of.close();
    }
}

TEST(PoseGraphMST, optimization)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG);

    // The dataset was taken from here: https://lucacarlone.mit.edu/datasets/
    // Connected paper:
    // L.Carlone, R.Tron, K.Daniilidis, and F.Dellaert.
    // Initialization Techniques for 3D SLAM : a Survey on Rotation Estimation and its Use in Pose Graph Optimization.
    // In IEEE Intl.Conf.on Robotics and Automation(ICRA), pages 4597 - 4604, 2015.

    std::string filename = cvtest::TS::ptr()->get_data_path() + "/cv/rgbd/sphere_bignoise_vertex3.g2o";

    Ptr<detail::PoseGraph> pgOptimizerOnly = readG2OFile(filename);
    Ptr<detail::PoseGraph> pgWithMSTAndOptimizer = readG2OFile(filename);
    Ptr<detail::PoseGraph> init = readG2OFile(filename);

    double lambda = 0.485;
    pgWithMSTAndOptimizer->initializePosesWithMST(lambda);

    // You may change logging level to view detailed optimization report
    // For example, set env. variable like this: OPENCV_LOG_LEVEL=INFO

    // geoScale=1 is experimental, not guaranteed to work on other problems
    // the rest are default params
    pgOptimizerOnly->createOptimizer(LevMarq::Settings().setGeoScale(1.0)
                        .setMaxIterations(100)
                        .setCheckRelEnergyChange(true)
                        .setRelEnergyDeltaTolerance(1e-6)
                        .setGeodesic(true));
    pgWithMSTAndOptimizer->createOptimizer(LevMarq::Settings().setGeoScale(1.0)
                        .setMaxIterations(100)
                        .setCheckRelEnergyChange(true)
                        .setRelEnergyDeltaTolerance(1e-6)
                        .setGeodesic(true));

    auto r1 = pgWithMSTAndOptimizer->optimize();
    auto r2 = pgOptimizerOnly->optimize();

    EXPECT_TRUE(r1.found);
    EXPECT_TRUE(r2.found);
    EXPECT_LE(r2.energy, 1.47723e+06);
    // Allow small tolerance due to optimization differences; final energy/iterations are effectively the same
    EXPECT_LE(std::abs(r1.energy - r2.energy), 1e-2);
    ASSERT_LE(std::abs(r1.iters - r2.iters), 1);

    // Add the "--test_debug" to arguments to see resulting pose graph nodes positions
    if (cvtest::debugLevel > 0)
    {
        /* --- OLD VERSION ---

        // Write OBJ for MST-initialized pose graph with optimizer
        std::string fname = "pg_with_mst_and_optimizer.obj";
        std::fstream of(fname, std::fstream::out);
        std::vector<size_t> ids = pgWithMSTAndOptimizer->getNodesIds();
        for (const size_t& id : ids)
        {
            Point3d d = pgWithMSTAndOptimizer->getNodePose(id).translation();
            of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
        }
        size_t esz = pgWithMSTAndOptimizer->getNumEdges();
        for (size_t i = 0; i < esz; i++)
        {
            size_t sid = pgWithMSTAndOptimizer->getEdgeStart(i), tid = pgWithMSTAndOptimizer->getEdgeEnd(i);
            of << "l " << sid + 1 << " " << tid + 1 << std::endl;
        }
        of.close();

        // Write OBJ for optimizer-only pose graph
        std::string fname = "pg_optimizer_only.obj";
        std::fstream of(fname, std::fstream::out);
        std::vector<size_t> ids = pgOptimizerOnly->getNodesIds();
        for (const size_t& id : ids)
        {
            Point3d d = pgOptimizerOnly->getNodePose(id).translation();
            of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
        }
        size_t esz = pgOptimizerOnly->getNumEdges();
        for (size_t i = 0; i < esz; i++)
        {
            size_t sid = pgOptimizerOnly->getEdgeStart(i), tid = pgOptimizerOnly->getEdgeEnd(i);
            of << "l " << sid + 1 << " " << tid + 1 << std::endl;
        }
        of.close();

        // Write OBJ for initial pose graph
        std::string fname = "pg_init.obj";
        std::fstream of(fname, std::fstream::out);
        std::vector<size_t> ids = init->getNodesIds();
        for (const size_t& id : ids)
        {
            Point3d d = init->getNodePose(id).translation();
            of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
        }
        size_t esz = init->getNumEdges();
        for (size_t i = 0; i < esz; i++)
        {
            size_t sid = init->getEdgeStart(i), tid = init->getEdgeEnd(i);
            of << "l " << sid + 1 << " " << tid + 1 << std::endl;
        }
        of.close();

        --- OLD VERSION --- */

        auto extractVertices = [](const Ptr<detail::PoseGraph>& pg) -> std::vector<Point3f>
        {
            std::vector<Point3f> vertices;
            for (const size_t& id : pg->getNodesIds())
            {
                Point3d d = pg->getNodePose(id).translation();
                vertices.emplace_back(static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
            }
            return vertices;
        };

        auto extractIndexes = [](const Ptr<detail::PoseGraph>& pg) -> std::vector<Vec3i>
        {
            std::vector<Vec3i> indexes;
            size_t esz = pg->getNumEdges();
            for (size_t i = 0; i < esz; i++)
            {
                size_t sid = pg->getEdgeStart(i), tid = pg->getEdgeEnd(i);
                indexes.emplace_back(static_cast<int>(sid), static_cast<int>(tid));
            }
            return indexes;
        };

        saveMesh("pg_with_mst_and_optimizer.obj", extractVertices(pgWithMSTAndOptimizer), extractIndexes(pgWithMSTAndOptimizer));
        saveMesh("pg_optimizer_only.obj", extractVertices(pgOptimizerOnly), extractIndexes(pgOptimizerOnly));
        saveMesh("pg_init2.obj", extractVertices(init), extractIndexes(init));
    }
}

// ------------------------------------------------------------------------------------------

// Wireframe meshes for debugging visualization purposes
struct Mesh
{
    std::vector<Point3f> pts;
    std::vector<Vec2i> lines;

    Mesh join(const Mesh& m2) const
    {
        Mesh mo;

        size_t sz1 = this->pts.size();
        std::copy(this->pts.begin(), this->pts.end(), std::back_inserter(mo.pts));
        std::copy(m2.pts.begin(), m2.pts.end(), std::back_inserter(mo.pts));

        std::copy(this->lines.begin(), this->lines.end(), std::back_inserter(mo.lines));
        std::transform(m2.lines.begin(), m2.lines.end(), std::back_inserter(mo.lines),
                       [sz1](Vec2i ab) { return Vec2i(ab[0] + (int)sz1, ab[1] + (int)sz1); });

        return mo;
    }

    Mesh transform(Affine3f a, float scale = 1.f) const
    {
        Mesh out;
        out.lines = this->lines;
        for (Point3f p : this->pts)
        {
            out.pts.push_back(a * (p * scale));
        }
        return out;
    }

    // 0-2 - min, 3-5 - max
    Vec6f getBoundingBox() const
    {
        float maxv = std::numeric_limits<float>::max();
        Vec3f xmin(maxv, maxv, maxv), xmax(-maxv, -maxv, -maxv);
        for (Point3f p : this->pts)
        {
            xmin[0] = min(p.x, xmin[0]); xmin[1] = min(p.y, xmin[1]); xmin[2] = min(p.z, xmin[2]);
            xmax[0] = max(p.x, xmax[0]); xmax[1] = max(p.y, xmax[1]); xmax[2] = max(p.z, xmax[2]);
        }
        return Vec6f(xmin[0], xmin[1], xmin[2], xmax[0], xmax[1], xmax[2]);
    }
};


Mesh seg7(int d)
{
    const std::vector<Point3f> pt = { {0, 0, 0}, {0, 1, 0},
                                      {1, 0, 0}, {1, 1, 0},
                                      {2, 0, 0}, {2, 1, 0} };

    std::vector<Mesh> seg(7);
    seg[0].pts = { pt[0], pt[1] };
    seg[1].pts = { pt[1], pt[3] };
    seg[2].pts = { pt[3], pt[5] };
    seg[3].pts = { pt[5], pt[4] };
    seg[4].pts = { pt[4], pt[2] };
    seg[5].pts = { pt[2], pt[0] };
    seg[6].pts = { pt[2], pt[3] };
    for (int i = 0; i < 7; i++)
        seg[i].lines = { {0, 1} };

    vector<Mesh> digits = {
        seg[0].join(seg[1]).join(seg[2]).join(seg[3]).join(seg[4]).join(seg[5]), // 0
        seg[1].join(seg[2]), // 1
        seg[0].join(seg[1]).join(seg[3]).join(seg[4]).join(seg[6]), // 2
        seg[0].join(seg[1]).join(seg[2]).join(seg[3]).join(seg[6]), // 3
        seg[1].join(seg[2]).join(seg[5]).join(seg[6]), // 4
        seg[0].join(seg[2]).join(seg[3]).join(seg[5]).join(seg[6]), // 5
        seg[0].join(seg[2]).join(seg[3]).join(seg[4]).join(seg[5]).join(seg[6]), // 6
        seg[0].join(seg[1]).join(seg[2]), // 7
        seg[0].join(seg[1]).join(seg[2]).join(seg[3]).join(seg[4]).join(seg[5]).join(seg[6]), // 8
        seg[0].join(seg[1]).join(seg[2]).join(seg[3]).join(seg[5]).join(seg[6]), // 9
        seg[6], // -
    };

    return digits[d];
}

Mesh drawId(size_t x)
{
    vector<int> digits;
    do
    {
        digits.push_back(x % 10);
        x /= 10;
    }
    while (x > 0);
    float spacing = 0.2f;
    Mesh m;
    for (size_t i = 0; i < digits.size(); i++)
    {
        Mesh digit = seg7(digits[digits.size() - 1 - i]);
        Vec6f bb = digit.getBoundingBox();
        digit = digit.transform(Affine3f().translate(-Vec3f(0, bb[1], 0)));
        Vec3f tr;
        if (m.pts.empty())
            tr = Vec3f();
        else
            tr = Vec3f(0, (m.getBoundingBox()[4] + spacing), 0);
        m = m.join(digit.transform( Affine3f().translate(tr) ));
    }
    return m;
}


Mesh drawFromTo(size_t f, size_t t)
{
    Mesh m;

    Mesh df = drawId(f);
    Mesh dp = seg7(10);
    Mesh dt = drawId(t);

    float spacing = 0.2f;
    m = m.join(df).join(dp.transform(Affine3f().translate(Vec3f(0, df.getBoundingBox()[4] + spacing, 0))))
                  .join(dt.transform(Affine3f().translate(Vec3f(0, df.getBoundingBox()[4] + 2*spacing + 1, 0))));

    return m;
}

Mesh drawPoseGraph(Ptr<detail::PoseGraph> pg)
{
    Mesh marker;
    marker.pts = { {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0} };
    marker.lines = { {0, 1}, {0, 2}, {0, 3}, {1, 4} };

    Mesh allMeshes;
    Affine3f margin = Affine3f().translate(Vec3f(0.1f, 0.1f, 0));
    std::vector<size_t> ids = pg->getNodesIds();
    for (const size_t& id : ids)
    {
        Affine3f pose = pg->getNodePose(id);

        Mesh m = marker.join(drawId(id).transform(margin, 0.25f)).transform(pose);
        allMeshes = allMeshes.join(m);
    }

    // edges
    margin = Affine3f().translate(Vec3f(0.05f, 0.05f, 0));
    for (size_t i = 0; i < pg->getNumEdges(); i++)
    {
        Affine3f pose = pg->getEdgePose(i);
        size_t sid = pg->getEdgeStart(i);
        size_t did = pg->getEdgeEnd(i);
        Affine3f spose = pg->getNodePose(sid);
        Affine3f dpose = spose * pose;

        Mesh m = marker.join(drawFromTo(sid, did).transform(margin, 0.125f)).transform(dpose);
        allMeshes = allMeshes.join(m);
    }

    return allMeshes;
}

void writeObj(const std::string& fname, const Mesh& m)
{
    // Write edge-only model of how nodes are located in space
    std::fstream of(fname, std::fstream::out);
    for (const Point3f& d : m.pts)
    {
        of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
    }

    for (const Vec2i& v : m.lines)
    {
        of << "l " << v[0] + 1 << " " << v[1] + 1 << std::endl;
    }

    of.close();
}


TEST(PoseGraph, simple)
{

    Ptr<detail::PoseGraph> pg = detail::PoseGraph::create();

    DualQuatf true0(1, 0, 0, 0, 0, 0, 0, 0);
    DualQuatf true1 = DualQuatf::createFromPitch((float)CV_PI / 3.0f, 10.0f, Vec3f(1, 1.5f, 1.2f), Vec3f());

    DualQuatf pose0 = true0;
    vector<DualQuatf> noise(7);
    for (size_t i = 0; i < noise.size(); i++)
    {
        float angle = cv::theRNG().uniform(-1.f, 1.f);
        float shift = cv::theRNG().uniform(-2.f, 2.f);
        Matx31f axis = Vec3f::randu(0.f, 1.f), moment = Vec3f::randu(0.f, 1.f);
        noise[i] = DualQuatf::createFromPitch(angle, shift,
            Vec3f(axis(0), axis(1), axis(2)),
            Vec3f(moment(0), moment(1), moment(2)));
    }

    DualQuatf pose1 = noise[0] * true1;

    DualQuatf diff = true1 * true0.inv();
    vector<DualQuatf> cfrom = { diff, diff * noise[1], noise[2] * diff };
    DualQuatf diffInv = diff.inv();
    vector<DualQuatf> cto = { diffInv, diffInv * noise[3], noise[4] * diffInv };

    pg->addNode(123, pose0.toAffine3(), true);
    pg->addNode(456, pose1.toAffine3(), false);

    Matx66f info = Matx66f::eye();
    for (int i = 0; i < 3; i++)
    {
        pg->addEdge(123, 456, cfrom[i].toAffine3(), info);
        pg->addEdge(456, 123, cto[i].toAffine3(), info);
    }

    Mesh allMeshes = drawPoseGraph(pg);

    // Add the "--test_debug" to arguments to see resulting pose graph nodes positions
    if (cvtest::debugLevel > 0)
    {
        writeObj("pg_simple_in.obj", allMeshes);
    }

    auto r = pg->optimize();

    Mesh after = drawPoseGraph(pg);

    // Add the "--test_debug" to arguments to see resulting pose graph nodes positions
    if (cvtest::debugLevel > 0)
    {
        writeObj("pg_simple_out.obj", after);
    }

    EXPECT_TRUE(r.found);
}
#else

TEST(PoseGraph, sphereG2O)
{
    throw SkipTestException("Build with Eigen required for pose graph optimization");
}

TEST(PoseGraphMST, optimization)
{
    throw SkipTestException("Build with Eigen required for pose graph optimization");
}

TEST(PoseGraph, simple)
{
    throw SkipTestException("Build with Eigen required for pose graph optimization");
}
#endif

TEST(LevMarq, Rosenbrock)
{
    auto f = [](double x, double y) -> double
    {
        return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
    };

    auto j = [](double x, double y) -> Matx12d
    {
        return {/*dx*/ -2.0 + 2.0 * x - 400.0 * x * y + 400.0 * x*x*x,
                /*dy*/ 200.0 * y - 200.0 * x*x,
                };
    };

    LevMarq solver(2, [f, j](InputOutputArray param, OutputArray err, OutputArray jv) -> bool
    {
            Vec2d v = param.getMat();
            double x = v[0], y = v[1];
            err.create(1, 1, CV_64F);
            err.getMat().at<double>(0) = f(x, y);
            if (jv.needed())
            {
                jv.create(1, 2, CV_64F);
                Mat(j(x, y)).copyTo(jv);
            }
            return true;
    },
    LevMarq::Settings().setGeodesic(true));

    Mat_<double> x (Vec2d(1, 3));

    auto r = solver.run(x);

    EXPECT_TRUE(r.found);
    EXPECT_LT(r.energy, 0.035);
    EXPECT_LE(r.iters, 17);
}


}} // namespace
