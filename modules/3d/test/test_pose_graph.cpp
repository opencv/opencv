// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

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
static Ptr<kinfu::detail::PoseGraph> readG2OFile(const std::string& g2oFileName)
{
    Ptr<kinfu::detail::PoseGraph> pg = kinfu::detail::PoseGraph::create();

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


TEST( PoseGraph, sphereG2O )
{
    // Test takes 15+ sec in Release mode and 400+ sec in Debug mode
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG);

    // The dataset was taken from here: https://lucacarlone.mit.edu/datasets/
    // Connected paper:
    // L.Carlone, R.Tron, K.Daniilidis, and F.Dellaert.
    // Initialization Techniques for 3D SLAM : a Survey on Rotation Estimation and its Use in Pose Graph Optimization.
    // In IEEE Intl.Conf.on Robotics and Automation(ICRA), pages 4597 - 4604, 2015.

    std::string filename = cvtest::TS::ptr()->get_data_path() + "rgbd/sphere_bignoise_vertex3.g2o";
    Ptr<kinfu::detail::PoseGraph> pg = readG2OFile(filename);

#ifdef HAVE_EIGEN
    // You may change logging level to view detailed optimization report
    // For example, set env. variable like this: OPENCV_LOG_LEVEL=INFO

    int iters = pg->optimize();

    ASSERT_GE(iters, 0);
    ASSERT_LE(iters, 36); // should converge in 36 iterations

    double energy = pg->calcEnergy();

    ASSERT_LE(energy, 1.47723e+06); // should converge to 1.47722e+06 or less

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
#else
    throw SkipTestException("Build with Eigen required for pose graph optimization");
#endif
}


}} // namespace