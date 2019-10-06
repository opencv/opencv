/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <math.h>
#include <string>
#include <map>
#include <set>

namespace opencv_test { namespace {

//Generates vector of all available PnPSolvers
std::vector<Ptr<PnPSolver>> generateSolvers()
{
    std::vector<Ptr<PnPSolver>> solvers;
    solvers.push_back(PnPSolverP3PComplete::create());
    solvers.push_back(PnPSolverAP3P::create());
    solvers.push_back(PnPSolverIPPE::create());
    solvers.push_back(PnPSolverIPPESquare::create());
    solvers.push_back(PnPSolverZhang::create());
    solvers.push_back(PnPSolverDLT::create());
    solvers.push_back(PnPSolverEPnP3D::create());
    return solvers;
}

//Generates vector of all available PnPRefiners
std::vector<Ptr<PnPRefiner>> generateRefiners()
{
    std::vector<Ptr<PnPRefiner>> refiners;
    refiners.push_back(PnPRefinerLM::create());
    refiners.push_back(PnPRefinerLMcpp::create());
    refiners.push_back(PnPRefinerVVS::create());
    refiners.push_back(Ptr<PnPRefiner>()); //last refiner is empty pointer. This is used when we run pnp without refining the solution from the PnPSolver
    return refiners;
}

/**
 * @brief solverName returns string associated with a PnPSolver. Used for printing to terminal if a test fails.
 * @param s the PnPSolver
 * @return name of s
 */
string solverName(const Ptr<PnPSolver> s)
{
    string ret;
    if (s.dynamicCast<PnPSolverP3PComplete>())
    {
        ret = "PnPSolverP3P";
    }
    else if (s.dynamicCast<PnPSolverAP3P>())
    {
        ret = "PnPSolverAP3P";
    }
    else if (s.dynamicCast<PnPSolverIPPE>())
    {
        ret = "PnPSolverIPPE";
    }
    else if (s.dynamicCast<PnPSolverZhang>())
    {
        ret = "PnPSolverZhang";
    }
    else if (s.dynamicCast<PnPSolverIPPESquare>())
    {
        ret = "PnPSolverIPPESquare";
    }
    else if (s.dynamicCast<PnPSolverDLT>())
    {
        ret = "PnPSolverLinearDLS";
    }
    else if (s.dynamicCast<PnPSolverEPnP3D>())
    {
        ret = "PnPSolverEPnP3D";
    }
    else if (s.dynamicCast<PnPSolverAutoSelect1>())
    {
        ret = "PnPSolverAuto";
    }
    CV_Assert(!ret.empty());
    return ret;

}

/**
 * @brief refinerName returns string associated with a PnPRefiner. Used for printing to terminal if a test fails.
 * @param r the PnPRefiner
 * @return name of r
 */
string refinerName(const Ptr<PnPRefiner> r)
{
    string ret;
    if (r.empty())
    {
        ret = "None";
    }
    else if (r.dynamicCast<PnPRefinerLM>())
    {
        ret = "PnPRefinerLM";
    }
    else if (r.dynamicCast<PnPRefinerLMcpp>())
    {
        ret = "PnPRefinerLMImpl2";
    }
    else if (r.dynamicCast<PnPRefinerVVS>())
    {
        ret = "PnPRefinerVVS";
    }
    CV_Assert(!ret.empty());
    return ret;

}

/**
 * @brief The ObjectPointShape enum holds all supported shapes for
 * the object points argument in pnp
 */
enum ObjectPointShape
{
    O_MAT_1CH_Nx3, O_MAT_3CH_1xN, MAT_3CH_Nx1, VEC_POINT3, OPS_LEN
};

/**
 * @brief The ImagePointShape enum holds all supported shapes for
 * the image points argument in pnp
 */
enum ImagePointShape
{
    I_MAT_1CH_Nx2, I_MAT2CH_1xN, I_MAT2CH_Nx1, VEC_POINT2, IPS_LEN
};

/**
 * @brief The ObjectGeometry enum holds all supported geometries for
 * the object points. This is either co-planar, co-planar square (a tag), 3D (non-coplanar)
 */
enum ObjectGeometry
{
    OBJECT_COPLANAR, OBJECT_SQUARE, OBJECT_3D, OBJ_GEOM_LEN //last entry holds enumeration count
};


/**
 * @brief The PnPProblem struct. Holds all data needed to solve a pnp problem. Includes ground truth pose.
 */
struct PnPProblem
{
    Mat oPts, imgPts, K, d, rvecGT, tvecGT; //object points, image points, intrinsic, distortion, rotation vector ground truth, translation
    //vector ground truth
    ObjectGeometry objectShape; //object shape geometry
    size_t getNumPts() const; //gets number of points
};

/**
 * @brief The ProblemFormat struct. Defines input types for Mats in a PnPProblem object
 */
struct ProblemFormat
{
    bool doubleObjectPoints; //true if object points are double otherwise false and object points are float
    bool doubleIntrinsics; //true if intrinsics are double otherwise false and intrinsics are float
    bool doubleDistCoeffs; //true if distortion coefficients are double otherwise false and they are float
    bool doubleImagePoints; //true if image points are double otherwise false and image points are float
    ObjectPointShape objShape;
    ImagePointShape imgShape;

    ProblemFormat();

    static vector<ProblemFormat> createAllFormats(); //creates all combinations of problem formats.
};


/**
 * @brief printProblemFormat prints problem format to terminal. Used if a test fails.
 * @param f problem format
 * @return string coding of problem format
 */
string printProblemFormat(const ProblemFormat & f);


/**
 * @brief The ProblemGenerator class. Base class for genetating a sequence of PnP problems
 */
class ProblemGenerator
{
public:
    ProblemGenerator(const cv::RNG & rng, double mNoiseSigma);
    virtual ~ProblemGenerator();

    /**
     * @brief generate generates the next problem in the sequence
     * @param prob generated problem
     */
    void generate(PnPProblem & prob) const;

    /**
     * @brief toCameraCoords computes the camera coordinate positions of the object points
     * @param prob pnp problem
     * @return camera coordinates of object points
     */
    cv::Mat toCameraCoords(const PnPProblem & prob) const;

    /**
     * @brief getCentroid gets centroid of a set of points. Must be 1-channel Nx3 double
     * @param pts points
     * @return centroid. 1-channel 3x1 double
     */
    cv::Mat getCentroid(const Mat & pts) const;

    /**
     * @brief frontOfCameraCheck checks if problem is valid by testing if all object points
     * in camera coordinates are in front of the camera (have positive z coefficients)
     * @param prob PnP problem
     * @return true if test passes, false otherwise
     */
    bool frontOfCameraCheck(const PnPProblem & prob) const;

    /**
     * @brief addWhiteNoiseImgPts adds zero mean I.I.D gaussian noise (white noise) to the
     * image points. Used to simulate real noise of measured point positions in an image
     * @param prob PnP problem
     */
    void addWhiteNoiseImgPts(PnPProblem & prob) const;


protected:
    mutable cv::RNG mRng;
    double mNoiseSigma; //image point noise to apply (standard deviation). If zero, no
    //noise is added

    /**
     * @brief create creates the next PnP problem in the sequenxe
     * @param prob created PnP problem
     */
    virtual void create(PnPProblem & prob) const = 0;  //override this

};


/**
 * @brief The ProblemGeneratorEpnP class generates random simulated pnp problems according to the EPnP paper @cite lepetit2009epnp
 */
class ProblemGeneratorEpnP : public ProblemGenerator
{
public:
    ProblemGeneratorEpnP(bool centred, size_t n, double noiseSigma, const cv::RNG & rng);

    /**
     * @brief generateK generates intrinsic matrix
     * @return intrinsic matrix
     */
    Mat generateK() const;

    /**
     * @brief generateD generates distortion coefficients
     * @return distortion coefficients
     */
    Mat generateD() const;


    /**
     * @brief generateCamPts generates object points defined in camera coordinates
     * @return object points in camera coordinates
     */
    Mat generateCamPts() const;

    /**
     * @brief generateRandomRvec generates random rotation vector (camera-to-object)
     * @return rotation vector
     */
    Mat generateRandomRvec() const;

private:
    size_t mNumPts; //numbe of points to generate
    bool mCentred; //set true if object points are centred (see EPnP paper)

    /**
     * @brief makeRandomObjectPts makes random set of object points
     * @param rvecGT rotation vector ground truth
     * @param tvecGT translation vector ground truth
     * @param camPts object points in camera coordinates
     * @param objPts object points in object coordinates
     */
    void makeRandomObjectPts(const Mat &rvecGT, const Mat &tvecGT, const Mat &camPts, Mat &objPts) const;


protected:

    /**
     * @brief create creates the next PnP problem in the sequenxe
     * @param prob created PnP problem
     */
    void create(PnPProblem & prob) const;

};

/**
 * @brief The ProblemGeneratorIPPE class generates random simulated pnp problems based on method in the IPPE paper @ref PnPSolverIPPE
 */
class ProblemGeneratorIPPE : public ProblemGenerator
{
public:
    ProblemGeneratorIPPE(int n, double w, double noiseSigma, const cv::RNG & rng);

    /**
     * @brief slantAngle computes the angle between the object's normal vector and the
     * line-of-sight passing through the centroid of the object. If this angle is close to
     * 90 degrees, the problem becomes ill-posed and cannot be solved. slantAngle is used to eliminate poses that
     * are ill-posed.
     * @param prob PnP problem
     * @return the slant angle
     */
    double slantAngle(const PnPProblem & prob) const;

    /**
     * @brief checkProblem checks if PnP problem is valid (object points in front of camera
     * and slant angle is not beyond 80 degrees
     * @param prob PnP problem
     * @return true if check passes, false otherwise
     */
    bool checkProblem(const PnPProblem & prob) const;

    /**
     * @brief generateRandomObjectPts generates a random set of object points.
     * Created randomly with uniform probability of x and y coordinate in the range +-mW
     * @param shape sets to OBJECT_COPLANAR
     * @return
     */
    virtual Mat generateObjectPts(ObjectGeometry & shape) const;

    /**
     * @brief generateRandomRVec generates random rotation vector (object-to-camera)
     * @return rotation vector
     */
    Mat generateRandomRVec() const;

    /**
     * @brief generateRandomTVec generates random translation vector (object-to-camera)
     * @return translation vector
     */
    Mat generateRandomTVec() const;

    /**
     * @brief generateK generates intrinsic matrix
     * @return intrinsic matrix
     */
    Mat generateK() const;

    /**
     * @brief generateD generates distortion coefficients
     * @return distortion coefficients
     */
    Mat generateD() const;

protected:
    /**
     * @brief create creates the next PnP problem in the sequenxe
     * @param prob created PnP problem
     */
    void create(PnPProblem & prob) const;
    int mNumPts; //number of points to generate
    double mW; //w parameter in page 14 first paragraph of IPPE paper

};

/**
 * @brief The ProblemGeneratorIPPETag class generates ranom problems with a square object (a tag)
 */
class ProblemGeneratorIPPETag : public ProblemGeneratorIPPE
{
public:

    /**
     * @brief ProblemGeneratorIPPETag constructor
     * @param tagHeight tag height, same as tag width.
     * @param noiseSigma amount of noise in image points (standard deviation)
     * @param rng random generator
     */
    ProblemGeneratorIPPETag(double tagHeight, double noiseSigma, const cv::RNG & rng);

    /**
     * @brief generateObjectPts generates object points (4 tag corners)
     * @param shape sets to OBJECT_SQUARE
     * @return generated object points
     */
    virtual Mat generateObjectPts(ObjectGeometry & shape) const override;
private:

};

/**
 * @brief The ProblemGeneratorFormatTest class generates problems used for testing correct problem format
 */
class ProblemGeneratorFormatTest : public ProblemGenerator
{
public:
    ProblemGeneratorFormatTest(const cv::RNG & rng, double noise);

    /**
     * @brief numberOfCases returns the number of problem cases
     * @return number of problem cases
     */
    size_t numberOfCases() const;
protected:

    /**
     * @brief create creates the next PnP problem in the sequenxe
     * @param prob created PnP problem
     */
    void create(PnPProblem & prob) const;
private:
    mutable size_t genCounter; //counter used to get the next problem in problems
    std::vector<PnPProblem> problems; //set of problems
    };


/**
 * @brief The ProblemGeneratorFailures class generates problems used for testing problems that
 * cause PnP solvers to fail. This is useful to collect a list of failure modes for each PnPSolver.
 */
class ProblemGeneratorFailures : public ProblemGenerator
{
public:
    ProblemGeneratorFailures(const cv::RNG & rng);

    size_t numberOfCases() const;

    /**
    * @brief generate generates next problem in the problem sequence
    * @param prob generated problem
    * @param solverFails vector holding names of all PnPSolvers that fail with prob
    */
    void generate(PnPProblem & prob, std::set<string> & solverFails) const;

protected:

    /**
     * @brief create this does nothing. You should use generate.
     * @param prob
     */
    void create(PnPProblem & prob) const;

private:
    mutable int genCounter; //used for keeping track of next problem to generate
    vector<PnPProblem> problems; //set of problem
    vector<std::set<string>> solverFails; //solverFails[i] holds names of all PnPSolvers
    //that fail for the corresponding problem problems[i]

};


/**
 * @brief computeTErr computes translation error according to standard definition (e.g. EPnP and IPPE papers)
 * @param tvecGT ground truth translation vector
 * @param tvecEst estimated translation vector
 * @return relative translation error in range 0 to 1
 */
static double computeTErr(const Mat & tvecGT, const Mat & tvecEst)
{
    double n1 = cv::norm(tvecGT-tvecEst);
    double n2 = cv::norm(tvecEst);
    return n1/n2;
}

/**
 * @brief computeTErr computes translation error according to standard definition (e.g. IPPE papers)
 * @param rvecGT ground truth translation vector
 * @param rvecEst estimated translation vector
 * @return rotation error defined as the minimal rotation angle needed to rotate matrix defined by rvecGT
 * to matrix defined by rvecEst. In the range 0 to 180 (degrees). 0 means no rotation error.
 */
static double computeRErr(const Mat & rvecGT, const Mat & rvecEst)
{
    Mat RGT, REst;
    cv::Rodrigues(rvecGT,RGT);
    cv::Rodrigues(rvecEst,REst);

    Mat RDiff = RGT.t()*REst;
    Mat rDiffVec;
    cv::Rodrigues(RDiff,rDiffVec);

    double n1 = cv::norm(rDiffVec) * 2.0*M_PI;
    return n1;
}


/**
 * @brief The ErrorTols struct defines error tolerances for testing precision of PnPSolvers and PnPRefiners
 */
struct ErrorTols
{
    double preRefineTrans,preRefineRot; //pre-refinement translation and rotation tolerences
    double postRefineTrans,postRefineRot; //post-refinement translation and rotation tolerences
    ErrorTols();
};


/**
 * @brief callPnP wrapper to calling pnp. A PnP problem is defined
 * by a PnPProblem p and a ProblemFormat f
 * @param p PnP problem
 * @param f problem format
 * @param rvecs rotation vectors
 * @param tvecs translation vectors
 * @param solver PnP solver. Use Ptr<PnPSolver>() to indicate no solver is used. If so, then values in rvecs and tvecs are
 * passed to the PnP refiner.
 * @param refiner PnP refiner. Use Ptr<PnPRefiner>() to indicate no refiner is used.
 * @param sortOnReprojErr sort results on reprojection error?
 * @param reprojectionError reprojection error of solutions
 * @return number of solutions
 */
int callPnP(const PnPProblem  & p,
                        const ProblemFormat & f,
                        vector<Mat> & rvecs, vector<Mat> & tvecs,
                        const Ptr<PnPSolver> solver, const Ptr<PnPRefiner> refiner,
                        bool sortOnReprojErr = true,
                        OutputArray reprojectionError = noArray());


//tests precision of all PnP refiners
TEST(Calib3d_PnP, refinerPrecision)
{
    cv::RNG rng = theRNG();
    const std::vector<Ptr<PnPRefiner>> refiners = generateRefiners();
    const vector<ProblemFormat> fs  = ProblemFormat::createAllFormats();
    const ErrorTols errTols;
    const ProblemGeneratorFormatTest pg(rng,0.0); //the problem generator. No noise is used.
    const size_t numGenerations = pg.numberOfCases();


    //loop over all problem formats:
    for (auto & f : fs)
    {
        //loop over all PnP problems
        for (size_t t = 0; t < numGenerations; t++)
        {
            Mat rvecInit, tvecInit; // stores the initial rotation and translation estimate
            PnPProblem prob;
            pg.generate(prob); //generates the PnP problem

            rvecInit = prob.rvecGT.clone();
            tvecInit = prob.tvecGT.clone();

            //perturb the initial estimates with small noise:
            rvecInit.at<double>(0) = rvecInit.at<double>(0) + rng.uniform(-0.1,0.1);
            rvecInit.at<double>(1) = rvecInit.at<double>(1) + rng.uniform(-0.1,0.1);
            rvecInit.at<double>(2) = rvecInit.at<double>(2) + rng.uniform(-0.1,0.1);

            tvecInit.at<double>(0) = tvecInit.at<double>(0) + rng.uniform(-1.0,1.0);
            tvecInit.at<double>(1) = tvecInit.at<double>(1) + rng.uniform(-1.0,1.0);
            tvecInit.at<double>(2) = tvecInit.at<double>(2) + rng.uniform(-1.0,1.0);

            //loop over all PnP refiners
            for (auto r : refiners)
            {
                if (r.empty())
                {
                    continue;
                }
                //compute reprojection errors of the initial pose estimate.
                //this can be done using  callSolvePnPGeneric with an empty PnP solver and
                //an empty PnP refiner
                vector<Mat> rvecs, tvecs;
                Mat reprojErrPreRefine, reprojErrPostRefine;
                rvecs.push_back(rvecInit.clone());
                tvecs.push_back(tvecInit.clone());

                auto numSolutions = callPnP(prob,f,rvecs,tvecs,Ptr<PnPSolver>(), Ptr<PnPRefiner>(), true,reprojErrPreRefine);
                EXPECT_TRUE(numSolutions==1);

                //refine the initial pose and get its new reprojection error:
                numSolutions = callPnP(prob,fs[0],rvecs,tvecs,Ptr<PnPSolver>(), r, true,reprojErrPostRefine);

                //compare reprojection error before and after refinement
                auto rErrorBefore = reprojErrPreRefine.at<double>(0);
                auto rErrorAfter = reprojErrPostRefine.at<double>(0);
                EXPECT_LE(rErrorAfter,rErrorBefore);

                //ensure that the refined poses are accurate within precision limits
                auto bestRerr = computeRErr(prob.rvecGT,rvecs[0]);
                auto bestTerr = computeTErr(prob.tvecGT,tvecs[0]);

                EXPECT_LE(bestRerr,1e-6);
                EXPECT_LE(bestTerr,1e-6);
            }
        }
    }
}

//tests precision of PnP refiners with noisy inputs.
//This tests if the final reprojection error is improved
//using a PnP refiner. This test is only valid for
//pnp refiners that minimize the reprojection error.
TEST(Calib3d_PnP, refinerWithNoise)
{
    std::vector<Ptr<PnPRefiner>> refiners;
    refiners.push_back(PnPRefinerLM::create());
    refiners.push_back(PnPRefinerLMcpp::create());
    //vvs dpes not minimizing reproj error

    const vector<ProblemFormat> fs  = ProblemFormat::createAllFormats();
    const ErrorTols errTols;
    map<string,float> successRateThresholds;
    successRateThresholds[solverName(PnPSolverP3PComplete::create())] = 1.0;  //PnPSolverP3PComplete fails often
    cv::RNG rng = theRNG();
    const ProblemGeneratorFormatTest pg(rng,0.05);
    const auto n = pg.numberOfCases();
    //loop over problem formats
    for (auto & f : fs)
    {
        //loop over PnP problems
        for (size_t t = 0; t < n; t++)
        {
            Mat rvecInit, tvecInit; //initial pose estimate
            PnPProblem prob;
            pg.generate(prob);
            rvecInit = prob.rvecGT.clone();
            tvecInit = prob.tvecGT.clone();

            //perturb initial pose estimates by small noise
            rvecInit.at<double>(0) = rvecInit.at<double>(0) + rng.uniform(-0.1,0.1);
            rvecInit.at<double>(1) = rvecInit.at<double>(1) + rng.uniform(-0.1,0.1);
            rvecInit.at<double>(2) = rvecInit.at<double>(2) + rng.uniform(-0.1,0.1);

            tvecInit.at<double>(0) = tvecInit.at<double>(0) + rng.uniform(-1.0,1.0);
            tvecInit.at<double>(1) = tvecInit.at<double>(1) + rng.uniform(-1.0,1.0);
            tvecInit.at<double>(2) = tvecInit.at<double>(2) + rng.uniform(-1.0,1.0);

            //loop over PnP refiners
            for (auto r : refiners)
            {
                if (r.empty())
                {
                    //don't consider an empty refiner
                    continue;
                }

                //compute reprojection error before refinement:
                vector<Mat> rvecs, tvecs;
                Mat reprojErrPreRefine, reprojErrPostRefine;
                rvecs.push_back(rvecInit.clone());
                tvecs.push_back(tvecInit.clone());

                auto numSolutions = callPnP(prob,f,rvecs,tvecs,Ptr<PnPSolver>(), Ptr<PnPRefiner>(), true,reprojErrPreRefine);
                EXPECT_TRUE(numSolutions==1);

                //refine and re-compute reprojection error:
                numSolutions = callPnP(prob,fs[0],rvecs,tvecs,Ptr<PnPSolver>(), r, true,reprojErrPostRefine);
                auto rErrorBefore = reprojErrPreRefine.at<double>(0);
                auto rErrorAfter = reprojErrPostRefine.at<double>(0);
                //ensure reprojection error after refinement is lower than before refinement
                EXPECT_LE(rErrorAfter,rErrorBefore);
            }
        }
    }
}

/**
 * @brief runMethods runs all combinations of PnPSolvers and PnPRefiners on PnP problems generated by a PnP problem generator.
 * If a PnPSolver cannot solve a given PnP problem then the test is skipped. This occurs if
 * e.g. the number of points in the PnP problem is not compatible with the PnPSolver (such as 5 points and PnPSolverP3P)
 * It is also skipped if the PnP solver cannot solve the problem because of its geometry. For example, if the PnP solver
 * only works with co-planar points, and the PnP problem has non-coplanar points.
 * @param probGenerator PnP problem generator
 * @param probFormat PnP problem format
 * @param numGenerations number of problems to generate
 * @param errTols error tolerances used for indicating if a solution is correct
 * @param successRateThresholds. For each PnPSolver, we define a success rate criteria.
 * This gives the proportion of successfully solved problems needed to pass the test.
 * For example,if successRateThresholds["solver"] = 1 then for the test to pass, all problems must be solved with
 * the PnPSolver named "solver" to the error tolerance defined in errTols.
 * If a PnP solver's name is not in successRateThresholds then it's success rate threshold is defined as 1.0
 * @return if the test passes or fails. The method returns on the first failed test.
 */
bool runMethods(const std::vector<Ptr<ProblemGenerator>> & probGenerator,
                const ProblemFormat & probFormat,
                size_t numGenerations,
                const ErrorTols & errTols,
                const map<string,float> & successRateThresholds);


//tests precision of all PnPSolver and PnPRefiner combinations, for all PnP problem formats.
TEST(Calib3d_PnP, formats)
{
    const vector<ProblemFormat> fs  = ProblemFormat::createAllFormats();
    const ErrorTols errTols;
    map<string,float> successRateThresholds;
    auto rng = theRNG();
    std::vector<Ptr<ProblemGenerator>> probGenerator;
    auto pPtr = makePtr<ProblemGeneratorFormatTest>(rng,0.0);
    probGenerator.push_back(pPtr);
    const size_t numTrials = pPtr->numberOfCases();
    for (auto & f : fs)
    {
        auto succ = runMethods(probGenerator,f,numTrials,errTols,successRateThresholds);
        if (!succ)
        {
            cout << "failed for format " << printProblemFormat(f) << endl;
        }
        EXPECT_TRUE(succ);
    }
}

//tests precision of all PnPSolver and PnPRefiner combinations for 3D (non-coplanar) objects
TEST(Calib3d_PnP, precision3DObjects1)
{
    const int numGenerations = 100; //number of generated problems
    const ErrorTols errTols;

    map<string,float> successRateThresholds;
    successRateThresholds[solverName(PnPSolverP3PComplete::create())] = 0.6f; //PnPSolverP3PComplete fails often to meet high precision
    successRateThresholds[solverName(PnPSolverEPnP3D::create())] = 0.1f; //PnPSolverEPnP3D fails often with 4 points

    ProblemFormat f;
    auto rng = theRNG();

    //define problem generators with number of points varying from 3 to 4
    std::vector<Ptr<ProblemGenerator>> probGenerator;
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,3,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,4,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,3,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,4,0.0,rng));

    EXPECT_TRUE(runMethods(probGenerator,f,numGenerations,errTols,successRateThresholds));
}


//tests precision of all PnPSolver and PnPRefiner combinations for 3D (non-coplanar) objects
TEST(Calib3d_PnP, precision3DObjects2)
{
    const int numGenerations = 100; //number of generated problems
    const ErrorTols errTols;

    map<string,float> successRateThresholds;
    ProblemFormat f;
    auto rng = theRNG();

    //define problem generators with number of points varying from 5 to 100
    std::vector<Ptr<ProblemGenerator>> probGenerator;
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,5,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,6,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,10,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(true,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,5,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,6,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,10,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorEpnP>(false,100,0.0,rng));

    EXPECT_TRUE(runMethods(probGenerator,f,numGenerations,errTols,successRateThresholds));
}

//tests precision of all PnPSolver and PnPRefiner combinations for coplanar objects
TEST(Calib3d_PnP, precisionCoPlanarObjects1)
{
    const int numGenerations = 100;
    const ErrorTols errTols;

    map<string,float> successRateThresholds;
    successRateThresholds[solverName(PnPSolverP3PComplete::create())] = 0.6f; //PnPSolverP3PComplete fails often to meet high precision
    successRateThresholds[solverName(PnPSolverZhang::create())] = 0.7f;

    const ProblemFormat f;
    auto rng = theRNG();

    //define problem generators with number of points varying from 3 to 100
    std::vector<Ptr<ProblemGenerator>> probGenerator;
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(3,300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(4,300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(5,300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(6,300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(10,300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(100,300,0.0,rng));

    EXPECT_TRUE(runMethods(probGenerator,f,numGenerations,errTols,successRateThresholds));
}

//tests precision of all PnPSolver and PnPRefiner combinations for coplanar objects.
//These objects are smaller than those defined in the previous test.
TEST(Calib3d_PnP, precisionCoPlanarObjects2)
{
    //same as zeroNoiseCoPlanarObjects1 but with smaller objects
    const int numTrials = 100;
    const ErrorTols errTols;

    map<string,float> successRateThresholds;
    successRateThresholds[solverName(PnPSolverP3PComplete::create())] = 0.1f; //PnPSolverP3P fails often to meet high precision
    successRateThresholds[solverName(PnPSolverZhang::create())] = 0.1f;  //PnPSolverZhang fails often to meet high precision

    const ProblemFormat f;
    auto rng = theRNG();

    std::vector<Ptr<ProblemGenerator>> probGenerator;
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(3,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(4,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(5,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(6,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(10,100,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPE>(100,100,0.0,rng));

    EXPECT_TRUE(runMethods(probGenerator,f,numTrials,errTols,successRateThresholds));
}

//tests precision of all PnPSolver and PnPRefiner combinations for planar tag objects.
TEST(Calib3d_PnP, precisionTagObjects)
{
    const int numTrials = 100;
    const ErrorTols errTols;

    map<string,float> successRateThresholds;
    successRateThresholds[solverName(PnPSolverP3PComplete::create())] = 0.9f;
    successRateThresholds[solverName(PnPSolverZhang::create())] = 0.9f;

    const ProblemFormat f;
    auto rng = theRNG();

    std::vector<Ptr<ProblemGenerator>> probGenerator;
    probGenerator.push_back(makePtr<ProblemGeneratorIPPETag>(300,0.0,rng));
    probGenerator.push_back(makePtr<ProblemGeneratorIPPETag>(1000,0.0,rng));

    EXPECT_TRUE(runMethods(probGenerator,f,numTrials,errTols,successRateThresholds));
}


//tests PnP solver failure cases. There exist some inputs for which a PnP solver
//cannot solve the problem. If the problem is actually solvable, then this indicates
//a limitation of the solver algorithm. It is useful to maintain a list of these cases.
TEST(Calib3d_PnP, solverFailures)
{
    const std::vector<Ptr<PnPSolver>> solvers = generateSolvers();
    ProblemFormat f;
    auto rng = theRNG();

    //the problem generator:
    ProblemGeneratorFailures pg(rng);

    double accTol = 1e-2; //pass precision tolerance
    const auto numCases = pg.numberOfCases();

    //loop over all cases:
    for (size_t t = 0; t < numCases; t++)
    {
        Mat rvecInit, tvecInit; //initial pose estimate
        //generate problem:
        PnPProblem prob;
        std::set<string> solverFails;
        pg.generate(prob,solverFails);
        for (auto s : solvers)
        {
            if (!s->validProblem(prob.oPts,prob.imgPts))
            {
                //ignore the solver if it cannot handle the geometry of the problem
                continue;
            }

            //try to solve the problem:
            vector<Mat> rvecs, tvecs;
            Mat reprojErrs;
            auto numSolutions = callPnP(prob,f,rvecs,tvecs,s, Ptr<PnPRefiner>(), true,reprojErrs);
            if (solverFails.find(solverName(s)) == solverFails.end())
            {
                //solver is not known to fail on this problem, so verify
                EXPECT_TRUE(numSolutions>0);
                if (numSolutions>0)
                {
                    double bestReprojErr = reprojErrs.at<double>(0);
                    if (bestReprojErr>=accTol)
                    {
                        cout <<"solver " << solverName(s) << " failed but it was not in failure list for case " << t << std::endl;
                    }
                    EXPECT_TRUE(bestReprojErr<accTol);
                }
            }
            else {
                //solver is known to fail on this problem, so verify
                if (numSolutions>0)
                {
                    double bestReprojErr = reprojErrs.at<double>(0);
                    if (bestReprojErr<accTol)
                    {
                        cout <<"solver " << solverName(s) << " passed but it was in failure list for case " << t << std::endl;
                    }
                    EXPECT_TRUE(bestReprojErr>=accTol);
                }
            }
        }
    }
}


size_t PnPProblem::getNumPts() const
{
    return static_cast<size_t>(oPts.rows);
}

ProblemFormat::ProblemFormat(): doubleObjectPoints(true), doubleIntrinsics(true), doubleDistCoeffs(true),
    doubleImagePoints(true), objShape(O_MAT_1CH_Nx3), imgShape(I_MAT_1CH_Nx2)
{

}

string printProblemFormat(const ProblemFormat & f)
{
    string out;

    out = "doubleObjectPoints: " + std::to_string(static_cast<int>(f.doubleObjectPoints)) + " " +
            "doubleImagePoints: " + std::to_string(static_cast<int>(f.doubleImagePoints)) + " " +
            "doubleIntrinsics: " + std::to_string(static_cast<int>(f.doubleIntrinsics)) + " " +
            "objShape: " + std::to_string(static_cast<int>(f.objShape)) + " " +
            "imgShape: " + std::to_string(static_cast<int>(f.imgShape));
    return out;
}

vector<ProblemFormat> ProblemFormat::createAllFormats()
{
    vector<ProblemFormat> formats;
    for (bool objDouble : { false, true })
    {
        for (bool imgDouble : { false, true })
        {
            for (bool distDouble : { false, true })
            {
                for (bool intrinsDouble : { false, true })
                {
                    for ( int opsInt = 0; opsInt != OPS_LEN; opsInt++ )
                    {
                        for ( int ipsInt = 0; ipsInt != IPS_LEN; ipsInt++ )
                        {
                            ProblemFormat f;
                            f.doubleObjectPoints = objDouble;
                            f.doubleImagePoints = imgDouble;
                            f.doubleIntrinsics = intrinsDouble;
                            f.doubleDistCoeffs = distDouble;
                            f.objShape = static_cast<ObjectPointShape>(opsInt);
                            f.imgShape = static_cast<ImagePointShape>(ipsInt);
                            formats.push_back(f);
                        }
                    }
                }
            }
        }
    }
    return formats;
}


ProblemGenerator::ProblemGenerator(const RNG &rng, double noiseSigma): mRng(rng), mNoiseSigma(noiseSigma)
{

}

ProblemGenerator::~ProblemGenerator() {}

void ProblemGenerator::generate(PnPProblem &prob) const
{
    create(prob);
    CV_Assert(frontOfCameraCheck(prob));
    addWhiteNoiseImgPts(prob);
}

Mat ProblemGenerator::toCameraCoords(const PnPProblem &prob) const
{
    Mat R;
    cv::Rodrigues(prob.rvecGT,R);
    const auto n = prob.getNumPts();
    Mat camPts(static_cast<int>(n),3,CV_64FC1);
    for (size_t i = 0; i < n; i++)
    {
        cv::Mat p(3,1,CV_64FC1);
        int iint = static_cast<int>(i);
        p.at<double>(0) = prob.oPts.at<double>(iint,0);
        p.at<double>(1) = prob.oPts.at<double>(iint,1);
        p.at<double>(2) = prob.oPts.at<double>(iint,2);

        p = R*p + prob.tvecGT;

        camPts.at<double>(iint,0) = p.at<double>(0);
        camPts.at<double>(iint,1) = p.at<double>(1);
        camPts.at<double>(iint,2) = p.at<double>(2);
    }
    return camPts;
}

Mat ProblemGenerator::getCentroid(const Mat &pts) const
{
    const int n = pts.rows;
    cv::Mat c(3,1,CV_64FC1);
    c.setTo(0.0);
    for (int i = 0; i < n; i++)
    {
        c.at<double>(0) = c.at<double>(0) + pts.at<double>(i,0);
        c.at<double>(1) = c.at<double>(1) + pts.at<double>(i,1);
        c.at<double>(2) = c.at<double>(2) + pts.at<double>(i,2);
    }
    c = c /static_cast<double>(n);
    return c;
}

bool ProblemGenerator::frontOfCameraCheck(const PnPProblem &prob) const
{
    const Mat camPts = toCameraCoords(prob);
    const auto n = prob.getNumPts();
    bool front = true;
    for (size_t i = 0; i < n; i++)
    {
        front = front & (camPts.at<double>(static_cast<int>(i),2)>0.0);
    }
    return front;
}

void ProblemGenerator::addWhiteNoiseImgPts(PnPProblem &prob) const
{
    for (size_t i = 0; i < prob.getNumPts(); i++)
    {
        int iint = static_cast<int>(i);
        prob.imgPts.at<double>(iint,0) = prob.imgPts.at<double>(iint,0) + mRng.gaussian(mNoiseSigma);
        prob.imgPts.at<double>(iint,1) = prob.imgPts.at<double>(iint,1) + mRng.gaussian(mNoiseSigma);
    }
}

ProblemGeneratorEpnP::ProblemGeneratorEpnP(bool centred, size_t n, double noiseSigma, const RNG &rng): ProblemGenerator(rng,noiseSigma), mNumPts(n), mCentred(centred)
{

}

void ProblemGeneratorEpnP::create(PnPProblem &prob) const
{
    prob.objectShape = OBJECT_3D;
    prob.K = generateK();
    prob.d = generateD();
    Mat camPts = generateCamPts();
    Mat r(3,1,CV_64FC1); r.setTo(0.0);
    Mat t(3,1,CV_64FC1); t.setTo(0.0);
    cv::projectPoints(camPts,r,t,prob.K,prob.d,prob.imgPts);
    Mat cent = getCentroid(camPts);

    prob.tvecGT = Mat(3,1,CV_64FC1);
    prob.tvecGT.at<double>(0) = cent.at<double>(0);
    prob.tvecGT.at<double>(1) = cent.at<double>(1);
    prob.tvecGT.at<double>(2) = cent.at<double>(2);

    Mat rv = generateRandomRvec();
    prob.rvecGT = Mat(3,1,CV_64FC1);
    rv.copyTo(prob.rvecGT);

    makeRandomObjectPts(prob.rvecGT,prob.tvecGT,camPts,prob.oPts);

}

Mat ProblemGeneratorEpnP::generateK() const
{
    Mat K(3,3,CV_64FC1);
    K.setTo(0.0);
    K.at<double>(0,0) = 800.0;
    K.at<double>(1,1) = 800.0;
    K.at<double>(2,2) = 1.0;
    K.at<double>(0,2) = 320.0;
    K.at<double>(1,2) = 240.0;
    return K;
}

Mat ProblemGeneratorEpnP::generateD() const
{
    return Mat();
}


Mat ProblemGeneratorEpnP::generateCamPts() const
{
    Mat camPts = Mat(static_cast<int>(mNumPts),3,CV_64FC1);
    for (size_t i = 0; i < mNumPts; i++)
    {
        const auto iint = static_cast<int>(i);
        if (mCentred)
        {
            camPts.at<double>(iint,0) = mRng.uniform(-2.0, 2.0);
            camPts.at<double>(iint,1) = mRng.uniform(-2.0, 2.0);
            camPts.at<double>(iint,2) = mRng.uniform(4.0, 8.0);
        }
        else {
            camPts.at<double>(iint,0) = mRng.uniform(1.0, 2.0);
            camPts.at<double>(iint,1) = mRng.uniform(1.0, 2.0);
            camPts.at<double>(iint,2) = mRng.uniform(4.0, 8.0);
        }
    }
    return camPts;
}

Mat ProblemGeneratorEpnP::generateRandomRvec() const
{
    double alpha=mRng.uniform(0.0, 2.0*M_PI);
    double beta=mRng.uniform(0.0, 2.0*M_PI);
    double gamma=mRng.uniform(0.0, 2.0*M_PI);

    Mat R(3,3,CV_64FC1);
    R.setTo(0.0);

    R.at<double>(0,0)=cos(alpha)*cos(gamma)-cos(beta)*sin(alpha)*sin(gamma);
    R.at<double>(1,0)=cos(gamma)*sin(alpha)+cos(alpha)*cos(beta)*sin(gamma);
    R.at<double>(2,0)=sin(beta)*sin(gamma);

    R.at<double>(0,1)=-cos(beta)*cos(gamma)*sin(alpha)-cos(alpha)*sin(gamma);
    R.at<double>(1,1)=cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma);
    R.at<double>(2,1)=cos(gamma)*sin(beta);

    R.at<double>(0,2)=sin(alpha)*sin(beta);
    R.at<double>(1,2)=-cos(alpha)*sin(beta);
    R.at<double>(2,2)=cos(beta);

    Mat rvec;
    cv::Rodrigues(R,rvec);
    return rvec;
}

void ProblemGeneratorEpnP::makeRandomObjectPts(const Mat &rvecGT, const Mat &tvecGT, const Mat &camPts, Mat &objPts) const
{
    Mat R,rt;
    cv::Rodrigues(rvecGT,R);
    objPts = Mat(static_cast<int>(mNumPts),3,CV_64FC1);
    rt = -R.t()*tvecGT;
    for (size_t i = 0; i < mNumPts; i++)
    {
        const auto iint = static_cast<int>(i);
        cv::Mat p(3,1,CV_64FC1);
        p.at<double>(0) = camPts.at<double>(iint,0);
        p.at<double>(1) = camPts.at<double>(iint,1);
        p.at<double>(2) = camPts.at<double>(iint,2);

        p = R.t()*p + rt;
        objPts.at<double>(iint,0) = p.at<double>(0);
        objPts.at<double>(iint,1) = p.at<double>(1);
        objPts.at<double>(iint,2) = p.at<double>(2);
    }

}


ProblemGeneratorIPPE::ProblemGeneratorIPPE(int n, double w, double noiseSigma, const RNG &rng):  ProblemGenerator(rng,noiseSigma), mNumPts(n), mW(w)
{

}

void ProblemGeneratorIPPE::create(PnPProblem &prob) const
{
    bool foundGoodProblem = false;
    while (!foundGoodProblem)
    {
        prob.K = generateK();
        prob.d = generateD();
        prob.tvecGT = generateRandomTVec();
        prob.rvecGT = generateRandomRVec();
        prob.oPts = generateObjectPts(prob.objectShape);

        cv::projectPoints(prob.oPts,prob.rvecGT,prob.tvecGT,prob.K,prob.d,prob.imgPts);
        foundGoodProblem = checkProblem(prob);
    }
}

double ProblemGeneratorIPPE::slantAngle(const PnPProblem &prob) const
{
    Mat ptsCam = this->toCameraCoords(prob);
    Mat cCam = this->getCentroid(ptsCam);
    Mat centroidVec = cCam / cv::norm(cCam);

    Point3d p0, p1, p2, v01, v02;
    p0.x = ptsCam.at<double>(0,0);
    p0.y = ptsCam.at<double>(0,1);
    p0.z = ptsCam.at<double>(0,2);

    p1.x = ptsCam.at<double>(1,0);
    p1.y = ptsCam.at<double>(1,1);
    p1.z = ptsCam.at<double>(1,2);

    p2.x = ptsCam.at<double>(2,0);
    p2.y = ptsCam.at<double>(2,1);
    p2.z = ptsCam.at<double>(2,2);

    v01 = p1-p0;
    v02 = p2-p0;

    Point3d crs = v01.cross(v02);
    double crsnrm = cv::norm(crs);
    crs.x = crs.x / crsnrm;
    crs.y = crs.y / crsnrm;
    crs.z = crs.z / crsnrm;

    Point3d centroidVecp(centroidVec);
    double dp = centroidVecp.dot(crs);
    double ang = acos(dp);
    ang = min(ang,M_PI - ang);
    double angDeg = 180.0 * (ang / M_PI);
    return angDeg;
}

bool ProblemGeneratorIPPE::checkProblem(const PnPProblem &prob) const
{
    bool frontCheck = frontOfCameraCheck(prob);
    bool angCheck = slantAngle(prob)< 80.0;
    return frontCheck && angCheck;
}

Mat ProblemGeneratorIPPE::generateObjectPts(ObjectGeometry &shape) const
{
    shape = OBJECT_COPLANAR;
    Mat opts(mNumPts,3,CV_64FC1);
    for (int i = 0; i < mNumPts; i++)
    {
        opts.at<double>(i,0) =  mRng.uniform(-mW / 2.0, mW / 2.0);
        opts.at<double>(i,1) =  mRng.uniform(-mW / 2.0, mW / 2.0);
        opts.at<double>(i,2) =  0.0;
    }

    Mat rv = generateRandomRVec();
    Mat rMat;
    cv::Rodrigues(rv, rMat);
    opts = opts*rMat.t();
    return opts;
}

Mat ProblemGeneratorIPPE::generateRandomRVec() const
{
    Mat rv(3,1,CV_64FC1);
    rv.at<double>(0) = mRng.uniform(0.0, 2.0*M_PI);
    rv.at<double>(1) = mRng.uniform(0.0, 2.0*M_PI);
    rv.at<double>(2) = mRng.uniform(0.0, 2.0*M_PI);
    return rv;
}

Mat ProblemGeneratorIPPE::generateRandomTVec() const
{
    Mat K = generateK();
    cv::Mat p(3,1,CV_64FC1);
    p.at<double>(0) = mRng.uniform(0.0,640.0);
    p.at<double>(1) = mRng.uniform(0.0,480.0);
    p.at<double>(2) = 1.0;

    Mat Kinv;
    cv::invert(K,Kinv);
    Mat pu = Kinv*p;

    double dpth = mRng.uniform(K.at<double>(0,0)/2.0,K.at<double>(0,0)*2.0);
    return pu*dpth;
}

Mat ProblemGeneratorIPPE::generateK() const
{
    Mat K(3,3,CV_64FC1);
    K.setTo(0.0);
    K.at<double>(0,0) = 800.0;
    K.at<double>(1,1) = 800.0;
    K.at<double>(2,2) = 1.0;
    K.at<double>(0,2) = 320.0;
    K.at<double>(1,2) = 240.0;
    return K;
}

Mat ProblemGeneratorIPPE::generateD() const
{
    return Mat();
}

ProblemGeneratorIPPETag::ProblemGeneratorIPPETag(double tagHeight, double noiseSigma, const RNG &rng): ProblemGeneratorIPPE(4,tagHeight,noiseSigma,rng)
{

}

Mat ProblemGeneratorIPPETag::generateObjectPts(ObjectGeometry &shape) const
{
    shape = OBJECT_SQUARE;
    Mat opts(mNumPts,3,CV_64FC1);
    opts.setTo(0.0);
    opts.at<double>(0,0) = -mW / 2.0;
    opts.at<double>(0,1) =  mW / 2.0;
    opts.at<double>(1,0) =  mW / 2.0;
    opts.at<double>(1,1) =  mW / 2.0;
    opts.at<double>(2,0) =  mW / 2.0;
    opts.at<double>(2,1) =  -mW / 2.0;
    opts.at<double>(3,0) = -mW / 2.0;
    opts.at<double>(3,1) =  -mW / 2.0;

    return opts;
}



ProblemGeneratorFormatTest::ProblemGeneratorFormatTest(const RNG &rng, double noise): ProblemGenerator(rng,noise), genCounter(0)
{
    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(3, 3) << 12.00604, -2.8654366, 18.472504,
                     7.6863389, 4.9355154, 11.146358,
                     14.260933, 2.8320458, 12.582781);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
    }
    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(4, 3) << 12.00604, -2.8654366, 18.472504,
                     7.6863389, 4.9355154, 11.146358,
                     14.260933, 2.8320458, 12.582781,
                     3.4562225, 8.2668982, 11.300434);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
    }
    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(5, 3) << 12.00604, -2.8654366, 18.472504,
                     7.6863389, 4.9355154, 11.146358,
                     14.260933, 2.8320458, 12.582781,
                     3.4562225, 8.2668982, 11.300434,
                     10.00604,  2.8654366, 15.472504);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
    }
    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(6, 3) << 12.00604, -2.8654366, 18.472504,
                     7.6863389, 4.9355154, 11.146358,
                     14.260933, 2.8320458, 12.582781,
                     3.4562225, 8.2668982, 11.300434,
                     10.00604,  2.8654366, 15.472504,
                     -4.6863389, 5.9355154, 13.146358);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
    }
    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(6, 3) <<  0, -2.8654366, 18.472504,
                     0, 4.9355154, 11.146358,
                     0, 2.8320458, 12.582781,
                     0, 8.2668982, 11.300434,
                     0,  2.8654366, 15.472504,
                     0, 5.9355154, 13.146358);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
    }

}

size_t ProblemGeneratorFormatTest::numberOfCases() const
{
    return problems.size();
}

void ProblemGeneratorFormatTest::create(PnPProblem &prob) const
{
    if (genCounter< problems.size())
    {
        prob = problems.at(genCounter);
        genCounter++;
    }
    else {
        prob = problems.at( problems.size()-1);
    }
}





ErrorTols::ErrorTols(): preRefineTrans(1e-2), preRefineRot(1e-2),
    postRefineTrans(1e-5), postRefineRot(1e-5)
{

}


int callPnP(const PnPProblem  & p,
                        const ProblemFormat & f,
                        vector<Mat> & rvecs, vector<Mat> & tvecs,
                        const Ptr<PnPSolver> solver, const Ptr<PnPRefiner> refiner,
                        bool sortOnReprojErr,
                        OutputArray reprojectionError)
{

    //some ugly logic needed to handle all possible problem formats. Would be nice
    //to template this but it doesn't work well with OpenCV's dynamic Mat typing.
    Mat intrinsics, dist;
    if (!f.doubleIntrinsics){p.K.convertTo(intrinsics,CV_32FC1);}
    else {p.K.convertTo(intrinsics,CV_64FC1);}

    if (!f.doubleDistCoeffs){p.d.convertTo(dist,CV_32FC1);}
    else {p.d.convertTo(dist,CV_64FC1);}

    Mat oPts, iPts;
    if (!f.doubleObjectPoints){p.oPts.convertTo(oPts,CV_32FC1);}
    else {p.oPts.convertTo(oPts,CV_64FC1);}

    if (!f.doubleImagePoints){p.imgPts.convertTo(iPts,CV_32FC1);}
    else {p.imgPts.convertTo(iPts,CV_64FC1);}

    CV_Assert(p.oPts.isContinuous());
    CV_Assert(p.imgPts.isContinuous());

    bool isVecObjPts = false;
    bool isVecImgPts = false;

    switch (f.objShape)
    {
    case O_MAT_1CH_Nx3:{
        oPts = oPts.reshape(1);
        break;}
    case O_MAT_3CH_1xN:{
        oPts = oPts.reshape(3);
        break;}
    case MAT_3CH_Nx1:{
        oPts = oPts.reshape(3).t();
        break;}
    case VEC_POINT3:{
        isVecObjPts = true;
        break;}
    case OPS_LEN:{
        break;}
    }

    switch (f.imgShape)
    {
    case I_MAT_1CH_Nx2:{
        iPts = iPts.reshape(1);
        break;
    }
    case I_MAT2CH_1xN:{
        iPts = iPts.reshape(2);
        break;}
    case I_MAT2CH_Nx1:{
        iPts = iPts.reshape(2).t();
        break;}
    case VEC_POINT2:{
        isVecImgPts = true;
        break;}
    case IPS_LEN:{
        break;}
    }
    int numSolutions;
    if ((!isVecObjPts) && (!isVecImgPts))
    {
        numSolutions = pnp(oPts,iPts,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
    }
    else {
        if (isVecObjPts)
        {
            if (f.doubleObjectPoints)
            {
                Point3d* ptr3 = (Point3d*)oPts.data;
                vector<Point3d> optsVec(ptr3, ptr3 + p.getNumPts());
                if (isVecImgPts)
                {
                    if (f.doubleImagePoints)
                    {
                        Point2d* ptr2 = (Point2d*)iPts.data;
                        vector<Point2d> iptsVec(ptr2, ptr2 + p.getNumPts());
                        numSolutions = pnp(optsVec,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                    }
                    else {
                        Point2f* ptr2 = (Point2f*)iPts.data;
                        vector<Point2f> iptsVec(ptr2, ptr2 + p.getNumPts());
                        numSolutions = pnp(optsVec,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                    }

                }
                else {
                    numSolutions = pnp(optsVec,iPts,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                }
            }
            else {

                Point3f* ptr3 = (Point3f*)oPts.data;
                vector<Point3f> optsVec(ptr3, ptr3 + p.getNumPts());
                if (isVecImgPts)
                {
                    if (f.doubleImagePoints)
                    {
                        Point2d* ptr2 = (Point2d*)iPts.data;
                        vector<Point2d> iptsVec(ptr2, ptr2 + p.getNumPts());
                        numSolutions = pnp(optsVec,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                    }
                    else {
                        Point2f* ptr2 = (Point2f*)iPts.data;
                        vector<Point2f> iptsVec(ptr2, ptr2 + p.getNumPts());
                        numSolutions = pnp(optsVec,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                    }

                }
                else {
                    numSolutions = pnp(optsVec,iPts,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
                }
            }
        }
        else
        {
            if (f.doubleImagePoints)
            {
                Point2d* ptr2 = (Point2d*)iPts.data;
                vector<Point2d> iptsVec(ptr2, ptr2 + p.getNumPts());
                numSolutions = pnp(oPts,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
            }
            else {
                Point2f* ptr2 = (Point2f*)iPts.data;
                vector<Point2f> iptsVec(ptr2, ptr2 + p.getNumPts());
                numSolutions = pnp(oPts,iptsVec,intrinsics,dist,rvecs,tvecs,solver,refiner,sortOnReprojErr,reprojectionError);
            }
        }
    }
    return numSolutions;
}
bool runMethods(const std::vector<Ptr<ProblemGenerator>> & probGenerator,
                const ProblemFormat & probFormat,
                size_t numTrials,
                const ErrorTols & errTols,
                const map<string,float> & successRateThresholds)
{

    std::vector<Ptr<PnPRefiner>> refiners = generateRefiners();
    std::vector<Ptr<PnPSolver>> solvers = generateSolvers();

    for (size_t pgIndx = 0; pgIndx < probGenerator.size(); pgIndx++)
    {
        auto pg  = probGenerator[pgIndx];
        std::vector<PnPProblem> probs;
        for (size_t t = 0; t < numTrials; t++)
        {
            PnPProblem p;
            pg->generate(p);
            probs.push_back(p);
        }
        for (auto s : solvers)
        {
            for (auto r : refiners)
            {
                double rErrorTol,tErrorTol;
                if (r.empty())
                {
                    rErrorTol = errTols.preRefineRot;
                    tErrorTol = errTols.preRefineTrans;
                }
                else {
                    rErrorTol = errTols.postRefineRot;
                    tErrorTol = errTols.postRefineTrans;
                }
                int numSuccesses = 0;
                int numAttempts = 0;
                for (size_t numTrial  = 0; numTrial < numTrials; numTrial++)
                {
                    PnPProblem p = probs[numTrial];

                    if (p.objectShape == OBJECT_3D && (s->requiresPlanarObject() | s->requiresPlanarTagObject()))
                    {
                        continue;
                    }
                    if (p.objectShape == OBJECT_COPLANAR && (s->requiresPlanarTagObject() | s->requires3DObject()))
                    {
                        continue;
                    }

                    if (s->validProblem(p.oPts,p.imgPts))
                    {
                        numAttempts++;
                        vector<Mat> rvecs,tvecs;
                        Mat reprojErrs;
                        const size_t numSolutions = callPnP(p,probFormat,rvecs,tvecs,s,r,true,reprojErrs);
                        if (numSolutions==0)
                        {
                            continue;
                        }
                        for (int i = 0; i < static_cast<int>(numSolutions -1); i++)
                        {
                            if (reprojErrs.at<double>(i+1) + 1e-6 <reprojErrs.at<double>(i))
                            {
                                cout << "Failure. Reprojection error order for problemGenerator: " << pgIndx << ", solver: " << solverName(s) << ", refiner: " << refinerName(r) << " is not descending" <<  endl;
                                return false;
                            }

                        }
                        double bestRerr = computeRErr(p.rvecGT,rvecs[0]);
                        double bestTerr = computeTErr(p.tvecGT,tvecs[0]);

                        for (size_t sol = 1; sol < numSolutions; sol++)
                        {
                            double tErr = computeTErr(p.tvecGT,tvecs[sol]);
                            double rErr = computeRErr(p.rvecGT,rvecs[sol]);

                            if (tErr<bestTerr)
                            {
                                bestTerr = tErr;
                            }
                            if (rErr<bestRerr)
                            {
                                bestRerr = rErr;
                            }
                        }
                        if ((bestRerr>rErrorTol) | (bestTerr>tErrorTol))
                        {
                            continue;
                        }
                        numSuccesses++;
                    }
                }
                if (numAttempts!=0)
                {
                    float rateThresh;
                    if ( successRateThresholds.find(solverName(s)) == successRateThresholds.end() ) {
                        rateThresh = 1.0;
                    } else {
                        rateThresh = successRateThresholds.at(solverName(s));
                    }

                    float rate = (float)numSuccesses/(float)numAttempts;
                    if (rate<rateThresh)
                    {
                        cout << "Failure. Success rate for problemGenerator: " << pgIndx << ", solver: " << solverName(s) << ", refiner: " << refinerName(r) << " is " << rate <<  endl;
                        return false;
                    }

                }
            }
        }
    }
    return true;
}


size_t ProblemGeneratorFailures::numberOfCases() const
{
    return problems.size();
}

void ProblemGeneratorFailures::create(PnPProblem &prob) const
{
    (void)prob;
}

void ProblemGeneratorFailures::generate(PnPProblem &prob, std::set<string> &sf) const
{
    size_t c = genCounter;
    if (c >= problems.size())
    {
        c = problems.size()-1;
    }
    else {
        genCounter++;
    }
    prob = problems.at(c);
    sf = solverFails.at(c);

}

ProblemGeneratorFailures::ProblemGeneratorFailures(const RNG &rng) : ProblemGenerator(rng, 0.0),  genCounter(0)
{
    {
        PnPProblem prob;
        double wdth = 20.0;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(4, 3) << -wdth / 2.0,  wdth / 2.0, 0,
                      wdth / 2.0,  wdth / 2.0, 0,
                      wdth / 2.0, -wdth / 2.0, 0,
                      -wdth / 2.0,  -wdth / 2.0, 0);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
        std::set<string> fails = {solverName(PnPSolverZhang::create()), //degenerate configuration for Zhang implementation
                                  solverName(PnPSolverDLT::create())};  //degenerate configuration (co-planar object)
 // object not square
        solverFails.push_back(fails);
    }

    {
        PnPProblem prob;
        prob.K = Mat::eye(3,3,CV_64FC1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat(Matx31d(0.9072420896651262, 0.09226497171882152, 0.8880772883671504));
        prob.tvecGT = Mat(Matx31d(7.376333362427632, 8.434449036856979, 13.79801619778456));
        prob.oPts = (Mat_<double>(6, 3) << 12.00604, -2.8654366, 0,
                     7.6863389, 4.9355154, 0,
                     14.260933, 2.8320458, 0,
                     3.4562225, 8.2668982, 0,
                     10.00604,  2.8654366, 0,
                     -4.6863389, 5.9355154, 0);
        projectPoints(prob.oPts, prob.rvecGT, prob.tvecGT, prob.K, prob.d, prob.imgPts);
        problems.push_back(prob);
        std::set<string> fails = {solverName(PnPSolverZhang::create()), //degenerate configuration (unknown reason)
                                  solverName(PnPSolverDLT::create()),  //degenerate configuration (co-planar object)
                                  solverName(PnPSolverIPPESquare::create())}; // object not square
        solverFails.push_back(fails);
    }

    {
        PnPProblem prob; // co-planar example
        prob.K = (Mat_<double>(3, 3) << 800, 0, 320,
                  0, 800, 240,
                  0,0,1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat();
        prob.tvecGT = Mat();
        prob.oPts = (Mat_<double>(5, 3) <<125.4911844070545, -141.6464856132886, 2.1035540918816,
                15.56081673101709, 146.8582460146606, -0.6425994678159378,
                77.35197691728332, 126.1239553416953, 0.1238782914796581,
                126.3792277274632, -141.7947735558015, 2.113747132966365,
                -102.4491895513266, 48.7904801704582, -1.350008425218844);
        prob.imgPts = (Mat_<double>(5, 2) << 211.3172241863653, 258.7392188558077,
                146.542077788426, 323.4183455020655,
                154.4149593976568, 343.2930499685378,
                211.4546846117436, 259.0963999763775,
                154.1419661384817, 239.8017744206864);
        problems.push_back(prob);
        std::set<string> fails = {solverName(PnPSolverDLT::create()),
                                  solverName(PnPSolverIPPESquare::create()),  //object not square
                                  solverName(PnPSolverEPnP3D::create())}; //degenerate configuration (unknown reason)
        solverFails.push_back(fails);
    }

    {
        PnPProblem prob;
        prob.K = (Mat_<double>(3, 3) << 800, 0, 320,
                  0, 800, 240,
                  0,0,1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat();
        prob.tvecGT = Mat();
        prob.oPts = (Mat_<double>(5, 3) << 54.50760732727414, 90.39974569723172, -16.5218902770834,
                -95.25016027022656, -8.847660462565507, -35.9773337903876,
                -85.12253904642863, 67.30713703924297, -64.86019671380593,
                124.1790118960073, 68.85302191697907, 21.97830276201228,
                53.77392574181322, 53.10773695353704, -0.6115363162486993);
        prob.imgPts = (Mat_<double>(5, 2) << 440.469369662699, 164.2063931256786,
                303.5507874119983, 116.5595777076413,
                351.0364046372179, 99.7866968358413,
                460.30720787272, 199.614585342235,
                417.6648503519709, 172.2234198887387);
        problems.push_back(prob);
        std::set<string> fails = {solverName(PnPSolverDLT::create()),
                                  solverName(PnPSolverIPPESquare::create()),  //object not square
                                  solverName(PnPSolverEPnP3D::create())}; //degenerate configuration (unknown reason)
        solverFails.push_back(fails);
    }

    {
        PnPProblem prob;
        prob.K = (Mat_<double>(3, 3) << 547.9413023815613, 0, 298.3554570004314,
                0, 548.17724002728, 230.6219405198623,
                0, 0, 1);
        prob.d = Mat::zeros(1,5,CV_64FC1);
        prob.rvecGT = Mat();
        prob.tvecGT = Mat();
        prob.oPts = (Mat_<double>(4, 3) << 13, 11, 10,
                13, 3, 2,
                7, 5, 4,
                11, 3, 2);
        prob.imgPts = (Mat_<double>(4, 2) << -8, 8,
                0, 8,
                -2, 2,
                0, 6);
        problems.push_back(prob);
        std::set<string> fails = {solverName(PnPSolverP3PComplete::create()), //degenerate configuration (unknown reason)
                                  solverName(PnPSolverZhang::create())}; //degenerate configuration (unknown reason)
        solverFails.push_back(fails);
    }

}

}
} // namespace
