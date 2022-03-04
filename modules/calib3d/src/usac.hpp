// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_USAC_USAC_HPP
#define OPENCV_USAC_USAC_HPP

namespace cv { namespace usac {
enum EstimationMethod { Homography, Fundamental, Fundamental8, Essential, Affine, P3P, P6P};
enum VerificationMethod { NullVerifier, SprtVerifier };
enum PolishingMethod { NonePolisher, LSQPolisher };
enum ErrorMetric {DIST_TO_LINE, SAMPSON_ERR, SGD_ERR, SYMM_REPR_ERR, FORW_REPR_ERR, RERPOJ};

// Abstract Error class
class Error : public Algorithm {
public:
    // set model to use getError() function
    virtual void setModelParameters (const Mat &model) = 0;
    // returns error of point wih @point_idx w.r.t. model
    virtual float getError (int point_idx) const = 0;
    virtual const std::vector<float> &getErrors (const Mat &model) = 0;
    virtual Ptr<Error> clone () const = 0;
};

// Symmetric Reprojection Error for Homography
class ReprojectionErrorSymmetric : public Error {
public:
    static Ptr<ReprojectionErrorSymmetric> create(const Mat &points);
};

// Forward Reprojection Error for Homography
class ReprojectionErrorForward : public Error {
public:
    static Ptr<ReprojectionErrorForward> create(const Mat &points);
};

// Sampson Error for Fundamental matrix
class SampsonError : public Error {
public:
    static Ptr<SampsonError> create(const Mat &points);
};

// Symmetric Geometric Distance (to epipolar lines) for Fundamental and Essential matrix
class SymmetricGeometricDistance : public Error {
public:
    static Ptr<SymmetricGeometricDistance> create(const Mat &points);
};

// Reprojection Error for Projection matrix
class ReprojectionErrorPmatrix : public Error {
public:
    static Ptr<ReprojectionErrorPmatrix> create(const Mat &points);
};

// Reprojection Error for Affine matrix
class ReprojectionErrorAffine : public Error {
public:
    static Ptr<ReprojectionErrorAffine> create(const Mat &points);
};

// Normalizing transformation of data points
class NormTransform : public Algorithm {
public:
    /*
     * @norm_points is output matrix of size pts_size x 4
     * @sample constains indices of points
     * @sample_number is number of used points in sample <0; sample_number)
     * @T1, T2 are output transformation matrices
     */
    virtual void getNormTransformation (Mat &norm_points, const std::vector<int> &sample,
                                        int sample_number, Matx33d &T1, Matx33d &T2) const = 0;
    static Ptr<NormTransform> create (const Mat &points);
};

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SOLVER ///////////////////////////////////////////
class MinimalSolver : public Algorithm {
public:
    // Estimate models from minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    // return minimal sample size required for estimation.
    virtual int getSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
    virtual Ptr<MinimalSolver> clone () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyMinimalSolver4ptsGEM : public MinimalSolver {
public:
    static Ptr<HomographyMinimalSolver4ptsGEM> create(const Mat &points_);
};

//-------------------------- FUNDAMENTAL MATRIX -----------------------
class FundamentalMinimalSolver7pts : public MinimalSolver {
public:
    static Ptr<FundamentalMinimalSolver7pts> create(const Mat &points_);
};

class FundamentalMinimalSolver8pts : public MinimalSolver {
public:
    static Ptr<FundamentalMinimalSolver8pts> create(const Mat &points_);
};

//-------------------------- ESSENTIAL MATRIX -----------------------
class EssentialMinimalSolverStewenius5pts : public MinimalSolver {
public:
    static Ptr<EssentialMinimalSolverStewenius5pts> create(const Mat &points_);
};

//-------------------------- PNP -----------------------
class PnPMinimalSolver6Pts : public MinimalSolver {
public:
    static Ptr<PnPMinimalSolver6Pts> create(const Mat &points_);
};

class P3PSolver : public MinimalSolver {
public:
    static Ptr<P3PSolver> create(const Mat &points_, const Mat &calib_norm_pts, const Matx33d &K);
};

//-------------------------- AFFINE -----------------------
class AffineMinimalSolver : public MinimalSolver {
public:
    static Ptr<AffineMinimalSolver> create(const Mat &points_);
};

//////////////////////////////////////// NON MINIMAL SOLVER ///////////////////////////////////////
class NonMinimalSolver : public Algorithm {
public:
    // Estimate models from non minimal sample. models.size() == number of found solutions
    virtual int estimate (const std::vector<int> &sample, int sample_size,
          std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getMinimumRequiredSampleSize() const = 0;
    // return maximum number of possible solutions.
    virtual int getMaxNumberOfSolutions () const = 0;
    virtual Ptr<NonMinimalSolver> clone () const = 0;
};

//-------------------------- HOMOGRAPHY MATRIX -----------------------
class HomographyNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<HomographyNonMinimalSolver> create(const Mat &points_);
};

//-------------------------- FUNDAMENTAL MATRIX -----------------------
class FundamentalNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<FundamentalNonMinimalSolver> create(const Mat &points_);
};

//-------------------------- ESSENTIAL MATRIX -----------------------
class EssentialNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<EssentialNonMinimalSolver> create(const Mat &points_);
};

//-------------------------- PNP -----------------------
class PnPNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<PnPNonMinimalSolver> create(const Mat &points);
};

class DLSPnP : public NonMinimalSolver {
public:
    static Ptr<DLSPnP> create(const Mat &points_, const Mat &calib_norm_pts, const Matx33d &K);
};

//-------------------------- AFFINE -----------------------
class AffineNonMinimalSolver : public NonMinimalSolver {
public:
    static Ptr<AffineNonMinimalSolver> create(const Mat &points_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SCORE ///////////////////////////////////////////
class Score {
public:
    int inlier_number;
    double score;
    Score () { // set worst case
        inlier_number = 0;
        score = std::numeric_limits<double>::max();
    }
    Score (int inlier_number_, double score_) { // copy constructor
        inlier_number = inlier_number_;
        score = score_;
    }
    // Compare two scores. Objective is minimization of score. Lower score is better.
    inline bool isBetter (const Score &score2) const {
        return score < score2.score;
    }
};

class GammaValues
{
    const double max_range_complete /*= 4.62*/, max_range_gamma /*= 1.52*/;
    const int max_size_table /* = 3000 */;

    std::vector<double> gamma_complete, gamma_incomplete, gamma;

    GammaValues();  // use getSingleton()

public:
    static const GammaValues& getSingleton();

    const std::vector<double>& getCompleteGammaValues() const;
    const std::vector<double>& getIncompleteGammaValues() const;
    const std::vector<double>& getGammaValues() const;
    double getScaleOfGammaCompleteValues () const;
    double getScaleOfGammaValues () const;
    int getTableSize () const;
};

////////////////////////////////////////// QUALITY ///////////////////////////////////////////
class Quality : public Algorithm {
public:
    virtual ~Quality() override = default;
    /*
     * Calculates number of inliers and score of the @model.
     * return Score with calculated inlier_number and score.
     * @model: Mat current model, e.g., H matrix.
     */
    virtual Score getScore (const Mat &model) const = 0;
    virtual Score getScore (const std::vector<float> &/*errors*/) const {
        CV_Error(cv::Error::StsNotImplemented, "getScore(errors)");
    }
    // get @inliers of the @model. Assume threshold is given
    // @inliers must be preallocated to maximum points size.
    virtual int getInliers (const Mat &model, std::vector<int> &inliers) const = 0;
    // get @inliers of the @model for given threshold
    virtual int getInliers (const Mat &model, std::vector<int> &inliers, double thr) const = 0;
    // Set the best score, so evaluation of the model can terminate earlier
    virtual void setBestScore (double best_score_) = 0;
    // set @inliers_mask: true if point i is inlier, false - otherwise.
    virtual int getInliers (const Mat &model, std::vector<bool> &inliers_mask) const = 0;
    virtual int getPointsSize() const = 0;
    virtual Ptr<Quality> clone () const = 0;
    static int getInliers (const Ptr<Error> &error, const Mat &model,
            std::vector<bool> &inliers_mask, double threshold);
    static int getInliers (const Ptr<Error> &error, const Mat &model,
            std::vector<int>  &inliers,      double threshold);
};

// RANSAC (binary) quality
class RansacQuality : public Quality {
public:
    static Ptr<RansacQuality> create(int points_size_, double threshold_,const Ptr<Error> &error_);
};

// M-estimator quality - truncated Squared error
class MsacQuality : public Quality {
public:
    static Ptr<MsacQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

// Marginlizing Sample Consensus quality, D. Barath et al.
class MagsacQuality : public Quality {
public:
    static Ptr<MagsacQuality> create(double maximum_thr, int points_size_,const Ptr<Error> &error_,
                             double tentative_inlier_threshold_, int DoF, double sigma_quantile,
                             double upper_incomplete_of_sigma_quantile,
                             double lower_incomplete_of_sigma_quantile, double C_);
};

// Least Median of Squares Quality
class LMedsQuality : public Quality {
public:
    static Ptr<LMedsQuality> create(int points_size_, double threshold_, const Ptr<Error> &error_);
};

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DEGENERACY //////////////////////////////////
class Degeneracy : public Algorithm {
public:
    virtual ~Degeneracy() override = default;
    /*
     * Check if sample causes degenerate configurations.
     * For example, test if points are collinear.
     */
    virtual bool isSampleGood (const std::vector<int> &/*sample*/) const {
        return true;
    }
    /*
     * Check if model satisfies constraints.
     * For example, test if epipolar geometry satisfies oriented constraint.
     */
    virtual bool isModelValid (const Mat &/*model*/, const std::vector<int> &/*sample*/) const {
        return true;
    }
    /*
     * Fix degenerate model.
     * Return true if model is degenerate, false - otherwise
     */
    virtual bool recoverIfDegenerate (const std::vector<int> &/*sample*/,const Mat &/*best_model*/,
                          Mat &/*non_degenerate_model*/, Score &/*non_degenerate_model_score*/) {
        return false;
    }
    virtual Ptr<Degeneracy> clone(int /*state*/) const { return makePtr<Degeneracy>(); }
};

class EpipolarGeometryDegeneracy : public Degeneracy {
public:
    static void recoverRank (Mat &model, bool is_fundamental_mat);
    static Ptr<EpipolarGeometryDegeneracy> create (const Mat &points_, int sample_size_);
};

class EssentialDegeneracy : public EpipolarGeometryDegeneracy {
public:
    static Ptr<EssentialDegeneracy>create (const Mat &points, int sample_size);
};

class HomographyDegeneracy : public Degeneracy {
public:
    static Ptr<HomographyDegeneracy> create(const Mat &points_);
};

class FundamentalDegeneracy : public EpipolarGeometryDegeneracy {
public:
    // threshold for homography is squared so is around 2.236 pixels
    static Ptr<FundamentalDegeneracy> create (int state, const Ptr<Quality> &quality_,
    const Mat &points_, int sample_size_, double homography_threshold);
};

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// ESTIMATOR //////////////////////////////////
class Estimator : public Algorithm{
public:
    /*
     * Estimate models with minimal solver.
     * Return number of valid solutions after estimation.
     * Return models accordingly to number of solutions.
     * Note, vector of models must allocated before.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModels (const std::vector<int> &sample, std::vector<Mat> &models) const = 0;
    /*
     * Estimate model with non-minimal solver.
     * Return number of valid solutions after estimation.
     * Note, not all degenerate tests are included in estimation.
     */
    virtual int
    estimateModelNonMinimalSample (const std::vector<int> &sample, int sample_size,
                       std::vector<Mat> &models, const std::vector<double> &weights) const = 0;
    // return minimal sample size required for minimal estimation.
    virtual int getMinimalSampleSize () const = 0;
    // return minimal sample size required for non-minimal estimation.
    virtual int getNonMinimalSampleSize () const = 0;
    // return maximum number of possible solutions of minimal estimation.
    virtual int getMaxNumSolutions () const = 0;
    // return maximum number of possible solutions of non-minimal estimation.
    virtual int getMaxNumSolutionsNonMinimal () const = 0;
    virtual Ptr<Estimator> clone() const = 0;
};

class HomographyEstimator : public Estimator {
public:
    static Ptr<HomographyEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class FundamentalEstimator : public Estimator {
public:
    static Ptr<FundamentalEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class EssentialEstimator : public Estimator {
public:
    static Ptr<EssentialEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_, const Ptr<Degeneracy> &degeneracy_);
};

class AffineEstimator : public Estimator {
public:
    static Ptr<AffineEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_);
};

class PnPEstimator : public Estimator {
public:
    static Ptr<PnPEstimator> create (const Ptr<MinimalSolver> &min_solver_,
            const Ptr<NonMinimalSolver> &non_min_solver_);
};

//////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// MODEL VERIFIER ////////////////////////////////////
class ModelVerifier : public Algorithm {
public:
    virtual ~ModelVerifier() override = default;
    // Return true if model is good, false - otherwise.
    virtual bool isModelGood(const Mat &model) = 0;
    // Return true if score was computed during evaluation.
    virtual bool getScore(Score &score) const = 0;
    // update verifier by given inlier number
    virtual void update (int highest_inlier_number) = 0;
    virtual const std::vector<float> &getErrors() const = 0;
    virtual bool hasErrors () const = 0;
    virtual Ptr<ModelVerifier> clone (int state) const = 0;
    static Ptr<ModelVerifier> create();
};

struct SPRT_history {
    /*
     * delta:
     * The probability of a data point being consistent
     * with a ‘bad’ model is modeled as a probability of
     * a random event with Bernoulli distribution with parameter
     * δ : p(1|Hb) = δ.

     * epsilon:
     * The probability p(1|Hg) = ε
     * that any randomly chosen data point is consistent with a ‘good’ model
     * is approximated by the fraction of inliers ε among the data
     * points

     * A is the decision threshold, the only parameter of the Adapted SPRT
     */
    double epsilon, delta, A;
    // number of samples processed by test
    int tested_samples; // k
    SPRT_history () : epsilon(0), delta(0), A(0) {
        tested_samples = 0;
    }
};

///////////////////////////////// SPRT VERIFIER /////////////////////////////////////////
/*
* Matas, Jiri, and Ondrej Chum. "Randomized RANSAC with sequential probability ratio test."
* Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 2. IEEE, 2005.
*/
class SPRT : public ModelVerifier {
public:
    // return constant reference of vector of SPRT histories for SPRT termination.
    virtual const std::vector<SPRT_history> &getSPRTvector () const = 0;
    static Ptr<SPRT> create (int state, const Ptr<Error> &err_, int points_size_,
       double inlier_threshold_, double prob_pt_of_good_model,
       double prob_pt_of_bad_model, double time_sample, double avg_num_models,
       ScoreMethod score_type_);
};

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// SAMPLER ///////////////////////////////////////
class Sampler : public Algorithm {
public:
    virtual ~Sampler() override = default;
    // set new points size
    virtual void setNewPointsSize (int points_size) = 0;
    // generate sample. Fill @sample with indices of points.
    virtual void generateSample (std::vector<int> &sample) = 0;
    virtual Ptr<Sampler> clone (int state) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// NEIGHBORHOOD GRAPH /////////////////////////////////////////
class NeighborhoodGraph : public Algorithm {
public:
    virtual ~NeighborhoodGraph() override = default;
    // Return neighbors of the point with index @point_idx_ in the graph.
    virtual const std::vector<int> &getNeighbors(int point_idx_) const = 0;
};

class RadiusSearchNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<RadiusSearchNeighborhoodGraph> create (const Mat &points, int points_size,
            double radius_, int flann_search_params, int num_kd_trees);
};

class FlannNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<FlannNeighborhoodGraph> create(const Mat &points, int points_size,
      int k_nearest_neighbors_, bool get_distances, int flann_search_params, int num_kd_trees);
    virtual const std::vector<double> &getNeighborsDistances (int idx) const = 0;
};

class GridNeighborhoodGraph : public NeighborhoodGraph {
public:
    static Ptr<GridNeighborhoodGraph> create(const Mat &points, int points_size,
            int cell_size_x_img1_, int cell_size_y_img1_,
            int cell_size_x_img2_, int cell_size_y_img2_, int max_neighbors);
};

////////////////////////////////////// UNIFORM SAMPLER ////////////////////////////////////////////
class UniformSampler : public Sampler {
public:
    static Ptr<UniformSampler> create(int state, int sample_size_, int points_size_);
};

/////////////////////////////////// PROSAC (SIMPLE) SAMPLER ///////////////////////////////////////
class ProsacSimpleSampler : public Sampler {
public:
    static Ptr<ProsacSimpleSampler> create(int state, int points_size_, int sample_size_,
           int max_prosac_samples_count);
};

////////////////////////////////////// PROSAC SAMPLER ////////////////////////////////////////////
class ProsacSampler : public Sampler {
public:
    static Ptr<ProsacSampler> create(int state, int points_size_, int sample_size_,
                                     int growth_max_samples);
    // return number of samples generated (for prosac termination).
    virtual int getKthSample () const = 0;
    // return constant reference of growth function of prosac sampler (for prosac termination)
    virtual const std::vector<int> &getGrowthFunction () const = 0;
    virtual void setTerminationLength (int termination_length) = 0;
};

////////////////////////// NAPSAC (N adjacent points sample consensus) SAMPLER ////////////////////
class NapsacSampler : public Sampler {
public:
    static Ptr<NapsacSampler> create(int state, int points_size_, int sample_size_,
          const Ptr<NeighborhoodGraph> &neighborhood_graph_);
};

////////////////////////////////////// P-NAPSAC SAMPLER /////////////////////////////////////////
class ProgressiveNapsac : public Sampler {
public:
    static Ptr<ProgressiveNapsac> create(int state, int points_size_, int sample_size_,
            const std::vector<Ptr<NeighborhoodGraph>> &layers, int sampler_length);
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// TERMINATION ///////////////////////////////////////////
class TerminationCriteria : public Algorithm {
public:
    // update termination object by given @model and @inlier number.
    // and return maximum number of predicted iteration
    virtual int update(const Mat &model, int inlier_number) = 0;
    // clone termination
    virtual Ptr<TerminationCriteria> clone () const = 0;
};

//////////////////////////////// STANDARD TERMINATION ///////////////////////////////////////////
class StandardTerminationCriteria : public TerminationCriteria {
public:
    static Ptr<StandardTerminationCriteria> create(double confidence, int points_size_,
               int sample_size_, int max_iterations_);
};

///////////////////////////////////// SPRT TERMINATION //////////////////////////////////////////
class SPRTTermination : public TerminationCriteria {
public:
    static Ptr<SPRTTermination> create(const std::vector<SPRT_history> &sprt_histories_,
               double confidence, int points_size_, int sample_size_, int max_iterations_);
};

///////////////////////////// PROGRESSIVE-NAPSAC-SPRT TERMINATION /////////////////////////////////
class SPRTPNapsacTermination : public TerminationCriteria {
public:
    static Ptr<SPRTPNapsacTermination> create(const std::vector<SPRT_history>&
        sprt_histories_, double confidence, int points_size_, int sample_size_,
        int max_iterations_, double relax_coef_);
};

////////////////////////////////////// PROSAC TERMINATION /////////////////////////////////////////
class ProsacTerminationCriteria : public TerminationCriteria {
public:
    static Ptr<ProsacTerminationCriteria> create(const Ptr<ProsacSampler> &sampler_,
         const Ptr<Error> &error_, int points_size_, int sample_size, double confidence,
         int max_iters, int min_termination_length, double beta, double non_randomness_phi,
         double inlier_thresh);
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// UTILS ////////////////////////////////////////////////
namespace Utils {
    /*
     * calibrate points: [x'; 1] = K^-1 [x; 1]
     * @points is matrix N x 4.
     * @norm_points is output matrix N x 4 with calibrated points.
     */
    void calibratePoints (const Matx33d &K1, const Matx33d &K2, const Mat &points, Mat &norm_points);
    void calibrateAndNormalizePointsPnP (const Matx33d &K, const Mat &pts, Mat &calib_norm_pts);
    void normalizeAndDecalibPointsPnP (const Matx33d &K, Mat &pts, Mat &calib_norm_pts);
    void decomposeProjection (const Mat &P, Matx33d &K_, Mat &R, Mat &t, bool same_focal=false);
    double getCalibratedThreshold (double threshold, const Matx33d &K1, const Matx33d &K2);
    float findMedian (std::vector<float> &array);
}
namespace Math {
    // return skew symmetric matrix
    Matx33d getSkewSymmetric(const Vec3d &v_);
    // eliminate matrix with m rows and n columns to be upper triangular.
    bool eliminateUpperTriangular (std::vector<double> &a, int m, int n);
    Matx33d rotVec2RotMat (const Vec3d &v);
    Vec3d rotMat2RotVec (const Matx33d &R);
}

///////////////////////////////////////// RANDOM GENERATOR /////////////////////////////////////
class RandomGenerator : public Algorithm {
public:
    virtual ~RandomGenerator() override = default;
    // interval is <0, max_range);
    virtual void resetGenerator (int max_range) = 0;
    // return sample filled with random numbers
    virtual void generateUniqueRandomSet (std::vector<int> &sample) = 0;
    // fill @sample of size @subset_size with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int subset_size,
                                                                    int max_range) = 0;
    // fill @sample of size @sample.size() with random numbers in range <0, @max_range)
    virtual void generateUniqueRandomSet (std::vector<int> &sample, int max_range) = 0;
    // return subset=sample size
    virtual void setSubsetSize (int subset_sz) = 0;
    virtual int getSubsetSize () const = 0;
    // return random number from <0, max_range), where max_range is from constructor
    virtual int getRandomNumber () = 0;
    // return random number from <0, max_rng)
    virtual int getRandomNumber (int max_rng) = 0;
    virtual const std::vector<int> &generateUniqueRandomSubset (std::vector<int> &array1,
            int size1) = 0;
    virtual Ptr<RandomGenerator> clone (int state) const = 0;
};

class UniformRandomGenerator : public RandomGenerator {
public:
    static Ptr<UniformRandomGenerator> create (int state);
    static Ptr<UniformRandomGenerator> create (int state, int max_range, int subset_size_);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOCAL OPTIMIZATION /////////////////////////////////////////
class LocalOptimization : public Algorithm {
public:
    virtual ~LocalOptimization() override = default;
    /*
     * Refine so-far-the-best RANSAC model in local optimization step.
     * @best_model: so-far-the-best model
     * @new_model: output refined new model.
     * @new_model_score: score of @new_model.
     * Returns bool if model was refined successfully, false - otherwise
     */
    virtual bool refineModel (const Mat &best_model, const Score &best_model_score,
                              Mat &new_model, Score &new_model_score) = 0;
    virtual Ptr<LocalOptimization> clone(int state) const = 0;
};

//////////////////////////////////// GRAPH CUT LO ////////////////////////////////////////
class GraphCut : public LocalOptimization {
public:
    static Ptr<GraphCut>
    create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
           const Ptr<Quality> &quality_, const Ptr<NeighborhoodGraph> &neighborhood_graph_,
           const Ptr<RandomGenerator> &lo_sampler_, double threshold_,
           double spatial_coherence_term, int gc_iters);
};

//////////////////////////////////// INNER + ITERATIVE LO ///////////////////////////////////////
class InnerIterativeLocalOptimization : public LocalOptimization {
public:
    static Ptr<InnerIterativeLocalOptimization>
    create(const Ptr<Estimator> &estimator_, const Ptr<Quality> &quality_,
           const Ptr<RandomGenerator> &lo_sampler_, int pts_size, double threshold_,
           bool is_iterative_, int lo_iter_sample_size_, int lo_inner_iterations,
           int lo_iter_max_iterations, double threshold_multiplier);
};

class SigmaConsensus : public LocalOptimization {
public:
    static Ptr<SigmaConsensus>
    create(const Ptr<Estimator> &estimator_, const Ptr<Error> &error_,
           const Ptr<Quality> &quality, const Ptr<ModelVerifier> &verifier_,
           int max_lo_sample_size, int number_of_irwls_iters_,
           int DoF, double sigma_quantile, double upper_incomplete_of_sigma_quantile,
           double C_, double maximum_thr);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// FINAL MODEL POLISHER //////////////////////////////////////
class FinalModelPolisher : public Algorithm {
public:
    virtual ~FinalModelPolisher() override = default;
    /*
     * Polish so-far-the-best RANSAC model in the end of RANSAC.
     * @model: input final RANSAC model.
     * @new_model: output polished model.
     * @new_score: score of output model.
     * Return true if polishing was successful, false - otherwise.
     */
    virtual bool polishSoFarTheBestModel (const Mat &model, const Score &best_model_score,
            Mat &new_model, Score &new_model_score) = 0;
};

///////////////////////////////////// LEAST SQUARES POLISHER //////////////////////////////////////
class LeastSquaresPolishing : public FinalModelPolisher {
public:
    static Ptr<LeastSquaresPolishing> create (const Ptr<Estimator> &estimator_,
        const Ptr<Quality> &quality_, int lsq_iterations);
};

/////////////////////////////////// RANSAC OUTPUT ///////////////////////////////////
class RansacOutput : public Algorithm {
public:
    virtual ~RansacOutput() override = default;
    static Ptr<RansacOutput> create(const Mat &model_,
        const std::vector<bool> &inliers_mask_,
        int time_mcs_, double score_, int number_inliers_, int number_iterations_,
        int number_estimated_models_, int number_good_models_);

    // Return inliers' indices. size of vector = number of inliers
    virtual const std::vector<int > &getInliers() = 0;
    // Return inliers mask. Vector of points size. 1-inlier, 0-outlier.
    virtual const std::vector<bool> &getInliersMask() const = 0;
    virtual int getTimeMicroSeconds() const = 0;
    virtual int getTimeMicroSeconds1() const = 0;
    virtual int getTimeMilliSeconds2() const = 0;
    virtual int getTimeSeconds3() const = 0;
    virtual int getNumberOfInliers() const = 0;
    virtual int getNumberOfMainIterations() const = 0;
    virtual int getNumberOfGoodModels () const = 0;
    virtual int getNumberOfEstimatedModels () const = 0;
    virtual const Mat &getModel() const = 0;
};

////////////////////////////////////////////// MODEL /////////////////////////////////////////////

class Model : public Algorithm {
public:
    virtual bool isFundamental () const = 0;
    virtual bool isHomography () const = 0;
    virtual bool isEssential () const = 0;
    virtual bool isPnP () const = 0;

    // getters
    virtual int getSampleSize () const = 0;
    virtual bool isParallel() const = 0;
    virtual int getMaxNumHypothesisToTestBeforeRejection() const = 0;
    virtual PolishingMethod getFinalPolisher () const = 0;
    virtual LocalOptimMethod getLO () const = 0;
    virtual ErrorMetric getError () const = 0;
    virtual EstimationMethod getEstimator () const = 0;
    virtual ScoreMethod getScore () const = 0;
    virtual int getMaxIters () const = 0;
    virtual double getConfidence () const = 0;
    virtual double getThreshold () const = 0;
    virtual VerificationMethod getVerifier () const = 0;
    virtual SamplingMethod getSampler () const = 0;
    virtual double getTimeForModelEstimation () const = 0;
    virtual double getSPRTdelta () const = 0;
    virtual double getSPRTepsilon () const = 0;
    virtual double getSPRTavgNumModels () const = 0;
    virtual NeighborSearchMethod getNeighborsSearch () const = 0;
    virtual int getKNN () const = 0;
    virtual int getCellSize () const = 0;
    virtual int getGraphRadius() const = 0;
    virtual double getRelaxCoef () const = 0;

    virtual int getFinalLSQIterations () const = 0;
    virtual int getDegreesOfFreedom () const = 0;
    virtual double getSigmaQuantile () const = 0;
    virtual double getUpperIncompleteOfSigmaQuantile () const = 0;
    virtual double getLowerIncompleteOfSigmaQuantile () const = 0;
    virtual double getC () const = 0;
    virtual double getMaximumThreshold () const = 0;
    virtual double getGraphCutSpatialCoherenceTerm () const = 0;
    virtual int getLOSampleSize () const = 0;
    virtual int getLOThresholdMultiplier() const = 0;
    virtual int getLOIterativeSampleSize() const = 0;
    virtual int getLOIterativeMaxIters() const = 0;
    virtual int getLOInnerMaxIters() const = 0;
    virtual const std::vector<int> &getGridCellNumber () const = 0;
    virtual int getRandomGeneratorState () const = 0;
    virtual int getMaxItersBeforeLO () const = 0;

    // setters
    virtual void setLocalOptimization (LocalOptimMethod lo_) = 0;
    virtual void setKNearestNeighhbors (int knn_) = 0;
    virtual void setNeighborsType (NeighborSearchMethod neighbors) = 0;
    virtual void setCellSize (int cell_size_) = 0;
    virtual void setParallel (bool is_parallel) = 0;
    virtual void setVerifier (VerificationMethod verifier_) = 0;
    virtual void setPolisher (PolishingMethod polisher_) = 0;
    virtual void setError (ErrorMetric error_) = 0;
    virtual void setLOIterations (int iters) = 0;
    virtual void setLOIterativeIters (int iters) = 0;
    virtual void setLOSampleSize (int lo_sample_size) = 0;
    virtual void setThresholdMultiplierLO (double thr_mult) = 0;
    virtual void setRandomGeneratorState (int state) = 0;

    virtual void maskRequired (bool required) = 0;
    virtual bool isMaskRequired () const = 0;
    static Ptr<Model> create(double threshold_, EstimationMethod estimator_, SamplingMethod sampler_,
         double confidence_=0.95, int max_iterations_=5000, ScoreMethod score_ =ScoreMethod::SCORE_METHOD_MSAC);
};

Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method,
                   double ransacReprojThreshold, OutputArray mask,
                   const int maxIters, const double confidence);

Mat findFundamentalMat( InputArray points1, InputArray points2,
    int method, double ransacReprojThreshold, double confidence,
    int maxIters, OutputArray mask=noArray());

bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
         InputArray cameraMatrix, InputArray distCoeffs,
         OutputArray rvec, OutputArray tvec,
         bool useExtrinsicGuess, int iterationsCount,
         float reprojectionError, double confidence,
         OutputArray inliers, int flags);

Mat findEssentialMat( InputArray points1, InputArray points2,
                      InputArray cameraMatrix1,
                      int method, double prob,
                      double threshold, OutputArray mask);

Mat estimateAffine2D(InputArray from, InputArray to, OutputArray inliers,
     int method, double ransacReprojThreshold, int maxIters,
     double confidence, int refineIters);

void saveMask (OutputArray mask, const std::vector<bool> &inliers_mask);
void setParameters (Ptr<Model> &params, EstimationMethod estimator, const UsacParams &usac_params,
        bool mask_need);
bool run (const Ptr<const Model> &params, InputArray points1, InputArray points2, int state,
      Ptr<RansacOutput> &ransac_output, InputArray K1_, InputArray K2_,
      InputArray dist_coeff1, InputArray dist_coeff2);
}}

#endif //OPENCV_USAC_USAC_HPP
