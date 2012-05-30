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

#ifndef __OPENCV_CONTRIB_HPP__
#define __OPENCV_CONTRIB_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#ifdef __cplusplus

/****************************************************************************************\
*                                   Adaptive Skin Detector                               *
\****************************************************************************************/

class CV_EXPORTS CvAdaptiveSkinDetector
{
private:
    enum {
        GSD_HUE_LT = 3,
        GSD_HUE_UT = 33,
        GSD_INTENSITY_LT = 15,
        GSD_INTENSITY_UT = 250
    };
    
    class CV_EXPORTS Histogram
    {
    private:
        enum {
            HistogramSize = (GSD_HUE_UT - GSD_HUE_LT + 1)
        };
        
    protected:
        int findCoverageIndex(double surfaceToCover, int defaultValue = 0);
        
    public:
        CvHistogram *fHistogram;
        Histogram();
        virtual ~Histogram();
        
        void findCurveThresholds(int &x1, int &x2, double percent = 0.05);
        void mergeWith(Histogram *source, double weight);
    };
    
    int nStartCounter, nFrameCount, nSkinHueLowerBound, nSkinHueUpperBound, nMorphingMethod, nSamplingDivider;
    double fHistogramMergeFactor, fHuePercentCovered;
    Histogram histogramHueMotion, skinHueHistogram;
    IplImage *imgHueFrame, *imgSaturationFrame, *imgLastGrayFrame, *imgMotionFrame, *imgFilteredFrame;
    IplImage *imgShrinked, *imgTemp, *imgGrayFrame, *imgHSVFrame;
    
protected:
    void initData(IplImage *src, int widthDivider, int heightDivider);
    void adaptiveFilter();
    
public:
    
    enum {
        MORPHING_METHOD_NONE = 0,
        MORPHING_METHOD_ERODE = 1,
        MORPHING_METHOD_ERODE_ERODE = 2,
        MORPHING_METHOD_ERODE_DILATE = 3
    };
    
    CvAdaptiveSkinDetector(int samplingDivider = 1, int morphingMethod = MORPHING_METHOD_NONE);
    virtual ~CvAdaptiveSkinDetector();
    
    virtual void process(IplImage *inputBGRImage, IplImage *outputHueMask);
};


/****************************************************************************************\
 *                                  Fuzzy MeanShift Tracker                               *
 \****************************************************************************************/

class CV_EXPORTS CvFuzzyPoint {
public:
    double x, y, value;
    
    CvFuzzyPoint(double _x, double _y);
};

class CV_EXPORTS CvFuzzyCurve {
private:
    std::vector<CvFuzzyPoint> points;
    double value, centre;
    
    bool between(double x, double x1, double x2);
    
public:
    CvFuzzyCurve();
    ~CvFuzzyCurve();
    
    void setCentre(double _centre);
    double getCentre();
    void clear();
    void addPoint(double x, double y);
    double calcValue(double param);
    double getValue();
    void setValue(double _value);
};

class CV_EXPORTS CvFuzzyFunction {
public:
    std::vector<CvFuzzyCurve> curves;
    
    CvFuzzyFunction();
    ~CvFuzzyFunction();
    void addCurve(CvFuzzyCurve *curve, double value = 0);
    void resetValues();
    double calcValue();
    CvFuzzyCurve *newCurve();
};

class CV_EXPORTS CvFuzzyRule {
private:
    CvFuzzyCurve *fuzzyInput1, *fuzzyInput2;
    CvFuzzyCurve *fuzzyOutput;
public:
    CvFuzzyRule();
    ~CvFuzzyRule();
    void setRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1);
    double calcValue(double param1, double param2);
    CvFuzzyCurve *getOutputCurve();
};

class CV_EXPORTS CvFuzzyController {
private:
    std::vector<CvFuzzyRule*> rules;
public:
    CvFuzzyController();
    ~CvFuzzyController();
    void addRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1);
    double calcOutput(double param1, double param2);
};

class CV_EXPORTS CvFuzzyMeanShiftTracker
{
private:
    class FuzzyResizer
    {
    private:
        CvFuzzyFunction iInput, iOutput;
        CvFuzzyController fuzzyController;
    public:
        FuzzyResizer();
        int calcOutput(double edgeDensity, double density);
    };
    
    class SearchWindow
    {
    public:
        FuzzyResizer *fuzzyResizer;
        int x, y;
        int width, height, maxWidth, maxHeight, ellipseHeight, ellipseWidth;
        int ldx, ldy, ldw, ldh, numShifts, numIters;
        int xGc, yGc;
        long m00, m01, m10, m11, m02, m20;
        double ellipseAngle;
        double density;
        unsigned int depthLow, depthHigh;
        int verticalEdgeLeft, verticalEdgeRight, horizontalEdgeTop, horizontalEdgeBottom;
        
        SearchWindow();
        ~SearchWindow();
        void setSize(int _x, int _y, int _width, int _height);
        void initDepthValues(IplImage *maskImage, IplImage *depthMap);
        bool shift();
        void extractInfo(IplImage *maskImage, IplImage *depthMap, bool initDepth);
        void getResizeAttribsEdgeDensityLinear(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        void getResizeAttribsInnerDensity(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        void getResizeAttribsEdgeDensityFuzzy(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh);
        bool meanShift(IplImage *maskImage, IplImage *depthMap, int maxIteration, bool initDepth);
    };
    
public:
    enum TrackingState
    {
        tsNone          = 0,
        tsSearching     = 1,
        tsTracking      = 2,
        tsSetWindow     = 3,
        tsDisabled      = 10
    };
    
    enum ResizeMethod {
        rmEdgeDensityLinear     = 0,
        rmEdgeDensityFuzzy      = 1,
        rmInnerDensity          = 2
    };
    
    enum {
        MinKernelMass           = 1000
    };
    
    SearchWindow kernel;
    int searchMode;
    
private:
    enum
    {
        MaxMeanShiftIteration   = 5,
        MaxSetSizeIteration     = 5
    };
    
    void findOptimumSearchWindow(SearchWindow &searchWindow, IplImage *maskImage, IplImage *depthMap, int maxIteration, int resizeMethod, bool initDepth);
    
public:
    CvFuzzyMeanShiftTracker();
    ~CvFuzzyMeanShiftTracker();
    
    void track(IplImage *maskImage, IplImage *depthMap, int resizeMethod, bool resetSearch, int minKernelMass = MinKernelMass);
};


namespace cv
{
    
    class CV_EXPORTS Octree
    {
    public:
        struct Node
        {
            Node() {}
            int begin, end;
            float x_min, x_max, y_min, y_max, z_min, z_max;
            int maxLevels;
            bool isLeaf;
            int children[8];
        };
        
        Octree();
        Octree( const vector<Point3f>& points, int maxLevels = 10, int minPoints = 20 );
        virtual ~Octree();
        
        virtual void buildTree( const vector<Point3f>& points, int maxLevels = 10, int minPoints = 20 );
        virtual void getPointsWithinSphere( const Point3f& center, float radius,
                                           vector<Point3f>& points ) const;
        const vector<Node>& getNodes() const { return nodes; }
    private:
        int minPoints;
        vector<Point3f> points;
        vector<Node> nodes;
        
        virtual void buildNext(size_t node_ind);
    };
    
    
    class CV_EXPORTS Mesh3D
    {
    public:
        struct EmptyMeshException {};
        
        Mesh3D();
        Mesh3D(const vector<Point3f>& vtx);
        ~Mesh3D();
        
        void buildOctree();
        void clearOctree();
        float estimateResolution(float tryRatio = 0.1f);
        void computeNormals(float normalRadius, int minNeighbors = 20);
        void computeNormals(const vector<int>& subset, float normalRadius, int minNeighbors = 20);
        
        void writeAsVrml(const String& file, const vector<Scalar>& colors = vector<Scalar>()) const;
        
        vector<Point3f> vtx;
        vector<Point3f> normals;
        float resolution;
        Octree octree;
        
        const static Point3f allzero;
    };
    
    class CV_EXPORTS SpinImageModel
    {
    public:
        
        /* model parameters, leave unset for default or auto estimate */
        float normalRadius;
        int minNeighbors;
        
        float binSize;
        int imageWidth;
        
        float lambda;
        float gamma;
        
        float T_GeometriccConsistency;
        float T_GroupingCorespondances;
        
        /* public interface */
        SpinImageModel();
        explicit SpinImageModel(const Mesh3D& mesh);
        ~SpinImageModel();
        
        void setLogger(std::ostream* log);
        void selectRandomSubset(float ratio);
        void setSubset(const vector<int>& subset);
        void compute();
        
        void match(const SpinImageModel& scene, vector< vector<Vec2i> >& result);
        
        Mat packRandomScaledSpins(bool separateScale = false, size_t xCount = 10, size_t yCount = 10) const;
        
        size_t getSpinCount() const { return spinImages.rows; }
        Mat getSpinImage(size_t index) const { return spinImages.row((int)index); }
        const Point3f& getSpinVertex(size_t index) const { return mesh.vtx[subset[index]]; }
        const Point3f& getSpinNormal(size_t index) const { return mesh.normals[subset[index]]; }
        
        const Mesh3D& getMesh() const { return mesh; }
        Mesh3D& getMesh() { return mesh; }
        
        /* static utility functions */
        static bool spinCorrelation(const Mat& spin1, const Mat& spin2, float lambda, float& result);
        
        static Point2f calcSpinMapCoo(const Point3f& point, const Point3f& vertex, const Point3f& normal);
        
        static float geometricConsistency(const Point3f& pointScene1, const Point3f& normalScene1,
                                          const Point3f& pointModel1, const Point3f& normalModel1,
                                          const Point3f& pointScene2, const Point3f& normalScene2,
                                          const Point3f& pointModel2, const Point3f& normalModel2);
        
        static float groupingCreteria(const Point3f& pointScene1, const Point3f& normalScene1,
                                      const Point3f& pointModel1, const Point3f& normalModel1,
                                      const Point3f& pointScene2, const Point3f& normalScene2,
                                      const Point3f& pointModel2, const Point3f& normalModel2,
                                      float gamma);
    protected:
        void defaultParams();
        
        void matchSpinToModel(const Mat& spin, vector<int>& indeces,
                              vector<float>& corrCoeffs, bool useExtremeOutliers = true) const;
        
        void repackSpinImages(const vector<uchar>& mask, Mat& spinImages, bool reAlloc = true) const;
        
        vector<int> subset;
        Mesh3D mesh;
        Mat spinImages;
        std::ostream* out;
    };
    
    class CV_EXPORTS TickMeter
    {
    public:
        TickMeter();
        void start();
        void stop();
        
        int64 getTimeTicks() const;
        double getTimeMicro() const;
        double getTimeMilli() const;
        double getTimeSec()   const;
        int64 getCounter() const;
        
        void reset();
    private:
        int64 counter;
        int64 sumTime;
        int64 startTime;
    };
    
    CV_EXPORTS std::ostream& operator<<(std::ostream& out, const TickMeter& tm);
    
    class CV_EXPORTS SelfSimDescriptor
    {
    public:
        SelfSimDescriptor();
        SelfSimDescriptor(int _ssize, int _lsize,
                          int _startDistanceBucket=DEFAULT_START_DISTANCE_BUCKET,
                          int _numberOfDistanceBuckets=DEFAULT_NUM_DISTANCE_BUCKETS,
                          int _nangles=DEFAULT_NUM_ANGLES);
        SelfSimDescriptor(const SelfSimDescriptor& ss);
        virtual ~SelfSimDescriptor();
        SelfSimDescriptor& operator = (const SelfSimDescriptor& ss);
        
        size_t getDescriptorSize() const;
        Size getGridSize( Size imgsize, Size winStride ) const;
        
        virtual void compute(const Mat& img, vector<float>& descriptors, Size winStride=Size(),
                             const vector<Point>& locations=vector<Point>()) const;
        virtual void computeLogPolarMapping(Mat& mappingMask) const;
        virtual void SSD(const Mat& img, Point pt, Mat& ssd) const;
        
        int smallSize;
        int largeSize;
        int startDistanceBucket;
        int numberOfDistanceBuckets;
        int numberOfAngles;
        
        enum { DEFAULT_SMALL_SIZE = 5, DEFAULT_LARGE_SIZE = 41,
            DEFAULT_NUM_ANGLES = 20, DEFAULT_START_DISTANCE_BUCKET = 3,
            DEFAULT_NUM_DISTANCE_BUCKETS = 7 };
    };
    
    
    typedef bool (*BundleAdjustCallback)(int iteration, double norm_error, void* user_data);
    
    class LevMarqSparse {
    public:
        LevMarqSparse();
        LevMarqSparse(int npoints, // number of points
                      int ncameras, // number of cameras
                      int nPointParams, // number of params per one point  (3 in case of 3D points)
                      int nCameraParams, // number of parameters per one camera
                      int nErrParams, // number of parameters in measurement vector
                      // for 1 point at one camera (2 in case of 2D projections)
                      Mat& visibility, // visibility matrix. rows correspond to points, columns correspond to cameras
                      // 1 - point is visible for the camera, 0 - invisible
                      Mat& P0, // starting vector of parameters, first cameras then points
                      Mat& X, // measurements, in order of visibility. non visible cases are skipped 
                      TermCriteria criteria, // termination criteria
                      
                      // callback for estimation of Jacobian matrices
                      void (CV_CDECL * fjac)(int i, int j, Mat& point_params,
                                             Mat& cam_params, Mat& A, Mat& B, void* data),
                      // callback for estimation of backprojection errors
                      void (CV_CDECL * func)(int i, int j, Mat& point_params,
                                             Mat& cam_params, Mat& estim, void* data),
                      void* data, // user-specific data passed to the callbacks
                      BundleAdjustCallback cb, void* user_data
                      );
        
        virtual ~LevMarqSparse();
        
        virtual void run( int npoints, // number of points
                         int ncameras, // number of cameras
                         int nPointParams, // number of params per one point  (3 in case of 3D points)
                         int nCameraParams, // number of parameters per one camera
                         int nErrParams, // number of parameters in measurement vector
                         // for 1 point at one camera (2 in case of 2D projections)
                         Mat& visibility, // visibility matrix. rows correspond to points, columns correspond to cameras
                         // 1 - point is visible for the camera, 0 - invisible
                         Mat& P0, // starting vector of parameters, first cameras then points
                         Mat& X, // measurements, in order of visibility. non visible cases are skipped 
                         TermCriteria criteria, // termination criteria
                         
                         // callback for estimation of Jacobian matrices
                         void (CV_CDECL * fjac)(int i, int j, Mat& point_params,
                                                Mat& cam_params, Mat& A, Mat& B, void* data),
                         // callback for estimation of backprojection errors
                         void (CV_CDECL * func)(int i, int j, Mat& point_params,
                                                Mat& cam_params, Mat& estim, void* data),
                         void* data // user-specific data passed to the callbacks
                         );
        
        virtual void clear();
        
        // useful function to do simple bundle adjustment tasks
        static void bundleAdjust(vector<Point3d>& points, // positions of points in global coordinate system (input and output)
                                 const vector<vector<Point2d> >& imagePoints, // projections of 3d points for every camera
                                 const vector<vector<int> >& visibility, // visibility of 3d points for every camera 
                                 vector<Mat>& cameraMatrix, // intrinsic matrices of all cameras (input and output)
                                 vector<Mat>& R, // rotation matrices of all cameras (input and output)
                                 vector<Mat>& T, // translation vector of all cameras (input and output)
                                 vector<Mat>& distCoeffs, // distortion coefficients of all cameras (input and output)
                                 const TermCriteria& criteria=
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON),
                                 BundleAdjustCallback cb = 0, void* user_data = 0);
        
    public:
        virtual void optimize(CvMat &_vis); //main function that runs minimization
        
        //iteratively asks for measurement for visible camera-point pairs
        void ask_for_proj(CvMat &_vis,bool once=false);
        //iteratively asks for Jacobians for every camera_point pair
        void ask_for_projac(CvMat &_vis);
        
        CvMat* err; //error X-hX
        double prevErrNorm, errNorm;
        double lambda;
        CvTermCriteria criteria;
        int iters;
        
        CvMat** U; //size of array is equal to number of cameras
        CvMat** V; //size of array is equal to number of points
        CvMat** inv_V_star; //inverse of V*
        
        CvMat** A;
        CvMat** B;
        CvMat** W;
        
        CvMat* X; //measurement 
        CvMat* hX; //current measurement extimation given new parameter vector 
        
        CvMat* prevP; //current already accepted parameter. 
        CvMat* P; // parameters used to evaluate function with new params
        // this parameters may be rejected 
        
        CvMat* deltaP; //computed increase of parameters (result of normal system solution )
        
        CvMat** ea; // sum_i  AijT * e_ij , used as right part of normal equation
        // length of array is j = number of cameras  
        CvMat** eb; // sum_j  BijT * e_ij , used as right part of normal equation
        // length of array is i = number of points
        
        CvMat** Yj; //length of array is i = num_points
        
        CvMat* S; //big matrix of block Sjk  , each block has size num_cam_params x num_cam_params 
        
        CvMat* JtJ_diag; //diagonal of JtJ,  used to backup diagonal elements before augmentation
        
        CvMat* Vis_index; // matrix which element is index of measurement for point i and camera j
        
        int num_cams;
        int num_points;
        int num_err_param;
        int num_cam_param;
        int num_point_param;
        
        //target function and jacobian pointers, which needs to be initialized 
        void (*fjac)(int i, int j, Mat& point_params, Mat& cam_params, Mat& A, Mat& B, void* data);
        void (*func)(int i, int j, Mat& point_params, Mat& cam_params, Mat& estim, void* data);
        
        void* data;
        
        BundleAdjustCallback cb;
        void* user_data;
    };   
    
    CV_EXPORTS int chamerMatching( Mat& img, Mat& templ,
                                  vector<vector<Point> >& results, vector<float>& cost,
                                  double templScale=1, int maxMatches = 20,
                                  double minMatchDistance = 1.0, int padX = 3,
                                  int padY = 3, int scales = 5, double minScale = 0.6, double maxScale = 1.6,
                                  double orientationWeight = 0.5, double truncate = 20);
    
    
    class CV_EXPORTS StereoVar
    {
    public:
        // Flags	
        enum {USE_INITIAL_DISPARITY = 1, USE_EQUALIZE_HIST = 2, USE_SMART_ID = 4, USE_AUTO_PARAMS = 8, USE_MEDIAN_FILTERING = 16};
        enum {CYCLE_O, CYCLE_V};
        enum {PENALIZATION_TICHONOV, PENALIZATION_CHARBONNIER, PENALIZATION_PERONA_MALIK};
        
        //! the default constructor
        CV_WRAP StereoVar();
        
        //! the full constructor taking all the necessary algorithm parameters
        CV_WRAP StereoVar(int levels, double pyrScale, int nIt, int minDisp, int maxDisp, int poly_n, double poly_sigma, float fi, float lambda, int penalization, int cycle, int flags);
        
        //! the destructor
        virtual ~StereoVar();
        
        //! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
        CV_WRAP_AS(compute) virtual void operator()(const Mat& left, const Mat& right, Mat& disp);
        
        CV_PROP_RW int		levels;
        CV_PROP_RW double	pyrScale;
        CV_PROP_RW int		nIt;
        CV_PROP_RW int		minDisp;
        CV_PROP_RW int		maxDisp;
        CV_PROP_RW int		poly_n;
        CV_PROP_RW double	poly_sigma;
        CV_PROP_RW float	fi;
        CV_PROP_RW float	lambda;
        CV_PROP_RW int		penalization;
        CV_PROP_RW int		cycle;
        CV_PROP_RW int		flags;
        
    private:
        void autoParams();
		void FMG(Mat &I1, Mat &I2, Mat &I2x, Mat &u, int level);
        void VCycle_MyFAS(Mat &I1_h, Mat &I2_h, Mat &I2x_h, Mat &u_h, int level);
        void VariationalSolver(Mat &I1_h, Mat &I2_h, Mat &I2x_h, Mat &u_h, int level);
    };
    
    CV_EXPORTS void polyfit(const Mat& srcx, const Mat& srcy, Mat& dst, int order);

	class CV_EXPORTS Directory
    {
	    public:
			static std::vector<std::string> GetListFiles  ( const std::string& path, const std::string & exten = "*", bool addPath = true );
			static std::vector<std::string> GetListFilesR ( const std::string& path, const std::string & exten = "*", bool addPath = true );
			static std::vector<std::string> GetListFolders( const std::string& path, const std::string & exten = "*", bool addPath = true );
    };

    /*
     * Generation of a set of different colors by the following way:
     * 1) generate more then need colors (in "factor" times) in RGB,
     * 2) convert them to Lab,
     * 3) choose the needed count of colors from the set that are more different from
     *    each other,
     * 4) convert the colors back to RGB
     */
    CV_EXPORTS void generateColors( std::vector<Scalar>& colors, size_t count, size_t factor=100 );


    /*
     *  Estimate the rigid body motion from frame0 to frame1. The method is based on the paper
     *  "Real-Time Visual Odometry from Dense RGB-D Images", F. Steinbucker, J. Strum, D. Cremers, ICCV, 2011.
     */
    enum { ROTATION          = 1,
           TRANSLATION       = 2,
           RIGID_BODY_MOTION = 4
         };
    CV_EXPORTS bool RGBDOdometry( Mat& Rt, const Mat& initRt,
                                  const Mat& image0, const Mat& depth0, const Mat& mask0,
                                  const Mat& image1, const Mat& depth1, const Mat& mask1,
                                  const Mat& cameraMatrix, float minDepth=0.f, float maxDepth=4.f, float maxDepthDiff=0.07f,
                                  const std::vector<int>& iterCounts=std::vector<int>(),
                                  const std::vector<float>& minGradientMagnitudes=std::vector<float>(),
                                  int transformType=RIGID_BODY_MOTION );

    /**
    *Bilinear interpolation technique.
    *
    *The value of a desired cortical pixel is obtained through a bilinear interpolation of the values
    *of the four nearest neighbouring Cartesian pixels to the center of the RF.
    *The same principle is applied to the inverse transformation.
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    class CV_EXPORTS LogPolar_Interp
    {
    public:
        
        LogPolar_Interp() {}

        /**
        *Constructor
        *\param w the width of the input image
        *\param h the height of the input image
        *\param center the transformation center: where the output precision is maximal
        *\param R the number of rings of the cortical image (default value 70 pixel)
        *\param ro0 the radius of the blind spot (default value 3 pixel)
        *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle. 
        *            \a 0 means that the retinal image is computed within the inscribed circle.
        *\param S the number of sectors of the cortical image (default value 70 pixel).
        *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
        *\param sp \a 1 (default value) means that the parameter \a S is internally computed. 
        *          \a 0 means that the parameter \a S is provided by the user.
        */
        LogPolar_Interp(int w, int h, Point2i center, int R=70, double ro0=3.0,
                        int interp=INTER_LINEAR, int full=1, int S=117, int sp=1);
        /**
        *Transformation from Cartesian image to cortical (log-polar) image.
        *\param source the Cartesian image
        *\return the transformed image (cortical image)
        */
        const Mat to_cortical(const Mat &source);
        /**
        *Transformation from cortical image to retinal (inverse log-polar) image.
        *\param source the cortical image
        *\return the transformed image (retinal image)
        */
        const Mat to_cartesian(const Mat &source);
        /**
        *Destructor
        */
        ~LogPolar_Interp();
    
    protected:
        
        Mat Rsri;
        Mat Csri;

        int S, R, M, N;
        int top, bottom,left,right;
        double ro0, romax, a, q;
        int interp;

        Mat ETAyx;
        Mat CSIyx;

        void create_map(int M, int N, int R, int S, double ro0);
    };

    /**
    *Overlapping circular receptive fields technique
    *
    *The Cartesian plane is divided in two regions: the fovea and the periphery.
    *The fovea (oversampling) is handled by using the bilinear interpolation technique described above, whereas in
    *the periphery we use the overlapping Gaussian circular RFs.
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    class CV_EXPORTS LogPolar_Overlapping
    {        
    public:
        LogPolar_Overlapping() {}
        
        /**
        *Constructor
        *\param w the width of the input image
        *\param h the height of the input image
        *\param center the transformation center: where the output precision is maximal
        *\param R the number of rings of the cortical image (default value 70 pixel)
        *\param ro0 the radius of the blind spot (default value 3 pixel)
        *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle. 
        *            \a 0 means that the retinal image is computed within the inscribed circle.
        *\param S the number of sectors of the cortical image (default value 70 pixel).
        *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
        *\param sp \a 1 (default value) means that the parameter \a S is internally computed. 
        *          \a 0 means that the parameter \a S is provided by the user.
        */
        LogPolar_Overlapping(int w, int h, Point2i center, int R=70,
                             double ro0=3.0, int full=1, int S=117, int sp=1);
        /**
        *Transformation from Cartesian image to cortical (log-polar) image.
        *\param source the Cartesian image
        *\return the transformed image (cortical image)
        */
        const Mat to_cortical(const Mat &source);
        /**
        *Transformation from cortical image to retinal (inverse log-polar) image.
        *\param source the cortical image
        *\return the transformed image (retinal image)
        */
        const Mat to_cartesian(const Mat &source);
        /**
        *Destructor
        */
        ~LogPolar_Overlapping();
        
    protected:
        
        Mat Rsri;
        Mat Csri;
        vector<int> Rsr;
        vector<int> Csr;
        vector<double> Wsr;

        int S, R, M, N, ind1;
        int top, bottom,left,right;
        double ro0, romax, a, q;

        struct kernel
        {
            kernel() { w = 0; }
            vector<double> weights;
            int w;
        };

        Mat ETAyx;
        Mat CSIyx;
        vector<kernel> w_ker_2D;

        void create_map(int M, int N, int R, int S, double ro0);
    };

    /**
    * Adjacent receptive fields technique
    *
    *All the Cartesian pixels, whose coordinates in the cortical domain share the same integer part, are assigned to the same RF.
    *The precision of the boundaries of the RF can be improved by breaking each pixel into subpixels and assigning each of them to the correct RF.
    *This technique is implemented from: Traver, V., Pla, F.: Log-polar mapping template design: From task-level requirements
    *to geometry parameters. Image Vision Comput. 26(10) (2008) 1354-1370
    *
    *More details can be found in http://dx.doi.org/10.1007/978-3-642-23968-7_5
    */
    class CV_EXPORTS LogPolar_Adjacent
    {
    public:
        LogPolar_Adjacent() {}
        
        /**
         *Constructor
         *\param w the width of the input image
         *\param h the height of the input image
         *\param center the transformation center: where the output precision is maximal
         *\param R the number of rings of the cortical image (default value 70 pixel)
         *\param ro0 the radius of the blind spot (default value 3 pixel)
         *\param smin the size of the subpixel (default value 0.25 pixel)
         *\param full \a 1 (default value) means that the retinal image (the inverse transform) is computed within the circumscribing circle. 
         *            \a 0 means that the retinal image is computed within the inscribed circle.
         *\param S the number of sectors of the cortical image (default value 70 pixel).
         *         Its value is usually internally computed to obtain a pixel aspect ratio equals to 1.
         *\param sp \a 1 (default value) means that the parameter \a S is internally computed. 
         *          \a 0 means that the parameter \a S is provided by the user.
         */  
        LogPolar_Adjacent(int w, int h, Point2i center, int R=70, double ro0=3.0, double smin=0.25, int full=1, int S=117, int sp=1);
        /**
         *Transformation from Cartesian image to cortical (log-polar) image.
         *\param source the Cartesian image
         *\return the transformed image (cortical image)
         */
        const Mat to_cortical(const Mat &source);
        /**
         *Transformation from cortical image to retinal (inverse log-polar) image.
         *\param source the cortical image
         *\return the transformed image (retinal image)
         */
        const Mat to_cartesian(const Mat &source);
        /**
         *Destructor
         */
        ~LogPolar_Adjacent();

    protected:
        struct pixel
        {
            pixel() { u = v = 0; a = 0.; }
            int u;
            int v;
            double a;
        };
        int S, R, M, N;
        int top, bottom,left,right;
        double ro0, romax, a, q;
        vector<vector<pixel> > L;
        vector<double> A;

        void subdivide_recursively(double x, double y, int i, int j, double length, double smin);
        bool get_uv(double x, double y, int&u, int&v);
        void create_map(int M, int N, int R, int S, double ro0, double smin);
    };
    
    CV_EXPORTS Mat subspaceProject(InputArray W, InputArray mean, InputArray src);
    CV_EXPORTS Mat subspaceReconstruct(InputArray W, InputArray mean, InputArray src);
    
    class CV_EXPORTS LDA
    {
    public:
        // Initializes a LDA with num_components (default 0) and specifies how
        // samples are aligned (default dataAsRow=true).
        LDA(int num_components = 0) :
            _num_components(num_components) {};

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        LDA(const Mat& src, vector<int> labels,
                int num_components = 0) :
                    _num_components(num_components)
        {
            this->compute(src, labels); //! compute eigenvectors and eigenvalues
        }

        // Initializes and performs a Discriminant Analysis with Fisher's
        // Optimization Criterion on given data in src and corresponding labels
        // in labels. If 0 (or less) number of components are given, they are
        // automatically determined for given data in computation.
        LDA(InputArray src, InputArray labels,
                int num_components = 0) :
                    _num_components(num_components)
        {
            this->compute(src, labels); //! compute eigenvectors and eigenvalues
        }

        // Serializes this object to a given filename.
        void save(const string& filename) const;

        // Deserializes this object from a given filename.
        void load(const string& filename);

        // Serializes this object to a given cv::FileStorage.
        void save(FileStorage& fs) const;

            // Deserializes this object from a given cv::FileStorage.
        void load(const FileStorage& node);

        // Destructor.
        ~LDA() {}

        //! Compute the discriminants for data in src and labels.
        void compute(InputArray src, InputArray labels);

        // Projects samples into the LDA subspace.
        Mat project(InputArray src);

        // Reconstructs projections from the LDA subspace.
        Mat reconstruct(InputArray src);

        // Returns the eigenvectors of this LDA.
        Mat eigenvectors() const { return _eigenvectors; };

        // Returns the eigenvalues of this LDA.
        Mat eigenvalues() const { return _eigenvalues; }
        
    protected:
        bool _dataAsRow;
        int _num_components;
        Mat _eigenvectors;
        Mat _eigenvalues;

        void lda(InputArray src, InputArray labels);
    };
    
    class CV_EXPORTS FaceRecognizer
    {
    public:
        //! virtual destructor
        virtual ~FaceRecognizer() {}

        // Trains a FaceRecognizer.
        virtual void train(InputArray src, InputArray labels) = 0;

        // Gets a prediction from a FaceRecognizer.
        virtual int predict(InputArray src) const = 0;

        // Serializes this object to a given filename.
        virtual void save(const string& filename) const;

        // Deserializes this object from a given filename.
        virtual void load(const string& filename);

        // Serializes this object to a given cv::FileStorage.
        virtual void save(FileStorage& fs) const = 0;

        // Deserializes this object from a given cv::FileStorage.
        virtual void load(const FileStorage& fs) = 0;
        
        // Returns eigenvectors (if any)
        virtual Mat eigenvectors() const { return Mat(); }
    };
    
    CV_EXPORTS Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0);
    CV_EXPORTS Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0);
    CV_EXPORTS Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8,
                                                            int grid_x=8, int grid_y=8);
    
    enum
    {
        COLORMAP_AUTUMN = 0,
        COLORMAP_BONE = 1,
        COLORMAP_JET = 2,
        COLORMAP_WINTER = 3,
        COLORMAP_RAINBOW = 4,
        COLORMAP_OCEAN = 5,
        COLORMAP_SUMMER = 6,
        COLORMAP_SPRING = 7,
        COLORMAP_COOL = 8,
        COLORMAP_HSV = 9,
        COLORMAP_PINK = 10,
        COLORMAP_HOT = 11,
        COLORMAP_MKPJ1 = 12,
        COLORMAP_MKPJ2 = 13
    };
    
    CV_EXPORTS void applyColorMap(InputArray src, OutputArray dst, int colormap);
    
    CV_EXPORTS bool initModule_contrib();
}


#include "opencv2/contrib/retina.hpp"

#endif

#endif

