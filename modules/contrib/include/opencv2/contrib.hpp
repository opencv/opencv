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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/objdetect.hpp"

namespace cv
{
class CV_EXPORTS Octree
{
public:
    struct Node
    {
        Node() { memset(this, 0, sizeof(Node)); }
        int begin, end;
        float x_min, x_max, y_min, y_max, z_min, z_max;
        int maxLevels;
        bool isLeaf;
        int children[8];
    };

    Octree();
    Octree( const std::vector<Point3f>& points, int maxLevels = 10, int minPoints = 20 );
    virtual ~Octree();

    virtual void buildTree( const std::vector<Point3f>& points, int maxLevels = 10, int minPoints = 20 );
    virtual void getPointsWithinSphere( const Point3f& center, float radius,
                                       std::vector<Point3f>& points ) const;
    const std::vector<Node>& getNodes() const { return nodes; }
private:
    int minPoints;
    std::vector<Point3f> points;
    std::vector<Node> nodes;

    virtual void buildNext(size_t node_ind);
};


class CV_EXPORTS Mesh3D
{
public:
    struct EmptyMeshException {};

    Mesh3D();
    Mesh3D(const std::vector<Point3f>& vtx);
    ~Mesh3D();

    void buildOctree();
    void clearOctree();
    float estimateResolution(float tryRatio = 0.1f);
    void computeNormals(float normalRadius, int minNeighbors = 20);
    void computeNormals(const std::vector<int>& subset, float normalRadius, int minNeighbors = 20);

    void writeAsVrml(const String& file, const std::vector<Scalar>& colors = std::vector<Scalar>()) const;

    std::vector<Point3f> vtx;
    std::vector<Point3f> normals;
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

    void selectRandomSubset(float ratio);
    void setSubset(const std::vector<int>& subset);
    void compute();

    void match(const SpinImageModel& scene, std::vector< std::vector<Vec2i> >& result);

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

    void matchSpinToModel(const Mat& spin, std::vector<int>& indeces,
                          std::vector<float>& corrCoeffs, bool useExtremeOutliers = true) const;

    void repackSpinImages(const std::vector<uchar>& mask, Mat& spinImages, bool reAlloc = true) const;

    std::vector<int> subset;
    Mesh3D mesh;
    Mat spinImages;
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

//CV_EXPORTS std::ostream& operator<<(std::ostream& out, const TickMeter& tm);

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

    virtual void compute(const Mat& img, std::vector<float>& descriptors, Size winStride=Size(),
                         const std::vector<Point>& locations=std::vector<Point>()) const;
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


CV_EXPORTS_W int chamerMatching( Mat& img, Mat& templ,
                              CV_OUT std::vector<std::vector<Point> >& results, CV_OUT std::vector<float>& cost,
                              double templScale=1, int maxMatches = 20,
                              double minMatchDistance = 1.0, int padX = 3,
                              int padY = 3, int scales = 5, double minScale = 0.6, double maxScale = 1.6,
                              double orientationWeight = 0.5, double truncate = 20);


class CV_EXPORTS_W StereoVar
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
    CV_WRAP_AS(compute) virtual void operator()(const Mat& left, const Mat& right, CV_OUT Mat& disp);

    CV_PROP_RW int      levels;
    CV_PROP_RW double   pyrScale;
    CV_PROP_RW int      nIt;
    CV_PROP_RW int      minDisp;
    CV_PROP_RW int      maxDisp;
    CV_PROP_RW int      poly_n;
    CV_PROP_RW double   poly_sigma;
    CV_PROP_RW float    fi;
    CV_PROP_RW float    lambda;
    CV_PROP_RW int      penalization;
    CV_PROP_RW int      cycle;
    CV_PROP_RW int      flags;

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
        static std::vector<String> GetListFiles  ( const String& path, const String & exten = "*", bool addPath = true );
        static std::vector<String> GetListFilesR ( const String& path, const String & exten = "*", bool addPath = true );
        static std::vector<String> GetListFolders( const String& path, const String & exten = "*", bool addPath = true );
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
    std::vector<int> Rsr;
    std::vector<int> Csr;
    std::vector<double> Wsr;

    int S, R, M, N, ind1;
    int top, bottom,left,right;
    double ro0, romax, a, q;

    struct kernel
    {
        kernel() { w = 0; }
        std::vector<double> weights;
        int w;
    };

    Mat ETAyx;
    Mat CSIyx;
    std::vector<kernel> w_ker_2D;

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
    std::vector<std::vector<pixel> > L;
    std::vector<double> A;

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
        _num_components(num_components) { }

    // Initializes and performs a Discriminant Analysis with Fisher's
    // Optimization Criterion on given data in src and corresponding labels
    // in labels. If 0 (or less) number of components are given, they are
    // automatically determined for given data in computation.
    LDA(InputArrayOfArrays src, InputArray labels,
            int num_components = 0) :
                _num_components(num_components)
    {
        this->compute(src, labels); //! compute eigenvectors and eigenvalues
    }

    // Serializes this object to a given filename.
    void save(const String& filename) const;

    // Deserializes this object from a given filename.
    void load(const String& filename);

    // Serializes this object to a given cv::FileStorage.
    void save(FileStorage& fs) const;

        // Deserializes this object from a given cv::FileStorage.
    void load(const FileStorage& node);

    // Destructor.
    ~LDA() {}

    //! Compute the discriminants for data in src and labels.
    void compute(InputArrayOfArrays src, InputArray labels);

    // Projects samples into the LDA subspace.
    Mat project(InputArray src);

    // Reconstructs projections from the LDA subspace.
    Mat reconstruct(InputArray src);

    // Returns the eigenvectors of this LDA.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this LDA.
    Mat eigenvalues() const { return _eigenvalues; }

protected:
    bool _dataAsRow;
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;

    void lda(InputArrayOfArrays src, InputArray labels);
};

class CV_EXPORTS_W FaceRecognizer : public Algorithm
{
public:
    //! virtual destructor
    virtual ~FaceRecognizer() {}

    // Trains a FaceRecognizer.
    CV_WRAP virtual void train(InputArrayOfArrays src, InputArray labels) = 0;

    // Updates a FaceRecognizer.
    CV_WRAP virtual void update(InputArrayOfArrays src, InputArray labels);

    // Gets a prediction from a FaceRecognizer.
    virtual int predict(InputArray src) const = 0;

    // Predicts the label and confidence for a given sample.
    CV_WRAP virtual void predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const = 0;

    // Serializes this object to a given filename.
    CV_WRAP virtual void save(const String& filename) const;

    // Deserializes this object from a given filename.
    CV_WRAP virtual void load(const String& filename);

    // Serializes this object to a given cv::FileStorage.
    virtual void save(FileStorage& fs) const = 0;

    // Deserializes this object from a given cv::FileStorage.
    virtual void load(const FileStorage& fs) = 0;

};

CV_EXPORTS_W Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
CV_EXPORTS_W Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
CV_EXPORTS_W Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8,
                                                        int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

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
    COLORMAP_HOT = 11
};

CV_EXPORTS_W void applyColorMap(InputArray src, OutputArray dst, int colormap);

CV_EXPORTS bool initModule_contrib();
}

#include "opencv2/contrib/openfabmap.hpp"

#endif
