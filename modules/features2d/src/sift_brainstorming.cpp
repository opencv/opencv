#include <mat.hpp>
#include "sift_cuda.cu"
#include "stdint.h"
#include "unistd.h"
#include <vector>


// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 4.5f; // 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#define USE_CUDA 1
#define DoG_TYPE_SHORT 0
#if DoG_TYPE_SHORT
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif


// Step 4
class SIFT_Range
{
public:
    SIFT_Range();
    SIFT_Range(int _rstart, int _cstart, int _nrow, int _ncol);
    int row0, col0;
    int num_row, num_col;
};


// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static
bool adjustLocalExtremaCuda(const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
                        int& layer, int& r, int& c, int nOctaveLayers,
                        float contrastThreshold, float edgeThreshold, float sigma)
{

    // declare unit conversion factors
    // DoG values are stored as integers (scaled by SIFT_FIXPT_SCALE to avoid floating point during construction
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0;     // sub-pixel offset in the scale (layer) direction
    float xr=0;     // sub-pixel offset in the row direction
    float xc=0;     //  sub-pixel offset in the column direction
    float contr=0;  // contrast value at the refined point
    int i = 0;      //  iteration counter

    // refinement loop: each iteration refetches the three layers becaue layer may have changed from prev iteration
    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
            return false;

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}

class findScaleSpaceExtremaCudaT
{
    public:
        findScaleSpaceExtremaCudaT(
            int _o,                                 // octave #
            int _i,                                 // layer number: [1, nOctaveLayer)
            int _threshold,                         // min abs value to even consider pixel as candidate
            int _idx,                               // precomputed index into dog_ptr for current layer
            int _step,                              // row stride - # elements to skip to get to next row
            int _cols,                              // width of image in octave
            int _nOctaveLayers,
            double _contrastThreshold,
            double _edgeThreshold,
            double _sigma,
            const std::vector<Mat>& _gauss_pyr,     // gaussian pyramid (for orientation calcs)
            const std::vector<Mat>& _dog_pyr,       // dog pyramid (for finding extrema / keypoints)
            std::vector<KeyPoint>& kpts)            // OUTPUT: found keypoints eventually go in here

            : o(_o),
            i(_i),
            threshold(_threshold),
            idx(_idx),
            step(_step),
            cols(_cols),
            nOctaveLayers(_nOctaveLayers),
            contrastThreshold(_contrastThreshold),
            edgeThreshold(_edgeThreshold),
            sigma(_sigma),
            gauss_pyr(_gauss_pyr),
            dog_pyr(_dog_pyr),
            kpts_(kpts)
        {
            // nothing
        }
        void process(const SIFT_Range &range)
        {
            const int rbegin = range.row0;
            const int rend = range.row0 + range.num_row;

            const int cbegin = range.col0;
            const int cend = range.col0 + range.num_col;

            static const int n = SIFT_ORI_HIST_BINS; // orientation histogram
            float hist[n]; // reused buffer for orientation calc

            const Mat& img =  this.dog_pyr[idx];     // current layer i
            const Mat& prev = this.dog_pyr[idx-1];   // current layer i-1
            const Mat& next = this.dog_pyr[idx+1];   // current layer i+1

            for( int r = rbegin; r < rend; r++)
            {
                // Accessing currptr[c] then gives you the pixel at (r, c).
                const sift_wt* currptr = img.ptr<sift_wt>(r); // pointer to row r in curr layer
                const sift_wt* prevptr = prev.ptr<sift_wt>(r); // pointer to row r previous layer
                const sift_wt* nextptr = next.ptr<sift_wt>(r);  // pointer to row r in next layer
                int c = SIFT_IMG_BORDER;
                
                // originally CPU vectorization optimization that processes multiple pixels at once using vector registers. 
                #if (!USE_CUDA) && !(DoG_TYPE_SHORT)
                    const int vecsize = VTraits<v_float32>::vlanes();
                    for( ; c <= cols-SIFT_IMG_BORDER - vecsize; c += vecsize)
                    {
                        v_float32 val = vx_load(&currptr[c]);
                        v_float32 _00,_01,_02;
                        v_float32 _10,    _12;
                        v_float32 _20,_21,_22;

                        v_float32 vmin,vmax;


                        v_float32 cond = v_gt(v_abs(val), vx_setall_f32((float)this->threshold));
                        if (!v_check_any(cond))
                        {
                            continue;
                        }

                        _00 = vx_load(&currptr[c-step-1]); _01 = vx_load(&currptr[c-step]); _02 = vx_load(&currptr[c-step+1]);
                        _10 = vx_load(&currptr[c     -1]);                                  _12 = vx_load(&currptr[c     +1]);
                        _20 = vx_load(&currptr[c+step-1]); _21 = vx_load(&currptr[c+step]); _22 = vx_load(&currptr[c+step+1]);

                        vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                        vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                        v_float32 condp = v_and(v_and(cond, v_gt(val, vx_setall_f32(0))), v_ge(val, vmax));
                        v_float32 condm = v_and(v_and(cond, v_lt(val, vx_setall_f32(0))), v_le(val, vmin));

                        cond = v_or(condp, condm);
                        if (!v_check_any(cond))
                        {
                            continue;
                        }

                        _00 = vx_load(&prevptr[c-step-1]); _01 = vx_load(&prevptr[c-step]); _02 = vx_load(&prevptr[c-step+1]);
                        _10 = vx_load(&prevptr[c     -1]);                                  _12 = vx_load(&prevptr[c     +1]);
                        _20 = vx_load(&prevptr[c+step-1]); _21 = vx_load(&prevptr[c+step]); _22 = vx_load(&prevptr[c+step+1]);

                        vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                        vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                        condp = v_and(condp, v_ge(val, vmax));
                        condm = v_and(condm, v_le(val, vmin));

                        cond = v_or(condp, condm);
                        if (!v_check_any(cond))
                        {
                            continue;
                        }

                        v_float32 _11p = vx_load(&prevptr[c]);
                        v_float32 _11n = vx_load(&nextptr[c]);

                        v_float32 max_middle = v_max(_11n,_11p);
                        v_float32 min_middle = v_min(_11n,_11p);

                        _00 = vx_load(&nextptr[c-step-1]); _01 = vx_load(&nextptr[c-step]); _02 = vx_load(&nextptr[c-step+1]);
                        _10 = vx_load(&nextptr[c     -1]);                                  _12 = vx_load(&nextptr[c     +1]);
                        _20 = vx_load(&nextptr[c+step-1]); _21 = vx_load(&nextptr[c+step]); _22 = vx_load(&nextptr[c+step+1]);

                        vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                        vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                        condp = v_and(condp, v_ge(val, v_max(vmax, max_middle)));
                        condm = v_and(condm, v_le(val, v_min(vmin, min_middle)));

                        cond = v_or(condp, condm);
                        if (!v_check_any(cond))
                        {
                            continue;
                        }

                        int mask = v_signmask(cond);
                        for (int k = 0; k<vecsize;k++)
                        {
                            if ((mask & (1<<k)) == 0)
                                continue;

                            CV_TRACE_REGION("pixel_candidate_simd");

                            KeyPoint kpt;
                            int r1 = r, c1 = c+k, layer = i;
                            if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                                    nOctaveLayers, (float)contrastThreshold,
                                                    (float)edgeThreshold, (float)sigma) )
                                continue;
                            float scl_octv = kpt.size*0.5f/(1 << o);
                            float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                            Point(c1, r1),
                                                            cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                            SIFT_ORI_SIG_FCTR * scl_octv,
                                                            hist, n);
                            float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                            for( int j = 0; j < n; j++ )
                            {
                                int l = j > 0 ? j - 1 : n - 1;
                                int r2 = j < n-1 ? j + 1 : 0;

                                if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                                {
                                    float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                    bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                                    kpt.angle = 360.f - (float)((360.f/n) * bin);
                                    if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                        kpt.angle = 0.f;

                                    kpts_.push_back(kpt);
                                }
                            }
                        }
                    }
                #elif (USE_CUDA) && !(DoG_TYPE_SHORT)

                    // Loading in the 4x8 chunk, plus neighbors (26-Neighbors per pixel) with overlap
                    for( ; c < cols-SIFT_IMG_BORDER; c++)
                    {
                        sift_wt val = currptr[c];
                        if (std::abs(val) <= threshold)
                            continue;

                        // Loading the 8 Current-Layer Neighbors
                        sift_wt _00,_01,_02;
                        sift_wt _10,    _12;
                        sift_wt _20,_21,_22;
                        _00 = currptr[c-step-1]; _01 = currptr[c-step]; _02 = currptr[c-step+1];
                        _10 = currptr[c     -1];                        _12 = currptr[c     +1];
                        _20 = currptr[c+step-1]; _21 = currptr[c+step]; _22 = currptr[c+step+1];
                        
                        // The 26-Neighbor Extremum Check
                        bool calculate = false;
                        if (val > 0)
                        {
                            // find max of same-layer nieghbors
                            sift_wt vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));

                            // compare max of neighbors to current value
                            if (val >= vmax)
                            {   
                                // if so, load previous layer neighbors too
                                _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                                _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                                _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];

                                // find max of prev-layer nieghbors
                                vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                                
                                // compare max of neighbors to current value
                                if (val >= vmax)
                                {
                                    // if so, load next layer neighbors too
                                    _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                                    _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                                    _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];

                                    // find max of next-layer nieghbors
                                    vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                                    
                                    // compare max of neighbors to current value
                                    if (val >= vmax)
                                    {
                                        // if so, compare current pixel to center pixels of prev and next layer
                                        sift_wt _11p = prevptr[c], _11n = nextptr[c];
                                        calculate = (val >= std::max(_11p,_11n));
                                    }
                                }
                            }

                        } else  { // val can't be zero here (first abs took care of zero), must be negative

                            // find min of current-layer neighbors
                            sift_wt vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));

                            // check if pixel is smaller
                            if (val <= vmin)
                            {
                                // if so, load in previous layer neighbors
                                _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                                _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                                _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];

                                // find min of prev-layer neighbors
                                vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));

                                // chekc if pixel is smaller
                                if (val <= vmin)
                                {
                                    // if so, then lead in next layer neighbors
                                    _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                                    _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                                    _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];

                                    // find min of neighbors
                                    vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));

                                    // check if pixel is smaller
                                    if (val <= vmin)
                                    {
                                        // if so, compare pixel to top and down neighbors
                                        sift_wt _11p = prevptr[c], _11n = nextptr[c];

                                        // calculate = True means we found local min
                                        calculate = (val <= std::min(_11p,_11n));
                                    }
                                }
                            }
                        }

                        // calculate == True -> found a local maxima/minima --> add to keypoints of interest
                        if (calculate)
                        {
                            KeyPoint kpt;
                            int r1 = r, c1 = c, layer = i;

                            // Sub-pixel refinement ->  adjusts (r1,c1,layer) to the true extremum location
                            
                            // Returns false if the refined point is too weak or on an edge → skip it
                            if( !adjustLocalExtremaCuda(dog_pyr, kpt, o, layer, r1, c1, nOctaveLayers, (float)contrastThreshold, (float)edgeThreshold, (float)sigma) )
                                continue;
                            
                            /****** END OF RACHEL's PORTION ********/

                            /****** START OF NITHILA's PORTION ********/
                            // refined point is strong, proceed to calculate orientation around keypoint
                            float scl_octv = kpt.size*0.5f/(1 << o);
                            float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer], // gaussian layer (not DoG)
                                                            Point(c1, r1),
                                                            cvRound(SIFT_ORI_RADIUS * scl_octv), // radius of histogram window
                                                            SIFT_ORI_SIG_FCTR * scl_octv, // gaussian weighting sigma
                                                            hist, n);
                            
                            // For each dominant orientation peak, emit a separate keypoint
                            float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);  // 80% of peak height
                            for( int j = 0; j < n; j++ )
                            {
                                int l = j > 0 ? j - 1 : n - 1; // left bin (wraps)
                                int r2 = j < n-1 ? j + 1 : 0; // right bin (wraps)
                                
                                // Is bin j a local peak above the threshold?
                                if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                                {
                                    // Parabolic interpolation for sub-bin precision
                                    float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                    bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin; // wrap to [0,n)

                                    kpt.angle = 360.f - (float)((360.f/n) * bin);  // convert bin to degrees
                                    if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                        kpt.angle = 0.f;

                                    kpts_.push_back(kpt); // ← final output, one entry per orientation
                                }
                            }
                        }
                    }

                #endif 
            }
        }
    private:
        int o, i;
        int threshold;
        int idx, step, cols;
        int nOctaveLayers;
        double contrastThreshold;
        double edgeThreshold;
        double sigma;
        const std::vector<Mat>& gauss_pyr;
        const std::vector<Mat>& dog_pyr;
        std::vector<KeyPoint>& kpts_;
        
    };
} 


// Step 3
void findScaleSpaceExtrema(
    int octave,
    int layer,
    int threshold,
    int idx,
    int step,
    int cols,
    int nOctaveLayers,
    double contrastThreshold,
    double edgeThreshold,
    double sigma,
    const std::vector<Mat>& gauss_pyr,
    const std::vector<Mat>& dog_pyr,
    std::vector<KeyPoint>& kpts,
    const cv::Range& range)
{
    CV_TRACE_FUNCTION();
    findScaleSpaceExtremaT(octave, layer, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, kpts).process(range);
}

// Step 2
class findScaleSpaceExtremaComputer
{
    private:
        int o, i; int threshold; int idx, step, cols; int nOctaveLayers; double contrastThreshold; double edgeThreshold; double sigma; 
        const std::vector<Mat>& gauss_pyr; const std::vector<Mat>& dog_pyr; TLSData<std::vector<KeyPoint> > &tls_kpts_struct;
    public:
        findScaleSpaceExtremaComputer(
            int _o, int _i, int _threshold, int _idx, int _step, int _cols, int _nOctaveLayers, double _contrastThreshold, double _edgeThreshold, 
            double _sigma, const std::vector<Mat>& _gauss_pyr, const std::vector<Mat>& _dog_pyr, TLSData<std::vector<KeyPoint> > &_tls_kpts_struct)

            : o(_o),
            i(_i),
            threshold(_threshold),
            idx(_idx),
            step(_step),
            cols(_cols),
            nOctaveLayers(_nOctaveLayers),
            contrastThreshold(_contrastThreshold),
            edgeThreshold(_edgeThreshold),
            sigma(_sigma),
            gauss_pyr(_gauss_pyr),
            dog_pyr(_dog_pyr),
            tls_kpts_struct(_tls_kpts_struct) { }
        void operator()( const cv::Range& range ) const CV_OVERRIDE
        {
            std::vector<KeyPoint>& kpts = tls_kpts_struct.getRef();
            CV_CPU_DISPATCH(findScaleSpaceExtrema, (o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, kpts, range), CV_CPU_DISPATCH_MODES_ALL);
        }
};

// Step 2
class findScaleSpaceExtrema_CUDA
{
    private:
        int o, i; int threshold; int idx, step, cols; int nOctaveLayers; double contrastThreshold; double edgeThreshold; double sigma; 
        const std::vector<Mat>& gauss_pyr; const std::vector<Mat>& dog_pyr; TLSData<std::vector<KeyPoint> > &tls_kpts_struct;
    public:
        findScaleSpaceExtrema_CUDA(int _o, int _i, int _threshold, int _idx, int _step, int _cols, int _nOctaveLayers, double _contrastThreshold, double _edgeThreshold, 
            double _sigma, const std::vector<Mat>& _gauss_pyr, const std::vector<Mat>& _dog_pyr, TLSData<std::vector<KeyPoint> > &_tls_kpts_struct)
            : o(_o),
            i(_i),
            threshold(_threshold),
            idx(_idx),
            step(_step),
            cols(_cols),
            nOctaveLayers(_nOctaveLayers),
            contrastThreshold(_contrastThreshold),
            edgeThreshold(_edgeThreshold),
            sigma(_sigma),
            gauss_pyr(_gauss_pyr),
            dog_pyr(_dog_pyr),
            tls_kpts_struct(_tls_kpts_struct) { }
        void operator()() const CV_OVERRIDE
        {
            std::vector<KeyPoint>& kpts = tls_kpts_struct.getRef();

             // 1. Allocate on GPU
            float* d_curr, *d_prev, * d_next;
            cudaMalloc(&d_prev, size_of(dog_pyr[idx-1]));
            cudaMalloc(&d_curr, size_of(dog_pyr[idx]));
            cudaMalloc(&d_next, size_of(dog_pyr[idx+1]));

            // MemCopy things over to Device
            cudaMemcpy(dog_pyr[idx-1], d_prev, size_of(dog_pyr[idx-1]), cudaMemcpyHostToDevice);
            cudaMemcpy(dog_pyr[idx], d_curr, size_of(dog_pyr[idx]), cudaMemcpyHostToDevice);
            cudaMemcpy(dog_pyr[idx+1], d_next, size_of(dog_pyr[idx+1]), cudaMemcpyHostToDevice);

            float* c_r, *c_c;
            int * c_count;
            
            cudaMalloc(&c_r, size_of(float) * dog_pyr[idx].size());
            cudaMalloc(&c_c, size_of(float) * dog_pyr[idx].size()); 
            cudaMalloc(&c_count, size_of(float));

            cudaMemcpy(dog_pyr[idx-1], d_prev, size_of(dog_pyr[idx-1]), cudaMemcpyHostToDevice);
            cudaMemcpy(dog_pyr[idx], d_curr, size_of(dog_pyr[idx]), cudaMemcpyHostToDevice);

            // this will be implemented by Nithila and Rachel
            findAndRefineExtremaKernel<<<grid, block>>>(d_prev, d_curr, d_next, rows, cols, step, threshold, c_r, c_c, c_count); // finds keypoints and adjusts local extrema and orientation calcs

            float row_cands, col_cands;
            int count_cands;
            
            // 3. download candidate (r,c) list back to CPU
            cudaMemcpy(c_count, &count_cands, size_of(float), cudaMemcpyDeviceToHost);
            if(count_cands == 0){
                return
            }

            cudaMemcpy(r_c, &row_cands, size_of(float) * count_cands, cudaMemcpyDeviceToHost);
            cudaMemcpy(c_c, &col_cands, size_of(float) * count_cands, cudaMemcpyDeviceToHost);

            // 4. append to keypoints vector
            for(int i = 0; i < *c_count; i++){
                kpts.push_back(KeyPoint(r_c[i], c_c[i]));
            }

            // 5. Frees up GPU memory
            cudaFree(d_prev);
            cudaFree(d_curr);
            cudaFree(d_next);
            cudaFree(c_r);
            cudaFree(c_c);
            cudaFree(c_count);
        }

};

// Step 1
void SIFT_Impl::findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,std: :vector<KeyPoint>& keypoints ) const
{
    const int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    keypoints.clear();
    
    #ifdef HAVE_CUDA
        for (int o = 0; o < nOctaves; o++)
            for (int i = 1; i <= nOctaveLayers; i++)
                findScaleSpaceExtrema_CUDA(o, i, threshold, ...);
    #else
        TLSDataAccumulator<std::vector<KeyPoint> > tls_kpts_struct;
        for( int o = 0; o < nOctaves; o++ )
            for( int i = 1; i <= nOctaveLayers; i++ )
            {
                const int idx = o*(nOctaveLayers+2)+i;
                const Mat& img = dog_pyr[idx];
                const int step = (int)img.step1();
                const int rows = img.rows, cols = img.cols;

                #ifdef HAVE_CUDA
                    findScaleSpaceExtrema_CUDA(o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, tls_kpts_struct);
                #else
                    parallel_for_(
                        Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER),
                        findScaleSpaceExtremaComputer(
                            o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, tls_kpts_struct)
                    );
                #endif
            }

        std::vector<std::vector<KeyPoint>*> kpt_vecs;
        tls_kpts_struct.gather(kpt_vecs);
        for (size_t i = 0; i < kpt_vecs.size(); ++i) {
            keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
        }
    #endif
}

/*
SIFT_Impl::findScaleSpaceExtrema()       ← outer loop over octaves/layers
    └── parallel_for_(..., findScaleSpaceExtremaComputer)   ← dispatches per row-range
            └── findScaleSpaceExtremaComputer::operator()   ← one thread's chunk
                    └── CV_CPU_DISPATCH → findScaleSpaceExtrema()  ← actual work
                            └── findScaleSpaceExtremaT::process()  ← pixel-level logic
*/

// Main Function for SIFT
void detectAndCompute(InputArray _image, InputArray _mask,std::vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints=False)
{
    CV_TRACE_FUNCTION();

    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat(), mask = _mask.getMat();

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma, enable_precise_upscale);
    std::vector<Mat> gpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - firstOctave;
    
    // NICO implements
    buildGaussianPyramid(base, gpyr, nOctaves);
    std::vector<Mat> dogpyr;
    buildDoGPyramid(gpyr, dogpyr);

    // Rachel and Nithila implements
    findScaleSpaceExtrema(gpyr, dogpyr, keypoints);

    KeyPointsFilter::removeDuplicatedSorted( keypoints );

    // lets just hope this doesn't get called LMAO
    if( nfeatures > 0 )
        KeyPointsFilter::retainBest(keypoints, nfeatures);

    if( firstOctave < 0 )
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            KeyPoint& kpt = keypoints[i];
            float scale = 1.f/(float)(1 << -firstOctave);
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }
    if( !mask.empty() )
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
    
    // Maggie implements
    if( _descriptors.needed() )
    {
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, descriptor_type);
        Mat descriptors = _descriptors.getMat();
        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
    }
}