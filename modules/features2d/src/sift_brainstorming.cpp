#include "precomp.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "sift_cuda.cu"
#include "sift.simd.hpp"
#include "sift.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

// Step 4
class findScaleSpaceExtremaT
{
    public:
        findScaleSpaceExtremaT(
            int _o,
            int _i,
            int _threshold,
            int _idx,
            int _step,
            int _cols,
            int _nOctaveLayers,
            double _contrastThreshold,
            double _edgeThreshold,
            double _sigma,
            const std::vector<Mat>& _gauss_pyr,
            const std::vector<Mat>& _dog_pyr,
            std::vector<KeyPoint>& kpts)

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
        void process(const cv::Range& range)
        {
            CV_TRACE_FUNCTION();

            const int begin = range.start;
            const int end = range.end;

            static const int n = SIFT_ORI_HIST_BINS;
            float CV_DECL_ALIGNED(CV_SIMD_WIDTH) hist[n];

            const Mat& img = dog_pyr[idx];
            const Mat& prev = dog_pyr[idx-1];
            const Mat& next = dog_pyr[idx+1];

            for( int r = begin; r < end; r++)
            {
                const sift_wt* currptr = img.ptr<sift_wt>(r);
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);
                int c = SIFT_IMG_BORDER;

                #if (CV_SIMD || CV_SIMD_SCALABLE) && !(DoG_TYPE_SHORT)
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

                #endif //CV_SIMD && !(DoG_TYPE_SHORT)

                // vector loop reminder, better predictibility and less branch density
                for( ; c < cols-SIFT_IMG_BORDER; c++)
                {
                    sift_wt val = currptr[c];
                    if (std::abs(val) <= threshold)
                        continue;

                    sift_wt _00,_01,_02;
                    sift_wt _10,    _12;
                    sift_wt _20,_21,_22;
                    _00 = currptr[c-step-1]; _01 = currptr[c-step]; _02 = currptr[c-step+1];
                    _10 = currptr[c     -1];                        _12 = currptr[c     +1];
                    _20 = currptr[c+step-1]; _21 = currptr[c+step]; _22 = currptr[c+step+1];

                    bool calculate = false;
                    if (val > 0)
                    {
                        sift_wt vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                        if (val >= vmax)
                        {
                            _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                            _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                            _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];
                            vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                            if (val >= vmax)
                            {
                                _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                                _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                                _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];
                                vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                                if (val >= vmax)
                                {
                                    sift_wt _11p = prevptr[c], _11n = nextptr[c];
                                    calculate = (val >= std::max(_11p,_11n));
                                }
                            }
                        }

                    } else  { // val can't be zero here (first abs took care of zero), must be negative
                        sift_wt vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
                        if (val <= vmin)
                        {
                            _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                            _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                            _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];
                            vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
                            if (val <= vmin)
                            {
                                _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                                _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                                _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];
                                vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
                                if (val <= vmin)
                                {
                                    sift_wt _11p = prevptr[c], _11n = nextptr[c];
                                    calculate = (val <= std::min(_11p,_11n));
                                }
                            }
                        }
                    }

                    if (calculate)
                    {
                        CV_TRACE_REGION("pixel_candidate");

                        KeyPoint kpt;
                        int r1 = r, c1 = c, layer = i;
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