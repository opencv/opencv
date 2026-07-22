#include <vector>
#include <queue>
#include <map>

#include "../precomp.hpp"
#include "opencv2/objdetect/aruco2.hpp"
#include "aruco2_dictionary.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/logger.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_objdetect.hpp"
#endif

//IF OpenCV 5
#if CV_VERSION_MAJOR >= 5
#include <opencv2/3d.hpp>
//IF OpenCV 4
#elif CV_VERSION_MAJOR >= 4
#include <opencv2/calib3d.hpp>
#endif

namespace {
using namespace cv;
using namespace cv::aruco2;

struct GPUDictCache {
    int markerSize;
    int num_markers;
    const uchar *bytes_ptr;
    void *context_ptr;
    cv::UMat u_dict_codes;
};

/** @brief The MarkerDetector class is detecting the markers in the image passed */
class MarkerDetector{
public:
    // The only function you need to call
    static inline std::vector<FiducialMarker> detect(const cv::Mat &img, const std::vector<DictionaryType> dictionaries,  const DetectionParameters &params=DetectionParameters(),std::vector<FiducialMarker> *candidatesOut=nullptr,cv::Mat ThresImageIn={});
    static inline std::vector<FiducialMarker> detect(const cv::UMat &img, const std::vector<DictionaryType> dictionaries,  const DetectionParameters &params=DetectionParameters(),std::vector<FiducialMarker> *candidatesOut=nullptr,cv::UMat ThresImageIn={});
private:
    static inline FiducialMarker sort( const  FiducialMarker &marker);
    static inline float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline int   getMarkerId(  cv::Mat  candidateBits,int &idx, int &nrotations, const DetectionParameters &params,Dictionary &dictionary);
    static inline int isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) ;
    static std::vector<std::vector<cv::Point>> visitedAwareTracingContour(cv::Mat &padded_io, size_t minSize = 1,float maxRevisited=0.1) ;
    static int getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize=1) ;
    static void thres255Adaptive(cv::Mat &in,cv::Mat &out,int off=2,int thres=5);
};

namespace _private {
struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f((m[0]*p.x+m[1]*p.y+m[2])/c,(m[3]*p.x+m[4]*p.y+m[5])/c);
    }
    cv::Mat H;
};
}
//Marker intersection. Tells the marker with most corners into another. 0 if no intersection or tie
int MarkerDetector::isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) {
    // Lambda for point-in-polygon test (Ray Casting)
    auto countInside = [](const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) -> int {
        int count = 0;
        for (const auto& pt : source) {
            bool inside = false;
            // Fixed 4-side loop logic
            for (int i = 0, j = 3; i < 4; j = i++) {
                if (((target[i].y > pt.y) != (target[j].y > pt.y)) &&
                    (pt.x < (target[j].x - target[i].x) * (pt.y - target[i].y) / (target[j].y - target[i].y) + target[i].x)) {
                    inside = !inside;
                }
            }
            if (inside) count++;
        }
        return count;
    };
    // Count how many corners of A are in B
    int aInB = countInside(a, b);
    // Count how many corners of B are in A
    int bInA = countInside(b, a);
    // Rule 1: Must contain at least one corner
    if (aInB == 0 && bInA == 0) return 0;
    // Rule 2: Compare counts
    if (aInB > bInA) return 1;
    if (bInA > aInB) return 2;
    // Default: Tie or no relative dominance
    return 0;
}

std::vector<FiducialMarker> MarkerDetector::detect(const cv::UMat &img, const std::vector<DictionaryType> dictionaries,
    const DetectionParameters &params, std::vector<FiducialMarker> *candidatesOut, cv::UMat ThresImIn) {
#ifndef HAVE_OPENCL
    cv::Mat mat = img.getMat(cv::ACCESS_READ);
    cv::Mat thresMat = ThresImIn.empty() ? cv::Mat() : ThresImIn.getMat(cv::ACCESS_READ);
    return detect(mat, dictionaries, params, candidatesOut, thresMat);
#else
    std::vector<FiducialMarker> DetectedMarkers;

    int width = img.cols;
    int height = img.rows;

    // GPU buffers allocated on stack (efficiently recycled by OpenCV's internal
    // pool)
    cv::UMat u_bwimage, u_mean, u_labels, u_thresh, u_count;
    cv::UMat u_harris, u_harris_norm;
    cv::UMat u_corners_out, u_corner_count, u_hash_keys, u_hash_counts;
    cv::UMat u_valid_corners_out, u_final_polygon_count;

    // Harris parameters
    int h_bsize = 3;  // Harris block size
    float h_k = 0.02; // Harris aperture Sobel

    // W_0 parameter for GPU scale computation
    float gpuW0 = 1024.f;

    int k = std::max(0, (int)std::floor(std::log2((float)width / gpuW0)));
    int h_ksize = (3 + 2 * k);               // Harris kernel size
    int nms_radius = std::max(1, 2 * k - 1); // NMS radius

    if (img.channels() == 3) {
        cv::cvtColor(img, u_bwimage, cv::COLOR_BGR2GRAY);
    } else {
        u_bwimage = img; // Already a 1-channel grayscale image (CV_8UC1)
    }

    // Create the label buffer (32-bit integers, 1 channel)
    u_labels.create(height, width, CV_32SC1);
    if (!u_labels.isContinuous()) {
        u_labels = u_labels.clone();
    }

    u_count.create(1, 1, CV_32SC1);

    // Retrieve the compiled OpenCL program from OpenCV's context-level cache
    cv::String errmsg;
    cv::ocl::Program program = cv::ocl::Context::getDefault().getProg(
        cv::ocl::objdetect::aruco2detect_oclsrc, "", errmsg);
    if (program.empty()) {
        return {}; // Fallback to CPU if compilation fails
    }

    size_t globalThreads[2] = {(size_t)width, (size_t)height};
    CV_UNUSED(globalThreads); // Silencing warning until used in GPU kernels

    if (ThresImIn.empty()) {
        int winSize = (int)params.boxFilterSize;
        if (winSize % 2 == 0) {
            winSize++;
        }
        cv::boxFilter(u_bwimage, u_mean, u_bwimage.type(),
                      cv::Size(winSize, winSize));

        u_thresh.create(height, width, CV_8UC1);
        // Force no memory padding
        if (!u_thresh.isContinuous()) {
            u_thresh = u_thresh.clone();
        }

        // 1. Preprocess and Init (Fused)
        // Concurrently thresholds the image and initializes the label array for
        // Union-Find
        cv::ocl::Kernel k_prep_init("preprocess_and_init", program);
        k_prep_init.args(cv::ocl::KernelArg::PtrReadOnly(u_bwimage),
                         cv::ocl::KernelArg::PtrReadOnly(u_mean),
                         cv::ocl::KernelArg::PtrWriteOnly(u_thresh),
                         cv::ocl::KernelArg::PtrReadWrite(u_labels), width, height,
                         (uchar)params.thres);
        k_prep_init.run(2, globalThreads, NULL, false);
    } else {
        CV_Assert(ThresImIn.size() == cv::Size(width, height) &&
                  ThresImIn.type() == CV_8UC1);
        u_thresh = ThresImIn;

        // 1. Init labels
        // Initialize the Union-Find label array when thresholded image is
        // pre-provided
        cv::ocl::Kernel k_init_labels("init_labels", program);
        k_init_labels.args(cv::ocl::KernelArg::PtrReadWrite(u_labels), width,
                           height);
        k_init_labels.run(2, globalThreads, NULL, false);
    }

    // 2. Merge (Union-Find)
    // Merges adjacent pixels with the same binary threshold state in the label
    // array
    cv::ocl::Kernel k_merge("uf_merge", program);
    k_merge.args(cv::ocl::KernelArg::PtrReadOnly(u_thresh),
                 cv::ocl::KernelArg::PtrReadWrite(u_labels), width, height);
    k_merge.run(2, globalThreads, NULL, false);

    // 3. Flatten (Union-Find)
    // Resolves the equivalence trees so that each pixel points directly to its
    // root component ID
    cv::ocl::Kernel k_flatten("uf_flatten", program);

    u_count.setTo(0);
    k_flatten.args(cv::ocl::KernelArg::PtrReadWrite(u_labels),
                   cv::ocl::KernelArg::PtrReadWrite(u_count), width, height);
    k_flatten.run(2, globalThreads, NULL, false);

    const int HASH_SIZE = 131072;
    const int max_points_per_hash = 8;

    u_corners_out.create(1, HASH_SIZE * max_points_per_hash * 2, CV_32SC1);

    u_corner_count.create(1, 1, CV_32SC1);
    u_corner_count.setTo(cv::Scalar(0));

    u_hash_keys.create(1, HASH_SIZE, CV_32SC1);
    u_hash_keys.setTo(cv::Scalar(0));

    u_hash_counts.create(1, HASH_SIZE, CV_32SC1);
    u_hash_counts.setTo(cv::Scalar(0));

    // Compute Harris corner response map on the thresholded image
    cv::cornerHarris(u_thresh, u_harris, h_bsize, h_ksize, h_k);

    cv::pow(u_harris, 0.5, u_harris);
    // Normalize values to the [0, 255] range
    cv::normalize(u_harris, u_harris_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1,
                  cv::noArray());

    // Step 4: Corner Extraction (NMS)
    // Finds corners within each labeled component and stores them in a spatial
    // hash table
    cv::ocl::Kernel k_nms("find_corners_nms", program);
    k_nms.args(cv::ocl::KernelArg::PtrReadOnly(u_harris_norm),
               cv::ocl::KernelArg::PtrReadOnly(u_labels),
               cv::ocl::KernelArg::PtrWriteOnly(u_corners_out),
               cv::ocl::KernelArg::PtrReadWrite(u_corner_count),
               cv::ocl::KernelArg::PtrReadWrite(u_hash_keys),
               cv::ocl::KernelArg::PtrReadWrite(u_hash_counts), nms_radius,
               HASH_SIZE, width, height, params.harrisThresh,
               max_points_per_hash);
    k_nms.run(2, globalThreads, NULL, true);

    //---------------------------------------------------------
    // COMPACTION AND SORTING (GPU-based)
    //---------------------------------------------------------

    int MAX_EXPECTED_MARKERS = 10000;

    u_valid_corners_out.create(
        1, MAX_EXPECTED_MARKERS * 8,
        CV_32SC1); // 8 integers per marker (4 pts * 2 coords)

    u_final_polygon_count.create(1, 1, CV_32SC1);
    u_final_polygon_count.setTo(cv::Scalar(0));

    cv::ocl::Kernel k_compact("compact_and_sort", program);
    k_compact.args(cv::ocl::KernelArg::PtrReadOnly(u_hash_counts),
                   cv::ocl::KernelArg::PtrReadOnly(u_corners_out),
                   cv::ocl::KernelArg::PtrWriteOnly(u_valid_corners_out),
                   cv::ocl::KernelArg::PtrReadWrite(u_final_polygon_count),
                   max_points_per_hash, MAX_EXPECTED_MARKERS,
                   params.minSize * params.minSize,
                   cv::ocl::KernelArg::PtrReadOnly(u_labels), width, height);

    // Launch one thread per hash table slot
    size_t globalThreadsCompact[1] = {(size_t)HASH_SIZE};
    k_compact.run(1, globalThreadsCompact, NULL, true);

    int num_polygons = 0;
    std::vector<int> cpu_valid_corners;
    cv::Mat mat_final_count = u_final_polygon_count.getMat(cv::ACCESS_READ);
    num_polygons = mat_final_count.at<int>(0, 0);
    if (num_polygons > 0) {
        cv::Mat mat_valid_corners = u_valid_corners_out.getMat(cv::ACCESS_READ);
        cpu_valid_corners.resize(num_polygons * 8);
        std::memcpy(cpu_valid_corners.data(), mat_valid_corners.ptr<int>(),
                    num_polygons * 8 * sizeof(int));
    }

    std::vector<FiducialMarker> candidateslocal;
    if (candidatesOut != nullptr) {
        candidatesOut->clear();
    } else {
        candidatesOut = &candidateslocal;
    }

    const int *ptr_valid = cpu_valid_corners.data();
    for (int i = 0; i < num_polygons; i++) {
        FiducialMarker marker;
        marker.corners.reserve(4);
        // Read the points already sorted by the kernel
        for (int j = 0; j < 4; j++) {
            marker.corners.emplace_back(
                cv::Point2f(ptr_valid[i * 8 + j * 2], ptr_valid[i * 8 + j * 2 + 1]));
        }
        marker.id = i; // Store original GPU index

        candidatesOut->push_back(marker);
    }

    //---------------------------------------------------------
    // STABILIZE CANDIDATES ORDER (CPU)
    //---------------------------------------------------------
    // The GPU returns candidates in a non-deterministic order due to atomic
    // operations. We sort them based on their centroid to ensure deterministic
    // iterations.
    std::sort(candidatesOut->begin(), candidatesOut->end(),
              [](const FiducialMarker &a, const FiducialMarker &b) {
                  float ca_x = 0, ca_y = 0;
                  for (const auto &p : a.corners) {
                      ca_x += p.x;
                      ca_y += p.y;
                  }
                  float cb_x = 0, cb_y = 0;
                  for (const auto &p : b.corners) {
                      cb_x += p.x;
                      cb_y += p.y;
                  }
                  if (std::abs(ca_y - cb_y) > 2.0f)
                      return ca_y < cb_y;
                  return ca_x < cb_x;
              });

    //---------------------------------------------------------
    // EXTRACT MARKERS ID (GPU First Pass + CPU Fallback)
    //---------------------------------------------------------
    if (num_polygons > 0) {
        // ---------------------------------------------------------
        // GPU PATH: Entire Pipeline on GPU
        // ---------------------------------------------------------

        // 1. Calculate average edge length of candidates to determine refinement
        // window size
        double avrgLen = 0;
        for (int i = 0; i < num_polygons; i++) {
            for (int j = 0; j < 4; j++) {
                float x1 = cpu_valid_corners[i * 8 + j * 2];
                float y1 = cpu_valid_corners[i * 8 + j * 2 + 1];
                float x2 = cpu_valid_corners[i * 8 + ((j + 1) % 4) * 2];
                float y2 = cpu_valid_corners[i * 8 + ((j + 1) % 4) * 2 + 1];
                avrgLen += std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
            }
        }
        avrgLen /= 4 * num_polygons;
        int halfwsize =
            std::min(int(3 * std::max(1.f, float(avrgLen) / float(34))), 9);

        // 2. Allocate tracking buffers on GPU
        cv::UMat u_candidate_matched(1, num_polygons, CV_32SC1);
        u_candidate_matched.setTo(cv::Scalar(0));

        cv::UMat u_detected_count(1, 1, CV_32SC1);
        u_detected_count.setTo(cv::Scalar(0));

        cv::UMat u_detected_markers(1, num_polygons * 10, CV_32FC1);

        // 3. Loop over dictionaries entirely on GPU
        for (size_t di = 0; di < dictionaries.size(); di++) {
            Dictionary dictInstance = getPredefinedDictionary(dictionaries[di]);
            cv::UMat u_dict_codes;

            thread_local static std::vector<GPUDictCache> dict_cache;
            void *current_ctx = cv::ocl::Context::getDefault().ptr();

            // Prune stale cache entries from other contexts to release GPU resources
            dict_cache.erase(std::remove_if(dict_cache.begin(), dict_cache.end(),
                                            [current_ctx](const GPUDictCache &c) {
                                                return c.context_ptr != current_ctx;
                                            }),
                             dict_cache.end());

            const GPUDictCache *cached_found = nullptr;
            for (const auto &cache : dict_cache) {
                if (cache.markerSize == dictInstance.markerSize &&
                    cache.num_markers == dictInstance.bytesList.rows &&
                    cache.bytes_ptr == dictInstance.bytesList.data) {
                    cached_found = &cache;
                    break;
                }
            }

            if (cached_found) {
                u_dict_codes = cached_found->u_dict_codes;
            } else {
                std::vector<uint64_t> gpu_dict_codes;
                gpu_dict_codes.reserve(dictInstance.bytesList.rows);
                for (int i = 0; i < dictInstance.bytesList.rows; i++) {
                    cv::Mat bits_mat = Dictionary::getBitsFromByteList(
                        dictInstance.bytesList.row(i), dictInstance.markerSize);
                    uint64_t code = 0;
                    int bit_idx = 0;
                    for (int r = 0; r < bits_mat.rows; r++) {
                        for (int c = 0; c < bits_mat.cols; c++) {
                            uint64_t bit = bits_mat.at<uchar>(r, c) ? 1 : 0;
                            code |= (bit << bit_idx);
                            bit_idx++;
                        }
                    }
                    gpu_dict_codes.push_back(code);
                }

                if (!gpu_dict_codes.empty()) {
                    cv::Mat mat_dict_codes(1, (int)gpu_dict_codes.size() * 8, CV_8UC1,
                                           (void *)gpu_dict_codes.data());
                    mat_dict_codes.copyTo(u_dict_codes);
                }

                GPUDictCache new_cache;
                new_cache.markerSize = dictInstance.markerSize;
                new_cache.num_markers = dictInstance.bytesList.rows;
                new_cache.bytes_ptr = dictInstance.bytesList.data;
                new_cache.context_ptr = current_ctx;
                new_cache.u_dict_codes = u_dict_codes;
                dict_cache.push_back(new_cache);
            }

            int max_attempts = params.maxAttemptsPerCandidate;
            if (max_attempts < 1)
                max_attempts = 1;

            cv::UMat u_identified_ids;
            cv::UMat u_identified_rotations;
            u_identified_ids.create(1, num_polygons * max_attempts, CV_32SC1);
            u_identified_rotations.create(1, num_polygons * max_attempts, CV_32SC1);
            u_identified_ids.setTo(cv::Scalar(-1));
            u_identified_rotations.setTo(cv::Scalar(-1));

            if (!u_dict_codes.empty()) {
                cv::ocl::Kernel k_identify("identify_candidates", program);
                unsigned int seed = 12345;

                k_identify.args(
                    cv::ocl::KernelArg::PtrReadOnly(u_bwimage), (int)u_bwimage.step,
                    width, height, cv::ocl::KernelArg::PtrReadOnly(u_valid_corners_out),
                    num_polygons, cv::ocl::KernelArg::PtrReadOnly(u_dict_codes),
                    (int)dictInstance.bytesList.rows, dictInstance.markerSize,
                    (int)(dictInstance.maxCorrectionBits * params.errorCorrectionRate),
                    (float)params.maxErroneousBitsInBorderRate,
                    (uchar)params.detectColorMode,
                    (uchar)params.gridBitSampling, max_attempts,
                    (unsigned int)seed,
                    cv::ocl::KernelArg::PtrReadOnly(u_candidate_matched),
                    cv::ocl::KernelArg::PtrWriteOnly(u_identified_ids),
                    cv::ocl::KernelArg::PtrWriteOnly(u_identified_rotations));
                size_t globalThreadsIdentify[1] = {(size_t)num_polygons *
                                                   (size_t)max_attempts};
                k_identify.run(1, globalThreadsIdentify, NULL, true);

                cv::ocl::Kernel k_refine("refine_and_collect_matches", program);
                k_refine.args(
                    cv::ocl::KernelArg::PtrReadOnly(u_bwimage), (int)u_bwimage.step,
                    width, height, cv::ocl::KernelArg::PtrReadOnly(u_valid_corners_out),
                    cv::ocl::KernelArg::PtrReadOnly(u_identified_ids),
                    cv::ocl::KernelArg::PtrReadOnly(u_identified_rotations),
                    cv::ocl::KernelArg::PtrReadWrite(u_candidate_matched), num_polygons,
                    max_attempts, (int)di, dictInstance.markerSize, halfwsize, 12,
                    0.005f, cv::ocl::KernelArg::PtrReadWrite(u_detected_count),
                    cv::ocl::KernelArg::PtrReadWrite(u_detected_markers), num_polygons);
                size_t globalThreadsRefine[1] = {(size_t)num_polygons};
                k_refine.run(1, globalThreadsRefine, NULL, true);
            }
        }

        // 4. Download final detected markers list to host
        cv::Mat mat_count = u_detected_count.getMat(cv::ACCESS_READ);
        int detected_count = mat_count.at<int>(0, 0);
        if (detected_count > 0) {
            cv::Mat mat_detected = u_detected_markers.getMat(cv::ACCESS_READ);
            const float *ptr_detected = mat_detected.ptr<float>();
            for (int i = 0; i < detected_count; i++) {
                int out_base = i * 10;
                int matched_id = (int)ptr_detected[out_base + 0];
                int dict_idx = (int)ptr_detected[out_base + 1];

                FiducialMarker marker;
                marker.id = matched_id;
                marker.dictionary = dictionaries[dict_idx];
                marker.corners.resize(4);
                marker.corners[0] =
                    cv::Point2f(ptr_detected[out_base + 2], ptr_detected[out_base + 3]);
                marker.corners[1] =
                    cv::Point2f(ptr_detected[out_base + 4], ptr_detected[out_base + 5]);
                marker.corners[2] =
                    cv::Point2f(ptr_detected[out_base + 6], ptr_detected[out_base + 7]);
                marker.corners[3] =
                    cv::Point2f(ptr_detected[out_base + 8], ptr_detected[out_base + 9]);
                DetectedMarkers.push_back(marker);
            }
        }
    }

    //---------------------------------------------------------
    // DELETE DUPLICATES (CPU)
    //---------------------------------------------------------
    std::sort(DetectedMarkers.begin(), DetectedMarkers.end(),
              [](const FiducialMarker &a, const FiducialMarker &b) {
                  return a.id < b.id;
              });
    {
        std::vector<bool> toRemove(DetectedMarkers.size(), false);
        for (int i = 0; i < int(DetectedMarkers.size()) - 1; i++) {
            for (int j = i + 1; j < int(DetectedMarkers.size()) && !toRemove[i];
                 j++) {
                // 1. Check if they are nearly identical in position (same or different
                // IDs/dictionaries)
                float dist = 0;
                for (int c = 0; c < 4; c++) {
                    dist += cv::norm(DetectedMarkers[i].corners[c] -
                                     DetectedMarkers[j].corners[c]);
                }
                if (dist < 5.0f) {
                    toRemove[j] = true;
                    continue;
                }

                // 2. Check if one is nested inside the other (only for the same ID)
                if (DetectedMarkers[i].id == DetectedMarkers[j].id) {
                    auto res =
                        isInto(DetectedMarkers[i].corners, DetectedMarkers[j].corners);
                    if (res != 0) {
                        float pixDistAvr = 0;
                        for (int c = 0; c < 4; c++) {
                            pixDistAvr += cv::norm(DetectedMarkers[i].corners[c] -
                                                   DetectedMarkers[j].corners[c]);
                        }
                        pixDistAvr /= 4.0f;

                        Dictionary dictInstance = getPredefinedDictionary(DetectedMarkers[i].dictionary);
                        int mSize = dictInstance.markerSize > 0 ? dictInstance.markerSize : 6;

                        double avrgLen = 0;
                        for (int c = 0; c < 4; c++) {
                            avrgLen += cv::norm(DetectedMarkers[i].corners[c] -
                                                DetectedMarkers[i].corners[(c + 1) % 4]);
                        }
                        avrgLen /= 4.0;
                        avrgLen /= mSize;

                        if (pixDistAvr > avrgLen) continue;

                        if (res == 1)
                            toRemove[i] = true;
                        else if (res == 2)
                            toRemove[j] = true;
                    }
                }
            }
        }
        // now remove the marked ones
        std::vector<FiducialMarker> DetectedMarkers2;
        for (unsigned int i = 0; i < DetectedMarkers.size(); i++)
            if (!toRemove[i])
                DetectedMarkers2.push_back(DetectedMarkers[i]);
        DetectedMarkers = DetectedMarkers2;
    }

    return DetectedMarkers;
#endif
}

std::vector<FiducialMarker>  MarkerDetector::detect(const cv::Mat &img,   const std::vector<DictionaryType> dictionaries,const DetectionParameters &params,std::vector<FiducialMarker> *candidatesOut,cv::Mat ThresImIn){

    DetectionParameters paramsCopy = params;
    if (paramsCopy.detectInvertedMarker && paramsCopy.detectColorMode == 0) {
        paramsCopy.detectColorMode = 1;
    }

    cv::Mat bwimage,thresImage;
    std::vector<FiducialMarker> DetectedMarkers;
    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;
    /////////////////// Adaptive Threshold to detect border

    if(ThresImIn.empty()){
        cv::boxFilter( bwimage, thresImage, bwimage.type(), cv::Size(paramsCopy.boxFilterSize, paramsCopy.boxFilterSize),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
        thresImage=thresImage-bwimage;
        cv::threshold(thresImage, thresImage, paramsCopy.thres, 255, cv::THRESH_BINARY);
    }
    else{
        thresImage=ThresImIn;
    }

    /////////////////// compute marker candidates by detecting contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::RNG rand;
    //cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    int  minSizeSq=paramsCopy.minSize*paramsCopy.minSize,minSize4=4*paramsCopy.minSize;
    contours=visitedAwareTracingContour(thresImage,minSize4,paramsCopy.maxTimesRevisited);

    //decide where to store the candidates. If candidatesOut is not null, store there, otherwise use a local variable
    std::vector<FiducialMarker> candidateslocal;
    if(candidatesOut!=nullptr) {
        candidatesOut->clear();
    }
    else{
        candidatesOut=&candidateslocal;
    }
    ///////////////// for each contour, approx to a rectangle
    for (size_t i = 0; i < contours.size(); i++)
    {
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        //check distance  between corners at least minSize pix
        if(  ((approxCurve[0].x-approxCurve[1].x)*(approxCurve[0].x-approxCurve[1].x) + (approxCurve[0].y-approxCurve[1].y)*(approxCurve[0].y-approxCurve[1].y))<minSizeSq) continue;
        if(  ((approxCurve[1].x-approxCurve[2].x)*(approxCurve[1].x-approxCurve[2].x) + (approxCurve[1].y-approxCurve[2].y)*(approxCurve[1].y-approxCurve[2].y))<minSizeSq) continue;
        if(  ((approxCurve[2].x-approxCurve[3].x)*(approxCurve[2].x-approxCurve[3].x) + (approxCurve[2].y-approxCurve[3].y)*(approxCurve[2].y-approxCurve[3].y))<minSizeSq) continue;
        if(  ((approxCurve[3].x-approxCurve[0].x)*(approxCurve[3].x-approxCurve[0].x) + (approxCurve[3].y-approxCurve[0].y)*(approxCurve[3].y-approxCurve[0].y))<minSizeSq) continue;
        // // add the points
        FiducialMarker marker;marker.corners.reserve(4);
        for (int j = 0; j < 4; j++)
            marker.corners.emplace_back( cv::Point2f( float(approxCurve[j].x),float(approxCurve[j].y)));
        //sort corner in clockwise direction
        marker=sort(marker);
        candidatesOut->push_back(marker);
    }


    //now, for each candidate check bits inside
    for(size_t di=0;di<dictionaries.size();di++){
        Dictionary dictInstance = getPredefinedDictionary(dictionaries[di]);
        std::vector<FiducialMarker> currDirMarkerDetected;
        cv::Mat bits(dictInstance.markerSize+2,dictInstance.markerSize+2,CV_8UC1),bitadaptive(dictInstance.markerSize+2,dictInstance.markerSize+2,CV_8UC1);

        for(auto it=candidatesOut->begin();it!=candidatesOut->end();){
            auto marker=*it;


            ////// extract the code. Obtain the intensities of the bits using  homography
            for(int i=0;i<int(paramsCopy.maxAttemptsPerCandidate) && marker.id==-1;i++){
                //if not first attempt, we may wanna produce small random alteration of the corners
                auto marker2=marker;
                if( i!=0) for(int c=0;c<4;c++) {marker2.corners[c].x+=rand.gaussian(0.75);marker2.corners[c].y+=rand.gaussian(0.75);}//if not first, alter corner location
                _private::Homographer hom(marker2.corners);
                for(int r=0;r<bits.rows;r++){
                    for(int c=0;c<bits.cols;c++){
                        if(paramsCopy.gridBitSampling == false  ){//sample only the central bit
                           // std::cout<<cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  )<<std::endl;
                            bits.at<uchar>(r,c)=uchar(0.5+getSubpixelValue(bwimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                        }
                        else{
                            //evaluate a grid of points (rows+cols) into each bit
                            double sum=0;
                            double intrabitIncR=1./double(bits.rows*bits.rows);
                            double intrabitIncC=1./double(bits.cols*bits.cols);

                            for (int sr = 0; sr < bits.rows; sr++) {
                                for (int sc = 0; sc < bits.cols; sc++) {
                                    // Only proceed if it is a border element (first row, last row, first col, or last col)
                                    if (sr == 0 || sr == bits.rows - 1 || sc == 0 || sc == bits.cols - 1) {
                                        auto pix = hom(cv::Point2f( (float(c) / float(bits.cols)) + (0.5 + float(sc)) * intrabitIncR,(float(r) / float(bits.rows)) + (0.5 + float(sr)) * intrabitIncC ));
                                        sum += getSubpixelValue(bwimage, pix);
                                    }
                                }
                            }
                            bits.at<uchar>(r,c)=uchar( 0.5+  sum/(2*bits.rows+2*bits.cols-4));
                        }
                    }
                }


                if(i==2){ // if not working the first time, try this time adaptive threshold into the bits to improve robustness to lighting
                    thres255Adaptive(bits,bitadaptive);
                    bitadaptive.copyTo(bits);
                }
                else{
                    cv::threshold(bits,bits,0,255,cv::THRESH_OTSU);
                }

                //now, analyze the inner code to see it if is a marker. If so, rotate to have the points properly sorted
                int nrotations=0;
                // 1. Invert bits immediately if we are strictly in Mode 1
                if (paramsCopy.detectColorMode == 1) {
                    bits = ~bits;
                }
                // 2. Perform the first check (covers Mode 0, Mode 1, and the first half of Mode 2)
                if (getMarkerId(bits, marker.id, nrotations, paramsCopy, dictInstance) == 0) {
                    // 3. If the check fails and we aren't in Mode 2, skip to the next iteration
                    if (paramsCopy.detectColorMode != 2) continue;
                    // 4. If we ARE in Mode 2, apply the fallback: invert and check one last time
                    bits = ~bits;
                    if (getMarkerId(bits, marker.id, nrotations, paramsCopy, dictInstance) == 0) continue;
                }
                std::rotate(marker.corners.begin(),marker.corners.begin() + 4 - nrotations,marker.corners.end());
            }
            if(marker.id!=-1) {
                marker.dictionary= dictionaries[di];
                currDirMarkerDetected.push_back(marker);
                //remove from candidate list
                it=candidatesOut->erase(it);
            }
            else it++;//go to next
        }


        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(currDirMarkerDetected.begin(), currDirMarkerDetected.end(),[](const FiducialMarker &a,const FiducialMarker &b){return a.id<b.id;});
        {
            std::vector<bool> toRemove(currDirMarkerDetected.size(), false);
            for (int i = 0; i < int(currDirMarkerDetected.size()) - 1; i++)
            {
                for (int j = i + 1; j < int(currDirMarkerDetected.size()) && !toRemove[i]; j++)
                {
                    if (currDirMarkerDetected[i].id == currDirMarkerDetected[j].id )
                    {
                        auto res=isInto(currDirMarkerDetected[i].corners,currDirMarkerDetected[j].corners);
                        //lets compute the average pixel distances
                        int pixDistAvr=0;
                        for(int c=0;c<4;c++){
                            pixDistAvr+=cv::norm(currDirMarkerDetected[i].corners[c]-currDirMarkerDetected[j].corners[c]);
                        }
                        pixDistAvr/=4;
                        //calculate the average length of the bits for this marker
                        double avrgLen=0;
                        for(int c=0;c<4;c++){
                            avrgLen+=cv::norm(currDirMarkerDetected[i].corners[c]-currDirMarkerDetected[i].corners[(c+1)%4]);
                        }
                        avrgLen/=4;
                        avrgLen/=dictInstance.markerSize;
                        //if the distance greter than avrgLen, do not remove
                        if(pixDistAvr>avrgLen)continue;

                        if( res==1)toRemove[i]=true;
                        else if( res==2)toRemove[j]=true;

                    }
                }
            }
            //now move to DetectedMarkers these not marked for removal
            for (size_t i = 0; i < currDirMarkerDetected.size(); i++)
                if (!toRemove[i]) DetectedMarkers.push_back(currDirMarkerDetected[i]);

        }
    }//for(size_t di=0;di<dictionaries.size();di++){


    ////// finally subpixel corner refinement
    if(DetectedMarkers.size()>0){
        double avrgLen=0;
        for(const auto &m:DetectedMarkers)
            for(int i=0;i<4;i++)
                avrgLen+=cv::norm(m.corners[i]-m.corners[(i+1)%4]);
        avrgLen/=4*DetectedMarkers.size();
        int halfwsize=std::min(int(3*std::max(1.f,float(avrgLen)/float(34))),9);
        std::vector<cv::Point2f> Corners;
        for (const auto &m:DetectedMarkers)
            Corners.insert(Corners.end(), m.corners.begin(),m.corners.end());
        cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
        // copy back to the markers
        for (size_t i = 0; i < DetectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++) DetectedMarkers[i].corners[c] = Corners[i * 4 + c];
    }
    return DetectedMarkers;//DONE
}
/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
int MarkerDetector:: getMarkerId(cv::Mat candidateBits, int &idx, int &nrotations, const DetectionParameters &params,Dictionary &dictionary){
    uint8_t typ=1;

    // analyze border bits
    int maximumErrorsInBorder =int(dictionary.markerSize * dictionary.markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =getBorderErrors(candidateBits, dictionary.markerSize);

    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong
    // take only inner bits
    cv::Mat onlyBits =candidateBits.rowRange(1,candidateBits.rows - 1).colRange(1, candidateBits.cols - 1);
    onlyBits/=255;
    // try to indentify the marker
    if(!dictionary.identify(onlyBits, idx, nrotations, params.errorCorrectionRate))
        return 0;
    return typ;



}
/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
int MarkerDetector::getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) {
    int sizeWithBorders = markerSize + 2 * borderSize;
    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}
float MarkerDetector::getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p) {
    // 1. Get integer coordinates
    const int ix = static_cast<int>(p.x);
    const int iy = static_cast<int>(p.y);


    //   Boundary Check: Ensure the 2x2 patch is within limits
    // We check ix+1 and iy+1 because the interpolation looks at the next pixel over.
    if (ix < 0 || iy < 0 || ix >= im_grey.cols - 1 || iy >= im_grey.rows - 1) {
        // Option A: Return a default value
        // Option B: Clamp the point to the nearest valid boundary
        return 0.0f;
    }

    // 2. Get fractional parts
    const float dx = p.x - ix;
    const float dy = p.y - iy;
    // 3. Optimized Pointer Access
    const uchar* ptr = im_grey.ptr<uchar>(iy) + ix;
    const size_t step = im_grey.step;
    // 4. Fetch the four pixels immediately as floats
    const float p00 = static_cast<float>(ptr[0]);        // Top-Left
    const float p01 = static_cast<float>(ptr[1]);        // Top-Right
    const float p10 = static_cast<float>(ptr[step]);     // Bottom-Left
    const float p11 = static_cast<float>(ptr[step + 1]); // Bottom-Right
    // 5. Separable Interpolation (3 Multiplications total)
    const float top = p00 + dx * (p01 - p00);// Interpolate Top Row Horizontally
    const float bot = p10 + dx * (p11 - p10);    // Interpolate Bottom Row Horizontallys
    // Interpolate Vertically between Top and Bottom results
    return top + dy * (bot - top);
}
FiducialMarker  MarkerDetector::sort( const  FiducialMarker &marker){
    FiducialMarker res_marker=marker;
    /// sort the points in clockwise order
    double dx1 = res_marker.corners[1].x - res_marker.corners[0].x;
    double dy1 = res_marker.corners[1].y - res_marker.corners[0].y;
    double dx2 = res_marker.corners[2].x - res_marker.corners[0].x;
    double dy2 = res_marker.corners[2].y - res_marker.corners[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);
    // if the third point is in the left side, then sort in anti-clockwise order
    if (o < 0.0)  std::swap(res_marker.corners[1], res_marker.corners[3]);
    return res_marker;
}

/**
 * @brief Traces the contours of a binary image using our visited aware Tracing algorithm.
 *
 * @param padded_io input binary image. It will be modified!
 *
 * This function scans a binary image (foreground as 255, background as 0) and
 * finds the external boundaries of all distinct objects.
 */
std::vector<std::vector<cv::Point>> MarkerDetector::visitedAwareTracingContour(cv::Mat &padded_io, size_t minSize, float maxRevisited ) {
    if (padded_io.empty() || padded_io.type() != CV_8UC1) return {};
    // 1. Fast Initialization and Padding
    int rows = padded_io.rows;
    int cols = padded_io.cols;
    int32_t step = padded_io.step;
    uchar* data = padded_io.data;
    // Fast clear of top and bottom rows
    memset(data, 0, cols);
    memset(data + (rows - 1) * step, 0, cols);
    // Fast clear of left and right columns
    for (int r = 1; r < rows - 1; ++r) {
        uchar* row_ptr = data + r * step;
        row_ptr[0] = 0;
        row_ptr[cols - 1] = 0;
    }
    // 2. Precompute Neighbor Offsets based on image stride This removes the need for Point arithmetic in the loop
    const int offsets[16]={-1,-step-1,-step,-step+1,1,step+1,step,step-1, -1,-step-1,-step,-step+1,1,step+1,step,step-1, };
    // Use static tables to avoid initialization overhead on every call // 8-connectivity offsets relative to center (0,0) // Order: W, NW, N, NE, E, SE, S, SW
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 }, dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    // Pre-allocate results
    std::vector<std::vector<cv::Point>> contours;contours.reserve(2048);
    std::vector<cv::Point> buffer;buffer.reserve(2048);
    const uchar FOREGROUND = 255, BACKGROUND = 0,VISITED = 100;
    // 3. Scanning Loop
    // We iterate using raw pointers for maximum speed
    int rowStep=1;//std::max(1,int(minSize/6));
    for (int r = 1; r < rows - 1; r+=rowStep) {
        uchar* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;  ) {
            ////findStartContourPoint
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                cv::v_uint8 v_zero = cv::vx_setzero_u8();
                for (; c <= cols - cv::VTraits<cv::v_uint8>::vlanes(); c+= cv::VTraits<cv::v_uint8>::vlanes())
                {
                    cv::v_uint8 vmask = (cv::v_ne(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                    if (v_check_any(vmask))
                    {
                        c += v_scan_forward(vmask);
                        break;
                    }
                }
#endif
                //process last tail
                for (; c < cols && !row_ptr[c]; ++c) ;//last tail
            }
            if( c==cols) break;//reached end of row
            if (row_ptr[c] == FOREGROUND ) {// --- 4. Tracing Loop  if is foreground
                buffer.clear();
                int curr_x = c, curr_y = r,search_idx = 1 ;
                uchar* curr_ptr = row_ptr + c,*start_ptr=curr_ptr;
                size_t ntimesRevisited=0;
                do {
                    buffer.emplace_back(curr_x, curr_y);// Add point
                    *curr_ptr = VISITED;// Mark as visited
                    //showImage(padded);
                    // Search for next foreground pixel. We search 8 neighbors starting from search_idx
                    for (int i = 0; i < 8; ++i) {
                        int idx = search_idx + i; // index into offsets (0..15)
                        uchar* neighbor = curr_ptr + offsets[idx]; // Fast pointer arithmetic
                        if (*neighbor != BACKGROUND) {
                            // Found next boundary pixel
                            curr_ptr = neighbor;
                            int dir = (idx & 7);                             // Update Integer Coordinates using the small static tables(Use modulo 8 to get the distinct direction 0-7)
                            int next_x=curr_x+dx[dir], next_y=curr_y+dy[dir];
                            ntimesRevisited+= int(*neighbor == VISITED);
                            curr_x = next_x;curr_y = next_y;
                            search_idx = (dir + 5) & 7;
                            break;
                        }
                    }
                } while (curr_ptr != start_ptr );
                size_t bufsize=buffer.size();
                if (ntimesRevisited<= float(bufsize)*maxRevisited && bufsize >= minSize) {
                    contours.push_back(buffer);
                }
            }
            c++;//move to next pixel
            ////findEndContourPoint
            if ( row_ptr[c]){
                {
#if (CV_SIMD || CV_SIMD_SCALABLE)

                    cv::v_uint8 v_zero = cv::vx_setzero_u8();
                    for (; c <=  cols - cv::VTraits<cv::v_uint8>::vlanes(); c += cv::VTraits<cv::v_uint8>::vlanes())
                    {
                        cv::v_uint8 vmask = (cv::v_eq(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                        if (cv::v_check_any(vmask))
                        {
                            c += cv::v_scan_forward(vmask);
                            break;
                        }
                    }
#endif
                }
                for (; c < cols && row_ptr[c]; ++c) ;//last tail
            }
        }
    }
    return contours;
}

void MarkerDetector::thres255Adaptive(cv::Mat &in,cv::Mat &out,int off,int thres){
    cv::boxFilter( in, out, in.type(), cv::Size(off*2+1, off*2+1),
                  cv::Point(-1,-1), true, 4 );

    for(int i = 0; i < in.rows; i++ )
    {
        const uchar* sdata = in.ptr(i);
        uchar* ddata = out.ptr(i);
        for(int j = 0; j < in.cols; j++ )
            ddata[j] = ((ddata[j]-thres  )< sdata[j]) *255;
    }

}


//////////////////////////////////// BOARD

std::vector<FiducialMarker> detect(DictionaryType dictionary, cv::Mat & src_gray,cv::Mat & thresImage,   int erosionIt,    const DetectionParameters &params){

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(thresImage, thresImage, kernel,{-1,-1},erosionIt);
    return MarkerDetector::detect(src_gray,{dictionary},params,nullptr,thresImage);
}

int cornerMaxDistance(const FiducialMarker &m1,const FiducialMarker &m2)
{
    int md=0;
    for(int i=0;i<4;i++){
        int d=int((m1.corners[i].x-m2.corners[i].x)*(m1.corners[i].x-m2.corners[i].x)+(m1.corners[i].y-m2.corners[i].y)*(m1.corners[i].y-m2.corners[i].y));
        if(d>md) md=d;
    }
    return md;
}
std::vector<FiducialMarker> detectBWMarkers(cv::Mat &src_gray,DictionaryType dictionary,const DetectionParameters &params={}){


    std::vector<FiducialMarker>  markers_black,markers_white;
    cv::Mat thresImage;

    //BLACK MARKERS
    cv::boxFilter( src_gray, thresImage, src_gray.type(), cv::Size(25,25),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-src_gray;
    cv::threshold(thresImage, thresImage, 3, 255, cv::THRESH_BINARY);
    //determine how many erosion iterations we will do, depending on the size of the image
    int maxErodeIterations= std::max(2, int( (2.*src_gray.cols/2000.)+0.5));
     std::vector<std::vector<FiducialMarker>  > markers_blackv(maxErodeIterations);
    cv::Range range(1, maxErodeIterations);
    cv::parallel_for_(range, [&](const cv::Range& r) {
        for(int i=r.start;i<r.end;i++){
            cv::Mat thres=thresImage.clone();
            markers_blackv[i]=detect(dictionary,src_gray,thres,i,params);//black markers
            //because we have shrink the borders for black markers, we will expand the corners a bit from the center
            for(auto & marker:markers_blackv[i]){
                std::vector<cv::Point2f> newPoints;
                for(int j=0;j<4;j++){
                    int idx0=j;
                    int idx1=(j+2) % 4;
                    auto dif=marker.corners[idx0]-marker.corners[idx1];
                    auto norm=cv::norm(dif);
                    auto p= marker.corners[idx1]+  ((dif)/norm)*(norm+ 2+i   );
                    newPoints.push_back(p);
                }
                //replace the points by the new ones
                for(int j=0;j<4;j++){
                    marker.corners[j]=newPoints[j];
                }
            }
        }
    });

    //merge them all into markers_black
    for(const auto &v:markers_blackv){
        for(const auto &m:v){
            markers_black.push_back(m);
        }
    }
    //now, we need to merge the different markers
    if(markers_blackv.size()>1){

        std::vector<FiducialMarker> unduplicated;
        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(markers_black.begin(), markers_black.end(),[](const FiducialMarker &a,const FiducialMarker &b){return a.id<b.id;});
        std::vector<bool> toRemove(markers_black.size(), false);
        for (int i = 0; i < int(markers_black.size()) - 1; i++)
        {
            for (int j = i + 1; j < int(markers_black.size()) && !toRemove[i]; j++)
            {
                if (markers_black[i].id == markers_black[j].id )
                {
                    if( cornerMaxDistance(markers_black[i],markers_black[j])<100)
                        toRemove[i]=true;
                }
            }
        }
        //now move to DetectedMarkers these not marked for removal
        for (unsigned int i = 0; i < markers_black.size(); i++)
            if (!toRemove[i]) unduplicated.push_back(markers_black[i]);
        markers_black=unduplicated;
    }


    //    markers_black=detect(board.dictionary,src_gray,thresImage,erodeIterations);//black markers





    ///WHITE MARKERS
    cv::boxFilter( src_gray, thresImage, src_gray.type(), cv::Size(15,15),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-src_gray;
    cv::threshold(thresImage, thresImage, 3, 255, cv::THRESH_BINARY);


    //draw a line between opposite corners to break the continous contour of the white markers, which will help to detect them.
    for(auto m:markers_black){
        auto n02=m[2]-m[0];
        auto n02norm=cv::norm(n02);
        auto p0=m[0]-(n02/n02norm)* (n02norm/8);
        auto p1=m[0]+(n02/n02norm)* (n02norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
        p0=m[2]-(n02/n02norm)* (n02norm/8);
        p1=m[2]+(n02/n02norm)* (n02norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);

        //same for the other diagonal
        auto n13=m[3]-m[1];
        auto n13norm=cv::norm(n13);
        p0=m[1]-(n13/n13norm)* (n13norm/8);
        p1=m[1]+(n13/n13norm)* (n13norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
        p0=m[3]-(n13/n13norm)* (n13norm/8);
        p1=m[3]+(n13/n13norm)* (n13norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
    }
    thresImage=255-thresImage;
    cv::Mat src_gray_inv = 255 - src_gray;
    markers_white = detect(dictionary, src_gray_inv, thresImage, 1,params); // white markers


    // Combine results from both
    std::vector<FiducialMarker> allMarkers;
    allMarkers.reserve(markers_black.size() + markers_white.size());
    allMarkers.insert(allMarkers.end(), markers_black.begin(), markers_black.end());
    allMarkers.insert(allMarkers.end(), markers_white.begin(), markers_white.end());

    return allMarkers;
}
std::vector<std::vector<int>> getConnectedComponents(cv::Mat graph_8uc1) {
    int n = graph_8uc1.rows;
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> components;

    // Ensure the matrix is square
    if (n != graph_8uc1.cols) {
        return components;
    }

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            // Start of a new connected component
            std::vector<int> currentComponent;
            std::queue<int> q;

            q.push(i);
            visited[i] = true;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                currentComponent.push_back(u);

                // Check the row in the adjacency matrix for neighbors
                // Using ptr<uchar> for faster row access than .at<uchar>
                const uchar* rowPtr = graph_8uc1.ptr<uchar>(u);
                for (int v = 0; v < n; ++v) {
                    if (rowPtr[v] != 0 && !visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
            components.push_back(currentComponent);
        }
    }

    return components;
}

std::shared_ptr<cv::flann::Index> buildFlannIndex(const std::vector<FiducialMarker> &markers) {
    //create a vector with all corners
    std::vector<cv::Point2f> corners;
    corners.reserve(markers.size()*4);
    for(const auto &m:markers){
        for(const auto &c:m.corners){
            corners.push_back(c);
        }
    }
    //now, create a flann index with these corners
    cv::flann::KDTreeIndexParams indexParams(1);
    cv::Mat data = cv::Mat(corners).reshape(1, static_cast<int>(corners.size()));
    return std::make_shared<cv::flann::Index>(data, indexParams);
}
//returns a set of markers that are connected, i.e. that have at least one corner closer than a threshold.
std::vector<std::vector<FiducialMarker>> connectedMarkerComponents(const std::vector<FiducialMarker> &markers){
    //create a vector with all corners
    //now, create a flann index with these corners
    auto flannIndex=buildFlannIndex(markers);

    cv::Mat  graph(markers.size(), markers.size(), CV_8UC1, cv::Scalar(0));

    //now, for each corner, find the neighbors closer than a threshold
    const float threshold=10.0f;
    for(size_t m=0;m<markers.size();m++){
        for(int i=0;i<4;i++){
            std::vector<int> indices;
            std::vector<float> dists;
            cv::Mat query = (cv::Mat_<float>(1, 2) << markers[m][i].x, markers[m][i].y); // Single 2D query point
            int n=flannIndex->radiusSearch(query, indices, dists, threshold*threshold, 4*markers.size());
            for(int x=0;x<n;x++){
                int idx=indices[x];
                graph.at<uchar>(m,idx/4)=1;
                graph.at<uchar>(idx/4,m)=1;
            }
        }
    }

    //std::cout<<"Graph:\n"<<graph<<std::endl;
    //obtain the different connected components available
    auto ccomps=getConnectedComponents(graph);


    //now, for each component, obtain the corresponding markers
    std::vector<std::vector<FiducialMarker>> res;
    for(const auto &comp:ccomps){
        res.push_back({});
        for(auto idx:comp){
            res.back().push_back(markers[idx]);
        }
    }
    return res;
}


//given a marker id and one of its corners, return the global corner id of that corner, which is a unique id for that corner in the whole board,

int getGlobalCornerID(int marker_id, int corner_id,const  cv::Size &bSize,const std::vector<int> &ids)
{


    auto getIdPos=[&](int id) -> std::pair<int, int>
    {
        auto it=std::find(ids.begin(),ids.end(),id);
        if(it==ids.end()) return {-1,-1};
        int idx=std::distance(ids.begin(),it);
        int row = idx / bSize.width;
        int col = idx % bSize.width;
        return {row, col};
    };

    //obtain the row, col of the marker_id
    auto row_col=getIdPos(marker_id);
    if(corner_id<=1){
        return (bSize.width+1) *row_col.first +  row_col.second+  corner_id;
    }
    else if(corner_id==2){
        return (bSize.width+1) *(row_col.first+1) +  row_col.second+ 1;
    }
    else  {
        return (bSize.width+1) *(row_col.first+1) +  row_col.second;

    }
}

//opposite of getGlobalCornerID, given a global corner id, return the marker ids and corner ids of that corner

std::vector<std::pair<int,int>> getMarkerCornersFromGlobalCornerID( int gid,cv::Size bSize,const std::vector<int> &ids)
{
    auto getId=[&](int row, int col)->int
    {
        if(row<0 || row>=bSize.height || col<0 || col>=bSize.width) return -1;
        int idx=row*bSize.width+col;
        return ids[idx];
    };
    std::vector<std::pair<int,int>> result;
    const int W = bSize.width;

    // Decompose gid into (cr, cc) in the (H+1) x (W+1) corner grid.
    // Inverse of  gid = (W+1) * cr + cc  used in getGlobalCornerID.
    const int cr = gid / (W + 1);
    const int cc = gid % (W + 1);

    // Up to 4 markers share a global corner. Sort order: 0=TL, 1=TR, 2=BR, 3=BL.
    // Board.getId() returns -1 for out-of-range (row,col), so it doubles as a bounds check.
    int id;

    // Marker at (cr, cc) sees this point as its top-left (0)
    id = getId(cr,     cc);
    if (id != -1) result.emplace_back(id, 0);

    // Marker at (cr, cc-1) sees this point as its top-right (1)
    id = getId(cr,     cc - 1);
    if (id != -1) result.emplace_back(id, 1);

    // Marker at (cr-1, cc-1) sees this point as its bottom-right (2)
    id = getId(cr - 1, cc - 1);
    if (id != -1) result.emplace_back(id, 2);

    // Marker at (cr-1, cc) sees this point as its bottom-left (3)
    id = getId(cr - 1, cc);
    if (id != -1) result.emplace_back(id, 3);

    return result;
}

}

namespace cv {
namespace aruco2 {

std::vector<FiducialMarker> detectFiducialMarkers(InputArray image,const std::vector<DictionaryType> &dictionaries,const DetectionParameters &detectorParams){
    CV_Assert(!image.empty());
    if (image.isUMat() && cv::ocl::useOpenCL()) {
        return MarkerDetector::detect(image.getUMat(), dictionaries, detectorParams);
    }
    return MarkerDetector::detect(image.getMat(), dictionaries, detectorParams);
}
std::vector<FiducialMarker> detectFiducialMarkers(InputArray image,DictionaryType dictionary,const DetectionParameters &detectorParams){
    return detectFiducialMarkers(image,std::vector<DictionaryType>{dictionary},detectorParams);
}



void getFiducialMarkerImage(OutputArray _img, DictionaryType dictionary, int id, int bitSize, bool externalBorder){

    //assert marker size >sidePixels
    auto dict=getPredefinedDictionary(dictionary);
    CV_Assert(bitSize > 0);

    cv::Mat marker;
    dict.getMarkerBits(id).convertTo(marker, CV_8UC1, 255.0);
    cv::Mat markerWithBorders;
    if(externalBorder){
        markerWithBorders.create(marker.rows+4,marker.cols+4,CV_8UC1);
        markerWithBorders=255;
        markerWithBorders(cv::Rect(1,1,marker.cols+2,marker.rows+2))=0;
        cv::Rect roi(2,2,marker.cols,marker.rows);
        marker.copyTo(markerWithBorders(roi));
    }
    else{
        markerWithBorders.create(marker.rows+2,marker.cols+2,CV_8UC1);
        markerWithBorders=0;
        cv::Rect roi(1,1,marker.cols,marker.rows);
        marker.copyTo(markerWithBorders(roi));
    }
    int sidePixels=markerWithBorders.cols*bitSize;
    _img.create(sidePixels, sidePixels, CV_8UC1);
     cv::resize(markerWithBorders, _img, cv::Size(sidePixels, sidePixels), 0, 0, cv::INTER_NEAREST);
 }

void drawFiducialMarkers(InputOutputArray _image, const std::vector<FiducialMarker> &markers, Scalar borderColor) {
    cv::Mat image = _image.getMat();
    Scalar cornerColor(255 - borderColor[0], borderColor[1], borderColor[2]);
    Scalar textColor  (255 - borderColor[0], 255 - borderColor[1], 255 - borderColor[2]);

    int thickness= 0.5+ std::max(float(3),3 * float(image.cols)/float(1920.));//adjust thickness to image dimensions
    for (const auto &marker : markers) {
        if (marker.corners.size() != 4) continue;
        // draw 4 sides
        for (int j = 0; j < 4; j++)
            cv::line(image, cv::Point(marker.corners[j]), cv::Point(marker.corners[(j+1)%4]), borderColor, thickness);
        // highlight first corner to show orientation
        cv::circle(image, cv::Point(marker.corners[0]), 3, cornerColor, -1);
        // draw id at the marker center
        cv::Point2f center(0, 0);
        for (const auto &c : marker.corners) center += c;
        center *= 0.25f;
        cv::putText(image, std::to_string(marker.id), cv::Point(center),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, thickness);
    }
}


void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
              InputArray rvec, InputArray tvec, float length) {
    std::vector<cv::Point3f> axisPoints = {
        {0.f, 0.f, 0.f}, {length, 0.f, 0.f}, {0.f, length, 0.f}, {0.f, 0.f, length}
    };
    std::vector<cv::Point2f> projected;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, projected);
    cv::Mat img = image.getMat();
    cv::line(img, projected[0], projected[1], Scalar(0,   0, 255), 2); // X red
    cv::line(img, projected[0], projected[2], Scalar(0, 255,   0), 2); // Y green
    cv::line(img, projected[0], projected[3], Scalar(255, 0,   0), 2); // Z blue
}

void getSolvePnpPoints(const FiducialMarker &marker, OutputArray objPoints, OutputArray imgPoints, float markerSize  ){
    std::vector<cv::Point3f> markerCorners={ {-markerSize/2.f,markerSize/2.f,0.f},{markerSize/2.f,markerSize/2.f,0.f},{markerSize/2.f,-markerSize/2.f,0.f},{-markerSize/2.f,-markerSize/2.f,0.f}};

    cv::Mat(markerCorners).copyTo(objPoints);
    cv::Mat(marker.corners).copyTo(imgPoints);
}



bool detectGridBoard(InputArray image, cv::Size gridSize, DictionaryType dictionary,
            CV_OUT GridBoard &board_, InputArray _ids){

    std::vector<int> ids;
    _ids.copyTo(ids);

    CV_Assert(gridSize.width > 0 && gridSize.height > 0  );
    CV_Assert(image.channels() == 1 || image.channels() == 3);
    //if ids is empty, we will consider all the ids in the dictionary up to gridSize.width*gridSize.height
    if(ids.empty()){
        int nMarkers=gridSize.width*gridSize.height;
        for(int i=0;i<nMarkers;i++){
            ids.push_back(i);
        }
    }

    //obtain the gray image
    cv::Mat src_gray;
    if(image.channels()==3)
        cvtColor(image, src_gray, cv::COLOR_BGR2GRAY);
    else src_gray=image.getMat();

    //detect all markers
    auto allMarkers=detectBWMarkers( src_gray,dictionary);
    //remove markers not belonging to the list of ids of the board
    allMarkers.erase(std::remove_if(allMarkers.begin(), allMarkers.end(),
                                    [ids](const FiducialMarker &m) {
                                        return std::find(ids.begin(), ids.end(), m.id) == ids.end();
                                    }), allMarkers.end());


    if(allMarkers.empty())return false;



    //obtain the connected components
    std::vector<std::vector<FiducialMarker> > connected_markers=connectedMarkerComponents(allMarkers);

    //lets detect possible inconsistencies, i.e., markers in the wrong order, possibly belonging to another board configuration
    std::vector<std::vector<FiducialMarker> > consistent_connected_markers;
    for(auto &comp:connected_markers){
        int threshold=10;
        auto findex=buildFlannIndex(comp);
        bool is_consistent=true;
        //for each corner, of each marker we will analyze its nearst neighbor. If is really connected, we will check if
        //the connection is consistent
        for(const auto &marker:comp){
            for(int c=0;c<4;c++){
                std::vector<int> indices;
                std::vector<float> dists;
                cv::Mat query = (cv::Mat_<float>(1, 2) << marker[c].x, marker[c].y); // Single 2D query point
                int nn=findex->radiusSearch(query, indices, dists, threshold*threshold, marker.size());
                nn=std::min(nn,int(indices.size()));//flann bug in <4.13
                for(int ix=0;ix<nn;ix++){
                    int idx=indices[ix];
                    if(comp[idx/4].id==marker.id) continue;//same marker
                    //check if the connection is consistent, i.e., if the global corner ids are the same
                    int gid1=getGlobalCornerID(marker.id,c,gridSize,ids);
                    int gid2=getGlobalCornerID(comp[idx/4].id,idx%4,gridSize,ids);
                    if(gid1!=gid2){
                        CV_LOG_WARNING(nullptr, "Marker " << marker.id << " corner " << c << " connected to marker " << comp[idx/4].id << " corner " << (idx%4) << " but global corner ids differ: " << gid1 << " vs " << gid2);
                        is_consistent=false;
                    }
                }
            }
            if(!is_consistent) break;
        }
        if(is_consistent)
            consistent_connected_markers.push_back(comp);
    }

    if(consistent_connected_markers.empty())return false;

    //select the one with more markers belonging to the board (at least 2)
    std::vector<FiducialMarker> connected_markers_filtered;
    for(const auto &comp:consistent_connected_markers){
        size_t validMarkers=0;
        for(const auto &m:comp){
            if(std::find(ids.begin(),ids.end(),m.id)!=ids.end()){
                validMarkers++;
            }
        }
        if(validMarkers>connected_markers_filtered.size() && validMarkers>=2){
            connected_markers_filtered=comp;
        }
    }


    if(connected_markers_filtered.empty()) return false;
    allMarkers=connected_markers_filtered;

    //now, lets focus on the global corners. Unify the corners for subpixel analysis
    //first, obtain the average position for each global corner id found
    std::map<int,std::pair<cv::Point2f,int> > global_corners;
    for(const auto &m:allMarkers){
        for(int c=0;c<4;c++){
            int gid=getGlobalCornerID(m.id,c,gridSize,ids);
            if (global_corners.find(gid)==global_corners.end())
                global_corners[gid]={m[c],1};
            else{
                auto &p=global_corners[gid];
                p.first+=m[c];
                p.second++;
            }
        }
    }
    //now, average
    for(auto &gc:global_corners)
        gc.second.first=gc.second.first / float(gc.second.second);

    //////  subpixel corner refinement
    // compute the window size for the subpixel refinement based on the size of the markers. Bigger makers, bigger search zone
    double avrgLen=0;
    for(const auto &m:allMarkers){
        for(int i=0;i<4;i++){
            avrgLen+=cv::norm(m[i]-m[(i+1)%4]);
        }
    }
    avrgLen/=4*allMarkers.size();
    //here is the formula to compute the half window size
    int halfwsize= std::min(int(3* std::max(1.f,float(avrgLen)/float(34) )),9);

    //add the global corners to a single vector
    std::vector<cv::Point2f> Corners;
    for (const auto &m:global_corners)
        Corners.push_back(m.second.first);
    //refine the corners
    cv::cornerSubPix(src_gray, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
    // copy back to the global corners
    int idex=0;
    for(auto &gc:global_corners){
        gc.second.first=Corners[idex++];
    }

    //now, assign the refined global corner positions back to the markers
    for(auto it_gc = global_corners.begin(); it_gc != global_corners.end(); ++it_gc){
        int gid = it_gc->first;
        cv::Point2f corner_c = it_gc->second.first;
        //find in how many markers it is involved
        auto marker_corners = getMarkerCornersFromGlobalCornerID(gid,gridSize,ids);
        for(auto it_mc = marker_corners.begin(); it_mc != marker_corners.end(); ++it_mc){
            int markerid = it_mc->first;
            int cornerid = it_mc->second;
            //see if the makrer id is detected
            auto markerid_copy=markerid;//capturing markerid only in C++20, lets use a copy for prev compilers
            auto it=std::find_if(allMarkers.begin(),allMarkers.end(),[markerid_copy](const FiducialMarker &m){return m.id==markerid_copy;});
            if(it!=allMarkers.end()){
                (*it).corners[cornerid]=corner_c;
            }
        }
    }


    //FINALLY. Populate board with the results

    board_.gridSize=gridSize;
    board_.dictionary=dictionary;
    board_.markers=allMarkers;

    //copy results to output arrays
    std::vector<int> gidv;
    std::vector<cv::Point2f> gcorners;
    for(auto it_gc = global_corners.begin(); it_gc != global_corners.end(); ++it_gc){
        board_.detectedBoardCorners.push_back({it_gc->first,it_gc->second.first});
        // gidv.push_back(it_gc->first);
        // gcorners.push_back(it_gc->second.first);
    }

     return true;
}


void getGridBoardImage(OutputArray img, Size bSize, DictionaryType dictionary,
                        int bitSize , InputArray _ids  ){
    std::vector<int> ids;
    _ids.copyTo(ids);
    auto dictInstance=getPredefinedDictionary(dictionary);
    int nmarkers=bSize.area();
    if(nmarkers > dictInstance.bytesList.rows)
        CV_Error(cv::Error::StsBadArg, "Number of markers exceeds the number of markers in the dictionary");
    CV_Assert(ids.empty() || (int)ids.size() == nmarkers);
    //if ids is empty, lets populate it
    if(ids.empty()){
        for(int i=0;i<nmarkers;i++){
            ids.push_back(i);
        }
    }

    int markerSizePix=dictInstance.markerSize *bitSize;
    int border=markerSizePix/4;
    cv::Size imgSize(markerSizePix*bSize.width + 2*border , markerSizePix*bSize.height+2*border);



    int markerIdx=0;
    cv::Mat outImage(imgSize, CV_8UC1,cv::Scalar(255));
    int startLineColor = 0;
    for(int y=0; y<bSize.height; y++){
        int curMarkerColor=startLineColor;
        for(int x= 0; x<bSize.width; x++,markerIdx++){
            cv::Mat markerImg;
            dictInstance.generateImageMarker(ids[markerIdx], markerSizePix, markerImg);
            int posX = x * markerSizePix + border;
            int posY = y * markerSizePix + border;
            if(curMarkerColor)
                markerImg=255-markerImg;
            markerImg.copyTo(outImage(cv::Rect(posX, posY, markerSizePix, markerSizePix)));
            curMarkerColor= curMarkerColor==1?0:1;
        }
        startLineColor=startLineColor==1?0:1;
    }

    for(int x=1;x<bSize.width;x+=2)
        cv::rectangle(outImage,cv::Rect(border+x*markerSizePix,0,markerSizePix,border),cv::Scalar::all(0),cv::FILLED);
    for(int x=bSize.height%2;x<bSize.width;x+=2)
        cv::rectangle(outImage,cv::Rect(border+x*markerSizePix,border+markerSizePix*bSize.height,markerSizePix,border),cv::Scalar::all(0),cv::FILLED);
    for(int y=1;y<bSize.height;y+=2)
        cv::rectangle(outImage,cv::Rect(0,border+y*markerSizePix,border,markerSizePix),cv::Scalar::all(0),cv::FILLED);
    for(int y=bSize.width%2;y<bSize.height;y+=2)
        cv::rectangle(outImage,cv::Rect(border+markerSizePix*bSize.width,border+y*markerSizePix,border,markerSizePix),cv::Scalar::all(0),cv::FILLED);

    cv::rectangle(outImage,cv::Rect(0,0,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(outImage.cols-border,0,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(0,outImage.rows-border,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(outImage.cols-border,outImage.rows-border,border,border),cv::Scalar::all(0),cv::FILLED);
    //copy result to img
    outImage.copyTo(img);

}
  void drawGridBoard(InputOutputArray image, const GridBoard &board,
                       Scalar color,bool drawMarkerIds ){
      for(size_t i=0;i<board.detectedBoardCorners.size();i++){
          //draw a circle for each detected corner
          cv::circle(image,board.detectedBoardCorners[i].second,3,color,-1);
          //draw id

          std::stringstream s;
          s << "id=" << board.detectedBoardCorners[i].first;
          putText(image, s.str(), board.detectedBoardCorners[i].second-cv::Point2f(15,10) , FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
      }

      if(drawMarkerIds){
          for(auto m:board.markers){
              cv::Point2f center(0,0);
              for(const auto &c:m.corners) center+=c;
              center*=0.25f;
              std::stringstream s;
              s << m.id;
              putText(image, s.str(), center , FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255)-color, 2);

          }

      }

  }

  void getSolvePnpPoints(const GridBoard &board, OutputArray objPoints, OutputArray imgPoints, float markerSize  ){

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    for(const auto &gc:board.detectedBoardCorners){
        int gid=gc.first;
        int row = gid / (board.gridSize.width+1);
        int col = gid % (board.gridSize.width+1);
        cv::Point3f p3d;
        p3d.x = col * markerSize;
        p3d.y = row * markerSize;
        p3d.z = 0;
        objectPoints.push_back(p3d);
        imagePoints.push_back(gc.second);
    }
    cv::Mat(objectPoints).copyTo(objPoints);
    cv::Mat(imagePoints).copyTo(imgPoints);
  }

  void getDiamondImage(OutputArray img,const DictionaryType &dictionary, const cv::Vec4i &ids,int bitSize){
        getGridBoardImage(img,cv::Size(2,2),dictionary,bitSize,ids);
  }

   std::vector<Diamond> detectDiamonds(InputArray image, DictionaryType dictionary  ){
       CV_Assert(image.channels() == 1 || image.channels() == 3);
       cv::Mat src_gray;
       //obtain the gray image
       if(image.channels()==3)
           cvtColor(image, src_gray, cv::COLOR_BGR2GRAY);
       else src_gray=image.getMat();

       //detect all markers
       auto allMarkers=detectBWMarkers( src_gray,dictionary);

       if(allMarkers.empty())return {};
       //obtain the connected components
       std::vector<std::vector<FiducialMarker> > connected_markers=connectedMarkerComponents(allMarkers);

       //discard these that have more or less than 4 elements

       std::vector<Diamond> diamonds_detected;
       int threshold=10*10;
       std::vector<int> indices;
       std::vector<float> dists;
       cv::Mat query(1,2,CV_32F);


       for(const auto &comp:connected_markers){
           if(comp.size()!=4) continue;
           //make sure they   share 3 corners
           auto flannIndex=buildFlannIndex(comp);
           int totalSharedCorners=0,cornerWithMostSharedCorners=-1;
           std::pair<int,int> centralCorner;//this is the central poisition, that let us know the order of the markers in the diamond
           for(size_t m=0;m<comp.size();m++){
               auto &marker=comp[m];
               for(int c=0;c<4;c++){
                   query.ptr<float>(0)[0]=marker[c].x;
                   query.ptr<float>(0)[1]=marker[c].y;
                   int nn=flannIndex->radiusSearch(query, indices, dists, threshold, 4);
                   nn=std::min(int(indices.size()),nn);
                   //  std::cout<<"nn="<<nn<<" for marker "<<marker.id<<" corner "<<c<<std::endl;
                   totalSharedCorners+=nn-1;
                   if( nn>cornerWithMostSharedCorners)
                   {
                       cornerWithMostSharedCorners=nn;
                       centralCorner={m,c};
                   }
               }
           }
           // std::cout<<"Component with markers "<<comp[0].id<<","<<comp[1].id<<","<<comp[2].id<<","<<comp[3].id<<" has "<<totalSharedCorners<<" shared corners"<<std::endl;
           // std::cout<<"Central corner is marker "<<centralCorner.first<<" corner "<<centralCorner.second<<" with "<<cornerWithMostSharedCorners<<" shared corners"<<std::endl;
           if(totalSharedCorners!=20) continue;

           //find the diamon id eximiing the poition of the central corner, which is the one with more shared corners.
           cv::Vec4i diamondIds;
           query.ptr<float>(0)[0]=comp[centralCorner.first][centralCorner.second].x;
           query.ptr<float>(0)[1]=comp[centralCorner.first][centralCorner.second].y;
           int nn=flannIndex->radiusSearch(query, indices, dists, threshold, 4);
           for(int i=0;i<nn;i++){

               int markerIdx=indices[i]/4;
               int cornerIdx=indices[i]%4;

               int idIndex;
               if( cornerIdx==2) idIndex=0;
               else if(cornerIdx==3) idIndex=1;
               else if(cornerIdx==1) idIndex=2;
               else idIndex=3;
               diamondIds[idIndex]=comp[markerIdx].id;

           }

           std::vector<cv::Point2f> board_corners(9);
           for(size_t m=0;m<comp.size();m++){
               for(int c=0;c<4;c++){
                   auto gid=getGlobalCornerID(comp[m].id,c, cv::Size(2,2),std::vector<int>{diamondIds[0],diamondIds[1],diamondIds[2],diamondIds[3]} );
                   board_corners[gid]=comp[m][c];
               }
           }

           //lets apply a corner refinement to the corners. first, estimate average lenght

           // compute the window size for the subpixel refinement based on the size of the markers. Bigger makers, bigger search zone
           double avrgLen=0;
           for(auto m:comp){
               for(int i=0;i<4;i++){
                   avrgLen+=cv::norm(m[i]-m[(i+1)%4]);
               }
           }
           avrgLen/=4*comp.size();
           //here is the formula to compute the half window size
           int halfwsize= std::min(int(3* std::max(1.f,float(avrgLen)/float(34) )),9);
           //now, subpix
           cv::cornerSubPix(src_gray, board_corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));



           Diamond _tdiamond;
           _tdiamond.id=diamondIds;
           _tdiamond.dictionary=dictionary;
           _tdiamond.markers=comp;
           _tdiamond.corners=board_corners;
           diamonds_detected.push_back(_tdiamond);
       }
       return diamonds_detected;
   }


   void drawDiamonds(InputOutputArray image, const std::vector<Diamond> &diamonds, Scalar color ,bool drawMarkerIds){
       for(const auto &d:diamonds){

           //draw lines between corners
           cv::line(image,cv::Point(d.corners[0]),cv::Point(d.corners[2]),color,2);
           cv::line(image,cv::Point(d.corners[2]),cv::Point(d.corners[8]),color,2);
           cv::line(image,cv::Point(d.corners[8]),cv::Point(d.corners[6]),color,2);
           cv::line(image,cv::Point(d.corners[6]),cv::Point(d.corners[0]),color,2);
           cv::Point2f center(0,0);
           for(size_t i=0;i<d.corners.size();i++){
               //draw a rect
               cv::rectangle(image,cv::Rect(d.corners[i]-cv::Point2f(3,3),cv::Size(6,6)),color,-1);
               center+=d.corners[i];
           }
           center/=float(d.corners.size());
           //draw its id (i)
           std::stringstream s;
           s << "id=" << d.id;
           putText(image, s.str(),center -cv::Point2f(50,10), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);



           if(drawMarkerIds){
               for(auto m:d.markers){
                   std::stringstream ss;
                   ss <<   m.id;
                   cv::Point2f m_center(0,0);
                   for(const auto &c:m.corners) m_center+=c;
                   m_center*=0.25f;
                   putText(image, ss.str(), m_center , FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255)-color, 2);
               }
           }
       }
   }
   void getSolvePnpPoints(const Diamond &diamond, OutputArray objPoints, OutputArray imgPoints, float markerSize  ){

       std::vector<cv::Point3f> objectPoints;
       std::vector<cv::Point2f> imagePoints;

       int idx=0;
       for(int row=0;row<3;row++){
           for(int col=0;col<3;col++,idx++){
               cv::Point3f p3d;
               p3d.x = (col - 1) * markerSize;
               p3d.y = (1 - row) * markerSize;
               p3d.z = 0;
               objectPoints.push_back(p3d);
               imagePoints.push_back(diamond.corners[idx]);
           }
       }

       cv::Mat(objectPoints).copyTo(objPoints);
       cv::Mat(imagePoints).copyTo(imgPoints);
   }
   std::vector<cv::aruco2::FiducialMarker> detectRArucoMarkers(InputArray image, cv::aruco2::DictionaryType dictionary ,
                                                               const cv::aruco2::DetectionParameters &detectorParams ){

       DetectionParameters params(detectorParams);
       params.gridBitSampling=true;//read the borders of the bit instead of the center
       params.detectColorMode=2;//detects both bw and wb markers
       return  detectFiducialMarkers(  image,dictionary,params);

   }
   void getRArucoMarkerImage(OutputArray img, cv::aruco2::DictionaryType dictionary , int id,int depth, int bitSize ,int innerBorders, bool externalBorder){
       CV_Assert(depth>=1);
       CV_Assert(innerBorders>=1);


       //first, create the canonical image
       std::vector<cv::Mat> canonicalImage(depth);
       getFiducialMarkerImage(canonicalImage[0],dictionary,id,1,true);

       int ntr=canonicalImage[0].rows+2*(innerBorders-1);
       int ntc=canonicalImage[0].cols+2*(innerBorders-1);
       cv::Mat bordered (ntr,ntc,canonicalImage[0].type(),cv::Scalar(255));
       cv::Mat roi=bordered(cv::Rect(innerBorders-1,innerBorders-1,canonicalImage[0].cols,canonicalImage[0].rows));
       canonicalImage[0].copyTo(roi);
       canonicalImage[0]=bordered;

       for(int d=1;d<depth;d++){
           cv::resize(canonicalImage[0],canonicalImage[d], {canonicalImage[d-1].cols*canonicalImage[0].cols,canonicalImage[d-1].rows*canonicalImage[0].rows},0,0,cv::INTER_NEAREST);
           //lets insert the markers in the islands
           for(int r=1+innerBorders;r<canonicalImage[0].rows-1-innerBorders;r++)
               for(int c=1+innerBorders;c<canonicalImage[0].cols-1-innerBorders;c++){
                   int x=c*canonicalImage[d-1].cols;
                   int y=r*canonicalImage[d-1].rows;
                   //copy the previous image into the island
                   cv::Mat subRoi=canonicalImage[d](cv::Rect(x,y,canonicalImage[d-1].cols,canonicalImage[d-1].rows));
                   cv::Mat imToCopy;
                   canonicalImage[d-1].copyTo(imToCopy);
                   if( canonicalImage[0].at<uchar>( {c,r})==0)
                       imToCopy=255-imToCopy;
                   imToCopy.copyTo(subRoi);

               }
       }

       cv::Mat finalCanonical;
       //lets remove the extra external borders of the final canonical image
       int bRowSize=canonicalImage.back().rows/canonicalImage[0].rows;
       int bColSize=canonicalImage.back().cols/canonicalImage[0].cols;
       int nRemoveBorder=innerBorders-int(externalBorder);
       canonicalImage.back().rowRange(nRemoveBorder*bRowSize,canonicalImage.back().rows-nRemoveBorder*bRowSize).colRange(nRemoveBorder*bColSize,canonicalImage.back().cols-nRemoveBorder*bColSize).copyTo(finalCanonical);
       //save
       cv::resize(finalCanonical,img, {finalCanonical.cols*bitSize,finalCanonical.rows*bitSize},0,0,cv::INTER_NEAREST);


   }

   } // namespace aruco2
   } // namespace cv
