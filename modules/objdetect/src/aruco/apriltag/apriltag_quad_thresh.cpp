// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2013-2016, The Regents of The University of Michigan.
//
// This software was developed in the APRIL Robotics Lab under the
// direction of Edwin Olson, ebolson@umich.edu. This software may be
// available under alternative licensing terms; contact the address above.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the Regents of The University of Michigan.

// limitation: image size must be <32768 in width and height. This is
// because we use a fixed-point 16 bit integer representation with one
// fractional bit.

#include "../../precomp.hpp"
#include "apriltag_quad_thresh.hpp"

//#define APRIL_DEBUG
#ifdef APRIL_DEBUG
    #include "opencv2/imgcodecs.hpp"
    #include <opencv2/imgproc.hpp>
#endif

namespace cv {
namespace aruco {

static void ptsort_(struct pt *pts, int sz); // forward delaration

static inline
void ptsort(struct pt *pts, int sz)
{
#define MAYBE_SWAP(arr,apos,bpos)                                   \
        if (arr[apos].theta > arr[bpos].theta) {                        \
            tmp = arr[apos]; arr[apos] = arr[bpos]; arr[bpos] = tmp;    \
        };

    if (sz <= 1)
        return;

    if (sz == 2) {
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    // NB: Using less-branch-intensive sorting networks here on the
    // hunch that it's better for performance.
    if (sz == 3) { // 3 element bubble sort is optimal
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    if (sz == 4) { // 4 element optimal sorting network.
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1); // sort each half, like a merge sort
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 0, 2); // minimum value is now at 0.
        MAYBE_SWAP(pts, 1, 3); // maximum value is now at end.
        MAYBE_SWAP(pts, 1, 2); // that only leaves the middle two.
        return;
    }

    if (sz == 5) {
        // this 9-step swap is optimal for a sorting network, but two
        // steps slower than a generic sort.
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1); // sort each half (3+2), like a merge sort
        MAYBE_SWAP(pts, 3, 4);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 0, 3); // minimum element now at 0
        MAYBE_SWAP(pts, 2, 4); // maximum element now at end
        MAYBE_SWAP(pts, 1, 2); // now resort the three elements 1-3.
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 1, 2);
        return;
    }

#undef MAYBE_SWAP

    ptsort_(pts, sz);
}

void ptsort_(struct pt *pts, int sz)
{
    // a merge sort with temp storage.

    // Use stack storage if it's not too big.
    cv::AutoBuffer<struct pt, 1024> _tmp_stack(sz);
    memcpy(_tmp_stack.data(), pts, sizeof(struct pt) * sz);

    int asz = sz/2;
    int bsz = sz - asz;

    struct pt *as = &_tmp_stack[0];
    struct pt *bs = &_tmp_stack[asz];

    ptsort(as, asz);
    ptsort(bs, bsz);

#define MERGE(apos,bpos)                        \
    if (as[apos].theta < bs[bpos].theta)        \
        pts[outpos++] = as[apos++];             \
    else                                        \
        pts[outpos++] = bs[bpos++];

    int apos = 0, bpos = 0, outpos = 0;
    while (apos + 8 < asz && bpos + 8 < bsz) {
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
    }

    while (apos < asz && bpos < bsz) {
        MERGE(apos,bpos);
    }

    if (apos < asz)
        memcpy(&pts[outpos], &as[apos], (asz-apos)*sizeof(struct pt));
    if (bpos < bsz)
        memcpy(&pts[outpos], &bs[bpos], (bsz-bpos)*sizeof(struct pt));

#undef MERGE
}

/**
 * lfps contains *cumulative* moments for N points, with
 * index j reflecting points [0,j] (inclusive).
 * fit a line to the points [i0, i1] (inclusive). i0, i1 are both (0, sz)
 * if i1 < i0, we treat this as a wrap around.
 */
void fit_line(struct line_fit_pt *lfps, int sz, int i0, int i1, double *lineparm, double *err, double *mse){
    CV_Assert(i0 != i1);
    CV_Assert(i0 >= 0 && i1 >= 0 && i0 < sz && i1 < sz);

    double Mx, My, Mxx, Myy, Mxy, W;
    int N; // how many points are included in the set?

    if (i0 < i1) {
        N = i1 - i0 + 1;

        Mx  = lfps[i1].Mx;
        My  = lfps[i1].My;
        Mxx = lfps[i1].Mxx;
        Mxy = lfps[i1].Mxy;
        Myy = lfps[i1].Myy;
        W   = lfps[i1].W;

        if (i0 > 0) {
            Mx  -= lfps[i0-1].Mx;
            My  -= lfps[i0-1].My;
            Mxx -= lfps[i0-1].Mxx;
            Mxy -= lfps[i0-1].Mxy;
            Myy -= lfps[i0-1].Myy;
            W   -= lfps[i0-1].W;
        }

    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        CV_Assert(i0 > 0);

        Mx  = lfps[sz-1].Mx   - lfps[i0-1].Mx;
        My  = lfps[sz-1].My   - lfps[i0-1].My;
        Mxx = lfps[sz-1].Mxx  - lfps[i0-1].Mxx;
        Mxy = lfps[sz-1].Mxy  - lfps[i0-1].Mxy;
        Myy = lfps[sz-1].Myy  - lfps[i0-1].Myy;
        W   = lfps[sz-1].W    - lfps[i0-1].W;

        Mx  += lfps[i1].Mx;
        My  += lfps[i1].My;
        Mxx += lfps[i1].Mxx;
        Mxy += lfps[i1].Mxy;
        Myy += lfps[i1].Myy;
        W   += lfps[i1].W;

        N = sz - i0 + i1 + 1;
    }

    CV_Assert(N >= 2);

    double Ex = Mx / W;
    double Ey = My / W;
    double Cxx = Mxx / W - Ex*Ex;
    double Cxy = Mxy / W - Ex*Ey;
    double Cyy = Myy / W - Ey*Ey;

    double nx, ny;

    if (1) {
        // on iOS about 5% of total CPU spent in these trig functions.
        // 85 ms per frame on 5S, example.pnm
        //
        // XXX this was using the double-precision atan2. Was there a case where
        // we needed that precision? Seems doubtful.
        float normal_theta = float(.5f * (CV_PI / 180)) * cv::fastAtan2((float)(-2*Cxy), (float)(Cyy - Cxx));
        nx = cosf(normal_theta);
        ny = sinf(normal_theta);
    } else {
        // 73.5 ms per frame on 5S, example.pnm
        double ty = -2*Cxy;
        double tx = (Cyy - Cxx);
        double mag = ty*ty + tx*tx;

        if (mag == 0) {
            nx = 1;
            ny = 0;
        } else {
            float norm = sqrtf((float)(ty*ty + tx*tx));
            tx /= norm;

            // ty is now sin(2theta)
            // tx is now cos(2theta). We want sin(theta) and cos(theta)

            // due to precision err, tx could still have slightly too large magnitude.
            if (tx > 1) {
                ny = 0;
                nx = 1;
            } else if (tx < -1) {
                ny = 1;
                nx = 0;
            } else {
                // half angle formula
                ny = sqrtf((1.0f - (float)tx)*0.5f);
                nx = sqrtf((1.0f + (float)tx)*0.5f);

                // pick a consistent branch cut
                if (ty < 0)
                    ny = - ny;
            }
        }
    }

    if (lineparm) {
        lineparm[0] = Ex;
        lineparm[1] = Ey;
        lineparm[2] = nx;
        lineparm[3] = ny;
    }

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    if (err)
        *err = nx*nx*N*Cxx + 2*nx*ny*N*Cxy + ny*ny*N*Cyy;

    // mean squared error
    if (mse)
        *mse = nx*nx*Cxx + 2*nx*ny*Cxy + ny*ny*Cyy;
}

int err_compare_descending(const void *_a, const void *_b){
    const double *a =  (const double*)_a;
    const double *b =  (const double*)_b;

    return ((*a) < (*b)) ? 1 : -1;
}

/**
  1. Identify A) white points near a black point and B) black points near a white point.

  2. Find the connected components within each of the classes above,
  yielding clusters of "white-near-black" and
  "black-near-white". (These two classes are kept separate). Each
  segment has a unique id.

  3. For every pair of "white-near-black" and "black-near-white"
  clusters, find the set of points that are in one and adjacent to the
  other. In other words, a "boundary" layer between the two
  clusters. (This is actually performed by iterating over the pixels,
  rather than pairs of clusters.) Critically, this helps keep nearby
  edges from becoming connected.
 **/
int quad_segment_maxima(const DetectorParameters &td, int sz, struct line_fit_pt *lfps, int indices[4]){

    // ksz: when fitting points, how many points on either side do we consider?
    // (actual "kernel" width is 2ksz).
    //
    // This value should be about: 0.5 * (points along shortest edge).
    //
    // If all edges were equally-sized, that would give a value of
    // sz/8. We make it somewhat smaller to account for tags at high
    // aspects.

    // XXX Tunable. Maybe make a multiple of JPEG block size to increase robustness
    // to JPEG compression artifacts?
    //int ksz = imin(20, sz / 12);
    int ksz = 20 < (sz/12)? 20: (sz/12);

    // can't fit a quad if there are too few points.
    if (ksz < 2)
        return 0;

    //    printf("sz %5d, ksz %3d\n", sz, ksz);

    std::vector<double> errs(sz);

    for (int i = 0; i < sz; i++) {
        fit_line(lfps, sz, (i + sz - ksz) % sz, (i + ksz) % sz, NULL, &errs[i], NULL);
    }

    // apply a low-pass filter to errs
    if (1) {
        std::vector<double> y(sz);

        // how much filter to apply?

        // XXX Tunable
        double sigma = 1; // was 3

        // cutoff = exp(-j*j/(2*sigma*sigma));
        // log(cutoff) = -j*j / (2*sigma*sigma)
        // log(cutoff)*2*sigma*sigma = -j*j;

        // how big a filter should we use? We make our kernel big
        // enough such that we represent any values larger than
        // 'cutoff'.

        // XXX Tunable (though not super useful to change)
        double cutoff = 0.05;
        int fsz = cvFloor(sqrt(-log(cutoff)*2*sigma*sigma)) + 1;
        fsz = 2*fsz + 1;

        // For default values of cutoff = 0.05, sigma = 3,
        // we have fsz = 17.
        std::vector<float> f(fsz);

        for (int i = 0; i < fsz; i++) {
            int j = i - fsz / 2;
            f[i] = (float)exp(-j*j/(2*sigma*sigma));
        }

        for (int iy = 0; iy < sz; iy++) {
            double acc = 0;

            for (int i = 0; i < fsz; i++) {
                acc += errs[(iy + i - fsz / 2 + sz) % sz] * f[i];
            }
            y[iy] = acc;
        }
        copy(y.begin(), y.end(), errs.begin());
    }

    std::vector<int> maxima(sz);
    std::vector<double> maxima_errs(sz);
    int nmaxima = 0;

    for (int i = 0; i < sz; i++) {
        if (errs[i] > errs[(i+1)%sz] && errs[i] > errs[(i+sz-1)%sz]) {
            maxima[nmaxima] = i;
            maxima_errs[nmaxima] = errs[i];
            nmaxima++;
        }
    }
    // if we didn't get at least 4 maxima, we can't fit a quad.
    if (nmaxima < 4)
        return 0;

    // select only the best maxima if we have too many
    int max_nmaxima = td.aprilTagMaxNmaxima;

    if (nmaxima > max_nmaxima) {
        std::vector<double> maxima_errs_copy(maxima_errs.begin(), maxima_errs.begin()+nmaxima);

        // throw out all but the best handful of maxima. Sorts descending.
        qsort(maxima_errs_copy.data(), nmaxima, sizeof(double), err_compare_descending);

        double maxima_thresh = maxima_errs_copy[max_nmaxima];
        int out = 0;
        for (int in = 0; in < nmaxima; in++) {
            if (maxima_errs[in] <= maxima_thresh)
                continue;
            maxima[out++] = maxima[in];
        }
        nmaxima = out;
    }

    int best_indices[4];
    double best_error = HUGE_VALF;

    double err01, err12, err23, err30;
    double mse01, mse12, mse23, mse30;
    double params01[4], params12[4], params23[4], params30[4];

    // disallow quads where the angle is less than a critical value.
    double max_dot = cos(td.aprilTagCriticalRad); //25*M_PI/180);

    for (int m0 = 0; m0 < nmaxima - 3; m0++) {
        int i0 = maxima[m0];

        for (int m1 = m0+1; m1 < nmaxima - 2; m1++) {
            int i1 = maxima[m1];

            fit_line(lfps, sz, i0, i1, params01, &err01, &mse01);

            if (mse01 > td.aprilTagMaxLineFitMse)
                continue;

            for (int m2 = m1+1; m2 < nmaxima - 1; m2++) {
                int i2 = maxima[m2];

                fit_line(lfps, sz, i1, i2, params12, &err12, &mse12);
                if (mse12 > td.aprilTagMaxLineFitMse)
                    continue;

                double dot = params01[2]*params12[2] + params01[3]*params12[3];
                if (fabs(dot) > max_dot)
                    continue;

                for (int m3 = m2+1; m3 < nmaxima; m3++) {
                    int i3 = maxima[m3];

                    fit_line(lfps, sz, i2, i3, params23, &err23, &mse23);
                    if (mse23 > td.aprilTagMaxLineFitMse)
                        continue;

                    fit_line(lfps, sz, i3, i0, params30, &err30, &mse30);
                    if (mse30 > td.aprilTagMaxLineFitMse)
                        continue;

                    double err = err01 + err12 + err23 + err30;
                    if (err < best_error) {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = i1;
                        best_indices[2] = i2;
                        best_indices[3] = i3;
                    }
                }
            }
        }
    }

    if (best_error == HUGE_VALF)
        return 0;

    for (int i = 0; i < 4; i++)
        indices[i] = best_indices[i];

    if (best_error / sz < td.aprilTagMaxLineFitMse)
        return 1;
    return 0;
}

/**
 * returns 0 if the cluster looks bad.
 */
int quad_segment_agg(int sz, struct line_fit_pt *lfps, int indices[4]){
    //int sz = zarray_size(cluster);

    zmaxheap_t *heap = zmaxheap_create(sizeof(struct remove_vertex*));

    // We will initially allocate sz rvs. We then have two types of
    // iterations: some iterations that are no-ops in terms of
    // allocations, and those that remove a vertex and allocate two
    // more children.  This will happen at most (sz-4) times.  Thus we
    // need: sz + 2*(sz-4) entries.

    int rvalloc_pos = 0;
    int rvalloc_size = 3*sz;
    cv::AutoBuffer<struct remove_vertex, 0> rvalloc_(std::max(1, rvalloc_size));
    memset(rvalloc_.data(), 0, sizeof(rvalloc_[0]) * rvalloc_.size()); // TODO Add AutoBuffer zero fill
    struct remove_vertex *rvalloc = rvalloc_.data();
    cv::AutoBuffer<struct segment, 0> segs_(std::max(1, sz)); // TODO Add AutoBuffer zero fill
    memset(segs_.data(), 0, sizeof(segs_[0]) * segs_.size());
    struct segment *segs = segs_.data();

    // populate with initial entries
    for (int i = 0; i < sz; i++) {
        struct remove_vertex *rv = &rvalloc[rvalloc_pos++];
        rv->i = i;
        if (i == 0) {
            rv->left = sz-1;
            rv->right = 1;
        } else {
            rv->left  = i-1;
            rv->right = (i+1) % sz;
        }

        fit_line(lfps, sz, rv->left, rv->right, NULL, NULL, &rv->err);

        //TODO is finite CV_Assert():
        CV_DbgAssert (!cvIsNaN(-rv->err) && "zmaxheap_add: Trying to add non-finite number to heap.  NaN's prohibited, could allow INF with testing");
        zmaxheap_add(heap, &rv, (float)-rv->err);

        segs[i].left = rv->left;
        segs[i].right = rv->right;
        segs[i].is_vertex = 1;
    }

    int nvertices = sz;

    while (nvertices > 4) {
        CV_Assert(rvalloc_pos < rvalloc_size);

        struct remove_vertex *rv;
        float err;

        int res = zmaxheap_remove_max(heap, &rv, &err);
        if (!res)
            return 0;
        CV_Assert(res);

        // is this remove_vertex valid? (Or has one of the left/right
        // vertices changes since we last looked?)
        if (!segs[rv->i].is_vertex ||
                !segs[rv->left].is_vertex ||
                !segs[rv->right].is_vertex) {
            continue;
        }

        // we now merge.
        CV_Assert(segs[rv->i].is_vertex);

        segs[rv->i].is_vertex = 0;
        segs[rv->left].right = rv->right;
        segs[rv->right].left = rv->left;

        // create the join to the left
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->left;
            child->left = segs[rv->left].left;
            child->right = rv->right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            //TODO is finite CV_Assert():
            CV_DbgAssert (!cvIsNaN(-child->err) && "zmaxheap_add: Trying to add non-finite number to heap.  NaN's prohibited, could allow INF with testing");
            zmaxheap_add(heap, &child, (float)-child->err);
        }

        // create the join to the right
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->right;
            child->left = rv->left;
            child->right = segs[rv->right].right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            //TODO is finite CV_Assert():
            CV_DbgAssert (!cvIsNaN(-child->err) && "zmaxheap_add: Trying to add non-finite number to heap.  NaN's prohibited, could allow INF with testing");
            zmaxheap_add(heap, &child, (float)-child->err);
        }

        // we now have one less vertex
        nvertices--;
    }

    zmaxheap_destroy(heap);

    int idx = 0;
    for (int i = 0; i < sz; i++) {
        if (segs[i].is_vertex) {
            indices[idx++] = i;
        }
    }

    return 1;
}

#define DO_UNIONFIND(dx, dy) if (im.data[y*s + dy*s + x + dx] == v) unionfind_connect(uf, y*w + x, y*w + dy*w + x + dx);
static void do_unionfind_line(unionfind_t *uf, Mat &im, int w, int s, int y){
    CV_Assert(y+1 < im.rows);
    CV_Assert(!im.empty());

    for (int x = 1; x < w - 1; x++) {
        uint8_t v = im.data[y*s + x];

        if (v == 127)
            continue;

        // (dx,dy) pairs for 8 connectivity:
        //          (REFERENCE) (1, 0)
        // (-1, 1)    (0, 1)    (1, 1)
        //
        DO_UNIONFIND(1, 0);
        DO_UNIONFIND(0, 1);
        if (v == 255) {
            DO_UNIONFIND(-1, 1);
            DO_UNIONFIND(1, 1);
        }
    }
}
#undef DO_UNIONFIND

/**
 *  return 1 if the quad looks okay, 0 if it should be discarded
 *  quad
 **/
int fit_quad(const DetectorParameters &_params, const Mat im, zarray_t *cluster, struct sQuad *quad){
    CV_Assert(cluster != NULL);

    int res = 0;

    int sz = _zarray_size(cluster);
    if (sz < 4) // can't fit a quad to less than 4 points
        return 0;

    /////////////////////////////////////////////////////////////
    // Step 1. Sort points so they wrap around the center of the
    // quad. We will constrain our quad fit to simply partition this
    // ordered set into 4 groups.

    // compute a bounding box so that we can order the points
    // according to their angle WRT the center.
    int32_t xmax = 0, xmin = INT32_MAX, ymax = 0, ymin = INT32_MAX;

    for (int pidx = 0; pidx < sz; pidx++) {
        struct pt *p;
        _zarray_get_volatile(cluster, pidx, &p);

        //(a > b) ? a : b;
        //xmax = imax(xmax, p->x);
        //xmin = imin(xmin, p->x);
        //ymax = imax(ymax, p->y);
        //ymin = imin(ymin, p->y);

        xmax = xmax > p->x? xmax : p->x;
        xmin = xmin < p->x? xmin : p->x;

        ymax = ymax > p->y? ymax : p->y;
        ymin = ymin < p->y? ymin : p->y;
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    double cx = (xmin + xmax) * 0.5 + 0.05118;
    double cy = (ymin + ymax) * 0.5 + -0.028581;

    double dot = 0;

    for (int pidx = 0; pidx < sz; pidx++) {
        struct pt *p;
        _zarray_get_volatile(cluster, pidx, &p);

        double dx = p->x - cx;
        double dy = p->y - cy;

        p->theta = cv::fastAtan2((float)dy, (float)dx) * (float)(CV_PI/180);

        dot += dx*p->gx + dy*p->gy;
    }

    // Ensure that the black border is inside the white border.
    if (dot < 0)
        return 0;

    // we now sort the points according to theta. This is a preparatory
    // step for segmenting them into four lines.
    if (1) {
        //        zarray_sort(cluster, pt_compare_theta);
        ptsort((struct pt*) cluster->data, sz);

        // remove duplicate points. (A byproduct of our segmentation system.)
        if (1) {
            int outpos = 1;

            struct pt *last;
            _zarray_get_volatile(cluster, 0, &last);

            for (int i = 1; i < sz; i++) {

                struct pt *p;
                _zarray_get_volatile(cluster, i, &p);

                if (p->x != last->x || p->y != last->y) {

                    if (i != outpos)  {
                        struct pt *out;
                        _zarray_get_volatile(cluster, outpos, &out);
                        memcpy(out, p, sizeof(struct pt));
                    }

                    outpos++;
                }

                last = p;
            }

            cluster->size = outpos;
            sz = outpos;
        }

    } else {
        // This is a counting sort in which we retain at most one
        // point for every bucket; the bucket index is computed from
        // theta. Since a good quad completes a complete revolution,
        // there's reason to think that we should get a good
        // distribution of thetas.  We might "lose" a few points due
        // to collisions, but this shouldn't affect quality very much.

        // XXX tunable. Increase to reduce the likelihood of "losing"
        // points due to collisions.
        int nbuckets = 4*sz;

#define ASSOC 2
        std::vector<std::vector<struct pt> > v(nbuckets, std::vector<struct pt>(ASSOC));

        // put each point into a bucket.
        for (int i = 0; i < sz; i++) {
            struct pt *p;
            _zarray_get_volatile(cluster, i, &p);

            CV_Assert(p->theta >= -CV_PI && p->theta <= CV_PI);

            int bucket = cvFloor((nbuckets - 1) * (p->theta + CV_PI) / (2*CV_PI));
            CV_Assert(bucket >= 0 && bucket < nbuckets);

            for (int j = 0; j < ASSOC; j++) {
                if (v[bucket][j].theta == 0) {
                    v[bucket][j] = *p;
                    break;
                }
            }
        }

        // collect the points from the buckets and put them back into the array.
        int outsz = 0;
        for (int i = 0; i < nbuckets; i++) {
            for (int j = 0; j < ASSOC; j++) {
                if (v[i][j].theta != 0) {
                    _zarray_set(cluster, outsz, &v[i][j], NULL);
                    outsz++;
                }
            }
        }

        _zarray_truncate(cluster, outsz);
        sz = outsz;
    }

    if (sz < 4)
        return 0;

    /////////////////////////////////////////////////////////////
    // Step 2. Precompute statistics that allow line fit queries to be
    // efficiently computed for any contiguous range of indices.

    cv::AutoBuffer<struct line_fit_pt, 64> lfps_(sz);
    memset(lfps_.data(), 0, sizeof(lfps_[0]) * lfps_.size()); // TODO Add AutoBuffer zero fill
    struct line_fit_pt *lfps = lfps_.data();

    for (int i = 0; i < sz; i++) {
        struct pt *p;
        _zarray_get_volatile(cluster, i, &p);

        if (i > 0) {
            memcpy(&lfps[i], &lfps[i-1], sizeof(struct line_fit_pt));
        }

        if (0) {
            // we now undo our fixed-point arithmetic.
            double delta = 0.5;
            double x = p->x * .5 + delta;
            double y = p->y * .5 + delta;
            double W;

            for (int dy = -1; dy <= 1; dy++) {
                int iy = cvFloor(y + dy);

                if (iy < 0 || iy + 1 >= im.rows)
                    continue;

                for (int dx = -1; dx <= 1; dx++) {
                    int ix = cvFloor(x + dx);

                    if (ix < 0 || ix + 1 >= im.cols)
                        continue;

                    int grad_x = im.data[iy * im.cols + ix + 1] -
                            im.data[iy * im.cols + ix - 1];

                    int grad_y = im.data[(iy+1) * im.cols + ix] -
                            im.data[(iy-1) * im.cols + ix];

                    W = sqrtf(float(grad_x*grad_x + grad_y*grad_y)) + 1;

                    //                    double fx = x + dx, fy = y + dy;
                    double fx = ix + .5, fy = iy + .5;
                    lfps[i].Mx  += W * fx;
                    lfps[i].My  += W * fy;
                    lfps[i].Mxx += W * fx * fx;
                    lfps[i].Mxy += W * fx * fy;
                    lfps[i].Myy += W * fy * fy;
                    lfps[i].W   += W;
                }
            }
        } else {
            // we now undo our fixed-point arithmetic.
            double delta = 0.5; // adjust for pixel center bias
            double x = p->x * .5 + delta;
            double y = p->y * .5 + delta;
            int ix = cvFloor(x), iy = cvFloor(y);
            double W = 1;

            if (ix > 0 && ix+1 < im.cols && iy > 0 && iy+1 < im.rows) {
                int grad_x = im.data[iy * im.cols + ix + 1] -
                        im.data[iy * im.cols + ix - 1];

                int grad_y = im.data[(iy+1) * im.cols + ix] -
                        im.data[(iy-1) * im.cols + ix];

                // XXX Tunable. How to shape the gradient magnitude?
                W = sqrt(grad_x*grad_x + grad_y*grad_y) + 1;
            }

            double fx = x, fy = y;
            lfps[i].Mx  += W * fx;
            lfps[i].My  += W * fy;
            lfps[i].Mxx += W * fx * fx;
            lfps[i].Mxy += W * fx * fy;
            lfps[i].Myy += W * fy * fy;
            lfps[i].W   += W;
        }
    }

    int indices[4];
    if (1) {
        if (!quad_segment_maxima(_params, _zarray_size(cluster), lfps, indices))
            goto finish;
    } else {
        if (!quad_segment_agg(sz, lfps, indices))
            goto finish;
    }

    //    printf("%d %d %d %d\n", indices[0], indices[1], indices[2], indices[3]);

    if (0) {
        // no refitting here; just use those points as the vertices.
        // Note, this is useful for debugging, but pretty bad in
        // practice since this code path also omits several
        // plausibility checks that save us tons of time in quad
        // decoding.
        for (int i = 0; i < 4; i++) {
            struct pt *p;
            _zarray_get_volatile(cluster, indices[i], &p);

            quad->p[i][0] = (float)(.5*p->x); // undo fixed-point arith.
            quad->p[i][1] = (float)(.5*p->y);
        }

        res = 1;

    } else {
        double lines[4][4];

        for (int i = 0; i < 4; i++) {
            int i0 = indices[i];
            int i1 = indices[(i+1)&3];

            if (0) {
                // if there are enough points, skip the points near the corners
                // (because those tend not to be very good.)
                if (i1-i0 > 8) {
                    int t = (i1-i0)/6;
                    if (t < 0)
                        t = -t;

                    i0 = (i0 + t) % sz;
                    i1 = (i1 + sz - t) % sz;
                }
            }

            double err;
            fit_line(lfps, sz, i0, i1, lines[i], NULL, &err);

            if (err > _params.aprilTagMaxLineFitMse) {
                res = 0;
                goto finish;
            }
        }

        for (int i = 0; i < 4; i++) {
            // solve for the intersection of lines (i) and (i+1)&3.
            // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
            // are the line directions.
            //
            // lambda0*u0 - lambda1*u1 = (p1 - p0)
            //
            // rearrange (solve for lambdas)
            //
            // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
            // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
            //
            // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
            // We want the unit vector, so we need the perpendiculars. Thus, below
            // we have swapped the x and y components and flipped the y components.

            const int i1 = (i + 1) & 3;
            double A00 =  lines[i][3],  A01 = -lines[i1][3];
            double A10 =  -lines[i][2],  A11 = lines[i1][2];
            double B0 = -lines[i][0] + lines[i1][0];
            double B1 = -lines[i][1] + lines[i1][1];

            double det = A00 * A11 - A10 * A01;
            if (fabs(det) < 0.001) {
                res = 0;
                goto finish;
            }

            // inverse.
            double det_inv = 1.0 / det;
            double W00 = A11 * det_inv, W01 = -A01 * det_inv;

            // solve
            double L0 = W00*B0 + W01*B1;

            // compute intersection
            quad->p[i][0] = (float)(lines[i][0] + L0*A00);
            quad->p[i][1] = (float)(lines[i][1] + L0*A10);

#if !defined(NDEBUG)
            {
                // we should get the same intersection starting
                // from point p1 and moving L1*u1.
                double W10 = -A10 * det_inv, W11 = A00 * det_inv;
                double L1 = W10*B0 + W11*B1;

                double x = lines[i1][0] - L1*A01;
                double y = lines[i1][1] - L1*A11;

                CV_Assert(fabs(x - quad->p[i][0]) < 0.001);
                CV_Assert(fabs(y - quad->p[i][1]) < 0.001);
            }
#endif // NDEBUG

            res = 1;
        }
    }

    // reject quads that are too small
    if (1) {
        double area = 0;

        // get area of triangle formed by points 0, 1, 2, 0
        double length[3], p;
        for (int i = 0; i < 3; i++) {
            int idxa = i; // 0, 1, 2,
            int idxb = (i+1) % 3; // 1, 2, 0
            //length[i] = sqrt(
            //                 sq(quad->p[idxb][0] - quad->p[idxa][0])
            //               + sq(quad->p[idxb][1] - quad->p[idxa][1]));
            double sq1 = quad->p[idxb][0] - quad->p[idxa][0];
            sq1 = sq1 * sq1;
            double sq2 = quad->p[idxb][1] - quad->p[idxa][1];
            sq2 = sq2 * sq2;
            length[i] = sqrt(sq1 + sq2);
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        // get area of triangle formed by points 2, 3, 0, 2
        for (int i = 0; i < 3; i++) {
            int idxs[] = { 2, 3, 0, 2 };
            int idxa = idxs[i];
            int idxb = idxs[i+1];
            //length[i] = sqrt(
            //                  sq(quad->p[idxb][0] - quad->p[idxa][0])
            //                + sq(quad->p[idxb][1] - quad->p[idxa][1]));
            double sq1 = quad->p[idxb][0] - quad->p[idxa][0];
            sq1 = sq1 * sq1;
            double sq2 = quad->p[idxb][1] - quad->p[idxa][1];
            sq2 = sq2 * sq2;
            length[i] = sqrt(sq1 + sq2);
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        // we don't actually know the family yet (quad detection is generic.)
        // This threshold is based on a 6x6 tag (which is actually 8x8)
        //        int d = fam->d + fam->black_border*2;
        int d = 8;
        if (area < d*d) {
            res = 0;
            goto finish;
        }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    if(1){
        double total = 0;

        for (int i = 0; i < 4; i++) {
            int i0 = i, i1 = (i+1)&3, i2 = (i+2)&3;

            double theta0 = atan2f(quad->p[i0][1] - quad->p[i1][1],
                    quad->p[i0][0] - quad->p[i1][0]);
            double theta1 = atan2f(quad->p[i2][1] - quad->p[i1][1],
                    quad->p[i2][0] - quad->p[i1][0]);

            double dtheta = theta0 - theta1;
            if (dtheta < 0)
                dtheta += 2*CV_PI;

            if (dtheta < _params.aprilTagCriticalRad || dtheta > (CV_PI - _params.aprilTagCriticalRad))
                res = 0;

            total += dtheta;
        }

        // looking for 2PI
        if (total < 6.2 || total > 6.4) {
            res = 0;
            goto finish;
        }
    }

    finish:

    return res;
}


static void do_quad(int nCidx0, int nCidx1, zarray_t &nClusters, int nW, int nH, zarray_t *nquads, const DetectorParameters &td, const Mat im){

    CV_Assert(nquads != NULL);

    //struct quad_task *task = (struct quad_task*) p;

    //zarray_t *clusters = nClusters;
    zarray_t *quads = nquads;
    int w = nW, h = nH;

    for (int cidx = nCidx0; cidx < nCidx1; cidx++) {

        zarray_t *cluster;
        _zarray_get(&nClusters, cidx, &cluster);

        if (_zarray_size(cluster) < td.aprilTagMinClusterPixels)
            continue;

        // a cluster should contain only boundary points around the
        // tag. it cannot be bigger than the whole screen. (Reject
        // large connected blobs that will be prohibitively slow to
        // fit quads to.) A typical point along an edge is added three
        // times (because it has 3 neighbors). The maximum perimeter
        // is 2w+2h.
        if (_zarray_size(cluster) > 3*(2*w+2*h)) {
            continue;
        }

        struct sQuad quad;
        memset(&quad, 0, sizeof(struct sQuad));

        if (fit_quad(td, im, cluster, &quad)) {
            //pthread_mutex_lock(&td.mutex);
            _zarray_add(quads, &quad);
            //pthread_mutex_unlock(&td.mutex);
        }
    }
}

void threshold(const Mat mIm, const DetectorParameters &parameters, Mat& mThresh){
    int w = mIm.cols, h = mIm.rows;
    int s = (unsigned) mIm.step;
    CV_Assert(w < 32768);
    CV_Assert(h < 32768);

    CV_Assert(mThresh.step == (unsigned)s);

    // The idea is to find the maximum and minimum values in a
    // window around each pixel. If it's a contrast-free region
    // (max-min is small), don't try to binarize. Otherwise,
    // threshold according to (max+min)/2.
    //
    // Mark low-contrast regions with value 127 so that we can skip
    // future work on these areas too.

    // however, computing max/min around every pixel is needlessly
    // expensive. We compute max/min for tiles. To avoid artifacts
    // that arise when high-contrast features appear near a tile
    // edge (and thus moving from one tile to another results in a
    // large change in max/min value), the max/min values used for
    // any pixel are computed from all 3x3 surrounding tiles. Thus,
    // the max/min sampling area for nearby pixels overlap by at least
    // one tile.
    //
    // The important thing is that the windows be large enough to
    // capture edge transitions; the tag does not need to fit into
    // a tile.

    // XXX Tunable. Generally, small tile sizes--- so long as they're
    // large enough to span a single tag edge--- seem to be a winner.
    const int tilesz = 4;

    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    int tw = w / tilesz;
    int th = h / tilesz;

    uint8_t *im_max = (uint8_t*)calloc(tw*th, sizeof(uint8_t));
    uint8_t *im_min = (uint8_t*)calloc(tw*th, sizeof(uint8_t));


    // first, collect min/max statistics for each tile
    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            uint8_t max = 0, min = 255;

            for (int dy = 0; dy < tilesz; dy++) {

                for (int dx = 0; dx < tilesz; dx++) {

                    uint8_t v = mIm.data[(ty*tilesz+dy)*s + tx*tilesz + dx];
                    if (v < min)
                        min = v;
                    if (v > max)
                        max = v;
                }
            }
            im_max[ty*tw+tx] = max;
            im_min[ty*tw+tx] = min;
        }
    }

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    uint8_t *im_max_tmp = (uint8_t*)calloc(tw*th, sizeof(uint8_t));
    uint8_t *im_min_tmp = (uint8_t*)calloc(tw*th, sizeof(uint8_t));

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            uint8_t max = 0, min = 255;

            for (int dy = -1; dy <= 1; dy++) {
                if (ty+dy < 0 || ty+dy >= th)
                    continue;
                for (int dx = -1; dx <= 1; dx++) {
                    if (tx+dx < 0 || tx+dx >= tw)
                        continue;

                    uint8_t m = im_max[(ty+dy)*tw+tx+dx];
                    if (m > max)
                        max = m;
                    m = im_min[(ty+dy)*tw+tx+dx];
                    if (m < min)
                        min = m;
                }
            }

            im_max_tmp[ty*tw + tx] = max;
            im_min_tmp[ty*tw + tx] = min;
        }
    }
    free(im_max);
    free(im_min);
    im_max = im_max_tmp;
    im_min = im_min_tmp;

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            int min_ = im_min[ty*tw + tx];
            int max_ = im_max[ty*tw + tx];

            // low contrast region? (no edges)
            if (max_ - min_ < parameters.aprilTagMinWhiteBlackDiff) {
                for (int dy = 0; dy < tilesz; dy++) {
                    int y = ty*tilesz + dy;

                    for (int dx = 0; dx < tilesz; dx++) {
                        int x = tx*tilesz + dx;

                        //threshim->buf[y*s+x] = 127;
                        mThresh.data[y*s+x] = 127;
                    }
                }
                continue;
            }

            // otherwise, actually threshold this tile.

            // argument for biasing towards dark; specular highlights
            // can be substantially brighter than white tag parts
            uint8_t thresh = saturate_cast<uint8_t>((max_ + min_) / 2);

            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;

                for (int dx = 0; dx < tilesz; dx++) {
                    int x = tx*tilesz + dx;

                    uint8_t v = mIm.data[y*s+x];
                    mThresh.data[y*s+x] = (v > thresh) ? 255 : 0;
                }
            }
        }
    }

    // we skipped over the non-full-sized tiles above. Fix those now.
    for (int y = 0; y < h; y++) {

        // what is the first x coordinate we need to process in this row?

        int x0;

        if (y >= th*tilesz) {
            x0 = 0; // we're at the bottom; do the whole row.
        } else {
            x0 = tw*tilesz; // we only need to do the right most part.
        }

        // compute tile coordinates and clamp.
        int ty = y / tilesz;
        if (ty >= th)
            ty = th - 1;

        for (int x = x0; x < w; x++) {
            int tx = x / tilesz;
            if (tx >= tw)
                tx = tw - 1;

            int max = im_max[ty*tw + tx];
            int min = im_min[ty*tw + tx];
            int thresh = min + (max - min) / 2;

            uint8_t v = mIm.data[y*s+x];
            if (v > thresh){
                mThresh.data[y*s+x] = 255;
            }
            else{
                mThresh.data[y*s+x] = 0;
            }
        }
    }
    free(im_min);
    free(im_max);

    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if (parameters.aprilTagDeglitch) {
        Mat tmp(h,w, mIm.type());
        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t max = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = mThresh.data[(y+dy)*s + x + dx];
                        if (v > max)
                            max = v;
                    }
                }
                tmp.data[y*s+x] = max;
            }
        }

        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t min = 255;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = tmp.data[(y+dy)*s + x + dx];
                        if (v < min)
                            min = v;
                    }
                }
                mThresh.data[y*s+x] = min;
            }
        }
    }

}

#ifdef APRIL_DEBUG
static void _darken(const Mat &im){
    for (int y = 0; y < im.rows; y++) {
        for (int x = 0; x < im.cols; x++) {
            im.data[im.cols*y+x] /= 2;
        }
    }
}
#endif

zarray_t *apriltag_quad_thresh(const DetectorParameters &parameters, const Mat & mImg, std::vector<std::vector<Point> > &contours){

    ////////////////////////////////////////////////////////
    // step 1. threshold the image, creating the edge image.

    int w = mImg.cols, h = mImg.rows;

    Mat thold(h, w, mImg.type());
    threshold(mImg, parameters, thold);

    int ts = thold.cols;

#ifdef APRIL_DEBUG
    imwrite("2.2 debug_threshold.pnm", thold);
#endif

    ////////////////////////////////////////////////////////
    // step 2. find connected components.

    unionfind_t *uf = unionfind_create(w * h);

    // TODO PARALLELIZE
    for (int y = 0; y < h - 1; y++) {
        do_unionfind_line(uf, thold, w, ts, y);
    }

    // XXX sizing??
    int nclustermap = 2*w*h - 1;

    struct uint64_zarray_entry **clustermap = (struct uint64_zarray_entry**)calloc(nclustermap, sizeof(struct uint64_zarray_entry*));

    for (int y = 1; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {

            uint8_t v0 = thold.data[y*ts + x];
            if (v0 == 127)
                continue;

            // XXX don't query this until we know we need it?
            uint64_t rep0 = unionfind_get_representative(uf, y*w + x);

            // whenever we find two adjacent pixels such that one is
            // white and the other black, we add the point half-way
            // between them to a cluster associated with the unique
            // ids of the white and black regions.
            //
            // We additionally compute the gradient direction (i.e., which
            // direction was the white pixel?) Note: if (v1-v0) == 255, then
            // (dx,dy) points towards the white pixel. if (v1-v0) == -255, then
            // (dx,dy) points towards the black pixel. p.gx and p.gy will thus
            // be -255, 0, or 255.
            //
            // Note that any given pixel might be added to multiple
            // different clusters. But in the common case, a given
            // pixel will be added multiple times to the same cluster,
            // which increases the size of the cluster and thus the
            // computational costs.
            //
            // A possible optimization would be to combine entries
            // within the same cluster.

#define DO_CONN(dx, dy)                                                 \
        if (1) {                                                    \
            uint8_t v1 = thold.data[y*ts + dy*ts + x + dx];         \
            \
            if (v0 + v1 == 255) {                                   \
                uint64_t rep1 = unionfind_get_representative(uf, y*w + dy*w + x + dx); \
                uint64_t clusterid;                                 \
                if (rep0 < rep1)                                    \
                clusterid = (rep1 << 32) + rep0;                \
                else                                                \
                clusterid = (rep0 << 32) + rep1;                \
                \
                /* XXX lousy hash function */                       \
        uint32_t clustermap_bucket = u64hash_2(clusterid) % nclustermap; \
        struct uint64_zarray_entry *entry = clustermap[clustermap_bucket]; \
        while (entry && entry->id != clusterid)     {       \
            entry = entry->next;                            \
        }                                                   \
        \
        if (!entry) {                                       \
            entry = (struct uint64_zarray_entry*)calloc(1, sizeof(struct uint64_zarray_entry)); \
            entry->id = clusterid;                          \
            entry->cluster = _zarray_create(sizeof(struct pt)); \
            entry->next = clustermap[clustermap_bucket];    \
            clustermap[clustermap_bucket] = entry;          \
        }                                                   \
        \
        struct pt p;                                        \
        p.x = saturate_cast<uint16_t>(2*x + dx);            \
        p.y = saturate_cast<uint16_t>(2*y + dy);            \
        p.gx = saturate_cast<uint16_t>(dx*((int) v1-v0));   \
        p.gy = saturate_cast<uint16_t>(dy*((int) v1-v0));   \
        _zarray_add(entry->cluster, &p);                    \
        }                                                   \
    }

    // do 4 connectivity. NB: Arguments must be [-1, 1] or we'll overflow .gx, .gy
    DO_CONN(1, 0);
    DO_CONN(0, 1);

    // do 8 connectivity
    DO_CONN(-1, 1);
    DO_CONN(1, 1);
}
}
#undef DO_CONN

#ifdef APRIL_DEBUG
Mat out = Mat::zeros(h, w, CV_8UC3);

uint32_t *colors = (uint32_t*) calloc(w*h, sizeof(*colors));

for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
        uint32_t v = unionfind_get_representative(uf, y*w+x);

        if (unionfind_get_set_size(uf, v) < parameters->aprilTagMinClusterPixels)
            continue;

        uint32_t color = colors[v];
        uint8_t r = color >> 16,
                g = color >> 8,
                b = color;

        if (color == 0) {
            const int bias = 50;
            r = bias + (random() % (200-bias));
            g = bias + (random() % (200-bias));
            b = bias + (random() % (200-bias));
            colors[v] = (r << 16) | (g << 8) | b;
        }
        out.at<Vec3b>(y, x)[0]=b;
        out.at<Vec3b>(y, x)[1]=g;
        out.at<Vec3b>(y, x)[2]=r;
    }
}
free(colors);
imwrite("2.3 debug_segmentation.pnm", out);
out = Mat::zeros(h, w, CV_8UC3);
#endif

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.
    zarray_t *clusters = _zarray_create(sizeof(zarray_t*)); //, uint64_zarray_hash_size(clustermap));
    CV_Assert(clusters != NULL);

    for (int i = 0; i < nclustermap; i++) {
        for (struct uint64_zarray_entry *entry = clustermap[i]; entry; entry = entry->next) {
            // XXX reject clusters here?
            _zarray_add(clusters, &entry->cluster);
        }
    }

#ifdef APRIL_DEBUG
for (int i = 0; i < _zarray_size(clusters); i++) {
    zarray_t *cluster;
    _zarray_get(clusters, i, &cluster);

    uint32_t r, g, b;

    const int bias = 50;
    r = bias + (random() % (200-bias));
    g = bias + (random() % (200-bias));
    b = bias + (random() % (200-bias));

    for (int j = 0; j < _zarray_size(cluster); j++) {
        struct pt *p;
        _zarray_get_volatile(cluster, j, &p);

        int x = p->x / 2;
        int y = p->y / 2;
        out.at<Vec3b>(y, x)[0]=b;
        out.at<Vec3b>(y, x)[1]=g;
        out.at<Vec3b>(y, x)[2]=r;
    }
}

imwrite("2.4 debug_clusters.pnm", out);
out = Mat::zeros(h, w, CV_8UC3);
#endif

    for (int i = 0; i < _zarray_size(clusters); i++) {
        zarray_t *cluster;
        _zarray_get(clusters, i, &cluster);

        std::vector<Point> cnt;
        for (int j = 0; j < _zarray_size(cluster); j++) {
            struct pt *p;
            _zarray_get_volatile(cluster, j, &p);

            Point pnt(p->x, p->y);
            cnt.push_back(pnt);
        }
        contours.push_back(cnt);
    }

    for (int i = 0; i < nclustermap; i++) {
        struct uint64_zarray_entry *entry = clustermap[i];
        while (entry) {
            struct uint64_zarray_entry *tmp = entry->next;
            free(entry);
            entry = tmp;
        }
    }
    free(clustermap);

    zarray_t *quads = _zarray_create(sizeof(struct sQuad));

    //int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * numberOfThreads);
    int chunksize = std::max(1, h / (10 * getNumThreads()));
    int sz = _zarray_size(clusters);

    // TODO PARALLELIZE
    for (int i = 0; i < sz; i += chunksize) {
        int min = sz < (i+chunksize)? sz: (i+chunksize);
        do_quad(i, min, *clusters, w, h, quads, parameters, mImg);
    }

#ifdef APRIL_DEBUG
mImg.copyTo(out);
_darken(out);
_darken(out);
srandom(0);

for (int i = 0; i < _zarray_size(quads); i++) {
    struct sQuad *quad;
    _zarray_get_volatile(quads, i, &quad);

    float rgb[3];
    int bias = 100;

    for (int i = 0; i < 3; i++)
        rgb[i] = bias + (random() % (255-bias));

    line(out, Point(quad->p[0][0], quad->p[0][1]), Point(quad->p[1][0], quad->p[1][1]), rgb[i]);
    line(out, Point(quad->p[1][0], quad->p[1][1]), Point(quad->p[2][0], quad->p[2][1]), rgb[i]);
    line(out, Point(quad->p[2][0], quad->p[2][1]), Point(quad->p[3][0], quad->p[3][1]), rgb[i]);
    line(out, Point(quad->p[3][0], quad->p[3][1]), Point(quad->p[0][0], quad->p[0][1]), rgb[i]);
}
imwrite("2.5 debug_lines.pnm", out);
#endif

    unionfind_destroy(uf);

    for (int i = 0; i < _zarray_size(clusters); i++) {
        zarray_t *cluster;
        _zarray_get(clusters, i, &cluster);
        _zarray_destroy(cluster);
    }
    _zarray_destroy(clusters);
    return quads;
}

void _apriltag(Mat im_orig, const DetectorParameters & _params, std::vector<std::vector<Point2f> > &candidates,
               std::vector<std::vector<Point> > &contours){

    ///////////////////////////////////////////////////////////
    /// Step 1. Detect quads according to requested image decimation
    /// and blurring parameters.
    Mat quad_im;

    if (_params.aprilTagQuadDecimate > 1){
        resize(im_orig, quad_im, Size(), 1/_params.aprilTagQuadDecimate, 1/_params.aprilTagQuadDecimate, INTER_AREA);
    }
    else {
        im_orig.copyTo(quad_im);
    }

    // Apply a Blur
    if (_params.aprilTagQuadSigma != 0) {
        // compute a reasonable kernel width by figuring that the
        // kernel should go out 2 std devs.
        //
        // max sigma          ksz
        // 0.499              1  (disabled)
        // 0.999              3
        // 1.499              5
        // 1.999              7

        float sigma = fabsf((float) _params.aprilTagQuadSigma);

        int ksz = cvFloor(4 * sigma); // 2 std devs in each direction
        ksz |= 1; // make odd number

        if (ksz > 1) {
            if (_params.aprilTagQuadSigma > 0)
                GaussianBlur(quad_im, quad_im, Size(ksz, ksz), sigma, sigma, BORDER_REPLICATE);
            else {
                Mat orig;
                quad_im.copyTo(orig);
                GaussianBlur(quad_im, quad_im, Size(ksz, ksz), sigma, sigma, BORDER_REPLICATE);

                // SHARPEN the image by subtracting the low frequency components.
                for (int y = 0; y < orig.rows; y++) {
                    for (int x = 0; x < orig.cols; x++) {
                        int vorig = orig.data[y*orig.step + x];
                        int vblur = quad_im.data[y*quad_im.step + x];

                        int v = 2*vorig - vblur;
                        if (v < 0)
                            v = 0;
                        if (v > 255)
                            v = 255;

                        quad_im.data[y*quad_im.step + x] = (uint8_t) v;
                    }
                }
            }
        }
    }

#ifdef APRIL_DEBUG
    imwrite("1.1 debug_preprocess.pnm", quad_im);
#endif

    ///////////////////////////////////////////////////////////
    /// Step 2. do the Threshold :: get the set of candidate quads
    zarray_t *quads = apriltag_quad_thresh(_params, quad_im, contours);

    CV_Assert(quads != NULL);

    // adjust centers of pixels so that they correspond to the
    // original full-resolution image.
    if (_params.aprilTagQuadDecimate > 1) {
        for (int i = 0; i < _zarray_size(quads); i++) {
            struct sQuad *q;
            _zarray_get_volatile(quads, i, &q);
            for (int j = 0; j < 4; j++) {
                q->p[j][0] *= _params.aprilTagQuadDecimate;
                q->p[j][1] *= _params.aprilTagQuadDecimate;
            }
        }
    }

#ifdef APRIL_DEBUG
    Mat im_quads = im_orig.clone();
    im_quads = im_quads*0.5;
    srandom(0);

    for (int i = 0; i < _zarray_size(quads); i++) {
        struct sQuad *quad;
        _zarray_get_volatile(quads, i, &quad);

        const int bias = 100;
        int color = bias + (random() % (255-bias));

        line(im_quads, Point(quad->p[0][0], quad->p[0][1]), Point(quad->p[1][0], quad->p[1][1]), color, 1);
        line(im_quads, Point(quad->p[1][0], quad->p[1][1]), Point(quad->p[2][0], quad->p[2][1]), color, 1);
        line(im_quads, Point(quad->p[2][0], quad->p[2][1]), Point(quad->p[3][0], quad->p[3][1]), color, 1);
        line(im_quads, Point(quad->p[3][0], quad->p[3][1]), Point(quad->p[0][0], quad->p[0][1]), color, 1);
    }
    imwrite("1.2 debug_quads_raw.pnm", im_quads);
#endif

    ////////////////////////////////////////////////////////////////
    /// Step 3. Save the output :: candidate corners
    for (int i = 0; i < _zarray_size(quads); i++) {
        struct sQuad *quad;
        _zarray_get_volatile(quads, i, &quad);

        std::vector<Point2f> corners;
        corners.push_back(Point2f(quad->p[3][0], quad->p[3][1]));   //pA
        corners.push_back(Point2f(quad->p[0][0], quad->p[0][1]));   //pB
        corners.push_back(Point2f(quad->p[1][0], quad->p[1][1]));   //pC
        corners.push_back(Point2f(quad->p[2][0], quad->p[2][1]));   //pD

        candidates.push_back(corners);
    }

    _zarray_destroy(quads);
}

}}
