/* Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *     Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer.
 *     Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *     The name of Contributor may not be used to endorse or
 *     promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 * Copyright (C) 2009, Liu Liu All rights reserved.
 *
 * OpenCV functions for MSER extraction
 *
 * 1. there are two different implementation of MSER, one for gray image, one for color image
 * 2. the gray image algorithm is taken from:
 *      Linear Time Maximally Stable Extremal Regions;
 *    the paper claims to be faster than union-find method;
 *    it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.
 * 3. the color image algorithm is taken from:
 *      Maximally Stable Colour Regions for Recognition and Match;
 *    it should be much slower than gray image method ( 3~4 times );
 *    the chi_table.h file is taken directly from the paper's source code:
 *    http://users.isy.liu.se/cvl/perfo/software/chi_table.h
 *    license (BSD-like) is located in the file: 3rdparty/mscr/chi_table_LICENSE.txt
 * 4. though the name is *contours*, the result actually is a list of point set.
 */

#include "precomp.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <limits>
#include "../3rdparty/mscr/chi_table.h"

namespace cv
{

using std::vector;

class MSER_Impl CV_FINAL : public MSER
{
public:
    struct Params
    {
        Params( int _delta=5, int _min_area=60, int _max_area=14400,
                   double _max_variation=0.25, double _min_diversity=.2,
                   int _max_evolution=200, double _area_threshold=1.01,
                   double _min_margin=0.003, int _edge_blur_size=5 )
        {
            delta = _delta;
            minArea = _min_area;
            maxArea = _max_area;
            maxVariation = _max_variation;
            minDiversity = _min_diversity;
            maxEvolution = _max_evolution;
            areaThreshold = _area_threshold;
            minMargin = _min_margin;
            edgeBlurSize = _edge_blur_size;
            pass2Only = false;
        }

        int delta;
        int minArea;
        int maxArea;
        double maxVariation;
        double minDiversity;
        bool pass2Only;

        int maxEvolution;
        double areaThreshold;
        double minMargin;
        int edgeBlurSize;
    };

    explicit MSER_Impl(const Params& _params) : params(_params) {}

    virtual ~MSER_Impl() CV_OVERRIDE {}

    void read( const FileNode& fn) CV_OVERRIDE
    {
      // if node is empty, keep previous value
      if (!fn["delta"].empty())
        fn["delta"] >> params.delta;
      if (!fn["minArea"].empty())
        fn["minArea"] >> params.minArea;
      if (!fn["maxArea"].empty())
        fn["maxArea"] >> params.maxArea;
      if (!fn["maxVariation"].empty())
        fn["maxVariation"] >> params.maxVariation;
      if (!fn["minDiversity"].empty())
        fn["minDiversity"] >> params.minDiversity;
      if (!fn["maxEvolution"].empty())
        fn["maxEvolution"] >> params.maxEvolution;
      if (!fn["areaThreshold"].empty())
        fn["areaThreshold"] >> params.areaThreshold;
      if (!fn["minMargin"].empty())
        fn["minMargin"] >> params.minMargin;
      if (!fn["edgeBlurSize"].empty())
        fn["edgeBlurSize"] >> params.edgeBlurSize;
      if (!fn["pass2Only"].empty())
        fn["pass2Only"] >> params.pass2Only;
    }
    void write( FileStorage& fs) const CV_OVERRIDE
    {
      if(fs.isOpened())
      {
        fs << "name" << getDefaultName();
        fs << "delta" << params.delta;
        fs << "minArea" << params.minArea;
        fs << "maxArea" << params.maxArea;
        fs << "maxVariation" << params.maxVariation;
        fs << "minDiversity" << params.minDiversity;
        fs << "maxEvolution" << params.maxEvolution;
        fs << "areaThreshold" << params.areaThreshold;
        fs << "minMargin" << params.minMargin;
        fs << "edgeBlurSize" << params.edgeBlurSize;
        fs << "pass2Only" << params.pass2Only;
      }
    }

    void setDelta(int delta) CV_OVERRIDE { params.delta = delta; }
    int getDelta() const CV_OVERRIDE { return params.delta; }

    void setMinArea(int minArea) CV_OVERRIDE { params.minArea = minArea; }
    int getMinArea() const CV_OVERRIDE { return params.minArea; }

    void setMaxArea(int maxArea) CV_OVERRIDE { params.maxArea = maxArea; }
    int getMaxArea() const CV_OVERRIDE { return params.maxArea; }

    void setMaxVariation(double maxVariation) CV_OVERRIDE { params.maxVariation = maxVariation; }
    double getMaxVariation() const CV_OVERRIDE { return params.maxVariation; }

    void setMinDiversity(double minDiversity) CV_OVERRIDE { params.minDiversity = minDiversity; }
    double getMinDiversity() const CV_OVERRIDE { return params.minDiversity; }

    void setMaxEvolution(int maxEvolution) CV_OVERRIDE { params.maxEvolution = maxEvolution; }
    int getMaxEvolution() const CV_OVERRIDE { return params.maxEvolution; }

    void setAreaThreshold(double areaThreshold) CV_OVERRIDE { params.areaThreshold = areaThreshold; }
    double getAreaThreshold() const CV_OVERRIDE { return params.areaThreshold; }

    void setMinMargin(double min_margin) CV_OVERRIDE { params.minMargin = min_margin; }
    double getMinMargin() const CV_OVERRIDE { return params.minMargin; }

    void setEdgeBlurSize(int edge_blur_size) CV_OVERRIDE { params.edgeBlurSize = edge_blur_size; }
    int getEdgeBlurSize() const CV_OVERRIDE { return params.edgeBlurSize; }

    void setPass2Only(bool f) CV_OVERRIDE { params.pass2Only = f; }
    bool getPass2Only() const CV_OVERRIDE { return params.pass2Only; }

    enum { DIR_SHIFT = 29, NEXT_MASK = ((1<<DIR_SHIFT)-1)  };

    struct Pixel
    {
        Pixel() : val(0) {}
        Pixel(int _val) : val(_val) {}

        int getGray(const Pixel* ptr0, const uchar* imgptr0, int mask) const
        {
            return imgptr0[this - ptr0] ^ mask;
        }
        int getNext() const { return (val & NEXT_MASK); }
        void setNext(int next) { val = (val & ~NEXT_MASK) | next; }

        int getDir() const { return (int)((unsigned)val >> DIR_SHIFT); }
        void setDir(int dir) { val = (val & NEXT_MASK) | (dir << DIR_SHIFT); }
        bool isVisited() const { return (val & ~NEXT_MASK) != 0; }

        int val;
    };
    typedef int PPixel;

    struct WParams
    {
        Params p;
        vector<vector<Point> >* msers;
        vector<Rect>* bboxvec;
        Pixel* pix0;
        int step;
    };

    // the history of region grown
    struct CompHistory
    {
        CompHistory()
        {
            parent_ = child_ = next_ = 0;
            val = size = 0;
            var = -1.f;
            head = 0;
            checked = false;
        }
        void updateTree( WParams& wp, CompHistory** _h0, CompHistory** _h1, bool final )
        {
            if( var >= 0.f )
                return;
            int delta = wp.p.delta;

            CompHistory* h0_ = 0, *h1_ = 0;
            CompHistory* c = child_;
            if( size >= wp.p.minArea )
            {
                for( ; c != 0; c = c->next_ )
                {
                    if( c->var < 0.f )
                        c->updateTree(wp, c == child_ ? &h0_ : 0, c == child_ ? &h1_ : 0, final);
                    if( c->var < 0.f )
                        return;
                }
            }

            // find h0 and h1 such that:
            //    h0->val >= h->val - delta and (h0->parent == 0 or h0->parent->val < h->val - delta)
            //    h1->val <= h->val + delta and (h1->child == 0 or h1->child->val < h->val + delta)
            // then we will adjust h0 and h1 as h moves towards latest
            CompHistory* h0 = this, *h1 = h1_ && h1_->size > size ? h1_ : this;
            if( h0_ )
            {
                for( h0 = h0_; h0 != this && h0->val < val - delta; h0 = h0->parent_ )
                    ;
            }
            else
            {
                for( ; h0->child_ && h0->child_->val >= val - delta; h0 = h0->child_ )
                    ;
            }

            for( ; h1->parent_ && h1->parent_->val <= val + delta; h1 = h1->parent_ )
                ;

            if( _h0 ) *_h0 = h0;
            if( _h1 ) *_h1 = h1;

            // when we do not well-defined ER(h->val + delta), we stop
            // the process of computing variances unless we are at the final step
            if( !final && !h1->parent_ && h1->val < val + delta )
                return;

            var = (float)(h1->size - h0->size)/size;
            c = child_;
            for( ; c != 0; c = c->next_ )
                c->checkAndCapture(wp);
            if( final && !parent_ )
                checkAndCapture(wp);
        }

        void checkAndCapture( WParams& wp )
        {
            if( checked )
                return;
            checked = true;
            if( size < wp.p.minArea || size > wp.p.maxArea || var < 0.f || var > wp.p.maxVariation )
                return;
            if( child_ )
            {
                CompHistory* c = child_;
                for( ; c != 0; c = c->next_ )
                {
                    if( c->var >= 0.f && var > c->var )
                        return;
                }
            }
            if( var > 0.f && parent_ && parent_->var >= 0.f && var >= parent_->var )
                return;
            int xmin = INT_MAX, ymin = INT_MAX, xmax = INT_MIN, ymax = INT_MIN, j = 0;
            wp.msers->push_back(vector<Point>());
            vector<Point>& region = wp.msers->back();
            region.resize(size);
            const Pixel* pix0 = wp.pix0;
            int step = wp.step;

            for( PPixel pix = head; j < size; j++, pix = pix0[pix].getNext() )
            {
                int y = pix/step;
                int x = pix - y*step;

                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);

                region[j] = Point(x, y);
            }

            wp.bboxvec->push_back(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        }

        CompHistory* child_;
        CompHistory* parent_;
        CompHistory* next_;
        int val;
        int size;
        float var;
        PPixel head;
        bool checked;
    };

    struct ConnectedComp
    {
        ConnectedComp()
        {
            init(0);
        }

        void init(int gray)
        {
            head = tail = 0;
            history = 0;
            size = 0;
            gray_level = gray;
        }

        // add history chunk to a connected component
        void growHistory(CompHistory*& hptr, WParams& wp, int new_gray_level, bool final)
        {
            if (new_gray_level < gray_level)
                new_gray_level = gray_level;

            CompHistory *h;
            if (history && history->val == gray_level)
            {
                h = history;
            }
            else
            {
                h = hptr++;
                h->parent_ = 0;
                h->child_ = history;
                h->next_ = 0;

                if (history)
                {
                    history->parent_ = h;
                }
            }
            CV_Assert(h != NULL);
            h->val = gray_level;
            h->size = size;
            h->head = head;
            h->var = FLT_MAX;
            h->checked = true;
            if (h->size >= wp.p.minArea)
            {
                h->var = -1.f;
                h->checked = false;
            }

            gray_level = new_gray_level;
            history = h;
            if (history && history->val != gray_level)
            {
                history->updateTree(wp, 0, 0, final);
            }
        }

        // merging two connected components
        void merge( ConnectedComp* comp1, ConnectedComp* comp2,
                    CompHistory*& hptr, WParams& wp )
        {
            if (comp1->gray_level < comp2->gray_level)
                std::swap(comp1, comp2);

            gray_level = comp1->gray_level;
            comp1->growHistory(hptr, wp, gray_level, false);
            comp2->growHistory(hptr, wp, gray_level, false);

            if (comp1->size == 0)
            {
                head = comp2->head;
                tail = comp2->tail;
            }
            else
            {
                head = comp1->head;
                wp.pix0[comp1->tail].setNext(comp2->head);
                tail = comp2->tail;
            }

            size = comp1->size + comp2->size;
            history = comp1->history;

            CompHistory *h1 = history->child_;
            CompHistory *h2 = comp2->history;
            // the child_'s size should be the large one
            if (h1 && h1->size > h2->size)
            {
                // add h2 as a child only if its size is large enough
                if(h2->size >= wp.p.minArea)
                {
                    h2->next_ = h1->next_;
                    h1->next_ = h2;
                    h2->parent_ = history;
                }
            }
            else
            {
                history->child_ = h2;
                h2->parent_ = history;
                // reserve h1 as a child only if its size is large enough
                if (h1 && h1->size >= wp.p.minArea)
                {
                    h2->next_ = h1;
                }
            }
        }

        PPixel head;
        PPixel tail;
        CompHistory* history;
        int gray_level;
        int size;
    };

    void detectRegions( InputArray image,
                        std::vector<std::vector<Point> >& msers,
                        std::vector<Rect>& bboxes ) CV_OVERRIDE;
    void detect( InputArray _src, vector<KeyPoint>& keypoints, InputArray _mask ) CV_OVERRIDE;

    void preprocess1( const Mat& img, int* level_size )
    {
        memset(level_size, 0, 256*sizeof(level_size[0]));

        int i, j, cols = img.cols, rows = img.rows;
        int step = cols;
        pixbuf.resize(step*rows);
        heapbuf.resize(cols*rows + 256);
        histbuf.resize(cols*rows);
        Pixel borderpix;
        borderpix.setDir(5);

        for( j = 0; j < step; j++ )
        {
            pixbuf[j] = pixbuf[j + (rows-1)*step] = borderpix;
        }

        for( i = 1; i < rows-1; i++ )
        {
            const uchar* imgptr = img.ptr(i);
            Pixel* pptr = &pixbuf[i*step];
            pptr[0] = pptr[cols-1] = borderpix;
            for( j = 1; j < cols-1; j++ )
            {
                int val = imgptr[j];
                level_size[val]++;
                pptr[j].val = 0;
            }
        }
    }

    void preprocess2( const Mat& img, int* level_size )
    {
        int i;

        for( i = 0; i < 128; i++ )
            std::swap(level_size[i], level_size[255-i]);

        if( !params.pass2Only )
        {
            int j, cols = img.cols, rows = img.rows;
            int step = cols;
            for( i = 1; i < rows-1; i++ )
            {
                Pixel* pptr = &pixbuf[i*step];
                for( j = 1; j < cols-1; j++ )
                {
                    pptr[j].val = 0;
                }
            }
        }
    }

    void pass( const Mat& img, vector<vector<Point> >& msers, vector<Rect>& bboxvec,
              Size size, const int* level_size, int mask )
    {
        CompHistory* histptr = &histbuf[0];
        int step = size.width;
        Pixel *ptr0 = &pixbuf[0], *ptr = &ptr0[step+1];
        const uchar* imgptr0 = img.ptr();
        Pixel** heap[256];
        ConnectedComp comp[257];
        ConnectedComp* comptr = &comp[0];
        WParams wp;
        wp.p = params;
        wp.msers = &msers;
        wp.bboxvec = &bboxvec;
        wp.pix0 = ptr0;
        wp.step = step;

        heap[0] = &heapbuf[0];
        heap[0][0] = 0;

        for( int i = 1; i < 256; i++ )
        {
            heap[i] = heap[i-1] + level_size[i-1] + 1;
            heap[i][0] = 0;
        }

        comptr->gray_level = 256;
        comptr++;
        comptr->gray_level = ptr->getGray(ptr0, imgptr0, mask);
        ptr->setDir(1);
        int dir[] = { 0, 1, step, -1, -step };
        for( ;; )
        {
            int curr_gray = ptr->getGray(ptr0, imgptr0, mask);
            int nbr_idx = ptr->getDir();
            // take tour of all the 4 directions
            for( ; nbr_idx <= 4; nbr_idx++ )
            {
                // get the neighbor
                Pixel* ptr_nbr = ptr + dir[nbr_idx];
                if( !ptr_nbr->isVisited() )
                {
                    // set dir=1, next=0
                    ptr_nbr->val = 1 << DIR_SHIFT;
                    int nbr_gray = ptr_nbr->getGray(ptr0, imgptr0, mask);
                    if( nbr_gray < curr_gray )
                    {
                        // when the value of neighbor smaller than current
                        // push current to boundary heap and make the neighbor to be the current one
                        // create an empty comp
                        *(++heap[curr_gray]) = ptr;
                        ptr->val = (nbr_idx+1) << DIR_SHIFT;
                        ptr = ptr_nbr;
                        comptr++;
                        comptr->init(nbr_gray);
                        curr_gray = nbr_gray;
                        nbr_idx = 0;
                        continue;
                    }
                    // otherwise, push the neighbor to boundary heap
                    *(++heap[nbr_gray]) = ptr_nbr;
                }
            }

            // set dir = nbr_idx, next = 0
            ptr->val = nbr_idx << DIR_SHIFT;
            int ptrofs = (int)(ptr - ptr0);
            CV_Assert(ptrofs != 0);

            // add a pixel to the pixel list
            if( comptr->tail )
                ptr0[comptr->tail].setNext(ptrofs);
            else
                comptr->head = ptrofs;
            comptr->tail = ptrofs;
            comptr->size++;
            // get the next pixel from boundary heap
            if( *heap[curr_gray] )
            {
                ptr = *heap[curr_gray];
                heap[curr_gray]--;
            }
            else
            {
                for( curr_gray++; curr_gray < 256; curr_gray++ )
                {
                    if( *heap[curr_gray] )
                        break;
                }
                if( curr_gray >= 256 )
                    break;

                ptr = *heap[curr_gray];
                heap[curr_gray]--;

                if (curr_gray < comptr[-1].gray_level)
                {
                    comptr->growHistory(histptr, wp, curr_gray, false);
                    CV_DbgAssert(comptr->size == comptr->history->size);
                }
                else
                {
                    // there must one pixel with the second component's gray level in the heap,
                    // so curr_gray is not large than the second component's gray level
                    comptr--;
                    CV_DbgAssert(curr_gray == comptr->gray_level);
                    comptr->merge(comptr, comptr + 1, histptr, wp);
                    CV_DbgAssert(curr_gray == comptr->gray_level);
                }
            }
        }

        for( ; comptr->gray_level != 256; comptr-- )
        {
            comptr->growHistory(histptr, wp, 256, true);
        }
    }

    Mat tempsrc;
    vector<Pixel> pixbuf;
    vector<Pixel*> heapbuf;
    vector<CompHistory> histbuf;

    Params params;
};

/*

TODO:
the color MSER has not been completely refactored yet. We leave it mostly as-is,
with just enough changes to convert C structures to C++ ones and
add support for color images into MSER_Impl::detectAndLabel.
*/
struct MSCRNode;

struct TempMSCR
{
    MSCRNode* head;
    MSCRNode* tail;
    double m; // the margin used to prune area later
    int size;
};

struct MSCRNode
{
    MSCRNode* shortcut;
    // to make the finding of root less painful
    MSCRNode* prev;
    MSCRNode* next;
    // a point double-linked list
    TempMSCR* tmsr;
    // the temporary msr (set to NULL at every re-initialise)
    TempMSCR* gmsr;
    // the global msr (once set, never to NULL)
    int index;
    // the index of the node, at this point, it should be x at the first 16-bits, and y at the last 16-bits.
    int rank;
    int reinit;
    int size, sizei;
    double dt, di;
    double s;
};

struct MSCREdge
{
    double chi;
    MSCRNode* left;
    MSCRNode* right;
};

static double ChiSquaredDistance( const uchar* x, const uchar* y )
{
    return (double)((x[0]-y[0])*(x[0]-y[0]))/(double)(x[0]+y[0]+1e-10)+
    (double)((x[1]-y[1])*(x[1]-y[1]))/(double)(x[1]+y[1]+1e-10)+
    (double)((x[2]-y[2])*(x[2]-y[2]))/(double)(x[2]+y[2]+1e-10);
}

static void initMSCRNode( MSCRNode* node )
{
    node->gmsr = node->tmsr = NULL;
    node->reinit = 0xffff;
    node->rank = 0;
    node->sizei = node->size = 1;
    node->prev = node->next = node->shortcut = node;
}

// the preprocess to get the edge list with proper gaussian blur
static int preprocessMSER_8uC3( MSCRNode* node,
                               MSCREdge* edge,
                               double* total,
                               const Mat& src,
                               Mat& dx,
                               Mat& dy,
                               int Ne,
                               int edgeBlurSize )
{
    int srccpt = (int)(src.step-src.cols*3);
    const uchar* srcptr = src.ptr();
    const uchar* lastptr = srcptr+3;
    double* dxptr = dx.ptr<double>();
    for ( int i = 0; i < src.rows; i++ )
    {
        for ( int j = 0; j < src.cols-1; j++ )
        {
            *dxptr = ChiSquaredDistance( srcptr, lastptr );
            dxptr++;
            srcptr += 3;
            lastptr += 3;
        }
        srcptr += srccpt+3;
        lastptr += srccpt+3;
    }
    srcptr = src.ptr();
    lastptr = srcptr+src.step;
    double* dyptr = dy.ptr<double>();
    for ( int i = 0; i < src.rows-1; i++ )
    {
        for ( int j = 0; j < src.cols; j++ )
        {
            *dyptr = ChiSquaredDistance( srcptr, lastptr );
            dyptr++;
            srcptr += 3;
            lastptr += 3;
        }
        srcptr += srccpt;
        lastptr += srccpt;
    }
    // get dx and dy and blur it
    if ( edgeBlurSize >= 1 )
    {
        GaussianBlur( dx, dx, Size(edgeBlurSize, edgeBlurSize), 0 );
        GaussianBlur( dy, dy, Size(edgeBlurSize, edgeBlurSize), 0 );
    }
    dxptr = dx.ptr<double>();
    dyptr = dy.ptr<double>();
    // assian dx, dy to proper edge list and initialize mscr node
    // the nasty code here intended to avoid extra loops
    MSCRNode* nodeptr = node;
    initMSCRNode( nodeptr );
    nodeptr->index = 0;
    *total += edge->chi = *dxptr;
    dxptr++;
    edge->left = nodeptr;
    edge->right = nodeptr+1;
    edge++;
    nodeptr++;
    for ( int i = 1; i < src.cols-1; i++ )
    {
        initMSCRNode( nodeptr );
        nodeptr->index = i;
        *total += edge->chi = *dxptr;
        dxptr++;
        edge->left = nodeptr;
        edge->right = nodeptr+1;
        edge++;
        nodeptr++;
    }
    initMSCRNode( nodeptr );
    nodeptr->index = src.cols-1;
    nodeptr++;
    for ( int i = 1; i < src.rows-1; i++ )
    {
        initMSCRNode( nodeptr );
        nodeptr->index = i<<16;
        *total += edge->chi = *dyptr;
        dyptr++;
        edge->left = nodeptr-src.cols;
        edge->right = nodeptr;
        edge++;
        *total += edge->chi = *dxptr;
        dxptr++;
        edge->left = nodeptr;
        edge->right = nodeptr+1;
        edge++;
        nodeptr++;
        for ( int j = 1; j < src.cols-1; j++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = (i<<16)|j;
            *total += edge->chi = *dyptr;
            dyptr++;
            edge->left = nodeptr-src.cols;
            edge->right = nodeptr;
            edge++;
            *total += edge->chi = *dxptr;
            dxptr++;
            edge->left = nodeptr;
            edge->right = nodeptr+1;
            edge++;
            nodeptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = (i<<16)|(src.cols-1);
        *total += edge->chi = *dyptr;
        dyptr++;
        edge->left = nodeptr-src.cols;
        edge->right = nodeptr;
        edge++;
        nodeptr++;
    }
    initMSCRNode( nodeptr );
    nodeptr->index = (src.rows-1)<<16;
    *total += edge->chi = *dxptr;
    dxptr++;
    edge->left = nodeptr;
    edge->right = nodeptr+1;
    edge++;
    *total += edge->chi = *dyptr;
    dyptr++;
    edge->left = nodeptr-src.cols;
    edge->right = nodeptr;
    edge++;
    nodeptr++;
    for ( int i = 1; i < src.cols-1; i++ )
    {
        initMSCRNode( nodeptr );
        nodeptr->index = ((src.rows-1)<<16)|i;
        *total += edge->chi = *dxptr;
        dxptr++;
        edge->left = nodeptr;
        edge->right = nodeptr+1;
        edge++;
        *total += edge->chi = *dyptr;
        dyptr++;
        edge->left = nodeptr-src.cols;
        edge->right = nodeptr;
        edge++;
        nodeptr++;
    }
    initMSCRNode( nodeptr );
    nodeptr->index = ((src.rows-1)<<16)|(src.cols-1);
    *total += edge->chi = *dyptr;
    edge->left = nodeptr-src.cols;
    edge->right = nodeptr;

    return Ne;
}

class LessThanEdge
{
public:
    bool operator()(const MSCREdge& a, const MSCREdge& b) const { return a.chi < b.chi; }
};

// to find the root of one region
static MSCRNode* findMSCR( MSCRNode* x )
{
    MSCRNode* prev = x;
    MSCRNode* next;
    for ( ; ; )
    {
        next = x->shortcut;
        x->shortcut = prev;
        if ( next == x ) break;
        prev= x;
        x = next;
    }
    MSCRNode* root = x;
    for ( ; ; )
    {
        prev = x->shortcut;
        x->shortcut = root;
        if ( prev == x ) break;
        x = prev;
    }
    return root;
}

// the stable mscr should be:
// bigger than minArea and smaller than maxArea
// differ from its ancestor more than minDiversity
static bool MSCRStableCheck( MSCRNode* x, const MSER_Impl::Params& params )
{
    if ( x->size <= params.minArea || x->size >= params.maxArea )
        return false;
    if ( x->gmsr == NULL )
        return true;
    double div = (double)(x->size-x->gmsr->size)/(double)x->size;
    return div > params.minDiversity;
}

static void
extractMSER_8uC3( const Mat& src,
                  vector<vector<Point> >& msers,
                  vector<Rect>& bboxvec,
                  const MSER_Impl::Params& params )
{
    bboxvec.clear();
    AutoBuffer<MSCRNode> mapBuf(src.cols*src.rows);
    MSCRNode* map = mapBuf.data();
    int Ne = src.cols*src.rows*2-src.cols-src.rows;
    AutoBuffer<MSCREdge> edgeBuf(Ne);
    MSCREdge* edge = edgeBuf.data();
    AutoBuffer<TempMSCR> mscrBuf(src.cols*src.rows);
    TempMSCR* mscr = mscrBuf.data();
    double emean = 0;
    Mat dx( src.rows, src.cols-1, CV_64FC1 );
    Mat dy( src.rows-1, src.cols, CV_64FC1 );
    Ne = preprocessMSER_8uC3( map, edge, &emean, src, dx, dy, Ne, params.edgeBlurSize );
    emean = emean / (double)Ne;
    std::sort(edge, edge + Ne, LessThanEdge());
    MSCREdge* edge_ub = edge+Ne;
    MSCREdge* edgeptr = edge;
    TempMSCR* mscrptr = mscr;
    // the evolution process
    for ( int i = 0; i < params.maxEvolution; i++ )
    {
        double k = (double)i/(double)params.maxEvolution*(TABLE_SIZE-1);
        int ti = cvFloor(k);
        double reminder = k-ti;
        double thres = emean*(chitab3[ti]*(1-reminder)+chitab3[ti+1]*reminder);
        // to process all the edges in the list that chi < thres
        while ( edgeptr < edge_ub && edgeptr->chi < thres )
        {
            MSCRNode* lr = findMSCR( edgeptr->left );
            MSCRNode* rr = findMSCR( edgeptr->right );
            // get the region root (who is responsible)
            if ( lr != rr )
            {
                // rank idea take from: N-tree Disjoint-Set Forests for Maximally Stable Extremal Regions
                if ( rr->rank > lr->rank )
                {
                    MSCRNode* tmp;
                    CV_SWAP( lr, rr, tmp );
                } else if ( lr->rank == rr->rank ) {
                    // at the same rank, we will compare the size
                    if ( lr->size > rr->size )
                    {
                        MSCRNode* tmp;
                        CV_SWAP( lr, rr, tmp );
                    }
                    lr->rank++;
                }
                rr->shortcut = lr;
                lr->size += rr->size;
                // join rr to the end of list lr (lr is a endless double-linked list)
                lr->prev->next = rr;
                lr->prev = rr->prev;
                rr->prev->next = lr;
                rr->prev = lr;
                // area threshold force to reinitialize
                if ( lr->size > (lr->size-rr->size)*params.areaThreshold )
                {
                    lr->sizei = lr->size;
                    lr->reinit = i;
                    if ( lr->tmsr != NULL )
                    {
                        lr->tmsr->m = lr->dt-lr->di;
                        lr->tmsr = NULL;
                    }
                    lr->di = edgeptr->chi;
                    lr->s = 1e10;
                }
                lr->dt = edgeptr->chi;
                if ( i > lr->reinit )
                {
                    double s = (double)(lr->size-lr->sizei)/(lr->dt-lr->di);
                    if ( s < lr->s )
                    {
                        // skip the first one and check stability
                        if ( i > lr->reinit+1 && MSCRStableCheck( lr, params ) )
                        {
                            if ( lr->tmsr == NULL )
                            {
                                lr->gmsr = lr->tmsr = mscrptr;
                                mscrptr++;
                            }
                            lr->tmsr->size = lr->size;
                            lr->tmsr->head = lr;
                            lr->tmsr->tail = lr->prev;
                            lr->tmsr->m = 0;
                        }
                        lr->s = s;
                    }
                }
            }
            edgeptr++;
        }
        if ( edgeptr >= edge_ub )
            break;
    }
    for ( TempMSCR* ptr = mscr; ptr < mscrptr; ptr++ )
        // to prune area with margin less than minMargin
        if ( ptr->m > params.minMargin )
        {
            MSCRNode* lpt = ptr->head;
            int xmin = INT_MAX, ymin = INT_MAX, xmax = INT_MIN, ymax = INT_MIN;
            msers.push_back(vector<Point>());
            vector<Point>& mser = msers.back();

            for ( int i = 0; i < ptr->size; i++ )
            {
                Point pt;
                pt.x = (lpt->index)&0xffff;
                pt.y = (lpt->index)>>16;
                xmin = std::min(xmin, pt.x);
                xmax = std::max(xmax, pt.x);
                ymin = std::min(ymin, pt.y);
                ymax = std::max(ymax, pt.y);

                lpt = lpt->next;
                mser.push_back(pt);
            }
            bboxvec.push_back(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        }
}

void MSER_Impl::detectRegions( InputArray _src, vector<vector<Point> >& msers, vector<Rect>& bboxes )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();

    msers.clear();
    bboxes.clear();

    if( src.rows < 3 || src.cols < 3 )
        CV_Error(Error::StsBadArg, "Input image is too small. Expected at least 3x3");

    Size size = src.size();

    if( src.type() == CV_8U )
    {
        int level_size[256];
        if( !src.isContinuous() )
        {
            src.copyTo(tempsrc);
            src = tempsrc;
        }

        // darker to brighter (MSER+)
        preprocess1( src, level_size );
        if( !params.pass2Only )
            pass( src, msers, bboxes, size, level_size, 0 );
        // brighter to darker (MSER-)
        preprocess2( src, level_size );
        pass( src, msers, bboxes, size, level_size, 255 );
    }
    else
    {
        CV_Assert( src.type() == CV_8UC3 || src.type() == CV_8UC4 );
        extractMSER_8uC3( src, msers, bboxes, params );
    }
}

void MSER_Impl::detect( InputArray _image, vector<KeyPoint>& keypoints, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    vector<Rect> bboxes;
    vector<vector<Point> > msers;
    Mat mask = _mask.getMat();

    detectRegions(_image, msers, bboxes);
    int i, ncomps = (int)msers.size();

    keypoints.clear();
    for( i = 0; i < ncomps; i++ )
    {
        Rect r = bboxes[i];
        // TODO check transformation from MSER region to KeyPoint
        RotatedRect rect = fitEllipse(Mat(msers[i]));
        float diam = std::sqrt(rect.size.height*rect.size.width);

        if( diam > std::numeric_limits<float>::epsilon() && r.contains(rect.center) &&
            (mask.empty() || mask.at<uchar>(cvRound(rect.center.y), cvRound(rect.center.x)) != 0) )
            keypoints.push_back( KeyPoint(rect.center, diam) );
    }
}

Ptr<MSER> MSER::create( int _delta, int _min_area, int _max_area,
      double _max_variation, double _min_diversity,
      int _max_evolution, double _area_threshold,
      double _min_margin, int _edge_blur_size )
{
    return makePtr<MSER_Impl>(
        MSER_Impl::Params(_delta, _min_area, _max_area,
                          _max_variation, _min_diversity,
                          _max_evolution, _area_threshold,
                          _min_margin, _edge_blur_size));
}

String MSER::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".MSER");
}

}
