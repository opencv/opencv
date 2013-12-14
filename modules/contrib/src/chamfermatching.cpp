/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008-2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

//
// The original code was written by
//          Marius Muja
// and later modified and prepared
//  for integration into OpenCV by
//        Antonella Cascitelli,
//        Marco Di Stefano and
//          Stefano Fabri
//        from Univ. of Rome
//

#include "precomp.hpp"
#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#  include "opencv2/highgui/highgui.hpp"
#endif
#include <iostream>
#include <queue>

namespace cv
{

using std::queue;

typedef std::pair<int,int> coordinate_t;
typedef float orientation_t;
typedef std::vector<coordinate_t> template_coords_t;
typedef std::vector<orientation_t> template_orientations_t;
typedef std::pair<Point, float> location_scale_t;

class ChamferMatcher
{

private:
    class Matching;
    int max_matches_;
    float min_match_distance_;

    ///////////////////////// Image iterators ////////////////////////////

    class ImageIterator
    {
    public:
        virtual ~ImageIterator() {}
        virtual bool hasNext() const = 0;
        virtual location_scale_t next() = 0;
    };

    class ImageRange
    {
    public:
        virtual ImageIterator* iterator() const = 0;
        virtual ~ImageRange() {}
    };

    // Sliding window

    class SlidingWindowImageRange : public ImageRange
    {
        int width_;
        int height_;
        int x_step_;
        int y_step_;
        int scales_;
        float min_scale_;
        float max_scale_;

    public:
        SlidingWindowImageRange(int width, int height, int x_step = 3, int y_step = 3, int _scales = 5, float min_scale = 0.6, float max_scale = 1.6) :
        width_(width), height_(height), x_step_(x_step),y_step_(y_step), scales_(_scales), min_scale_(min_scale), max_scale_(max_scale)
        {
        }


        ImageIterator* iterator() const;
    };

    class LocationImageRange : public ImageRange
    {
        const std::vector<Point>& locations_;

        int scales_;
        float min_scale_;
        float max_scale_;

        LocationImageRange(const LocationImageRange&);
        LocationImageRange& operator=(const LocationImageRange&);

    public:
        LocationImageRange(const std::vector<Point>& locations, int _scales = 5, float min_scale = 0.6, float max_scale = 1.6) :
        locations_(locations), scales_(_scales), min_scale_(min_scale), max_scale_(max_scale)
        {
        }

        ImageIterator* iterator() const
        {
            return new LocationImageIterator(locations_, scales_, min_scale_, max_scale_);
        }
    };


    class LocationScaleImageRange : public ImageRange
    {
        const std::vector<Point>& locations_;
        const std::vector<float>& scales_;

        LocationScaleImageRange(const LocationScaleImageRange&);
        LocationScaleImageRange& operator=(const LocationScaleImageRange&);
    public:
        LocationScaleImageRange(const std::vector<Point>& locations, const std::vector<float>& _scales) :
        locations_(locations), scales_(_scales)
        {
            assert(locations.size()==_scales.size());
        }

        ImageIterator* iterator() const
        {
            return new LocationScaleImageIterator(locations_, scales_);
        }
    };




public:
    /**
     * Class that represents a template for chamfer matching.
     */
    class Template
    {
        friend class ChamferMatcher::Matching;
        friend class ChamferMatcher;


    public:
        std::vector<Template*> scaled_templates;
        std::vector<int> addr;
        int addr_width;
        float scale;
        template_coords_t coords;

        template_orientations_t orientations;
        Size size;
        Point center;

    public:
        Template() : addr_width(-1)
        {
        }

        Template(Mat& edge_image, float scale_ = 1);

        ~Template()
        {
            for (size_t i=0;i<scaled_templates.size();++i) {
                delete scaled_templates[i];
            }
            scaled_templates.clear();
            coords.clear();
            orientations.clear();
        }
        void show() const;



    private:
        /**
         * Resizes a template
         *
         * @param scale Scale to be resized to
         */
        Template* rescale(float scale);

        std::vector<int>& getTemplateAddresses(int width);
    };



    /**
     * Used to represent a matching result.
     */

    class Match
    {
    public:
        float cost;
        Point offset;
        const Template* tpl;
    };

    typedef std::vector<Match> Matches;

private:
    /**
     * Implements the chamfer matching algorithm on images taking into account both distance from
     * the template pixels to the nearest pixels and orientation alignment between template and image
     * contours.
     */
    class Matching
    {
        float truncate_;
        bool use_orientation_;

        std::vector<Template*> templates;
    public:
        Matching(bool use_orientation = true, float _truncate = 10) : truncate_(_truncate), use_orientation_(use_orientation)
        {
        }

        ~Matching()
        {
            for (size_t i = 0; i<templates.size(); i++) {
                delete templates[i];
            }
        }

        /**
         * Add a template to the detector from an edge image.
         * @param templ An edge image
         */
        void addTemplateFromImage(Mat& templ, float scale = 1.0);

        /**
         * Run matching using an edge image.
         * @param edge_img Edge image
         * @return a match object
         */
        ChamferMatcher::Matches* matchEdgeImage(Mat& edge_img, const ImageRange& range, float orientation_weight = 0.5, int max_matches = 20, float min_match_distance = 10.0);

        void addTemplate(Template& template_);

    private:

        float orientation_diff(float o1, float o2)
        {
            return fabs(o1-o2);
        }

        /**
         * Computes the chamfer matching cost for one position in the target image.
         * @param offset Offset where to compute cost
         * @param dist_img Distance transform image.
         * @param orientation_img Orientation image.
         * @param tpl Template
         * @param templ_orientations Orientations of the target points.
         * @return matching result
         */
        ChamferMatcher::Match* localChamferDistance(Point offset, Mat& dist_img, Mat& orientation_img, Template* tpl,  float orientation_weight);

    private:
        /**
         * Matches all templates.
         * @param dist_img Distance transform image.
         * @param orientation_img Orientation image.
         */
        ChamferMatcher::Matches* matchTemplates(Mat& dist_img, Mat& orientation_img, const ImageRange& range, float orientation_weight);

        void computeDistanceTransform(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a, float b);
        void computeEdgeOrientations(Mat& edge_img, Mat& orientation_img);
        void fillNonContourOrientations(Mat& annotated_img, Mat& orientation_img);


    public:
        /**
         * Finds a contour in an edge image. The original image is altered by removing the found contour.
         * @param templ_img Edge image
         * @param coords Coordinates forming the contour.
         * @return True while a contour is still found in the image.
         */
        static bool findContour(Mat& templ_img, template_coords_t& coords);

        /**
         * Computes contour points orientations using the approach from:
         *
         * Matas, Shao and Kittler - Estimation of Curvature and Tangent Direction by
         * Median Filtered Differencing
         *
         * @param coords Contour points
         * @param orientations Contour points orientations
         */
        static void findContourOrientations(const template_coords_t& coords, template_orientations_t& orientations);


        /**
         * Computes the angle of a line segment.
         *
         * @param a One end of the line segment
         * @param b The other end.
         * @param dx
         * @param dy
         * @return Angle in radians.
         */
        static float getAngle(coordinate_t a, coordinate_t b, int& dx, int& dy);

        /**
         * Finds a point in the image from which to start contour following.
         * @param templ_img
         * @param p
         * @return
         */

        static bool findFirstContourPoint(Mat& templ_img, coordinate_t& p);
        /**
         * Method that extracts a single continuous contour from an image given a starting point.
         * When it extracts the contour it tries to maintain the same direction (at a T-join for example).
         *
         * @param templ_
         * @param coords
         * @param direction
         */
        static void followContour(Mat& templ_img, template_coords_t& coords, int direction);


    };




    class LocationImageIterator : public ImageIterator
    {
        const std::vector<Point>& locations_;

        size_t iter_;

        int scales_;
        float min_scale_;
        float max_scale_;

        float scale_;
        float scale_step_;
        int scale_cnt_;

        bool has_next_;

        LocationImageIterator(const LocationImageIterator&);
        LocationImageIterator& operator=(const LocationImageIterator&);

    public:
        LocationImageIterator(const std::vector<Point>& locations, int _scales, float min_scale, float max_scale);

        bool hasNext() const {
            return has_next_;
        }

        location_scale_t next();
    };

    class LocationScaleImageIterator : public ImageIterator
    {
        const std::vector<Point>& locations_;
        const std::vector<float>& scales_;

        size_t iter_;

        bool has_next_;

        LocationScaleImageIterator(const LocationScaleImageIterator&);
        LocationScaleImageIterator& operator=(const LocationScaleImageIterator&);

    public:
        LocationScaleImageIterator(const std::vector<Point>& locations, const std::vector<float>& _scales) :
        locations_(locations), scales_(_scales)
        {
            assert(locations.size()==_scales.size());
            reset();
        }

        void reset()
        {
            iter_ = 0;
            has_next_ = (locations_.size()==0 ? false : true);
        }

        bool hasNext() const {
            return has_next_;
        }

        location_scale_t next();
    };

    class SlidingWindowImageIterator : public ImageIterator
    {
        int x_;
        int y_;
        float scale_;
        float scale_step_;
        int scale_cnt_;

        bool has_next_;

        int width_;
        int height_;
        int x_step_;
        int y_step_;
        int scales_;
        float min_scale_;
        float max_scale_;


    public:

        SlidingWindowImageIterator(int width, int height, int x_step, int y_step, int scales, float min_scale, float max_scale);

        bool hasNext() const {
            return has_next_;
        }

        location_scale_t next();
    };




    int count;
    Matches matches;
    int pad_x;
    int pad_y;
    int scales;
    float minScale;
    float maxScale;
    float orientation_weight;
    float truncate;
    Matching * chamfer_;

public:
    ChamferMatcher(int _max_matches = 20, float _min_match_distance = 1.0, int _pad_x = 3,
                   int _pad_y = 3, int _scales = 5, float _minScale = 0.6, float _maxScale = 1.6,
                   float _orientation_weight = 0.5, float _truncate = 20)
    {
        max_matches_ = _max_matches;
        min_match_distance_ = _min_match_distance;
        pad_x = _pad_x;
        pad_y = _pad_y;
        scales = _scales;
        minScale = _minScale;
        maxScale = _maxScale;
        orientation_weight = _orientation_weight;
        truncate = _truncate;
        count = 0;

        matches.resize(max_matches_);
        chamfer_ = new Matching(true);
    }

    ~ChamferMatcher()
    {
        delete chamfer_;
    }

    void showMatch(Mat& img, int index = 0);
    void showMatch(Mat& img, Match match_);

    const Matches& matching(Template&, Mat&);

private:
    ChamferMatcher(const ChamferMatcher&);
    ChamferMatcher& operator=(const ChamferMatcher&);
    void addMatch(float cost, Point offset, const Template* tpl);


};


///////////////////// implementation ///////////////////////////

ChamferMatcher::SlidingWindowImageIterator::SlidingWindowImageIterator( int width,
                                                                        int height,
                                                                        int x_step = 3,
                                                                        int y_step = 3,
                                                                        int _scales = 5,
                                                                        float min_scale = 0.6,
                                                                        float max_scale = 1.6) :

                                                                            width_(width),
                                                                            height_(height),
                                                                            x_step_(x_step),
                                                                            y_step_(y_step),
                                                                            scales_(_scales),
                                                                            min_scale_(min_scale),
                                                                            max_scale_(max_scale)
{
    x_ = 0;
    y_ = 0;
    scale_cnt_ = 0;
    scale_ = min_scale_;
    has_next_ = true;
    scale_step_ = (max_scale_-min_scale_)/scales_;
}

location_scale_t ChamferMatcher::SlidingWindowImageIterator::next()
{
    location_scale_t next_val = std::make_pair(Point(x_,y_),scale_);

    x_ += x_step_;

    if (x_ >= width_) {
        x_ = 0;
        y_ += y_step_;

        if (y_ >= height_) {
            y_ = 0;
            scale_ += scale_step_;
            scale_cnt_++;

            if (scale_cnt_ == scales_) {
                has_next_ = false;
                scale_cnt_ = 0;
                scale_ = min_scale_;
            }
        }
    }

    return next_val;
}



ChamferMatcher::ImageIterator* ChamferMatcher::SlidingWindowImageRange::iterator() const
{
    return new SlidingWindowImageIterator(width_, height_, x_step_, y_step_, scales_, min_scale_, max_scale_);
}



ChamferMatcher::LocationImageIterator::LocationImageIterator(const std::vector<Point>& locations,
                                                                int _scales = 5,
                                                                float min_scale = 0.6,
                                                                float max_scale = 1.6) :
                                                                    locations_(locations),
                                                                    scales_(_scales),
                                                                    min_scale_(min_scale),
                                                                    max_scale_(max_scale)
{
    iter_ = 0;
    scale_cnt_ = 0;
    scale_ = min_scale_;
    has_next_ = (locations_.size()==0 ? false : true);
    scale_step_ = (max_scale_-min_scale_)/scales_;
}

location_scale_t ChamferMatcher::LocationImageIterator:: next()
{
    location_scale_t next_val = std::make_pair(locations_[iter_],scale_);

    iter_ ++;
    if (iter_==locations_.size()) {
        iter_ = 0;
        scale_ += scale_step_;
        scale_cnt_++;

        if (scale_cnt_ == scales_) {
            has_next_ = false;
            scale_cnt_ = 0;
            scale_ = min_scale_;
        }
    }

    return next_val;
}


location_scale_t ChamferMatcher::LocationScaleImageIterator::next()
{
    location_scale_t next_val = std::make_pair(locations_[iter_],scales_[iter_]);

    iter_ ++;
    if (iter_==locations_.size()) {
        iter_ = 0;

        has_next_ = false;
    }

    return next_val;
}



bool ChamferMatcher::Matching::findFirstContourPoint(Mat& templ_img, coordinate_t& p)
{
    for (int y=0;y<templ_img.rows;++y) {
        for (int x=0;x<templ_img.cols;++x) {
            if (templ_img.at<uchar>(y,x)!=0) {
                p.first = x;
                p.second = y;
                return true;
            }
        }
    }
    return false;
}



void ChamferMatcher::Matching::followContour(Mat& templ_img, template_coords_t& coords, int direction = -1)
{
    const int dir[][2] = { {-1,-1}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1} };
    coordinate_t next;
    unsigned char ptr;

    assert (direction==-1 || !coords.empty());

    coordinate_t crt = coords.back();

    // mark the current pixel as visited
    templ_img.at<uchar>(crt.second,crt.first) = 0;
    if (direction==-1) {
        for (int j = 0; j<7; ++j) {
            next.first = crt.first + dir[j][1];
            next.second = crt.second + dir[j][0];
            if (next.first >= 0 && next.first < templ_img.cols &&
                next.second >= 0 && next.second < templ_img.rows){
                ptr = templ_img.at<uchar>(next.second, next.first);
                if (ptr!=0) {
                    coords.push_back(next);
                    followContour(templ_img, coords,j);
                    // try to continue contour in the other direction
                    reverse(coords.begin(), coords.end());
                    followContour(templ_img, coords, (j+4)%8);
                    break;
                }
            }
        }
    }
    else {
        int k = direction;
        int k_cost = 3;
        next.first = crt.first + dir[k][1];
        next.second = crt.second + dir[k][0];
        if (next.first >= 0 && next.first < templ_img.cols &&
                next.second >= 0 && next.second < templ_img.rows){
            ptr = templ_img.at<uchar>(next.second, next.first);
            if (ptr!=0) {
                k_cost = std::abs(dir[k][1]) + std::abs(dir[k][0]);
            }
            int p = k;
            int n = k;

            for (int j = 0 ;j<3; ++j) {
                p = (p + 7) % 8;
                n = (n + 1) % 8;
                next.first = crt.first + dir[p][1];
                next.second = crt.second + dir[p][0];
                if (next.first >= 0 && next.first < templ_img.cols &&
                    next.second >= 0 && next.second < templ_img.rows){
                    ptr = templ_img.at<uchar>(next.second, next.first);
                    if (ptr!=0) {
                        int p_cost = std::abs(dir[p][1]) + std::abs(dir[p][0]);
                        if (p_cost<k_cost) {
                            k_cost = p_cost;
                            k = p;
                        }
                    }
                    next.first = crt.first + dir[n][1];
                    next.second = crt.second + dir[n][0];
                    if (next.first >= 0 && next.first < templ_img.cols &&
                    next.second >= 0 && next.second < templ_img.rows){
                        ptr = templ_img.at<uchar>(next.second, next.first);
                        if (ptr!=0) {
                            int n_cost = std::abs(dir[n][1]) + std::abs(dir[n][0]);
                            if (n_cost<k_cost) {
                                k_cost = n_cost;
                                k = n;
                            }
                        }
                    }
                }
            }

            if (k_cost!=3) {
                next.first = crt.first + dir[k][1];
                next.second = crt.second + dir[k][0];
                if (next.first >= 0 && next.first < templ_img.cols &&
                    next.second >= 0 && next.second < templ_img.rows) {
                    coords.push_back(next);
                    followContour(templ_img, coords, k);
                }
            }
        }
    }
}


bool ChamferMatcher::Matching::findContour(Mat& templ_img, template_coords_t& coords)
{
    coordinate_t start_point;

    bool found = findFirstContourPoint(templ_img,start_point);
    if (found) {
        coords.push_back(start_point);
        followContour(templ_img, coords);
        return true;
    }

    return false;
}


float ChamferMatcher::Matching::getAngle(coordinate_t a, coordinate_t b, int& dx, int& dy)
{
    dx = b.first-a.first;
    dy = -(b.second-a.second);  // in image coordinated Y axis points downward
        float angle = atan2((float)dy,(float)dx);

    if (angle<0) {
                angle+=(float)CV_PI;
    }

    return angle;
}



void ChamferMatcher::Matching::findContourOrientations(const template_coords_t& coords, template_orientations_t& orientations)
{
    const int M = 5;
    int coords_size = (int)coords.size();

    std::vector<float> angles(2*M);
        orientations.insert(orientations.begin(), coords_size, float(-3*CV_PI)); // mark as invalid in the beginning

    if (coords_size<2*M+1) {  // if contour not long enough to estimate orientations, abort
        return;
    }

    for (int i=M;i<coords_size-M;++i) {
        coordinate_t crt = coords[i];
        coordinate_t other;
        int k = 0;
        int dx, dy;
        // compute previous M angles
        for (int j=M;j>0;--j) {
            other = coords[i-j];
            angles[k++] = getAngle(other,crt, dx, dy);
        }
        // compute next M angles
        for (int j=1;j<=M;++j) {
            other = coords[i+j];
            angles[k++] = getAngle(crt, other, dx, dy);
        }

        // get the middle two angles
        std::nth_element(angles.begin(), angles.begin()+M-1,  angles.end());
        std::nth_element(angles.begin()+M-1, angles.begin()+M,  angles.end());
        //        sort(angles.begin(), angles.end());

        // average them to compute tangent
        orientations[i] = (angles[M-1]+angles[M])/2;
    }
}

//////////////////////// Template /////////////////////////////////////

ChamferMatcher::Template::Template(Mat& edge_image, float scale_) : addr_width(-1), scale(scale_)
{
    template_coords_t local_coords;
    template_orientations_t local_orientations;

    while (ChamferMatcher::Matching::findContour(edge_image, local_coords)) {
        ChamferMatcher::Matching::findContourOrientations(local_coords, local_orientations);

        coords.insert(coords.end(), local_coords.begin(), local_coords.end());
        orientations.insert(orientations.end(), local_orientations.begin(), local_orientations.end());
        local_coords.clear();
        local_orientations.clear();
    }


    size = edge_image.size();
    Point min, max;
    min.x = size.width;
    min.y = size.height;
    max.x = 0;
    max.y = 0;

    center = Point(0,0);
    for (size_t i=0;i<coords.size();++i) {
        center.x += coords[i].first;
        center.y += coords[i].second;

        if (min.x>coords[i].first) min.x = coords[i].first;
        if (min.y>coords[i].second) min.y = coords[i].second;
        if (max.x<coords[i].first) max.x = coords[i].first;
        if (max.y<coords[i].second) max.y = coords[i].second;
    }

    size.width = max.x - min.x;
    size.height = max.y - min.y;
    int coords_size = (int)coords.size();

    center.x /= MAX(coords_size, 1);
    center.y /= MAX(coords_size, 1);

    for (int i=0;i<coords_size;++i) {
        coords[i].first -= center.x;
        coords[i].second -= center.y;
    }
}


vector<int>& ChamferMatcher::Template::getTemplateAddresses(int width)
{
    if (addr_width!=width) {
        addr.resize(coords.size());
        addr_width = width;

        for (size_t i=0; i<coords.size();++i) {
            addr[i] = coords[i].second*width+coords[i].first;
        }
    }
    return addr;
}


/**
 * Resizes a template
 *
 * @param scale Scale to be resized to
 */
ChamferMatcher::Template* ChamferMatcher::Template::rescale(float new_scale)
{

    if (fabs(scale-new_scale)<1e-6) return this;

    for (size_t i=0;i<scaled_templates.size();++i) {
        if (fabs(scaled_templates[i]->scale-new_scale)<1e-6) {
            return scaled_templates[i];
        }
    }

    float scale_factor = new_scale/scale;

    Template* tpl = new Template();
    tpl->scale = new_scale;

    tpl->center.x = int(center.x*scale_factor+0.5);
    tpl->center.y = int(center.y*scale_factor+0.5);

    tpl->size.width = int(size.width*scale_factor+0.5);
    tpl->size.height = int(size.height*scale_factor+0.5);

    tpl->coords.resize(coords.size());
    tpl->orientations.resize(orientations.size());
    for (size_t i=0;i<coords.size();++i) {
        tpl->coords[i].first = int(coords[i].first*scale_factor+0.5);
        tpl->coords[i].second = int(coords[i].second*scale_factor+0.5);
        tpl->orientations[i] = orientations[i];
    }
    scaled_templates.push_back(tpl);

    return tpl;

}



void ChamferMatcher::Template::show() const
{
    int pad = 50;
    //Attention size is not correct
    Mat templ_color (Size(size.width+(pad*2), size.height+(pad*2)), CV_8UC3);
    templ_color.setTo(0);

    for (size_t i=0;i<coords.size();++i) {

        int x = center.x+coords[i].first+pad;
        int y = center.y+coords[i].second+pad;
        templ_color.at<Vec3b>(y,x)[1]=255;
        //CV_PIXEL(unsigned char, templ_color,x,y)[1] = 255;

        if (i%3==0) {
                        if (orientations[i] < -CV_PI) {
                continue;
            }
            Point p1;
            p1.x = x;
            p1.y = y;
            Point p2;
            p2.x = x + pad*(int)(sin(orientations[i])*100)/100;
            p2.y = y + pad*(int)(cos(orientations[i])*100)/100;

            line(templ_color, p1,p2, CV_RGB(255,0,0));
        }
    }

    circle(templ_color,Point(center.x + pad, center.y + pad),1,CV_RGB(0,255,0));

#ifdef HAVE_OPENCV_HIGHGUI
    namedWindow("templ",1);
    imshow("templ",templ_color);

    cvWaitKey(0);
#else
    CV_Error(CV_StsNotImplemented, "OpenCV has been compiled without GUI support");
#endif

    templ_color.release();
}


//////////////////////// Matching /////////////////////////////////////


void ChamferMatcher::Matching::addTemplateFromImage(Mat& templ, float scale)
{
    Template* cmt = new Template(templ, scale);
    templates.clear();
    templates.push_back(cmt);
    cmt->show();
}

void ChamferMatcher::Matching::addTemplate(Template& template_){
    templates.clear();
    templates.push_back(&template_);
}
/**
 * Alternative version of computeDistanceTransform, will probably be used to compute distance
 * transform annotated with edge orientation.
 */
void ChamferMatcher::Matching::computeDistanceTransform(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a = 1.0, float b = 1.5)
{
    int d[][2] = { {-1,-1}, { 0,-1}, { 1,-1},
            {-1,0},          { 1,0},
            {-1,1}, { 0,1}, { 1,1} };


    Size s = edges_img.size();
    int w = s.width;
    int h = s.height;
    // set distance to the edge pixels to 0 and put them in the queue
    std::queue<std::pair<int,int> > q;

    for (int y=0;y<h;++y) {
        for (int x=0;x<w;++x) {
            // initialize
            if (&annotate_img!=NULL) {
                annotate_img.at<Vec2i>(y,x)[0]=x;
                annotate_img.at<Vec2i>(y,x)[1]=y;
            }

            uchar edge_val = edges_img.at<uchar>(y,x);
            if( (edge_val!=0) ) {
                q.push(std::make_pair(x,y));
                dist_img.at<float>(y,x)= 0;
            }
            else {
                dist_img.at<float>(y,x)=-1;
            }
        }
    }

    // breadth first computation of distance transform
    std::pair<int,int> crt;
    while (!q.empty()) {
        crt = q.front();
        q.pop();

        int x = crt.first;
        int y = crt.second;

        float dist_orig = dist_img.at<float>(y,x);
        float dist;

        for (size_t i=0;i<sizeof(d)/sizeof(d[0]);++i) {
            int nx = x + d[i][0];
            int ny = y + d[i][1];

            if (nx<0 || ny<0 || nx>=w || ny>=h) continue;

            if (std::abs(d[i][0]+d[i][1])==1) {
                dist = (dist_orig)+a;
            }
            else {
                dist = (dist_orig)+b;
            }

            float dt = dist_img.at<float>(ny,nx);

            if (dt==-1 || dt>dist) {
                dist_img.at<float>(ny,nx) = dist;
                q.push(std::make_pair(nx,ny));

                if (&annotate_img!=NULL) {
                    annotate_img.at<Vec2i>(ny,nx)[0]=annotate_img.at<Vec2i>(y,x)[0];
                    annotate_img.at<Vec2i>(ny,nx)[1]=annotate_img.at<Vec2i>(y,x)[1];
                }
            }
        }
    }
    // truncate dt

    if (truncate_dt>0) {
        Mat dist_img_thr = dist_img.clone();
        threshold(dist_img, dist_img_thr, truncate_dt,0.0 ,THRESH_TRUNC);
        dist_img_thr.copyTo(dist_img);
    }
}


void ChamferMatcher::Matching::computeEdgeOrientations(Mat& edge_img, Mat& orientation_img)
{
    Mat contour_img(edge_img.size(), CV_8UC1);

        orientation_img.setTo(3*(-CV_PI));
    template_coords_t coords;
    template_orientations_t orientations;

    while (ChamferMatcher::Matching::findContour(edge_img, coords)) {

        ChamferMatcher::Matching::findContourOrientations(coords, orientations);

        // set orientation pixel in orientation image
        for (size_t i = 0; i<coords.size();++i) {
            int x = coords[i].first;
            int y = coords[i].second;
                        //            if (orientations[i]>-CV_PI)
            //    {
            //CV_PIXEL(unsigned char, contour_img, x, y)[0] = 255;
            contour_img.at<uchar>(y,x)=255;
            //    }
            //CV_PIXEL(float, orientation_img, x, y)[0] = orientations[i];
            orientation_img.at<float>(y,x)=orientations[i];
        }


        coords.clear();
        orientations.clear();
    }

    //imwrite("contours.pgm", contour_img);
}


void ChamferMatcher::Matching::fillNonContourOrientations(Mat& annotated_img, Mat& orientation_img)
{
    int cols = annotated_img.cols;
    int rows = annotated_img.rows;

    assert(orientation_img.cols==cols && orientation_img.rows==rows);

    for (int y=0;y<rows;++y) {
        for (int x=0;x<cols;++x) {
            int xorig = annotated_img.at<Vec2i>(y,x)[0];
            int yorig = annotated_img.at<Vec2i>(y,x)[1];

            if (x!=xorig || y!=yorig) {
                //orientation_img.at<float>(yorig,xorig)=orientation_img.at<float>(y,x);
                orientation_img.at<float>(y,x)=orientation_img.at<float>(yorig,xorig);
            }
        }
    }
}


ChamferMatcher::Match* ChamferMatcher::Matching::localChamferDistance(Point offset, Mat& dist_img, Mat& orientation_img,
        ChamferMatcher::Template* tpl, float alpha)
{
    int x = offset.x;
    int y = offset.y;

    float beta = 1-alpha;

    std::vector<int>& addr = tpl->getTemplateAddresses(dist_img.cols);

    float* ptr = dist_img.ptr<float>(y)+x;


    float sum_distance = 0;
    for (size_t i=0; i<addr.size();++i) {
        if(addr[i] < (dist_img.cols*dist_img.rows) - (offset.y*dist_img.cols + offset.x)){
            sum_distance += *(ptr+addr[i]);
        }
    }

    float cost = (sum_distance/truncate_)/addr.size();


    if (&orientation_img!=NULL) {
        float* optr = orientation_img.ptr<float>(y)+x;
        float sum_orientation = 0;
        int cnt_orientation = 0;

        for (size_t i=0;i<addr.size();++i) {

            if(addr[i] < (orientation_img.cols*orientation_img.rows) - (offset.y*orientation_img.cols + offset.x)){
                                if (tpl->orientations[i]>=-CV_PI && (*(optr+addr[i]))>=-CV_PI) {
                    sum_orientation += orientation_diff(tpl->orientations[i], (*(optr+addr[i])));
                    cnt_orientation++;
                }
            }
        }

        if (cnt_orientation>0) {
                        cost = (float)(beta*cost+alpha*(sum_orientation/(2*CV_PI))/cnt_orientation);
        }

    }

    if(cost > 0){
        ChamferMatcher::Match* istance = new ChamferMatcher::Match();
        istance->cost = cost;
        istance->offset = offset;
        istance->tpl = tpl;

        return istance;
    }

    return NULL;
}


ChamferMatcher::Matches* ChamferMatcher::Matching::matchTemplates(Mat& dist_img, Mat& orientation_img, const ImageRange& range, float _orientation_weight)
{

    ChamferMatcher::Matches* pmatches(new Matches());
    // try each template
    for(size_t i = 0; i < templates.size(); i++) {
        ImageIterator* it = range.iterator();
        while (it->hasNext()) {
            location_scale_t crt = it->next();

            Point loc = crt.first;
            float scale = crt.second;
            Template* tpl = templates[i]->rescale(scale);


            if (loc.x-tpl->center.x<0 || loc.x+tpl->size.width/2>=dist_img.cols) continue;
            if (loc.y-tpl->center.y<0 || loc.y+tpl->size.height/2>=dist_img.rows) continue;

            ChamferMatcher::Match* is = localChamferDistance(loc, dist_img, orientation_img, tpl, _orientation_weight);
            if(is)
            {
                pmatches->push_back(*is);
                delete is;
            }
        }

        delete it;
    }
    return pmatches;
}



/**
 * Run matching using an edge image.
 * @param edge_img Edge image
 * @return a match object
 */
ChamferMatcher::Matches* ChamferMatcher::Matching::matchEdgeImage(Mat& edge_img, const ImageRange& range,
                    float _orientation_weight, int /*max_matches*/, float /*min_match_distance*/)
{
    CV_Assert(edge_img.channels()==1);

    Mat dist_img;
    Mat annotated_img;
    Mat orientation_img;

    annotated_img.create(edge_img.size(), CV_32SC2);
    dist_img.create(edge_img.size(),CV_32FC1);
    dist_img.setTo(0);
    // Computing distance transform
    computeDistanceTransform(edge_img,dist_img, annotated_img, truncate_);


    //orientation_img = NULL;
    if (use_orientation_) {
        orientation_img.create(edge_img.size(), CV_32FC1);
        orientation_img.setTo(0);
        Mat edge_clone = edge_img.clone();
        computeEdgeOrientations(edge_clone, orientation_img );
        edge_clone.release();
        fillNonContourOrientations(annotated_img, orientation_img);
    }


    // Template matching
    ChamferMatcher::Matches* pmatches = matchTemplates(    dist_img,
                                                        orientation_img,
                                                        range,
                                                        _orientation_weight);


    if (use_orientation_) {
        orientation_img.release();
    }
    dist_img.release();
    annotated_img.release();

    return pmatches;
}


void ChamferMatcher::addMatch(float cost, Point offset, const Template* tpl)
{
    bool new_match = true;
    for (int i=0; i<count; ++i) {
        if (std::abs(matches[i].offset.x-offset.x)+std::abs(matches[i].offset.y-offset.y)<min_match_distance_) {
            // too close, not a new match
            new_match = false;
            // if better cost, replace existing match
            if (cost<matches[i].cost) {
                matches[i].cost = cost;
                matches[i].offset = offset;
                matches[i].tpl = tpl;
            }
            // re-bubble to keep ordered
            int k = i;
            while (k>0) {
                if (matches[k-1].cost>matches[k].cost) {
                    std::swap(matches[k-1],matches[k]);
                }
                k--;
            }

            break;
        }
    }

    if (new_match) {
        // if we don't have enough matches yet, add it to the array
        if (count<max_matches_) {
            matches[count].cost = cost;
            matches[count].offset = offset;
            matches[count].tpl = tpl;
            count++;
        }
        // otherwise find the right position to insert it
        else {
            // if higher cost than the worst current match, just ignore it
            if (matches[count-1].cost<cost) {
                return;
            }

            int j = 0;
            // skip all matches better than current one
            while (matches[j].cost<cost) j++;

            // shift matches one position
            int k = count-2;
            while (k>=j) {
                matches[k+1] = matches[k];
                k--;
            }

            matches[j].cost = cost;
            matches[j].offset = offset;
            matches[j].tpl = tpl;
        }
    }
}

void ChamferMatcher::showMatch(Mat& img, int index)
{
    if (index>=count) {
        std::cout << "Index too big.\n" << std::endl;
    }

    assert(img.channels()==3);

    Match match = matches[index];

    const template_coords_t& templ_coords = match.tpl->coords;
    int x, y;
    for (size_t i=0;i<templ_coords.size();++i) {
        x = match.offset.x + templ_coords[i].first;
        y = match.offset.y + templ_coords[i].second;

        if ( x > img.cols-1 || x < 0 || y > img.rows-1 || y < 0) continue;
        img.at<Vec3b>(y,x)[0]=0;
        img.at<Vec3b>(y,x)[2]=0;
        img.at<Vec3b>(y,x)[1]=255;
    }
}

void ChamferMatcher::showMatch(Mat& img, Match match)
{
    assert(img.channels()==3);

    const template_coords_t& templ_coords = match.tpl->coords;
    for (size_t i=0;i<templ_coords.size();++i) {
        int x = match.offset.x + templ_coords[i].first;
        int y = match.offset.y + templ_coords[i].second;
        if ( x > img.cols-1 || x < 0 || y > img.rows-1 || y < 0) continue;
        img.at<Vec3b>(y,x)[0]=0;
        img.at<Vec3b>(y,x)[2]=0;
        img.at<Vec3b>(y,x)[1]=255;
    }
    match.tpl->show();
}

const ChamferMatcher::Matches& ChamferMatcher::matching(Template& tpl, Mat& image_){
    chamfer_->addTemplate(tpl);

    matches.clear();
    matches.resize(max_matches_);
    count = 0;


    Matches* matches_ = chamfer_->matchEdgeImage(    image_,
                                                    ChamferMatcher::
                                                        SlidingWindowImageRange(image_.cols,
                                                                                image_.rows,
                                                                                pad_x,
                                                                                pad_y,
                                                                                scales,
                                                                                minScale,
                                                                                maxScale),
                                                    orientation_weight,
                                                    max_matches_,
                                                    min_match_distance_);



    for(int i = 0; i < (int)matches_->size(); i++){
        addMatch(matches_->at(i).cost, matches_->at(i).offset, matches_->at(i).tpl);
    }

    matches_->clear();
    delete matches_;
    matches_ = NULL;

    matches.resize(count);


    return matches;

}


int chamerMatching( Mat& img, Mat& templ,
                    std::vector<std::vector<Point> >& results, std::vector<float>& costs,
                    double templScale, int maxMatches, double minMatchDistance, int padX,
                    int padY, int scales, double minScale, double maxScale,
                    double orientationWeight, double truncate )
{
    CV_Assert(img.type() == CV_8UC1 && templ.type() == CV_8UC1);

    ChamferMatcher matcher_(maxMatches, (float)minMatchDistance, padX, padY, scales,
                            (float)minScale, (float)maxScale,
                            (float)orientationWeight, (float)truncate);

    ChamferMatcher::Template template_(templ, (float)templScale);
    ChamferMatcher::Matches match_instances = matcher_.matching(template_, img);

    size_t i, nmatches = match_instances.size();

    results.resize(nmatches);
    costs.resize(nmatches);

    int bestIdx = -1;
    double minCost = DBL_MAX;

    for( i = 0; i < nmatches; i++ )
    {
        const ChamferMatcher::Match& match = match_instances[i];
        double cval = match.cost;
        if( cval < minCost)
        {
            minCost = cval;
            bestIdx = (int)i;
        }
        costs[i] = (float)cval;

        const template_coords_t& templ_coords = match.tpl->coords;
        std::vector<Point>& templPoints = results[i];
        size_t j, npoints = templ_coords.size();
        templPoints.resize(npoints);

        for (j = 0; j < npoints; j++ )
        {
            int x = match.offset.x + templ_coords[j].first;
            int y = match.offset.y + templ_coords[j].second;
            templPoints[j] = Point(x,y);
        }
    }

    return bestIdx;
}

}
