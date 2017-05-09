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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
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

#include "precomp.hpp"
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////
// Default LSD parameters
// SIGMA_SCALE 0.6    - Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
// QUANT       2.0    - Bound to the quantization error on the gradient norm.
// ANG_TH      22.5   - Gradient angle tolerance in degrees.
// LOG_EPS     0.0    - Detection threshold: -log10(NFA) > log_eps
// DENSITY_TH  0.7    - Minimal density of region points in rectangle.
// N_BINS      1024   - Number of bins in pseudo-ordering of gradient modulus.

struct SEGMENT
{
  float x1, y1, x2, y2, angle;
};

/////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv{

  class FastLineDetectorImpl : public FastLineDetector
  {
    public:

      /**
       * Create a LineSegmentDetectorImpl object. Specifying scale, number of subdivisions for the image, should the lines be refined and other constants as follows:
       *
       * @param _refine       How should the lines found be refined?
       *                      LSD_REFINE_NONE - No refinement applied.
       *                      LSD_REFINE_STD  - Standard refinement is applied. E.g. breaking arches into smaller line approximations.
       *                      LSD_REFINE_ADV  - Advanced refinement. Number of false alarms is calculated,
       *                                    lines are refined through increase of precision, decrement in size, etc.
       * @param _scale        The scale of the image that will be used to find the lines. Range (0..1].
       * @param _sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
       * @param _quant        Bound to the quantization error on the gradient norm.
       * @param _ang_th       Gradient angle tolerance in degrees.
       * @param _log_eps      Detection threshold: -log10(NFA) > _log_eps
       * @param _density_th   Minimal density of aligned region points in rectangle.
       * @param _n_bins       Number of bins in pseudo-ordering of gradient modulus.
       */
      FastLineDetectorImpl(int _length_threshold = 10, float _distance_threshold = 1.6f);

      /**
       * Detect lines in the input image.
       *
       * @param _image    A grayscale(CV_8UC1) input image.
       *                  If only a roi needs to be selected, use
       *                  lsd_ptr->detect(image(roi), ..., lines);
       *                  lines += Scalar(roi.x, roi.y, roi.x, roi.y);
       * @param _lines    Return: A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line.
       *                          Where Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
       *                          Returned lines are strictly oriented depending on the gradient.
       * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
       * @param prec      Return: Vector of precisions with which the lines are found.
       * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%.
       *                          The bigger the value, logarithmically better the detection.
       *                              * -1 corresponds to 10 mean false alarms
       *                              * 0 corresponds to 1 mean false alarm
       *                              * 1 corresponds to 0.1 mean false alarms
       *                          This vector will be calculated _only_ when the objects type is REFINE_ADV
       */
      void detect(InputArray _image, OutputArray _lines);
      // OutputArray width = noArray(), OutputArray prec = noArray(),
      // OutputArray nfa = noArray());

      /**
       * Draw lines on the given canvas.
       *
       * @param image     The image, where lines will be drawn.
       *                  Should have the size of the image, where the lines were found
       * @param lines     The lines that need to be drawn
       */
      void drawSegments(InputOutputArray _image, InputArray lines);

    private:
      int imagewidth, imageheight, threshold_length;
      float threshold_dist;

      FastLineDetectorImpl& operator= (const FastLineDetectorImpl&); // to quiet MSVC
      template<class T>
        void incidentPoint(const Mat& l, T& pt);

      void mergeLines(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged);

      bool mergeSegments(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged);

      int getNumBranch( const Mat& img, const Point pt);

      bool getPointChain( const Mat& img, Point pt, Point& chained_pt, int& direction, int step );

      double distPointLine( const Mat& p, Mat& l );

      void extractSegments(const std::vector<Point2i>& points, std::vector<SEGMENT>& segments );

      void lineDetection(const Mat& src, std::vector<SEGMENT>& segments_all, bool merge = false );

      void pointInboardTest(const Mat& src, Point2i& pt);

      void getAngle(SEGMENT& seg);

      void additionalOperationsOnSegments(const Mat& src, SEGMENT& seg);

      void drawSegment(Mat& mat, const SEGMENT& seg, Scalar bgr = Scalar(0,255,0),
          int thickness = 1, bool directed = true);
  };

  /////////////////////////////////////////////////////////////////////////////////////////

  CV_EXPORTS Ptr<FastLineDetector> createFastLineDetector(
      int _length_threshold, float _distance_threshold)
  {
    return makePtr<FastLineDetectorImpl>(
        _length_threshold, _distance_threshold);
  }

  /////////////////////////////////////////////////////////////////////////////////////////

  FastLineDetectorImpl::FastLineDetectorImpl(int _length_threshold, float _distance_threshold)
    :threshold_length(_length_threshold), threshold_dist(_distance_threshold)
  {
    CV_Assert(_length_threshold > 0 && _distance_threshold > 0);
  }

  void FastLineDetectorImpl::detect(InputArray _image, OutputArray _lines)
  {
    CV_INSTRUMENT_REGION()

      Mat image = _image.getMat();
    CV_Assert(!image.empty() && image.type() == CV_8UC1);

    std::vector<Vec4f> lines;
    std::vector<SEGMENT> segments;
    lineDetection(image, segments);
    for(size_t i = 0; i < segments.size(); ++i)
    {
      const SEGMENT seg = segments[i];
      Vec4f line(seg.x1, seg.y1, seg.x2, seg.y2);
      lines.push_back(line);
    }
    Mat(lines).copyTo(_lines);
  }

  void FastLineDetectorImpl::drawSegments(InputOutputArray _image, InputArray lines)
  {
    CV_INSTRUMENT_REGION()

      CV_Assert(!_image.empty() && (_image.channels() == 1 || _image.channels() == 3));

    Mat gray;
    if (_image.channels() == 1)
    {
      gray = _image.getMatRef();
    }
    else if (_image.channels() == 3)
    {
      cvtColor(_image, gray, CV_BGR2GRAY);
    }

    // Create a 3 channel image in order to draw colored lines
    std::vector<Mat> planes;
    planes.push_back(gray);
    planes.push_back(gray);
    planes.push_back(gray);

    merge(planes, _image);

    Mat _lines;
    _lines = lines.getMat();
    int N = _lines.checkVector(4);

    // Draw segments
    for(int i = 0; i < N; ++i)
    {
      const Vec4f& v = _lines.at<Vec4f>(i);
      Point2f b(v[0], v[1]);
      Point2f e(v[2], v[3]);
      line(_image.getMatRef(), b, e, Scalar(0, 0, 255), 1);
    }
  }

  void FastLineDetectorImpl::mergeLines(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
  {
    double xg = 0.0, yg = 0.0;
    double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
    float ax = 0, bx = 0, cx = 0, dx = 0;
    float ay = 0, by = 0, cy = 0, dy = 0;
    double li = 0.0, lj = 0.0;
    double thi = 0.0, thj = 0.0, thr = 0.0;
    double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

    ax = seg1.x1;
    ay = seg1.y1;

    bx = seg1.x2;
    by = seg1.y2;
    cx = seg2.x1;
    cy = seg2.y1;

    dx = seg2.x2;
    dy = seg2.y2;

    float dlix = (bx - ax);
    float dliy = (by - ay);
    float dljx = (dx - cx);
    float dljy = (dy - cy);

    li = sqrt((double) (dlix * dlix) + (double) (dliy * dliy));
    lj = sqrt((double) (dljx * dljx) + (double) (dljy * dljy));

    xg = (li * (double) (ax + bx) + lj * (double) (cx + dx))
      / (double) (2.0 * (li + lj));
    yg = (li * (double) (ay + by) + lj * (double) (cy + dy))
      / (double) (2.0 * (li + lj));

    if(dlix == 0.0f) thi = CV_PI / 2.0;
    else thi = atan(dliy / dlix);

    if(dljx == 0.0f) thj = CV_PI / 2.0;
    else thj = atan(dljy / dljx);

    if (fabs(thi - thj) <= CV_PI / 2.0)
    {
      thr = (li * thi + lj * thj) / (li + lj);
    }
    else
    {
      double tmp = thj - CV_PI * (thj / fabs(thj));
      thr = li * thi + lj * tmp;
      thr /= (li + lj);
    }

    axg = ((double) ay - yg) * sin(thr) + ((double) ax - xg) * cos(thr);
    bxg = ((double) by - yg) * sin(thr) + ((double) bx - xg) * cos(thr);
    cxg = ((double) cy - yg) * sin(thr) + ((double) cx - xg) * cos(thr);
    dxg = ((double) dy - yg) * sin(thr) + ((double) dx - xg) * cos(thr);

    delta1xg = min(axg,min(bxg,min(cxg,dxg)));
    delta2xg = max(axg,max(bxg,max(cxg,dxg)));

    delta1x = delta1xg * cos(thr) + xg;
    delta1y = delta1xg * sin(thr) + yg;
    delta2x = delta2xg * cos(thr) + xg;
    delta2y = delta2xg * sin(thr) + yg;

    seg_merged.x1 = (float)delta1x;
    seg_merged.y1 = (float)delta1y;
    seg_merged.x2 = (float)delta2x;
    seg_merged.y2 = (float)delta2y;
  }

  double FastLineDetectorImpl::distPointLine(const Mat& p, Mat& l)
  {
    double x = l.at<double>(0,0);
    double y = l.at<double>(1,0);
    double w = sqrt(x*x+y*y);

    l.at<double>(0,0) = x / w;
    l.at<double>(1,0) = y / w;
    l.at<double>(2,0) = l.at<double>(2,0) / w;

    return l.dot(p);
  }

  bool FastLineDetectorImpl::mergeSegments(const SEGMENT& seg1, const SEGMENT& seg2, SEGMENT& seg_merged)
  {
    double o[] = { 0.0, 0.0, 1.0 };
    double a[] = { 0.0, 0.0, 1.0 };
    double b[] = { 0.0, 0.0, 1.0 };
    double c[3];

    o[0] = ( seg2.x1 + seg2.x2 ) / 2.0;
    o[1] = ( seg2.y1 + seg2.y2 ) / 2.0;

    a[0] = seg1.x1;
    a[1] = seg1.y1;
    b[0] = seg1.x2;
    b[1] = seg1.y2;

    Mat ori = Mat(3, 1, CV_64FC1, o).clone();
    Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
    Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
    Mat l1 = Mat(3, 1, CV_64FC1, c).clone();

    l1 = p1.cross(p2);

    Point2f seg1mid, seg2mid;
    seg1mid.x = (seg1.x1 + seg1.x2) /2.0f;
    seg1mid.y = (seg1.y1 + seg1.y2) /2.0f;
    seg2mid.x = (seg2.x1 + seg2.x2) /2.0f;
    seg2mid.y = (seg2.y1 + seg2.y2) /2.0f;

    double seg1len, seg2len;
    seg1len = sqrt((seg1.x1 - seg1.x2)*(seg1.x1 - seg1.x2)+(seg1.y1 - seg1.y2)*(seg1.y1 - seg1.y2));
    seg2len = sqrt((seg2.x1 - seg2.x2)*(seg2.x1 - seg2.x2)+(seg2.y1 - seg2.y2)*(seg2.y1 - seg2.y2));

    double middist = sqrt((seg1mid.x - seg2mid.x)*(seg1mid.x - seg2mid.x) + (seg1mid.y - seg2mid.y)*(seg1mid.y - seg2mid.y));

    float angdiff = seg1.angle - seg2.angle;
    angdiff = fabs(angdiff);

    double dist = distPointLine(ori, l1);

    if ( fabs( dist ) <= threshold_dist * 2.0
        && middist <= seg1len / 2.0 + seg2len / 2.0 + 20.0
        && angdiff <= CV_PI / 180.0f * 5.0f) {
      mergeLines(seg1, seg2, seg_merged);
      return true;
    } else {
      return false;
    }
  }

  template<class T>
    void FastLineDetectorImpl::incidentPoint(const Mat& l, T& pt)
    {
      double a[] = { (double)pt.x, (double)pt.y, 1.0 };
      double b[] = { l.at<double>(0,0), l.at<double>(1,0), 0.0 };
      double c[3];

      Mat xk = Mat(3, 1, CV_64FC1, a).clone();
      Mat lh = Mat(3, 1, CV_64FC1, b).clone();
      Mat lk = Mat(3, 1, CV_64FC1, c).clone();

      lk = xk.cross(lh);
      xk = lk.cross(l);

      xk.convertTo(xk, -1, 1.0 / xk.at<double>(2,0));

      pt.x = (float)xk.at<double>(0,0) < 0.0f ? 0.0f : (float)xk.at<double>(0,0)
        >= (imagewidth - 1.0f) ? (imagewidth - 1.0f) : (float)xk.at<double>(0,0);
      pt.y = (float)xk.at<double>(1,0) < 0.0f ? 0.0f : (float)xk.at<double>(1,0)
        >= (imageheight - 1.0f) ? (imageheight - 1.0f) : (float)xk.at<double>(1,0);
    }

  void FastLineDetectorImpl::extractSegments(const std::vector<Point2i>& points, std::vector<SEGMENT>& segments )
  {
    bool is_line;

    int i, j;
    SEGMENT seg;
    Point2i ps, pe, pt;

    std::vector<Point2i> l_points;

    int total = points.size();

    for ( i = 0; i + threshold_length < total; i++ ) {
      ps = points[i];
      pe = points[i + threshold_length];

      double a[] = { (double)ps.x, (double)ps.y, 1 };
      double b[] = { (double)pe.x, (double)pe.y, 1 };
      double c[3], d[3];

      Mat p1 = Mat(3, 1, CV_64FC1, a).clone();
      Mat p2 = Mat(3, 1, CV_64FC1, b).clone();
      Mat p = Mat(3, 1, CV_64FC1, c).clone();
      Mat l = Mat(3, 1, CV_64FC1, d).clone();
      l = p1.cross(p2);

      is_line = true;

      l_points.clear();
      l_points.push_back(ps);

      for ( j = 1; j < threshold_length; j++ ) {
        pt.x = points[i+j].x;
        pt.y = points[i+j].y;

        p.at<double>(0,0) = (double)pt.x;
        p.at<double>(1,0) = (double)pt.y;
        p.at<double>(2,0) = 1.0;

        double dist = distPointLine(p, l);

        if ( fabs( dist ) > threshold_dist ) {
          is_line = false;
          break;
        }
        l_points.push_back(pt);
      }

      // Line check fail, test next point
      if ( is_line == false )
        continue;

      l_points.push_back(pe);

      Vec4f line;
      fitLine( Mat(l_points), line, CV_DIST_L2, 0, 0.01, 0.01);
      a[0] = line[2];
      a[1] = line[3];
      b[0] = line[2] + line[0];
      b[1] = line[3] + line[1];

      p1 = Mat(3, 1, CV_64FC1, a).clone();
      p2 = Mat(3, 1, CV_64FC1, b).clone();

      l = p1.cross(p2);

      incidentPoint(l, ps);

      // Extending line
      for ( j = threshold_length + 1; i + j < total; j++ ) {
        pt.x = points[i+j].x;
        pt.y = points[i+j].y;

        p.at<double>(0,0) = (double)pt.x;
        p.at<double>(1,0) = (double)pt.y;
        p.at<double>(2,0) = 1.0;

        double dist = distPointLine(p, l);

        if ( fabs( dist ) > threshold_dist ) {
          j--;
          break;
        }
        pe = pt;
        l_points.push_back(pt);
      }
      fitLine( Mat(l_points), line, CV_DIST_L2, 0, 0.01, 0.01);
      a[0] = line[2];
      a[1] = line[3];
      b[0] = line[2] + line[0];
      b[1] = line[3] + line[1];

      p1 = Mat(3, 1, CV_64FC1, a).clone();
      p2 = Mat(3, 1, CV_64FC1, b).clone();

      l = p1.cross(p2);

      Point2f e1, e2;
      e1.x = ps.x;
      e1.y = ps.y;
      e2.x = pe.x;
      e2.y = pe.y;

      incidentPoint(l, e1);
      incidentPoint(l, e2);
      seg.x1 = e1.x;
      seg.y1 = e1.y;
      seg.x2 = e2.x;
      seg.y2 = e2.y;

      segments.push_back(seg);
      i = i + j;
    }
  }

  void FastLineDetectorImpl::pointInboardTest(const Mat& src, Point2i& pt)
  {
    pt.x = pt.x <= 5.0f ? 5.0f : pt.x >= src.cols - 5.0f ? src.cols - 5.0f : pt.x;
    pt.y = pt.y <= 5.0f ? 5.0f : pt.y >= src.rows - 5.0f ? src.rows - 5.0f : pt.y;
  }

  int FastLineDetectorImpl::getNumBranch( const Mat & img, const Point pt)
  {
    int ri, ci;
    int indices[8][2]={ {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1},{-1,0}, {-1,1}, {0,1} };

    int num_branch = 0;
    for ( int i = 0; i < 8; i++ ) {
      ci = pt.x + indices[i][1];
      ri = pt.y + indices[i][0];

      if ( ri < 0 || ri == img.rows || ci < 0 || ci == img.cols )
        continue;

      if ( img.at<unsigned char>(ri, ci) == 0 )
        continue;
      num_branch++;
    }
    return num_branch;
  }

  bool FastLineDetectorImpl::getPointChain( const Mat& img, Point pt, Point& chained_pt, int& direction, int step )
  {
    int ri, ci;
    int indices[8][2]={ {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1},{-1,0}, {-1,1}, {0,1} };

    int min_dir_diff = 7;
    Point consistent_pt;
    int consistent_direction;
    for ( int i = 0; i < 8; i++ ) {
      ci = pt.x + indices[i][1];
      ri = pt.y + indices[i][0];

      if ( ri < 0 || ri == img.rows || ci < 0 || ci == img.cols )
        continue;

      if ( img.at<unsigned char>(ri, ci) == 0 )
        continue;

      if(step == 0) {
        chained_pt.x = ci;
        chained_pt.y = ri;
        direction = i;
        return true;
      }
      else {
        int dir_diff = abs(i - direction);
        dir_diff = dir_diff > 4 ? 8 - dir_diff : dir_diff;
        if(dir_diff <= min_dir_diff)
        {
          min_dir_diff = dir_diff;
          consistent_pt.x = ci;
          consistent_pt.y = ri;
          consistent_direction = i;
        }
      }
    }
    if(min_dir_diff <= 2) {
      chained_pt.x = consistent_pt.x;
      chained_pt.y = consistent_pt.y;
      direction = consistent_direction;
      return true;
    }
    return false;
  }

  void FastLineDetectorImpl::lineDetection(const Mat& src, std::vector<SEGMENT>& segments_all, bool merge )
  {
    int r, c;
    imageheight=src.rows; imagewidth=src.cols;

    std::vector<Point2i> points;
    std::vector<SEGMENT> segments, segments_tmp;
    Mat canny = src.clone();
    Canny(src, canny, 50, 50, 3);
    // imshow("Canny", canny);
    // moveWindow("Canny", 100, 100);

    canny.colRange(0,6).rowRange(0,6) = 0;
    canny.colRange(src.cols-5,src.cols).rowRange(src.rows-5,src.rows) = 0;

    SEGMENT seg, seg1, seg2;

    for ( r = 0; r < imageheight; r++ ) {
      for ( c = 0; c < imagewidth; c++ ) {
        // Find seeds - skip for non-seeds
        if ( canny.at<unsigned char>(r,c) == 0 )
          continue;

        // Found seeds
        Point2i pt = Point2i(c,r);

        points.push_back(pt);
        canny.at<unsigned char>(pt.y, pt.x) = 0;

        int direction = 0;
        int step = 0;
        while(getPointChain(canny, pt, pt, direction, step)) {
          points.push_back(pt);
          step++;
          canny.at<unsigned char>(pt.y, pt.x) = 0;
        }

        if ( points.size() < (unsigned int)threshold_length + 1 ) {
          points.clear();
          continue;
        }

        extractSegments(points, segments);

        if ( segments.size() == 0 ) {
          points.clear();
          continue;
        }
        for ( int i = 0; i < (int)segments.size(); i++ ) {
          seg = segments[i];
          float length = sqrt((seg.x1 - seg.x2)*(seg.x1 - seg.x2) + (seg.y1 - seg.y2)*(seg.y1 - seg.y2));
          if(length < threshold_length)
            continue;
          if( (seg.x1 <= 5.0f && seg.x2 <= 5.0f)
              || (seg.y1 <= 5.0f && seg.y2 <= 5.0f)
              || (seg.x1 >= imagewidth - 5.0f && seg.x2 >= imagewidth - 5.0f)
              || (seg.y1 >= imageheight - 5.0f && seg.y2 >= imageheight - 5.0f) )
            continue;

          additionalOperationsOnSegments(src, seg);
          if(!merge) {
            segments_all.push_back(seg);
          }
          segments_tmp.push_back(seg);
        }
        points.clear();
        segments.clear();
      }
    }
    if(!merge)
      return;

    bool is_merged = false;
    int ith = segments_tmp.size() - 1;
    int jth = ith - 1;
    while(true)
    {
      seg1 = segments_tmp[ith];
      seg2 = segments_tmp[jth];
      SEGMENT seg_merged;
      is_merged = mergeSegments(seg1, seg2, seg_merged);
      if(is_merged == true)
      {
        seg2 = seg_merged;
        additionalOperationsOnSegments(src, seg2);
        std::vector<SEGMENT>::iterator it = segments_tmp.begin() + ith;
        *it = seg2;
        segments_tmp.erase(segments_tmp.begin()+jth);
        ith--;
        jth = ith - 1;
      }
      else
      {
        jth--;
      }
      if(jth < 0) {
        ith--;
        jth = ith - 1;
      }
      if(ith == 1 && jth == 0)
        break;
    }
    segments_all = segments_tmp;
  }

  void FastLineDetectorImpl::getAngle(SEGMENT& seg)
  {
    float dx = (float)(seg.x2 - seg.x1);
    float dy = (float)(seg.y2 - seg.y1);
    double ang = 0.0;

    if(dx == 0.0f) {
      if(dy > 0)
        ang = CV_PI / 2.0;
      else
        ang = -1.0 * CV_PI / 2.0;
    }
    else if(dy == 0.0f) {
      if(dx > 0)
        ang = 0.0;
      else
        ang = CV_PI;
    }
    else if(dx < 0.0f && dy > 0.0f)
      ang = CV_PI + atan( dy/dx );
    else if(dx > 0.0f && dy < 0.0f)
      ang = 2*CV_PI + atan( dy/dx );
    else if(dx < 0.0f && dy < 0.0f)
      ang = CV_PI + atan( dy/dx );
    else
      ang = atan( dy/dx );

    if(ang > 2.0 * CV_PI)
      ang -= 2.0 * CV_PI;
    seg.angle = (float)ang;
  }

  void FastLineDetectorImpl::additionalOperationsOnSegments(const Mat& src, SEGMENT& seg)
  {
    if(seg.x1 == 0.0f && seg.x2 == 0.0f && seg.y1 == 0.0f && seg.y2 == 0.0f)
      return;

    getAngle(seg);
    double ang = (double)seg.angle;

    Point2f start = Point2f(seg.x1, seg.y1);
    Point2f end = Point2f(seg.x2, seg.y2);

    double dx = 0.0, dy = 0.0;
    dx = (double) end.x - (double) start.x;
    dy = (double) end.y - (double) start.y;

    int num_points = 10;
    Point2f *points = new Point2f[num_points];

    points[0] = start;
    points[num_points - 1] = end;
    for (int i = 0; i < num_points; i++) {
      if (i == 0 || i == num_points - 1)
        continue;
      points[i].x = points[0].x + (dx / double(num_points - 1) * (double) i);
      points[i].y = points[0].y + (dy / double(num_points - 1) * (double) i);
    }

    Point2i *points_right = new Point2i[num_points];
    Point2i *points_left = new Point2i[num_points];
    double gap = 1.0;

    for(int i = 0; i < num_points; i++) {
      points_right[i].x = cvRound(points[i].x + gap*cos(90.0 * CV_PI / 180.0 + ang));
      points_right[i].y = cvRound(points[i].y + gap*sin(90.0 * CV_PI / 180.0 + ang));
      points_left[i].x = cvRound(points[i].x - gap*cos(90.0 * CV_PI / 180.0 + ang));
      points_left[i].y = cvRound(points[i].y - gap*sin(90.0 * CV_PI / 180.0 + ang));
      pointInboardTest(src, points_right[i]);
      pointInboardTest(src, points_left[i]);
    }

    int iR = 0, iL = 0;
    for(int i = 0; i < num_points; i++) { 
      iR += src.at<unsigned char>(points_right[i].y, points_right[i].x);
      iL += src.at<unsigned char>(points_left[i].y, points_left[i].x);
    }

    if(iR > iL)
    {
      std::swap(seg.x1, seg.x2);
      std::swap(seg.y1, seg.y2);
      ang = ang + CV_PI;
      if(ang >= 2.0*CV_PI)
        ang = ang - 2.0 * CV_PI;
      seg.angle = (float)ang;
    }

    delete[] points;
    delete[] points_right;
    delete[] points_left;

    return;
  }

  void FastLineDetectorImpl::drawSegment(Mat& mat, const SEGMENT& seg, Scalar bgr, int thickness, bool directed)
  {
    Point2i p1;

    double gap = 10.0;
    double ang = (double)seg.angle;
    double arrow_angle = 30.0;

    p1.x = round(seg.x2 - gap*cos(arrow_angle * CV_PI / 180.0 + ang));
    p1.y = round(seg.y2 - gap*sin(arrow_angle * CV_PI / 180.0 + ang));
    pointInboardTest(mat, p1);

    line(mat, Point(round(seg.x1), round(seg.y1)), Point(round(seg.x2),
          round(seg.y2)), bgr, thickness, 1);
    if(directed)
      line(mat, Point(round(seg.x2), round(seg.y2)), p1, bgr, thickness, 1);
  }
} // namespace cv
