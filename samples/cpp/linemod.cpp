#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

// Function prototypes
void subtractPlane(const Mat& depth, Mat& mask, vector<Point>& chain, double f);

vector<Point> maskFromTemplate(const vector<linemod::Template>& templates,
                                      int num_modalities, Point offset, Size size,
                                      Mat& mask, Mat& dst);

void templateConvexHull(const vector<linemod::Template>& templates,
                        int num_modalities, Point offset, Size size,
                        Mat& dst);

void drawResponse(const vector<linemod::Template>& templates,
                  int num_modalities, Mat& dst, Point offset, int T);

Mat displayQuantized(const Mat& quantized);

// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
  static void start(const string& a_img_name)
  {
    setMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
  }
  static int event(void)
  {
    int l_event = m_event;
    m_event = -1;
    return l_event;
  }
  static int x(void)
  {
    return m_x;
  }
  static int y(void)
  {
    return m_y;
  }

private:
  static void cv_on_mouse(int a_event, int a_x, int a_y, int, void *)
  {
    m_event = a_event;
    m_x = a_x;
    m_y = a_y;
  }

  static int m_event;
  static int m_x;
  static int m_y;
};
int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;

static void help()
{
  printf("Usage: openni_demo [templates.yml]\n\n"
         "Place your object on a planar, featureless surface. With the mouse,\n"
         "frame it in the 'color' window and right click to learn a first template.\n"
         "Then press 'l' to enter online learning mode, and move the camera around.\n"
         "When the match score falls between 90-95%% the demo will add a new template.\n\n"
         "Keys:\n"
         "\t h   -- This help page\n"
         "\t l   -- Toggle online learning\n"
         "\t m   -- Toggle printing match result\n"
         "\t t   -- Toggle printing timings\n"
         "\t w   -- Write learned templates to disk\n"
         "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
         "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
  Timer() : start_(0), time_(0) {}

  void start()
  {
    start_ = getTickCount();
  }

  void stop()
  {
    CV_Assert(start_ != 0);
    int64 end = getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time()
  {
    double ret = time_ / getTickFrequency();
    time_ = 0;
    return ret;
  }

private:
  int64 start_, time_;
};

// Functions to store detector and templates in single XML/YAML file
static Ptr<linemod::Detector> readLinemod(const string& filename)
{
  Ptr<linemod::Detector> detector = new linemod::Detector;
  FileStorage fs(filename, FileStorage::READ);
  detector->read(fs.root());

  FileNode fn = fs["classes"];
  for (FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
    detector->readClass(*i);

  return detector;
}

static void writeLinemod(const Ptr<linemod::Detector>& detector, const string& filename)
{
  FileStorage fs(filename, FileStorage::WRITE);
  detector->write(fs);

  vector<string> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}


int main(int argc, char * argv[])
{
  // Various settings and flags
  bool show_match_result = true;
  bool show_timings = false;
  bool learn_online = false;
  int num_classes = 0;
  int matching_threshold = 80;
  /// @todo Keys for changing these?
  Size roi_size(200, 200);
  int learning_lower_bound = 90;
  int learning_upper_bound = 95;

  // Timers
  Timer extract_timer;
  Timer match_timer;

  // Initialize HighGUI
  help();
  namedWindow("color");
  namedWindow("normals");
  Mouse::start("color");

  // Initialize LINEMOD data structures
  Ptr<linemod::Detector> detector;
  string filename;
  if (argc == 1)
  {
    filename = "linemod_templates.yml";
    detector = linemod::getDefaultLINEMOD();
  }
  else
  {
    detector = readLinemod(argv[1]);

    vector<string> ids = detector->classIds();
    num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
           argv[1], num_classes, detector->numTemplates());
    if (!ids.empty())
    {
      printf("Class ids:\n");
      copy(ids.begin(), ids.end(), ostream_iterator<string>(cout, "\n"));
    }
  }
  int num_modalities = (int)detector->getModalities().size();

  // Open Kinect sensor
  VideoCapture capture( CV_CAP_OPENNI );
  if (!capture.isOpened())
  {
    printf("Could not open OpenNI-capable sensor\n");
    return -1;
  }
  capture.set(CV_CAP_PROP_OPENNI_REGISTRATION, 1);
  double focal_length = capture.get(CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH);
  //printf("Focal length = %f\n", focal_length);

  // Main loop
  Mat color, depth;
  for(;;)
  {
    // Capture next color/depth pair
    capture.grab();
    capture.retrieve(depth, CV_CAP_OPENNI_DEPTH_MAP);
    capture.retrieve(color, CV_CAP_OPENNI_BGR_IMAGE);

    vector<Mat> sources;
    sources.push_back(color);
    sources.push_back(depth);
    Mat display = color.clone();

    if (!learn_online)
    {
      Point mouse(Mouse::x(), Mouse::y());
      int event = Mouse::event();

      // Compute ROI centered on current mouse location
      Point roi_offset(roi_size.width / 2, roi_size.height / 2);
      Point pt1 = mouse - roi_offset; // top left
      Point pt2 = mouse + roi_offset; // bottom right

      if (event == CV_EVENT_RBUTTONDOWN)
      {
        // Compute object mask by subtracting the plane within the ROI
        vector<Point> chain;
        chain.push_back(pt1);
        chain.push_back(Point(pt2.x, pt1.y));
        chain.push_back(pt2);
        chain.push_back(Point(pt1.x, pt2.y));
        Mat mask;
        subtractPlane(depth, mask, chain, focal_length);

        imshow("mask", mask);

        // Extract template
        string class_id = format("class%d", num_classes);
        Rect bb;
        extract_timer.start();
        int template_id = detector->addTemplate(sources, class_id, mask, &bb);
        extract_timer.stop();
        if (template_id != -1)
        {
          printf("*** Added template (id %d) for new object class %d***\n",
                 template_id, num_classes);
          //printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height);
        }

        ++num_classes;
      }

      // Draw ROI for display
      rectangle(display, pt1, pt2, CV_RGB(0,0,0), 3);
      rectangle(display, pt1, pt2, CV_RGB(255,255,0), 1);
    }

    // Perform matching
    vector<linemod::Match> matches;
    vector<string> class_ids;
    vector<Mat> quantized_images;
    match_timer.start();
    detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);
    match_timer.stop();

    int classes_visited = 0;
    set<string> visited;

    for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
    {
      linemod::Match m = matches[i];

      if (visited.insert(m.class_id).second)
      {
        ++classes_visited;

        if (show_match_result)
        {
          printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                 m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
        }

        // Draw matching template
        const vector<linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
        drawResponse(templates, num_modalities, display, Point(m.x, m.y), detector->getT(0));

        if (learn_online == true)
        {
          /// @todo Online learning possibly broken by new gradient feature extraction,
          /// which assumes an accurate object outline.

          // Compute masks based on convex hull of matched template
          Mat color_mask, depth_mask;
          vector<Point> chain = maskFromTemplate(templates, num_modalities,
                                                        Point(m.x, m.y), color.size(),
                                                        color_mask, display);
          subtractPlane(depth, depth_mask, chain, focal_length);

          imshow("mask", depth_mask);

          // If pretty sure (but not TOO sure), add new template
          if (learning_lower_bound < m.similarity && m.similarity < learning_upper_bound)
          {
            extract_timer.start();
            int template_id = detector->addTemplate(sources, m.class_id, depth_mask);
            extract_timer.stop();
            if (template_id != -1)
            {
              printf("*** Added template (id %d) for existing object class %s***\n",
                     template_id, m.class_id.c_str());
            }
          }
        }
      }
    }

    if (show_match_result && matches.empty())
      printf("No matches found...\n");
    if (show_timings)
    {
      printf("Training: %.2fs\n", extract_timer.time());
      printf("Matching: %.2fs\n", match_timer.time());
    }
    if (show_match_result || show_timings)
      printf("------------------------------------------------------------\n");

    imshow("color", display);
    imshow("normals", quantized_images[1]);

    FileStorage fs;
    char key = (char)waitKey(10);
    if( key == 'q' )
        break;

    switch (key)
    {
      case 'h':
        help();
        break;
      case 'm':
        // toggle printing match result
        show_match_result = !show_match_result;
        printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
        break;
      case 't':
        // toggle printing timings
        show_timings = !show_timings;
        printf("Show timings %s\n", show_timings ? "ON" : "OFF");
        break;
      case 'l':
        // toggle online learning
        learn_online = !learn_online;
        printf("Online learning %s\n", learn_online ? "ON" : "OFF");
        break;
      case '[':
        // decrement threshold
        matching_threshold = std::max(matching_threshold - 1, -100);
        printf("New threshold: %d\n", matching_threshold);
        break;
      case ']':
        // increment threshold
        matching_threshold = std::min(matching_threshold + 1, +100);
        printf("New threshold: %d\n", matching_threshold);
        break;
      case 'w':
        // write model to disk
        writeLinemod(detector, filename);
        printf("Wrote detector and templates to %s\n", filename.c_str());
        break;
      default:
        ;
    }
  }
  return 0;
}

static void reprojectPoints(const vector<Point3d>& proj, vector<Point3d>& real, double f)
{
  real.resize(proj.size());
  double f_inv = 1.0 / f;

  for (int i = 0; i < (int)proj.size(); ++i)
  {
    double Z = proj[i].z;
    real[i].x = (proj[i].x - 320.) * (f_inv * Z);
    real[i].y = (proj[i].y - 240.) * (f_inv * Z);
    real[i].z = Z;
  }
}

static void filterPlane(Mat & ap_depth, vector<Mat> & a_masks, vector<Point> & a_chain, double f)
{
  const int l_num_cost_pts = 200;

  float l_thres = 4;

  Mat lp_mask = Mat::zeros(ap_depth.rows, ap_depth.cols, CV_8UC1);

  vector<Point> l_chain_vector;

  float l_chain_length = 0;
  float * lp_seg_length = new float[a_chain.size()];

  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    float x_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x);
    float y_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y);
    lp_seg_length[l_i] = sqrt(x_diff*x_diff + y_diff*y_diff);
    l_chain_length += lp_seg_length[l_i];
  }
  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    if (lp_seg_length[l_i] > 0)
    {
      int l_cur_num = cvRound(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length);
      float l_cur_len = lp_seg_length[l_i] / l_cur_num;

      for (int l_j = 0; l_j < l_cur_num; ++l_j)
      {
        float l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]);

        Point l_pts;

        l_pts.x = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x) + a_chain[l_i].x);
        l_pts.y = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y) + a_chain[l_i].y);

        l_chain_vector.push_back(l_pts);
      }
    }
  }
  vector<Point3d> lp_src_3Dpts(l_chain_vector.size());

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    lp_src_3Dpts[l_i].x = l_chain_vector[l_i].x;
    lp_src_3Dpts[l_i].y = l_chain_vector[l_i].y;
    lp_src_3Dpts[l_i].z = ap_depth.at<uchar>(cvRound(lp_src_3Dpts[l_i].y), cvRound(lp_src_3Dpts[l_i].x));
    //lp_mask.at<uchar>((int)lp_src_3Dpts[l_i].Y,(int)lp_src_3Dpts[l_i].X) = 255;
  }
  //imshow("hallo2", lp_mask);

  reprojectPoints(lp_src_3Dpts, lp_src_3Dpts, f);

  Mat lp_pts = Mat((int)l_chain_vector.size(), 4, CV_32F);
  Mat lp_v = Mat(4, 4, CV_32F);
  Mat lp_w = Mat(4, 1, CV_32F);

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    lp_pts.at<float>(l_i, 0) = (float)lp_src_3Dpts[l_i].x;
    lp_pts.at<float>(l_i, 1) = (float)lp_src_3Dpts[l_i].y;
    lp_pts.at<float>(l_i, 2) = (float)lp_src_3Dpts[l_i].z;
    lp_pts.at<float>(l_i, 3) = 1.0f;
  }
  SVD::compute(lp_pts, lp_w, Mat(), lp_v);

  float l_n[4] = {lp_v.at<float>(0, 3),
                  lp_v.at<float>(1, 3),
                  lp_v.at<float>(2, 3),
                  lp_v.at<float>(3, 3)};

  float l_norm = sqrt(l_n[0] * l_n[0] + l_n[1] * l_n[1] + l_n[2] * l_n[2]);

  l_n[0] /= l_norm;
  l_n[1] /= l_norm;
  l_n[2] /= l_norm;
  l_n[3] /= l_norm;

  float l_max_dist = 0;

  for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
  {
    float l_dist =  l_n[0] * lp_pts.at<float>(l_i, 0) +
                    l_n[1] * lp_pts.at<float>(l_i, 1) +
                    l_n[2] * lp_pts.at<float>(l_i, 2) +
                    l_n[3] * lp_pts.at<float>(l_i, 3);

    if (fabs(l_dist) > l_max_dist)
      l_max_dist = l_dist;
  }
  //cerr << "plane: " << l_n[0] << ";" << l_n[1] << ";" << l_n[2] << ";" << l_n[3] << " maxdist: " << l_max_dist << " end" << endl;
  int l_minx = ap_depth.cols;
  int l_miny = ap_depth.rows;
  int l_maxx = 0;
  int l_maxy = 0;

  for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
  {
    l_minx = min(l_minx, a_chain[l_i].x);
    l_miny = min(l_miny, a_chain[l_i].y);
    l_maxx = max(l_maxx, a_chain[l_i].x);
    l_maxy = max(l_maxy, a_chain[l_i].y);
  }
  int l_w = l_maxx - l_minx + 1;
  int l_h = l_maxy - l_miny + 1;
  int l_nn = (int)a_chain.size();

  vector<Point> lp_chain;

  for (int l_i = 0; l_i < l_nn; ++l_i)
    lp_chain.push_back(a_chain[l_i]);

  // Create the proper structure for the fillPoly function
  vector< vector<Point> > filled;
  filled.push_back(lp_chain);
  fillPoly(lp_mask, filled, Scalar(255, 255, 255));

  //imshow("hallo1", lp_mask);

  vector<Point3d> lp_dst_3Dpts(l_h * l_w);

  int l_ind = 0;

  for (int l_r = 0; l_r < l_h; ++l_r)
  {
    for (int l_c = 0; l_c < l_w; ++l_c)
    {
      lp_dst_3Dpts[l_ind].x = l_c + l_minx;
      lp_dst_3Dpts[l_ind].y = l_r + l_miny;
      lp_dst_3Dpts[l_ind].z = ap_depth.at<uchar>(l_r + l_miny, l_c + l_minx);
      ++l_ind;
    }
  }
  reprojectPoints(lp_dst_3Dpts, lp_dst_3Dpts, f);

  l_ind = 0;

  for (int l_r = 0; l_r < l_h; ++l_r)
  {
    for (int l_c = 0; l_c < l_w; ++l_c)
    {
      float l_dist = (float)(l_n[0] * lp_dst_3Dpts[l_ind].x + l_n[1] * lp_dst_3Dpts[l_ind].y + lp_dst_3Dpts[l_ind].z * l_n[2] + l_n[3]);

      ++l_ind;

      if (lp_mask.at<uchar>(l_r + l_miny, l_c + l_minx) != 0)
      {
        if (fabs(l_dist) < max(l_thres, (l_max_dist * 2.0f)))
        {
          for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
          {
            int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
            int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

            a_masks[l_p].at<uchar>(l_row, l_col) = 0;
          }
        }
        else
        {
          for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
          {
            int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
            int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

            a_masks[l_p].at<uchar>(l_row, l_col) = 255;
          }
        }
      }
    }
  }
}

void subtractPlane(const Mat& depth, Mat& mask, vector<Point>& chain, double f)
{
  mask = Mat::zeros(depth.rows, depth.cols, CV_8U);
  vector<Mat> tmp;
  tmp.push_back(mask);
  Mat temp_depth = depth.clone();
  filterPlane(temp_depth, tmp, chain, f);
}

vector<Point> maskFromTemplate(const vector<linemod::Template>& templates,
                                      int num_modalities, Point offset, Size size,
                                      Mat& mask, Mat& dst)
{
  templateConvexHull(templates, num_modalities, offset, size, mask);

  const int OFFSET = 30;
  dilate(mask, mask, Mat(), Point(-1,-1), OFFSET);

  Mat mask_copy = mask.clone();
  vector<Mat> contours;
  findContours(mask_copy, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

  vector<Point> l_pts1 = contours[0]; // to use as input to cv_primesensor::filter_plane

  // Create the lines from the contour set, on the dst matrix
  // Add first element to the end to make the contour complete
  l_pts1.push_back(l_pts1[0]);
  for(size_t i = 0; i < l_pts1.size() - 1; i++)
  {
    Point pt0 = l_pts1[i];
    Point pt1 = l_pts1[i+1];
    line(dst, pt0, pt1, Scalar(0,255,0), 2);
  }

  return l_pts1;
}

// Adapted from cv_show_angles
Mat displayQuantized(const Mat& quantized)
{
  Mat color(quantized.size(), CV_8UC3);
  for (int r = 0; r < quantized.rows; ++r)
  {
    const uchar* quant_r = quantized.ptr(r);
    Vec3b* color_r = color.ptr<Vec3b>(r);

    for (int c = 0; c < quantized.cols; ++c)
    {
      Vec3b& bgr = color_r[c];
      switch (quant_r[c])
      {
        case 0:   bgr[0]=  0; bgr[1]=  0; bgr[2]=  0;    break;
        case 1:   bgr[0]= 55; bgr[1]= 55; bgr[2]= 55;    break;
        case 2:   bgr[0]= 80; bgr[1]= 80; bgr[2]= 80;    break;
        case 4:   bgr[0]=105; bgr[1]=105; bgr[2]=105;    break;
        case 8:   bgr[0]=130; bgr[1]=130; bgr[2]=130;    break;
        case 16:  bgr[0]=155; bgr[1]=155; bgr[2]=155;    break;
        case 32:  bgr[0]=180; bgr[1]=180; bgr[2]=180;    break;
        case 64:  bgr[0]=205; bgr[1]=205; bgr[2]=205;    break;
        case 128: bgr[0]=230; bgr[1]=230; bgr[2]=230;    break;
        case 255: bgr[0]=  0; bgr[1]=  0; bgr[2]=255;    break;
        default:  bgr[0]=  0; bgr[1]=255; bgr[2]=  0;    break;
      }
    }
  }

  return color;
}

// Adapted from cv_line_template::convex_hull
void templateConvexHull(const vector<linemod::Template>& templates,
                        int num_modalities, Point offset, Size size,
                        Mat& dst)
{
  vector<Point> points;
  for (int m = 0; m < num_modalities; ++m)
  {
    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      linemod::Feature f = templates[m].features[i];
      points.push_back(Point(f.x, f.y) + offset);
    }
  }

  vector<Point> hull;
  convexHull(points, hull);

  dst = Mat::zeros(size, CV_8U);
  const int hull_count = (int)hull.size();
  const Point* hull_pts = &hull[0];
  fillPoly(dst, &hull_pts, &hull_count, 1, Scalar(255));
}

void drawResponse(const vector<linemod::Template>& templates,
                  int num_modalities, Mat& dst, Point offset, int T)
{
  static const Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      linemod::Feature f = templates[m].features[i];
      Point pt(f.x + offset.x, f.y + offset.y);
      circle(dst, pt, T / 2, color);
    }
  }
}
