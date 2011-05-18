#include "autocalib.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

void focalsFromHomography(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
    CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));

    const double* h = reinterpret_cast<const double*>(H.data);

    double d1, d2; // Denominators
    double v1, v2; // Focal squares value candidates

    f1_ok = true;
    d1 = h[6] * h[7];
    d2 = (h[7] - h[6]) * (h[7] + h[6]);
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
    if (v1 < v2) swap(v1, v2);
    if (v1 > 0 && v2 > 0) f1 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
    else if (v1 > 0) f1 = sqrt(v1);
    else f1_ok = false;

    f0_ok = true;
    d1 = h[0] * h[3] + h[1] * h[4];
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
    v1 = -h[2] * h[5] / d1;
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
    if (v1 < v2) swap(v1, v2);
    if (v1 > 0 && v2 > 0) f0 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
    else if (v1 > 0) f0 = sqrt(v1);
    else f0_ok = false;
}


double estimateFocal(const vector<Mat> &images, const vector<ImageFeatures> &/*features*/, 
                     const vector<MatchesInfo> &pairwise_matches)
{
    const int num_images = static_cast<int>(images.size());

    vector<double> focals;
    for (int src_idx = 0; src_idx < num_images; ++src_idx)
    {
        for (int dst_idx = 0; dst_idx < num_images; ++dst_idx)
        {
            const MatchesInfo &m = pairwise_matches[src_idx*num_images + dst_idx];
            if (m.H.empty())
                continue;

            double f0, f1;
            bool f0ok, f1ok;
            focalsFromHomography(m.H, f0, f1, f0ok, f1ok);
            if (f0ok && f1ok) focals.push_back(sqrt(f0*f1));
        }
    }

    if (focals.size() + 1 >= images.size())
    {
        nth_element(focals.begin(), focals.end(), focals.begin() + focals.size()/2);
        return focals[focals.size()/2];
    }

    LOGLN("Can't estimate focal length, will use naive approach");
    double focals_sum = 0;
    for (int i = 0; i < num_images; ++i)
        focals_sum += images[i].rows + images[i].cols;
    return focals_sum / num_images;
}
