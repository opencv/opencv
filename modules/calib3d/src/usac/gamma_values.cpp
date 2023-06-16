// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {

class GammaValuesImpl : public GammaValues {
    std::vector<double> gamma_complete, gamma_incomplete, gamma;
    double scale_complete_values, scale_gamma_values;
    int max_size_table, DoF;
public:
    GammaValuesImpl (int DoF_, int max_size_table_) {
        max_size_table = max_size_table_;
        max_size_table = max_size_table_;
        DoF = DoF_;
        /*
         * Gamma values for degrees of freedom n = 2 and sigma quantile 99% of chi distribution
         * (squared root of chi-squared distribution), in the range <0; 4.62> for complete values
         * and <0, 1.52> for gamma values.
         * Number of anchor points is 50. Other values are approximated using linear interpolation
         */
        const int number_of_anchor_points = 50;
        std::vector<double> gamma_complete_anchor, gamma_incomplete_anchor, gamma_anchor;
        if (DoF == 2) {
            const double max_thr = 7.5, gamma_quantile = 3.04;
            scale_complete_values = max_size_table_ / max_thr;
            scale_gamma_values = gamma_quantile * max_size_table_ / max_thr ;

            gamma_complete_anchor = std::vector<double>
                    {1.77245385e+00, 1.02824699e+00, 7.69267629e-01, 5.99047749e-01,
                     4.75998050e-01, 3.83008633e-01, 3.10886473e-01, 2.53983661e-01,
                     2.08540472e-01, 1.71918718e-01, 1.42197872e-01, 1.17941854e-01,
                     9.80549104e-02, 8.16877552e-02, 6.81739145e-02, 5.69851046e-02,
                     4.76991202e-02, 3.99417329e-02, 3.35126632e-02, 2.81470710e-02,
                     2.36624697e-02, 1.99092598e-02, 1.67644090e-02, 1.41264487e-02,
                     1.19114860e-02, 1.00500046e-02, 8.48428689e-03, 7.16632498e-03,
                     6.05612291e-03, 5.12031042e-03, 4.33100803e-03, 3.66489504e-03,
                     3.10244213e-03, 2.62514027e-03, 2.22385863e-03, 1.88454040e-03,
                     1.59749690e-03, 1.35457835e-03, 1.14892453e-03, 9.74756909e-04,
                     8.27205063e-04, 7.02161552e-04, 5.96160506e-04, 5.06275903e-04,
                     4.30036278e-04, 3.65353149e-04, 3.10460901e-04, 2.63866261e-04,
                     2.24305797e-04, 1.90558599e-04};

            gamma_incomplete_anchor = std::vector<double>
                    {0.        , 0.0364325 , 0.09423626, 0.15858163, 0.22401622, 0.28773243,
                     0.34820493, 0.40463362, 0.45665762, 0.50419009, 0.54731575, 0.58622491,
                     0.62116968, 0.65243473, 0.68031763, 0.70511575, 0.72711773, 0.74668782,
                     0.76389332, 0.77907386, 0.79244816, 0.80421561, 0.81455692, 0.8236351 ,
                     0.83159653, 0.83857228, 0.84467927, 0.85002158, 0.85469163, 0.85877132,
                     0.86233307, 0.86544086, 0.86815108, 0.87052421, 0.87258093, 0.87437198,
                     0.87593103, 0.87728759, 0.87846752, 0.87949345, 0.88038518, 0.88116002,
                     0.88183307, 0.88241755, 0.88292497, 0.88336537, 0.88374751, 0.88407901,
                     0.88436652, 0.88461696};

            gamma_anchor = std::vector<double>
                    {1.77245385e+00, 5.93375722e-01, 3.05833272e-01, 1.68019955e-01,
                     9.52188705e-02, 5.49876141e-02, 3.21629603e-02, 1.89881161e-02,
                     1.12897384e-02, 6.75016002e-03, 4.05426969e-03, 2.44422283e-03,
                     1.47822780e-03, 8.96425642e-04, 5.44879754e-04, 3.31873268e-04,
                     2.02499478e-04, 1.23458651e-04, 7.55593392e-05, 4.63032752e-05,
                     2.84078946e-05, 1.74471428e-05, 1.07257506e-05, 6.59955061e-06,
                     4.06400013e-06, 2.50448635e-06, 1.54449028e-06, 9.53085308e-07,
                     5.88490160e-07, 3.63571768e-07, 2.24734099e-07, 1.38982938e-07,
                     8.59913580e-08, 5.31026827e-08, 3.28834964e-08, 2.03707922e-08,
                     1.26240063e-08, 7.82595771e-09, 4.85312084e-09, 3.01051575e-09,
                     1.86805770e-09, 1.15947962e-09, 7.19869372e-10, 4.47050615e-10,
                     2.77694421e-10, 1.72536278e-10, 1.07224039e-10, 6.66497131e-11,
                     4.14376355e-11, 2.57079508e-11};
        } else if (DoF == 4) {
            const double max_thr = 2.5, gamma_quantile = 3.64;
            scale_complete_values = max_size_table_ / max_thr;
            scale_gamma_values = gamma_quantile * max_size_table_ / max_thr ;
            gamma_complete_anchor = std::vector<double>
                {0.88622693, 0.87877828, 0.86578847, 0.84979442, 0.83179176, 0.81238452,
                0.79199067, 0.77091934, 0.74940836, 0.72764529, 0.70578051, 0.68393585,
                0.66221071, 0.64068639, 0.61942952, 0.59849449, 0.57792561, 0.55766078,
                0.53792634, 0.51864482, 0.49983336, 0.48150466, 0.46366759, 0.44632776,
                0.42948797, 0.41314862, 0.39730804, 0.38196282, 0.36710806, 0.35273761,
                0.33884422, 0.32541979, 0.31245545, 0.29988151, 0.28781065, 0.2761701 ,
                0.26494924, 0.25413723, 0.24372308, 0.23369573, 0.22404405, 0.21475696,
                0.2058234 , 0.19723241, 0.18897314, 0.18103488, 0.17340708, 0.16607937,
                0.15904157, 0.15225125};
            gamma_incomplete_anchor = std::vector<double>
                {0.00000000e+00, 2.26619558e-04, 1.23631005e-03, 3.28596265e-03,
                6.50682297e-03, 1.09662062e-02, 1.66907233e-02, 2.36788942e-02,
                3.19091043e-02, 4.13450655e-02, 5.19397673e-02, 6.36384378e-02,
                7.63808171e-02, 9.01029320e-02, 1.04738496e-01, 1.20220023e-01,
                1.36479717e-01, 1.53535010e-01, 1.71152805e-01, 1.89349599e-01,
                2.08062142e-01, 2.27229225e-01, 2.46791879e-01, 2.66693534e-01,
                2.86880123e-01, 3.07300152e-01, 3.27904735e-01, 3.48647611e-01,
                3.69485130e-01, 3.90376227e-01, 4.11282379e-01, 4.32167556e-01,
                4.52998149e-01, 4.73844336e-01, 4.94473655e-01, 5.14961263e-01,
                5.35282509e-01, 5.55414767e-01, 5.75337352e-01, 5.95031429e-01,
                6.14479929e-01, 6.33667460e-01, 6.52580220e-01, 6.71205917e-01,
                6.89533681e-01, 7.07553988e-01, 7.25258581e-01, 7.42640393e-01,
                7.59693477e-01, 7.76494059e-01};
            gamma_anchor = std::vector<double>
                {8.86226925e-01, 8.38460922e-01, 7.64931722e-01, 6.85680218e-01,
                6.07663201e-01, 5.34128389e-01, 4.66574835e-01, 4.05560768e-01,
                3.51114357e-01, 3.02965249e-01, 2.60682396e-01, 2.23758335e-01,
                1.91661077e-01, 1.63865725e-01, 1.39873108e-01, 1.19220033e-01,
                1.01484113e-01, 8.62162923e-02, 7.32253576e-02, 6.21314285e-02,
                5.26713657e-02, 4.46151697e-02, 3.77626859e-02, 3.19403783e-02,
                2.69982683e-02, 2.28070945e-02, 1.92557199e-02, 1.62487939e-02,
                1.37046640e-02, 1.15535264e-02, 9.73579631e-03, 8.20068208e-03,
                6.90494092e-03, 5.80688564e-03, 4.88587254e-03, 4.10958296e-03,
                3.45555079e-03, 2.90474053e-03, 2.44103551e-03, 2.05079975e-03,
                1.72250366e-03, 1.44640449e-03, 1.21427410e-03, 1.01916714e-03,
                8.55224023e-04, 7.17503448e-04, 6.01840372e-04, 5.04725511e-04,
                4.23203257e-04, 3.54478559e-04};
        } else CV_Error(cv::Error::StsNotImplemented, "Not implemented for specific DoF!");
        // allocate tables
        gamma_complete = std::vector<double>(max_size_table);
        gamma_incomplete = std::vector<double>(max_size_table);
        gamma = std::vector<double>(max_size_table);

        // do linear interpolation of gamma values
        const int step = (int)((double)max_size_table / (number_of_anchor_points-1));
        int arr_cnt = 0;
        for (int i = 0; i < number_of_anchor_points-1; i++) {
             const double complete_x0 = gamma_complete_anchor[i], step_complete = (gamma_complete_anchor[i+1] - complete_x0) / step;
             const double incomplete_x0 = gamma_incomplete_anchor[i], step_incomplete = (gamma_incomplete_anchor[i+1] - incomplete_x0) / step;
             const double gamma_x0 = gamma_anchor[i], step_gamma = (gamma_anchor[i+1] - gamma_x0) / step;

             for (int j = 0; j < step; j++) {
                 gamma_complete[arr_cnt] = complete_x0 + j * step_complete;
                 gamma_incomplete[arr_cnt] = incomplete_x0 + j * step_incomplete;
                 gamma[arr_cnt++] = gamma_x0 + j * step_gamma;
             }
        }
        if (arr_cnt < max_size_table) {
            // if array was not totally filled (in some cases can happen) then copy last values
            std::fill(gamma_complete.begin()+arr_cnt, gamma_complete.end(), gamma_complete[arr_cnt-1]);
            std::fill(gamma_incomplete.begin()+arr_cnt, gamma_incomplete.end(), gamma_incomplete[arr_cnt-1]);
            std::fill(gamma.begin()+arr_cnt, gamma.end(), gamma[arr_cnt-1]);
        }
    }

    const std::vector<double> &getCompleteGammaValues() const override { return gamma_complete; }
    const std::vector<double> &getIncompleteGammaValues() const override { return gamma_incomplete; }
    const std::vector<double> &getGammaValues() const override { return gamma; }
    double getScaleOfGammaCompleteValues () const override { return scale_complete_values; }
    double getScaleOfGammaValues () const override { return scale_gamma_values; }
    int getTableSize () const override { return max_size_table; }
};
Ptr<GammaValues> GammaValues::create(int DoF, int max_size_table) {
    return makePtr<GammaValuesImpl>(DoF, max_size_table);
}
}}  // namespace
