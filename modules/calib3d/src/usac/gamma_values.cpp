// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {

GammaValues::GammaValues()
    : max_range_complete(4.62)
    , max_range_gamma(1.52)
    , max_size_table(3000)
{
    /*
     * Gamma values for degrees of freedom n = 2 and sigma quantile 99% of chi distribution
     * (squared root of chi-squared distribution), in the range <0; 4.62> for complete values
     * and <0, 1.52> for gamma values.
     * Number of anchor points is 50. Other values are approximated using linear interpolation
     */
    const int number_of_anchor_points = 50;
    std::vector<double> gamma_complete_anchor = std::vector<double>
       {1.7724538509055159, 1.182606138403832, 0.962685372890749, 0.8090013493715409,
        0.6909325812483967, 0.5961199186942078, 0.5179833984918483, 0.45248091153099873,
        0.39690029823142897, 0.34930995878395804, 0.3082742109224103, 0.2726914551904204,
        0.2416954924567404, 0.21459196516027726, 0.190815580770884, 0.16990026519723456,
        0.15145770273372564, 0.13516150988807635, 0.12073530906427948, 0.10794357255251595,
        0.0965844793065712, 0.08648426334883624, 0.07749268706639856, 0.06947937608738222,
        0.062330823249820304, 0.05594791865006951, 0.05024389794830681, 0.045142626552664405,
        0.040577155977706246, 0.03648850256745103, 0.03282460924226794, 0.029539458909083157,
        0.02659231432268328, 0.023947063970062663, 0.021571657306774475, 0.01943761564987864,
        0.017519607407598645, 0.015795078236273064, 0.014243928262247118, 0.012848229767187478,
        0.011591979769030827, 0.010460882783057988, 0.009442159753944173, 0.008524379737926344,
        0.007697311406424555, 0.006951791856026042, 0.006279610558635573, 0.005673406581042374,
        0.005126577454218803, 0.004633198286725555};

    std::vector<double> gamma_incomplete_anchor = std::vector<double>
        {0.0, 0.01773096912803939, 0.047486924846289004, 0.08265437835139826, 0.120639343491371,
         0.15993024714868515, 0.19954558593754865, 0.23881753504915218, 0.2772830648361923,
         0.3146208784488923, 0.3506114446939783, 0.385110056889967, 0.41802785670077697,
         0.44931803198258047, 0.47896553567848993, 0.5069792897777948, 0.5333861945970247,
         0.5582264802664578, 0.581550074874317, 0.6034137543595729, 0.6238789008764282,
         0.6430097394182639, 0.6608719532994989, 0.6775316015953519, 0.6930542783709592,
         0.7075044661695132, 0.7209450459078338, 0.733436932830201, 0.7450388140484766,
         0.7558069678435577, 0.7657951486073097, 0.7750545242776943, 0.7836336555215403,
         0.7915785078697124, 0.798932489600361, 0.8057365094688473, 0.8120290494534339,
         0.8178462485678104, 0.8232219945197348, 0.8281880205973585, 0.8327740056635289,
         0.8370076755516281, 0.8409149044990385, 0.8445198155381767, 0.8478448790000731,
         0.8509110084798414, 0.8537376537738418, 0.8563428904304485, 0.8587435056647642,
         0.8609550804762539};

    std::vector<double> gamma_anchor = std::vector<double>
        {1.7724538509055159, 1.427187162582056, 1.2890382454046982, 1.186244737282388,
         1.1021938955410173, 1.0303674512016956, 0.9673796229113404, 0.9111932804012203,
         0.8604640514722175, 0.814246149432561, 0.7718421763436497, 0.7327190195355812,
         0.6964573670982434, 0.6627197089339725, 0.6312291454822467, 0.6017548373556638,
         0.5741017071093776, 0.5481029597580317, 0.523614528104858, 0.5005108666212138,
         0.478681711577816, 0.4580295473431646, 0.43846759792922513, 0.41991821541471996,
         0.40231157253054745, 0.38558459136185, 0.3696800574963841, 0.3545458813847714,
         0.340134477710645, 0.32640224021796493, 0.3133090943985706, 0.3008181141790485,
         0.28889519159238314, 0.2775087506098113, 0.2666294980086962, 0.2562302054837794,
         0.24628551826026082, 0.2367717863030556, 0.22766691488600885, 0.21895023182476064,
         0.2106023691144937, 0.2026051570714723, 0.19494152937027823, 0.18759543761063277,
         0.1805517742482484, 0.17379630289125447, 0.16731559510356395, 0.1610969729740903,
         0.1551284568099053, 0.14939871739550692};

    // allocate tables
    gamma_complete = std::vector<double>(max_size_table);
    gamma_incomplete = std::vector<double>(max_size_table);
    gamma = std::vector<double>(max_size_table);

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

const std::vector<double>& GammaValues::getCompleteGammaValues() const { return gamma_complete; }
const std::vector<double>& GammaValues::getIncompleteGammaValues() const { return gamma_incomplete; }
const std::vector<double>& GammaValues::getGammaValues() const { return gamma; }
double GammaValues::getScaleOfGammaCompleteValues () const { return gamma_complete.size() / max_range_complete; }
double GammaValues::getScaleOfGammaValues () const { return gamma.size() / max_range_gamma; }
int GammaValues::getTableSize () const { return max_size_table; }

/* static */
const GammaValues& GammaValues::getSingleton()
{
    static GammaValues g_gammaValues;
    return g_gammaValues;
}

}}  // namespace
