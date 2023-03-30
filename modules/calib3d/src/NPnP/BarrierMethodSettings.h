//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H
#define PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H

#include <memory>

namespace NPnP
{
    class BarrierMethodSettings
    {
    public:
        double epsilon;
        int binary_search_depth;
        bool verbose;
        int max_inner_iterations;
        double miu;
        double valid_result_threshold;

        BarrierMethodSettings(double a_epsilon, int a_binary_search_depth, bool a_verbose,
                              int a_max_inner_iterations, double a_miu,
                              double a_valid_result_threshold)
            : epsilon(a_epsilon), binary_search_depth(a_binary_search_depth),
              verbose(a_verbose), max_inner_iterations(a_max_inner_iterations), miu(a_miu),
              valid_result_threshold(a_valid_result_threshold) {}

        static std::shared_ptr<BarrierMethodSettings>
        init(double epsilon = 4E-8, int binary_search_depth = 20, bool verbose = true,
             int max_inner_iterations = 20, double miu = 50,
             double valid_result_threshold = 0.001)
        {
            return std::make_shared<BarrierMethodSettings>(
                epsilon, binary_search_depth, verbose, max_inner_iterations, miu,
                valid_result_threshold);
        }
    };
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_BARRIERMETHODSETTINGS_H
