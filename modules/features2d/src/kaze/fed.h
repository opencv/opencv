#ifndef __OPENCV_FEATURES_2D_FED_H__
#define __OPENCV_FEATURES_2D_FED_H__

//******************************************************************************
//******************************************************************************

// Includes
#include <vector>

//*************************************************************************************
//*************************************************************************************

// Declaration of functions
int fed_tau_by_process_time(const float& T, const int& M, const float& tau_max,
                            const bool& reordering, std::vector<float>& tau);
int fed_tau_by_cycle_time(const float& t, const float& tau_max,
                          const bool& reordering, std::vector<float> &tau) ;
int fed_tau_internal(const int& n, const float& scale, const float& tau_max,
                     const bool& reordering, std::vector<float> &tau);
bool fed_is_prime_internal(const int& number);

//*************************************************************************************
//*************************************************************************************

#endif // __OPENCV_FEATURES_2D_FED_H__
