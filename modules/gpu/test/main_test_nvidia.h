#ifndef __main_test_nvidia_h__
#define __main_test_nvidia_h__

#include<string>

enum OutputLevel
{
    OutputLevelNone,
    OutputLevelCompact,
    OutputLevelFull
};

bool nvidia_NPPST_Integral_Image(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Squared_Integral_Image(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_RectStdDev(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Resize(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Vector_Operations(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Transpose(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Vector_Operations(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Haar_Cascade_Loader(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Haar_Cascade_Application(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Hypotheses_Filtration(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Visualization(const std::string& test_data_path, OutputLevel outputLevel);

#endif
