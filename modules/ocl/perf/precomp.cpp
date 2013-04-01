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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

// This program test most of the functions in ocl module and generate data metrix of x-factor in .csv files
// All images needed in this test are in samples/gpu folder.
// For haar template, haarcascade_frontalface_alt.xml shouold be in working directory
void TestSystem::run()
{
    if (is_list_mode_)
    {
        for (vector<Runnable *>::iterator it = tests_.begin(); it != tests_.end(); ++it)
        {
            cout << (*it)->name() << endl;
        }

        return;
    }

    // Run test initializers
    for (vector<Runnable *>::iterator it = inits_.begin(); it != inits_.end(); ++it)
    {
        if ((*it)->name().find(test_filter_, 0) != string::npos)
        {
            (*it)->run();
        }
    }

    printHeading();
    writeHeading();

    // Run tests
    for (vector<Runnable *>::iterator it = tests_.begin(); it != tests_.end(); ++it)
    {
        try
        {
            if ((*it)->name().find(test_filter_, 0) != string::npos)
            {
                cout << endl << (*it)->name() << ":\n";

                setCurrentTest((*it)->name());
                //fprintf(record_,"%s\n",(*it)->name().c_str());

                (*it)->run();
                finishCurrentSubtest();
            }
        }
        catch (const Exception &)
        {
            // Message is printed via callback
            resetCurrentSubtest();
        }
        catch (const runtime_error &e)
        {
            printError(e.what());
            resetCurrentSubtest();
        }
    }

    printSummary();
    writeSummary();
}


void TestSystem::finishCurrentSubtest()
{
    if (cur_subtest_is_empty_)
        // There is no need to print subtest statistics
    {
        return;
    }

    double cpu_time = cpu_elapsed_ / getTickFrequency() * 1000.0;
    double gpu_time = gpu_elapsed_ / getTickFrequency() * 1000.0;
    double gpu_full_time = gpu_full_elapsed_ / getTickFrequency() * 1000.0;

    double speedup = static_cast<double>(cpu_elapsed_) / std::max(1.0, gpu_elapsed_);
    speedup_total_ += speedup;

    double fullspeedup = static_cast<double>(cpu_elapsed_) / std::max(1.0, gpu_full_elapsed_);
    speedup_full_total_ += fullspeedup;

    if (speedup > top_)
    {
        speedup_faster_count_++;
    }
    else if (speedup < bottom_)
    {
        speedup_slower_count_++;
    }
    else
    {
        speedup_equal_count_++;
    }

    if (fullspeedup > top_)
    {
        speedup_full_faster_count_++;
    }
    else if (fullspeedup < bottom_)
    {
        speedup_full_slower_count_++;
    }
    else
    {
        speedup_full_equal_count_++;
    }

    // compute min, max and
    std::sort(gpu_times_.begin(), gpu_times_.end());
    double gpu_min = gpu_times_.front() / getTickFrequency() * 1000.0;
    double gpu_max = gpu_times_.back() / getTickFrequency() * 1000.0;
    double deviation = 0;

    if (gpu_times_.size() > 1)
    {
        double sum = 0;

        for (size_t i = 0; i < gpu_times_.size(); i++)
        {
            int64 diff = gpu_times_[i] - static_cast<int64>(gpu_elapsed_);
            double diff_time = diff * 1000 / getTickFrequency();
            sum += diff_time * diff_time;
        }

        deviation = std::sqrt(sum / gpu_times_.size());
    }

    printMetrics(cpu_time, gpu_time, gpu_full_time, speedup, fullspeedup);
    writeMetrics(cpu_time, gpu_time, gpu_full_time, speedup, fullspeedup, gpu_min, gpu_max, deviation);

    num_subtests_called_++;
    resetCurrentSubtest();
}


double TestSystem::meanTime(const vector<int64> &samples)
{
    double sum = accumulate(samples.begin(), samples.end(), 0.);
    return sum / samples.size();
}


void TestSystem::printHeading()
{
    cout << endl;
    cout << setiosflags(ios_base::left);
    cout << TAB << setw(10) << "CPU, ms" << setw(10) << "GPU, ms"
         << setw(14) << "SPEEDUP" << setw(14) << "GPUTOTAL, ms" << setw(14) << "TOTALSPEEDUP"
         << "DESCRIPTION\n";

    cout << resetiosflags(ios_base::left);
}

void TestSystem::writeHeading()
{
    if (!record_)
    {
        recordname_ += "_OCL.csv";
        record_ = fopen(recordname_.c_str(), "w");
    }

    fprintf(record_, "NAME,DESCRIPTION,CPU (ms),GPU (ms),SPEEDUP,GPUTOTAL (ms),TOTALSPEEDUP,GPU Min (ms),GPU Max (ms), Standard deviation (ms)\n");

    fflush(record_);
}

void TestSystem::printSummary()
{
    cout << setiosflags(ios_base::fixed);
    cout << "\naverage GPU speedup: x"
         << setprecision(3) << speedup_total_ / std::max(1, num_subtests_called_)
         << endl;
    cout << "\nGPU exceeded: "
         << setprecision(3) << speedup_faster_count_
         << "\nGPU passed: "
         << setprecision(3) << speedup_equal_count_
         << "\nGPU failed: "
         << setprecision(3) << speedup_slower_count_
         << endl;
    cout << "\nGPU exceeded rate: "
         << setprecision(3) << (float)speedup_faster_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPU passed rate: "
         << setprecision(3) << (float)speedup_equal_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPU failed rate: "
         << setprecision(3) << (float)speedup_slower_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << endl;
    cout << "\naverage GPUTOTAL speedup: x"
         << setprecision(3) << speedup_full_total_ / std::max(1, num_subtests_called_)
         << endl;
    cout << "\nGPUTOTAL exceeded: "
         << setprecision(3) << speedup_full_faster_count_
         << "\nGPUTOTAL passed: "
         << setprecision(3) << speedup_full_equal_count_
         << "\nGPUTOTAL failed: "
         << setprecision(3) << speedup_full_slower_count_
         << endl;
    cout << "\nGPUTOTAL exceeded rate: "
         << setprecision(3) << (float)speedup_full_faster_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPUTOTAL passed rate: "
         << setprecision(3) << (float)speedup_full_equal_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << "\nGPUTOTAL failed rate: "
         << setprecision(3) << (float)speedup_full_slower_count_ / std::max(1, num_subtests_called_) * 100
         << "%"
         << endl;
    cout << resetiosflags(ios_base::fixed);
}


void TestSystem::printMetrics(double cpu_time, double gpu_time, double gpu_full_time, double speedup, double fullspeedup)
{
    cout << TAB << setiosflags(ios_base::left);
    stringstream stream;

    stream << cpu_time;
    cout << setw(10) << stream.str();

    stream.str("");
    stream << gpu_time;
    cout << setw(10) << stream.str();

    stream.str("");
    stream << "x" << setprecision(3) << speedup;
    cout << setw(14) << stream.str();

    stream.str("");
    stream << gpu_full_time;
    cout << setw(14) << stream.str();

    stream.str("");
    stream << "x" << setprecision(3) << fullspeedup;
    cout << setw(14) << stream.str();

    cout << cur_subtest_description_.str();
    cout << resetiosflags(ios_base::left) << endl;
}

void TestSystem::writeMetrics(double cpu_time, double gpu_time, double gpu_full_time, double speedup, double fullspeedup, double gpu_min, double gpu_max, double std_dev)
{
    if (!record_)
    {
        recordname_ += ".csv";
        record_ = fopen(recordname_.c_str(), "w");
    }

    fprintf(record_, "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", itname_changed_ ? itname_.c_str() : "",
            cur_subtest_description_.str().c_str(),
            cpu_time, gpu_time, speedup, gpu_full_time, fullspeedup,
            gpu_min, gpu_max, std_dev);

    if (itname_changed_)
    {
        itname_changed_ = false;
    }

    fflush(record_);
}

void TestSystem::writeSummary()
{
    if (!record_)
    {
        recordname_ += ".csv";
        record_ = fopen(recordname_.c_str(), "w");
    }

    fprintf(record_, "\nAverage GPU speedup: %.3f\n"
            "exceeded: %d (%.3f%%)\n"
            "passed: %d (%.3f%%)\n"
            "failed: %d (%.3f%%)\n"
            "\nAverage GPUTOTAL speedup: %.3f\n"
            "exceeded: %d (%.3f%%)\n"
            "passed: %d (%.3f%%)\n"
            "failed: %d (%.3f%%)\n",
            speedup_total_ / std::max(1, num_subtests_called_),
            speedup_faster_count_, (float)speedup_faster_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_equal_count_, (float)speedup_equal_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_slower_count_, (float)speedup_slower_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_total_ / std::max(1, num_subtests_called_),
            speedup_full_faster_count_, (float)speedup_full_faster_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_equal_count_, (float)speedup_full_equal_count_ / std::max(1, num_subtests_called_) * 100,
            speedup_full_slower_count_, (float)speedup_full_slower_count_ / std::max(1, num_subtests_called_) * 100
           );
    fflush(record_);
}

void TestSystem::printError(const std::string &msg)
{
	if(msg != "CL_INVALID_BUFFER_SIZE")
	{
		cout << TAB << "[error: " << msg << "] " << cur_subtest_description_.str() << endl;
	}
}

void gen(Mat &mat, int rows, int cols, int type, Scalar low, Scalar high)
{
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


string abspath(const string &relpath)
{
    return TestSystem::instance().workingDir() + relpath;
}


int CV_CDECL cvErrorCallback(int /*status*/, const char * /*func_name*/,
                             const char *err_msg, const char * /*file_name*/,
                             int /*line*/, void * /*userdata*/)
{
    TestSystem::instance().printError(err_msg);
    return 0;
}


