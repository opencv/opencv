// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

enum VideoBackendMode
{
    MODE_CAMERA,
    MODE_STREAM,
    MODE_WRITER,
};

static
void dumpBackendInfo(VideoCaptureAPIs backend, enum VideoBackendMode mode)
{
    std::string name;
    try
    {
        name = videoio_registry::getBackendName(backend);
    }
    catch (const std::exception& e)
    {
        ADD_FAILURE() << "Can't query name of backend=" << backend << ": " << e.what();
    }
    catch (...)
    {
        ADD_FAILURE() << "Can't query name of backend=" << backend << ": unknown C++ exception";
    }
    bool isBuiltIn = true;
    try
    {
        isBuiltIn = videoio_registry::isBackendBuiltIn(backend);
    }
    catch (const std::exception& e)
    {
        ADD_FAILURE() << "Failed isBackendBuiltIn(backend=" << backend << "): " << e.what();
        cout << name << " - UNKNOWN TYPE" << endl;
        return;
    }
    if (isBuiltIn)
    {
        cout << name << " - BUILTIN" << endl;
        return;
    }

    std::string description = "NO_DESCRIPTION";
    int version_ABI = 0;
    int version_API = 0;
    try
    {
        if (mode == MODE_CAMERA)
            description = videoio_registry::getCameraBackendPluginVersion(backend, version_ABI, version_API);
        else if (mode == MODE_STREAM)
            description = videoio_registry::getStreamBackendPluginVersion(backend, version_ABI, version_API);
        else if (mode == MODE_WRITER)
            description = videoio_registry::getWriterBackendPluginVersion(backend, version_ABI, version_API);
        else
            CV_Error(Error::StsInternal, "");
        cout << name << " - PLUGIN (" << description << ") ABI=" << version_ABI << " API=" << version_API << endl;
        return;
    }
    catch (const cv::Exception& e)
    {
        if (e.code == Error::StsNotImplemented)
        {
            cout << name << " - PLUGIN - NOT LOADED" << endl;
            return;
        }
        ADD_FAILURE() << "Failed getBackendPluginDescription(backend=" << backend << "): " << e.what();
    }
    catch (const std::exception& e)
    {
        ADD_FAILURE() << "Failed getBackendPluginDescription(backend=" << backend << "): " << e.what();
    }
    cout << name << " - PLUGIN (ERROR on quering information)" << endl;
}

TEST(VideoIO_Plugins, query)
{
    const std::vector<cv::VideoCaptureAPIs> camera_backends = cv::videoio_registry::getCameraBackends();
    cout << "== Camera APIs (" << camera_backends.size() << "):" << endl;
    for (auto backend : camera_backends)
    {
        dumpBackendInfo(backend, MODE_CAMERA);
    }

    const std::vector<cv::VideoCaptureAPIs> stream_backends = cv::videoio_registry::getStreamBackends();
    cout << "== Stream capture APIs (" << stream_backends.size() << "):" << endl;
    for (auto backend : stream_backends)
    {
        dumpBackendInfo(backend, MODE_STREAM);
    }

    const std::vector<cv::VideoCaptureAPIs> writer_backends = cv::videoio_registry::getWriterBackends();
    cout << "== Writer APIs (" << writer_backends.size() << "):" << endl;
    for (auto backend : writer_backends)
    {
        dumpBackendInfo(backend, MODE_WRITER);
    }
}

}}
