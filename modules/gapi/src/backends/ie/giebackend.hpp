// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation

#ifndef OPENCV_GAPI_GIEBACKEND_HPP
#define OPENCV_GAPI_GIEBACKEND_HPP

// Include anyway - cv::gapi::ie::backend() still needs to be defined
#include "opencv2/gapi/infer/ie.hpp"

#ifdef HAVE_INF_ENGINE

#include <ade/util/algorithm.hpp> // type_list_index
#include <condition_variable>

#include <inference_engine.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv {
namespace gimpl {
namespace ie {

struct IECompiled {
#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
    InferenceEngine::InferencePlugin   this_plugin;
#else
    InferenceEngine::Core              this_core;
#endif
    InferenceEngine::ExecutableNetwork this_network;
    InferenceEngine::InferRequest      this_request;
};

// FIXME: Structure which collect all necessary sync primitives
// will be deleted when the async request pool appears
class SyncPrim {
public:
    void wait() {
        std::unique_lock<std::mutex> l(m_mutex);
        m_cv.wait(l, [this]{ return !m_is_busy; });
    }

    void release_and_notify() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_is_busy = false;
        }
        m_cv.notify_one();
    }

    void acquire() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_is_busy = true;
    }

private:
    // To wait until the async request isn't over
    std::condition_variable m_cv;
    // To avoid spurious cond var wake up
    bool m_is_busy = false;
    // To sleep until condition variable wakes up
    std::mutex m_mutex;
};

class GIEExecutable final: public GIslandExecutable
{
    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    // The only executable stuff in this graph
    // (assuming it is always single-op)
    ade::NodeHandle this_nh;
    IECompiled this_iec;

    // List of all resources in graph (both internal and external)
    std::vector<ade::NodeHandle> m_dataNodes;

    // Sync primitive
    SyncPrim m_sync;

public:
    GIEExecutable(const ade::Graph                   &graph,
                  const std::vector<ade::NodeHandle> &nodes);

    virtual inline bool canReshape() const override { return false; }
    virtual inline void reshape(ade::Graph&, const GCompileArgs&) override {
        GAPI_Assert(false); // Not implemented yet
    }

    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override {
        GAPI_Assert(false && "Not implemented");
    }

    virtual void run(GIslandExecutable::IInput  &in,
                     GIslandExecutable::IOutput &out) override;

};

}}}

#endif // HAVE_INF_ENGINE
#endif // OPENCV_GAPI_GIEBACKEND_HPP
