// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_GAPI_FLUID_BACKEND_HPP
#define OPENCV_GAPI_FLUID_BACKEND_HPP

// FIXME? Actually gfluidbackend.hpp is not included anywhere
// and can be placed in gfluidbackend.cpp

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>
#include <opencv2/gapi/fluid/gfluidbuffer.hpp>

// PRIVATE STUFF!
#include "backends/common/gbackend.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv { namespace gimpl {

struct FluidUnit
{
    static const char *name() { return "FluidUnit"; }
    GFluidKernel k;
    gapi::fluid::BorderOpt border;
    int border_size;
    int window;
    std::vector<int> line_consumption;
    double ratio;
};

struct FluidUseOwnBorderBuffer
{
    static const char *name() { return "FluidUseOwnBorderBuffer"; }
    bool use;
};

struct FluidData
{
    static const char *name() { return "FluidData"; }

    // FIXME: This structure starts looking like "FluidBuffer" meta
    int  latency         = 0;
    int  skew            = 0;
    int  max_consumption = 1;
    int  border_size     = 0;
    int  lpi_write       = 1;
    bool internal        = false; // is node internal to any fluid island
    gapi::fluid::BorderOpt border;
};

struct agent_data_t {
     GFluidKernel::Kind  kind;
     ade::NodeHandle     nh;
     std::vector<int>    in_buffer_ids;
     std::vector<int>    out_buffer_ids;
 };

struct FluidAgent
{
public:
    virtual ~FluidAgent() = default;
    FluidAgent(const ade::Graph &g, ade::NodeHandle nh);

    GFluidKernel k;
    ade::NodeHandle op_handle; // FIXME: why it is here??//
    std::string op_name;

    // <  0 - not a buffer
    // >= 0 - a buffer with RcID
    std::vector<int> in_buffer_ids;
    std::vector<int> out_buffer_ids;

    cv::GArgs in_args;
    std::vector<cv::gapi::fluid::View>   in_views; // sparce list of IN views
    std::vector<cv::gapi::fluid::Buffer*> out_buffers;

    // FIXME Current assumption is that outputs have EQUAL SIZES
    int m_outputLines = 0;
    int m_producedLines = 0;

    // Execution methods
    void reset();
    bool canWork() const;
    bool canRead() const;
    bool canWrite() const;
    void doWork();
    bool done() const;

    void debug(std::ostream& os);

    // FIXME:
    // refactor (implement a more solid replacement or
    // drop this method completely)
    virtual void setRatio(double ratio) = 0;

private:
    // FIXME!!!
    // move to another class
    virtual int firstWindow(std::size_t inPort) const = 0;
    virtual std::pair<int,int> linesReadAndnextWindow(std::size_t inPort) const = 0;
};

//helper data structure for accumulating graph traversal/analysis data
struct FluidGraphInputData {

    std::vector<agent_data_t>               m_agents_data;
    std::vector<std::size_t>                m_scratch_users;
    std::unordered_map<int, std::size_t>    m_id_map;           // GMat id -> buffer idx map
    std::map<std::size_t, ade::NodeHandle>  m_all_gmat_ids;

    std::size_t                             m_mat_count;
};
//local helper function to traverse the graph once and pass the results to multiple instances of GFluidExecutable
FluidGraphInputData fluidExtractInputDataFromGraph(const ade::Graph &m_g, const std::vector<ade::NodeHandle> &nodes);

class GFluidExecutable final: public GIslandExecutable
{
    GFluidExecutable(const GFluidExecutable&) = delete;  // due std::unique_ptr in members list

    const ade::Graph &m_g;
    GModel::ConstGraph m_gm;

    std::vector<std::unique_ptr<FluidAgent>> m_agents;
    std::vector<cv::gapi::fluid::Buffer> m_buffers;

    std::vector<FluidAgent*> m_script;

    using Magazine = detail::magazine<cv::gapi::own::Scalar, cv::detail::VectorRef, cv::detail::OpaqueRef>;
    Magazine m_res;

    std::size_t m_num_int_buffers; // internal buffers counter (m_buffers - num_scratch)
    std::vector<std::size_t> m_scratch_users;

    std::unordered_map<int, std::size_t> m_id_map; // GMat id -> buffer idx map
    std::map<std::size_t, ade::NodeHandle> m_all_gmat_ids;

    void bindInArg (const RcDesc &rc, const GRunArg &arg);
    void bindOutArg(const RcDesc &rc, const GRunArgP &arg);
    void packArg   (GArg &in_arg, const GArg &op_arg);

    void initBufferRois(std::vector<int>& readStarts, std::vector<cv::gapi::own::Rect>& rois, const std::vector<gapi::own::Rect> &out_rois);
    void makeReshape(const std::vector<cv::gapi::own::Rect>& out_rois);
    std::size_t total_buffers_size() const;

public:
    virtual inline bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph& g, const GCompileArgs& args) override;

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;

    void run(std::vector<InObj>  &input_objs,
             std::vector<OutObj> &output_objs);


     GFluidExecutable(const ade::Graph                          &g,
                      const FluidGraphInputData                 &graph_data,
                      const std::vector<cv::gapi::own::Rect>    &outputRois);
};


class GParallelFluidExecutable final: public GIslandExecutable {
    GParallelFluidExecutable(const GParallelFluidExecutable&) = delete;  // due std::unique_ptr in members list

    std::vector<std::unique_ptr<GFluidExecutable>> tiles;
    decltype(GFluidParallelFor::parallel_for) parallel_for;
public:
    GParallelFluidExecutable(const ade::Graph                       &g,
                             const FluidGraphInputData              &graph_data,
                             const std::vector<GFluidOutputRois>    &parallelOutputRois,
                             const decltype(parallel_for)           &pfor);


    virtual inline bool canReshape() const override { return false; }
    virtual void reshape(ade::Graph& g, const GCompileArgs& args) override;

    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) override;
};
}} // cv::gimpl


#endif // OPENCV_GAPI_FLUID_BACKEND_HPP
