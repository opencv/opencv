#ifndef OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP
#define OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP

#include <iomanip>

struct PerfReport {
    std::string name;
    double  avg_latency        = 0.0;
    int64_t min_latency        = 0;
    int64_t max_latency        = 0;
    int64_t first_latency      = 0;
    double  throughput         = 0.0;
    int64_t elapsed            = 0;
    int64_t warmup_time        = 0;
    int64_t num_late_frames    = 0;
    std::vector<int64_t> latencies;

    std::string toStr(bool expanded = false) const;
};

std::string PerfReport::toStr(bool expand) const {
    std::stringstream ss;
    ss << name << ": \n"
       << "  Warm up time:   " << warmup_time << " ms\n"
       << "  Execution time: " << elapsed << " ms\n"
       << "  Frames:         " << num_late_frames << "/" << latencies.size() << " (late/all)\n"
       << "  Latency:\n"
       << "    first: " << first_latency << " ms\n"
       << "    min:   " << min_latency   << " ms\n"
       << "    max:   " << max_latency   << " ms\n"
       << "    avg:   " << std::fixed << std::setprecision(3) << avg_latency << " ms\n"
       << "  Throughput: " << std::fixed << std::setprecision(3) << throughput << " FPS";
    if (expand) {
        for (size_t i = 0; i < latencies.size(); ++i) {
            ss << "\nFrame:" << i << "\nLatency: "
               << latencies[i] << " ms";
        }
    }

    return ss.str();
}

class StopCriterion {
public:
    using Ptr = std::unique_ptr<StopCriterion>;

    virtual void start() = 0;
    virtual void iter()  = 0;
    virtual bool done()  = 0;
    virtual ~StopCriterion() = default;
};

class Pipeline {
public:
    using Ptr = std::shared_ptr<Pipeline>;

    Pipeline(std::string&&                  name,
             cv::GComputation&&             comp,
             std::shared_ptr<DummySource>&& src,
             StopCriterion::Ptr             stop_criterion,
             cv::GCompileArgs&&             args,
             const size_t                   num_outputs);

    void compile();
    void run();

    const PerfReport& report() const;
    const std::string& name() const { return m_name;}

    virtual ~Pipeline() = default;

protected:
    virtual void    _compile() = 0;
    virtual int64_t run_iter() = 0;
    virtual void    init() {};
    virtual void    deinit() {};

    std::string                  m_name;
    cv::GComputation             m_comp;
    std::shared_ptr<DummySource> m_src;
    StopCriterion::Ptr           m_stop_criterion;
    cv::GCompileArgs             m_args;
    size_t                       m_num_outputs;
    PerfReport                   m_perf;
};

Pipeline::Pipeline(std::string&&                  name,
                   cv::GComputation&&             comp,
                   std::shared_ptr<DummySource>&& src,
                   StopCriterion::Ptr             stop_criterion,
                   cv::GCompileArgs&&             args,
                   const size_t                   num_outputs)
    : m_name(std::move(name)),
      m_comp(std::move(comp)),
      m_src(std::move(src)),
      m_stop_criterion(std::move(stop_criterion)),
      m_args(std::move(args)),
      m_num_outputs(num_outputs) {
    m_perf.name = m_name;
}

void Pipeline::compile() {
    m_perf.warmup_time =
        utils::measure<std::chrono::milliseconds>([this]() {
        _compile();
    });
}

void Pipeline::run() {
    using namespace std::chrono;

    init();
    auto start = high_resolution_clock::now();
    m_stop_criterion->start();
    while (true) {
        m_perf.latencies.push_back(run_iter());
        m_perf.elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        m_stop_criterion->iter();

        if (m_stop_criterion->done()) {
            deinit();
            break;
        }
    }

    m_perf.avg_latency   = utils::avg(m_perf.latencies);
    m_perf.min_latency   = utils::min(m_perf.latencies);
    m_perf.max_latency   = utils::max(m_perf.latencies);
    m_perf.first_latency = m_perf.latencies[0];

    // NB: Count how many executions don't fit into camera latency interval.
    m_perf.num_late_frames =
        std::count_if(m_perf.latencies.begin(), m_perf.latencies.end(),
                [this](int64_t latency) {
                    return static_cast<double>(latency) > m_src->latency();
                });

    m_perf.throughput =
        (m_perf.latencies.size() / static_cast<double>(m_perf.elapsed)) * 1000;
}

const PerfReport& Pipeline::report() const {
    return m_perf;
}

class StreamingPipeline : public Pipeline {
public:
    using Pipeline::Pipeline;

private:
    void _compile() override {
        m_compiled =
            m_comp.compileStreaming({m_src->descr_of()},
                                     cv::GCompileArgs(m_args));
    }

    virtual void init() override {
        using namespace std::chrono;
        // NB: N-1 buffers + timestamp.
        m_out_mats.resize(m_num_outputs - 1);
        for (auto& m : m_out_mats) {
            m_pipeline_outputs += cv::gout(m);
        }
        m_pipeline_outputs += cv::gout(m_start_ts);
        m_compiled.setSource(m_src);
        m_compiled.start();
    }

    virtual void deinit() override {
        m_compiled.stop();
    }

    virtual int64_t run_iter() override {
        m_compiled.pull(cv::GRunArgsP{m_pipeline_outputs});
        return utils::timestamp<std::chrono::milliseconds>() - m_start_ts;
    }

    cv::GStreamingCompiled m_compiled;
    cv::GRunArgsP        m_pipeline_outputs;
    std::vector<cv::Mat> m_out_mats;
    int64_t              m_start_ts;
};

class RegularPipeline : public Pipeline {
public:
    using Pipeline::Pipeline;

private:
    void _compile() override {
        m_compiled =
            m_comp.compile({m_src->descr_of()},
                            cv::GCompileArgs(m_args));
    }

    virtual void init() override {
        m_out_mats.resize(m_num_outputs);
        for (auto& m : m_out_mats) {
            m_pipeline_outputs += cv::gout(m);
        }
    }

    virtual int64_t run_iter() override {
        using namespace std::chrono;
        cv::gapi::wip::Data d;
        m_src->pull(d);
        auto in_mat = cv::util::get<cv::Mat>(d);
        return utils::measure<milliseconds>([&]{
            m_compiled(cv::gin(in_mat), cv::GRunArgsP{m_pipeline_outputs});
        });
    }

    cv::GCompiled        m_compiled;
    cv::GRunArgsP        m_pipeline_outputs;
    std::vector<cv::Mat> m_out_mats;
};

enum class PLMode {
    REGULAR,
    STREAMING
};

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP
