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

class Pipeline {
public:
    using Ptr = std::shared_ptr<Pipeline>;

    Pipeline(std::string&&                  name,
             cv::GComputation&&             comp,
             std::shared_ptr<DummySource>&& src,
             cv::GCompileArgs&&             args,
             const size_t                   num_outputs);

    void compile();
    void run(double work_time_ms);
    const PerfReport& report() const;
    const std::string& name() const { return m_name;}

    virtual ~Pipeline() = default;

protected:
    struct RunPerf {
        int64_t              elapsed   = 0;
        std::vector<int64_t> latencies;
    };

    virtual void _compile() = 0;
    virtual RunPerf _run(double work_time_ms) = 0;

    std::string                  m_name;
    cv::GComputation             m_comp;
    std::shared_ptr<DummySource> m_src;
    cv::GCompileArgs             m_args;
    size_t                       m_num_outputs;
    PerfReport                   m_perf;
};

Pipeline::Pipeline(std::string&&                  name,
                   cv::GComputation&&             comp,
                   std::shared_ptr<DummySource>&& src,
                   cv::GCompileArgs&&             args,
                   const size_t                   num_outputs)
    : m_name(std::move(name)),
      m_comp(std::move(comp)),
      m_src(std::move(src)),
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

void Pipeline::run(double work_time_ms) {
    auto run_perf = _run(work_time_ms);

    m_perf.elapsed       = run_perf.elapsed;
    m_perf.latencies     = std::move(run_perf.latencies);
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

    Pipeline::RunPerf _run(double work_time_ms) override {
        // NB: Setup.
        using namespace std::chrono;
        // NB: N-1 buffers + timestamp.
        std::vector<cv::Mat> out_mats(m_num_outputs - 1);
        int64_t start_ts = -1;
        cv::GRunArgsP pipeline_outputs;
        for (auto& m : out_mats) {
            pipeline_outputs += cv::gout(m);
        }
        pipeline_outputs += cv::gout(start_ts);
        m_compiled.setSource(m_src);

        // NB: Start execution & measure performance statistics.
        Pipeline::RunPerf perf;
        auto start = high_resolution_clock::now();
        m_compiled.start();
        while (m_compiled.pull(cv::GRunArgsP{pipeline_outputs})) {
            int64_t latency = utils::timestamp<milliseconds>() - start_ts;

            perf.latencies.push_back(latency);
            perf.elapsed = duration_cast<milliseconds>(
                    high_resolution_clock::now() - start).count();

            if (perf.elapsed >= work_time_ms) {
                m_compiled.stop();
                break;
            }
        };
        return perf;
    }

    cv::GStreamingCompiled m_compiled;
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

    Pipeline::RunPerf _run(double work_time_ms) override {
        // NB: Setup
        using namespace std::chrono;
        cv::gapi::wip::Data d;
        std::vector<cv::Mat> out_mats(m_num_outputs);
        cv::GRunArgsP pipeline_outputs;
        for (auto& m : out_mats) {
            pipeline_outputs += cv::gout(m);
        }

        // NB: Start execution & measure performance statistics.
        Pipeline::RunPerf perf;
        auto start = high_resolution_clock::now();
        while (m_src->pull(d)) {
            auto in_mat = cv::util::get<cv::Mat>(d);
            int64_t latency = utils::measure<milliseconds>([&]{
                m_compiled(cv::gin(in_mat), cv::GRunArgsP{pipeline_outputs});
            });

            perf.latencies.push_back(latency);
            perf.elapsed = duration_cast<milliseconds>(
                    high_resolution_clock::now() - start).count();

            if (perf.elapsed >= work_time_ms) {
                break;
            }
        };
        return perf;
    }

    cv::GCompiled m_compiled;
};

enum class PLMode {
    REGULAR,
    STREAMING
};

#endif // OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP
