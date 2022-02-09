#ifndef OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP
#define OPENCV_GAPI_PIPELINE_MODELING_TOOL_PIPELINE_HPP

struct PerfReport {
    std::string               name;
    double  avg_latency       = 0.0;
    double  throughput        = 0.0;
    int64_t first_run_latency = 0;
    int64_t elapsed           = 0;
    int64_t compilation_time  = 0;
    std::vector<int64_t> latencies;

    std::string toStr(bool expanded = false) const;
};

std::string PerfReport::toStr(bool expand) const {
    std::stringstream ss;
    ss << name << ": Compilation time: " << compilation_time << " ms; "
       << "Average latency: " << avg_latency << " ms; Throughput: "
       << throughput << " FPS; First latency: "
       << first_run_latency << " ms";

    if (expand) {
        ss << "\nTotal processed frames: " << latencies.size()
           << "\nTotal elapsed time: "     << elapsed << " ms" << std::endl;
        for (size_t i = 0; i < latencies.size(); ++i) {
            ss << std::endl;
            ss << "Frame:" << i << "\nLatency: "
               << latencies[i] << " ms";
        }
    }

    return ss.str();
}

class Pipeline {
public:
    using Ptr = std::shared_ptr<Pipeline>;

    Pipeline(std::string&&                       name,
             cv::GComputation&&                  comp,
             cv::gapi::wip::IStreamSource::Ptr&& src,
             cv::GCompileArgs&&                  args,
             const size_t                        num_outputs);

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

    std::string                       m_name;
    cv::GComputation                  m_comp;
    cv::gapi::wip::IStreamSource::Ptr m_src;
    cv::GCompileArgs                  m_args;
    size_t                            m_num_outputs;
    PerfReport                        m_perf;
};

Pipeline::Pipeline(std::string&&                       name,
                   cv::GComputation&&                  comp,
                   cv::gapi::wip::IStreamSource::Ptr&& src,
                   cv::GCompileArgs&&                  args,
                   const size_t                        num_outputs)
    : m_name(std::move(name)),
      m_comp(std::move(comp)),
      m_src(std::move(src)),
      m_args(std::move(args)),
      m_num_outputs(num_outputs) {
    m_perf.name = m_name;
}

void Pipeline::compile() {
    m_perf.compilation_time =
        utils::measure<std::chrono::milliseconds>([this]() {
        _compile();
    });
}

void Pipeline::run(double work_time_ms) {
    auto run_perf = _run(work_time_ms);

    m_perf.elapsed   = run_perf.elapsed;
    m_perf.latencies = std::move(run_perf.latencies);

    m_perf.avg_latency =
        std::accumulate(m_perf.latencies.begin(),
                        m_perf.latencies.end(),
                        0.0) / static_cast<double>(m_perf.latencies.size());
    m_perf.throughput =
        (m_perf.latencies.size() / static_cast<double>(m_perf.elapsed)) * 1000;

    m_perf.first_run_latency = m_perf.latencies[0];
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
