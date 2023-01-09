#include <iostream>
#include <fstream>
#include <thread>
#include <exception>
#include <unordered_map>
#include <vector>

#include <opencv2/gapi.hpp>
#include <opencv2/highgui.hpp> // cv::CommandLineParser
#include <opencv2/core/utils/filesystem.hpp>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif

#include "pipeline_modeling_tool/dummy_source.hpp"
#include "pipeline_modeling_tool/utils.hpp"
#include "pipeline_modeling_tool/pipeline_builder.hpp"

enum class AppMode {
    REALTIME,
    BENCHMARK
};

static AppMode strToAppMode(const std::string& mode_str) {
    if (mode_str == "realtime") {
        return AppMode::REALTIME;
    } else if (mode_str == "benchmark") {
        return AppMode::BENCHMARK;
    } else {
        throw std::logic_error("Unsupported AppMode: " + mode_str +
                "\nPlease chose between: realtime and benchmark");
    }
}

template <typename T>
T read(const cv::FileNode& node) {
    return static_cast<T>(node);
}

static cv::FileNode check_and_get_fn(const cv::FileNode& fn,
                                     const std::string&  field,
                                     const std::string&  uplvl) {
    const bool is_map = fn.isMap();
    if (!is_map || fn[field].empty()) {
        throw std::logic_error(uplvl + " must contain field: " + field);
    }
    return fn[field];
}

static cv::FileNode check_and_get_fn(const cv::FileStorage& fs,
                                     const std::string&     field,
                                     const std::string&     uplvl) {
    auto fn = fs[field];
    if (fn.empty()) {
        throw std::logic_error(uplvl + " must contain field: " + field);
    }
    return fn;
}

template <typename T, typename FileT>
T check_and_read(const FileT& f,
                 const std::string& field,
                 const std::string& uplvl) {
    auto fn = check_and_get_fn(f, field, uplvl);
    return read<T>(fn);
}

template <typename T>
cv::optional<T> readOpt(const cv::FileNode& fn) {
    return fn.empty() ? cv::optional<T>() : cv::optional<T>(read<T>(fn));
}

template <typename T>
std::vector<T> readList(const cv::FileNode& fn,
                        const std::string& field,
                        const std::string& uplvl) {
    auto fn_field = check_and_get_fn(fn, field, uplvl);
    if (!fn_field.isSeq()) {
        throw std::logic_error(field + " in " + uplvl + " must be a sequence");
    }

    std::vector<T> vec;
    for (auto iter : fn_field) {
        vec.push_back(read<T>(iter));
    }
    return vec;
}

template <typename T>
std::vector<T> readVec(const cv::FileNode& fn,
                       const std::string& field,
                       const std::string& uplvl) {
    auto fn_field = check_and_get_fn(fn, field, uplvl);

    std::vector<T> vec;
    fn_field >> vec;
    return vec;
}

static int strToPrecision(const std::string& precision) {
    static std::unordered_map<std::string, int> str_to_precision = {
        {"U8", CV_8U}, {"FP32", CV_32F}, {"FP16", CV_16F}
    };
    auto it = str_to_precision.find(precision);
    if (it == str_to_precision.end()) {
        throw std::logic_error("Unsupported precision: " + precision);
    }
    return it->second;
}

template <>
OutputDescr read<OutputDescr>(const cv::FileNode& fn) {
    auto dims      = readVec<int>(fn, "dims", "output");
    auto str_prec = check_and_read<std::string>(fn, "precision", "output");
    return OutputDescr{dims, strToPrecision(str_prec)};
}

template <>
Edge read<Edge>(const cv::FileNode& fn) {
    auto from = check_and_read<std::string>(fn, "from", "edge");
    auto to   = check_and_read<std::string>(fn, "to", "edge");

    auto splitNameAndPort = [](const std::string& str) {
        auto pos = str.find(':');
        auto name =
            pos == std::string::npos ? str : std::string(str.c_str(), pos);
        size_t port =
            pos == std::string::npos ? 0 : std::atoi(str.c_str() + pos + 1);
        return std::make_pair(name, port);
    };

    auto p1 = splitNameAndPort(from);
    auto p2 = splitNameAndPort(to);
    return Edge{Edge::P{p1.first, p1.second}, Edge::P{p2.first, p2.second}};
}

static std::string getModelsPath() {
    static char* models_path_c = std::getenv("PIPELINE_MODELS_PATH");
    static std::string models_path = models_path_c ? models_path_c : ".";
    return models_path;
}

template <>
ModelPath read<ModelPath>(const cv::FileNode& fn) {
    using cv::utils::fs::join;
    if (!fn["xml"].empty() && !fn["bin"].empty()) {
        return ModelPath{LoadPath{join(getModelsPath(), fn["xml"].string()),
                                  join(getModelsPath(), fn["bin"].string())}};
    } else if (!fn["blob"].empty()){
        return ModelPath{ImportPath{join(getModelsPath(), fn["blob"].string())}};
    } else {
        const std::string emsg = R""""(
        Path to OpenVINO model must be specified in either of two formats:
1.
  xml: path to *.xml
  bin: path to *.bin
2.
  blob: path to *.blob
        )"""";
        throw std::logic_error(emsg);
    }
}

static PLMode strToPLMode(const std::string& mode_str) {
    if (mode_str == "streaming") {
        return PLMode::STREAMING;
    } else if (mode_str == "regular") {
        return PLMode::REGULAR;
    } else {
        throw std::logic_error("Unsupported PLMode: " + mode_str +
                "\nPlease chose between: streaming and regular");
    }
}

static cv::gapi::ie::InferMode strToInferMode(const std::string& infer_mode) {
    if (infer_mode == "async") {
        return cv::gapi::ie::InferMode::Async;
    } else if (infer_mode == "sync") {
        return cv::gapi::ie::InferMode::Sync;
    } else {
        throw std::logic_error("Unsupported Infer mode: " + infer_mode +
                "\nPlease chose between: async and sync");
    }
}

template <>
CallParams read<CallParams>(const cv::FileNode& fn) {
    auto name =
        check_and_read<std::string>(fn, "name", "node");
    // FIXME: Impossible to read size_t due OpenCV limitations.
    auto call_every_nth_opt = readOpt<int>(fn["call_every_nth"]);
    auto call_every_nth = call_every_nth_opt.value_or(1);
    if (call_every_nth <= 0) {
        throw std::logic_error(
                name + " call_every_nth must be greater than zero\n"
                "Current call_every_nth: " + std::to_string(call_every_nth));
    }
    return CallParams{std::move(name), static_cast<size_t>(call_every_nth)};
}

template <typename V>
std::map<std::string, V> readMap(const cv::FileNode& fn) {
    std::map<std::string, V> map;
    for (auto item : fn) {
        map.emplace(item.name(), read<V>(item));
    }
    return map;
}

template <>
InferParams read<InferParams>(const cv::FileNode& fn) {
    auto name =
        check_and_read<std::string>(fn, "name", "node");

    InferParams params;
    params.path          = read<ModelPath>(fn);
    params.device        = check_and_read<std::string>(fn, "device", name);
    params.input_layers  = readList<std::string>(fn, "input_layers", name);
    params.output_layers = readList<std::string>(fn, "output_layers", name);
    params.config        = readMap<std::string>(fn["config"]);

    auto out_prec_str = readOpt<std::string>(fn["output_precision"]);
    if (out_prec_str.has_value()) {
        params.out_precision =
            cv::optional<int>(strToPrecision(out_prec_str.value()));
    }
    return params;
}

template <>
DummyParams read<DummyParams>(const cv::FileNode& fn) {
    auto name =
        check_and_read<std::string>(fn, "name", "node");

    DummyParams params;
    params.time = check_and_read<double>(fn, "time", name);
    if (params.time < 0) {
        throw std::logic_error(name + " time must be positive");
    }
    params.output = check_and_read<OutputDescr>(fn, "output", name);

    return params;
}

static std::vector<std::string> parseExecList(const std::string& exec_list) {
    std::vector<std::string> pl_types;
    std::stringstream ss(exec_list);
    std::string pl_type;
    while (getline(ss, pl_type, ',')) {
        pl_types.push_back(pl_type);
    }
    return pl_types;
}

static void loadConfig(const std::string&                        filename,
                             std::map<std::string, std::string>& config) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to load config: " + filename);
    }

    cv::FileNode root = fs.root();
    for (auto it = root.begin(); it != root.end(); ++it) {
        auto device = *it;
        if (!device.isMap()) {
            throw std::runtime_error("Failed to parse config: " + filename);
        }
        for (auto item : device) {
            config.emplace(item.name(), item.string());
        }
    }
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    timeBeginPeriod(1);
#endif
    try {
        const std::string keys =
        "{ h help      |           | Print this help message. }"
        "{ cfg         |           | Path to the config which is either"
                                   " YAML file or string. }"
        "{ load_config |           | Optional. Path to XML/YAML/JSON file"
                                   " to load custom IE parameters. }"
        "{ cache_dir   |           | Optional. Enables caching of loaded models"
                                   " to specified directory. }"
        "{ log_file    |           | Optional. If file is specified, app will"
                                   " dump expanded execution information. }"
        "{ pl_mode     | streaming | Optional. Pipeline mode: streaming/regular"
                                   " if it's specified will be applied for"
                                   " every pipeline. }"
        "{ qc          | 1         | Optional. Calculated automatically by G-API"
                                   " if set to 0. If it's specified will be"
                                   " applied for every pipeline. }"
        "{ app_mode    | realtime  | Application mode (realtime/benchmark). }"
        "{ drop_frames | false     | Drop frames if they come earlier than pipeline is completed. }"
        "{ exec_list   |           | A comma-separated list of pipelines that"
                                   " will be executed. Spaces around commas"
                                   " are prohibited. }"
        "{ infer_mode  | async     | OpenVINO inference mode (async/sync). }";

        cv::CommandLineParser cmd(argc, argv, keys);
        if (cmd.has("help")) {
            cmd.printMessage();
            return 0;
        }

        const auto cfg         = cmd.get<std::string>("cfg");
        const auto load_config = cmd.get<std::string>("load_config");
        const auto cached_dir  = cmd.get<std::string>("cache_dir");
        const auto log_file    = cmd.get<std::string>("log_file");
        const auto cmd_pl_mode = strToPLMode(cmd.get<std::string>("pl_mode"));
        const auto qc          = cmd.get<int>("qc");
        const auto app_mode    = strToAppMode(cmd.get<std::string>("app_mode"));
        const auto exec_str    = cmd.get<std::string>("exec_list");
        const auto infer_mode  = strToInferMode(cmd.get<std::string>("infer_mode"));
        const auto drop_frames = cmd.get<bool>("drop_frames");

        cv::FileStorage fs;
        if (cfg.empty()) {
            throw std::logic_error("Config must be specified via --cfg option");
        }
        // NB: *.yml
        if (cfg.size() < 5) {
            throw std::logic_error("--cfg string must contain at least 5 symbols"
                                   " to determine if it's a file (*.yml) a or string");
        }
        if (cfg.substr(cfg.size() - 4, cfg.size()) == ".yml") {
            if (!fs.open(cfg, cv::FileStorage::READ)) {
                throw std::logic_error("Failed to open config file: " + cfg);
            }
        } else {
            fs = cv::FileStorage(cfg, cv::FileStorage::FORMAT_YAML |
                                      cv::FileStorage::MEMORY);
        }

        std::map<std::string, std::string> gconfig;
        if (!load_config.empty()) {
            loadConfig(load_config, gconfig);
        }
        // NB: Takes priority over config from file
        if (!cached_dir.empty()) {
            gconfig =
                std::map<std::string, std::string>{{"CACHE_DIR", cached_dir}};
        }

        auto opt_work_time_ms = readOpt<double>(fs["work_time"]);
        cv::optional<int64_t> opt_work_time_mcs;
        if (opt_work_time_ms) {
            const double work_time_ms = opt_work_time_ms.value();
            if (work_time_ms < 0) {
                throw std::logic_error("work_time must be positive");
            }
            opt_work_time_mcs = cv::optional<int64_t>(utils::ms_to_mcs(work_time_ms));
        }

        auto pipelines_fn = check_and_get_fn(fs, "Pipelines", "Config");
        if (!pipelines_fn.isMap()) {
            throw std::logic_error("Pipelines field must be a map");
        }

        auto exec_list = !exec_str.empty() ? parseExecList(exec_str)
                                           : pipelines_fn.keys();


        std::vector<Pipeline::Ptr> pipelines;
        pipelines.reserve(exec_list.size());
        // NB: Build pipelines based on config information
        PipelineBuilder builder;
        for (const auto& name : exec_list) {
            const auto& pl_fn = check_and_get_fn(pipelines_fn, name, "Pipelines");
            builder.setName(name);
            StopCriterion::Ptr stop_criterion;
            auto opt_num_iters = readOpt<int>(pl_fn["num_iters"]);
            // NB: num_iters for specific pipeline takes priority over global work_time.
            if (opt_num_iters) {
                stop_criterion.reset(new NumItersCriterion(opt_num_iters.value()));
            } else if (opt_work_time_mcs) {
                stop_criterion.reset(new ElapsedTimeCriterion(opt_work_time_mcs.value()));
            } else {
                throw std::logic_error(
                        "Failed: Pipeline " + name + " doesn't have stop criterion!\n"
                        "Please specify either work_time: <value> in the config root"
                        " or num_iters: <value> for specific pipeline.");
            }
            builder.setStopCriterion(std::move(stop_criterion));

            // NB: Set source
            {
                const auto& src_fn = check_and_get_fn(pl_fn, "source", name);
                auto src_name =
                    check_and_read<std::string>(src_fn, "name", "source");
                auto latency =
                    check_and_read<double>(src_fn, "latency", "source");
                auto output =
                    check_and_read<OutputDescr>(src_fn, "output", "source");
                // NB: In case BENCHMARK mode sources work with zero latency.
                if (app_mode == AppMode::BENCHMARK) {
                    latency = 0.0;
                }
                auto src = std::make_shared<DummySource>(latency, output, drop_frames);
                builder.setSource(src_name, src);
            }

            const auto& nodes_fn = check_and_get_fn(pl_fn, "nodes", name);
            if (!nodes_fn.isSeq()) {
                throw std::logic_error("nodes in " + name + " must be a sequence");
            }

            for (auto node_fn : nodes_fn) {
                auto call_params = read<CallParams>(node_fn);
                auto node_type =
                    check_and_read<std::string>(node_fn, "type", "node");
                if (node_type == "Dummy") {
                    builder.addDummy(call_params, read<DummyParams>(node_fn));
                } else if (node_type == "Infer") {
                    auto infer_params = read<InferParams>(node_fn);
                    try {
                        utils::mergeMapWith(infer_params.config, gconfig);
                    } catch (std::exception& e) {
                        std::stringstream ss;
                        ss << "Failed to merge global and local config for Infer node: "
                           << call_params.name << std::endl << e.what();
                        throw std::logic_error(ss.str());
                    }
                    infer_params.mode = infer_mode;
                    builder.addInfer(call_params, infer_params);
                } else {
                    throw std::logic_error("Unsupported node type: " + node_type);
                }
            }

            const auto edges_fn = check_and_get_fn(pl_fn, "edges", name);
            if (!edges_fn.isSeq()) {
                throw std::logic_error("edges in " + name + " must be a sequence");
            }
            for (auto edge_fn : edges_fn) {
                auto edge = read<Edge>(edge_fn);
                builder.addEdge(edge);
            }

            auto cfg_pl_mode = readOpt<std::string>(pl_fn["mode"]);
            // NB: Pipeline mode from config takes priority over cmd.
            auto pl_mode = cfg_pl_mode.has_value()
                ? strToPLMode(cfg_pl_mode.value()) : cmd_pl_mode;
            // NB: Using drop_frames with streaming pipelines will follow to
            // incorrect performance results.
            if (drop_frames && pl_mode == PLMode::STREAMING) {
                throw std::logic_error(
                        "--drop_frames option is supported only for pipelines in \"regular\" mode");
            }

            builder.setMode(pl_mode);

            // NB: Queue capacity from config takes priority over cmd.
            auto config_qc = readOpt<int>(pl_fn["queue_capacity"]);
            auto queue_capacity = config_qc.has_value() ? config_qc.value() : qc;
            // NB: 0 is special constant that means
            // queue capacity should be calculated automatically.
            if (queue_capacity != 0) {
                builder.setQueueCapacity(queue_capacity);
            }

            auto dump = readOpt<std::string>(pl_fn["dump"]);
            if (dump) {
                builder.setDumpFilePath(dump.value());
            }

            pipelines.emplace_back(builder.build());
        }

        // NB: Compille pipelines
        for (size_t i = 0; i < pipelines.size(); ++i) {
            pipelines[i]->compile();
        }

        // NB: Execute pipelines
        std::vector<std::exception_ptr> eptrs(pipelines.size(), nullptr);
        std::vector<std::thread> threads(pipelines.size());
        for (size_t i = 0; i < pipelines.size(); ++i) {
            threads[i] = std::thread([&, i]() {
                try {
                    pipelines[i]->run();
                } catch (...) {
                    eptrs[i] = std::current_exception();
                }
            });
        }

        std::ofstream file;
        if (!log_file.empty()) {
            file.open(log_file);
        }

        for (size_t i = 0; i < threads.size(); ++i) {
            threads[i].join();
        }

        for (size_t i = 0; i < threads.size(); ++i) {
            if (eptrs[i] != nullptr) {
                try {
                    std::rethrow_exception(eptrs[i]);
                } catch (std::exception& e) {
                    throw std::logic_error(pipelines[i]->name() + " failed: " + e.what());
                }
            }
            if (file.is_open()) {
                file << pipelines[i]->report().toStr(true) << std::endl;
            }
            std::cout << pipelines[i]->report().toStr() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        throw;
    }
    return 0;
}
