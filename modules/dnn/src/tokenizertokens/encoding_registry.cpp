#include "encoding.hpp"
#include "encoding_registry.hpp"

#include <mutex>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional>


namespace cv {namespace dnn { namespace tokenizer {


CV_EXPORTS void registerEncoding(const std::string &name, EncodingCtor ctor) {
    std::lock_guard<std::mutex> lock(mtx);
    ctors[name] = std::move(ctor);
}

CV_EXPORTS const Encoding& getEncoding(const std::string &name) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = instances.find(name);
        if (it!=instances.end())
            return it->second;
    }

    EncodingCtor ctor;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = ctors.find(name);
        if (it == ctors.end()) 
            throw std::invalid_argument("Unknown encoding: " + name);
        ctor = it->second;
    }

    Encoding enc = ctor();
    std::lock_guard<std::mutex> lock(mtx);
    auto [inserted, ok] = instances.emplace(name, std::move(enc));
    return inserted->second;
}

CV_EXPORTS std::vector<std::string> listEncodingNames() {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::string> names;
    names.reserve(ctors.size());
    for (auto &p : ctors) 
        names.push_back(p.first);
    return names;
}

}}}