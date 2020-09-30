#ifndef OPENCV_GAPI_COMMON_SERIALIZATION_HPP
#define OPENCV_GAPI_COMMON_SERIALIZATION_HPP

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <ade/util/iota_range.hpp> // used in the vector<</>>

#include "compiler/gmodel.hpp"
#include "opencv2/gapi/render/render_types.hpp"
#include "opencv2/gapi/s11n.hpp" // basic interfaces

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(disable: 4702)
#endif

namespace cv {
namespace gapi {
namespace s11n {

struct GSerialized {
    std::vector<cv::gimpl::Op> m_ops;
    std::vector<cv::gimpl::Data> m_datas;
    cv::gimpl::DataObjectCounter m_counter;
    cv::gimpl::Protocol m_proto;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IStream/OStream

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, cv::gimpl::Data::Storage  s);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is, cv::gimpl::Data::Storage &s);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::DataObjectCounter &c);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::DataObjectCounter &c);

GAPI_EXPORTS I::OStream& operator<< (I::OStream& os, const cv::gimpl::Protocol &p);
GAPI_EXPORTS I::IStream& operator>> (I::IStream& is,       cv::gimpl::Protocol &p);

//Forward declaration
//I::OStream& operator<< (I::OStream& os, const cv::GRunArg &arg);
//I::IStream& operator>> (I::IStream& is, cv::GRunArg &arg);

// The top-level serialization routine.
// Note it is just a single function which takes a GModel and a list of nodes
// and writes the data to the stream (recursively)
GAPI_EXPORTS void serialize( I::OStream& os
                           , const ade::Graph &g
                           , const std::vector<ade::NodeHandle> &nodes);

// The top-level serialization routine.
// Note it is just a single function which takes a GModel and a list of nodes
// and writes the data to the stream (recursively)
GAPI_EXPORTS void serialize( I::OStream& os
                           , const ade::Graph &g
                           , const cv::gimpl::Protocol &p
                           , const std::vector<ade::NodeHandle> &nodes);

// The top-level deserialization routineS.
// Unfortunately the deserialization is a two-step process:
// 1. First we decode a stream into some intermediate representation
//     (called "GSerialized");
// 2. Then we produce an ade::Graph from this intermediate representation.
//
// An ade::Graph can't be produced from the stream immediately
// since every GCompiled object has its own unique ade::Graph, so
// we can't do it once and for all since every compilation process
// is individual and _is_ altering the ade::Graph state (structure and metadata).
// At the same time, we can't hold the reference to "is" within the GComputation
// forever since this input stream may be associated with an external resource
// and have side effects.
//
// Summarizing, the `deserialize()` happens *once per GComputation* immediately
// during the cv::gapi::deserialize<GComputation>(), and `reconstruct()` happens
// on every compilation process issued for this GComputation.
GAPI_EXPORTS GSerialized deserialize(I::IStream& is);
GAPI_EXPORTS void reconstruct(const GSerialized &s, ade::Graph &g);

// FIXME: Basic Stream implementaions //////////////////////////////////////////

// Basic in-memory stream implementations.
class GAPI_EXPORTS ByteMemoryOutStream final: public I::OStream {
    std::vector<char> m_storage;

    //virtual I::OStream& operator << (uint32_t) override;
    //virtual I::OStream& operator<< (uint32_t) final;
public:
    const std::vector<char>& data() const;

    virtual I::OStream& operator<< (bool) override;
    virtual I::OStream& operator<< (char) override;
    virtual I::OStream& operator<< (unsigned char) override;
    virtual I::OStream& operator<< (short) override;
    virtual I::OStream& operator<< (unsigned short) override;
    virtual I::OStream& operator<< (int) override;
    //virtual I::OStream& operator<< (std::size_t) override;
    virtual I::OStream& operator<< (float) override;
    virtual I::OStream& operator<< (double) override;
    virtual I::OStream& operator<< (const std::string&) override;
    virtual I::OStream& operator<< (uint32_t) override;
    virtual I::OStream& operator<< (uint64_t) override;
};

class GAPI_EXPORTS ByteMemoryInStream final: public I::IStream {
    const std::vector<char>& m_storage;
    size_t m_idx = 0u;

    void check(std::size_t n) { (void) n; GAPI_DbgAssert(m_idx+n-1 < m_storage.size()); }
    uint32_t getU32() { uint32_t v{}; *this >> v; return v; };

    //virtual I::IStream& operator>> (uint32_t &) final;

public:
    explicit ByteMemoryInStream(const std::vector<char> &data);

    virtual I::IStream& operator>> (bool &) override;
    virtual I::IStream& operator>> (std::vector<bool>::reference) override;
    virtual I::IStream& operator>> (char &) override;
    virtual I::IStream& operator>> (unsigned char &) override;
    virtual I::IStream& operator>> (short &) override;
    virtual I::IStream& operator>> (unsigned short &) override;
    virtual I::IStream& operator>> (int &) override;
    virtual I::IStream& operator>> (float &) override;
    virtual I::IStream& operator>> (double &) override;
    //virtual I::IStream& operator>> (std::size_t &) override;
    virtual I::IStream& operator >> (uint32_t &) override;
    virtual I::IStream& operator >> (uint64_t &) override;
    virtual I::IStream& operator>> (std::string &) override;
};

GAPI_EXPORTS void serialize(I::OStream& os, const cv::GMetaArgs &ma);
GAPI_EXPORTS void serialize(I::OStream& os, const cv::GRunArgs &ra);
GAPI_EXPORTS GMetaArgs meta_args_deserialize(I::IStream& is);
GAPI_EXPORTS GRunArgs run_args_deserialize(I::IStream& is);

} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_COMMON_SERIALIZATION_HPP
