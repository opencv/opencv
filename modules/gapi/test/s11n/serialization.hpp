// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "compiler/gmodel.hpp"
#ifdef _WIN32
#include <winsock.h>
#else
#include <netinet/in.h>
//#include <arpa/inet.h>
#endif
#include <iostream>
#include <fstream>
#include <string.h>

#include "logger.hpp"

namespace cv
{
namespace gimpl
{
// FIXME? Name is too long?
namespace s11n {

// FIXME? Split RcDesc,Kernel,Op,Data etc
// into serializable/non-serializable parts?
// All structures below are partial copies of default (non-serializable) ones
struct Kernel
{
    std::string name;
    std::string tag;
};

struct RcDesc
{
    GShape shape;
    int    id;
    bool operator==(const RcDesc& rc) const { return id == rc.id && shape == rc.shape; }
};

struct Op
{
    Kernel k;
    // FIXME: GArg needs to be serialized properly
    //GArg data
    std::vector<int>   kind;
    std::vector<int>   opaque_kind;
    std::vector<RcDesc> outs;
    std::vector<RcDesc> ins;
    //opaque args
    std::vector<int> opaque_ints;
    std::vector<double> opaque_doubles;
    std::vector<cv::Size> opaque_cvsizes;
    std::vector<bool> opaque_bools;
    std::vector<cv::Scalar> opaque_cvscalars;
    std::vector<cv::Point> opaque_cvpoints;
    std::vector<cv::Mat> opaque_cvmats;
    std::vector<cv::Rect> opaque_cvrects;
};

struct Data
{
    // GModel::Data consists of shape+(int)rc
    RcDesc rc;
    GMetaArg meta;
    // storage?

    bool operator==(const Data& d) const { return rc == d.rc; }
};

struct GSerialized
{
    // Need to monitor ins/outs of the graph?
    // Remove m_?
    std::vector<Op> m_ops;
    std::vector<Data> m_datas;
};

GSerialized serialize(const gimpl::GModel::ConstGraph& m_gm, const std::vector<ade::NodeHandle>& nodes);

// Stream interfaces, so far temporary
namespace I {
    struct OStream {
        virtual void put(uint32_t) = 0;
        virtual ~OStream() = default;
    };

    struct IStream {
        virtual uint getUInt32() = 0;
        virtual ~IStream() = default;
    };
} // namespace I


//Graph dump operators
I::OStream& operator<< (I::OStream& os, const Kernel &k);
I::OStream& operator<< (I::OStream& os, const std::string &str);
I::OStream& operator<< (I::OStream& os, const std::vector<int> &ints);
I::OStream& operator<< (I::OStream& os, const std::vector<RcDesc> &descs);
I::OStream& operator<< (I::OStream& os, const RcDesc &desc);
I::OStream& operator<< (I::OStream& os, const std::vector<double> &doubles);
I::OStream& operator<< (I::OStream& os, const double &double_val);
I::OStream& operator<< (I::OStream& os, const std::vector<cv::Size> &cvsizes);
I::OStream& operator<< (I::OStream& os, const cv::Size &cvsize);
I::OStream& operator<< (I::OStream& os, const std::vector<bool> &bools);
I::OStream& operator<< (I::OStream& os, const bool &bool_val);
I::OStream& operator<< (I::OStream& os, const std::vector<cv::Scalar> &cvscalars);
I::OStream& operator<< (I::OStream& os, const cv::Scalar &cvscalar);
I::OStream& operator<< (I::OStream& os, const std::vector<cv::Point> &cvpoints);
I::OStream& operator<< (I::OStream& os, const cv::Point &cvpoint);
I::OStream& operator<< (I::OStream& os, const cv::GMatDesc &cvmatdesc);
I::OStream& operator<< (I::OStream& os, const std::vector<cv::Mat> &cvmats);
I::OStream& operator<< (I::OStream& os, const cv::Mat &cvmat);
I::OStream& operator<< (I::OStream& os, const std::vector<cv::Rect> &cvrects);
I::OStream& operator<< (I::OStream& os, const cv::Rect &cvrect);
I::OStream& operator<< (I::OStream& os, const std::vector<Data> &datas);
I::OStream& operator<< (I::OStream& os, const Data &data);
I::OStream& operator<< (I::OStream& os, const std::vector<Op> &ops);
I::OStream& operator<< (I::OStream& os, const Op &op);
I::OStream& operator<< (I::OStream& os, const int &atom);
I::OStream& operator<< (I::OStream& os, uint32_t atom);

void dumpGSerialized(const GSerialized s, I::OStream &ofs_serialized);

//Graph restore operators
I::IStream& operator>> (I::IStream& is, Kernel& k);
I::IStream& operator>> (I::IStream& is, std::string& str);
I::IStream& operator>> (I::IStream& is, std::vector<int>& ints);
I::IStream& operator>> (I::IStream& is, std::vector<RcDesc>& descs);
I::IStream& operator>> (I::IStream& is, RcDesc& desc);
I::IStream& operator>> (I::IStream& is, std::vector<double>& doubles);
I::IStream& operator>> (I::IStream& is, double& double_val);
I::IStream& operator>> (I::IStream& is, std::vector<cv::Size>& cvsizes);
I::IStream& operator>> (I::IStream& is, cv::Size& cvsize);
I::IStream& operator>> (I::IStream& is, std::vector<bool>& bools);
I::IStream& operator>> (I::IStream& is, bool& bool_val);
I::IStream& operator>> (I::IStream& is, std::vector<cv::Scalar>& cvscalars);
I::IStream& operator>> (I::IStream& is, cv::Scalar& cvscalar);
I::IStream& operator>> (I::IStream& is, std::vector<cv::Point>& cvpoints);
I::IStream& operator>> (I::IStream& is, cv::Point& cvpoint);
I::IStream& operator>> (I::IStream& is, cv::GMatDesc& cvmatdesc);
I::IStream& operator>> (I::IStream& is, cv::Mat & cvmat);
I::IStream& operator>> (I::IStream& is, std::vector<cv::Mat>& cvmats);
I::IStream& operator>> (I::IStream& is, std::vector<cv::Rect>& cvrects);
I::IStream& operator>> (I::IStream& is, cv::Rect& cvrect);
I::IStream& operator>> (I::IStream& is, Data& data);
I::IStream& operator>> (I::IStream& is, std::vector<Data>& datas);
I::IStream& operator>> (I::IStream& is, Op& op);
I::IStream& operator>> (I::IStream& is, std::vector<Op>& ops);
I::IStream& operator>> (I::IStream& is, int &atom);
I::IStream& operator>> (I::IStream& is, uint32_t &atom);

void readGSerialized(GSerialized &s, I::IStream &ifs_serialized);
std::vector<ade::NodeHandle> reconstructGModel(ade::Graph &g, const GSerialized &s);

// FIXME: Basic Stream implementaions //////////////////////////////////////////

// Basic (dummy) stream implementations.
class SerializationStream final: public I::OStream {
    std::vector<uint> m_dump_storage{};

public:
    SerializationStream() = default;
    char* getData();
    size_t getSize();
    void putAtom(uint new_atom);

    // Implement OStream interface
    virtual void put(uint32_t) override;
};

class DeSerializationStream final: public I::IStream {
    std::vector<uint> m_dump_storage{};
    size_t m_storage_index = 0;

public:
    DeSerializationStream(char* data, size_t sz);
    char* getData();
    size_t getSize();
    void putAtom(uint& new_atom);
    uint getAtom();

    // Implement IStream interface
    virtual uint32_t getUInt32() override;
};


} // namespace s11n
} // namespace gimpl

namespace detail
{
    template<> struct GTypeTraits<cv::gimpl::s11n::RcDesc>
    {
        static constexpr const ArgKind kind = ArgKind::GOBJREF;
    };
} // namespace detail
} // namespace cv
