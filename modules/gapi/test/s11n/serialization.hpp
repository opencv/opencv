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
void deserialize(const GSerialized& gs);
void mkDataNode(ade::Graph& g, const Data& data);
void mkOpNode(ade::Graph& g, const Op& op);
std::vector<ade::NodeHandle> linkNodes(ade::Graph& g);
void putData(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void putOp(GSerialized& s, const GModel::ConstGraph& cg, const ade::NodeHandle nh);
void printOp(const Op& op);
void printData(const Data& data);
void printGSerialized(const GSerialized s);
void cleanGSerializedOps(GSerialized &s);
void cleanupGSerializedDatas(GSerialized &s);

class SerializationStream
{
    std::vector<uint> m_dump_storage{};
public:
    SerializationStream() = default;
    ~SerializationStream()
    {
        m_dump_storage.empty();
    }
    char* getData()
    {
        return (char*)m_dump_storage.data();
    };
    size_t getSize()
    {
        return (size_t)(m_dump_storage.size()*sizeof(uint));
    };
    void putAtom(uint& new_atom)
    {
        m_dump_storage.push_back(new_atom);
    };
};

//Graph dump operators
template<typename _Tp> static inline
SerializationStream& operator << (SerializationStream& os, const _Tp &value);
template<typename _Tp> static inline
SerializationStream&  operator << (SerializationStream& os, const std::vector<_Tp> &values);
SerializationStream& operator << (SerializationStream& os, const Kernel &k);
SerializationStream& operator << (SerializationStream& os, const std::string &str);
SerializationStream& operator << (SerializationStream& os, const std::vector<int> &ints);
SerializationStream& operator << (SerializationStream& os, const std::vector<RcDesc> &descs);
SerializationStream& operator << (SerializationStream& os, const RcDesc &desc);
SerializationStream& operator << (SerializationStream& os, const std::vector<double> &doubles);
SerializationStream& operator << (SerializationStream& os, const double &double_val);
SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Size> &cvsizes);
SerializationStream& operator << (SerializationStream& os, const cv::Size &cvsize);
SerializationStream& operator << (SerializationStream& os, const std::vector<bool> &bools);
SerializationStream& operator << (SerializationStream& os, const bool &bool_val);
SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Scalar> &cvscalars);
SerializationStream& operator << (SerializationStream& os, const cv::Scalar &cvscalar);
SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Point> &cvpoints);
SerializationStream& operator << (SerializationStream& os, const cv::Point &cvpoint);
SerializationStream& operator << (SerializationStream& os, const cv::GMatDesc &cvmatdesc);
SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Mat> &cvmats);
SerializationStream& operator << (SerializationStream& os, const cv::Mat &cvmat);
SerializationStream& operator << (SerializationStream& os, const std::vector<cv::Rect> &cvrects);
SerializationStream& operator << (SerializationStream& os, const cv::Rect &cvrect);
SerializationStream& operator << (SerializationStream& os, const std::vector<Data> &datas);
SerializationStream& operator << (SerializationStream& os, const Data &data);
SerializationStream& operator << (SerializationStream& os, const std::vector<Op> &ops);
SerializationStream& operator << (SerializationStream& os, const Op &op);
SerializationStream& operator << (SerializationStream& os, const int &atom);
SerializationStream& operator << (SerializationStream& os, const uint &atom);

void dumpGSerialized(const GSerialized s, SerializationStream &ofs_serialized);

class DeSerializationStream
{
    std::vector<uint> m_dump_storage{};
    size_t m_storage_index = 0;
public:
    DeSerializationStream(char* data, size_t sz)
    {
        uint* uint_data = (uint*)data;
        size_t uint_size = sz / sizeof(uint);
        for (size_t i = 0; i < uint_size; i++)
        {
            m_dump_storage.push_back(uint_data[i]);
        }
    }
    ~DeSerializationStream()
    {
        m_dump_storage.empty();
        m_storage_index = 0;
    }
    char* getData()
    {
        return (char*)m_dump_storage.data();
    };
    size_t getSize()
    {
        return (size_t)(m_dump_storage.size() * sizeof(uint));
    };
    void putAtom(uint& new_atom)
    {
        m_dump_storage.push_back(new_atom);
    };
    uint getAtom()
    {
        uint next_atom = m_dump_storage.data()[m_storage_index++];
        return next_atom;
    };
    //void init();
    //std::ostream& operator << (std::ostream& os, uint atom);
};

//Graph restore operators
template<typename _Tp> static inline
DeSerializationStream& operator >> (DeSerializationStream& is, /*const*/ _Tp& value);
DeSerializationStream& operator >> (DeSerializationStream& is, Kernel& k);
DeSerializationStream& operator >> (DeSerializationStream& is, std::string& str);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<int>& ints);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<RcDesc>& descs);
DeSerializationStream& operator >> (DeSerializationStream& is, RcDesc& desc);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<double>& doubles);
DeSerializationStream& operator >> (DeSerializationStream& is, double& double_val);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Size>& cvsizes);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::Size& cvsize);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<bool>& bools);
DeSerializationStream& operator >> (DeSerializationStream& is, bool& bool_val);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Scalar>& cvscalars);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::Scalar& cvscalar);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Point>& cvpoints);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::Point& cvpoint);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::GMatDesc& cvmatdesc);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::Mat & cvmat);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Mat>& cvmats);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<cv::Rect>& cvrects);
DeSerializationStream& operator >> (DeSerializationStream& is, cv::Rect& cvrect);
DeSerializationStream& operator >> (DeSerializationStream& is, Data& data);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<Data>& datas);
DeSerializationStream& operator >> (DeSerializationStream& is, Op& op);
DeSerializationStream& operator >> (DeSerializationStream& is, std::vector<Op>& ops);
DeSerializationStream& operator >> (DeSerializationStream& is, int &atom);
DeSerializationStream& operator >> (DeSerializationStream& is, uint &atom);

void readGSerialized(GSerialized &s, DeSerializationStream &ifs_serialized);

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
