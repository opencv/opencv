// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_INSTR_HPP
#define OPENCV_UTILS_INSTR_HPP

#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/tls.hpp>

namespace cv {

//! @addtogroup core_utils
//! @{

#ifdef CV_COLLECT_IMPL_DATA
CV_EXPORTS void setImpl(int flags); // set implementation flags and reset storage arrays
CV_EXPORTS void addImpl(int flag, const char* func = 0); // add implementation and function name to storage arrays
// Get stored implementation flags and functions names arrays
// Each implementation entry correspond to function name entry, so you can find which implementation was executed in which function
CV_EXPORTS int getImpl(std::vector<int> &impl, std::vector<String> &funName);

CV_EXPORTS bool useCollection(); // return implementation collection state
CV_EXPORTS void setUseCollection(bool flag); // set implementation collection state

#define CV_IMPL_PLAIN  0x01 // native CPU OpenCV implementation
#define CV_IMPL_OCL    0x02 // OpenCL implementation
#define CV_IMPL_IPP    0x04 // IPP implementation
#define CV_IMPL_MT     0x10 // multithreaded implementation

#undef CV_IMPL_ADD
#define CV_IMPL_ADD(impl)                                                   \
    if(cv::useCollection())                                                 \
    {                                                                       \
        cv::addImpl(impl, CV_Func);                                         \
    }
#endif

// Instrumentation external interface
namespace instr
{

#if !defined OPENCV_ABI_CHECK

enum TYPE
{
    TYPE_GENERAL = 0,   // OpenCV API function, e.g. exported function
    TYPE_MARKER,        // Information marker
    TYPE_WRAPPER,       // Wrapper function for implementation
    TYPE_FUN,           // Simple function call
};

enum IMPL
{
    IMPL_PLAIN = 0,
    IMPL_IPP,
    IMPL_OPENCL,
};

struct NodeDataTls
{
    NodeDataTls()
    {
        m_ticksTotal = 0;
    }
    uint64      m_ticksTotal;
};

class CV_EXPORTS NodeData
{
public:
    NodeData(const char* funName = 0, const char* fileName = NULL, int lineNum = 0, void* retAddress = NULL, bool alwaysExpand = false, cv::instr::TYPE instrType = TYPE_GENERAL, cv::instr::IMPL implType = IMPL_PLAIN);
    NodeData(NodeData &ref);
    ~NodeData();
    NodeData& operator=(const NodeData&);

    cv::String          m_funName;
    cv::instr::TYPE     m_instrType;
    cv::instr::IMPL     m_implType;
    const char*         m_fileName;
    int                 m_lineNum;
    void*               m_retAddress;
    bool                m_alwaysExpand;
    bool                m_funError;

    volatile int         m_counter;
    volatile uint64      m_ticksTotal;
    TLSDataAccumulator<NodeDataTls> m_tls;
    int                  m_threads;

    // No synchronization
    double getTotalMs()   const { return ((double)m_ticksTotal / cv::getTickFrequency()) * 1000; }
    double getMeanMs()    const { return (((double)m_ticksTotal/m_counter) / cv::getTickFrequency()) * 1000; }
};
bool operator==(const NodeData& lhs, const NodeData& rhs);

typedef Node<NodeData> InstrNode;

CV_EXPORTS InstrNode* getTrace();

#endif // !defined OPENCV_ABI_CHECK


CV_EXPORTS bool       useInstrumentation();
CV_EXPORTS void       setUseInstrumentation(bool flag);
CV_EXPORTS void       resetTrace();

enum FLAGS
{
    FLAGS_NONE              = 0,
    FLAGS_MAPPING           = 0x01,
    FLAGS_EXPAND_SAME_NAMES = 0x02,
};

CV_EXPORTS void       setFlags(FLAGS modeFlags);
static inline void    setFlags(int modeFlags) { setFlags((FLAGS)modeFlags); }
CV_EXPORTS FLAGS      getFlags();

} // namespace instr

//! @}

} // namespace

#endif // OPENCV_UTILS_TLS_HPP
