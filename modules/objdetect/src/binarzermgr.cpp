#include "binarzermgr.hpp"

#include "qbarsource.hpp"

using namespace zxing;

namespace cv {
BinarizerMgr::BinarizerMgr() :m_iNowRotateIndex(0), m_iNextOnceBinarizer(-1)
{
    m_vecRotateBinarizer.push_back(Hybrid);
    m_vecRotateBinarizer.push_back(FastWindow);
    m_vecRotateBinarizer.push_back(SimpleAdaptive);
}

BinarizerMgr::~BinarizerMgr()
{
}

zxing::Ref<Binarizer> BinarizerMgr::Binarize(zxing::Ref<LuminanceSource> source)
{
    BINARIZER binarizerIdx = m_vecRotateBinarizer[m_iNowRotateIndex];
    if (m_iNextOnceBinarizer >= 0)
    {
        binarizerIdx = (BINARIZER)m_iNextOnceBinarizer;
    }
    
    zxing::Ref<Binarizer> binarizer;
    
    switch (binarizerIdx)
    {
        case Hybrid:
            binarizer = new HybridBinarizer(source);
            break;
        case FastWindow:
            binarizer = new FastWindowBinarizer(source);
            break;
        case SimpleAdaptive:
            binarizer = new SimpleAdaptiveBinarizer(source);
            break;
        default:
            binarizer = new HybridBinarizer(source);
            break;
    }
    
    return binarizer;
}

void BinarizerMgr::SwitchBinarizer()
{
    m_iNowRotateIndex = (m_iNowRotateIndex+1) % m_vecRotateBinarizer.size();
}

int BinarizerMgr::GetCurBinarizer()
{
    if (m_iNextOnceBinarizer != -1) return m_iNextOnceBinarizer;
    return m_vecRotateBinarizer[m_iNowRotateIndex];
}

void BinarizerMgr::SetNextOnceBinarizer(int iBinarizerIndex)
{
    m_iNextOnceBinarizer = iBinarizerIndex;
}
}  // namesapce cv