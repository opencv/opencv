#pragma  once
#include "zxing/zxing.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/binarizer.hpp"
#include "zxing/common/global_histogram_binarizer.hpp"
#include "zxing/common/hybrid_binarizer.hpp"
#include "zxing/common/fast_window_binarizer.hpp"
#include "zxing/common/simple_adaptive_binarizer.hpp"

namespace cv {
class BinarizerMgr
{
    enum BINARIZER
    {
        Hybrid = 0,
        FastWindow = 1,
        SimpleAdaptive = 2,
        GlobalHistogram = 3,
        OTSU = 4,
        Niblack = 5,
        Adaptive = 6,
        HistogramBackground = 7
    };
    
public:
    BinarizerMgr();
    ~BinarizerMgr();
    
    zxing::Ref<zxing::Binarizer> Binarize(zxing::Ref<zxing::LuminanceSource> source);
    
    void SwitchBinarizer();
    
    int GetCurBinarizer();
    
    void SetNextOnceBinarizer(int iBinarizerIndex);
    
private:
    int m_iNowRotateIndex;
    int m_iNextOnceBinarizer;
    std::vector<BINARIZER> m_vecRotateBinarizer;
};
}  // namesapce cv