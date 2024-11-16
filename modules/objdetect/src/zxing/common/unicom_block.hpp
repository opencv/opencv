#pragma once
#include "counted.hpp"
#include "bit_matrix.hpp"
#include <vector>
#include <cstring>

namespace zxing
{
class UnicomBlock : public Counted
{
public:
    UnicomBlock(int iMaxHeight, int iMaxWidth);
    ~UnicomBlock();
    
    void Init();
    void Reset(Ref<BitMatrix> poImage);
    
    unsigned short GetUnicomBlockIndex(int y, int x);
    
    int GetUnicomBlockSize(int y, int x);
    
    int GetMinPoint(int y, int x, int &iMinY, int &iMinX);
    int GetMaxPoint(int y, int x, int &iMaxY, int &iMaxX);
    
private:
    void Bfs(int y, int x);
    
    int m_iHeight;
    int m_iWidth;
    
    unsigned short m_iNowIdx;
    bool m_bInit;
    std::vector<unsigned short> m_vcIndex;
    std::vector<unsigned short> m_vcCount;
    std::vector<int> m_vcMinPnt;
    std::vector<int> m_vcMaxPnt;
    std::vector<int> m_vcQueue;
    static short SEARCH_POS[4][2];
    
    Ref<BitMatrix> m_poImage;
};
}  // namespace zxing
