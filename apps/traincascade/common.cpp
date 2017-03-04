#include "common.h"

int updateCascade(MBLBPCascadef * pCascade, int stepWidth)
{
    if(!pCascade)
        return 0;

    for(int cidx = 0; cidx < pCascade->count; cidx++)
    {
        bool next_stage = false;

        MBLBPStagef * pStage = pCascade->stages[cidx];
        double fSum = 0.0;

        for(int i = 0; i < pStage->count; i++)
        {
            MBLBPWeakf * pWeak = pStage->weak_classifiers + i;

            int x = pWeak->x;
            int y = pWeak->y;
            int w = pWeak->cellwidth;
            int h = pWeak->cellheight;

            pWeak->offsets[ 0] = y * stepWidth + (x      );
            pWeak->offsets[ 1] = y * stepWidth + (x + w  );
            pWeak->offsets[ 2] = y * stepWidth + (x + w*2);
            pWeak->offsets[ 3] = y * stepWidth + (x + w*3);

            pWeak->offsets[ 4] = (y+h) * stepWidth + (x      );
            pWeak->offsets[ 5] = (y+h) * stepWidth + (x + w  );
            pWeak->offsets[ 6] = (y+h) * stepWidth + (x + w*2);
            pWeak->offsets[ 7] = (y+h) * stepWidth + (x + w*3);
        
            pWeak->offsets[ 8] = (y+h*2) * stepWidth + (x      );
            pWeak->offsets[ 9] = (y+h*2) * stepWidth + (x + w  );
            pWeak->offsets[10] = (y+h*2) * stepWidth + (x + w*2);
            pWeak->offsets[11] = (y+h*2) * stepWidth + (x + w*3);
        
            pWeak->offsets[12] = (y+h*3) * stepWidth + (x      );
            pWeak->offsets[13] = (y+h*3) * stepWidth + (x + w  );
            pWeak->offsets[14] = (y+h*3) * stepWidth + (x + w*2);
            pWeak->offsets[15] = (y+h*3) * stepWidth + (x + w*3);       
        }

    }
    return 1;
}



bool detectAt(const Mat &sum, MBLBPCascadef * pCascade, int sum_offset)
{
    double confidence=0.0;

    for(int sidx = 0; sidx < pCascade->count; sidx++)
    {
        MBLBPStagef * pStage = pCascade->stages[ sidx ];
        double fSum = 0.0;

        for(int i = 0; i < pStage->count; i++)
        {
            MBLBPWeakf * weak =  pStage->weak_classifiers + i;
            int lbp_code = LBPcode(sum, weak->offsets, sum_offset);
            fSum += weak->look_up_table[lbp_code];
        }

        if(fSum < pStage->threshold)
            return false;
        else
            confidence = fSum - pStage->threshold;

    }

    return (confidence>=0.0);
}