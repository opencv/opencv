// FaceDetection.h: interface for the FaceDetection class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CVFACEDETECTION_H_
#define _CVFACEDETECTION_H_

#include "cvfacetemplate.h"
#include "cvface.h"
#include "cvboostingtemplate.h"

typedef struct CvContourRect
{
	int		iNumber;  //порядковый номер атрибута
	int		iType;    //тип объекта
	int		iFlags;   //свободное поле
	CvSeq	*seqContour; //адрес начала записи объекта 
	int		iContourLength;  //длина записи векторов
	CvRect	r;    //описаный прямоугольник
	CvPoint pCenter; // center of rect
	int		iColor;//    цвет заполнения контура
} CvContourRect;

//class Face;

class ListElem
{
public:
	ListElem();
	ListElem(Face * pFace,ListElem * pHead);
	virtual ~ListElem();
	ListElem * m_pNext;
	ListElem * m_pPrev;
	Face * m_pFace;
};//class ListElem

class List
{
public:
	List();
	int AddElem(Face * pFace);
	virtual ~List();
	Face* GetData();
        long m_FacesCount;
private:
	ListElem * m_pHead;
	ListElem * m_pCurElem;
};//class List


class FaceDetection  
{
public:
	void FindFace(IplImage* img);
	void CreateResults(CvSeq * lpSeq);
	FaceDetection();
	virtual ~FaceDetection();
	void SetBoosting(bool bBoosting) {m_bBoosting = bBoosting;}
	bool isPostBoosting() {return m_bBoosting;}
protected:

	IplImage* m_imgGray;
	IplImage* m_imgThresh;
	int m_iNumLayers;
	CvMemStorage* m_mstgContours;
	CvSeq* m_seqContours[MAX_LAYERS];
	CvMemStorage* m_mstgRects;
	CvSeq* m_seqRects;
	
	bool m_bBoosting;
	List * m_pFaceList;

protected:
	void ResetImage();
	void FindContours(IplImage* imgGray);
	void AddContours2Rect(CvSeq*  seq, int color, int iLayer);
	void ThresholdingParam(IplImage* imgGray, int iNumLayers, int& iMinLevel, int& iMaxLevel, int& iStep);
	void FindCandidats();
	void PostBoostingFindCandidats(IplImage * FaceImage);
};

inline void ReallocImage(IplImage** ppImage, CvSize sz, long lChNum)
{
    IplImage* pImage;
    if( ppImage == NULL ) 
		return;
    pImage = *ppImage;
    if( pImage != NULL )
    {
        if (pImage->width != sz.width || pImage->height != sz.height || pImage->nChannels != lChNum)
            cvReleaseImage( &pImage );
    }
    if( pImage == NULL )
        pImage = cvCreateImage( sz, IPL_DEPTH_8U, lChNum);
    *ppImage = pImage;
};



#endif // !defined(AFX_FACEDETECTION_H__55865033_D8E5_4DD5_8925_34C2285BB1BE__INCLUDED_)
