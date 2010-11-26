/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"

// Function cvCreateBGStatModel creates and returns initialized BG model.
// Parameters:
//      first_frame   - frame from video sequence
//      model_type ñ type of BG model (CV_BG_MODEL_MOG, CV_BG_MODEL_FGD,Ö)
//      parameters  - (optional) if NULL the default parameters of the algorithm will be used
static CvBGStatModel* cvCreateBGStatModel( IplImage* first_frame, int model_type, void* params )
{
    CvBGStatModel* bg_model = NULL;
    
    if( model_type == CV_BG_MODEL_FGD || model_type == CV_BG_MODEL_FGD_SIMPLE )
        bg_model = cvCreateFGDStatModel( first_frame, (CvFGDStatModelParams*)params );
    else if( model_type == CV_BG_MODEL_MOG )
        bg_model = cvCreateGaussianBGModel( first_frame, (CvGaussBGStatModelParams*)params );
    
    return bg_model;
}

/* FOREGROUND DETECTOR INTERFACE */
class CvFGDetectorBase : public CvFGDetector
{
protected:
    CvBGStatModel*  m_pFG;
    int             m_FGType;
    void*           m_pFGParam; /* Foreground parameters. */
    CvFGDStatModelParams        m_ParamFGD;
    CvGaussBGStatModelParams    m_ParamMOG;
    const char*                       m_SaveName;
    const char*                       m_LoadName;
public:
    virtual void SaveState(CvFileStorage* )
    {
        if( m_FGType == CV_BG_MODEL_FGD || m_FGType == CV_BG_MODEL_FGD_SIMPLE )
        {
            if( m_SaveName ) /* File name is not empty */
            {
                //cvSaveStatModel(m_SaveName, (CvFGDStatModel*)m_pFG);
            }
        }
    };
    virtual void LoadState(CvFileStorage* , CvFileNode* )
    {
        if( m_FGType == CV_BG_MODEL_FGD || m_FGType == CV_BG_MODEL_FGD_SIMPLE )
        {
            if( m_LoadName ) /* File name is not empty */
            {
                //cvRestoreStatModel(m_LoadName, (CvFGDStatModel*)m_pFG);
            }
        }
    };
    CvFGDetectorBase(int type, void* param)
    {
        m_pFG = NULL;
        m_FGType = type;
        m_pFGParam = param;
        if( m_FGType == CV_BG_MODEL_FGD || m_FGType == CV_BG_MODEL_FGD_SIMPLE )
        {
            if(m_pFGParam)
            {
              m_ParamFGD = *(CvFGDStatModelParams*)m_pFGParam;
            }
            else
            {
                m_ParamFGD.Lc = CV_BGFG_FGD_LC;
                m_ParamFGD.N1c = CV_BGFG_FGD_N1C;
                m_ParamFGD.N2c = CV_BGFG_FGD_N2C;
                m_ParamFGD.Lcc = CV_BGFG_FGD_LCC;
                m_ParamFGD.N1cc = CV_BGFG_FGD_N1CC;
                m_ParamFGD.N2cc = CV_BGFG_FGD_N2CC;
                m_ParamFGD.delta = CV_BGFG_FGD_DELTA;
                m_ParamFGD.alpha1 = CV_BGFG_FGD_ALPHA_1;
                m_ParamFGD.alpha2 = CV_BGFG_FGD_ALPHA_2;
                m_ParamFGD.alpha3 = CV_BGFG_FGD_ALPHA_3;
                m_ParamFGD.T = CV_BGFG_FGD_T;
                m_ParamFGD.minArea = CV_BGFG_FGD_MINAREA;
                m_ParamFGD.is_obj_without_holes = 1;
                m_ParamFGD.perform_morphing = 1;
            }
            AddParam("LC",&m_ParamFGD.Lc);
            AddParam("alpha1",&m_ParamFGD.alpha1);
            AddParam("alpha2",&m_ParamFGD.alpha2);
            AddParam("alpha3",&m_ParamFGD.alpha3);
            AddParam("N1c",&m_ParamFGD.N1c);
            AddParam("N2c",&m_ParamFGD.N2c);
            AddParam("N1cc",&m_ParamFGD.N1cc);
            AddParam("N2cc",&m_ParamFGD.N2cc);
            m_SaveName = 0;
            m_LoadName = 0;
            AddParam("SaveName",&m_SaveName);
            AddParam("LoadName",&m_LoadName);
            AddParam("ObjWithoutHoles",&m_ParamFGD.is_obj_without_holes);
            AddParam("Morphology",&m_ParamFGD.perform_morphing);

        SetModuleName("FGD");
        }
        else if( m_FGType == CV_BG_MODEL_MOG )			// "MOG" == "Mixture Of Gaussians"
        {
            if(m_pFGParam)
            {
                m_ParamMOG = *(CvGaussBGStatModelParams*)m_pFGParam;
            }
            else
            {                              // These constants are all from cvaux/include/cvaux.h
                m_ParamMOG.win_size      = CV_BGFG_MOG_WINDOW_SIZE;
                m_ParamMOG.bg_threshold  = CV_BGFG_MOG_BACKGROUND_THRESHOLD;

                m_ParamMOG.std_threshold = CV_BGFG_MOG_STD_THRESHOLD;
                m_ParamMOG.weight_init   = CV_BGFG_MOG_WEIGHT_INIT;

                m_ParamMOG.variance_init = CV_BGFG_MOG_SIGMA_INIT*CV_BGFG_MOG_SIGMA_INIT;
                m_ParamMOG.minArea       = CV_BGFG_MOG_MINAREA;
                m_ParamMOG.n_gauss       = CV_BGFG_MOG_NGAUSSIANS;
            }
            AddParam("NG",&m_ParamMOG.n_gauss);

            SetModuleName("MOG");
        }

    };
    ~CvFGDetectorBase()
    {
        if(m_pFG)cvReleaseBGStatModel( &m_pFG );
    }
    void ParamUpdate()
    {
        if(m_pFG)cvReleaseBGStatModel( &m_pFG );
    }

    inline IplImage* GetMask()
    {
        return m_pFG?m_pFG->foreground:NULL;
    };

    /* Process current image: */
    virtual void    Process(IplImage* pImg)
    {
        if(m_pFG == NULL)
        {
            void* param = m_pFGParam;
            if( m_FGType == CV_BG_MODEL_FGD || m_FGType == CV_BG_MODEL_FGD_SIMPLE )
            {
                param = &m_ParamFGD;
            }
            else if( m_FGType == CV_BG_MODEL_MOG )
            {
                param = &m_ParamMOG;
            }
            m_pFG = cvCreateBGStatModel(
                pImg,
                m_FGType,
                param);
            LoadState(0, 0);
        }
        else
        {
            cvUpdateBGStatModel( pImg, m_pFG );
        }
    };

    /* Release foreground detector: */
    virtual void    Release()
    {
        SaveState(0);
        if(m_pFG)cvReleaseBGStatModel( &m_pFG );
    };
};

CvFGDetector* cvCreateFGDetectorBase(int type, void *param)
{
    return (CvFGDetector*) new CvFGDetectorBase(type, param);
}
