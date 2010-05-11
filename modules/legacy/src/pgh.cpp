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
//                For Open Source Computer Vision Library
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

#define _CV_ACOS_TABLE_SIZE  513

static const float icv_acos_table[_CV_ACOS_TABLE_SIZE] = {
    3.14159265f, 3.05317551f, 3.01651113f, 2.98834964f, 2.96458497f, 2.94362719f,
    2.92466119f, 2.90720289f, 2.89093699f, 2.87564455f, 2.86116621f, 2.84738169f,
    2.83419760f, 2.82153967f, 2.80934770f, 2.79757211f, 2.78617145f, 2.77511069f,
    2.76435988f, 2.75389319f, 2.74368816f, 2.73372510f, 2.72398665f, 2.71445741f,
    2.70512362f, 2.69597298f, 2.68699438f, 2.67817778f, 2.66951407f, 2.66099493f,
    2.65261279f, 2.64436066f, 2.63623214f, 2.62822133f, 2.62032277f, 2.61253138f,
    2.60484248f, 2.59725167f, 2.58975488f, 2.58234828f, 2.57502832f, 2.56779164f,
    2.56063509f, 2.55355572f, 2.54655073f, 2.53961750f, 2.53275354f, 2.52595650f,
    2.51922417f, 2.51255441f, 2.50594525f, 2.49939476f, 2.49290115f, 2.48646269f,
    2.48007773f, 2.47374472f, 2.46746215f, 2.46122860f, 2.45504269f, 2.44890314f,
    2.44280867f, 2.43675809f, 2.43075025f, 2.42478404f, 2.41885841f, 2.41297232f,
    2.40712480f, 2.40131491f, 2.39554173f, 2.38980439f, 2.38410204f, 2.37843388f,
    2.37279910f, 2.36719697f, 2.36162673f, 2.35608768f, 2.35057914f, 2.34510044f,
    2.33965094f, 2.33423003f, 2.32883709f, 2.32347155f, 2.31813284f, 2.31282041f,
    2.30753373f, 2.30227228f, 2.29703556f, 2.29182309f, 2.28663439f, 2.28146900f,
    2.27632647f, 2.27120637f, 2.26610827f, 2.26103177f, 2.25597646f, 2.25094195f,
    2.24592786f, 2.24093382f, 2.23595946f, 2.23100444f, 2.22606842f, 2.22115104f,
    2.21625199f, 2.21137096f, 2.20650761f, 2.20166166f, 2.19683280f, 2.19202074f,
    2.18722520f, 2.18244590f, 2.17768257f, 2.17293493f, 2.16820274f, 2.16348574f,
    2.15878367f, 2.15409630f, 2.14942338f, 2.14476468f, 2.14011997f, 2.13548903f,
    2.13087163f, 2.12626757f, 2.12167662f, 2.11709859f, 2.11253326f, 2.10798044f,
    2.10343994f, 2.09891156f, 2.09439510f, 2.08989040f, 2.08539725f, 2.08091550f,
    2.07644495f, 2.07198545f, 2.06753681f, 2.06309887f, 2.05867147f, 2.05425445f,
    2.04984765f, 2.04545092f, 2.04106409f, 2.03668703f, 2.03231957f, 2.02796159f,
    2.02361292f, 2.01927344f, 2.01494300f, 2.01062146f, 2.00630870f, 2.00200457f,
    1.99770895f, 1.99342171f, 1.98914271f, 1.98487185f, 1.98060898f, 1.97635399f,
    1.97210676f, 1.96786718f, 1.96363511f, 1.95941046f, 1.95519310f, 1.95098292f,
    1.94677982f, 1.94258368f, 1.93839439f, 1.93421185f, 1.93003595f, 1.92586659f,
    1.92170367f, 1.91754708f, 1.91339673f, 1.90925250f, 1.90511432f, 1.90098208f,
    1.89685568f, 1.89273503f, 1.88862003f, 1.88451060f, 1.88040664f, 1.87630806f,
    1.87221477f, 1.86812668f, 1.86404371f, 1.85996577f, 1.85589277f, 1.85182462f,
    1.84776125f, 1.84370256f, 1.83964848f, 1.83559892f, 1.83155381f, 1.82751305f,
    1.82347658f, 1.81944431f, 1.81541617f, 1.81139207f, 1.80737194f, 1.80335570f,
    1.79934328f, 1.79533460f, 1.79132959f, 1.78732817f, 1.78333027f, 1.77933581f,
    1.77534473f, 1.77135695f, 1.76737240f, 1.76339101f, 1.75941271f, 1.75543743f,
    1.75146510f, 1.74749565f, 1.74352900f, 1.73956511f, 1.73560389f, 1.73164527f,
    1.72768920f, 1.72373560f, 1.71978441f, 1.71583556f, 1.71188899f, 1.70794462f,
    1.70400241f, 1.70006228f, 1.69612416f, 1.69218799f, 1.68825372f, 1.68432127f,
    1.68039058f, 1.67646160f, 1.67253424f, 1.66860847f, 1.66468420f, 1.66076139f,
    1.65683996f, 1.65291986f, 1.64900102f, 1.64508338f, 1.64116689f, 1.63725148f,
    1.63333709f, 1.62942366f, 1.62551112f, 1.62159943f, 1.61768851f, 1.61377831f,
    1.60986877f, 1.60595982f, 1.60205142f, 1.59814349f, 1.59423597f, 1.59032882f,
    1.58642196f, 1.58251535f, 1.57860891f, 1.57470259f, 1.57079633f, 1.56689007f,
    1.56298375f, 1.55907731f, 1.55517069f, 1.55126383f, 1.54735668f, 1.54344917f,
    1.53954124f, 1.53563283f, 1.53172389f, 1.52781434f, 1.52390414f, 1.51999323f,
    1.51608153f, 1.51216900f, 1.50825556f, 1.50434117f, 1.50042576f, 1.49650927f,
    1.49259163f, 1.48867280f, 1.48475270f, 1.48083127f, 1.47690845f, 1.47298419f,
    1.46905841f, 1.46513106f, 1.46120207f, 1.45727138f, 1.45333893f, 1.44940466f,
    1.44546850f, 1.44153038f, 1.43759024f, 1.43364803f, 1.42970367f, 1.42575709f,
    1.42180825f, 1.41785705f, 1.41390346f, 1.40994738f, 1.40598877f, 1.40202755f,
    1.39806365f, 1.39409701f, 1.39012756f, 1.38615522f, 1.38217994f, 1.37820164f,
    1.37422025f, 1.37023570f, 1.36624792f, 1.36225684f, 1.35826239f, 1.35426449f,
    1.35026307f, 1.34625805f, 1.34224937f, 1.33823695f, 1.33422072f, 1.33020059f,
    1.32617649f, 1.32214834f, 1.31811607f, 1.31407960f, 1.31003885f, 1.30599373f,
    1.30194417f, 1.29789009f, 1.29383141f, 1.28976803f, 1.28569989f, 1.28162688f,
    1.27754894f, 1.27346597f, 1.26937788f, 1.26528459f, 1.26118602f, 1.25708205f,
    1.25297262f, 1.24885763f, 1.24473698f, 1.24061058f, 1.23647833f, 1.23234015f,
    1.22819593f, 1.22404557f, 1.21988898f, 1.21572606f, 1.21155670f, 1.20738080f,
    1.20319826f, 1.19900898f, 1.19481283f, 1.19060973f, 1.18639955f, 1.18218219f,
    1.17795754f, 1.17372548f, 1.16948589f, 1.16523866f, 1.16098368f, 1.15672081f,
    1.15244994f, 1.14817095f, 1.14388370f, 1.13958808f, 1.13528396f, 1.13097119f,
    1.12664966f, 1.12231921f, 1.11797973f, 1.11363107f, 1.10927308f, 1.10490563f,
    1.10052856f, 1.09614174f, 1.09174500f, 1.08733820f, 1.08292118f, 1.07849378f,
    1.07405585f, 1.06960721f, 1.06514770f, 1.06067715f, 1.05619540f, 1.05170226f,
    1.04719755f, 1.04268110f, 1.03815271f, 1.03361221f, 1.02905939f, 1.02449407f,
    1.01991603f, 1.01532509f, 1.01072102f, 1.00610363f, 1.00147268f, 0.99682798f,
    0.99216928f, 0.98749636f, 0.98280898f, 0.97810691f, 0.97338991f, 0.96865772f,
    0.96391009f, 0.95914675f, 0.95436745f, 0.94957191f, 0.94475985f, 0.93993099f,
    0.93508504f, 0.93022170f, 0.92534066f, 0.92044161f, 0.91552424f, 0.91058821f,
    0.90563319f, 0.90065884f, 0.89566479f, 0.89065070f, 0.88561619f, 0.88056088f,
    0.87548438f, 0.87038629f, 0.86526619f, 0.86012366f, 0.85495827f, 0.84976956f,
    0.84455709f, 0.83932037f, 0.83405893f, 0.82877225f, 0.82345981f, 0.81812110f,
    0.81275556f, 0.80736262f, 0.80194171f, 0.79649221f, 0.79101352f, 0.78550497f,
    0.77996593f, 0.77439569f, 0.76879355f, 0.76315878f, 0.75749061f, 0.75178826f,
    0.74605092f, 0.74027775f, 0.73446785f, 0.72862033f, 0.72273425f, 0.71680861f,
    0.71084240f, 0.70483456f, 0.69878398f, 0.69268952f, 0.68654996f, 0.68036406f,
    0.67413051f, 0.66784794f, 0.66151492f, 0.65512997f, 0.64869151f, 0.64219789f,
    0.63564741f, 0.62903824f, 0.62236849f, 0.61563615f, 0.60883911f, 0.60197515f,
    0.59504192f, 0.58803694f, 0.58095756f, 0.57380101f, 0.56656433f, 0.55924437f,
    0.55183778f, 0.54434099f, 0.53675018f, 0.52906127f, 0.52126988f, 0.51337132f,
    0.50536051f, 0.49723200f, 0.48897987f, 0.48059772f, 0.47207859f, 0.46341487f,
    0.45459827f, 0.44561967f, 0.43646903f, 0.42713525f, 0.41760600f, 0.40786755f,
    0.39790449f, 0.38769946f, 0.37723277f, 0.36648196f, 0.35542120f, 0.34402054f,
    0.33224495f, 0.32005298f, 0.30739505f, 0.29421096f, 0.28042645f, 0.26594810f,
    0.25065566f, 0.23438976f, 0.21693146f, 0.19796546f, 0.17700769f, 0.15324301f,
    0.12508152f, 0.08841715f, 0.00000000f
};


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCalcPGH
//    Purpose:
//      Calculates PGH(pairwise geometric histogram) for contour given.
//    Context:
//    Parameters:
//      contour  - pointer to input contour object.
//      pgh      - output histogram
//      ang_dim  - number of angle bins (vertical size of histogram)
//      dist_dim - number of distance bins (horizontal size of histogram)
//    Returns:
//      CV_OK or error code
//    Notes:
//F*/
static CvStatus
icvCalcPGH( const CvSeq * contour, float *pgh, int angle_dim, int dist_dim )
{
    char local_buffer[(1 << 14) + 32];
    float *local_buffer_ptr = (float *)cvAlignPtr(local_buffer,32);
    float *buffer = local_buffer_ptr;
    double angle_scale = (angle_dim - 0.51) / icv_acos_table[0];
    double dist_scale = DBL_EPSILON;
    int buffer_size;
    int i, count, pass;
    int *pghi = (int *) pgh;
    int hist_size = angle_dim * dist_dim;
    CvSeqReader reader1, reader2;       /* external and internal readers */

    if( !contour || !pgh )
        return CV_NULLPTR_ERR;

    if( angle_dim <= 0 || angle_dim > 180 || dist_dim <= 0 )
        return CV_BADRANGE_ERR;

    if( !CV_IS_SEQ_POINT_SET( contour ))
        return CV_BADFLAG_ERR;

    memset( pgh, 0, hist_size * sizeof( pgh[0] ));

    count = contour->total;

    /* allocate buffer for distances */
    buffer_size = count * sizeof( float );

    if( buffer_size > (int)sizeof(local_buffer) - 32 )
    {
        buffer = (float *) cvAlloc( buffer_size );
        if( !buffer )
            return CV_OUTOFMEM_ERR;
    }

    cvStartReadSeq( contour, &reader1, 0 );
    cvStartReadSeq( contour, &reader2, 0 );

    /* calc & store squared edge lengths, calculate maximal distance between edges */
    for( i = 0; i < count; i++ )
    {
        CvPoint pt1, pt2;
        double dx, dy;

        CV_READ_EDGE( pt1, pt2, reader1 );

        dx = pt2.x - pt1.x;
        dy = pt2.y - pt1.y;
        buffer[i] = (float)(1./sqrt(dx * dx + dy * dy));
    }

    /* 
       do 2 passes. 
       First calculates maximal distance.
       Second calculates histogram itself.
     */
    for( pass = 1; pass <= 2; pass++ )
    {
        double dist_coeff = 0, angle_coeff = 0;

        /* run external loop */
        for( i = 0; i < count; i++ )
        {
            CvPoint pt1, pt2;
            int dx, dy;
            int dist = 0;

            CV_READ_EDGE( pt1, pt2, reader1 );

            dx = pt2.x - pt1.x;
            dy = pt2.y - pt1.y;

            if( (dx | dy) != 0 )
            {
                int j;

                if( pass == 2 )
                {
                    dist_coeff = buffer[i] * dist_scale;
                    angle_coeff = buffer[i] * (_CV_ACOS_TABLE_SIZE / 2);
                }

                /* run internal loop (for current edge) */
                for( j = 0; j < count; j++ )
                {
                    CvPoint pt3, pt4;

                    CV_READ_EDGE( pt3, pt4, reader2 );

                    if( i != j )        /* process edge pair */
                    {
                        int d1 = (pt3.y - pt1.y) * dx - (pt3.x - pt1.x) * dy;
                        int d2 = (pt4.y - pt1.y) * dx - (pt2.x - pt1.x) * dy;
                        int cross_flag;
                        int *hist_row = 0;

                        if( pass == 2 )
                        {
                            int dp = (pt4.x - pt3.x) * dx + (pt4.y - pt3.y) * dy;

                            dp = cvRound( dp * angle_coeff * buffer[j] ) +
                                (_CV_ACOS_TABLE_SIZE / 2);
                            dp = MAX( dp, 0 );
                            dp = MIN( dp, _CV_ACOS_TABLE_SIZE - 1 );
                            hist_row = pghi + dist_dim *
                                cvRound( icv_acos_table[dp] * angle_scale );

                            d1 = cvRound( d1 * dist_coeff );
                            d2 = cvRound( d2 * dist_coeff );
                        }

                        cross_flag = (d1 ^ d2) < 0;

                        d1 = CV_IABS( d1 );
                        d2 = CV_IABS( d2 );

                        if( pass == 2 )
                        {
                            if( d1 >= dist_dim )
                                d1 = dist_dim - 1;
                            if( d2 >= dist_dim )
                                d2 = dist_dim - 1;

                            if( !cross_flag )
                            {
                                if( d1 > d2 )   /* make d1 <= d2 */
                                {
                                    d1 ^= d2;
                                    d2 ^= d1;
                                    d1 ^= d2;
                                }

                                for( ; d1 <= d2; d1++ )
                                    hist_row[d1]++;
                            }
                            else
                            {
                                for( ; d1 >= 0; d1-- )
                                    hist_row[d1]++;
                                for( ; d2 >= 0; d2-- )
                                    hist_row[d2]++;
                            }
                        }
                        else    /* 1st pass */
                        {
                            d1 = CV_IMAX( d1, d2 );
                            dist = CV_IMAX( dist, d1 );
                        }
                    }           /* end of processing of edge pair */

                }               /* end of internal loop */

                if( pass == 1 )
                {
                    double scale = dist * buffer[i];

                    dist_scale = MAX( dist_scale, scale );
                }
            }
        }                       /* end of external loop */

        if( pass == 1 )
        {
            dist_scale = (dist_dim - 0.51) / dist_scale;
        }

    }                           /* end of pass on loops */


    /* convert hist to floats */
    for( i = 0; i < hist_size; i++ )
    {
        ((float *) pghi)[i] = (float) pghi[i];
    }

    if( buffer != local_buffer_ptr )
        cvFree( &buffer );

    return CV_OK;
}


CV_IMPL void
cvCalcPGH( const CvSeq * contour, CvHistogram * hist )
{
    int size[CV_MAX_DIM];
    int dims;
    
    if( !CV_IS_HIST(hist))
        CV_Error( CV_StsBadArg, "The histogram header is invalid " );

    if( CV_IS_SPARSE_HIST( hist ))
        CV_Error( CV_StsUnsupportedFormat, "Sparse histogram are not supported" );

    dims = cvGetDims( hist->bins, size ); 

    if( dims != 2 )
        CV_Error( CV_StsBadSize, "The histogram must be two-dimensional" );

    if( !CV_IS_SEQ_POINT_SET( contour ) || CV_SEQ_ELTYPE( contour ) != CV_32SC2 )
        CV_Error( CV_StsUnsupportedFormat, "The contour is not valid or the point type is not supported" );

    IPPI_CALL( icvCalcPGH( contour, ((CvMatND*)(hist->bins))->data.fl, size[0], size[1] ));
}


/* End of file. */
