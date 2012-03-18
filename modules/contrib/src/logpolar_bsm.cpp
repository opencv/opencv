/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                         License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2012, Willow Garage Inc., all rights reserved.
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
//   * The names of the copyright holders may not be used to endorse or promote products
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

/*************************************************************************************

The LogPolar Blind Spot Model code has been contributed by Fabio Solari.

More details can be found in:

M. Chessa, S. P. Sabatini, F. Solari and F. Tatti (2011)
A Quantitative Comparison of Speed and Reliability for Log-Polar Mapping Techniques,
Computer Vision Systems - 8th International Conference,
ICVS 2011, Sophia Antipolis, France, September 20-22, 2011
(http://dx.doi.org/10.1007/978-3-642-23968-7_5)

***************************************************************************************/

#include "precomp.hpp"

#include <cmath>
#include <vector>

namespace cv
{
    
//------------------------------------interp-------------------------------------------
LogPolar_Interp::LogPolar_Interp(int w, int h, Point2i center, int R, double ro0, int interp, int full, int S, int sp)
{
    if ( (center.x!=w/2 || center.y!=h/2) && full==0) full=1;

    if (center.x<0) center.x=0;
    if (center.y<0) center.y=0;
    if (center.x>=w) center.x=w-1;
    if (center.y>=h) center.y=h-1;

    if (full){
        int rtmp;

        if (center.x<=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)(w-center.x)*(w-center.x));
        if (center.x>=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)center.x*center.x);
        if (center.x>=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)center.x*center.x);
        if (center.x<=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)(w-center.x)*(w-center.x));

        M=2*rtmp; N=2*rtmp;

        top = M/2 - center.y;
        bottom = M/2 - (h-center.y);
        left = M/2 - center.x;
        right = M/2 - (w - center.x);

    }else{
        top=bottom=left=right=0;
        M=w; N=h;
    }

    if (sp){
        int jc=M/2-1, ic=N/2-1;
        int romax=min(ic, jc);
        double a=exp(log((double)(romax/2-1)/(double)ro0)/(double)R);
        S=(int) floor(2*M_PI/(a-1)+0.5);
    }

    this->interp=interp;

    create_map(M, N, R, S, ro0);
}

void LogPolar_Interp::create_map(int M, int N, int R, int S, double ro0)
{
    this->M=M;
    this->N=N;
    this->R=R;
    this->S=S;
    this->ro0=ro0;

    int jc=N/2-1, ic=M/2-1;
    romax=min(ic, jc);
    a=exp(log((double)romax/(double)ro0)/(double)R);
    q=((double)S)/(2*M_PI);

    Rsri = Mat::zeros(S,R,CV_32FC1);
    Csri = Mat::zeros(S,R,CV_32FC1);
    ETAyx = Mat::zeros(N,M,CV_32FC1);
    CSIyx = Mat::zeros(N,M,CV_32FC1);

    for(int v=0; v<S; v++)
    {
        for(int u=0; u<R; u++)
        {
            Rsri.at<float>(v,u)=(float)(ro0*pow(a,u)*sin(v/q)+jc);
            Csri.at<float>(v,u)=(float)(ro0*pow(a,u)*cos(v/q)+ic); 
        }
    }

    for(int j=0; j<N; j++)
    {
        for(int i=0; i<M; i++)
        {
            double theta;
            if(i>=ic)
                theta=atan((double)(j-jc)/(double)(i-ic));
            else
                theta=atan((double)(j-jc)/(double)(i-ic))+M_PI;

            if(theta<0)
                theta+=2*M_PI;

            ETAyx.at<float>(j,i)=(float)(q*theta);

            double ro2=(j-jc)*(j-jc)+(i-ic)*(i-ic);
            CSIyx.at<float>(j,i)=(float)(0.5*log(ro2/(ro0*ro0))/log(a));
        }
    }
}

const Mat LogPolar_Interp::to_cortical(const Mat &source)
{
    Mat out(S,R,CV_8UC1,Scalar(0));
    
    Mat source_border;
    copyMakeBorder(source,source_border,top,bottom,left,right,BORDER_CONSTANT,Scalar(0));

    remap(source_border,out,Csri,Rsri,interp);

    return out;
}


const Mat LogPolar_Interp::to_cartesian(const Mat &source)
{
    Mat out(N,M,CV_8UC1,Scalar(0));

    Mat source_border;
    
    if (interp==INTER_NEAREST || interp==INTER_LINEAR){
        copyMakeBorder(source,source_border,0,1,0,0,BORDER_CONSTANT,Scalar(0));
        Mat rowS0 = source_border.row(S);
        source_border.row(0).copyTo(rowS0);
    } else if (interp==INTER_CUBIC){
        copyMakeBorder(source,source_border,0,2,0,0,BORDER_CONSTANT,Scalar(0));
        Mat rowS0 = source_border.row(S);
        Mat rowS1 = source_border.row(S+1);
        source_border.row(0).copyTo(rowS0);
        source_border.row(1).copyTo(rowS1);
    } else if (interp==INTER_LANCZOS4){
        copyMakeBorder(source,source_border,0,4,0,0,BORDER_CONSTANT,Scalar(0));
        Mat rowS0 = source_border.row(S);
        Mat rowS1 = source_border.row(S+1);
        Mat rowS2 = source_border.row(S+2);
        Mat rowS3 = source_border.row(S+3);
        source_border.row(0).copyTo(rowS0);
        source_border.row(1).copyTo(rowS1);
        source_border.row(2).copyTo(rowS2);
        source_border.row(3).copyTo(rowS3);
    }
    remap(source_border,out,CSIyx,ETAyx,interp);

    Mat out_cropped=out(Range(top,N-1-bottom),Range(left,M-1-right));

    return out_cropped;
}

LogPolar_Interp::~LogPolar_Interp()
{
}

//------------------------------------overlapping----------------------------------

LogPolar_Overlapping::LogPolar_Overlapping(int w, int h, Point2i center, int R, double ro0, int full, int S, int sp)
{
    if ( (center.x!=w/2 || center.y!=h/2) && full==0) full=1;

    if (center.x<0) center.x=0;
    if (center.y<0) center.y=0;
    if (center.x>=w) center.x=w-1;
    if (center.y>=h) center.y=h-1;

    if (full){
        int rtmp;

        if (center.x<=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)(w-center.x)*(w-center.x));
        if (center.x>=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)center.x*center.x);
        if (center.x>=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)center.x*center.x);
        if (center.x<=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)(w-center.x)*(w-center.x));

        M=2*rtmp; N=2*rtmp;

        top = M/2 - center.y;
        bottom = M/2 - (h-center.y);
        left = M/2 - center.x;
        right = M/2 - (w - center.x);

    }else{
        top=bottom=left=right=0;
        M=w; N=h;
    }


    if (sp){
        int jc=M/2-1, ic=N/2-1;
        int romax=min(ic, jc);
        double a=exp(log((double)(romax/2-1)/(double)ro0)/(double)R);
        S=(int) floor(2*M_PI/(a-1)+0.5);
    }

    create_map(M, N, R, S, ro0);
}

void LogPolar_Overlapping::create_map(int M, int N, int R, int S, double ro0)
{
    this->M=M;
    this->N=N;
    this->R=R;
    this->S=S;
    this->ro0=ro0;

    int jc=N/2-1, ic=M/2-1;
    romax=min(ic, jc);
    a=exp(log((double)romax/(double)ro0)/(double)R);
    q=((double)S)/(2*M_PI);
    ind1=0;

    Rsri=Mat::zeros(S,R,CV_32FC1);
    Csri=Mat::zeros(S,R,CV_32FC1);
    ETAyx=Mat::zeros(N,M,CV_32FC1);
    CSIyx=Mat::zeros(N,M,CV_32FC1);
    Rsr.resize(R*S);
    Csr.resize(R*S);
    Wsr.resize(R);
    w_ker_2D.resize(R*S);

    for(int v=0; v<S; v++)
    {
        for(int u=0; u<R; u++)
        {
            Rsri.at<float>(v,u)=(float)(ro0*pow(a,u)*sin(v/q)+jc);
            Csri.at<float>(v,u)=(float)(ro0*pow(a,u)*cos(v/q)+ic); 
            Rsr[v*R+u]=(int)floor(Rsri.at<float>(v,u));
            Csr[v*R+u]=(int)floor(Csri.at<float>(v,u));            
        }
    }

    bool done=false;
    
    for(int i=0; i<R; i++)
    {
        Wsr[i]=ro0*(a-1)*pow(a,i-1);
        if((Wsr[i]>1)&&(done==false))
        {
            ind1=i;
            done =true;
        }
    }
    
    for(int j=0; j<N; j++)
    {
        for(int i=0; i<M; i++)//mdf
        {
            double theta;
            if(i>=ic)
                theta=atan((double)(j-jc)/(double)(i-ic));
            else
                theta=atan((double)(j-jc)/(double)(i-ic))+M_PI;

            if(theta<0)
                theta+=2*M_PI;

            ETAyx.at<float>(j,i)=(float)(q*theta);
            
            double ro2=(j-jc)*(j-jc)+(i-ic)*(i-ic);
            CSIyx.at<float>(j,i)=(float)(0.5*log(ro2/(ro0*ro0))/log(a));
        }
    }

    for(int v=0; v<S; v++)
        for(int u=ind1; u<R; u++)
        {
            //double sigma=Wsr[u]/2.0;
            double sigma=Wsr[u]/3.0;//modf
            int w=(int) floor(3*sigma+0.5);
            w_ker_2D[v*R+u].w=w;
            w_ker_2D[v*R+u].weights.resize((2*w+1)*(2*w+1));
            double dx=Csri.at<float>(v,u)-Csr[v*R+u];
            double dy=Rsri.at<float>(v,u)-Rsr[v*R+u];
            double tot=0;
            for(int j=0; j<2*w+1; j++)
                for(int i=0; i<2*w+1; i++)
                {
                    (w_ker_2D[v*R+u].weights)[j*(2*w+1)+i]=exp(-(pow(i-w-dx, 2)+pow(j-w-dy, 2))/(2*sigma*sigma));
                    tot+=(w_ker_2D[v*R+u].weights)[j*(2*w+1)+i];
                }
            for(int j=0; j<(2*w+1); j++)
                for(int i=0; i<(2*w+1); i++)
                    (w_ker_2D[v*R+u].weights)[j*(2*w+1)+i]/=tot;
        }
}

const Mat LogPolar_Overlapping::to_cortical(const Mat &source)
{
    Mat out(S,R,CV_8UC1,Scalar(0));

    Mat source_border;
    copyMakeBorder(source,source_border,top,bottom,left,right,BORDER_CONSTANT,Scalar(0));

    remap(source_border,out,Csri,Rsri,INTER_LINEAR);

    int wm=w_ker_2D[R-1].w;
    vector<int> IMG((M+2*wm+1)*(N+2*wm+1), 0);

    for(int j=0; j<N; j++)
        for(int i=0; i<M; i++)
            IMG[(M+2*wm+1)*(j+wm)+i+wm]=source_border.at<uchar>(j,i);

    for(int v=0; v<S; v++)
        for(int u=ind1; u<R; u++)
        {
            int w=w_ker_2D[v*R+u].w;
            double tmp=0;
            for(int rf=0; rf<(2*w+1); rf++)
            {
                for(int cf=0; cf<(2*w+1); cf++)
                {
                    double weight=(w_ker_2D[v*R+u]).weights[rf*(2*w+1)+cf];
                    tmp+=IMG[(M+2*wm+1)*((rf-w)+Rsr[v*R+u]+wm)+((cf-w)+Csr[v*R+u]+wm)]*weight;
                }
            }
            out.at<uchar>(v,u)=(uchar) floor(tmp+0.5);
        }

    return out;
}

const Mat LogPolar_Overlapping::to_cartesian(const Mat &source)
{
    Mat out(N,M,CV_8UC1,Scalar(0));

    Mat source_border;
    copyMakeBorder(source,source_border,0,1,0,0,BORDER_CONSTANT,Scalar(0));
    Mat rowS = source_border.row(S);
    source_border.row(0).copyTo(rowS);
    remap(source_border,out,CSIyx,ETAyx,INTER_LINEAR);

    int wm=w_ker_2D[R-1].w;
    
    vector<double> IMG((N+2*wm+1)*(M+2*wm+1), 0.);
    vector<double> NOR((N+2*wm+1)*(M+2*wm+1), 0.);

    for(int v=0; v<S; v++)
        for(int u=ind1; u<R; u++)
        {
            int w=w_ker_2D[v*R+u].w;
            for(int j=0; j<(2*w+1); j++)
            {
                for(int i=0; i<(2*w+1); i++)
                {
                    int ind=(M+2*wm+1)*((j-w)+Rsr[v*R+u]+wm)+(i-w)+Csr[v*R+u]+wm;
                    IMG[ind]+=((w_ker_2D[v*R+u]).weights[j*(2*w+1)+i])*source.at<uchar>(v, u);
                    NOR[ind]+=((w_ker_2D[v*R+u]).weights[j*(2*w+1)+i]);
                }
            }
        }

    for(int i=0; i<((N+2*wm+1)*(M+2*wm+1)); i++)
        IMG[i]/=NOR[i];

    //int xc=M/2-1, yc=N/2-1;

    for(int j=wm; j<N+wm; j++)
        for(int i=wm; i<M+wm; i++)
        {
            /*if(NOR[(M+2*wm+1)*j+i]>0)
                ret[M*(j-wm)+i-wm]=(int) floor(IMG[(M+2*wm+1)*j+i]+0.5);*/
            //int ro=(int)floor(sqrt((double)((j-wm-yc)*(j-wm-yc)+(i-wm-xc)*(i-wm-xc))));
            int csi=(int) floor(CSIyx.at<float>(j-wm,i-wm));

            if((csi>=(ind1-(w_ker_2D[ind1]).w))&&(csi<R))
                out.at<uchar>(j-wm,i-wm)=(uchar) floor(IMG[(M+2*wm+1)*j+i]+0.5);
        }

    Mat out_cropped=out(Range(top,N-1-bottom),Range(left,M-1-right));
    return out_cropped;
}
 
LogPolar_Overlapping::~LogPolar_Overlapping()
{
}

//----------------------------------------adjacent---------------------------------------

LogPolar_Adjacent::LogPolar_Adjacent(int w, int h, Point2i center, int R, double ro0, double smin, int full, int S, int sp)
{
    if ( (center.x!=w/2 || center.y!=h/2) && full==0) full=1;

    if (center.x<0) center.x=0;
    if (center.y<0) center.y=0;
    if (center.x>=w) center.x=w-1;
    if (center.y>=h) center.y=h-1;

    if (full){
        int rtmp;

        if (center.x<=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)(w-center.x)*(w-center.x));
        if (center.x>=w/2 && center.y>=h/2)
            rtmp=(int)sqrt((float)center.y*center.y + (float)center.x*center.x);
        if (center.x>=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)center.x*center.x);
        if (center.x<=w/2 && center.y<=h/2)
            rtmp=(int)sqrt((float)(h-center.y)*(h-center.y) + (float)(w-center.x)*(w-center.x));

        M=2*rtmp; N=2*rtmp;

        top = M/2 - center.y;
        bottom = M/2 - (h-center.y);
        left = M/2 - center.x;
        right = M/2 - (w - center.x);

    }else{
        top=bottom=left=right=0;
        M=w; N=h;
    }

    if (sp){
        int jc=M/2-1, ic=N/2-1;
        int romax=min(ic, jc);
        double a=exp(log((double)(romax/2-1)/(double)ro0)/(double)R);
        S=(int) floor(2*M_PI/(a-1)+0.5);
    }

    create_map(M, N, R, S, ro0, smin);
}


void LogPolar_Adjacent::create_map(int M, int N, int R, int S, double ro0, double smin)
{
    LogPolar_Adjacent::M=M;
    LogPolar_Adjacent::N=N;
    LogPolar_Adjacent::R=R;
    LogPolar_Adjacent::S=S;
    LogPolar_Adjacent::ro0=ro0;
    romax=min(M/2.0, N/2.0);

    a=exp(log(romax/ro0)/(double)R);
    q=S/(2*M_PI);

    A.resize(R*S);
    L.resize(M*N);

    for(int i=0; i<R*S; i++)
        A[i]=0;

    double xc=M/2.0, yc=N/2.0;

    for(int j=0; j<N; j++)
        for(int i=0; i<M; i++)
        {
            double x=i+0.5-xc, y=j+0.5-yc;
            subdivide_recursively(x, y, i, j, 1, smin);
        }
}


void LogPolar_Adjacent::subdivide_recursively(double x, double y, int i, int j, double length, double smin)
{   
    if(length<=smin)
    {
        int u, v;
        if(get_uv(x, y, u, v))
        {
            pixel p;
            p.u=u;
            p.v=v;
            p.a=length*length;
            L[M*j+i].push_back(p);
            A[v*R+u]+=length*length;
        }
    }

    if(length>smin)
    {
        double xs[4], ys[4];
        int us[4], vs[4];

        xs[0]=xs[3]=x+length/4.0;
        xs[1]=xs[2]=x-length/4.0;
        ys[1]=ys[0]=y+length/4.0;
        ys[2]=ys[3]=y-length/4.0;

        for(int z=0; z<4; z++)
            get_uv(xs[z], ys[z], us[z], vs[z]);

        bool c=true;
        for(int w=1; w<4; w++)
        {
            if(us[w]!=us[w-1])
                c=false;
            if(vs[w]!=vs[w-1])
                c=false;
        }

        if(c)
        {
            if(us[0]!=-1)
            {
                pixel p;
                p.u=us[0];
                p.v=vs[0];
                p.a=length*length;
                L[M*j+i].push_back(p);
                A[vs[0]*R+us[0]]+=length*length;
            }
        }

        else
        {
            for(int z=0; z<4; z++)
                if(us[z]!=-1)
                    subdivide_recursively(xs[z], ys[z], i, j, length/2.0, smin);
        }
    }
}


const Mat LogPolar_Adjacent::to_cortical(const Mat &source)
{
    Mat source_border;
    copyMakeBorder(source,source_border,top,bottom,left,right,BORDER_CONSTANT,Scalar(0));

    vector<double> map(R*S, 0.);

    for(int j=0; j<N; j++)
        for(int i=0; i<M; i++)
        {   
            for(size_t z=0; z<(L[M*j+i]).size(); z++)
            {
                map[R*((L[M*j+i])[z].v)+((L[M*j+i])[z].u)]+=((L[M*j+i])[z].a)*(source_border.at<uchar>(j,i));
            }
        }

    for(int i=0; i<R*S; i++)
        map[i]/=A[i];

    Mat out(S,R,CV_8UC1,Scalar(0));

    for(int i=0; i<S; i++)
        for(int j=0;j<R;j++)
            out.at<uchar>(i,j)=(uchar) floor(map[i*R+j]+0.5);

    return out;
}

const Mat LogPolar_Adjacent::to_cartesian(const Mat &source)
{
    vector<double> map(M*N, 0.);

    for(int j=0; j<N; j++)
        for(int i=0; i<M; i++)
        {
            for(size_t z=0; z<(L[M*j+i]).size(); z++)
            {
                map[M*j+i]+=(L[M*j+i])[z].a*source.at<uchar>((L[M*j+i])[z].v,(L[M*j+i])[z].u);
            }
        }

    Mat out(N,M,CV_8UC1,Scalar(0));

    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            out.at<uchar>(i,j)=(uchar) floor(map[i*M+j]+0.5);

    Mat out_cropped=out(Range(top,N-1-bottom),Range(left,M-1-right));
    return out_cropped;
}


bool LogPolar_Adjacent::get_uv(double x, double y, int&u, int&v)
{
    double ro=sqrt(x*x+y*y), theta;
    if(x>0)
        theta=atan(y/x);
    else
        theta=atan(y/x)+M_PI;

    if(ro<ro0||ro>romax)
    {
        u=-1;
        v=-1;
        return false;
    }
    else
    {
        u= (int) floor(log(ro/ro0)/log(a));
        if(theta>=0)
            v= (int) floor(q*theta);
        else
            v= (int) floor(q*(theta+2*M_PI));
        return true;
    }   
}

LogPolar_Adjacent::~LogPolar_Adjacent()
{
}

}

