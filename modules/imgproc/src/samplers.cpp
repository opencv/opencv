/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

//此处更改了Include的path，原本是#include "precomp.hpp"
#include "modules/imgproc/src/precomp.hpp"

//（补充）向量化需要的两个头文件：
//  1.<type_traits>：提供 std::is_same，用来在【编译期】判断模板类型是不是 float。
//  2."opencv2/core/hal/intrin.hpp"：OpenCV 的通用 SIMD 抽象层，
//    提供 v_float32、vx_load / vx_store / vx_setall_f32 / v_fma，以及 VTraits<...>::vlanes()。
#include <type_traits>
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
//（1）第一个函数：static const uchar* adjustRect(...)
//      1.它的作用是：
//          当你要从源图像中提取一个矩形区域，但这个区域部分超出了图像边界时，
//          adjustRect 会帮你算出真正能取到的有效区域，
//          并调整源图像指针，让后续的插值代码只处理图像内的像素。

//      2.函数参数解释：
//          const uchar* src：源图像数据的指针。指向我们希望取出的采样区域的左上角。
//          size_t src_step：源图像每行的字节数（步长）。
//          int pix_size：源图像每个像素的字节数（例如，灰度图像是1，RGB图像是3）。
//          Size src_size：源图像的尺寸（宽度和高度）。————一个特殊的class来封装这部分信息
//          Size win_size：我们希望提取的矩形区域的尺寸。
//          Point ip：我们希望提取的矩形区域的左上角在源图像中的坐标。
//          Rect* pRect：输出参数，返回实际能提取的矩形区域在 win_size 中的位置和大小。
//      
//      3.src_step / pix_size是不是源图像的宽度？
//          1.不一定是。因为为了保证内存对齐，OpenCV在存储图像时，常常对每一行做对齐。
//            比如要求每行起始地址是 16 字节、32 字节或 64 字节的倍数。
//            如果一行的像素数乘以每个像素的字节数不是对齐的倍数，OpenCV就会在每行末尾加上一些填充字节。  
//          2.所以，src_step 是源图像每行的实际字节数，可能大于： 图像宽度 * pix_size。
//      
//      4.为什么这里传入的指针指向的是左上角而非中心点？
//          1.因为 adjustRect 的职责是计算“实际能提取的矩形区域”，而矩形区域在内存里的表达方式，天然就是左上角 + 宽高。
//          2.它不需要知道采样区域的“中心点”，我们使用采样区域的左上角，和一个size对象来描述这个矩形区域的大小

//      5.返回的指针是：
//          1.对齐窗口第 0 列（哪怕它在图像缓冲区之外）
//          2.对齐窗口第 0 行；只有窗口上边界超出图像时，才变成图像第 0 行

//      6.最终得到的rect：
//          rect 是有效区域在窗口坐标系里的位置和大小，不是源图坐标系。
//          rect.x：窗口内从第几列开始是有效的
//          rect.y：窗口内从第几行开始是有效的
//          rect.width：窗口内有效列数
//          rect.height：窗口内有效行数
static const uchar*
adjustRect( const uchar* src, size_t src_step, int pix_size,
           Size src_size, Size win_size,
           Point ip, Rect* pRect )
{
    Rect rect;//此变量用于存储实际能提取的源图像的矩形区域在 win_size 中的位置和大小

//（2）先理解源图像内的坐标是如何表示的
//  1.在OpenCV中，图像的坐标系是以左上角为原点（0,0），x轴向右，y轴向下。
//  2.所以，ip.x 和 ip.y 分别表示我们希望提取的矩形区域的左上角在源图像中的横纵坐标。
//  3.那么ip.x >= 0表示左上角在图像的左边界的右边或者正好在左边界上，ip.x < 0表示左上角在图像的左边界的左边（超出图像范围）。
    if( ip.x >= 0 )
    {

//（2）src不是一个作为参数传入的常量const？为什么这里可以直接修改src的值？
//  1.在C++中，函数参数是按值传递的，也就是说，函数内部的 src 是一个局部变量，它是传入的指针的副本。
//  2.虽然 src 是一个指针类型的局部变量，但它本身是可以被修改的。
//  3.但是，src 指向的内存内容是 const uchar*，所以不能通过 src 修改图像数据的内容。
        src += ip.x*pix_size;
        rect.x = 0;
    }

    else
    {
        rect.x = -ip.x;
        if( rect.x > win_size.width )
            rect.x = win_size.width;
    }
//（3）所以第一个分支的意思是：
//  1.如果采样区域的左上角在图像的左边界的右边或者正好在左边界上（ip.x >= 0），
//    那么我们可以直接从 src 指针开始读取数据，所以 rect.x = 0

//  2.如果采样区域的起点比图像左边界更左（ip.x < 0）：
//    说明采样区域左边有一部分落在图像之外，这些像素取不到。
//    rect.x = -ip.x，表示"左边要跳过多少列"，即输出图中有效数据的起始列。
//    例如 ip.x = -3 → rect.x = 3，左边 3 列无源像素，有效数据从第 3 列开始。
//    再看上限保护：若 rect.x 已超过采样区域宽度（整块都在图像左侧之外），
//    就截断成 win_size.width，表示这一行没有任何有效像素。
//      


    if( ip.x < src_size.width - win_size.width )
        rect.width = win_size.width;
    else
    {
        rect.width = src_size.width - ip.x - 1;
        if( rect.width < 0 )
        {
            src += rect.width*pix_size;
            rect.width = 0;
        }
        CV_Assert( rect.width <= win_size.width );
    }
//剩下部分懒得看了，知道功能就行

    if( ip.y >= 0 )
    {
        src += ip.y * src_step;
        rect.y = 0;
    }
    else
        rect.y = -ip.y;

    if( ip.y < src_size.height - win_size.height )
        rect.height = win_size.height;
    else
    {
        rect.height = src_size.height - ip.y - 1;
        if( rect.height < 0 )
        {
            src += rect.height*src_step;
            rect.height = 0;
        }
    }

    *pRect = rect;
    return src - rect.x*pix_size;
}

//==============================================part2==================================================
//（1）这段代码定义了一组函数对象（functor），用来在 getRectSubPix_Cn_ 里做定点数运算和类型转换。
//      1.函数对象functor和一般的函数function有什么区别？
//          1.函数对象就是一个重载了 operator() 的类或结构体的实例。 
//              因为重载了 operator()，它的对象可以像函数一样用 obj(args) 来调用。
//          2.最简单的例子：
        //       struct Add {
        //            int operator()(int a, int b) const {
        //                return a + b;
        //            }
        //      };

        //       int main() {
        //          Add add;              // 创建一个对象
        //          int x = add(3, 4);    // 像函数一样调用，实际调用 add.operator()(3, 4)
        //          // x == 7
        //      }
//          3.operator就是一个根object类定义的抽象函数。有点像java里面的object的print函数
//          4.所以说实际上，这个函数对象的功能，就是允许我们采用调用对象的方法，来调用函数


//      2.定点数运算是什么？
//          在图像插值里，权重是 0~1 的浮点数，比如 a11 = 0.3、a12 = 0.7。
//          如果直接用浮点算插值：float s0 = src[j]*0.3f + src[j+1]*0.7f;
//          开销大。并且很多时候对于比如 8U → 8U 这种像素值只有 0~255 的情况，浮点运算其实是浪费的，而且慢。
//          所以我们把0~1的权重，映射到一个较大的整数区间
//          ————比如这里是SUBPIX_SHIFT=16，也就是2^16 = 65536，即为0~65536这个区间。精度足够大
//          所以整体计算过程变成了：
//              先处理a11和a12两个权重，也就是都乘以65536，或者说移位16bit，然后带入原式子，得到新的s0
//              然后s0采用移位运算而不是除法，刚好移位16bit，得到原本的s0


enum { SUBPIX_SHIFT=16 };

struct scale_fixpt
{
//（2）这个函数就负责更改权重。把浮点数映射成整数
    int operator()(float a) const { return cvRound(a*(1 << SUBPIX_SHIFT)); }
};

//（3）把定点整数还原成 8 位无符号像素值（uchar，0~255），并且带四舍五入。
//      1.a + (1 << (SUBPIX_SHIFT-1)，为什么要把定点整数加上2^15，再除以2^16？
//      2.const 出现在参数列表之后、函数体之前，这个位置有专门的含义——它修饰的是隐式的 this 指针
struct cast_8u
{
    uchar operator()(int a) const { return (uchar)((a + (1 << (SUBPIX_SHIFT-1))) >> SUBPIX_SHIFT); }
};
//（4）把浮点数四舍五入后转成 8 位像素值。
//      1.比如：输入是浮点结果，比如 123.6。然后cvRound(123.6) 返回 124。最后转成 uchar，得到 124。
//      2.这里的uchar不是指针，别误会了。只是一个无符号的char的数据类型。作用是强制转换
struct cast_flt_8u
{
    uchar operator()(float a) const { return (uchar)cvRound(a); }
};

//（5）先理解template是什么？
//      1.template 和 int、class、if、for 一样，都是关键字。它专门用来声明模板。
//      2.模板不是类，是“生成类的模具”
//      3.template<typename T>。这一行是模板声明头。
//          它告诉编译器：下一段定义的东西是一个模板，里面用到了一个还没确定的类型，叫 T。
//          template<class T>   // 和 template<typename T> 等价。只是换一个写法罢了
//      4.然后后面跟着一个模板类型。比如：
//          struct Box {
//              T value;
//              T get() const { return value; }
//          };
//      5.Box 本身不是一个类，它是类模板的名字。只有写成 Box<int> 才是类。
//      6.这种写法，其实就类似于java里的泛型

//（6）所以下面这个struct构造体，本质啥也没干，是一个恒等函数。
//      1.输入什么，就原样返回什么。nop = no operation，空操作。
//      2.它是一个“什么都不做”的函数对象，用来在模板里占位。
//          当某条代码路径不需要缩放、也不需要类型转换时，就用它代替，让模板不用为“没有操作”再写一份特化代码。
//      3.结合下面的函数getRectSubPix_Cn才理解这部分的含义
template<typename _Tp>
struct nop
{
    _Tp operator()(_Tp a) const { return a; }
};

//================================================part3==========================================
//（1）第一句是模板声明头
//  1.它告诉编译器：
//  下面定义的函数是一个模板函数，它用到了五个还没确定的类型，分别叫 _Tp、_DTp、_WTp、ScaleOp、CastOp。
//  2.调用该函数的时候才知道。比如getRectSubPix_Cn_<uchar, uchar, int, scale_fixpt, cast_8u>(...);
//  3.那么从参数意义角度看，这几个类型是什么？
//      1._Tp 是源图像中，C++用来储存每个像素的数据类型。比如8位灰度像素，使用的类型就是uchar。
//          那么这里const _Tp* src就是指向原图像的指针。_Tp类型决定了指针类型，从而得知一个元素占几个byte
//      2._DTp* dst是目标像素类型。同样的这里也是指针
//      3._WTp：中间计算类型。
//      4.ScaleOp：权重缩放操作。ScaleOp 是一个函数对象类型，负责把浮点权重转成 _WTp。
//          它决定“权重怎么从浮点变成中间类型”。也就是WTp
//      5.CastOp：输出转换操作。CastOp 是一个函数对象类型，负责把中间结果 _WTp 转成目标类型 _DTp。
//          它决定“算完的结果怎么写进目标”。
template<typename _Tp, typename _DTp, typename _WTp, class ScaleOp, class CastOp>
void getRectSubPix_Cn_(const _Tp* src, size_t src_step, Size src_size,
                       _DTp* dst, size_t dst_step, Size win_size, Point2f center, int cn )
{
    ScaleOp scale_op; //这里声明的变量，是一个函数对象，决定我们如何把原本的权重映射成定点数
    CastOp cast_op;//把定点数decode成目标类型的数
    Point ip;
    _WTp a11, a12, a21, a22, b1, b2;//声明几个中间类型的变量
    float a, b;
    int i, j, c;

    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;

    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );

    a = center.x - ip.x;
    b = center.y - ip.y;
    a11 = scale_op((1.f-a)*(1.f-b));
    a12 = scale_op(a*(1.f-b));
    a21 = scale_op((1.f-a)*b);
    a22 = scale_op(a*b);
    b1 = scale_op(1.f - b);
    b2 = scale_op(b);

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    if( 0 <= ip.x && ip.x < src_size.width - win_size.width &&
       0 <= ip.y && ip.y < src_size.height - win_size.height)
    {
        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x*cn;

//（2）向量化路径：只覆盖 CV_32FC1（单通道 float）、且窗口完全在图内的情况
#if (CV_SIMD || CV_SIMD_SCALABLE)

//（3）此处if含义：同时要求源类型 float、目标类型 float、中间类型 float、通道数 1。
//      0.先看这个语法；

//      1.为什么需要三个地方的数都是float？
//          1.也就是，我们希望：
//              原图像内，每一个像素的值，在内存中按照float来储存
//              输出图像内，每一个像素的值，在内存中按照float来储存
//              在双线性插值的计算过程中，用来存权重、乘积、累加结果的临时变量和 SIMD 寄存器，元素类型都是 float
//          2.这样处理过程，代码最直接最简单：下面的指令可以直接用。
//              不需要考虑类型转换、没有定点舍入、没有饱和处理
                //v_float32
                //vx_load
                //vx_store
                //v_fma
                //v_mul
                //vx_setall_f32
//          3.如果不是float，而是比如：CV_8U -> CV_8U 
//              使用 _WTp = int，权重是 scale_fixpt 转成的定点整数，目标图也是int类型
//              最后还要 cast_8u 做舍入、右移、截断到 uchar 

//      2.为什么希望cn==1，也就是单通道？
//          如果是多通道，我们每一次只处理一个通道
//          那么读取一个元素的一个通道的时候，就要跳跃式访问内存了
//          如果是单通道，直接连续访问内存即可。开销小
        if constexpr (std::is_same<_Tp, float>::value &&
                      std::is_same<_DTp, float>::value &&
                      std::is_same<_WTp, float>::value)

        {
            if( cn == 1 )
            {
                const int VECSZ = VTraits<v_float32>::vlanes();

                //1.权重在整个 patch 内不变，提到循环外只构造一次,所以加上const关键字
                //2.v_float32 不是 C++ 内建类型，而是 OpenCV 在 SIMD 抽象层里定义的一个类型名。
                    //它通常是一个 typedef 或类型别名，表示“一个包含多个 float 的向量寄存器”。
                    //具体底层类型取决于平台和编译选项
                //3.vx_setall_f32：
                //接收一个标量 float，返回一个 v_float32 向量，并且把这个标量值复制到向量的每一个 lane 中。
                //比如：vx_setall_f32(0.5f)得到的向量类似：[0.5, 0.5, 0.5, 0.5]
                const v_float32 va11 = vx_setall_f32((float)a11);
                const v_float32 va12 = vx_setall_f32((float)a12);
                const v_float32 va21 = vx_setall_f32((float)a21);
                const v_float32 va22 = vx_setall_f32((float)a22);

                for( i = 0; i < win_size.height; i++, src += src_step, dst += dst_step )
                {
                    const float *srow  = (const float*)src;              // 当前行
                    const float *srow2 = (const float*)(src + src_step); // 下一行
                    float *drow = (float*)dst;//采样结果行

                    for( j = 0; j <= win_size.width - VECSZ; j += VECSZ )
                    {
                //4.vx_load 从给定地址开始，连续加载 VTraits<v_float32>::vlanes() 个 float，
                //  然后打包成一个 v_float32 向量。
                //5.这里传入的是指针。
                //  然后连续读入之后，四个向量的第一个元素，分别恰好是窗口该行第一列，
                //  希望从源图采样的左上，左下，右上，右下像素
                //6.因为该行第一列的右上像素，恰好是第二列的左上像素。所以连续错位地读取这些像素，是可行的
                        v_float32 vs00 = vx_load(srow  + j    ); // 左上
                        v_float32 vs10 = vx_load(srow  + j + 1); // 右上
                        v_float32 vs01 = vx_load(srow2 + j    ); // 左下
                        v_float32 vs11 = vx_load(srow2 + j + 1); // 右下

                        // s = 左上*a11 + 右上*a12 + 左下*a21 + 右下*a22
                //这里进行的加法和乘法，是元素之间彼此独立地进行的。于是就可以加速了
                        vx_store(drow + j, v_fma(vs00, va11,
                                          v_fma(vs10, va12,
                                          v_fma(vs01, va21, v_mul(vs11, va22)))));
                    }

                    // 标量尾巴：宽度不是 VECSZ 整数倍时剩下的列
                    for( ; j < win_size.width; j++ )
                        drow[j] = (float)(srow[j  ]*a11 + srow[j+1  ]*a12 +
                                          srow2[j ]*a21 + srow2[j+1]*a22);
                }
                return; // 该分支已处理完整个 patch，直接返回
            }
        }
#endif

        //-------- 标量路径：cn==3、8U 定点、无 SIMD 等所有其余情况 --------
        win_size.width *= cn;

        for( i = 0; i < win_size.height; i++, src += src_step, dst += dst_step )
        {
            for( j = 0; j <= win_size.width - 2; j += 2 )
            {
                _WTp s0 = src[j]*a11 + src[j+cn]*a12 + src[j+src_step]*a21 + src[j+src_step+cn]*a22;
                _WTp s1 = src[j+1]*a11 + src[j+cn+1]*a12 + src[j+src_step+1]*a21 + src[j+src_step+cn+1]*a22;
                dst[j] = cast_op(s0);
                dst[j+1] = cast_op(s1);
            }

            for( ; j < win_size.width; j++ )
            {
                _WTp s0 = src[j]*a11 + src[j+cn]*a12 + src[j+src_step]*a21 + src[j+src_step+cn]*a22;
                dst[j] = cast_op(s0);
            }
        }
    }
    else
    {
        Rect r;
        src = (const _Tp*)adjustRect( (const uchar*)src, src_step*sizeof(*src),
                                     sizeof(*src)*cn, src_size, win_size, ip, &r);

        for( i = 0; i < win_size.height; i++, dst += dst_step )
        {
            const _Tp *src2 = src + src_step;
            _WTp s0;

            if( i < r.y || i >= r.height )
                src2 -= src_step;

            for( c = 0; c < cn; c++ )
            {
                s0 = src[r.x*cn + c]*b1 + src2[r.x*cn + c]*b2;
                for( j = 0; j < r.x; j++ )
                    dst[j*cn + c] = cast_op(s0);
                s0 = src[r.width*cn + c]*b1 + src2[r.width*cn + c]*b2;
                for( j = r.width; j < win_size.width; j++ )
                    dst[j*cn + c] = cast_op(s0);
            }

            for( j = r.x*cn; j < r.width*cn; j++ )
            {
                s0 = src[j]*a11 + src[j+cn]*a12 + src2[j]*a21 + src2[j+cn]*a22;
                dst[j] = cast_op(s0);
            }

            if( i < r.height )
                src = src2;
        }
    }
}

static void getRectSubPix_8u32f
( const uchar* src, size_t src_step, Size src_size,
 float* dst, size_t dst_step, Size win_size, Point2f center0, int cn )
{
    Point2f center = center0;
    Point ip;

    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;

    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );

    if( cn == 1 &&
       0 <= ip.x && ip.x + win_size.width < src_size.width &&
       0 <= ip.y && ip.y + win_size.height < src_size.height &&
       win_size.width > 0 && win_size.height > 0 )
    {
        float a = center.x - ip.x;
        float b = center.y - ip.y;
        a = MAX(a,0.0001f);
        float a12 = a*(1.f-b);
        float a22 = a*b;
        float b1 = 1.f - b;
        float b2 = b;
        double s = (1. - a)/a;

        src_step /= sizeof(src[0]);
        dst_step /= sizeof(dst[0]);

        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x;

        for( ; win_size.height--; src += src_step, dst += dst_step )
        {
            float prev = (1 - a)*(b1*src[0] + b2*src[src_step]);
            for( int j = 0; j < win_size.width; j++ )
            {
                float t = a12*src[j+1] + a22*src[j+1+src_step];
                dst[j] = prev + t;
                prev = (float)(t*s);
            }
        }
    }
    else
    {
        getRectSubPix_Cn_<uchar, float, float, nop<float>, nop<float> >
        (src, src_step, src_size, dst, dst_step, win_size, center0, cn );
    }
}

}


void cv::getRectSubPix( InputArray _image, Size patchSize, Point2f center,
                       OutputArray _patch, int patchType )
{
    CV_INSTRUMENT_REGION();

    Mat image = _image.getMat();
    int depth = image.depth(), cn = image.channels();
    int ddepth = patchType < 0 ? depth : CV_MAT_DEPTH(patchType);

    CV_Assert( cn == 1 || cn == 3 );

    _patch.create(patchSize, CV_MAKETYPE(ddepth, cn));
    Mat patch = _patch.getMat();

    CALL_HAL(getRectSubPix, cv_hal_getRectSubPix,
             image.type(), image.data, image.step, image.cols, image.rows,
             patch.type(), patch.data, patch.step, patch.cols, patch.rows,
             center.x, center.y);

    if( depth == CV_8U && ddepth == CV_8U )
        getRectSubPix_Cn_<uchar, uchar, int, scale_fixpt, cast_8u>
        (image.ptr(), image.step, image.size(), patch.ptr(), patch.step, patch.size(), center, cn);
    else if( depth == CV_8U && ddepth == CV_32F )
        getRectSubPix_8u32f
        (image.ptr(), image.step, image.size(), patch.ptr<float>(), patch.step, patch.size(), center, cn);
    else if( depth == CV_32F && ddepth == CV_32F )
        getRectSubPix_Cn_<float, float, float, nop<float>, nop<float> >
        (image.ptr<float>(), image.step, image.size(), patch.ptr<float>(), patch.step, patch.size(), center, cn);
    else
        CV_Error( cv::Error::StsUnsupportedFormat, "Unsupported combination of input and output formats");
}
