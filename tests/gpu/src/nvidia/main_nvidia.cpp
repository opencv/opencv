#pragma warning (disable : 4408 4201 4100)
 
#include <cstdio>

#include "NCV.hpp"
#include "NCVHaarObjectDetection.hpp"

#include "TestIntegralImage.h"
#include "TestIntegralImageSquared.h"
#include "TestRectStdDev.h"
#include "TestResize.h"
#include "TestCompact.h"
#include "TestTranspose.h"

#include "TestDrawRects.h"
#include "TestHypothesesGrow.h"
#include "TestHypothesesFilter.h"
#include "TestHaarCascadeLoader.h"
#include "TestHaarCascadeApplication.h"

#include "NCVAutoTestLister.hpp"
#include "NCVTestSourceProvider.hpp"


template <class T_in, class T_out>
void generateIntegralTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<T_in> &src,
                           Ncv32u maxWidth, Ncv32u maxHeight)
{
    for (Ncv32f _i=1.0; _i<maxWidth; _i*=1.2f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "LinIntImgW%dH%d", i, 2);
        testLister.add(new TestIntegralImage<T_in, T_out>(testName, src, i, 2));
    }
    for (Ncv32f _i=1.0; _i<maxHeight; _i*=1.2f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "LinIntImgW%dH%d", 2, i);
        testLister.add(new TestIntegralImage<T_in, T_out>(testName, src, 2, i));
    }

    //test VGA
    testLister.add(new TestIntegralImage<T_in, T_out>("LinIntImg_VGA", src, 640, 480));

    //TODO: add tests of various resolutions up to 4096x4096
}


void generateSquaredIntegralTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv8u> &src,
                                  Ncv32u maxWidth, Ncv32u maxHeight)
{
    for (Ncv32f _i=1.0; _i<maxWidth; _i*=1.2f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "SqIntImgW%dH%d", i, 32);
        testLister.add(new TestIntegralImageSquared(testName, src, i, 32));
    }
    for (Ncv32f _i=1.0; _i<maxHeight; _i*=1.2f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "SqIntImgW%dH%d", 32, i);
        testLister.add(new TestIntegralImageSquared(testName, src, 32, i));
    }

    //test VGA
    testLister.add(new TestIntegralImageSquared("SqLinIntImg_VGA", src, 640, 480));

    //TODO: add tests of various resolutions up to 4096x4096
}


void generateRectStdDevTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv8u> &src,
                             Ncv32u maxWidth, Ncv32u maxHeight)
{
    NcvRect32u rect(1,1,18,18);

    for (Ncv32f _i=32; _i<maxHeight/2 && _i < maxWidth/2; _i*=1.2f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "RectStdDevW%dH%d", i*2, i);
        testLister.add(new TestRectStdDev(testName, src, i*2, i, rect, 1, true));
        testLister.add(new TestRectStdDev(testName, src, i*2, i, rect, 1.5, false));
        testLister.add(new TestRectStdDev(testName, src, i-1, i*2-1, rect, 1, false));
        testLister.add(new TestRectStdDev(testName, src, i-1, i*2-1, rect, 2.5, true));
    }

    //test VGA
    testLister.add(new TestRectStdDev("RectStdDev_VGA", src, 640, 480, rect, 1, true));

    //TODO: add tests of various resolutions up to 4096x4096
}


template <class T>
void generateResizeTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<T> &src)
{
    //test VGA
    for (Ncv32u i=1; i<480; i+=3)
    {
        char testName[80];
        sprintf(testName, "TestResize_VGA_s%d", i);
        testLister.add(new TestResize<T>(testName, src, 640, 480, i, true));
        testLister.add(new TestResize<T>(testName, src, 640, 480, i, false));
    }

    //test HD
    for (Ncv32u i=1; i<1080; i+=5)
    {
        char testName[80];
        sprintf(testName, "TestResize_1080_s%d", i);
        testLister.add(new TestResize<T>(testName, src, 1920, 1080, i, true));
        testLister.add(new TestResize<T>(testName, src, 1920, 1080, i, false));
    }

    //TODO: add tests of various resolutions up to 4096x4096
}


void generateNPPSTVectorTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv32u> &src, Ncv32u maxLength)
{
    //compaction
    for (Ncv32f _i=256.0; _i<maxLength; _i*=1.1f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "Compaction%d", i);
        testLister.add(new TestCompact(testName, src, i, 0xFFFFFFFF, 30));
    }
    for (Ncv32u i=1; i<260; i++)
    {
        char testName[80];
        sprintf(testName, "Compaction%d", i);
        testLister.add(new TestCompact(testName, src, i, 0xC001C0DE, 70));
        testLister.add(new TestCompact(testName, src, i, 0xC001C0DE, 0));
        testLister.add(new TestCompact(testName, src, i, 0xC001C0DE, 100));
    }
    for (Ncv32u i=256*256-256; i<256*256+257; i++)
    {
        char testName[80];
        sprintf(testName, "Compaction%d", i);
        testLister.add(new TestCompact(testName, src, i, 0xFFFFFFFF, 40));
    }
    for (Ncv32u i=256*256*256-10; i<256*256*256+10; i++)
    {
        char testName[80];
        sprintf(testName, "Compaction%d", i);
        testLister.add(new TestCompact(testName, src, i, 0x00000000, 2));
    }
}


template <class T>
void generateTransposeTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<T> &src)
{
    for (int i=2; i<64; i+=4)
    {
        for (int j=2; j<64; j+=4)
        {
            char testName[80];
            sprintf(testName, "TestTranspose_%dx%d", i, j);
            testLister.add(new TestTranspose<T>(testName, src, i, j));
        }
    }

    for (int i=1; i<128; i+=1)
    {
        for (int j=1; j<2; j+=1)
        {
            char testName[80];
            sprintf(testName, "TestTranspose_%dx%d", i, j);
            testLister.add(new TestTranspose<T>(testName, src, i, j));
        }
    }

    testLister.add(new TestTranspose<T>("TestTranspose_VGA", src, 640, 480));
    testLister.add(new TestTranspose<T>("TestTranspose_HD1080", src, 1920, 1080));

    //regression tests
    testLister.add(new TestTranspose<T>("TestTranspose_reg_0", src, 1072, 375));
}


template <class T>
void generateDrawRectsTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<T> &src, NCVTestSourceProvider<Ncv32u> &src32u,
                            Ncv32u maxWidth, Ncv32u maxHeight)
{
    for (Ncv32f _i=16.0; _i<maxWidth; _i*=1.1f)
    {
        Ncv32u i = (Ncv32u)_i;
        Ncv32u j = maxHeight * i / maxWidth;
        if (!j) continue;
        char testName[80];
        sprintf(testName, "DrawRectsW%dH%d", i, j);

        if (sizeof(T) == sizeof(Ncv32u))
        {
            testLister.add(new TestDrawRects<T>(testName, src, src32u, i, j, i*j/1000+1, (T)0xFFFFFFFF));
        }
        else if (sizeof(T) == sizeof(Ncv8u))
        {
            testLister.add(new TestDrawRects<T>(testName, src, src32u, i, j, i*j/1000+1, (T)0xFF));
        }
        else
        {
            ncvAssertPrintCheck(false, "Attempted to instantiate non-existing DrawRects test suite");
        }
    }

    //test VGA
    testLister.add(new TestDrawRects<T>("DrawRects_VGA", src, src32u, 640, 480, 640*480/1000, (T)0xFF));

    //TODO: add tests of various resolutions up to 4096x4096
}


void generateVectorTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv32u> &src, Ncv32u maxLength)
{
    //growth
    for (Ncv32f _i=10.0; _i<maxLength; _i*=1.1f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "VectorGrow%d", i);
        testLister.add(new TestHypothesesGrow(testName, src, 20, 20, 2.2f, i, i/2, i, i/4));
        testLister.add(new TestHypothesesGrow(testName, src, 10, 42, 1.2f, i, i, i, 0));
    }
    testLister.add(new TestHypothesesGrow("VectorGrow01b", src, 10, 42, 1.2f, 10, 0, 10, 1));
    testLister.add(new TestHypothesesGrow("VectorGrow11b", src, 10, 42, 1.2f, 10, 1, 10, 1));
    testLister.add(new TestHypothesesGrow("VectorGrow10b", src, 10, 42, 1.2f, 10, 1, 10, 0));
    testLister.add(new TestHypothesesGrow("VectorGrow00b", src, 10, 42, 1.2f, 10, 0, 10, 0));
}


void generateHypothesesFiltrationTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv32u> &src, Ncv32u maxLength)
{
    for (Ncv32f _i=1.0; _i<maxLength; _i*=1.1f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "HypFilter%d", i);
        testLister.add(new TestHypothesesFilter(testName, src, i, 3, 0.2f));
        testLister.add(new TestHypothesesFilter(testName, src, i, 0, 0.2f));
        testLister.add(new TestHypothesesFilter(testName, src, i, 1, 0.1f));
    }
}


void generateHaarLoaderTests(NCVAutoTestLister &testLister)
{
    testLister.add(new TestHaarCascadeLoader("haarcascade_eye.xml", "haarcascade_eye.xml"));
    testLister.add(new TestHaarCascadeLoader("haarcascade_frontalface_alt.xml", "haarcascade_frontalface_alt.xml"));
    testLister.add(new TestHaarCascadeLoader("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_alt2.xml"));
    testLister.add(new TestHaarCascadeLoader("haarcascade_frontalface_alt_tree.xml", "haarcascade_frontalface_alt_tree.xml"));
    testLister.add(new TestHaarCascadeLoader("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye_tree_eyeglasses.xml"));
}


void generateHaarApplicationTests(NCVAutoTestLister &testLister, NCVTestSourceProvider<Ncv8u> &src,
                                  Ncv32u maxWidth, Ncv32u maxHeight)
{
    for (Ncv32u i=20; i<512; i+=11)
    {
        for (Ncv32u j=20; j<128; j+=5)
        {
            char testName[80];
            sprintf(testName, "HaarAppl%d_%d", i, j);
            testLister.add(new TestHaarCascadeApplication(testName, src, "haarcascade_frontalface_alt.xml", j, i));
        }
    }
    for (Ncv32f _i=20.0; _i<maxWidth; _i*=1.1f)
    {
        Ncv32u i = (Ncv32u)_i;
        char testName[80];
        sprintf(testName, "HaarAppl%d", i);
        testLister.add(new TestHaarCascadeApplication(testName, src, "haarcascade_frontalface_alt.xml", i, i));
    }
}


static void devNullOutput(const char *msg)
{
}


bool main_nvidia()
{
    printf("Testing NVIDIA Computer Vision SDK\n");
    printf("==================================\n");

    ncvSetDebugOutputHandler(devNullOutput);

    NCVAutoTestLister testListerII("NPPST Integral Image"                   );//,,true, false);
    NCVAutoTestLister testListerSII("NPPST Squared Integral Image"          );//,,true, false);
    NCVAutoTestLister testListerRStdDev("NPPST RectStdDev"                  );//,,true, false);
    NCVAutoTestLister testListerResize("NPPST Resize"                       );//,,true, false);
    NCVAutoTestLister testListerNPPSTVectorOperations("NPPST Vector Operations"  );//,,true, false);
    NCVAutoTestLister testListerTranspose("NPPST Transpose"                 );//,,true, false);

    NCVAutoTestLister testListerVectorOperations("Vector Operations"        );//,,true, false);
    NCVAutoTestLister testListerHaarLoader("Haar Cascade Loader"            );//,,true, false);
    NCVAutoTestLister testListerHaarAppl("Haar Cascade Application"         );//,,true, false);
    NCVAutoTestLister testListerHypFiltration("Hypotheses Filtration"       );//,,true, false);
    NCVAutoTestLister testListerVisualize("Visualization"                   );//,,true, false);

    printf("Initializing data source providers\n");
    NCVTestSourceProvider<Ncv32u> testSrcRandom_32u(2010, 0, 0xFFFFFFFF, 4096, 4096);
    NCVTestSourceProvider<Ncv8u> testSrcRandom_8u(2010, 0, 255, 4096, 4096);
    NCVTestSourceProvider<Ncv64u> testSrcRandom_64u(2010, 0, 0xFFFFFFFFFFFFFFFF, 4096, 4096);
    NCVTestSourceProvider<Ncv8u> testSrcFacesVGA_8u("../../data/group_1_640x480_VGA.pgm");
    NCVTestSourceProvider<Ncv32f> testSrcRandom_32f(2010, -1.0f, 1.0f, 4096, 4096);

    printf("Generating NPPST test suites\n");
    generateIntegralTests<Ncv8u, Ncv32u>(testListerII, testSrcRandom_8u, 4096, 4096);
    generateIntegralTests<Ncv32f, Ncv32f>(testListerII, testSrcRandom_32f, 4096, 4096);
    generateSquaredIntegralTests(testListerSII, testSrcRandom_8u, 4096, 4096);
    generateRectStdDevTests(testListerRStdDev, testSrcRandom_8u, 4096, 4096);
    generateResizeTests(testListerResize, testSrcRandom_32u);
    generateResizeTests(testListerResize, testSrcRandom_64u);
    generateNPPSTVectorTests(testListerNPPSTVectorOperations, testSrcRandom_32u, 4096*4096);
    generateTransposeTests(testListerTranspose, testSrcRandom_32u);
    generateTransposeTests(testListerTranspose, testSrcRandom_64u);

    printf("Generating NCV test suites\n");
    generateDrawRectsTests(testListerVisualize, testSrcRandom_8u, testSrcRandom_32u, 4096, 4096);
    generateDrawRectsTests(testListerVisualize, testSrcRandom_32u, testSrcRandom_32u, 4096, 4096);
    generateVectorTests(testListerVectorOperations, testSrcRandom_32u, 4096*4096);
    generateHypothesesFiltrationTests(testListerHypFiltration, testSrcRandom_32u, 1024);
    generateHaarLoaderTests(testListerHaarLoader);
    generateHaarApplicationTests(testListerHaarAppl, testSrcFacesVGA_8u, 1280, 720);

    // Indicate if at least one test failed
    bool passed = true;

    // Invoke all tests
    passed &= testListerII.invoke();
    passed &= testListerSII.invoke();
    passed &= testListerRStdDev.invoke();
    passed &= testListerResize.invoke();
    passed &= testListerNPPSTVectorOperations.invoke();
    passed &= testListerTranspose.invoke();
    passed &= testListerVisualize.invoke();
    passed &= testListerVectorOperations.invoke();
    passed &= testListerHypFiltration.invoke();
    passed &= testListerHaarLoader.invoke();
    passed &= testListerHaarAppl.invoke();

    return passed;
}
