#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

static vector<int> readAudioFile(string file, int audioStream)
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, audioStream,
                            CAP_PROP_VIDEO_STREAM, -1,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

    cap.open(file, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't open file" << endl;
        return {};
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    vector<int> inputAudio;
    vector<int> frameVec;
    Mat frame;
    for (;;)
    {
        if (cap.grab())
        {
            cap.retrieve(frame, audioBaseIndex);
            frameVec = frame;
            inputAudio.insert(inputAudio.end(), frameVec.begin(), frameVec.end());
        }
        else
        {
            cout << "Number of samples: " << inputAudio.size() << endl;
            break;
        }
    }
    return inputAudio;
}

static vector<int> readAudioMicrophone()
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, -1   };

    cap.open(0, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open microphone" << endl;
        return {};
    }

    const int audioBaseIndex =  static_cast<int>(cap.get(CAP_PROP_AUDIO_BASE_INDEX));
    const int numberOfChannels =  static_cast<int>(cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS));
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString( static_cast<int>(cap.get(CAP_PROP_AUDIO_DATA_DEPTH))) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    const double cvTickFreq = getTickFrequency();
    int64 sysTimeCurr = getTickCount();
    int64 sysTimePrev = sysTimeCurr;

    vector<int> frameVec;
    vector<int> inputAudio;
    Mat frame;
    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < 10)
    {
        if (cap.grab())
        {
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                cap.retrieve(frame, audioBaseIndex + nCh);
                frameVec = frame;
                inputAudio.insert(inputAudio.end(), frameVec.begin(), frameVec.end());
                sysTimeCurr = getTickCount();
            }
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }
    cout << "Number of samples: " << inputAudio.size() << endl;
    return inputAudio;
}

static int getSamplingRate(string file, int audioStream)
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, audioStream,
                            CAP_PROP_VIDEO_STREAM, -1,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

    cap.open(file, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't open file" << endl;
        return -1;
    }
    return static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
}

static int getSamplingRate()
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, -1   };

    cap.open(0, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open microphone" << endl;
        return -1;
    }
    return static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
}

static Mat drawAmplitude(vector<int>& inputAudio)
{
    int rows = 400;
    int cols = 900;
    Scalar color = Scalar(247,111,87);
    int thickness = 5;
    int frameVectorRows = 500;
    int middle = frameVectorRows / 2;
    // usually the input data is too big, so it is necessary
    // to reduce size using interpolation of data
    int frameVectorCols = 40000;
    if (static_cast<int>(inputAudio.size()) < frameVectorCols)
    {
        frameVectorCols = static_cast<int>(inputAudio.size());
    }

    Mat img = Mat(frameVectorRows, frameVectorCols, CV_8UC3 , Scalar(255,255,255)); // white background

    vector<double>reshape_audio(inputAudio.size());
    for (size_t i = 0; i < inputAudio.size(); ++i)
    {
        reshape_audio[i]=static_cast<double>(inputAudio[i]);
    }

    Mat img_frameVector_resize( 1, static_cast<int>(reshape_audio.size()), CV_64F , reshape_audio.data());
    resize(img_frameVector_resize, img_frameVector_resize, Size(frameVectorCols, 1), INTER_LINEAR);
    reshape_audio = img_frameVector_resize;

    // normalization data by maximum element
    double minCv; double maxCv; Point minLoc; Point maxLoc;
    minMaxLoc(inputAudio, &minCv, &maxCv, &minLoc, &maxLoc);

    int max_item = static_cast<int>(max(abs(minCv), abs(maxCv)));

    // if all data values are zero (silence)
    if (max_item == 0)
    {
        max_item = 1;
    }

    for (size_t i = 0; i < reshape_audio.size(); ++i)
    {
        if (reshape_audio[i] > 0)
        {
            reshape_audio[i] = middle - reshape_audio[i] * middle / max_item;
        }
        else
        {
            reshape_audio[i] = frameVectorRows - middle - reshape_audio[i] * middle / max_item;
        }
    }

    for (int i = 1; i < static_cast<int>(reshape_audio.size()); ++i)
    {
        line(img, Point(i-1, static_cast<int>(reshape_audio[i-1])), Point(i, static_cast<int>(reshape_audio[i])), color, thickness);
    }
    resize(img, img, Size(cols, rows), INTER_AREA );

    return img;
}

static Mat drawAmplitudeScale(Mat& inputImg, const vector<int>& inputAudio, int samplingRate,
                       string grid = "off",
                       int rows = 800, int cols = 900,
                       int xmarkup = 5, int ymarkup = 5,
                       int xmin = 0, int xmax = 0)
{
    // function of layout drawing for graph of volume amplitudes
    // x axis for time
    // y axis for amplitudes

    // parameters for the new image size
    int preCol = 100;
    int aftCol = 100;
    int preLine = 40;
    int aftLine = 50;

    int frameVectorRows = inputImg.rows;
    int frameVectorCols = inputImg.cols;

    int totalRows = preLine + frameVectorRows + aftLine;
    int totalCols = preCol + frameVectorCols + aftCol;

    Mat imgTotal = Mat(totalRows, totalCols, CV_8UC3, Scalar(255, 255, 255));
    inputImg.copyTo(imgTotal(Rect(preCol, preLine, inputImg.cols, inputImg.rows)));

    double delta;
    // calculating values on x axis
    if (xmax == 0)
    {
        xmax = static_cast<int>(inputAudio.size()) / samplingRate;
    }
    std::vector<double> xList(xmarkup);
    if (xmax >= xmarkup)
    {
        delta = (xmax - xmin) / (xmarkup - 1);
        for (int i = 0; i < xmarkup; ++i)
        {
            xList[i] = (xmin + delta * i);
        }
    }
    else
    {
        // this case is used to display a dynamic update
        vector<double> tmpXList;
        for (int i = xmin; i < xmax; ++i)
        {
            tmpXList.push_back(i + 1);
        }
        int k = 0;
        for (int i = xmarkup - static_cast<int>(tmpXList.size()); i < xmarkup; ++i)
        {
            xList[i] = tmpXList[k];
            k += 1;
        }
    }

    // calculating values on y axis
    double minCv; double maxCv; Point minLoc; Point maxLoc;
    minMaxLoc(inputAudio, &minCv, &maxCv, &minLoc, &maxLoc);
    int ymin = static_cast<int>(minCv);
    int ymax = static_cast<int>(maxCv);

    std::vector<double> yList(ymarkup);
    delta = (ymax - ymin) / (ymarkup - 1);
    for (int i = 0; i < ymarkup; ++i)
    {
        yList[i] = ymin + delta * i;
    }

    // parameters for layout drawing
    int textThickness = 1;
    int gridThickness = 1;
    Scalar gridColor(0, 0, 0);
    Scalar textColor(0, 0, 0);
    float fontScale = 0.5;

    // horizontal axis
    line(imgTotal, Point(preCol, totalRows - aftLine), Point(preCol + frameVectorCols, totalRows - aftLine),
        gridColor, gridThickness);
    // vertical axis
    line(imgTotal, Point(preCol, preLine), Point(preCol, preLine + frameVectorRows),
        gridColor, gridThickness);

    // parameters for layout calculation
    int serifSize = 10;
    int indentDownX = serifSize * 2;
    int indentDownY = serifSize / 2;
    int indentLeftX = serifSize;
    int indentLeftY = 2 * preCol / 3;


    // drawing layout for x axis
    int numX = frameVectorCols / (xmarkup - 1);
    for (size_t i = 0; i < xList.size(); ++i)
    {
        int a1 = static_cast<int>(preCol + i * numX);
        int a2 = frameVectorRows + preLine;

        int b1 = a1;
        int b2 = a2 + serifSize;

        if (grid == "on")
        {
            int d1 = a1;
            int d2 = preLine;
            line(imgTotal, Point(a1, a2), Point(d1, d2), gridColor, gridThickness);
        }
        line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
        putText(imgTotal, to_string(int(xList[i])), Point(b1 - indentLeftX, b2 + indentDownX),
                FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
    }

    // drawing layout for y axis
    int numY = frameVectorRows / (ymarkup - 1);
    for (size_t i = 0; i < yList.size(); ++i) {
        int a1 = preCol;
        int a2 = static_cast<int>(totalRows - aftLine - i * numY);
        int b1 = preCol - serifSize;
        int b2 = a2;
        if (grid == "on")
        {
            int d1 = preCol + frameVectorCols;
            int d2 = a2;
            line(imgTotal, Point(a1, a2), Point(d1, d2), gridColor, gridThickness);
        }
        line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
        putText(imgTotal, to_string(int(yList[i])), Point(b1 - indentLeftY, b2 + indentDownY),
                FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
    }
    resize(imgTotal, imgTotal, Size(cols, rows), INTER_AREA );
    return imgTotal;
}

static vector<vector<double>> STFT(const vector<int>& inputAudio,
                           std::string windowType = "Rect",
                           int windLen = 256, int overlap = 128)
{
    int timeStep = windLen - overlap;
    Mat dstMat;
    vector<double> stftRow;
    vector<double> WindType;
    if (windowType == "Hann")
    {
        double pi = 2 * acos(-1.0);
        for (int j = 1 - windLen; j < windLen; j+=2)
        {
            WindType.push_back(j * (0.5 * (1 - cos(pi * j / (windLen - 1)))));
        }
    }
    else if (windowType == "Hamming")
    {
        double pi = 2 * acos(-1.0);
        for (int j = 1 - windLen; j < windLen; j+=2)
        {
            WindType.push_back(j * (0.53836 - 0.46164 * (cos(pi * j / (windLen - 1)))));
        }
    }
    for (size_t i = 0; i < inputAudio.size(); i += timeStep)
    {
        vector<double>section(windLen, 0);
        for (int j = 0; j < windLen; ++j)
        {
            section[j] = inputAudio[j + i];
        }
        if (windowType == "Hann" || windowType == "Hamming")
        {
            for (size_t j = 0; j < section.size(); ++j)
            {
                section[j] *= WindType[j];
            }
        }

        dft(section, dstMat, DFT_COMPLEX_OUTPUT);

        for (int j = 0; j < dstMat.cols / 4; ++j)
        {
            double complModule = sqrt(dstMat.at<double>(2*j) * dstMat.at<double>(2*j) +
                                      dstMat.at<double>(2*j+1) * dstMat.at<double>(2*j+1));
            stftRow.push_back(complModule);
        }
    }

    size_t xSize = inputAudio.size() / timeStep + 1;
    // we need only the first part of the spectrum, the second part is symmetrical
    size_t ySize = dstMat.cols / 4;

    vector<vector<double>> stft(ySize, vector<double>(xSize, 0.));
    for (size_t i = 0; i < xSize; ++i)
    {
        for (size_t j = 0; j < ySize; ++j)
        {
            // write elements with transposition and convert it to the decibel scale
            double stftElem = stftRow[ i * ySize + j];
            if (stftElem != 0.)
            {
                stft[j][i] = 10 * log10(stftElem);
            }
        }
    }
    return stft;
}

static Mat drawSpectrogram(const vector<vector<double>>& stft)
{
    int rows = 400;
    int cols = 900;
    int frameVectorRows = static_cast<int>(stft.size());
    int frameVectorCols = static_cast<int>(stft[0].size());

    // Normalization of image values from 0 to 255 to get more contrast image
    // and this normalization will be taken into account in the scale drawing
    int colormapImageRows = 255;

    double minCv; double maxCv; Point minLoc; Point maxLoc;
    minMaxLoc(stft[0], &minCv, &maxCv, &minLoc, &maxLoc);
    double maxStft = max(abs(maxCv), abs(minCv));

    for (int i = 1; i < frameVectorRows; ++i)
    {
        minMaxLoc( stft[i], &minCv, &maxCv, &minLoc, &maxLoc);
        maxStft = max(maxStft, max(abs(maxCv), abs(minCv)));
    }
    // if maxStft is zero (silence)
    if (maxStft == 0.)
    {
        maxStft = 1;
    }
    Mat imgSpec = Mat(frameVectorRows, frameVectorCols, CV_8UC1, Scalar(255, 255, 255));

    for (int i = 0; i < frameVectorRows; ++i)
    {
        for (int j = 0; j < frameVectorCols; ++j)
        {
            imgSpec.at<uchar>(frameVectorRows - i - 1, j) = static_cast<uchar>(stft[i][j] * colormapImageRows / maxStft);
        }
    }
    applyColorMap(imgSpec, imgSpec, COLORMAP_INFERNO);
    resize(imgSpec, imgSpec, Size(cols, rows), INTER_AREA);
    return imgSpec;
}

static Mat drawSpectrogramColorbar(Mat& inputImg, const vector<int>& inputAudio,
                            int samplingRate, const vector<vector<double>>& stft,
                            int rows = 800, int cols = 900,
                            int xmarkup = 5, int ymarkup = 5, int zmarkup = 5,
                            int xmin = 0, int xmax = 0)
{
    // function of layout drawing for the three-dimensional graph of the spectrogram
    // x axis for time
    // y axis for frequencies
    // z axis for magnitudes of frequencies shown by color scale

    // parameters for the new image size
    int preCol = 100;
    int aftCol = 100;
    int preLine = 40;
    int aftLine = 50;
    int colColor = 20;
    int indCol = 20;

    int frameVectorRows = inputImg.rows;
    int frameVectorCols = inputImg.cols;

    int totalRows = preLine + frameVectorRows + aftLine;
    int totalCols = preCol + frameVectorCols + aftCol;

    Mat imgTotal = Mat(totalRows, totalCols, CV_8UC3 , Scalar(255, 255, 255));
    inputImg.copyTo(imgTotal(Rect(preCol, preLine, frameVectorCols, frameVectorRows)));

    // colorbar image due to drawSpectrogram(..) picture has been normalised from 255 to 0,
    // so here colorbar has values from 255 to 0
    int colorArrSize = 256;
    Mat imgColorBar = Mat (colorArrSize, colColor, CV_8UC1 , Scalar(255,255,255));
    for (int i = 0; i < colorArrSize; ++i)
    {
        for( int j = 0; j < colColor; ++j)
        {
            imgColorBar.at<uchar>(i, j) = static_cast<uchar>(colorArrSize - 1 - i); // from 255 to 0
        }
    }

    applyColorMap(imgColorBar, imgColorBar, COLORMAP_INFERNO);
    resize(imgColorBar, imgColorBar, Size(colColor, frameVectorRows), INTER_AREA);
    imgColorBar.copyTo(imgTotal(Rect(preCol + frameVectorCols + indCol, preLine, colColor, frameVectorRows)));


    double delta;
    // calculating values on x axis
    if (xmax == 0)
    {
        xmax = static_cast<int>(inputAudio.size()) / samplingRate + 1;
    }
    vector<double> xList(xmarkup, 0);
    if (xmax >= xmarkup)
    {
        delta = (xmax - xmin) / (xmarkup - 1);
        for(int i = 0; i < xmarkup; ++i)
        {
            xList[i] = xmin + delta * i;
        }
    }
    else
    {
        // this case is used to display a dynamic update
        vector<double> tmpXList;
        for(int i = xmin; i < xmax; ++i)
        {
            tmpXList.push_back(i + 1);
        }
        int k = 0;
        for (int i = xmarkup - static_cast<int>(tmpXList.size()); i < xmarkup; ++i)
        {
            xList[i] = tmpXList[k];
            k += 1;
        }
    }

    // calculating values on y axis
    // according to the Nyquist sampling theorem,
    // signal should posses frequencies equal to half of sampling rate
    int ymin = 0;
    int ymax = static_cast<int>(samplingRate / 2);

    vector<double> yList;
    delta = (ymax - ymin) / (ymarkup - 1);
    for(int i = 0; i < ymarkup; ++i)
    {
        yList.push_back(ymin + delta * i);
    }

    // calculating values on z axis
    double minCv; double maxCv; Point minLoc; Point maxLoc;
    minMaxLoc( stft[0], &minCv, &maxCv, &minLoc, &maxLoc);
    double zmin = minCv, zmax = maxCv;

    std::vector<double> zList;
    for (size_t i = 1; i < stft.size(); ++i)
    {
        minMaxLoc( stft[i], &minCv, &maxCv, &minLoc, &maxLoc);
        zmax = max(zmax, maxCv);
        zmin = min(zmin, minCv);
    }
    delta = (zmax - zmin) / (zmarkup - 1);
    for(int i = 0; i < zmarkup; ++i)
    {
        zList.push_back(zmin + delta * i);
    }

    // parameters for layout drawing
    int textThickness = 1;
    int gridThickness = 1;
    Scalar gridColor(0,0,0);
    Scalar textColor(0,0,0);
    float fontScale = 0.5;

    int serifSize = 10;
    int indentDownX = serifSize * 2;
    int indentDownY = serifSize / 2;
    int indentLeftX = serifSize;
    int indentLeftY = 2 * preCol / 3;

    // horizontal axis
    line(imgTotal, Point(preCol, totalRows - aftLine), Point(preCol + frameVectorCols, totalRows - aftLine),
                         gridColor, gridThickness);
    // vertical axis
    line(imgTotal, Point(preCol, preLine), Point(preCol, preLine + frameVectorRows),
                         gridColor, gridThickness);

    // drawing layout for x axis
    int numX = frameVectorCols / (xmarkup - 1);
    for (size_t i = 0; i < xList.size(); ++i)
    {
        int a1 = static_cast<int>(preCol + i * numX);
        int a2 = frameVectorRows + preLine;

        int b1 = a1;
        int b2 = a2 + serifSize;

        line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
        putText(imgTotal, to_string(static_cast<int>(xList[i])), Point(b1 - indentLeftX, b2 + indentDownX),
                FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
    }

    // drawing layout for y axis
    int numY = frameVectorRows / (ymarkup - 1);
    for (size_t i = 0; i < yList.size(); ++i)
    {
        int a1 = preCol;
        int a2 = static_cast<int>(totalRows - aftLine - i * numY);

        int b1 = preCol - serifSize;
        int b2 = a2;

        line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
        putText(imgTotal, to_string(static_cast<int>(yList[i])), Point(b1 - indentLeftY, b2 + indentDownY),
                FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
    }

    // drawing layout for z axis
    int numZ = frameVectorRows / (zmarkup - 1);
    for (size_t i = 0; i < zList.size(); ++i)
    {
        int a1 = preCol + frameVectorCols + indCol + colColor;
        int a2 = static_cast<int>(totalRows - aftLine - i * numZ);

        int b1 = a1 + serifSize;
        int b2 = a2;

        line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
        putText(imgTotal, to_string(static_cast<int>(zList[i])), Point(b1 + 10, b2 + indentDownY),
                FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
    }
    resize(imgTotal, imgTotal, Size(cols, rows), INTER_AREA );
    return imgTotal;
}

static Mat concatenateImages(Mat& img1, Mat& img2, int rows = 800, int cols = 900)
{
    // first image will be under the second image
    int totalRows = img1.rows + img2.rows;
    int totalCols = max(img1.cols , img2.cols);
    // if images columns do not match, the difference is filled in white
    Mat imgTotal = Mat (totalRows, totalCols, CV_8UC3 , Scalar(255, 255, 255));

    img1.copyTo(imgTotal(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(imgTotal(Rect(0, img1.rows, img2.cols, img2.rows)));

    resize(imgTotal, imgTotal, Size(cols, rows));
    return imgTotal;
}

static void dynamicFile(const string file, int audioStream = 1, const string graph = "ampl_and_spec",
                 int frameSizeTime = 5, int updateTime = 1, int waitTime = 1000,
                 string windowType = "Rect", int windLen = 256, int overlap = 128,
                 string grid = "off", int rows = 700, int cols = 900,
                 int xmarkup = 5, int ymarkup = 5, int zmarkup = 5)
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, audioStream,
                            CAP_PROP_VIDEO_STREAM, -1,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

    cap.open(file, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open file" << endl;
        return;
    }

    const int audioBaseIndex = static_cast<int>(cap.get(CAP_PROP_AUDIO_BASE_INDEX));
    const int numberOfChannels = static_cast<int>(cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS));
    int samplingRate = static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));

    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString(static_cast<int>(cap.get(CAP_PROP_AUDIO_DATA_DEPTH))) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    int step = static_cast<int>(updateTime * samplingRate);
    int frameSize = static_cast<int>(frameSizeTime * samplingRate);

    // since the dimensional grid is counted in integer seconds,
    // if duration of audio frame is less than xmarkup, to avoid an incorrect display,
    // xmarkup will be taken equal to duration
    if (frameSizeTime <= xmarkup)
    {
        xmarkup = frameSizeTime;
    }

    vector<int> buffer;
    vector<int> frameVector;
    vector<int> section(frameSize, 0);
    vector<vector<double>>stft;
    Mat frame, imgAmplitude, imgSpec, imgTotal;
    int currentSamples = 0;
    int xmin = 0;
    int xmax = 0;

    for (;;)
    {
        if (cap.grab())
        {
            cap.retrieve(frame, audioBaseIndex);
            frameVector = frame;
            buffer.insert(buffer.end(), frameVector.begin(), frameVector.end());
            int bufferSize = static_cast<int>(buffer.size());
            if (bufferSize >= step)
            {
                currentSamples += bufferSize;
                section.erase(section.begin(), section.begin() + step);
                section.insert(section.end(), buffer.begin(), buffer.end());
                buffer.erase(buffer.begin(), buffer.begin() + step);
                if (currentSamples < frameSize)
                {
                    xmin = 0;
                    xmax = (currentSamples) / samplingRate;
                }
                else
                {
                    xmin = (currentSamples - frameSize) / samplingRate + 1;
                    xmax = (currentSamples) / samplingRate;
                }

                if (graph == "ampl")
                {
                    imgAmplitude = drawAmplitude(section);
                    imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, grid,
                                                          rows, cols, xmarkup, ymarkup, xmin, xmax);
                    imshow("Display amplitude graph", imgAmplitude);
                    waitKey(waitTime);
                }
                else if (graph == "spec")
                {
                    stft = STFT(section, windowType, windLen, overlap);
                    imgSpec = drawSpectrogram(stft);
                    imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                      rows, cols, xmarkup, ymarkup, zmarkup,
                                                      xmin, xmax);
                    imshow("Display spectrogram", imgSpec);
                    waitKey(waitTime);
                }
                else if (graph == "ampl_and_spec")
                {
                    imgAmplitude = drawAmplitude(section);
                    imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, grid,
                                                      rows, cols, xmarkup, ymarkup, xmin, xmax);
                    stft = STFT(section, windowType, windLen, overlap);
                    imgSpec = drawSpectrogram(stft);
                    imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                      rows, cols, xmarkup, ymarkup, zmarkup,
                                                      xmin, xmax);
                    imgTotal = concatenateImages(imgAmplitude, imgSpec);
                    imshow("Display amplitude graph and spectrogram", imgTotal);
                    waitKey(waitTime);
                }
            }
        }
        else
        {
            break;
        }
    }

}

static void dynamicMicrophone(int microTime = 10, const string graph = "ampl_and_spec",
                       int frameSizeTime = 5, int updateTime = 1, int waitTime = 1000,
                       string windowType = "Rect", int windLen = 256, int overlap = 128,
                       string grid = "off", int rows = 700, int cols = 900,
                       int xmarkup = 5, int ymarkup = 5, int zmarkup = 5)
{
    VideoCapture cap;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, -1   };

    cap.open(0, CAP_MSMF, params);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't to open microphone" << endl;
        return;
    }

    const int audioBaseIndex = static_cast<int>(cap.get(CAP_PROP_AUDIO_BASE_INDEX));
    const int numberOfChannels = static_cast<int>(cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS));
    int samplingRate = static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString(static_cast<int>(cap.get(CAP_PROP_AUDIO_DATA_DEPTH))) << endl;
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    const double cvTickFreq = getTickFrequency();
    int64 sysTimeCurr = getTickCount();
    int64 sysTimePrev = sysTimeCurr;

    int step = (updateTime * samplingRate);
    int frameSize = (frameSizeTime * samplingRate);
    // since the dimensional grid is counted in integer seconds,
    // if duration of audio frame is less than xmarkup, to avoid an incorrect display,
    // xmarkup will be taken equal to duration
    if (frameSizeTime <= xmarkup)
    {
        xmarkup = frameSizeTime;
    }

    vector<int> frameVector;
    vector<int> buffer;
    vector<int> section(frameSize, 0);
    Mat frame, imgAmplitude, imgSpec, imgTotal;

    int currentSamples = 0;
    vector<vector<double>> stft;
    int xmin = 0;
    int xmax = 0;
    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < microTime)
    {
        if (cap.grab())
        {
            for (int nCh = 0; nCh < numberOfChannels; nCh++)
            {
                cap.retrieve(frame, audioBaseIndex + nCh);
                frameVector = frame;
                buffer.insert(buffer.end(), frameVector.begin(), frameVector.end());
                sysTimeCurr = getTickCount();

                int bufferSize = static_cast<int>(buffer.size());
                if (bufferSize >= step)
                {
                    currentSamples += step;
                    section.erase(section.begin(), section.begin() + step);
                    section.insert(section.end(), buffer.begin(), buffer.end());
                    buffer.erase(buffer.begin(), buffer.begin() + step);

                    if (currentSamples < frameSize)
                    {
                        xmin = 0;
                        xmax = (currentSamples) / samplingRate;
                    }
                    else
                    {
                        xmin = (currentSamples - frameSize) / samplingRate + 1;
                        xmax = (currentSamples) / samplingRate;
                    }

                    if (graph == "ampl")
                    {
                        imgAmplitude = drawAmplitude(section);
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, grid,
                                                          rows, cols, xmarkup, ymarkup, xmin, xmax);
                        imshow("Display amplitude graph", imgAmplitude);
                        waitKey(waitTime);
                    }
                    else if (graph == "spec")
                    {
                        stft = STFT(section, windowType, windLen, overlap);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                        rows, cols, xmarkup, ymarkup, zmarkup, xmin, xmax);
                        imshow("Display spectrogram", imgSpec);
                        waitKey(waitTime);
                    }
                    else if (graph == "ampl_and_spec")
                    {
                        imgAmplitude = drawAmplitude(section);
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, grid,
                                                          rows, cols, xmarkup, ymarkup, xmin, xmax);
                        stft = STFT(section, windowType, windLen, overlap);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                        rows, cols, xmarkup, ymarkup, zmarkup, xmin, xmax);
                        imgTotal = concatenateImages(imgAmplitude, imgSpec);
                        imshow("Display amplitude graph and spectrogram", imgTotal);
                        waitKey(waitTime);
                    }
                }
            }
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }

}

static bool checkArgs(CommandLineParser parser)
{
    string inputType = parser.get<string>("inputType");
    if ((inputType != "file") && (inputType != "microphone"))
    {
        cout << "Error: " << inputType << " input method doesnt exist" << endl;
        return false;
    }

    string draw = parser.get<string>("draw");
    if ((draw != "static") && (draw != "dynamic"))
    {
        cout << "Error: " << draw << " draw type doesnt exist" << endl;
        return false;
    }

    string graph = parser.get<string>("graph");
    if ((graph != "ampl") && (graph != "spec") && (graph != "ampl_and_spec"))
    {
        cout << "Error: " << graph << " type of graph doesnt exist" << endl;
        return false;
    }

    string windowType = parser.get<string>("windowType");
    if ((windowType != "Rect") && (windowType != "Hann") && (windowType != "Hamming"))
    {
        cout << "Error: " << windowType << " type of window doesnt exist" << endl;
        return false;
    }

    if (parser.get<int>("windLen") <= 0)
    {
        cout << "Error: windLen = " << parser.get<int>("windLen") << " - incorrect value. Must be > 0" << endl;
        return false;
    }

    if (parser.get<int>("overlap") <= 0)
    {
        cout << "Error: overlap = " << parser.get<int>("overlap") << " - incorrect value. Must be > 0" << endl;
        return false;
    }

    string grid = parser.get<string>("grid");
    if ((grid != "on") && (grid != "off"))
    {
        cout << "Error: " << grid << " grid type doesnt exist" << endl;
        return false;
    }
    if (parser.get<int>("rows") <= 0)
    {
        cout << "Error: rows = " << parser.get<int>("rows") << " - incorrect value. Must be > 0" << endl;
        return false;
    }
    if (parser.get<int>("cols") <= 0)
    {
        cout << "Error: cols = " << parser.get<int>("cols") << " - incorrect value. Must be > 0" << endl;
        return false;
    }

    if (parser.get<int>("xmarkup") < 2)
    {
        cout << "Error: xmarkup = " << parser.get<int>("xmarkup") << " - incorrect value. Must be >= 2" << endl;
        return false;
    }
    if (parser.get<int>("ymarkup") < 2)
    {
        cout << "Error: ymarkup = " << parser.get<int>("ymarkup") << " - incorrect value. Must be >= 2" << endl;
        return false;
    }
    if (parser.get<int>("zmarkup") < 2)
    {
        cout << "Error: zmarkup = " << parser.get<int>("zmarkup") << " - incorrect value. Must be >= 2" << endl;
        return false;
    }

    if (parser.get<int>("microTime") <= 0)
    {
        cout << "Error: microTime = " << parser.get<int>("microTime") << " - incorrect value. Must be > 0" << endl;
        return false;
    }
    if (parser.get<int>("frameSizeTime") <= 0)
    {
        cout << "Error: frameSizeTime = " << parser.get<int>("frameSizeTime") << " - incorrect value. Must be > 0" << endl;
        return false;
    }
    if (parser.get<int>("updateTime") <= 0)
    {
        cout << "Error: updateTime = " << parser.get<int>("updateTime") << " - incorrect value. Must be > 0" << endl;
        return false;
    }
    if (parser.get<int>("waitTime") < 0)
    {
        cout << "Error: waitTime = " << parser.get<int>("waitTime") << " - incorrect value. Must be >= 0" << endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |               | this sample draws a volume graph and/or spectrogram of audio/video files and microphone \n\t\tDefault usage: ./Spectrogram.exe}"
        "{inputType i    | file          | file or microphone                       }"
        "{draw d         | static        | type of drawing: \n\t\t\tstatic - for plotting graph(s) across the entire input audio \n\t\t\tdynamic - for plotting graph(s) in a time-updating window}"
        "{graph g        | ampl_and_spec | type of graph: amplitude graph or/and spectrogram. Please use tags below : \n\t\t\tampl - draw the amplitude graph \n\t\t\tspec - draw the spectrogram\n\t\t\tampl_and_spec - draw the amplitude graph and spectrogram on one image under each other}"
        "{audio a        |../../../samples/data/Megamind.avi| name and path to file                    }"
        "{audioStream s  | 1             | CAP_PROP_AUDIO_STREAM value. Select audio stream number }"
        "{windowType t   | Rect          | type of window for STFT. Please use tags below : \n\t\t\tRect/Hann/Hamming }"
        "{windLen l      | 256           | size of window for STFT                  }"
        "{overlap o      | 128           | overlap of windows for STFT (on/off)     }"

        "{grid           | off           | grid on amplitude graph                  }"

        "{rows r         | 800           | rows of output image                     }"
        "{cols c         | 900           | cols of output image                     }"

        "{xmarkup x      | 5             | number of x axis divisions (time asix)   }"
        "{ymarkup y      | 5             | number of y axis divisions (frequency or/and amplitude axis) }"
        "{zmarkup z      | 5             | number of z axis divisions (colorbar)    }"

        "{microTime m    | 20            | time of recording audio with microphone in seconds }"
        "{frameSizeTime f| 5             | size of sliding window in seconds        }"
        "{updateTime u   | 1             | update time of sliding window in seconds }"
        "{waitTime w     | 10            | parameter to cv.waitKey() for dynamic update, takes values in milliseconds }"
        ;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!checkArgs(parser))
    {
        return 1;
    }

    string inputType = parser.get<string>("inputType");
    string draw = parser.get<string>("draw");
    string graph = parser.get<string>("graph");
    string audio = parser.get<string>("audio");
    int audioStream = parser.get<int>("audioStream");

    string windowType = parser.get<string>("windowType");
    int windLen = parser.get<int>("windLen");
    int overlap = parser.get<int>("overlap");

    string grid = parser.get<string>("grid");

    int rows = parser.get<int>("rows");
    int cols = parser.get<int>("cols");

    int xmarkup = parser.get<int>("xmarkup");
    int ymarkup = parser.get<int>("ymarkup");
    int zmarkup = parser.get<int>("zmarkup");

    int microTime = parser.get<int>("microTime");
    int frameSizeTime = parser.get<int>("frameSizeTime");
    int updateTime = parser.get<int>("updateTime");
    int waitTime = parser.get<int>("waitTime");

    if (draw == "static")
    {
        vector<int>inputAudio;
        int samplingRate = 0;
        if (inputType == "file")
        {
            if (audio.empty())
            {
                cout << "Can't open file" << endl;
                return 1;
            }
            inputAudio = readAudioFile(audio, audioStream);
            samplingRate = getSamplingRate(audio, audioStream);
        }
        else if (inputType == "microphone")
        {
            inputAudio = readAudioMicrophone();
            samplingRate = getSamplingRate();
        }
        if ((inputAudio.size() == 0) || samplingRate == 0)
        {
            std::cout << "Can't read audio" << std::endl;
            return 0;
        }

        int duration = static_cast<int>(inputAudio.size()) / samplingRate;

        // since the dimensional grid is counted in integer seconds,
        // if the input audio has an incomplete last second,
        // then it is filled with zeros to complete
        int remainder = static_cast<int>(inputAudio.size()) % samplingRate;
        if (remainder)
        {
            int sizeToFullSec = samplingRate - remainder;
            for (int j = 0; j < sizeToFullSec; ++j)
            {
                inputAudio.push_back(0);
            }
            duration += 1;
            cout << "update duration of audio to full last second with " <<
                     sizeToFullSec << " zero samples" << endl;
            cout << "new number of samples " << inputAudio.size() << endl;
        }
        cout << "duration of audio = "<< duration << " seconds" << endl;

        // since the dimensional grid is counted in integer seconds,
        // if duration of file is less than xmarkup, to avoid an incorrect display,
        // xmarkup will be taken equal to duration
        if (duration <= xmarkup)
        {
            xmarkup = duration + 1;
        }

        if (graph == "ampl")
        {
            Mat imgAmplitude = drawAmplitude(inputAudio);
            imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate, grid,
                                              rows, cols, xmarkup, ymarkup);
            imshow("Display amplitude graph", imgAmplitude);
            waitKey(0);
        }
        else if (graph == "spec")
        {
            vector<vector<double>>stft = STFT(inputAudio, windowType, windLen, overlap);
            Mat imgSpec = drawSpectrogram(stft);
            imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft,
                                              rows, cols, xmarkup, ymarkup, zmarkup);
            imshow("Display spectrogram", imgSpec);
            waitKey(0);
        }
        else if (graph == "ampl_and_spec")
        {
            Mat imgAmplitude = drawAmplitude(inputAudio);
            imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate,
                                              grid, rows, cols, xmarkup, ymarkup);
            vector<vector<double>>stft = STFT(inputAudio, windowType, windLen, overlap);
            Mat imgSpec = drawSpectrogram(stft);
            imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft,
                                              rows, cols, xmarkup, ymarkup, zmarkup);
            Mat imgTotal = concatenateImages(imgAmplitude, imgSpec, rows, cols);
            imshow("Display amplitude graph and spectrogram", imgTotal);
            waitKey(0);
        }
    }
    else if (draw == "dynamic")
    {
        if (inputType == "file")
        {
            dynamicFile(audio, audioStream, graph, frameSizeTime, updateTime, waitTime,
                        windowType, windLen, overlap,
                        grid, rows, cols, xmarkup, ymarkup, zmarkup);
        }
        else if (inputType == "microphone")
        {
            dynamicMicrophone(microTime, graph, frameSizeTime, updateTime, waitTime,
                              windowType, windLen, overlap,
                              grid, rows, cols, xmarkup, ymarkup, zmarkup);
        }
    }

    return 0;
}