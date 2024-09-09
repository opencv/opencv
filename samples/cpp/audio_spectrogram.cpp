#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

class AudioDrawing
{
public:
    AudioDrawing(const CommandLineParser& parser) {
        if (!initAndCheckArgs(parser))
        {
            cerr << "Error: Wrong input arguments" << endl;
            exit(0);
        }
        Draw();
    }

    void Draw() {
        if (draw == "static")
        {
            vector<int>inputAudio = {};
            int samplingRate = 0;
            if (inputType == "file")
            {
                samplingRate = readAudioFile(audio, inputAudio);
            }
            else if (inputType == "microphone")
            {
                samplingRate = readAudioMicrophone(inputAudio);
            }
            if ((inputAudio.size() == 0) || samplingRate <= 0)
            {
                cerr << "Error: problems with audio reading, check input arguments" << endl;
                return;
            }

            int duration = static_cast<int>(inputAudio.size()) / samplingRate;

            int remainder = static_cast<int>(inputAudio.size()) % samplingRate;
            if (remainder)
            {
                int sizeToFullSec = samplingRate - remainder;
                for (int j = 0; j < sizeToFullSec; ++j)
                {
                    inputAudio.push_back(0);
                }
                duration += 1;
                cout << "Update duration of audio to full last second with " <<
                        sizeToFullSec << " zero samples" << endl;
                cout << "New number of samples " << inputAudio.size() << endl;
            }
            cout << "Duration of audio = " << duration << " seconds" << endl;

            if (duration <= xmarkup)
            {
                xmarkup = duration + 1;
            }

            if (graph == "ampl")
            {
                Mat imgAmplitude = drawAmplitude(inputAudio);
                imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate);
                // 注释掉显示图像的部分
                // imshow("Display amplitude graph", imgAmplitude);
                // waitKey(0);
                saveImage(imgAmplitude, "amplitude_graph.png");
            }
            else if (graph == "spec")
            {
                vector<vector<double>>stft = STFT(inputAudio);
                Mat imgSpec = drawSpectrogram(stft);
                imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft);
                // 注释掉显示图像的部分
                // imshow("Display spectrogram", imgSpec);
                // waitKey(0);
                saveImage(imgSpec, "spectrogram.png");
            }
            else if (graph == "ampl_and_spec")
            {
                Mat imgAmplitude = drawAmplitude(inputAudio);
                imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate);
                vector<vector<double>>stft = STFT(inputAudio);
                Mat imgSpec = drawSpectrogram(stft);
                imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft);
                Mat imgTotal = concatenateImages(imgAmplitude, imgSpec);
                // 注释掉显示图像的部分
                // imshow("Display amplitude graph and spectrogram", imgTotal);
                // waitKey(0);
                saveImage(imgTotal, "amplitude_and_spectrogram.png");
            }
        }
        else if (draw == "dynamic")
        {
            if (inputType == "file")
            {
                dynamicFile(audio);
            }
            else if (inputType == "microphone")
            {
                dynamicMicrophone();
            }
        }
    }

    ~AudioDrawing() {
    }
int readAudioFile(string file, vector<int>& inputAudio)
{
    VideoCapture cap;
    cap.open(file); // Use basic open to check if file can be opened without specific parameters
    if (!cap.isOpened())
    {
        cerr << "Error : Can't read audio file: '" << file << "' with audioStream = " << audioStream << endl;
        return -1;
    }
    cout << "Video file opened successfully" << endl;

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    const int numberOfChannels = (int)cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS);
    cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString((int)cap.get(CAP_PROP_AUDIO_DATA_DEPTH)) << endl;
    int samplingRate = static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
    cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND) << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
    cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

    vector<int> frameVec;
    Mat frame;
    for (;;)
    {
        if (cap.grab())
        {
            cout << "Frame grabbed successfully" << endl;
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
    return samplingRate;
}


    int readAudioMicrophone(vector<int>& inputAudio)
    {
        VideoCapture cap;
        vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                                CAP_PROP_VIDEO_STREAM, -1   };

        cap.open(0, CAP_ANY, params);
        if (!cap.isOpened())
        {
            cerr << "Error: Can't open microphone" << endl;
            return -1;
        }

        const int audioBaseIndex =  static_cast<int>(cap.get(CAP_PROP_AUDIO_BASE_INDEX));
        const int numberOfChannels =  static_cast<int>(cap.get(CAP_PROP_AUDIO_TOTAL_CHANNELS));
        cout << "CAP_PROP_AUDIO_DATA_DEPTH: " << depthToString( static_cast<int>(cap.get(CAP_PROP_AUDIO_DATA_DEPTH))) << endl;
        int samplingRate = static_cast<int>(cap.get(CAP_PROP_AUDIO_SAMPLES_PER_SECOND));
        cout << "CAP_PROP_AUDIO_SAMPLES_PER_SECOND: " << samplingRate << endl;
        cout << "CAP_PROP_AUDIO_TOTAL_CHANNELS: " << numberOfChannels << endl;
        cout << "CAP_PROP_AUDIO_TOTAL_STREAMS: " << cap.get(CAP_PROP_AUDIO_TOTAL_STREAMS) << endl;

        const double cvTickFreq = getTickFrequency();
        int64 sysTimeCurr = getTickCount();
        int64 sysTimePrev = sysTimeCurr;

        vector<int> frameVec;
        Mat frame;
        while ((sysTimeCurr - sysTimePrev) / cvTickFreq < microTime)
        {
            if (cap.grab())
            {
                cap.retrieve(frame, audioBaseIndex);
                frameVec = frame;
                inputAudio.insert(inputAudio.end(), frameVec.begin(), frameVec.end());
                sysTimeCurr = getTickCount();
            }
            else
            {
                cerr << "Error: Grab error" << endl;
                break;
            }
        }
        cout << "Number of samples: " << inputAudio.size() << endl;
        return samplingRate;
    }

    Mat drawAmplitude(vector<int>& inputAudio)
    {
        Scalar color = Scalar(247,111,87);
        int thickness = 5;
        int frameVectorRows = 500;
        int middle = frameVectorRows / 2;

        int frameVectorCols = 40000;
        if (static_cast<int>(inputAudio.size()) < frameVectorCols)
        {
            frameVectorCols = static_cast<int>(inputAudio.size());
        }

        Mat img(frameVectorRows, frameVectorCols, CV_8UC3 , Scalar(255,255,255)); // white background

        vector<double>reshapeAudio(inputAudio.size());
        for (size_t i = 0; i < inputAudio.size(); ++i)
        {
            reshapeAudio[i]=static_cast<double>(inputAudio[i]);
        }

        Mat img_frameVector( 1, static_cast<int>(reshapeAudio.size()), CV_64F , reshapeAudio.data());
        Mat img_frameVector_resize;
        resize(img_frameVector, img_frameVector_resize, Size(frameVectorCols, 1), INTER_LINEAR);
        reshapeAudio = img_frameVector_resize;

        normalize(reshapeAudio, reshapeAudio, 1.0, 0.0, NORM_INF);

        for (size_t i = 0; i < reshapeAudio.size(); ++i)
        {
            reshapeAudio[i] = middle - reshapeAudio[i] * middle;
        }

        for (int i = 1; i < static_cast<int>(reshapeAudio.size()); ++i)
        {
            line(img, Point(i-1, static_cast<int>(reshapeAudio[i-1])), Point(i, static_cast<int>(reshapeAudio[i])), color, thickness);
        }
        Mat resImage;
        resize(img, resImage, Size(900, 400), INTER_AREA );
        return resImage;
    }

    Mat drawAmplitudeScale(Mat& inputImg, const vector<int>& inputAudio, int samplingRate,
                           int xmin = 0, int xmax = 0)
    {
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

        if (xmax == 0)
        {
            xmax = static_cast<int>(inputAudio.size()) / samplingRate;
        }
        std::vector<double> xList(xmarkup);
        if (xmax >= xmarkup)
        {
            double deltax = (xmax - xmin) / (xmarkup - 1);
            for (int i = 0; i < xmarkup; ++i)
            {
                xList[i] = (xmin + deltax * i);
            }
        }
        else
        {
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

        double minCv; double maxCv; Point minLoc; Point maxLoc;
        minMaxLoc(inputAudio, &minCv, &maxCv, &minLoc, &maxLoc);
        int ymin = static_cast<int>(minCv);
        int ymax = static_cast<int>(maxCv);

        std::vector<double> yList(ymarkup);
        double deltay = (ymax - ymin) / (ymarkup - 1);
        for (int i = 0; i < ymarkup; ++i)
        {
            yList[i] = ymin + deltay * i;
        }

        int textThickness = 1;
        int gridThickness = 1;
        Scalar gridColor(0, 0, 0);
        Scalar textColor(0, 0, 0);
        float fontScale = 0.5;

        line(imgTotal, Point(preCol, totalRows - aftLine), Point(preCol + frameVectorCols, totalRows - aftLine),
            gridColor, gridThickness);

        line(imgTotal, Point(preCol, preLine), Point(preCol, preLine + frameVectorRows),
            gridColor, gridThickness);

        int serifSize = 10;
        int indentDownX = serifSize * 2;
        int indentDownY = serifSize / 2;
        int indentLeftX = serifSize;
        int indentLeftY = 2 * preCol / 3;

        int numX = frameVectorCols / (xmarkup - 1);
        for (size_t i = 0; i < xList.size(); ++i)
        {
            int a1 = static_cast<int>(preCol + i * numX);
            int a2 = frameVectorRows + preLine;

            int b1 = a1;
            int b2 = a2 + serifSize;

            if (enableGrid)
            {
                int d1 = a1;
                int d2 = preLine;
                line(imgTotal, Point(a1, a2), Point(d1, d2), gridColor, gridThickness);
            }
            line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
            putText(imgTotal, to_string(int(xList[i])), Point(b1 - indentLeftX, b2 + indentDownX),
                    FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
        }

        int numY = frameVectorRows / (ymarkup - 1);
        for (size_t i = 0; i < yList.size(); ++i) {
            int a1 = preCol;
            int a2 = static_cast<int>(totalRows - aftLine - i * numY);
            int b1 = preCol - serifSize;
            int b2 = a2;
            if (enableGrid)
            {
                int d1 = preCol + frameVectorCols;
                int d2 = a2;
                line(imgTotal, Point(a1, a2), Point(d1, d2), gridColor, gridThickness);
            }
            line(imgTotal, Point(a1, a2), Point(b1, b2), gridColor, gridThickness);
            putText(imgTotal, to_string(int(yList[i])), Point(b1 - indentLeftY, b2 + indentDownY),
                    FONT_HERSHEY_SIMPLEX, fontScale, textColor, textThickness);
        }
        Mat resImage;
        resize(imgTotal, resImage, Size(cols, rows), INTER_AREA );
        return resImage;
    }

    vector<vector<double>> STFT(const vector<int>& inputAudio)
    {
        int timeStep = windLen - overlap;
        Mat dstMat;
        vector<double> stftRow;
        vector<double> WindType;
        if (windowType == "Hann")
        {
            for (int j = 1 - windLen; j < windLen; j+=2)
            {
                WindType.push_back(j * (0.5 * (1 - cos(CV_PI * j / (windLen - 1)))));
            }
        }
        else if (windowType == "Hamming")
        {
            for (int j = 1 - windLen; j < windLen; j+=2)
            {
                WindType.push_back(j * (0.53836 - 0.46164 * (cos(CV_PI * j / (windLen - 1)))));
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
        size_t ySize = dstMat.cols / 4;

        vector<vector<double>> stft(ySize, vector<double>(xSize, 0.));
        for (size_t i = 0; i < xSize; ++i)
        {
            for (size_t j = 0; j < ySize; ++j)
            {
                double stftElem = stftRow[ i * ySize + j];
                if (stftElem != 0.)
                {
                    stft[j][i] = 10 * log10(stftElem);
                }
            }
        }
        return stft;
    }

    Mat drawSpectrogram(const vector<vector<double>>& stft)
    {
        int frameVectorRows = static_cast<int>(stft.size());
        int frameVectorCols = static_cast<int>(stft[0].size());

        int colormapImageRows = 255;

        double minCv; double maxCv; Point minLoc; Point maxLoc;
        minMaxLoc(stft[0], &minCv, &maxCv, &minLoc, &maxLoc);
        double maxStft = max(abs(maxCv), abs(minCv));

        for (int i = 1; i < frameVectorRows; ++i)
        {
            minMaxLoc( stft[i], &minCv, &maxCv, &minLoc, &maxLoc);
            maxStft = max(maxStft, max(abs(maxCv), abs(minCv)));
        }
        if (maxStft == 0.)
        {
            maxStft = 1;
        }
        Mat imgSpec(frameVectorRows, frameVectorCols, CV_8UC1, Scalar(255, 255, 255));

        for (int i = 0; i < frameVectorRows; ++i)
        {
            for (int j = 0; j < frameVectorCols; ++j)
            {
                imgSpec.at<uchar>(frameVectorRows - i - 1, j) = static_cast<uchar>(stft[i][j] * colormapImageRows / maxStft);
            }
        }
        applyColorMap(imgSpec, imgSpec, COLORMAP_INFERNO);
        Mat resImage;
        resize(imgSpec, resImage, Size(900, 400), INTER_AREA);
        return resImage;
    }

    Mat drawSpectrogramColorbar(Mat& inputImg, const vector<int>& inputAudio,
                                int samplingRate, const vector<vector<double>>& stft,
                                int xmin = 0, int xmax = 0)
    {
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

        if (xmax == 0)
        {
            xmax = static_cast<int>(inputAudio.size()) / samplingRate + 1;
        }
        vector<double> xList(xmarkup, 0);
        if (xmax >= xmarkup)
        {
            double deltax = (xmax - xmin) / (xmarkup - 1);
            for(int i = 0; i < xmarkup; ++i)
            {
                xList[i] = xmin + deltax * i;
            }
        }
        else
        {
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

        int ymin = 0;
        int ymax = static_cast<int>(samplingRate / 2);

        vector<double> yList;
        double deltay = (ymax - ymin) / (ymarkup - 1);
        for(int i = 0; i < ymarkup; ++i)
        {
            yList.push_back(ymin + deltay * i);
        }

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
        double deltaz = (zmax - zmin) / (zmarkup - 1);
        for(int i = 0; i < zmarkup; ++i)
        {
            zList.push_back(zmin + deltaz * i);
        }

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

        line(imgTotal, Point(preCol, totalRows - aftLine), Point(preCol + frameVectorCols, totalRows - aftLine),
                            gridColor, gridThickness);

        line(imgTotal, Point(preCol, preLine), Point(preCol, preLine + frameVectorRows),
                            gridColor, gridThickness);

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
        Mat resImage;
        resize(imgTotal, resImage, Size(cols, rows), INTER_AREA );
        return resImage;
    }

    Mat concatenateImages(Mat& img1, Mat& img2)
    {
        int totalRows = img1.rows + img2.rows;
        int totalCols = max(img1.cols , img2.cols);

        Mat imgTotal = Mat (totalRows, totalCols, CV_8UC3 , Scalar(255, 255, 255));

        img1.copyTo(imgTotal(Rect(0, 0, img1.cols, img1.rows)));
        img2.copyTo(imgTotal(Rect(0, img1.rows, img2.cols, img2.rows)));
        return imgTotal;
    }

    void dynamicFile(const string file)
    {
        VideoCapture cap;
        vector<int> params {    CAP_PROP_AUDIO_STREAM, audioStream,
                                CAP_PROP_VIDEO_STREAM, -1,
                                CAP_PROP_AUDIO_DATA_DEPTH, CV_16S   };

        cap.open(file, CAP_ANY, params);
        if (!cap.isOpened())
        {
            cerr << "Error : Can't read audio file: '" << audio << "' with audioStream = " << audioStream << endl;
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
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, xmin, xmax);
                        // imshow("Display amplitude graph", imgAmplitude);
                        waitKey(waitTime);
                        saveImage(imgAmplitude, "dynamic_amplitude_graph.png");
                    }
                    else if (graph == "spec")
                    {
                        stft = STFT(section);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft, xmin, xmax);
                        // imshow("Display spectrogram", imgSpec);
                        waitKey(waitTime);
                        saveImage(imgSpec, "dynamic_spectrogram.png");
                    }
                    else if (graph == "ampl_and_spec")
                    {
                        imgAmplitude = drawAmplitude(section);
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, xmin, xmax);
                        stft = STFT(section);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft, xmin, xmax);
                        imgTotal = concatenateImages(imgAmplitude, imgSpec);
                        // imshow("Display amplitude graph and spectrogram", imgTotal);
                        waitKey(waitTime);
                        saveImage(imgTotal, "dynamic_amplitude_and_spectrogram.png");
                    }
                }
            }
            else
            {
                break;
            }
        }
    }

    void dynamicMicrophone()
    {
        VideoCapture cap;
        vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                                CAP_PROP_VIDEO_STREAM, -1   };

        cap.open(0, CAP_MSMF, params);
        if (!cap.isOpened())
        {
            cerr << "Error: Can't open microphone" << endl;
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
        waitTime = updateTime * 1000;
        while ((sysTimeCurr - sysTimePrev) / cvTickFreq < microTime)
        {
            if (cap.grab())
            {
                cap.retrieve(frame, audioBaseIndex);
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
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, xmin, xmax);
                        // imshow("Display amplitude graph", imgAmplitude);
                        waitKey(waitTime);
                        saveImage(imgAmplitude, "dynamic_microphone_amplitude_graph.png");
                    }
                    else if (graph == "spec")
                    {
                        stft = STFT(section);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft, xmin, xmax);
                        // imshow("Display spectrogram", imgSpec);
                        waitKey(waitTime);
                        saveImage(imgSpec, "dynamic_microphone_spectrogram.png");
                    }
                    else if (graph == "ampl_and_spec")
                    {
                        imgAmplitude = drawAmplitude(section);
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate, xmin, xmax);
                        stft = STFT(section);
                        imgSpec = drawSpectrogram(stft);
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft, xmin, xmax);
                        imgTotal = concatenateImages(imgAmplitude, imgSpec);
                        // imshow("Display amplitude graph and spectrogram", imgTotal);
                        waitKey(waitTime);
                        saveImage(imgTotal, "dynamic_microphone_amplitude_and_spectrogram.png");
                    }
                }
            }
            else
            {
                cerr << "Error: Grab error" << endl;
                break;
            }
        }
    }

    bool initAndCheckArgs(const CommandLineParser& parser)
    {
        inputType = parser.get<string>("inputType");
        if ((inputType != "file") && (inputType != "microphone"))
        {
            cout << "Error: " << inputType << " input method doesnt exist" << endl;
            return false;
        }

        draw = parser.get<string>("draw");
        if ((draw != "static") && (draw != "dynamic"))
        {
            cout << "Error: " << draw << " draw type doesnt exist" << endl;
            return false;
        }

        graph = parser.get<string>("graph");
        if ((graph != "ampl") && (graph != "spec") && (graph != "ampl_and_spec"))
        {
            cout << "Error: " << graph << " type of graph doesnt exist" << endl;
            return false;
        }

        audio = samples::findFile(parser.get<std::string>("audio"));

        audioStream = parser.get<int>("audioStream");
        if (audioStream < 0)
        {
            cout << "Error: audioStream = " << audioStream << " - incorrect value. Must be >= 0" << endl;
            return false;
        }
        windowType = parser.get<string>("windowType");
        if ((windowType != "Rect") && (windowType != "Hann") && (windowType != "Hamming"))
        {
            cout << "Error: " << windowType << " type of window doesnt exist" << endl;
            return false;
        }

        windLen = parser.get<int>("windLen");
        if (windLen <= 0)
        {
            cout << "Error: windLen = " << windLen << " - incorrect value. Must be > 0" << endl;
            return false;
        }

        overlap = parser.get<int>("overlap");
        if (overlap <= 0)
        {
            cout << "Error: overlap = " << overlap << " - incorrect value. Must be > 0" << endl;
            return false;
        }

        enableGrid = parser.get<bool>("enableGrid");

        rows = parser.get<int>("rows");
        if (rows <= 0)
        {
            cout << "Error: rows = " << rows << " - incorrect value. Must be > 0" << endl;
            return false;
        }
        cols = parser.get<int>("cols");

        if (cols <= 0)
        {
            cout << "Error: cols = " << cols << " - incorrect value. Must be > 0" << endl;
            return false;
        }
        xmarkup = parser.get<int>("xmarkup");
        if (xmarkup < 2)
        {
            cout << "Error: xmarkup = " << xmarkup << " - incorrect value. Must be >= 2" << endl;
            return false;
        }
        ymarkup = parser.get<int>("ymarkup");
        if (ymarkup < 2)
        {
            cout << "Error: ymarkup = " << ymarkup << " - incorrect value. Must be >= 2" << endl;
            return false;
        }
        zmarkup = parser.get<int>("zmarkup");
        if (zmarkup < 2)
        {
            cout << "Error: zmarkup = " << zmarkup << " - incorrect value. Must be >= 2" << endl;
            return false;
        }
        microTime = parser.get<int>("microTime");
        if (microTime <= 0)
        {
            cout << "Error: microTime = " << microTime << " - incorrect value. Must be > 0" << endl;
            return false;
        }
        frameSizeTime = parser.get<int>("frameSizeTime");
        if (frameSizeTime <= 0)
        {
            cout << "Error: frameSizeTime = " << frameSizeTime << " - incorrect value. Must be > 0" << endl;
            return false;
        }
        updateTime = parser.get<int>("updateTime");
        if (updateTime <= 0)
        {
            cout << "Error: updateTime = " << updateTime << " - incorrect value. Must be > 0" << endl;
            return false;
        }
        waitTime = parser.get<int>("waitTime");
        if (waitTime < 0)
        {
            cout << "Error: waitTime = " << waitTime << " - incorrect value. Must be >= 0" << endl;
            return false;
        }
        return true;
    }

private :
    string inputType;
    string draw;
    string graph;
    string audio;
    int audioStream;

    string windowType;
    int windLen;
    int overlap;

    bool enableGrid;

    int rows;
    int cols;

    int xmarkup;
    int ymarkup;
    int zmarkup;

    int microTime;
    int frameSizeTime;
    int updateTime;
    int waitTime;

    void saveImage(const Mat& image, const string& filename) {
        string outputDir = "audio_spectrogram";
        system(("mkdir -p " + outputDir).c_str());
        string outputPath = outputDir + "/" + filename;
        imwrite(outputPath, image);
        cout << "Image saved at: " << outputPath << endl;
    }
};

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ? |               | this sample draws a volume graph and/or spectrogram of audio/video files and microphone \n\t\tDefault usage: ./Spectrogram.exe}"
        "{inputType i    | file          | file or microphone                       }"
        "{draw d         | static        | type of drawing: \n\t\t\tstatic - for plotting graph(s) across the entire input audio \n\t\t\tdynamic - for plotting graph(s) in a time-updating window}"
        "{graph g        | ampl_and_spec | type of graph: amplitude graph or/and spectrogram. Please use tags below : \n\t\t\tampl - draw the amplitude graph \n\t\t\tspec - draw the spectrogram\n\t\t\tampl_and_spec - draw the amplitude graph and spectrogram on one image under each other}"
        "{audio a        | Megamind.avi  | name and path to file                    }"
        "{audioStream s  | 1             | CAP_PROP_AUDIO_STREAM value. Select audio stream number }"
        "{windowType t   | Rect          | type of window for STFT. Please use tags below : \n\t\t\tRect/Hann/Hamming }"
        "{windLen l      | 256           | size of window for STFT                  }"
        "{overlap o      | 128           | overlap of windows for STFT              }"

        "{enableGrid     | false         | grid on the amplitude graph              }"

        "{rows r         | 400           | rows of output image                     }"
        "{cols c         | 900           | cols of output image                     }"

        "{xmarkup x      | 5             | number of x axis divisions (time asix)   }"
        "{ymarkup y      | 5             | number of y axis divisions (frequency or/and amplitude axis) }"
        "{zmarkup z      | 5             | number of z axis divisions (colorbar)    }"

        "{microTime m    | 20            | time of recording audio with microphone in seconds }"
        "{frameSizeTime f| 5             | size of sliding window in seconds        }"
        "{updateTime u   | 1             | update time of sliding window in seconds }"
        "{waitTime w     | 10            | parameter to cv.waitKey() for dynamic update of file input, takes values in milliseconds }"
        ;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    AudioDrawing draw(parser);
    return 0;
}

