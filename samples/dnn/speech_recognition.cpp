#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <random>
#include <numeric>
using namespace cv;
using namespace std;

class FilterbankFeatures {

//     Initializes pre-processing class. Default values are the values used by the Jasper
//     architecture for pre-processing. For more details, refer to the paper here:
//     https://arxiv.org/abs/1904.03288

private:
    int sample_rate = 16000;
    double window_size = 0.02;
    double window_stride = 0.01;
    int win_length = static_cast<int>(sample_rate * window_size); // Number of samples in window
    int hop_length = static_cast<int>(sample_rate * window_stride); // Number of steps to advance between frames
    int n_fft = 512; // Size of window for STFT

    // Parameters for filterbanks calculation
    int n_filt = 64;
    double lowfreq = 0.;
    double highfreq = sample_rate / 2;

public:
    // Mel filterbanks preparation
    double hz_to_mel(double frequencies)
    {
        //Converts frequencies from hz to mel scale
        // Fill in the linear scale
        double f_min = 0.0;
        double f_sp = 200.0 / 3;
        double mels = (frequencies - f_min) / f_sp;
        // Fill in the log-scale part
        double min_log_hz = 1000.0;  // beginning of log region (Hz)
        double min_log_mel = (min_log_hz - f_min) / f_sp;  // same (Mels)
        double logstep = std::log(6.4) / 27.0;  // step size for log region

        if (frequencies >= min_log_hz)
        {
            mels = min_log_mel + std::log(frequencies / min_log_hz) / logstep;
        }
        return mels;
    }

    vector<double> mel_to_hz(vector<double>& mels)
    {
        // Converts frequencies from mel to hz scale

        // Fill in the linear scale
        double f_min = 0.0;
        double f_sp = 200.0 / 3;
        vector<double> freqs;
        for (size_t i = 0; i < mels.size(); i++)
        {
            freqs.push_back(f_min + f_sp * mels[i]);
        }

        // And now the nonlinear scale
        double min_log_hz = 1000.0;  // beginning of log region (Hz)
        double min_log_mel = (min_log_hz - f_min) / f_sp;  // same (Mels)
        double logstep = std::log(6.4) / 27.0;  // step size for log region

        for(size_t i = 0; i < mels.size(); i++)
        {
            if (mels[i] >= min_log_mel)
            {
                freqs[i] = min_log_hz * exp(logstep * (mels[i] - min_log_mel));
            }
        }
        return freqs;
    }

    vector<double> mel_frequencies(int n_mels, double fmin, double fmax)
    {
        // Calculates n mel frequencies between 2 frequencies
        double min_mel = hz_to_mel(fmin);
        double max_mel = hz_to_mel(fmax);

        vector<double> mels;
        double step = (max_mel - min_mel) / (n_mels - 1);
        for(double i = min_mel; i < max_mel; i += step)
        {
            mels.push_back(i);
        }
        mels.push_back(max_mel);

        vector<double> res = mel_to_hz(mels);
        return res;
    }

    vector<vector<double>> mel(int n_mels, double fmin, double fmax)
    {
        //  Generates mel filterbank matrix

        double num = 1 + n_fft / 2;
        vector<vector<double>> weights(n_mels, vector<double>(static_cast<int>(num), 0.));

        // Center freqs of each FFT bin
        vector<double> fftfreqs;
        double step = (sample_rate / 2) / (num - 1);
        for(double i = 0; i <= sample_rate / 2; i += step)
        {
            fftfreqs.push_back(i);
        }
        // 'Center freqs' of mel bands - uniformly spaced between limits
        vector<double> mel_f = mel_frequencies(n_mels + 2, fmin, fmax);

        vector<double> fdiff;
        for(size_t i = 1; i < mel_f.size(); ++i)
        {
            fdiff.push_back(mel_f[i]- mel_f[i - 1]);
        }

        vector<vector<double>> ramps(mel_f.size(), vector<double>(fftfreqs.size()));
        for (size_t i = 0; i < mel_f.size(); ++i)
        {
            for (size_t j = 0; j < fftfreqs.size(); ++j)
            {
                ramps[i][j] = mel_f[i] - fftfreqs[j];
            }
        }

        double lower, upper, enorm;
        for (int i = 0; i < n_mels; ++i)
        {
            // using Slaney-style mel which is scaled to be approx constant energy per channel
            enorm = 2./(mel_f[i + 2] - mel_f[i]);

            for (int j = 0; j < static_cast<int>(num); ++j)
            {
                // lower and upper slopes for all bins
                lower = (-1) * ramps[i][j] / fdiff[i];
                upper = ramps[i + 2][j] / fdiff[i + 1];

                weights[i][j] = max(0., min(lower, upper)) * enorm;
            }
        }
        return weights;
    }

    // STFT preparation
    vector<double> pad_window_center(vector<double>&data, int size)
    {
        // Pad the window out to n_fft size
        int n = static_cast<int>(data.size());
        int lpad = static_cast<int>((size - n) / 2);
        vector<double> pad_array;

        for(int i = 0; i < lpad; ++i)
        {
            pad_array.push_back(0.);
        }

        for(size_t i = 0; i < data.size(); ++i)
        {
            pad_array.push_back(data[i]);
        }

        for(int i = 0; i < lpad; ++i)
        {
            pad_array.push_back(0.);
        }
        return pad_array;
    }

    vector<vector<double>> frame(vector<double>& x)
    {
        // Slices a data array into overlapping frames.
        int n_frames = static_cast<int>(1 + (x.size() - n_fft) / hop_length);
        vector<vector<double>> new_x(n_fft, vector<double>(n_frames));

        for (int i = 0; i < n_fft; ++i)
        {
            for (int j = 0; j < n_frames; ++j)
            {
                new_x[i][j] = x[i + j * hop_length];
            }
        }
        return new_x;
    }

    vector<double> hanning()
    {
        // https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
        vector<double> window_tensor;
        for (int j = 1 - win_length; j < win_length; j+=2)
        {
            window_tensor.push_back(1 - (0.5 * (1 - cos(CV_PI * j / (win_length - 1)))));
        }
        return window_tensor;
    }

    vector<vector<double>> stft_power(vector<double>& y)
    {
        // Short Time Fourier Transform. The STFT represents a signal in the time-frequency
        // domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
        // https://en.wikipedia.org/wiki/Short-time_Fourier_transform

        // Pad the time series so that frames are centered
        vector<double> new_y;
        int num = int(n_fft / 2);

        for (int i = 0; i < num; ++i)
        {
            new_y.push_back(y[num - i]);
        }
        for (size_t i = 0; i < y.size(); ++i)
        {
            new_y.push_back(y[i]);
        }
        for (size_t i = y.size() - 2; i >= y.size() - num - 1; --i)
        {
            new_y.push_back(y[i]);
        }

        // Compute a window function
        vector<double> window_tensor = hanning();

        // Pad the window out to n_fft size
        vector<double> fft_window = pad_window_center(window_tensor, n_fft);

        // Window the time series
        vector<vector<double>> y_frames = frame(new_y);

        // Multiply on fft_window
        for (size_t i = 0; i < y_frames.size(); ++i)
        {
            for (size_t j = 0; j < y_frames[0].size(); ++j)
            {
                y_frames[i][j] *= fft_window[i];
            }
        }

        // Transpose frames for computing stft
        vector<vector<double>> y_frames_transpose(y_frames[0].size(), vector<double>(y_frames.size()));
        for (size_t i = 0; i < y_frames[0].size(); ++i)
        {
            for (size_t j = 0; j < y_frames.size(); ++j)
            {
                y_frames_transpose[i][j] = y_frames[j][i];
            }
        }

        // Short Time Fourier Transform
        // and get power of spectrum
        vector<vector<double>> spectrum_power(y_frames_transpose[0].size() / 2 + 1 );
        for (size_t i = 0; i < y_frames_transpose.size(); ++i)
        {
            Mat dstMat;
            dft(y_frames_transpose[i], dstMat, DFT_COMPLEX_OUTPUT);

            // we need only the first part of the spectrum, the second part is symmetrical
            for (int j = 0; j < static_cast<int>(y_frames_transpose[0].size()) / 2 + 1; ++j)
            {
                double power_re = dstMat.at<double>(2 * j) * dstMat.at<double>(2 * j);
                double power_im = dstMat.at<double>(2 * j + 1) * dstMat.at<double>(2 * j + 1);
                spectrum_power[j].push_back(power_re + power_im);
            }
        }
        return spectrum_power;
    }

    Mat calculate_features(vector<double>& x)
    {
        // Calculates filterbank features matrix.

        // Do preemphasis
        std::default_random_engine generator;
        std::normal_distribution<double> normal_distr(0, 1);
        double dither = 1e-5;
        for(size_t i = 0; i < x.size(); ++i)
        {
            x[i] += dither * static_cast<double>(normal_distr(generator));
        }
        double preemph = 0.97;
        for (size_t i =  x.size() - 1; i > 0; --i)
        {
            x[i] -= preemph * x[i-1];
        }

        // Calculate Short Time Fourier Transform and get power of spectrum
        auto spectrum_power = stft_power(x);

        vector<vector<double>> filterbanks = mel(n_filt, lowfreq, highfreq);

        // Calculate log of multiplication of filterbanks matrix on spectrum_power matrix
        vector<vector<double>> x_stft(filterbanks.size(), vector<double>(spectrum_power[0].size(), 0));

        for (size_t i = 0; i < filterbanks.size(); ++i)
        {
            for (size_t j = 0; j < filterbanks[0].size(); ++j)
            {
                for (size_t k = 0; k < spectrum_power[0].size(); ++k)
                {
                    x_stft[i][k] += filterbanks[i][j] * spectrum_power[j][k];
                }
            }
            for (size_t k = 0; k < spectrum_power[0].size(); ++k)
            {
                x_stft[i][k] = std::log(x_stft[i][k] + 1e-20);
            }
        }

        // normalize data
        auto elments_num = x_stft[0].size();
        for(size_t i = 0; i < x_stft.size(); ++i)
        {
            double x_mean = std::accumulate(x_stft[i].begin(), x_stft[i].end(), 0.) / elments_num; // arithmetic mean
            double x_std = 0; // standard deviation
            for(size_t j = 0; j < elments_num; ++j)
            {
                double subtract = x_stft[i][j] - x_mean;
                x_std += subtract * subtract;
            }
            x_std /= elments_num;
            x_std = sqrt(x_std) + 1e-10; // make sure x_std is not zero

            for(size_t j = 0; j < elments_num; ++j)
            {
                x_stft[i][j] = (x_stft[i][j] - x_mean) / x_std; // standard score
            }
        }

        Mat calculate_features(static_cast<int>(x_stft.size()), static_cast<int>(x_stft[0].size()), CV_32F);
        for(int i = 0; i < calculate_features.size[0]; ++i)
        {
            for(int j = 0; j < calculate_features.size[1]; ++j)
            {
                calculate_features.at<float>(i, j) = static_cast<float>(x_stft[i][j]);
            }
        }
        return calculate_features;
    }
};

class Decoder {
    // Used for decoding the output of jasper model
private:
    unordered_map<int, char> labels_map = fillMap();
    int blank_id = 28;

public:
    unordered_map<int, char> fillMap()
    {
        vector<char> labels={' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p'
                                ,'q','r','s','t','u','v','w','x','y','z','\''};
        unordered_map<int, char> map;
        for(int i = 0; i < static_cast<int>(labels.size()); ++i)
        {
            map[i] = labels[i];
        }
        return map;
    }

    string decode(Mat& x)
    {
        // Takes output of Jasper model and performs ctc decoding algorithm to
        // remove duplicates and special symbol. Returns prediction

        vector<int> prediction;
        for(int i = 0; i < x.size[1]; ++i)
        {
            double maxEl = -1e10;
            int ind = 0;
            for(int j = 0; j < x.size[2]; ++j)
            {
                if (maxEl <= x.at<float>(0, i, j))
                {
                    maxEl = x.at<float>(0, i, j);
                    ind = j;
                }
            }
            prediction.push_back(ind);
        }
        // CTC decoding procedure
        vector<double> decoded_prediction = {};
        int previous = blank_id;

        for(int i = 0; i < static_cast<int>(prediction.size()); ++i)
        {
            if (( prediction[i] != previous || previous == blank_id) &&  prediction[i] != blank_id)
            {
                decoded_prediction.push_back(prediction[i]);
            }
            previous = prediction[i];
        }

        string hypotheses = {};
        for(size_t i = 0; i < decoded_prediction.size(); ++i)
        {
            auto it = labels_map.find(static_cast<char>(decoded_prediction[i]));
            if (it != labels_map.end())
                hypotheses.push_back(it->second);
        }
        return hypotheses;
    }

};

static string predict(Mat& features, dnn::Net net, Decoder decoder)
{
    // Passes the features through the Jasper model and decodes the output to english transcripts.

    // expand 2d features matrix to 3d
    vector<int> sizes = {1, static_cast<int>(features.size[0]),
                              static_cast<int>(features.size[1])};
    features = features.reshape(0, sizes);

    // make prediction
    net.setInput(features);
    Mat output = net.forward();

    // decode output to transcript
    auto prediction = decoder.decode(output);
    return prediction;
}

static int readAudioFile(vector<double>& inputAudio, string file, int audioStream)
{
    VideoCapture cap;
    int samplingRate = 16000;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, audioStream,
                            CAP_PROP_VIDEO_STREAM, -1,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_32F,
                            CAP_PROP_AUDIO_SAMPLES_PER_SECOND, samplingRate
                            };
    cap.open(file, CAP_ANY, params);
    if (!cap.isOpened())
    {
        cerr << "Error : Can't read audio file: '" << file << "' with audioStream = " << audioStream << endl;
        return -1;
    }
    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    vector<double> frameVec;
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
            break;
        }
    }
    return samplingRate;
}

static int readAudioMicrophone(vector<double>& inputAudio, int microTime)
{
    VideoCapture cap;
    int samplingRate = 16000;
    vector<int> params {    CAP_PROP_AUDIO_STREAM, 0,
                            CAP_PROP_VIDEO_STREAM, -1,
                            CAP_PROP_AUDIO_DATA_DEPTH, CV_32F,
                            CAP_PROP_AUDIO_SAMPLES_PER_SECOND, samplingRate
                            };
    cap.open(0, CAP_ANY, params);
    if (!cap.isOpened())
    {
        cerr << "Error: Can't open microphone" << endl;
        return -1;
    }

    const int audioBaseIndex = (int)cap.get(CAP_PROP_AUDIO_BASE_INDEX);
    vector<double> frameVec;
    Mat frame;
    if (microTime <= 0)
    {
        cerr << "Error: Duration of audio chunk must be > 0" << endl;
        return -1;
    }
    size_t sizeOfData = static_cast<size_t>(microTime * samplingRate);
    while (inputAudio.size() < sizeOfData)
    {
        if (cap.grab())
        {
            cap.retrieve(frame, audioBaseIndex);
            frameVec = frame;
            inputAudio.insert(inputAudio.end(), frameVec.begin(), frameVec.end());
        }
        else
        {
            cerr << "Error: Grab error" << endl;
            break;
        }
    }
    return samplingRate;
}

int main(int argc, char** argv)
{
    const String keys =
        "{help h usage ?     |                          | This script runs Jasper Speech recognition model }"
        "{input_file i       |                          | Path to input audio file. If not specified, microphone input will be used }"
        "{audio_duration t   | 15                       | Duration of audio chunk to be captured from microphone }"
        "{audio_stream a     | 0                        | CAP_PROP_AUDIO_STREAM value     }"
        "{show_spectrogram s | false                    | Show a spectrogram of the input audio: true / false / 1 / 0 }"
        "{model m            | jasper.onnx              | Path to the onnx file of Jasper. You can download the converted onnx model "
                                                          "from https://drive.google.com/drive/folders/1wLtxyao4ItAg8tt4Sb63zt6qXzhcQoR6?usp=sharing}"
        "{backend b          | dnn::DNN_BACKEND_DEFAULT | Select a computation backend: "
                                                          "dnn::DNN_BACKEND_DEFAULT, "
                                                          "dnn::DNN_BACKEND_INFERENCE_ENGINE, "
                                                          "dnn::DNN_BACKEND_OPENCV }"
        "{target t           | dnn::DNN_TARGET_CPU      | Select a target device: "
                                                          "dnn::DNN_TARGET_CPU, "
                                                          "dnn::DNN_TARGET_OPENCL, "
                                                          "dnn::DNN_TARGET_OPENCL_FP16 }"
        ;
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Load Network
    dnn::Net net = dnn::readNetFromONNX(parser.get<std::string>("model"));
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));

    // Get audio
    vector<double>inputAudio = {};
    int samplingRate = 0;
    if (parser.has("input_file"))
    {
        string audio = samples::findFile(parser.get<std::string>("input_file"));
        samplingRate = readAudioFile(inputAudio, audio, parser.get<int>("audio_stream"));
    }
    else
    {
        samplingRate = readAudioMicrophone(inputAudio, parser.get<int>("audio_duration"));
    }

    if ((inputAudio.size() == 0) || samplingRate <= 0)
    {
        cerr << "Error: problems with audio reading, check input arguments" << endl;
        return -1;
    }

    if (inputAudio.size() / samplingRate < 6)
    {
        cout << "Warning: For predictable network performance duration of audio must exceed 6 sec."
                " Audio will be extended with zero samples" << endl;
        for(int i = static_cast<int>(inputAudio.size()) - 1; i < samplingRate * 6; ++i)
        {
            inputAudio.push_back(0);
        }
    }

    // Calculate features
    FilterbankFeatures filter;
    auto calculated_features = filter.calculate_features(inputAudio);

    // Show spectogram if required
    if (parser.get<bool>("show_spectrogram") == true)
    {
        Mat spectogram;
        normalize(calculated_features, spectogram, 0, 255, NORM_MINMAX, CV_8U);
        applyColorMap(spectogram, spectogram, COLORMAP_INFERNO);
        imshow("spectogram", spectogram);
        waitKey(0);
    }

    Decoder decoder;
    string prediction = predict(calculated_features, net, decoder);
    for( auto &transcript: prediction)
    {
        cout << transcript;
    }

    return 0;
}
