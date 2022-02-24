import numpy as np
import cv2 as cv
import argparse
import os

'''
 You can download the converted onnx model from https://drive.google.com/drive/folders/1wLtxyao4ItAg8tt4Sb63zt6qXzhcQoR6?usp=sharing
 or convert the model yourself.

 You can get the original pre-trained Jasper model from NVIDIA : https://ngc.nvidia.com/catalog/models/nvidia:jasper_pyt_onnx_fp16_amp/files
    Download and unzip : `$ wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/jasper_pyt_onnx_fp16_amp/versions/20.10.0/zip -O jasper_pyt_onnx_fp16_amp_20.10.0.zip && unzip -o ./jasper_pyt_onnx_fp16_amp_20.10.0.zip && unzip -o ./jasper_pyt_onnx_fp16_amp.zip`

 you can get the script to convert the model here : https://gist.github.com/spazewalker/507f1529e19aea7e8417f6e935851a01

 You can convert the model using the following steps:
     1. Import onnx and load the original model
        ```
        import onnx
        model = onnx.load("./jasper-onnx/1/model.onnx")
        ```

     3. Change data type of input layer
        ```
        inp = model.graph.input[0]
        model.graph.input.remove(inp)
        inp.type.tensor_type.elem_type = 1
        model.graph.input.insert(0,inp)
        ```

     4. Change the data type of output layer
        ```
        out = model.graph.output[0]
        model.graph.output.remove(out)
        out.type.tensor_type.elem_type = 1
        model.graph.output.insert(0,out)
        ```

     5. Change the data type of every initializer and cast it's values from FP16 to FP32
        ```
        for i,init in enumerate(model.graph.initializer):
            model.graph.initializer.remove(init)
            init.data_type = 1
            init.raw_data = np.frombuffer(init.raw_data, count=np.product(init.dims), dtype=np.float16).astype(np.float32).tobytes()
            model.graph.initializer.insert(i,init)
        ```

     6. Add an additional reshape node to handle the inconsistant input from python and c++ of openCV.
        see https://github.com/opencv/opencv/issues/19091
        Make & insert a new node with 'Reshape' operation & required initializer
        ```
            tensor = numpy_helper.from_array(np.array([0,64,-1]),name='shape_reshape')
            model.graph.initializer.insert(0,tensor)
            node = onnx.helper.make_node(op_type='Reshape',inputs=['input__0','shape_reshape'], outputs=['input_reshaped'], name='reshape__0')
            model.graph.node.insert(0,node)
            model.graph.node[1].input[0] = 'input_reshaped'
        ```

     7. Finally save the model
        ```
        with open('jasper_dynamic_input_float.onnx','wb') as f:
            onnx.save_model(model,f)
        ```

    Original Repo : https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
 '''

class FilterbankFeatures:
    def __init__(self,
                 sample_rate=16000, window_size=0.02, window_stride=0.01,
                 n_fft=512, preemph=0.97, n_filt=64, lowfreq=0,
                 highfreq=None, log=True, dither=1e-5):
        '''
            Initializes pre-processing class. Default values are the values used by the Jasper
            architecture for pre-processing. For more details, refer to the paper here:
            https://arxiv.org/abs/1904.03288
        '''
        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride) # stride
        self.n_fft = n_fft or 2 ** np.ceil(np.log2(self.win_length))
        self.log = log
        self.dither = dither
        self.n_filt = n_filt
        self.preemph = preemph
        highfreq = highfreq or sample_rate / 2
        self.window_tensor = np.hanning(self.win_length)

        self.filterbanks = self.mel(sample_rate, self.n_fft, n_mels=n_filt, fmin=lowfreq, fmax=highfreq)
        self.filterbanks.dtype=np.float32
        self.filterbanks = np.expand_dims(self.filterbanks,0)

    def normalize_batch(self, x, seq_len):
        '''
            Normalizes the features.
        '''
        x_mean = np.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype)
        x_std = np.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype)
        for i in range(x.shape[0]):
            x_mean[i, :] = np.mean(x[i, :, :seq_len[i]],axis=1)
            x_std[i, :] = np.std(x[i, :, :seq_len[i]],axis=1)
        # make sure x_std is not zero
        x_std += 1e-10
        return (x - np.expand_dims(x_mean,2)) / np.expand_dims(x_std,2)

    def calculate_features(self, x, seq_len):
        '''
            Calculates filterbank features.
            args:
                x : mono channel audio
                seq_len : length of the audio sample
            returns:
                x : filterbank features
        '''
        dtype = x.dtype

        seq_len = np.ceil(seq_len / self.hop_length)
        seq_len = np.array(seq_len,dtype=np.int32)

        # dither
        if self.dither > 0:
            x += self.dither * np.random.randn(*x.shape)

        # do preemphasis
        if self.preemph is not None:
            x = np.concatenate(
                (np.expand_dims(x[0],-1), x[1:] - self.preemph * x[:-1]), axis=0)

        # Short Time Fourier Transform
        x  = self.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                  win_length=self.win_length,
                  fft_window=self.window_tensor)

        # get power spectrum
        x = (x**2).sum(-1)

        # dot with filterbank energies
        x = np.matmul(np.array(self.filterbanks,dtype=x.dtype), x)

        # log features if required
        if self.log:
            x = np.log(x + 1e-20)

        # normalize if required
        x = self.normalize_batch(x, seq_len).astype(dtype)
        return x

    # Mel Frequency calculation
    def hz_to_mel(self, frequencies):
        '''
            Converts frequencies from hz to mel scale. Input can be a number or a vector.
        '''
        frequencies = np.asanyarray(frequencies)

        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (frequencies - f_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        if frequencies.ndim:
            # If we have array data, vectorize
            log_t = frequencies >= min_log_hz
            mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
        elif frequencies >= min_log_hz:
            # If we have scalar data, directly
            mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
        return mels

    def mel_to_hz(self, mels):
        '''
            Converts frequencies from mel to hz scale. Input can be a number or a vector.
        '''
        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        if mels.ndim:
            # If we have vector data, vectorize
            log_t = mels >= min_log_mel
            freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        elif mels >= min_log_mel:
            # If we have scalar data, check directly
            freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

        return freqs

    def mel_frequencies(self, n_mels=128, fmin=0.0, fmax=11025.0):
        '''
            Calculates n mel frequencies between 2 frequencies
            args:
                n_mels : number of bands
                fmin : min frequency
                fmax : max frequency
            returns:
                mels : vector of mel frequencies
        '''
        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = self.hz_to_mel(fmin)
        max_mel = self.hz_to_mel(fmax)

        mels = np.linspace(min_mel, max_mel, n_mels)

        return self.mel_to_hz(mels)

    def mel(self, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32):
        '''
            Generates mel filterbank
            args:
                sr : Sampling rate
                n_fft : number of FFT components
                n_mels : number of Mel bands to generate
                fmin : lowest frequency (in Hz)
                fmax : highest frequency (in Hz). sr/2.0 if None
                dtype : the data type of the output basis.
            returns:
                mels : Mel transform matrix
        '''
        # default Max freq = half of sampling rate
        if fmax is None:
            fmax = float(sr) / 2

        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = self.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Using Slaney-style mel which is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
        return weights

    # STFT preperation
    def pad_window_center(self, data, size, axis=-1, **kwargs):
        '''
            Centers the data and pads.
            args:
                data : Vector to be padded and centered
                size : Length to pad data
                axis : Axis along which to pad and center the data
                kwargs : arguments passed to np.pad
            return : centered and padded data
        '''
        kwargs.setdefault("mode", "constant")
        n = data.shape[axis]
        lpad = int((size - n) // 2)
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (lpad, int(size - n - lpad))
        if lpad < 0:
            raise Exception(
                ("Target size ({:d}) must be at least input size ({:d})").format(size, n)
            )
        return np.pad(data, lengths, **kwargs)

    def frame(self, x, frame_length, hop_length):
        '''
            Slices a data array into (overlapping) frames.
            args:
                x : array to frame
                frame_length : length of frame
                hop_length : Number of steps to advance between frames
            return : A framed view of `x`
        '''
        if x.shape[-1] < frame_length:
            raise Exception(
                "Input is too short (n={:d})"
                " for frame_length={:d}".format(x.shape[-1], frame_length)
            )
        x = np.asfortranarray(x)
        n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
        strides = np.asarray(x.strides)
        new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def dtype_r2c(self, d, default=np.complex64):
        '''
            Find the complex numpy dtype corresponding to a real dtype.
            args:
                d : The real-valued dtype to convert to complex.
                default : The default complex target type, if `d` does not match a known dtype
            return : The complex dtype
        '''
        mapping = {
            np.dtype(np.float32): np.complex64,
            np.dtype(np.float64): np.complex128,
        }
        dt = np.dtype(d)
        if dt.kind == "c":
            return dt
        return np.dtype(mapping.get(dt, default))

    def stft(self, y, n_fft, hop_length=None, win_length=None, fft_window=None, pad_mode='reflect', return_complex=False):
        '''
            Short Time Fourier Transform. The STFT represents a signal in the time-frequency
            domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
            args:
                y : input signal
                n_fft : length of the windowed signal after padding with zeros.
                hop_length : number of audio samples between adjacent STFT columns.
                win_length : Each frame of audio is windowed by window of length win_length and
                    then padded with zeros to match n_fft
                fft_window : a vector or array of length `n_fft` having values computed by a
                    window function
                pad_mode : mode while padding the singnal
                return_complex : returns array with complex data type if `True`
            return : Matrix of short-term Fourier transform coefficients.
        '''
        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = int(win_length // 4)
        if y.ndim!=1:
            raise Exception(f'Invalid input shape. Only Mono Channeled audio supported. Input must have shape (Audio,). Got {y.shape}')

        # Pad the window out to n_fft size
        fft_window = self.pad_window_center(fft_window, n_fft)

        # Reshape so that the window can be broadcast
        fft_window = fft_window.reshape((-1, 1))

        # Pad the time series so that frames are centered
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

        # Window the time series.
        y_frames = self.frame(y, frame_length=n_fft, hop_length=hop_length)

        # Convert data type to complex
        dtype = self.dtype_r2c(y.dtype)

        # Pre-allocate the STFT matrix
        stft_matrix = np.empty( (int(1 + n_fft // 2), y_frames.shape[-1]), dtype=dtype, order="F")

        stft_matrix = np.fft.rfft( fft_window * y_frames, axis=0)
        return stft_matrix if return_complex==True else np.stack((stft_matrix.real,stft_matrix.imag),axis=-1)

class Decoder:
    '''
        Used for decoding the output of jasper model.
    '''
    def __init__(self):
        labels=[' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',"'"]
        self.labels_map = {i: label for i,label in enumerate(labels)}
        self.blank_id = 28

    def decode(self,x):
        """
            Takes output of Jasper model and performs ctc decoding algorithm to
            remove duplicates and special symbol. Returns prediction
        """
        x = np.argmax(x,axis=-1)
        hypotheses = []
        prediction = x.tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = self.blank_id
        for p in prediction:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([self.labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
        return hypotheses

def predict(features, net, decoder):
    '''
        Passes the features through the Jasper model and decodes the output to english transcripts.
        args:
            features : input features, calculated using FilterbankFeatures class
            net : Jasper model dnn.net object
            decoder : Decoder object
        return : Predicted text
    '''
    # make prediction
    net.setInput(features)
    output = net.forward()

    # decode output to transcript
    prediction = decoder.decode(output.squeeze(0))
    return prediction[0]

def readAudioFile(file, audioStream):
    cap = cv.VideoCapture(file)
    samplingRate = 16000
    params = np.asarray([cv.CAP_PROP_AUDIO_STREAM, audioStream,
              cv.CAP_PROP_VIDEO_STREAM, -1,
              cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_32F,
              cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, samplingRate
              ])
    cap.open(file, cv.CAP_ANY, params)
    if cap.isOpened() is False:
        print("Error : Can't read audio file:", file, "with audioStream = ", audioStream)
        return
    audioBaseIndex = int (cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    inputAudio = []
    while(1):
        if (cap.grab()):
            frame = np.asarray([])
            frame = cap.retrieve(frame, audioBaseIndex)
            for i in range(len(frame[1][0])):
                inputAudio.append(frame[1][0][i])
        else:
            break
    inputAudio = np.asarray(inputAudio, dtype=np.float64)
    return inputAudio, samplingRate

def readAudioMicrophone(microTime):
    cap = cv.VideoCapture()
    samplingRate = 16000
    params = np.asarray([cv.CAP_PROP_AUDIO_STREAM, 0,
              cv.CAP_PROP_VIDEO_STREAM, -1,
              cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_32F,
              cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, samplingRate
              ])
    cap.open(0, cv.CAP_ANY, params)
    if cap.isOpened() is False:
        print("Error: Can't open microphone")
        print("Error: problems with audio reading, check input arguments")
        return
    audioBaseIndex = int(cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    cvTickFreq = cv.getTickFrequency()
    sysTimeCurr = cv.getTickCount()
    sysTimePrev = sysTimeCurr
    inputAudio = []
    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < microTime):
        if (cap.grab()):
            frame = np.asarray([])
            frame = cap.retrieve(frame, audioBaseIndex)
            for i in range(len(frame[1][0])):
                inputAudio.append(frame[1][0][i])
            sysTimeCurr = cv.getTickCount()
        else:
            print("Error: Grab error")
            break
    inputAudio = np.asarray(inputAudio, dtype=np.float64)
    print("Number of samples: ", len(inputAudio))
    return inputAudio, samplingRate

if __name__ == '__main__':

    # Computation backends supported by layers
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
    # Target Devices for computation
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16)

    parser = argparse.ArgumentParser(description='This script runs Jasper Speech recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_type', type=str, required=True, help='file or microphone')
    parser.add_argument('--micro_time', type=int, default=15, help='Duration of microphone work in seconds. Must be more than 6 sec')
    parser.add_argument('--input_audio', type=str, help='Path to input audio file. OR Path to a txt file with relative path to multiple audio files in different lines')
    parser.add_argument('--audio_stream', type=int, default=0, help='CAP_PROP_AUDIO_STREAM value')
    parser.add_argument('--show_spectrogram', action='store_true', help='Whether to show a spectrogram of the input audio.')
    parser.add_argument('--model', type=str, default='jasper.onnx', help='Path to the onnx file of Jasper. default="jasper.onnx"')
    parser.add_argument('--output', type=str, help='Path to file where recognized audio transcript must be saved. Leave this to print on console.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help='Select a computation backend: '
                        "%d: automatically (by default) "
                        "%d: OpenVINO Inference Engine "
                        "%d: OpenCV Implementation " % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Select a target device: '
                        "%d: CPU target (by default) "
                        "%d: OpenCL "
                        "%d: OpenCL FP16 " % targets)

    args, _ = parser.parse_known_args()

    if args.input_audio and not os.path.isfile(args.input_audio):
        raise OSError("Input audio file does not exist")
    if not os.path.isfile(args.model):
        raise OSError("Jasper model file does not exist")

    features = []
    if args.input_type == "file":
        if args.input_audio.endswith('.txt'):
            with open(args.input_audio) as f:
                content = f.readlines()
                content = [x.strip() for x in content]
                audio_file_paths = content
            for audio_file_path in audio_file_paths:
                if not os.path.isfile(audio_file_path):
                    raise OSError("Audio file({audio_file_path}) does not exist")
        else:
            audio_file_paths = [args.input_audio]
        audio_file_paths = [os.path.abspath(x) for x in audio_file_paths]

        # Read audio Files
        for audio_file_path in audio_file_paths:
            audio = readAudioFile(audio_file_path, args.audio_stream)
            if audio is None:
                raise Exception(f"Can't read {args.input_audio}. Try a different format")
            features.append(audio[0])
    elif args.input_type == "microphone":
        # Read audio from microphone
        audio = readAudioMicrophone(args.micro_time)
        if audio is None:
            raise Exception(f"Can't open microphone. Try a different format")
        features.append(audio[0])
    else:
        raise Exception(f"input_type {args.input_type} doesn't exist. Please enter 'file' or 'microphone'")

    # Get Filterbank Features
    feature_extractor = FilterbankFeatures()
    for i in range(len(features)):
        X = features[i]
        seq_len = np.array([X.shape[0]], dtype=np.int32)
        features[i] = feature_extractor.calculate_features(x=X, seq_len=seq_len)

    # Load Network
    net = cv.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)

    # Show spectogram if required
    if args.show_spectrogram and not args.input_audio.endswith('.txt'):
        img = cv.normalize(src=features[0][0], dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        img = cv.applyColorMap(img, cv.COLORMAP_JET)
        cv.imshow('spectogram', img)
        cv.waitKey(0)

    # Initialize decoder
    decoder = Decoder()

    # Make prediction
    prediction = []
    print("Predicting...")
    for feature in features:
        print(f"\rAudio file {len(prediction)+1}/{len(features)}", end='')
        prediction.append(predict(feature, net, decoder))
    print("")

    # save transcript if required
    if args.output:
        with open(args.output,'w') as f:
            for pred in prediction:
                f.write(pred+'\n')
        print("Transcript was written to {}".format(args.output))
    else:
        print(prediction)
    cv.destroyAllWindows()
