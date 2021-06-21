import numpy as np
import cv2 as cv
import argparse
import os
import soundfile as sf # Temporary import to load audio files

class FilterbankFeatures():
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
        x = self.normalize_batch(x, seq_len)

        # mask to zero any values beyond seq_len in batch,
        # pad to multiple of `pad_align` (for efficiency)
        max_len = x.shape[-1]
        mask = np.arange(max_len, dtype=seq_len.dtype)
        mask = np.tile(mask,(x.shape[0],1))
        mask = mask >= np.expand_dims(seq_len,1)
        x = np.ma.array(x,mask=np.tile(mask,(1,x.shape[1],1)), fill_value=0).filled()
        x.dtype=dtype
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
            # If we have scalar data, heck directly
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

class Decoder():
    '''
        Used for decoding the output of jasper model.
    '''
    def __init__(self):
        labels=[' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',"'"]
        self.labels_map = {i: labels[i] for i in range(28)}
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

if __name__ == '__main__':

    # Computation backends supported by layers
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_OPENCV)
    # Target Devices for computation
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL)

    parser = argparse.ArgumentParser(description='This script runs Jasper Speech recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_audio', type=str, help='Path to input audio file.')
    parser.add_argument('--show_spectrogram', action='store_true', help='Whether to show a spectrogram of the input audio.')
    parser.add_argument('--model', type=str, default='jasper.onnx', help='Path to the onnx file of Jasper. default="jasper.onnx"')
    parser.add_argument('--output', type=str, help='Path to file where recognized audio transcript must be saved. Leave this to print on console.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help='Select a computation backend: '
                        "%d: automatically (by default) "
                        "%d: OpenCV Implementation" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Select a target device: '
                        "%d: CPU target (by default)"
                        "%d: OpenCL" % targets)
    
    args, _ = parser.parse_known_args()

    if args.input_audio and not os.path.isfile(args.input_audio):
        raise OSError("Input audio file does not exist")
    if not os.path.isfile(args.model):
        raise OSError("Jasper model file does not exist")

    # Read audio File
    try:
        audio = sf.read(args.input_audio)
        X=audio[0]
        seq_len=np.array([audio[1]],dtype=np.int32)
    except:
        raise Exception(f"Soundfile cannot read {args.input_audio}. Try a different format")

    # Load Network
    net = cv.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    
    # Get Filterbank Features
    feature_extractor=FilterbankFeatures()
    features = feature_extractor.calculate_features(x=X,seq_len=seq_len)

    # Show spectogram if required
    if args.show_spectrogram:
        # img = plt.imshow((features[0]), origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
        # plt.show()
        # code below doesn't show correct plot
        img = cv.applyColorMap((features[0]).astype(np.uint8),cv.COLORMAP_INFERNO)
        cv.imshow('spectogram',img)
        cv.waitKey(0)

    # Initialize decoder
    decoder = Decoder()

    # make prediction
    net.setInput(features)
    output = net.forward()

    # decode output to transcript
    prediction = decoder.decode(output[0])
    
    # save transcript if required
    if args.output:
        with open(args.output,'w') as f:
            f.write(prediction)
        print("Done")
    else:
        print(prediction)
    cv.destroyAllWindows() 
