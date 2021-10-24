import cv2 as cv
import numpy as np
import math
import argparse


def readAudioFile(file, audioStream):
    cap = cv.VideoCapture(file)

    params = [cv.CAP_PROP_AUDIO_STREAM, audioStream,
              cv.CAP_PROP_VIDEO_STREAM, -1,
              cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_16S]
    params = np.asarray(params)

    cap.open(file, cv.CAP_MSMF, params)
    if cap.isOpened() == False:
        print("Cant read file")
        return
    audioBaseIndex = int(cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    numberOfChannels = int(cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))

    print("CAP_PROP_AUDIO_DATA_DEPTH: ", str((int(cap.get(cv.CAP_PROP_AUDIO_DATA_DEPTH)))))
    print("CAP_PROP_AUDIO_SAMPLES_PER_SECOND: ", cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    print("CAP_PROP_AUDIO_TOTAL_CHANNELS: ", numberOfChannels)
    print("CAP_PROP_AUDIO_TOTAL_STREAMS: ", cap.get(cv.CAP_PROP_AUDIO_TOTAL_STREAMS))

    frame = []
    frame = np.asarray(frame)
    inputAudio = []

    while (1):
        if (cap.grab()):
            frame = []
            frame = np.asarray(frame)
            frame = cap.retrieve(frame, audioBaseIndex)
            for i in range(len(frame[1][0])):
                inputAudio.append(frame[1][0][i])
        else:
            break

    inputAudio = np.asarray(inputAudio)
    samplingRate = int(cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    return samplingRate, inputAudio


def readAudioMicrophone():
    cap = cv.VideoCapture()

    params = [cv.CAP_PROP_AUDIO_STREAM, 0, cv.CAP_PROP_VIDEO_STREAM, -1]
    params = np.asarray(params)

    cap.open(0, cv.CAP_MSMF, params)
    audioBaseIndex = int(cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    numberOfChannels = int(cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))

    print("CAP_PROP_AUDIO_DATA_DEPTH: ", str((int(cap.get(cv.CAP_PROP_AUDIO_DATA_DEPTH)))))
    print("CAP_PROP_AUDIO_SAMPLES_PER_SECOND: ", cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    print("CAP_PROP_AUDIO_TOTAL_CHANNELS: ", numberOfChannels)
    print("CAP_PROP_AUDIO_TOTAL_STREAMS: ", cap.get(cv.CAP_PROP_AUDIO_TOTAL_STREAMS))

    cvTickFreq = cv.getTickFrequency()
    sysTimeCurr = cv.getTickCount()
    sysTimePrev = sysTimeCurr

    frame = []
    frame = np.asarray(frame)
    inputAudio = []

    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < 10):
        if (cap.grab()):
            for nCh in range(numberOfChannels):
                frame = []
                frame = np.asarray(frame)
                frame = cap.retrieve(frame, audioBaseIndex + nCh)
                for i in range(len(frame[1][0])):
                    inputAudio.append(frame[1][0][i])
                sysTimeCurr = cv.getTickCount()
        else:
            break

    inputAudio = np.asarray(inputAudio)
    samplingRate = int(cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))

    return samplingRate, inputAudio


def drawAmplitude(inputAudio):
    rows = 400
    cols = 900
    color = (247, 111, 87)
    thickness = 5
    frameVectorRows = 500
    middle = frameVectorRows // 2

    # usually the input data is too big, so it is necessary
    # to reduce size using interpolation of data
    frameVectorCols = 40000
    if len(inputAudio) < frameVectorCols:
        frameVectorCols = len(inputAudio)

    img = np.zeros((frameVectorRows, frameVectorCols, 3), np.uint8)
    img += 255  # white background

    audio = np.array(0)
    audio = cv.resize(inputAudio, (1, frameVectorCols), interpolation=cv.INTER_LINEAR)
    reshapeAudio = np.reshape(audio, (-1))

    # normalization data by maximum element
    minCv, maxCv, _, _ = cv.minMaxLoc(reshapeAudio)
    maxElem = int(max(abs(minCv), abs(maxCv)))
    # if all data values are zero (silence)
    if maxElem == 0:
        maxElem = 1
    for i in range(len(reshapeAudio)):
        if reshapeAudio[i] > 0:
            reshapeAudio[i] = middle - reshapeAudio[i] * middle // maxElem
        else:
            reshapeAudio[i] = frameVectorRows - middle - reshapeAudio[i] * middle // maxElem

    for i in range(1, frameVectorCols, 1):
        cv.line(img, (i - 1, int(reshapeAudio[i - 1])), (i, int(reshapeAudio[i])), color, thickness)

    img = cv.resize(img, (cols, rows), interpolation=cv.INTER_AREA)
    return img


def drawAmplitudeScale(inputImg, inputAudio, samplingRate,
                       grid="off",
                       rows=800, cols=900,
                       xmarkup=5, ymarkup=5,
                       xmin=0, xmax=None):
    # function of layout drawing for graph of volume amplitudes
    # x axis for time
    # y axis for amplitudes

    # parameters for the new image size
    preCol = 100
    aftCol = 100
    preLine = 40
    aftLine = 50

    frameVectorRows = inputImg.shape[0]
    frameVectorCols = inputImg.shape[1]

    totalRows = preLine + frameVectorRows + aftLine
    totalCols = preCol + frameVectorCols + aftCol

    imgTotal = np.zeros((totalRows, totalCols, 3), np.uint8)
    imgTotal += 255  # white background
    imgTotal[preLine: preLine + frameVectorRows, preCol: preCol + frameVectorCols] = inputImg

    # calculating values on x axis
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(inputAudio) / samplingRate

    if xmax > xmarkup:
        xList = np.linspace(xmin, xmax, xmarkup).astype(int)
    else:
        # this case is used to display a dynamic update
        tmp = np.arange(xmin, xmax, 1).astype(int) + 1
        xList = np.concatenate((np.zeros(xmarkup - len(tmp)), tmp[:]), axis=None)

    # calculating values on y axis
    ymin = np.min(inputAudio)
    ymax = np.max(inputAudio)
    yList = np.linspace(ymin, ymax, ymarkup)

    # parameters for layout drawing
    textThickness = 1
    gridThickness = 1
    gridColor = (0, 0, 0)
    textColor = (0, 0, 0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    # horizontal axis under the graph
    cv.line(imgTotal, (preCol, totalRows - aftLine),
            (preCol + frameVectorCols, totalRows - aftLine),
            gridColor, gridThickness)
    # vertical axis for amplitude
    cv.line(imgTotal, (preCol, preLine), (preCol, preLine + frameVectorRows),
            gridColor, gridThickness)

    # parameters for layout calculation
    serifSize = 10
    indentDownX = serifSize * 2
    indentDownY = serifSize // 2
    indentLeftX = serifSize
    indentLeftY = 2 * preCol // 3

    # drawing layout for x axis
    numX = frameVectorCols // (xmarkup - 1)
    for i in range(len(xList)):
        a1 = preCol + i * numX
        a2 = frameVectorRows + preLine
        b1 = a1
        b2 = a2 + serifSize
        if (grid == 'on'):
            d1 = a1
            d2 = preLine
            cv.line(imgTotal, (a1, a2), (d1, d2), gridColor, gridThickness)
        cv.line(imgTotal, (a1, a2), (b1, b2), gridColor, gridThickness)
        cv.putText(imgTotal, str(int(xList[i])), (b1 - indentLeftX, b2 + indentDownX),
                   font, fontScale, textColor, textThickness)

    # drawing layout for y axis
    numY = frameVectorRows // (ymarkup - 1)
    for i in range(len(yList)):
        a1 = preCol
        a2 = totalRows - aftLine - i * numY
        b1 = preCol - serifSize
        b2 = a2
        if (grid == 'on'):
            d1 = preCol + frameVectorCols
            d2 = a2
            cv.line(imgTotal, (a1, a2), (d1, d2), gridColor, gridThickness)
        cv.line(imgTotal, (a1, a2), (b1, b2), gridColor, gridThickness)
        cv.putText(imgTotal, str(int(yList[i])), (b1 - indentLeftY, b2 + indentDownY),
                   font, fontScale, textColor, textThickness)
    imgTotal = cv.resize(imgTotal, (cols, rows), interpolation=cv.INTER_AREA)
    return imgTotal


def STFT(inputAudio, windowType="Rect", windLen=256, overlap=128):
    time_step = windLen - overlap
    stft = []

    if windowType == "Hann":
        Hann_wind = []
        for i in range (1 - windLen, windLen, 2):
            Hann_wind.append(i*(0.5 + 0.5 * math.cos(math.pi * i / (windLen - 1))))
        Hann_wind = np.asarray(Hann_wind)

    elif windowType == "Hamming":
        Hamming_wind = []
        for i in range (1 - windLen, windLen, 2):
            Hamming_wind.append(i*(0.53836 - 0.46164 * (math.cos(2 * math.pi * i / (windLen - 1)))))
        Hamming_wind = np.asarray(Hamming_wind)

    for index in np.arange(0, len(inputAudio), time_step).astype(int):

        section = inputAudio[index:index + windLen]
        zeroArray = np.zeros(windLen - len(section))
        section = np.concatenate((section, zeroArray), axis=None)

        if windowType == "Hann":
            section *= Hann_wind
        elif windowType == "Hamming":
            section *= Hamming_wind

        dst = np.empty(0)
        dst = cv.dft(section, dst, flags=cv.DFT_COMPLEX_OUTPUT)
        reshape_dst = np.reshape(dst, (-1))
        # we need only the first part of the spectrum, the second part is symmetrical
        complexArr = np.zeros(len(dst) // 4, dtype=complex)
        for i in range(len(dst) // 4):
            complexArr[i] = complex(reshape_dst[2 * i], reshape_dst[2 * i + 1])
        stft.append(np.abs(complexArr))

    stft = np.array(stft).transpose()
    # convert elements to the decibel scale
    np.log10(stft, out=stft, where=(stft != 0.))
    return 10 * stft


def drawSpectrogram(stft):
    rows = 400
    cols = 900
    colorbar = cv.COLORMAP_INFERNO

    # Normalization of image values from 0 to 255 to get more contrast image
    # and this normalization will be taken into account in the scale drawing
    colormapImageRows = 255
    frameVectorRows = stft.shape[0]
    frameVectorCols = stft.shape[1]

    # maxStft = np.max(np.abs(stft))
    minCv, maxCv, _, _ = cv.minMaxLoc(stft)
    maxStft = int(max(abs(minCv), abs(maxCv)))
    # if maxStft still zero (silence)
    if maxStft == 0:
        maxStft = 1

    imgSpec = np.zeros((frameVectorRows, frameVectorCols, 3), np.uint8)

    for i in range(frameVectorRows):
        for j in range(frameVectorCols):
            imgSpec[frameVectorRows - i - 1, j] = int(stft[i][j]) * colormapImageRows // maxStft

    imgSpec = cv.applyColorMap(imgSpec, colorbar)
    imgSpec = cv.resize(imgSpec, (cols, rows), interpolation=cv.INTER_LINEAR)
    return imgSpec


def drawSpectrogramColorbar(inputImg, inputAudio, samplingRate, stft,
                            rows=800, cols=900,
                            xmarkup=5, ymarkup=5, zmarkup=5,
                            xmin=0, xmax=None):
    # function of layout drawing for the three-dimensional graph of the spectrogram
    # x axis for time
    # y axis for frequencies
    # z axis for magnitudes of frequencies shown by color scale

    # parameters for the new image size
    preCol = 100
    aftCol = 100
    preLine = 40
    aftLine = 50
    colColor = 20
    ind_col = 20

    frameVectorRows = inputImg.shape[0]
    frameVectorCols = inputImg.shape[1]

    totalRows = preLine + frameVectorRows + aftLine
    totalCols = preCol + frameVectorCols + aftCol + colColor

    imgTotal = np.zeros((totalRows, totalCols, 3), np.uint8)
    imgTotal += 255  # white background
    imgTotal[preLine: preLine + frameVectorRows, preCol: preCol + frameVectorCols] = inputImg

    # colorbar image due to drawSpectrogram(..) picture has been normalised from 255 to 0,
    # so here colorbar has values from 255 to 0
    colorArrSize = 256
    imgColorBar = np.zeros((colorArrSize, colColor, 1), np.uint8)

    for i in range(colorArrSize):
        imgColorBar[i] += colorArrSize - 1 - i

    imgColorBar = cv.applyColorMap(imgColorBar, cv.COLORMAP_INFERNO)
    imgColorBar = cv.resize(imgColorBar, (colColor, frameVectorRows), interpolation=cv.INTER_AREA)  #

    imgTotal[preLine: preLine + frameVectorRows,
    preCol + frameVectorCols + ind_col:
    preCol + frameVectorCols + ind_col + colColor] = imgColorBar

    # calculating values on x axis
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(inputAudio) / samplingRate
    if xmax > xmarkup:
        xList = np.linspace(xmin, xmax, xmarkup).astype(int)
    else:
        # this case is used to display a dynamic update
        tmpXList = np.arange(xmin, xmax, 1).astype(int) + 1
        xList = np.concatenate((np.zeros(xmarkup - len(tmpXList)), tmpXList[:]), axis=None)

    # calculating values on y axis
    # according to the Nyquist sampling theorem,
    # signal should posses frequencies equal to half of sampling rate
    ymin = 0
    ymax = int(samplingRate / 2.)
    yList = np.linspace(ymin, ymax, ymarkup).astype(int)

    # calculating values on z axis
    zList = np.linspace(np.min(stft), np.max(stft), zmarkup)

    # parameters for layout drawing
    textThickness = 1
    textColor = (0, 0, 0)
    gridThickness = 1
    gridColor = (0, 0, 0)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    serifSize = 10
    indentDownX = serifSize * 2
    indentDownY = serifSize // 2
    indentLeftX = serifSize
    indentLeftY = 2 * preCol // 3

    # horizontal axis
    cv.line(imgTotal, (preCol, totalRows - aftLine), (preCol + frameVectorCols, totalRows - aftLine),
            gridColor, gridThickness)
    # vertical axis
    cv.line(imgTotal, (preCol, preLine), (preCol, preLine + frameVectorRows),
            gridColor, gridThickness)

    # drawing layout for x axis
    numX = frameVectorCols // (xmarkup - 1)
    for i in range(len(xList)):
        a1 = preCol + i * numX
        a2 = frameVectorRows + preLine
        b1 = a1
        b2 = a2 + serifSize
        cv.line(imgTotal, (a1, a2), (b1, b2), gridColor, gridThickness)
        cv.putText(imgTotal, str(int(xList[i])), (b1 - indentLeftX, b2 + indentDownX),
                   font, fontScale, textColor, textThickness)

    # drawing layout for y axis
    numY = frameVectorRows // (ymarkup - 1)
    for i in range(len(yList)):
        a1 = preCol
        a2 = totalRows - aftLine - i * numY
        b1 = preCol - serifSize
        b2 = a2
        cv.line(imgTotal, (a1, a2), (b1, b2), gridColor, gridThickness)
        cv.putText(imgTotal, str(int(yList[i])), (b1 - indentLeftY, b2 + indentDownY),
                   font, fontScale, textColor, textThickness)

    # drawing layout for z axis
    numZ = frameVectorRows // (zmarkup - 1)
    for i in range(len(zList)):
        a1 = preCol + frameVectorCols + ind_col + colColor
        a2 = totalRows - aftLine - i * numZ
        b1 = a1 + serifSize
        b2 = a2
        cv.line(imgTotal, (a1, a2), (b1, b2), gridColor, gridThickness)
        cv.putText(imgTotal, str(int(zList[i])), (b1 + 10, b2 + indentDownY),
                   font, fontScale, textColor, textThickness)
    imgTotal = cv.resize(imgTotal, (cols, rows), interpolation=cv.INTER_AREA)
    return imgTotal


def concatenateImages(img1, img2, rows=800, cols=900):
    # first image will be under the second image
    totalRows = img1.shape[0] + img2.shape[0]
    totalCols = max(img1.shape[1], img2.shape[1])

    # if images columns do not match, the difference is filled in white
    imgTotal = np.zeros((totalRows, totalCols, 3), np.uint8)
    imgTotal += 255

    imgTotal[:img1.shape[0], :img1.shape[1]] = img1
    imgTotal[img2.shape[0]:, :img2.shape[1]] = img2

    imgTotal = cv.resize(imgTotal, (cols, rows))
    return imgTotal


def dynamicFile(file, audioStream, graph="ampl_and_spec",
                frameSizeTime=5, updateTime=1, waitTime=10,
                windowType="Rect", windLen=256, overlap=128,
                grid="off", rows=800, cols=900,
                xmarkup=5, ymarkup=5, zmarkup=5):
    cap = cv.VideoCapture(file)
    params = [cv.CAP_PROP_AUDIO_STREAM, audioStream,
              cv.CAP_PROP_VIDEO_STREAM, -1,
              cv.CAP_PROP_AUDIO_DATA_DEPTH, cv.CV_16S]
    params = np.asarray(params)

    cap.open(file, cv.CAP_MSMF, params)
    if cap.isOpened() == False:
        print("ERROR! Can't to open file")
        return

    audioBaseIndex = int(cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    numberOfChannels = int(cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))
    samplingRate = int(cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))

    print("CAP_PROP_AUDIO_DATA_DEPTH: ", str((int(cap.get(cv.CAP_PROP_AUDIO_DATA_DEPTH)))))
    print("CAP_PROP_AUDIO_SAMPLES_PER_SECOND: ", cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    print("CAP_PROP_AUDIO_TOTAL_CHANNELS: ", numberOfChannels)
    print("CAP_PROP_AUDIO_TOTAL_STREAMS: ", cap.get(cv.CAP_PROP_AUDIO_TOTAL_STREAMS))

    step = int(updateTime * samplingRate)
    frameSize = int(frameSizeTime * samplingRate)
    # since the dimensional grid is counted in integer seconds,
    # if duration of audio frame is less than xmarkup, to avoid an incorrect display,
    # xmarkup will be taken equal to duration
    if frameSizeTime <= xmarkup:
        xmarkup = frameSizeTime

    buffer = []
    section = np.zeros(frameSize, dtype=np.int16)
    currentSamples = 0

    while (1):
        if (cap.grab()):
            frame = []
            frame = np.asarray(frame)
            frame = cap.retrieve(frame, audioBaseIndex)

            for i in range(len(frame[1][0])):
                buffer.append(frame[1][0][i])

            buffer_size = len(buffer)
            if (buffer_size >= step):

                section = list(section)
                currentSamples += step

                del section[0:step]
                section.extend(buffer[0:step])
                del buffer[0:step]

                section = np.asarray(section)

                if currentSamples < frameSize:
                    xmin = 0
                    xmax = (currentSamples) / samplingRate
                else:
                    xmin = (currentSamples - frameSize) / samplingRate + 1
                    xmax = (currentSamples) / samplingRate

                if graph == "ampl":
                    imgAmplitude = drawAmplitude(section)
                    imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate,
                                                      grid, rows, cols,
                                                      xmarkup, ymarkup,
                                                      xmin, xmax)
                    cv.imshow("Display amplitude graph", imgAmplitude)
                    cv.waitKey(waitTime)

                elif graph == "spec":
                    stft = STFT(section, windowType, windLen, overlap)
                    imgSpec = drawSpectrogram(stft)
                    imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                      rows, cols,
                                                      xmarkup, ymarkup, zmarkup,
                                                      xmin, xmax)
                    cv.imshow("Display spectrogram", imgSpec)
                    cv.waitKey(waitTime)

                elif graph == "ampl_and_spec":

                    imgAmplitude = drawAmplitude(section)
                    stft = STFT(section, windowType, windLen, overlap)
                    imgSpec = drawSpectrogram(stft)

                    imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate,
                                                      grid, rows, cols,
                                                      xmarkup, ymarkup,
                                                      xmin, xmax)
                    imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                      rows, cols,
                                                      xmarkup, ymarkup, zmarkup,
                                                      xmin, xmax)

                    imgTotal = concatenateImages(imgAmplitude, imgSpec, rows, cols)
                    cv.imshow("Display amplitude graph and spectrogram", imgTotal)
                    cv.waitKey(waitTime)

        else:
            break


def dynamicMicrophone(microTime=10, graph="ampl_and_spec",
                      frameSizeTime=5, updateTime=1, waitTime=10,
                      windowType="Rect", windLen=256, overlap=128,
                      grid="off", rows=800, cols=900,
                      xmarkup=5, ymarkup=5, zmarkup=5):
    cap = cv.VideoCapture()
    params = [cv.CAP_PROP_AUDIO_STREAM, 0, cv.CAP_PROP_VIDEO_STREAM, -1]
    params = np.asarray(params)

    cap.open(0, cv.CAP_MSMF, params)
    if cap.isOpened() == False:
        print("ERROR! Can't to open file")
        return
    audioBaseIndex = int(cap.get(cv.CAP_PROP_AUDIO_BASE_INDEX))
    numberOfChannels = int(cap.get(cv.CAP_PROP_AUDIO_TOTAL_CHANNELS))

    print("CAP_PROP_AUDIO_DATA_DEPTH: ", str((int(cap.get(cv.CAP_PROP_AUDIO_DATA_DEPTH)))))
    print("CAP_PROP_AUDIO_SAMPLES_PER_SECOND: ", cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
    print("CAP_PROP_AUDIO_TOTAL_CHANNELS: ", numberOfChannels)
    print("CAP_PROP_AUDIO_TOTAL_STREAMS: ", cap.get(cv.CAP_PROP_AUDIO_TOTAL_STREAMS))

    frame = []
    frame = np.asarray(frame)
    samplingRate = int(cap.get(cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))

    step = int(updateTime * samplingRate)
    frameSize = int(frameSizeTime * samplingRate)
    xmarkup = frameSizeTime

    currentSamples = 0

    buffer = []
    section = np.zeros(frameSize, dtype=np.int16)

    cvTickFreq = cv.getTickFrequency()
    sysTimeCurr = cv.getTickCount()
    sysTimePrev = sysTimeCurr

    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < microTime):

        if (cap.grab()):
            for nCh in range(numberOfChannels):
                frame = []
                frame = np.asarray(frame)
                frame = cap.retrieve(frame, audioBaseIndex + nCh)

                for i in range(len(frame[1][0])):
                    buffer.append(frame[1][0][i])

                sysTimeCurr = cv.getTickCount()
                buffer_size = len(buffer)
                if (buffer_size >= step):

                    section = list(section)
                    currentSamples += step

                    del section[0:step]
                    section.extend(buffer[0:step])
                    del buffer[0:step]

                    section = np.asarray(section)

                    if currentSamples < frameSize:
                        xmin = 0
                        xmax = (currentSamples) / samplingRate
                    else:
                        xmin = (currentSamples - frameSize) / samplingRate + 1
                        xmax = (currentSamples) / samplingRate

                    if graph == "ampl":
                        imgAmplitude = drawAmplitude(section)
                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate,
                                                          grid, rows, cols,
                                                          xmarkup, ymarkup,
                                                          xmin, xmax)
                        cv.imshow("Display amplitude graph", imgAmplitude)
                        cv.waitKey(waitTime)

                    elif graph == "spec":
                        stft = STFT(section, windowType, windLen, overlap)
                        imgSpec = drawSpectrogram(stft)
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                          rows, cols,
                                                          xmarkup, ymarkup, zmarkup,
                                                          xmin, xmax)
                        cv.imshow("Display spectrogram", imgSpec)
                        cv.waitKey(waitTime)

                    elif graph == "ampl_and_spec":
                        imgAmplitude = drawAmplitude(section)
                        stft = STFT(section, windowType, windLen, overlap)
                        imgSpec = drawSpectrogram(stft)

                        imgAmplitude = drawAmplitudeScale(imgAmplitude, section, samplingRate,
                                                          grid, rows, cols,
                                                          xmarkup, ymarkup,
                                                          xmin, xmax)
                        imgSpec = drawSpectrogramColorbar(imgSpec, section, samplingRate, stft,
                                                          rows, cols,
                                                          xmarkup, ymarkup, zmarkup,
                                                          xmin, xmax)

                        imgTotal = concatenateImages(imgAmplitude, imgSpec, rows, cols)
                        cv.imshow("Display amplitude graph and spectrogram", imgTotal)
                        cv.waitKey(waitTime)

        else:
            break


def checkArgs(args):
    if args.inputMethod != "file" and args.inputMethod != "microphone":
        print("Error: ", args.inputMethod, " input method doesnt exist")
        return False
    if args.draw != "static" and args.draw != "dynamic":
        print("Error: ", args.draw, " draw type doesnt exist")
        return False
    if args.graph != "ampl" and args.graph != "spec" and args.graph != "ampl_and_spec":
        print("Error: ", args.graph, " type of graph doesnt exist")
        return False
    if args.windowType != "Rect" and args.windowType != "Hann" and args.windowType != "Hamming":
        print("Error: ", args.windowType, " type of window doesnt exist")
        return False
    if args.windLen <= 0:
        print("Error: windLen = ", args.windLen, " - incorrect value. Must be > 0")
        return False
    if args.overlap <= 0:
        print("Error: overlap = ", args.overlap, " - incorrect value. Must be > 0")
        return False
    if args.grid != "on" and args.grid != "off":
        print("Error: ", args.draw, " grid type doesnt exist")
        return False
    if args.rows <= 0:
        print("Error: rows = ", args.rows, " - incorrect value. Must be > 0")
        return False
    if args.cols <= 0:
        print("Error: cols = ", args.cols, " - incorrect value. Must be > 0")
        return False
    if args.xmarkup < 2:
        print("Error: xmarkup = ", args.xmarkup, " - incorrect value. Must be >= 2")
        return False
    if args.ymarkup < 2:
        print("Error: ymarkup = ", args.ymarkup, " - incorrect value. Must be >= 2")
        return False
    if args.zmarkup < 2:
        print("Error: zmarkup = ", args.zmarkup, " - incorrect value. Must be >= 2")
        return False
    if args.microTime <= 0:
        print("Error: microTime = ", args.microTime, " - incorrect value. Must be > 0")
        return False
    if args.frameSizeTime <= 0:
        print("Error: frameSizeTime = ", args.frameSizeTime, " - incorrect value. Must be > 0")
        return False
    if args.updateTime <= 0:
        print("Error: updateTime = ", args.updateTime, " - incorrect value. Must be > 0")
        return False
    if args.waitTime < 0:
        print("Error: waitTime = ", args.waitTime, " - incorrect value. Must be >= 0")
        return False
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''this sample draws a volume graph and/or spectrogram of audio/video files and microphone\nDefault usage: ./Spectrogram.exe''')

    parser.add_argument("-i", "--inputMethod", dest="inputMethod", type=str, default="file", help="file or microphone")
    parser.add_argument("-d", "--draw", dest="draw", type=str, default="static",
                        help="type of drawing: static - for plotting graph(s) across the entire input audio; dynamic - for plotting graph(s) in a time-updating window")
    parser.add_argument("-g", "--graph", dest="graph", type=str, default="ampl_and_spec",
                        help="type of graph: amplitude graph or/and spectrogram. Please use tags below : ampl - draw the amplitude graph; spec - draw the spectrogram; ampl_and_spec - draw the amplitude graph and spectrogram on one image under each other")

    parser.add_argument("-a", "--audio", dest="audio", type=str, default='../data/Megamind.avi',
                        help="name and path to file")
    parser.add_argument("-s", "--audioStream", dest="audioStream", type=int, default=1,
                        help=" CAP_PROP_AUDIO_STREAM value")

    parser.add_argument("-t", '--windowType', dest="windowType", type=str, default="Rect",
                        help="type of window for STFT. Please use tags below : Rect/Hann/Hamming")
    parser.add_argument("-l", '--windLen', dest="windLen", type=int, default=256, help="size of window for STFT")
    parser.add_argument("-o", '--overlap', dest="overlap", type=int, default=128, help="overlap of windows for STFT")

    parser.add_argument("-gd", '--grid', dest="grid", type=str, default="off", help="grid on amplitude graph(on/off)")

    parser.add_argument("-r", '--rows', dest="rows", type=int, default=800, help="rows of output image")
    parser.add_argument("-c", '--cols', dest="cols", type=int, default=900, help="cols of output image")

    parser.add_argument("-x", '--xmarkup', dest="xmarkup", type=int, default=5,
                        help="number of x axis divisions (time asix)")
    parser.add_argument("-y", '--ymarkup', dest="ymarkup", type=int, default=5,
                        help="number of y axis divisions (frequency or/and amplitude axis)")  # ?
    parser.add_argument("-z", '--zmarkup', dest="zmarkup", type=int, default=5,
                        help="number of z axis divisions (colorbar)")  # ?

    parser.add_argument("-m", '--microTime', dest="microTime", type=int, default=20,
                        help="time of recording audio with microphone in seconds")
    parser.add_argument("-f", '--frameSizeTime', dest="frameSizeTime", type=int, default=5,
                        help="size of sliding window in seconds")
    parser.add_argument("-u", '--updateTime', dest="updateTime", type=int, default=1,
                        help="update time of sliding window in seconds")
    parser.add_argument("-w", '--waitTime', dest="waitTime", type=int, default=10,
                        help="parameter to cv.waitKey() for dynamic update, takes values in milliseconds")

    args = parser.parse_args()

    if checkArgs(args) is False:
        exit()

    if args.draw == "static":

        if args.inputMethod == "file":
            samplingRate, inputAudio = readAudioFile(args.audio, args.audioStream)

        elif args.inputMethod == "microphone":
            samplingRate, inputAudio = readAudioMicrophone()

        if samplingRate == 0 or len(inputAudio) == 0:
            print("Cant read audio")
            exit()

        duration = len(inputAudio) // samplingRate

        # since the dimensional grid is counted in integer seconds,
        # if the input audio has an incomplete last second,
        # then it is filled with zeros to complete
        remainder = len(inputAudio) % samplingRate
        if remainder != 0:
            sizeToFullSec = samplingRate - remainder
            zeroArr = np.zeros(sizeToFullSec)
            inputAudio = np.concatenate((inputAudio, zeroArr), axis=0)
            duration += 1
            print("update duration of audio to full second with ",
                  sizeToFullSec, " zero samples")
            print("new number of samples ", len(inputAudio))

        if duration <= args.xmarkup:
            args.xmarkup = duration + 1

        if args.graph == "ampl":
            imgAmplitude = drawAmplitude(inputAudio)
            imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate,
                                              args.grid, args.rows, args.cols,
                                              args.xmarkup, args.ymarkup)
            cv.imshow("Display window", imgAmplitude)
            cv.waitKey(0)

        elif args.graph == "spec":
            stft = STFT(inputAudio, args.windowType, args.windLen, args.overlap)
            imgSpec = drawSpectrogram(stft)
            imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft,
                                              args.rows, args.cols,
                                              args.xmarkup, args.ymarkup, args.zmarkup)
            cv.imshow("Display window", imgSpec)
            cv.waitKey(0)

        elif args.graph == "ampl_and_spec":
            imgAmplitude = drawAmplitude(inputAudio)
            imgAmplitude = drawAmplitudeScale(imgAmplitude, inputAudio, samplingRate,
                                              args.grid, args.rows, args.cols,
                                              args.xmarkup, args.ymarkup)

            stft = STFT(inputAudio, args.windowType, args.windLen, args.overlap)
            imgSpec = drawSpectrogram(stft)
            imgSpec = drawSpectrogramColorbar(imgSpec, inputAudio, samplingRate, stft,
                                              args.rows, args.cols,
                                              args.xmarkup, args.ymarkup, args.zmarkup)

            imgTotal = concatenateImages(imgAmplitude, imgSpec, args.rows, args.cols)
            cv.imshow("Display window", imgTotal)
            cv.waitKey(0)

    elif args.draw == "dynamic":

        if args.inputMethod == "file":
            dynamicFile(args.audio, args.audioStream, args.graph,
                        args.frameSizeTime, args.updateTime, args.waitTime,
                        args.windowType, args.windLen, args.overlap,
                        args.grid, args.rows, args.cols,
                        args.xmarkup, args.ymarkup, args.zmarkup)

        elif args.inputMethod == "microphone":
            dynamicMicrophone(args.microTime, args.graph,
                              args.frameSizeTime, args.updateTime, args.waitTime,
                              args.windowType, args.windLen, args.overlap,
                              args.grid, args.rows, args.cols,
                              args.xmarkup, args.ymarkup, args.zmarkup)
