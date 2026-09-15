// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Text-to-speech synthesis with Qwen3-TTS: reads a sentence and writes the spoken audio
as a .wav file.

How to use:
    Sample command to run:
        `./example_dnn_qwen3_tts --model=<onnx dir> --tokenizer=<tokenizer dir> --text="Hello."`

    Download the cpu_fp32 variant of the model (a directory of ONNX graphs) from
        https://huggingface.co/onnx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice
    and pass that directory with --model. Speech synthesis runs six of its graphs:
        text_embed     : text_ids[1,S] i64            -> [1,S,1024] f32
        codec_embed    : codec_ids[1,S] i64           -> [1,S,1024] f32
        talker_cache   : inputs_embeds[1,T,1024] f32 + position_ids[3,1,T] i64
                         + attention_mask[1,T] i64 + 56 past K/V [1,8,past,128] f32
                         -> logits[1,T,3072] f32, hidden_states[1,T,1024] f32
        code_predictor : talker_hidden[1,1024] f32 + codec_ids[1,16] i64 -> [1,15,2048] f32
        residual_embed : codec_ids[1,16] i64          -> [1,1024] f32
        tok_decoder    : audio_codes[1,25,16] i64     -> waveform[1,1,L] f32

    --tokenizer takes a directory holding config.json and tokenizer.json of any Qwen2-family
    tokenizer; Qwen3-TTS reuses that vocabulary and adds only ids this sample never encodes.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "common.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

const string about =
        "Use this script for text-to-speech synthesis with Qwen3-TTS using OpenCV.\n\n"
        "Download the model directory from HuggingFace (see the file header), then run:\n"
        "\t Example: ./example_dnn_qwen3_tts --model=<onnx dir> --tokenizer=<tokenizer dir> "
        "--text=\"OpenCV is an open source computer vision library.\" --speaker=serena\n\n"
        "Synthesis is greedy, so the same text always produces the same audio.\n\n";

const string param_keys =
    "{ help h     |               | Print help message. }"
    "{ model m    |               | Directory with the Qwen3-TTS ONNX graphs. }"
    "{ tokenizer t|               | Directory with config.json and tokenizer.json. }"
    "{ text       | OpenCV is an open source computer vision library. | Text to synthesize. }"
    "{ speaker    | serena        | Speaker name: serena, vivian, uncle_fu, ryan, aiden, "
    "ono_anna, sohee, eric or dylan. }"
    "{ language   | english       | Language tag, or auto to let the model decide. }"
    "{ output o   | qwen3_tts.wav | Path of the .wav file to write. }"
    "{ max_frames | 400           | Stop after this many code frames (about 12 per second). }";

const string backend_keys = format(
    "{ backend    | default | Choose one of computation backends: "
                              "default: automatically (by default), "
                              "openvino: Intel's Deep Learning Inference Engine, "
                              "opencv: OpenCV implementation, "
                              "vkcom: VKCOM, "
                              "cuda: CUDA, "
                              "webnn: WebNN }");

const string target_keys = format(
    "{ target     | cpu | Choose one of target computation devices: "
                          "cpu: CPU target (by default), "
                          "opencl: OpenCL, "
                          "opencl_fp16: OpenCL fp16 (half-float precision), "
                          "vpu: VPU, "
                          "vulkan: Vulkan, "
                          "cuda: CUDA, "
                          "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

// Audio and graph geometry, all fixed by the export.
const int SAMPLE_RATE = 24000;
const int HIDDEN = 1024;
const int GROUPS = 16;       // codes per frame
const int DEC_FRAMES = 25;   // tok_decoder is exported at a fixed frame count
const int LAYERS = 28, KV_HEADS = 8, HEAD_DIM = 128;

// Control ids, from the model's config.json: the first three are top-level, the rest come
// from its talker_config. Codes in the top 1024 of the vocabulary are control ids.
const int64_t TTS_BOS = 151672, TTS_EOS = 151673, TTS_PAD = 151671;
const int64_t CODEC_PAD = 2148, CODEC_BOS = 2149, CODEC_EOS = 2150;
const int64_t CODEC_THINK = 2154, CODEC_NOTHINK = 2155;
const int64_t CODEC_THINK_BOS = 2156, CODEC_THINK_EOS = 2157;
const int VOCAB = 3072, CONTROL_IDS = 1024;
const double REPETITION_PENALTY = 1.05;
const int MAX_CONSECUTIVE_REPEAT = 12;

// config.json talker_config: spk_id and codec_language_id.
const map<string, int64_t> SPEAKER_IDS {
    {"serena", 3066}, {"vivian", 3065}, {"uncle_fu", 3010}, {"ryan", 3061}, {"aiden", 2861},
    {"ono_anna", 2873}, {"sohee", 2864}, {"eric", 2875}, {"dylan", 2878}
};
const map<string, int64_t> LANGUAGE_IDS {
    {"chinese", 2055}, {"english", 2050}, {"german", 2053}, {"italian", 2070},
    {"portuguese", 2071}, {"spanish", 2054}, {"japanese", 2058}, {"korean", 2064},
    {"french", 2061}, {"russian", 2069}, {"beijing_dialect", 2074}, {"sichuan_dialect", 2062}
};

static void requireFile(const string& path)
{
    ifstream f(path);
    if (!f.good())
        CV_Error(Error::StsError, "cannot open " + path);
}

static Mat feed3d(const Mat& seq)
{
    const int sizes[3] = {1, seq.rows, seq.cols};
    return seq.reshape(1, 3, sizes);
}

static Mat idsMat(const vector<int64_t>& ids)
{
    const int sizes[2] = {1, (int)ids.size()};
    Mat m(2, sizes, CV_64S);
    memcpy(m.ptr<int64_t>(), ids.data(), ids.size() * sizeof(int64_t));
    return m;
}

static Mat embed(Net& net, const vector<int64_t>& ids)
{
    net.setInput(idsMat(ids), "codec_ids");
    Mat out = net.forward();
    return out.reshape(1, (int)out.total() / HIDDEN).clone();
}

static Mat embedText(Net& net, const vector<int64_t>& ids)
{
    net.setInput(idsMat(ids), "text_ids");
    Mat out = net.forward();
    return out.reshape(1, (int)out.total() / HIDDEN).clone();
}

static Mat vcat(const vector<Mat>& parts)
{
    int rows = 0;
    for (const Mat& p : parts)
        rows += p.rows;
    Mat out(rows, HIDDEN, CV_32F);
    int r = 0;
    for (const Mat& p : parts)
    {
        p.copyTo(out.rowRange(r, r + p.rows));
        r += p.rows;
    }
    return out;
}

static Mat addRow(const Mat& seq, const Mat& row)
{
    Mat out = seq.clone();
    for (int r = 0; r < out.rows; r++)
        out.row(r) += row;
    return out;
}

static int argmaxRange(const float* v, int count)
{
    int best = 0;
    for (int i = 1; i < count; i++)
    {
        if (v[i] > v[best])
            best = i;
    }
    return best;
}

static void writeWav(const string& path, const vector<float>& samples, int rate)
{
    ofstream f(path, ios::binary);
    if (!f)
        CV_Error(Error::StsError, "cannot write " + path);

    const uint32_t dataBytes = (uint32_t)(samples.size() * sizeof(int16_t));
    const uint32_t fmtSize = 16, riffSize = 36 + dataBytes;
    const uint16_t pcm = 1, channels = 1, bits = 16, blockAlign = channels * bits / 8;
    const uint32_t byteRate = rate * blockAlign;

    f.write("RIFF", 4);
    f.write((const char*)&riffSize, 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    f.write((const char*)&fmtSize, 4);
    f.write((const char*)&pcm, 2);
    f.write((const char*)&channels, 2);
    f.write((const char*)&rate, 4);
    f.write((const char*)&byteRate, 4);
    f.write((const char*)&blockAlign, 2);
    f.write((const char*)&bits, 2);
    f.write("data", 4);
    f.write((const char*)&dataBytes, 4);
    for (float s : samples)
    {
        const int16_t v = saturate_cast<int16_t>(s * 32767.f);
        f.write((const char*)&v, 2);
    }
}

static int64_t specialTokenId(const string& tokenizerDir, const string& token)
{
    const string path = tokenizerDir + "tokenizer_config.json";
    requireFile(path);
    FileStorage fs(path, FileStorage::READ);
    CV_Assert(fs.isOpened());
    FileNode added = fs["added_tokens_decoder"];
    CV_Assert(!added.empty());
    for (FileNodeIterator it = added.begin(); it != added.end(); ++it)
    {
        string content;
        (*it)["content"] >> content;
        if (content == token)
            return (int64_t)stol((*it).name());
    }
    CV_Error(Error::StsError, "token " + token + " is missing from " + path);
}

static int64_t lookupId(const map<string, int64_t>& table, const string& name, const string& what)
{
    map<string, int64_t>::const_iterator it = table.find(name);
    if (it == table.end())
        CV_Error(Error::StsBadArg, "unknown " + what + ": " + name);
    return it->second;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help") || !parser.has("model") || !parser.has("tokenizer"))
    {
        parser.printMessage();
        return 0;
    }

    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_WARNING);

    const string modelDir = parser.get<String>("model");
    const string text = parser.get<String>("text");
    const string language = parser.get<String>("language");
    const string outPath = parser.get<String>("output");
    const int maxFrames = parser.get<int>("max_frames");
    const int backendId = getBackendID(parser.get<String>("backend"));
    const int targetId = getTargetID(parser.get<String>("target"));

    string tokenizerDir = parser.get<String>("tokenizer");
    if (!tokenizerDir.empty() && tokenizerDir.back() != '/')
        tokenizerDir += '/';

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    const bool autoLanguage = language == "auto";
    const int64_t speakerId = lookupId(SPEAKER_IDS, parser.get<String>("speaker"), "speaker");
    const int64_t languageId = autoLanguage ? 0 : lookupId(LANGUAGE_IDS, language, "language");

    const string tokenizerConfig = tokenizerDir + "config.json";
    requireFile(tokenizerConfig);
    requireFile(tokenizerDir + "tokenizer.json");
    Tokenizer tokenizer = Tokenizer::load(tokenizerConfig);

    const vector<int> rolePiece = tokenizer.encode("assistant\n");
    const vector<int> textPiece = tokenizer.encode(text);
    CV_Assert(!rolePiece.empty() && !textPiece.empty());

    vector<int64_t> roleIds {specialTokenId(tokenizerDir, "<|im_start|>")};
    roleIds.insert(roleIds.end(), rolePiece.begin(), rolePiece.end());
    const vector<int64_t> textIds(textPiece.begin(), textPiece.end());

    const char* graphs[] = {"text_embed", "codec_embed", "talker_cache",
                            "code_predictor", "residual_embed", "tok_decoder"};
    Net nets[6];
    for (int i = 0; i < 6; i++)
    {
        const string path = modelDir + "/" + graphs[i] + ".onnx";
        requireFile(path);
        cout << "loading  : " << graphs[i] << endl;
        nets[i] = readNetFromONNX(path);
        nets[i].setPreferableBackend(backendId);
        nets[i].setPreferableTarget(targetId);
    }
    Net& textEmbed = nets[0];
    Net& codecEmbed = nets[1];
    Net& talker = nets[2];
    Net& codePredictor = nets[3];
    Net& residualEmbed = nets[4];
    Net& tokDecoder = nets[5];

    // Prefill: role embeddings, then the codec tag block, then the text and its closing tags.
    Mat special = embedText(textEmbed, {TTS_BOS, TTS_EOS, TTS_PAD});
    Mat bosEmbed = special.row(0), eosEmbed = special.row(1), padEmbed = special.row(2);

    const vector<int64_t> tags = autoLanguage
            ? vector<int64_t>{CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS}
            : vector<int64_t>{CODEC_THINK, CODEC_THINK_BOS, languageId, CODEC_THINK_EOS};
    Mat codecInput = vcat({embed(codecEmbed, tags), embed(codecEmbed, {speakerId}),
                           embed(codecEmbed, {CODEC_PAD, CODEC_BOS})});

    Mat padBlock(codecInput.rows - 1, HIDDEN, CV_32F);
    for (int r = 0; r < padBlock.rows - 1; r++)
        padEmbed.copyTo(padBlock.row(r));
    bosEmbed.copyTo(padBlock.row(padBlock.rows - 1));

    Mat textBody = embedText(textEmbed, textIds);
    const vector<int64_t> padRun((size_t)textBody.rows + 1, CODEC_PAD);
    Mat textBlock = vcat({textBody, eosEmbed}) + embed(codecEmbed, padRun);
    Mat startBlock = addRow(embed(codecEmbed, {CODEC_BOS}), padEmbed);

    Mat talkerIn = vcat({embedText(textEmbed, roleIds),
                         padBlock + codecInput.rowRange(0, codecInput.rows - 1),
                         textBlock, startBlock});

    const int pastSizes[4] = {1, KV_HEADS, 0, HEAD_DIM};
    Mat emptyPast(4, pastSizes, CV_32F);
    vector<string> pastNames;
    for (int l = 0; l < LAYERS; l++)
    {
        for (int kv = 0; kv < 2; kv++)
            pastNames.push_back(l == 0 && kv == 0 ? "past_kv" : format("past_kv_%d_%d", l, kv));
    }

    cout << "text     : \"" << text << "\" (" << textIds.size() << " tokens)" << endl;

    vector<int64_t> codes;
    vector<char> seenFirst((size_t)(VOCAB - CONTROL_IDS), 0);
    int64_t lastFirst = -1;
    int repeatRun = 0;
    vector<vector<char>> seenSub(GROUPS, vector<char>((size_t)(VOCAB - CONTROL_IDS), 0));
    vector<int64_t> lastSub(GROUPS, -1);
    vector<int> repeatRunSub(GROUPS, 0);
    int frames = 0;
    for (; frames < maxFrames; frames++)
    {
        const int total = talkerIn.rows;
        const int posSizes[3] = {3, 1, total};
        Mat positions(3, posSizes, CV_64S);
        for (int row = 0; row < 3; row++)
        {
            for (int t = 0; t < total; t++)
                positions.ptr<int64_t>(row, 0)[t] = t;
        }
        const int maskSizes[2] = {1, total};
        Mat mask(2, maskSizes, CV_64S, Scalar(1));

        talker.setInput(feed3d(talkerIn), "inputs_embeds");
        talker.setInput(positions, "position_ids");
        talker.setInput(mask, "attention_mask");
        for (const string& name : pastNames)
            talker.setInput(emptyPast, name);

        vector<Mat> talkerOut;
        talker.forward(talkerOut, vector<String>{"logits", "hidden_states"});
        const Mat& logits = talkerOut[0];
        const Mat& hidden = talkerOut[1];

        const float* step = logits.ptr<float>(0, logits.size[1] - 1);
        vector<float> allowed(step, step + (VOCAB - CONTROL_IDS));
        for (size_t i = 0; i < allowed.size(); i++)
            if (seenFirst[i])
                allowed[i] = allowed[i] < 0 ? (float)(allowed[i] * REPETITION_PENALTY)
                                            : (float)(allowed[i] / REPETITION_PENALTY);
        const int first = argmaxRange(allowed.data(), (int)allowed.size());
        if (step[CODEC_EOS] > allowed[first])
            break;
        repeatRun = (first == lastFirst) ? repeatRun + 1 : 1;
        if (repeatRun >= MAX_CONSECUTIVE_REPEAT)
            break;
        lastFirst = first;
        seenFirst[first] = 1;

        vector<int64_t> frame((size_t)GROUPS, 0);
        frame[0] = first;
        Mat talkerHidden(1, HIDDEN, CV_32F);
        memcpy(talkerHidden.ptr<float>(), hidden.ptr<float>(0, hidden.size[1] - 1),
               HIDDEN * sizeof(float));

        bool stuckSub = false;
        for (int g = 1; g < GROUPS; g++)
        {
            codePredictor.setInput(talkerHidden, "talker_hidden");
            codePredictor.setInput(idsMat(frame), "codec_ids");
            Mat groupLogits = codePredictor.forward();
            const float* gp = groupLogits.ptr<float>(0, g - 1);
            vector<float> gallowed(gp, gp + groupLogits.size[2]);
            for (size_t i = 0; i < gallowed.size(); i++)
                if (seenSub[g][i])
                    gallowed[i] = gallowed[i] < 0 ? (float)(gallowed[i] * REPETITION_PENALTY)
                                                  : (float)(gallowed[i] / REPETITION_PENALTY);
            const int val = argmaxRange(gallowed.data(), (int)gallowed.size());
            repeatRunSub[g] = (val == lastSub[g]) ? repeatRunSub[g] + 1 : 1;
            if (repeatRunSub[g] >= MAX_CONSECUTIVE_REPEAT)
                stuckSub = true;
            lastSub[g] = val;
            seenSub[g][val] = 1;
            frame[g] = val;
        }
        if (stuckSub)
            break;
        codes.insert(codes.end(), frame.begin(), frame.end());

        residualEmbed.setInput(idsMat(frame), "codec_ids");
        talkerIn = vcat({talkerIn, residualEmbed.forward().reshape(1, 1) + padEmbed});
    }

    const int frameCount = (int)codes.size() / GROUPS;
    cout << "frames   : " << frameCount << endl;
    if (frameCount == 0)
        CV_Error(Error::StsError, "the model produced no code frames");

    // tok_decoder takes a fixed number of frames, so a short tail is padded by repeating its
    // frames and the extra audio is trimmed back off.
    vector<float> waveform;
    for (int start = 0; start < frameCount; start += DEC_FRAMES)
    {
        const int have = min(DEC_FRAMES, frameCount - start);
        const int chunkSizes[3] = {1, DEC_FRAMES, GROUPS};
        Mat chunk(3, chunkSizes, CV_64S);
        for (int t = 0; t < DEC_FRAMES; t++)
            memcpy(chunk.ptr<int64_t>(0, t), &codes[(size_t)(start + t % have) * GROUPS],
                   GROUPS * sizeof(int64_t));

        tokDecoder.setInput(chunk, "audio_codes");
        Mat audio = tokDecoder.forward();
        int count = (int)audio.total();
        if (have < DEC_FRAMES)
            count = cvRound(count * (double)have / DEC_FRAMES);
        const float* p = audio.ptr<float>();
        waveform.insert(waveform.end(), p, p + count);
    }

    writeWav(outPath, waveform, SAMPLE_RATE);
    cout << "wrote    : " << outPath << " ("
         << waveform.size() / (float)SAMPLE_RATE << " s)" << endl;
    return 0;
}
