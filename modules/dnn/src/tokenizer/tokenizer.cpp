// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "utils.hpp"
#include "unicode.hpp"
#include "core_bpe.hpp"
#include "core_gemma.hpp"
#include "core_unigram.hpp"
#include "core_wordpiece.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {
enum class TokenizerFamily { Auto, BPE, SentencePiece, Unigram, WordPiece };
}

// The id layout a TemplateProcessing post_processor declares, from its 'single'
// and 'pair' templates. N chunks reuse the pair layout as
// pairPrefix chunk0 separator chunk1 ... chunkN-1 pairSuffix.
struct TemplateWrap {
    std::vector<int> prefixIds;      // before the sequence, from 'single'
    std::vector<int> suffixIds;      // after the sequence, from 'single'
    std::vector<int> pairPrefixIds;  // before the first chunk, from 'pair'
    std::vector<int> separatorIds;   // between two chunks, from 'pair'
    std::vector<int> pairSuffixIds;  // after the last chunk, from 'pair'
    bool hasPair = false;            // no 'pair' template means no way to join chunks
};

// Appends one chunk's ids, without a wrap of its own.
typedef std::function<void(const std::string&, std::vector<int>&)> ChunkEncoder;

static std::vector<int> encodeChunksWithWrap(const std::vector<std::string>& textChunks,
                                             const TemplateWrap& wrap,
                                             const ChunkEncoder& encodeBody)
{
    CV_Assert(textChunks.size() > 1);
    if (!wrap.hasPair)
        CV_Error(cv::Error::StsNotImplemented,
            "this tokenizer's post_processor declares no 'pair' template, so it cannot "
            "encode more than one text chunk");

    std::vector<int> ids = wrap.pairPrefixIds;
    for (size_t i = 0; i < textChunks.size(); i++) {
        if (i > 0)
            ids.insert(ids.end(), wrap.separatorIds.begin(), wrap.separatorIds.end());
        encodeBody(textChunks[i], ids);
    }
    ids.insert(ids.end(), wrap.pairSuffixIds.begin(), wrap.pairSuffixIds.end());
    return ids;
}

static CoreBPE buildTokenizerFromJson(cv::FileStorage& fs,
                          std::unordered_set<std::string>* outSpecial = nullptr);

// Strips the multi-MB base64 "precompiled_charsmap" before FileStorage parses it.
static cv::FileStorage openTokenizerJson(const std::string& jsonPath,
    std::string* outCharsmap = nullptr)
{
    std::ifstream in(jsonPath, std::ios::binary);
    if (!in.is_open())
        CV_Error(cv::Error::StsError, "Failed to open tokenizer.json: " + jsonPath);

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    std::string charsmap = extractAndStripBase64Field(text, "precompiled_charsmap");
    if (outCharsmap)
        *outCharsmap = std::move(charsmap);

    cv::FileStorage fs(text, cv::FileStorage::MEMORY | cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        CV_Error(cv::Error::StsError, "Failed to parse tokenizer.json: " + jsonPath);
    return fs;
}

struct Tokenizer::Impl {
    virtual ~Impl() {}
    virtual std::vector<int> encode(const std::string& text) = 0;
    // Not an encode() overload: that would hide one half of the pair in every subclass.
    virtual std::vector<int> encodeChunks(const std::vector<std::string>& textChunks) {
        if (textChunks.size() == 1)
            return encode(textChunks[0]);
        CV_Error(cv::Error::StsNotImplemented,
            "this tokenizer does not support encoding several text chunks as one sequence");
    }
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

class BpeTokenizerImpl : public Tokenizer::Impl {
public:
    BpeTokenizerImpl(CoreBPE core, std::unordered_set<std::string> special = {});

    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    Ptr<CoreBPE> coreBPE_;
    std::unordered_set<std::string> allowedSpecial_;
};

BpeTokenizerImpl::BpeTokenizerImpl(CoreBPE core, std::unordered_set<std::string> special)
    : coreBPE_(makePtr<CoreBPE>(std::move(core)))
    , allowedSpecial_(std::move(special)) {}

std::vector<int> BpeTokenizerImpl::encode(const std::string& text) {
    CV_Assert(coreBPE_);
    std::vector<uint32_t> tok = coreBPE_->encode(text, allowedSpecial_).first;
    return std::vector<int>(tok.begin(), tok.end());
}

std::string BpeTokenizerImpl::decode(const std::vector<int>& tokens) {
    CV_Assert(coreBPE_);
    std::vector<uint32_t> t32(tokens.begin(), tokens.end());
    const std::vector<std::uint8_t> optBytes = coreBPE_->decodeBytes(t32);
    if (optBytes.empty())
        CV_Error(cv::Error::StsError, "Invalid decode.");
    return std::string(reinterpret_cast<const char*>(optBytes.data()), optBytes.size());
}

class SentencePieceTokenizerImpl : public Tokenizer::Impl {
public:
    SentencePieceTokenizerImpl(CoreGemmaBPE model,
                                std::unordered_set<std::string> special = {},
                                TemplateWrap wrap = TemplateWrap());

    std::vector<int> encode(const std::string& text) override;
    std::vector<int> encodeChunks(const std::vector<std::string>& textChunks) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    void encodeBody(const std::string& text, std::vector<int>& ids) const;

    CoreGemmaBPE model_;
    std::unordered_set<std::string> allowedSpecial_;
    TemplateWrap wrap_;
};

SentencePieceTokenizerImpl::SentencePieceTokenizerImpl(CoreGemmaBPE model,
                                                        std::unordered_set<std::string> special,
                                                        TemplateWrap wrap)
    : model_(std::move(model)), allowedSpecial_(std::move(special)),
      wrap_(std::move(wrap)) {}

void SentencePieceTokenizerImpl::encodeBody(const std::string& text, std::vector<int>& ids) const {
    std::vector<int> body = model_.encode(text, allowedSpecial_);
    ids.insert(ids.end(), body.begin(), body.end());
}

std::vector<int> SentencePieceTokenizerImpl::encode(const std::string& text) {
    std::vector<int> ids = wrap_.prefixIds;
    encodeBody(text, ids);
    ids.insert(ids.end(), wrap_.suffixIds.begin(), wrap_.suffixIds.end());
    return ids;
}

std::vector<int> SentencePieceTokenizerImpl::encodeChunks(const std::vector<std::string>& textChunks) {
    if (textChunks.size() == 1)
        return encode(textChunks[0]);
    return encodeChunksWithWrap(textChunks, wrap_,
        [this](const std::string& chunk, std::vector<int>& ids) { encodeBody(chunk, ids); });
}

std::string SentencePieceTokenizerImpl::decode(const std::vector<int>& tokens) {
    // Strip only the wrap added here; CoreGemmaBPE handles the rest.
    size_t begin = 0, end = tokens.size();
    if (end - begin >= wrap_.prefixIds.size() &&
        std::equal(wrap_.prefixIds.begin(), wrap_.prefixIds.end(), tokens.begin()))
        begin += wrap_.prefixIds.size();
    if (end - begin >= wrap_.suffixIds.size() &&
        std::equal(wrap_.suffixIds.rbegin(), wrap_.suffixIds.rend(), tokens.rbegin()))
        end -= wrap_.suffixIds.size();
    if (begin == 0 && end == tokens.size())
        return model_.decode(tokens);
    return model_.decode(std::vector<int>(tokens.begin() + begin, tokens.begin() + end));
}

// CoreUnigram already adds eos/strips specials; don't redo here.
class UnigramTokenizerImpl : public Tokenizer::Impl {
public:
    UnigramTokenizerImpl(CoreUnigram model, std::unordered_set<std::string> special = {},
                          TemplateWrap wrap = TemplateWrap());

    std::vector<int> encode(const std::string& text) override;
    std::vector<int> encodeChunks(const std::vector<std::string>& textChunks) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    CoreUnigram model_;
    std::unordered_set<std::string> allowedSpecial_;
    // CoreUnigram applies the single-sequence wrap itself; this is for chunk layout.
    TemplateWrap wrap_;
};

UnigramTokenizerImpl::UnigramTokenizerImpl(CoreUnigram model, std::unordered_set<std::string> special,
                                            TemplateWrap wrap)
    : model_(std::move(model)), allowedSpecial_(std::move(special)), wrap_(std::move(wrap)) {}

std::vector<int> UnigramTokenizerImpl::encode(const std::string& text) {
    return model_.encode(text, allowedSpecial_);
}

std::vector<int> UnigramTokenizerImpl::encodeChunks(const std::vector<std::string>& textChunks) {
    if (textChunks.size() == 1)
        return encode(textChunks[0]);
    return encodeChunksWithWrap(textChunks, wrap_,
        [this](const std::string& chunk, std::vector<int>& ids) {
            model_.encodeBody(chunk, allowedSpecial_, ids);
        });
}

std::string UnigramTokenizerImpl::decode(const std::vector<int>& tokens) {
    return model_.decode(tokens);
}

// Matches reference BasicTokenizer's ASCII punctuation ranges exactly.
static bool isBertAsciiPunct(uint32_t cpt)
{
    return (cpt >= 33 && cpt <= 47) || (cpt >= 58 && cpt <= 64) ||
           (cpt >= 91 && cpt <= 96) || (cpt >= 123 && cpt <= 126);
}

// Don't drop \t\n\r; reference treats them as whitespace.
static bool isBertControl(uint32_t cpt, unicode_cpt_flags flags)
{
    return flags.is_control && cpt != '\t' && cpt != '\n' && cpt != '\r';
}

// CJK ranges per the reference BasicTokenizer._is_chinese_char.
static bool isBertChineseChar(uint32_t cpt)
{
    return (cpt >= 0x4E00 && cpt <= 0x9FFF) ||
           (cpt >= 0x3400 && cpt <= 0x4DBF) ||
           (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
           (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
           (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
           (cpt >= 0x2B820 && cpt <= 0x2CEAF) ||
           (cpt >= 0xF900 && cpt <= 0xFAFF) ||
           (cpt >= 0x2F800 && cpt <= 0x2FA1F);
}

class WordPieceTokenizerImpl : public Tokenizer::Impl {
public:
    WordPieceTokenizerImpl(CoreWordPiece model,
                            bool cleanText,
                            bool handleChineseChars,
                            bool stripAccents,
                            bool lowercase,
                            TemplateWrap wrap,
                            std::unordered_map<std::string, int> specialToId);

    std::vector<int> encode(const std::string& text) override;
    std::vector<int> encodeChunks(const std::vector<std::string>& textChunks) override;
    std::string decode(const std::vector<int>& tokens) override;

private:
    std::string normalize(const std::string& text) const;

    // normalize() already collapsed whitespace; punctuation is split here too.
    static std::vector<std::string> preTokenize(const std::string& text);

    void encodeNormalized(const std::string& text, std::vector<int>& ids) const;

    // Special/added tokens are matched literally before normalize()/preTokenize().
    void encodeSegment(const std::string& text, std::vector<int>& ids) const;

    bool isSpecialId(int id) const;

    CoreWordPiece model_;
    bool cleanText_;
    bool handleChineseChars_;
    bool stripAccents_;
    bool lowercase_;
    TemplateWrap wrap_;
    // Every id the template contributes, dropped by decode() wherever it sits.
    std::unordered_set<int> wrapIds_;
    // Literal added/special tokens (e.g. "[MASK]") bypassing normalize/preTokenize/encode.
    std::unordered_map<std::string, int> specialToId_;
};

WordPieceTokenizerImpl::WordPieceTokenizerImpl(CoreWordPiece model,
                                                bool cleanText,
                                                bool handleChineseChars,
                                                bool stripAccents,
                                                bool lowercase,
                                                TemplateWrap wrap,
                                                std::unordered_map<std::string, int> specialToId)
    : model_(std::move(model)),
      cleanText_(cleanText),
      handleChineseChars_(handleChineseChars),
      stripAccents_(stripAccents),
      lowercase_(lowercase),
      wrap_(std::move(wrap)),
      specialToId_(std::move(specialToId))
{
    for (const std::vector<int>* ids : { &wrap_.prefixIds, &wrap_.suffixIds,
                                         &wrap_.pairPrefixIds, &wrap_.separatorIds,
                                         &wrap_.pairSuffixIds })
        wrapIds_.insert(ids->begin(), ids->end());
}

std::string WordPieceTokenizerImpl::normalize(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        unicode_cpt_flags flags = unicode_cpt_flags_from_cpt(cpt);
        if (cleanText_ && (cpt == 0 || cpt == 0xFFFD || isBertControl(cpt, flags))) {
            continue;
        }
        if (cleanText_ && flags.is_whitespace) {
            out += ' ';
            continue;
        }
        if (handleChineseChars_ && isBertChineseChar(cpt)) {
            out += ' ';
            out += unicode_cpt_to_utf8(cpt);
            out += ' ';
            continue;
        }
        // Drop a standalone nonspacing mark; a spacing mark is a vowel, not an accent.
        if (stripAccents_ && flags.is_accent_mark && !unicode_cpt_is_spacing_mark(cpt)) {
            continue;
        }
        uint32_t effective = stripAccents_ ? unicode_strip_accent_base(cpt) : cpt;
        if (lowercase_) effective = unicode_tolower(effective);
        out += unicode_cpt_to_utf8(effective);
    }
    return out;
}

std::vector<std::string> WordPieceTokenizerImpl::preTokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
        unicode_cpt_flags flags = unicode_cpt_flags_from_cpt(cpt);
        bool isWs = flags.is_whitespace || cpt == ' ';
        bool isPunct = flags.is_punctuation || isBertAsciiPunct(cpt);
        if (isWs) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            continue;
        }
        if (isPunct) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            out.push_back(unicode_cpt_to_utf8(cpt));
            continue;
        }
        cur += unicode_cpt_to_utf8(cpt);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void WordPieceTokenizerImpl::encodeNormalized(const std::string& text, std::vector<int>& ids) const {
    for (const std::string& word : preTokenize(normalize(text))) {
        std::vector<int> wordIds = model_.encode(word);
        ids.insert(ids.end(), wordIds.begin(), wordIds.end());
    }
}

void WordPieceTokenizerImpl::encodeSegment(const std::string& text, std::vector<int>& ids) const {
    if (specialToId_.empty()) {
        encodeNormalized(text, ids);
        return;
    }
    splitOnSpecialTokens(text, specialToId_,
        [](const std::string&) { return true; },
        [&](const std::string& literal) { encodeNormalized(literal, ids); },
        [&](int id) { ids.push_back(id); });
}

std::vector<int> WordPieceTokenizerImpl::encode(const std::string& text) {
    std::vector<int> ids = wrap_.prefixIds;
    encodeSegment(text, ids);
    ids.insert(ids.end(), wrap_.suffixIds.begin(), wrap_.suffixIds.end());
    return ids;
}

std::vector<int> WordPieceTokenizerImpl::encodeChunks(const std::vector<std::string>& textChunks) {
    if (textChunks.size() == 1)
        return encode(textChunks[0]);
    return encodeChunksWithWrap(textChunks, wrap_,
        [this](const std::string& chunk, std::vector<int>& ids) { encodeSegment(chunk, ids); });
}

std::string WordPieceTokenizerImpl::decode(const std::vector<int>& tokens) {
    std::vector<int> filtered;
    filtered.reserve(tokens.size());
    for (int id : tokens) {
        if (wrapIds_.count(id)) continue;
        if (isSpecialId(id)) continue;
        filtered.push_back(id);
    }
    return model_.decode(filtered);
}

bool WordPieceTokenizerImpl::isSpecialId(int id) const {
    return std::any_of(specialToId_.begin(), specialToId_.end(),
                        [id](const std::pair<const std::string, int>& kv) { return kv.second == id; });
}

// Rewrites (?i:...) for std::regex. Only flat ASCII alternations translate;
// anything else is rejected.
static std::string expandCaseInsensitiveGroups(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in.compare(i, 4, "(?i:") != 0) {
            out += in[i++];
            continue;
        }
        size_t j = i + 4;
        int depth = 1;
        bool inClass = false;
        while (j < in.size() && depth > 0) {
            const char c = in[j];
            if (c == '\\' && j + 1 < in.size()) { j += 2; continue; }  // escapes are opaque
            if (inClass) {
                if (c == ']') inClass = false;
            } else if (c == '[') {
                inClass = true;
            } else if (c == '(') {
                depth++;
            } else if (c == ')') {
                depth--;
            }
            if (depth > 0) j++;
        }
        if (depth != 0)
            CV_Error(cv::Error::StsParseError,
                "tokenizer.json: unterminated '(?i:' group in the pre_tokenizer regex");

        const std::string inner = in.substr(i + 4, j - (i + 4));
        if (inner.find('[') != std::string::npos || inner.find('(') != std::string::npos)
            CV_Error(cv::Error::StsNotImplemented,
                "tokenizer.json: character classes and nested groups inside '(?i:...)' are "
                "not supported in the pre_tokenizer regex, only flat literal alternations: "
                + inner);

        out += "(?:";
        int braceDepth = 0;
        for (size_t k = 0; k < inner.size(); ++k) {
            const char c = inner[k];
            if (c == '\\' && k + 1 < inner.size()) {  // copy escapes verbatim
                out += c;
                out += inner[++k];
                continue;
            }
            if (c == '{') ++braceDepth;
            else if (c == '}') --braceDepth;
            // Never expand inside \p{...} or a {m,n} quantifier.
            if (braceDepth == 0 && std::isalpha(static_cast<unsigned char>(c))) {
                out += '[';
                out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                out += ']';
            } else {
                out += c;
            }
        }
        out += ")";
        i = (j < in.size()) ? j + 1 : j;
    }
    return out;
}

static std::string adaptHfPreTokenizerRegex(const std::string& raw)
{
    return expandCaseInsensitiveGroups(raw);
}

static bool findEmbeddedSplitRegex(const cv::FileNode& preTok, std::string& outRegex)
{
    if (preTok.empty()) return false;
    std::string type;
    preTok["type"] >> type;
    if (type == "Sequence") {
        cv::FileNode list = preTok["pretokenizers"];
        for (auto it = list.begin(); it != list.end(); ++it) {
            cv::FileNode child = *it;
            cv::FileNode regexNode = child["pattern"]["Regex"];
            if (!regexNode.empty() && regexNode.isString()) {
                regexNode >> outRegex;
                return true;
            }
        }
        return false;
    }
    cv::FileNode regexNode = preTok["pattern"]["Regex"];
    if (!regexNode.empty() && regexNode.isString()) {
        regexNode >> outRegex;
        return true;
    }
    return false;
}

// True if 'preTok' is (or contains, inside a Sequence) a ByteLevel pre_tokenizer.
// Byte-level BPE (GPT2/GPT4/Qwen-style) always routes text through a ByteLevel
// pre_tokenizer; SentencePiece-derived byte-fallback BPE (Gemma/Llama-style)
// never does. Used as a family discriminator: 'byte_fallback' alone is a
// decoder property, not proof of the SentencePiece family, so it must not be
// trusted on its own.
static bool hasByteLevelPreTokenizer(const cv::FileNode& preTok)
{
    if (preTok.empty()) return false;
    std::string type;
    preTok["type"] >> type;
    if (type == "ByteLevel") return true;
    if (type == "Sequence") {
        cv::FileNode list = preTok["pretokenizers"];
        for (auto it = list.begin(); it != list.end(); ++it) {
            cv::FileNode child = *it;
            std::string childType;
            child["type"] >> childType;
            if (childType == "ByteLevel") return true;
        }
    }
    return false;
}

static std::string detectSplitPattern(const cv::FileStorage& fs, const std::string& modelType)
{
    std::string raw;
    cv::FileNode preTok = fs["pre_tokenizer"];
    if (findEmbeddedSplitRegex(preTok, raw))
        return adaptHfPreTokenizerRegex(raw);
    // A bare ByteLevel pre_tokenizer (GPT-2 shape) carries no regex of its own and is
    // the only case R50K is right for. cl100k/o200k split differently and must not
    // silently inherit GPT-2's pattern.
    if ((preTok.empty() || hasByteLevelPreTokenizer(preTok)) &&
        (modelType.empty() || modelType == "BPE"))
        return R50K_UTF8;
    std::string preTokType;
    if (!preTok.empty())
        preTok["type"] >> preTokType;
    if (preTokType.empty())
        preTokType = "(none)";
    CV_Error(cv::Error::StsError,
        "No pre_tokenizer split regex in tokenizer.json and no default split pattern for "
        "pre_tokenizer '" + preTokType + "' / model type '" +
        (modelType.empty() ? std::string("(none)") : modelType) + "'");
}

// Resolves a post_processor SpecialToken id to a vocab id, or -1 if unknown.
typedef std::function<int(const std::string&)> SpecialIdResolver;

// Splits one template ('single' or 'pair') into the runs of SpecialToken ids around
// its $A/$B placeholders: slot 0 precedes the first placeholder, slot i follows the
// i-th one. A template without placeholders yields one slot.
static std::vector<std::vector<int>> readTemplateSlots(const cv::FileNode& tmpl,
                                                       const SpecialIdResolver& resolve)
{
    std::vector<std::vector<int>> slots(1);
    for (auto it = tmpl.begin(); it != tmpl.end(); ++it) {
        cv::FileNode entry = *it;
        if (!entry["Sequence"].empty()) {
            slots.push_back(std::vector<int>());
            continue;
        }
        cv::FileNode special = entry["SpecialToken"];
        if (special.empty())
            continue;
        std::string tokId;
        special["id"] >> tokId;
        const int id = resolve(tokId);
        if (id < 0) {
            CV_LOG_WARNING(NULL, "tokenizer.json: post_processor references special token '"
                << tokId << "' that has no id in this tokenizer; it will not be emitted");
            continue;
        }
        slots.back().push_back(id);
    }
    return slots;
}

// The 'pair' template is what makes multi-chunk encoding possible: its ids between
// $A and $B separate every neighbouring pair of chunks.
static TemplateWrap readTemplateProcessingWrap(const cv::FileNode& postProc,
                                               const SpecialIdResolver& resolve)
{
    TemplateWrap wrap;
    if (postProc.empty())
        return wrap;
    std::string postType;
    postProc["type"] >> postType;
    if (postType != "TemplateProcessing")
        return wrap;

    const std::vector<std::vector<int>> single = readTemplateSlots(postProc["single"], resolve);
    wrap.prefixIds = single.front();
    for (size_t i = 1; i < single.size(); i++)
        wrap.suffixIds.insert(wrap.suffixIds.end(), single[i].begin(), single[i].end());

    const std::vector<std::vector<int>> pair = readTemplateSlots(postProc["pair"], resolve);
    if (pair.size() == 3) {
        wrap.pairPrefixIds = pair[0];
        wrap.separatorIds = pair[1];
        wrap.pairSuffixIds = pair[2];
        wrap.hasPair = true;
    } else if (pair.size() > 1) {
        CV_LOG_WARNING(NULL, "tokenizer.json: post_processor 'pair' template does not have "
            "exactly two sequence placeholders; encoding several text chunks is disabled");
    }
    return wrap;
}

// The common case: template tokens come from 'added_tokens'.
static SpecialIdResolver addedTokenResolver(const std::unordered_map<std::string, int>& specialToId)
{
    return [specialToId](const std::string& name) -> int {
        auto found = specialToId.find(name);
        return found == specialToId.end() ? -1 : found->second;
    };
}

static Ptr<Tokenizer::Impl> buildSentencePieceTokenizerImpl(
        cv::FileStorage& fs,
        std::unordered_set<std::string>* outSpecial = nullptr)
{

    cv::FileNode model_node = fs["model"];
    CV_CheckFalse(model_node.empty(), "tokenizer.json missing 'model'");

    std::string model_type;
    model_node["type"] >> model_type;
    if (!model_type.empty() && model_type != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a byte-fallback BPE model in tokenizer.json, got: " + model_type);

    CoreGemmaBPE gemma;

    cv::FileNode vocab_node = model_node["vocab"];
    CV_CheckFalse(vocab_node.empty(), "tokenizer.json model missing 'vocab'");

    int maxId = -1;
    for (auto it = vocab_node.begin(); it != vocab_node.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece = entry.name();
        int id = (int)entry;
        if (id < 0)
            CV_Error(cv::Error::StsBadArg,
                "tokenizer.json: vocab entry '" + piece + "' has a negative id");
        if (id > maxId) maxId = id;
        gemma.pieceToId[piece] = id;
    }

    gemma.idToPiece.resize(maxId + 1);
    for (const auto& kv : gemma.pieceToId)
        gemma.idToPiece[kv.second] = kv.first;

    cv::FileNode merges_node = model_node["merges"];
    if (!merges_node.empty()) {
        uint32_t rank = 0;
        for (auto it = merges_node.begin(); it != merges_node.end(); ++it) {
            cv::FileNode entry = *it;
            std::string a, b;
            if (entry.isString()) {
                std::string merge_str;
                entry >> merge_str;
                size_t sp = merge_str.find(' ');
                if (sp == std::string::npos) {
                    ++rank;
                    continue;
                }
                a = merge_str.substr(0, sp);
                b = merge_str.substr(sp + 1);
            } else if (entry.size() == 2) {
                entry[0] >> a;
                entry[1] >> b;
            } else {
                ++rank;
                continue;
            }
            gemma.addMerge(a, b, rank);
            ++rank;
        }
    }

    std::unordered_set<std::string> special;
    std::unordered_map<std::string, int> specialToId;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            int id = -1;         cv::read(t["id"], id, id);
            std::string content; t["content"] >> content;
            if (id >= 0 && !content.empty()) {
                gemma.specialToId[content] = id;
                gemma.idToSpecial[id]      = content;
                specialToId[content]       = id;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    // The post_processor declares the wrap, not how 'merges' is serialized.
    TemplateWrap wrap = readTemplateProcessingWrap(fs["post_processor"],
                                                   addedTokenResolver(specialToId));

    return makePtr<SentencePieceTokenizerImpl>(std::move(gemma), std::move(special),
                                               std::move(wrap));
}

static void appendUnigramNormalizerStep(const cv::FileNode& node,
                                        std::vector<UnigramNormalizerStep>& out,
                                        std::string& unhandledNormalization)
{
    if (node.empty() || node.isNone())
        return;

    std::string type;
    node["type"] >> type;

    if (type == "Sequence") {
        cv::FileNode seq = node["normalizers"];
        for (auto it = seq.begin(); it != seq.end(); ++it)
            appendUnigramNormalizerStep(*it, out, unhandledNormalization);
        return;
    }

    UnigramNormalizerStep step;
    if (type == "Precompiled") {
        step.kind = UnigramNormalizerStep::PRECOMPILED;
    } else if (type == "Lowercase") {
        step.kind = UnigramNormalizerStep::LOWERCASE;
    } else if (type == "StripAccents") {
        step.kind = UnigramNormalizerStep::STRIP_ACCENTS;
    } else if (type == "Replace") {
        cv::FileNode pattern = node["pattern"];
        if (pattern.empty() || pattern["String"].empty()) {
            CV_LOG_WARNING(NULL, "tokenizer.json: 'Replace' normalizer with a non-String "
                "pattern is not supported and will be skipped; token ids may differ "
                "from the reference tokenizer");
            return;
        }
        step.kind = UnigramNormalizerStep::REPLACE;
        pattern["String"] >> step.from;
        node["content"] >> step.to;
        if (step.from.empty())
            return;
    } else if (type == "NFKD" || type == "NFKC" || type == "NFD" || type == "NFC") {
        // Reported once the whole chain is known; a Precompiled charsmap subsumes it.
        unhandledNormalization = type;
        return;
    } else {
        CV_LOG_WARNING(NULL, "tokenizer.json: unsupported normalizer step '" << type
            << "' will be skipped; token ids may differ from the reference tokenizer");
        return;
    }
    out.push_back(step);
}

static std::vector<UnigramNormalizerStep> readUnigramNormalizerSteps(const cv::FileNode& node)
{
    std::vector<UnigramNormalizerStep> steps;
    std::string unhandledNormalization;
    appendUnigramNormalizerStep(node, steps, unhandledNormalization);
    if (!unhandledNormalization.empty()) {
        const bool subsumed = std::any_of(steps.begin(), steps.end(),
            [](const UnigramNormalizerStep& s) {
                return s.kind == UnigramNormalizerStep::PRECOMPILED;
            });
        // Braces required: CV_LOG_DEBUG expands to a for-loop unless NDEBUG strips it.
        if (subsumed) {
            CV_LOG_DEBUG(NULL, "tokenizer.json: normalizer step '" << unhandledNormalization
                << "' is subsumed by the Precompiled charsmap; skipping");
        } else {
            CV_LOG_WARNING(NULL, "tokenizer.json: normalizer step '" << unhandledNormalization
                << "' is not implemented and no Precompiled charsmap covers it; token ids "
                   "may differ from the reference tokenizer");
        }
    }
    // An absent normalizer still runs the default PRECOMPILED step.
    if (steps.empty())
        steps.resize(1);
    return steps;
}

static Ptr<Tokenizer::Impl> buildUnigramTokenizerImpl(
        cv::FileStorage& fs, const std::string& charsmapB64,
        std::unordered_set<std::string>* outSpecial = nullptr)
{

    cv::FileNode modelNode = fs["model"];
    CV_CheckFalse(modelNode.empty(), "tokenizer.json missing 'model'");

    std::string modelType;
    modelNode["type"] >> modelType;
    if (!modelType.empty() && modelType != "Unigram")
        CV_Error(cv::Error::StsError,
            "Expected a Unigram model in tokenizer.json, got: " + modelType);

    cv::FileNode vocabNode = modelNode["vocab"];
    CV_CheckFalse(vocabNode.empty(), "tokenizer.json model missing 'vocab'");

    std::vector<std::pair<std::string, float>> vocab;
    vocab.reserve(vocabNode.size());
    for (auto it = vocabNode.begin(); it != vocabNode.end(); ++it) {
        cv::FileNode entry = *it;
        std::string piece;
        double score = 0.0;
        if (entry.size() >= 2) {
            entry[0] >> piece;
            entry[1] >> score;
        }
        vocab.emplace_back(piece, (float)score);
    }

    int unkId = -1;
    cv::read(modelNode["unk_id"], unkId, unkId);
    // -1 is the documented "no unk piece" sentinel (CoreUnigram handles it).
    // Anything else must be a valid vocab index -- reject a corrupt/out-of-
    // range unk_id now instead of letting it flow into CoreUnigram and come
    // back out as an invalid token id from encode().
    if (unkId != -1 && (unkId < 0 || (size_t)unkId >= vocab.size()))
        CV_Error(cv::Error::StsError, "tokenizer.json: 'unk_id' is out of range for 'vocab'");

    std::unordered_set<std::string> special;
    std::unordered_map<std::string, int> specialToId;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"] >> isSpecial;
            int tid = -1;           cv::read(t["id"], tid, tid);
            std::string content;    t["content"] >> content;
            if (isSpecial && tid >= 0 && !content.empty()) {
                specialToId[content] = tid;
                special.insert(content);
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    // The post_processor declares the wrap, not the name of any one added token.
    TemplateWrap wrap = readTemplateProcessingWrap(fs["post_processor"],
                                                   addedTokenResolver(specialToId));

    CoreUnigram unigram(vocab, unkId, buildUnigramPrecompiledNormalizer(charsmapB64),
                        specialToId, wrap.prefixIds, wrap.suffixIds,
                        readUnigramNormalizerSteps(fs["normalizer"]));

    return makePtr<UnigramTokenizerImpl>(std::move(unigram), std::move(special), std::move(wrap));
}

static Ptr<Tokenizer::Impl> buildWordPieceTokenizerImpl(cv::FileStorage& fs)
{
    cv::FileNode modelNode = fs["model"];
    CV_CheckFalse(modelNode.empty(), "tokenizer.json missing 'model'");

    std::string modelType;
    modelNode["type"] >> modelType;
    if (!modelType.empty() && modelType != "WordPiece")
        CV_Error(cv::Error::StsError,
            "Expected a WordPiece model in tokenizer.json, got: " + modelType);

    cv::FileNode vocabNode = modelNode["vocab"];
    CV_CheckFalse(vocabNode.empty(), "tokenizer.json model missing 'vocab'");

    std::string unkToken = "[UNK]";
    cv::read(modelNode["unk_token"], unkToken, unkToken);
    std::string continuingSubwordPrefix = "##";
    cv::read(modelNode["continuing_subword_prefix"], continuingSubwordPrefix, continuingSubwordPrefix);
    int maxInputCharsPerWord = 100;
    cv::read(modelNode["max_input_chars_per_word"], maxInputCharsPerWord, maxInputCharsPerWord);

    std::unordered_map<std::string, int> vocab;
    for (auto it = vocabNode.begin(); it != vocabNode.end(); ++it) {
        cv::FileNode entry = *it;
        vocab[entry.name()] = (int)entry;
    }

    CoreWordPiece core(vocab, unkToken, continuingSubwordPrefix,
                        (size_t)std::max(1, maxInputCharsPerWord));

    // See specialToId_ / encodeSegment() for bypass-of-normal-split usage.
    std::unordered_map<std::string, int> specialToId;
    cv::FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"] >> isSpecial;
            int tid = -1;           cv::read(t["id"], tid, tid);
            std::string content;    t["content"] >> content;
            if (isSpecial && tid >= 0 && !content.empty())
                specialToId[content] = tid;
        }
    }

    // Only a flat (non-Sequence) BertNormalizer node is read. A present-but-null
    // "strip_accents" is HF's encoding for "follow lowercase" -- it must resolve
    // to the current "lowercase" value, not false, to match the reference tokenizer.
    bool cleanText = true, handleChineseChars = true, lowercase = true, stripAccents;
    cv::FileNode normNode = fs["normalizer"];
    if (!normNode.empty()) {
        cv::read(normNode["clean_text"], cleanText, cleanText);
        cv::read(normNode["handle_chinese_chars"], handleChineseChars, handleChineseChars);
        cv::read(normNode["lowercase"], lowercase, lowercase);
        cv::FileNode stripAccentsNode = normNode["strip_accents"];
        if (!stripAccentsNode.empty() && !stripAccentsNode.isNone())
            stripAccentsNode >> stripAccents;
        else
            stripAccents = lowercase;
    } else {
        stripAccents = lowercase;
    }

    // [CLS]/[SEP] are usually in 'added_tokens' but always in the vocab.
    TemplateWrap wrap = readTemplateProcessingWrap(fs["post_processor"],
        [&](const std::string& name) -> int {
            auto found = specialToId.find(name);
            if (found != specialToId.end())
                return found->second;
            int id = -1;
            return core.tryGetId(name, id) ? id : -1;
        });

    // Older WordPiece exports ship no post_processor; BERT's wrap is the convention
    // for them, with [SEP] separating chunks the way the template would.
    int clsId = -1, sepId = -1;
    core.tryGetId("[CLS]", clsId);
    core.tryGetId("[SEP]", sepId);
    if (wrap.prefixIds.empty() && clsId >= 0)
        wrap.prefixIds.push_back(clsId);
    if (wrap.suffixIds.empty() && sepId >= 0)
        wrap.suffixIds.push_back(sepId);
    if (!wrap.hasPair && !wrap.suffixIds.empty()) {
        wrap.pairPrefixIds = wrap.prefixIds;
        wrap.separatorIds = wrap.suffixIds;
        wrap.pairSuffixIds = wrap.suffixIds;
        wrap.hasPair = true;
    }

    return makePtr<WordPieceTokenizerImpl>(std::move(core), cleanText, handleChineseChars,
                                            stripAccents, lowercase, std::move(wrap),
                                            std::move(specialToId));
}

static Ptr<Tokenizer::Impl> buildBPETokenizerImpl(cv::FileStorage& fs)
{
    std::unordered_set<std::string> special;
    CoreBPE core = buildTokenizerFromJson(fs, &special);
    return makePtr<BpeTokenizerImpl>(std::move(core), std::move(special));
}

static Ptr<Tokenizer::Impl> buildFromTokenizerDir(const std::string& dir,
        TokenizerFamily familyOverride = TokenizerFamily::Auto)
{
    std::string tokJson = dir + "tokenizer.json";
    std::string charsmapB64;
    cv::FileStorage fs = openTokenizerJson(tokJson, &charsmapB64);

    cv::FileNode model = fs["model"];
    if (model.empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json has no 'model' field; raw rank-table tokenizers are not "
            "supported by this loader: " + tokJson);

    std::string modelType;
    model["type"] >> modelType;

    if (familyOverride == TokenizerFamily::Unigram)
        return buildUnigramTokenizerImpl(fs, charsmapB64);
    if (familyOverride == TokenizerFamily::WordPiece)
        return buildWordPieceTokenizerImpl(fs);
    if (familyOverride == TokenizerFamily::SentencePiece)
        return buildSentencePieceTokenizerImpl(fs);
    if (familyOverride == TokenizerFamily::BPE)
        return buildBPETokenizerImpl(fs);

    // Some older HF tokenizer.json snapshots omit the model "type" field
    // entirely. Detect that case via each model kind's distinctive schema
    // instead of falling through to the BPE path below and erroring out on
    // the missing 'merges' table.
    bool looksLikeUnigram = modelType == "Unigram" ||
        (modelType.empty() &&
         !model["unk_id"].empty() &&
         model["unk_token"].empty() &&
         model["merges"].empty());
    if (looksLikeUnigram)
        return buildUnigramTokenizerImpl(fs, charsmapB64);

    bool looksLikeWordPiece = modelType == "WordPiece" ||
        (modelType.empty() &&
         !model["unk_token"].empty() &&
         !model["continuing_subword_prefix"].empty() &&
         model["merges"].empty());
    if (looksLikeWordPiece)
        return buildWordPieceTokenizerImpl(fs);

    bool byteFallback = false;
    model["byte_fallback"] >> byteFallback;
    // byte_fallback alone isn't a reliable discriminator (see hasByteLevelPreTokenizer);
    // require ByteLevel's absence too before routing to SentencePiece.
    bool looksLikeSentencePiece = byteFallback &&
        (modelType == "BPE" || modelType.empty()) &&
        !hasByteLevelPreTokenizer(fs["pre_tokenizer"]);
    if (looksLikeSentencePiece)
        return buildSentencePieceTokenizerImpl(fs);

    if (!modelType.empty() && modelType != "BPE")
        CV_Error(cv::Error::StsError,
            "Unsupported tokenizer model type '" + modelType + "' in " + tokJson +
            " (only BPE-family, Unigram and WordPiece models are currently supported)");

    if (model["merges"].empty())
        CV_Error(cv::Error::StsError,
            "tokenizer.json model has no 'merges' table in " + tokJson +
            " (only merge-based BPE models are currently supported)");

    return buildBPETokenizerImpl(fs);
}

Tokenizer::Tokenizer() : impl_(nullptr) {}

std::vector<int> Tokenizer::encode(const std::string& text)
{
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->encode(text);
}

std::vector<int> Tokenizer::encode(const std::vector<std::string>& textChunks)
{
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    CV_CheckFalse(textChunks.empty(), "Tokenizer::encode(): the chunk list must not be empty");
    return impl_->encodeChunks(textChunks);
}

std::string Tokenizer::decode(const std::vector<int>& tokens)
{
    if (!impl_) CV_Error(cv::Error::StsError, "Tokenizer impl null");
    return impl_->decode(tokens);
}

static std::vector<uint8_t> tokenToBytes(const std::string& tokenUtf8)
{
    std::vector<std::uint8_t> out;
    auto cps = unicode_cpts_from_utf8(tokenUtf8);
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        const std::string one = unicode_cpt_to_utf8(cp);
        out.push_back(unicode_utf8_to_byte(one));
    }
    return out;
}

static CoreBPE buildTokenizerFromJson(cv::FileStorage& fs,
                          std::unordered_set<std::string>* outSpecial)
{
    cv::FileNode model = fs["model"];
    CV_CheckFalse(model.empty(), "tokenizer.json missing 'model'");
    cv::FileNode vocab = model["vocab"];
    CV_CheckFalse(vocab.empty(), "tokenizer.json missing model.vocab");

    std::string modelType;
    model["type"] >> modelType;
    if (!modelType.empty() && modelType != "BPE")
        CV_Error(cv::Error::StsError,
            "Expected a BPE model in tokenizer.json, got: " + modelType);

    std::string pattern = detectSplitPattern(fs, modelType);

    std::unordered_set<std::string> skipTokens;
    FileNode addedPeek = fs["added_tokens"];
    if (!addedPeek.empty()) {
        for (auto it = addedPeek.begin(); it != addedPeek.end(); ++it) {
            cv::FileNode t = *it;
            bool isSpecial = false; t["special"]  >> isSpecial;
            std::string content;     t["content"]  >> content;
            if (isSpecial && !content.empty())
                skipTokens.insert(content);
        }
    }

    ByteVecRankMap mergeableRanks;
    mergeableRanks.reserve((size_t)vocab.size());
    int maxId = -1;

    for (cv::FileNodeIterator it = vocab.begin(); it != vocab.end(); ++it) {
        FileNode val = *it;
        std::string token = val.name();
        if (skipTokens.find(token) != skipTokens.end()) continue;
        int id = (int)val;
        mergeableRanks.emplace(tokenToBytes(token), (uint32_t)id);
        if (id > maxId) maxId = id;
    }

    std::unordered_map<std::string, uint32_t> specialTokens;
    FileNode added = fs["added_tokens"];
    if (!added.empty()) {
        for (auto it = added.begin(); it != added.end(); ++it) {
            cv::FileNode t = *it;
            bool special = false; t["special"] >> special;
            int id = -1;          cv::read(t["id"], id, id);
            std::string content;  t["content"] >> content;
            if (special && id >= 0 && !content.empty()) {
                specialTokens.emplace(content, (uint32_t)id);
                if (id > maxId) maxId = id;
                if (outSpecial) outSpecial->insert(content);
            }
        }
    }

    return CoreBPE(std::move(mergeableRanks), std::move(specialTokens), pattern);
}

Tokenizer Tokenizer::load(const std::string& model_config)
{
    // Accepts the model directory or the config.json path itself.
    std::string cfgPath = model_config;
    std::string dir;
    if (model_config.empty() || model_config.back() == '/' || model_config.back() == '\\'
        || utils::fs::isDirectory(model_config))
    {
        dir = model_config;
        if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
            dir += '/';
        cfgPath = dir + "config.json";
    }
    else
    {
        size_t pos = model_config.find_last_of("/\\");
        dir = (pos == std::string::npos) ? std::string() : model_config.substr(0, pos + 1);
    }

    cv::FileStorage cfg(cfgPath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!cfg.isOpened())
        CV_Error(cv::Error::StsError, "Could not open config.json: " + cfgPath);

    std::string methodType = "BPE";
    cv::FileNode methodNode = cfg["method"];
    bool hasMethod = !methodNode.empty() && !methodNode.isNone();
    if (hasMethod)
        methodNode >> methodType;
    // Gemma is byte-fallback BPE under the hood, hence SentencePiece here too.
    static const std::pair<const char*, TokenizerFamily> kFamilies[] = {
        { "BPE",           TokenizerFamily::BPE },
        { "Gemma",         TokenizerFamily::SentencePiece },
        { "SentencePiece", TokenizerFamily::SentencePiece },
        { "Unigram",       TokenizerFamily::Unigram },
        { "WordPiece",     TokenizerFamily::WordPiece },
    };
    // config.json without a "method" key (e.g. stock HF config.json) keeps
    // auto-detecting from tokenizer.json; only an explicit "method" routes.
    TokenizerFamily methodOverride = TokenizerFamily::Auto;
    if (hasMethod) {
        auto methodIt = std::find_if(std::begin(kFamilies), std::end(kFamilies),
                                      [&](const std::pair<const char*, TokenizerFamily>& f) { return methodType == f.first; });
        if (methodIt == std::end(kFamilies))
            CV_Error(cv::Error::StsError,
                "Unsupported tokenizer method: '" + methodType + "'. Supported: BPE, Gemma, SentencePiece, Unigram, WordPiece");
        methodOverride = methodIt->second;
    }

    Tokenizer tok;
    tok.impl_ = buildFromTokenizerDir(dir, methodOverride);
    return tok;
}

CV__DNN_INLINE_NS_END
}}
