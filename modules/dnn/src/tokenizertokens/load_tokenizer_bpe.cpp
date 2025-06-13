#include <sstream>
#include <stdexcept>
#include "core_bpe.hpp"

namespace cv { namespace dnn { namespace tokenizer {


// Read a local file or URL into a string.  verify SHA-256.
std::string readFileOrUrl(const std::string& pathOrUrl,
                          const std::string& expextedHash) {
    // TODO: not finihsed 
    // if pathOrUrl starts with [http], do a HTTP GET,
    // otherwise use std::ifstream.  Then, if expectedHash is non-empty,
    // compute SHA-256 of contents and compare.
    // sketch only:
    std::string contents = "....";
    // if (expectedHash != "" && sha256(contents) != expectedHash) throw ...
    return contents;    
}

// Parse a “.tiktoken” BPE file into mergeable_ranks.
ByteVecRankMap loadTokenizerBPE(const std::string& filePath,
                                const std::string& expectedHash) {

    std::string contents = readFileOrUrl(filePath, expectedHash);
    ByteVecRankMap out;
    std::istringstream lines(contents);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        auto sep = line.find(' ');
        auto tokenB64 = line.substr(0, sep);
        auto rankStr = line.substr(sep + 1);
        // TODO: implement funcitonality for base64
        ByteVec tokenBytes = base64Decode(tokenB64);
        Rank rank = static_cast<Rank>(std::stoi(rankStr));
        out.emplace(std::move(tokenBytes), rank);
    }
    return out;
}

ByteVecRankMap dataGymToMergeableBpeRanks(const std::string& vocabBpeFile,
                                          const std::string& encoderJsonFile,
                                          const std::string& vocabBpeHash,
                                          const std::string& encoderJsonHash) {
    
    // TODO; implement logic later
    return {};
}

}}}