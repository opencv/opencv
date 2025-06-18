#include <string>
#include "core_bpe.hpp" 

namespace cv { namespace dnn { namespace tokenizer {

/**
 * Read an entire file or HTTP/HTTPS URL into a binary string.
 * If @p expectedHash is non‑empty, the function computes the SHA‑256 of the
 * contents and throws std::runtime_error if it doesn’t match.
 */
std::string readFileOrUrl(const std::string& pathOrUrl,
                          const std::string& expectedHash = "");

/**
 * Parse a modern “.tiktoken” merge table (Base‑64 + rank per line) and return
 * a mapping ByteVec -> Rank.  If @p expectedHash is provided, the file’s
 * SHA‑256 is verified first.
 */
ByteVecRankMap loadTokenizerBPE(const std::string& filePath,
                                const std::string& expectedHash = "");

/**
 * Reconstruct GPT‑2’s mergeable_ranks from the original data‑gym files
 * (encoder.json + vocab.bpe) and return the complete mapping.
 * The two SHA‑256 hashes are mandatory for network fetches but may be empty
 * if you’re loading from a trusted local path.
 */
ByteVecRankMap dataGymToMergeableBpeRanks(const std::string& vocabBpeFile,
                                          const std::string& encoderJsonFile,
                                          const std::string& vocabBpeHash = "",
                                          const std::string& encoderJsonHash = "");

}}}