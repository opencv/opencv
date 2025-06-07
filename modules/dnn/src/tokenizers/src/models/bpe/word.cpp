#include "word.hpp"

#include <algorithm>
#include <queue>
#include <random>

namespace cv { namespace dnn { namespace tokenizer {

struct MergeEntry {
    std::size_t pos;        // index of left-hand symbol in symbols
    std::uint32_t rank;     // lower rank -> merge earlier
    std::uint32_t newId;    // resulting token id after merging
};

struct MergeCmp {
    // get a min‑heap on rank, then on position.
    bool operator()(const MergeEntry &a, const MergeEntry &b) const noexcept {
        return (a.rank == b.rank) ? (a.pos > b.pos) : (a.rank > b.rank);
    }
};

void Word::add(std::uint32_t c, std::size_t byte_len) {
    std::ptrdiff_t len = static_cast<std::ptrdiff_t>(symbols.size());
    std::ptrdiff_t prev = -1;
    if (!symbols.empty()) {
        symbols.back().next = len;  // wire-up next from previous symbol
        prev = len - 1;
    }
    symbols.push_back(Symbol{c, prev, -1, byte_len});
}

std::vector<std::pair<Pair, int>> Word::merge(std::uint32_t c1, 
                                            std::uint32_t c2, 
                                            std::uint32_t replacement,
                                            std::size_t max_length) {

    std::vector<std::pair<Pair, int>> delta;
    std::size_t i = 0;
    while (i < symbols.size()) {
        if (symbols[i].c == c1 && i + 1 < symbols.size() && symbols[i+1].c == c2) {
            const Symbol first = symbols[i];
            const Symbol second = symbols[i+1];

            Symbol merged {
                replacement,
                first.prev,
                second.next,
                first.len + second.len
            };

            // left context updates
            if (i > 0) {
                delta.emplace_back(Pair{symbols[i-1].c, first.c}, -1);
                if (symbols[i-1].len + merged.len < max_length)
                    delta.emplace_back(Pair{symbols[i-1].c, replacement}, 1);
            }

            // perfrom in-place replacement 
            symbols.insert(symbols.begin() + i, merged);
            symbols.erase(symbols.begin() + i + 1, symbols.begin() + i + 3);

            // right context updates
            if (i < symbols.size()  - 1) {
                delta.emplace_back(Pair{second.c, symbols[i+1].c}, -1);
                if (symbols[i+1].len + merged.len < max_length) 
                    delta.emplace_back(Pair{replacement, symbols[i+1].c}, 1);
            }
        }
        ++i;
    }
    return delta;
}

void Word::mergeAll(const std::unordered_map<Pair, std::pair<std::uint32_t, std::uint32_t>, PairHash>& merges,
                        std::optional<float> dropout) {
    using PQ = std::priority_queue<MergeEntry, std::vector<MergeEntry>, MergeCmp>;
    PQ queue;
    queue = PQ(MergeCmp(), {});

    // seed priority queue with all valid adjacent pairs
    for (std::size_t idx = 0; idx + 1 < symbols.size(); ++idx) {
        Pair p{symbols[idx].c, symbols[idx + 1].c};
        auto it = merges.find(p);
        if (it != merges.end()) {
            queue.push(
                MergeEntry{
                    idx,
                    it->second.first,
                    it->second.second
                }
            );
        }
    }

    std::vector<MergeEntry> skipped; 
    skipped.reserve(queue.size());
    
}


}}}