// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pcc.h"
#include <queue>
namespace cv{

void EntropyCoder::encodeCharVectorToStream(
    std::vector<unsigned char> &inputCharVector,
    std::ostream &outputStream)
{
    // histogram of char frequency
    std::uint64_t hist[257];

    // partition of symbol ranges from cumulative frequency, define by left index
    std::uint32_t part_idx[257];

    // define range limits
    const std::uint32_t adjust_limit = static_cast<std::uint32_t> (1) << 24;
    const std::uint32_t bottom_limit = static_cast<std::uint32_t> (1) << 16;

    // encoding variables
    std::uint32_t low, range;
    size_t readPos;
    std::uint8_t symbol;

    auto input_size = static_cast<size_t> (inputCharVector.size());

    // output vector ready
    std::vector<unsigned char> outputCharVector_;
    outputCharVector_.clear();
    outputCharVector_.reserve(sizeof(unsigned char) * input_size);


    // Calculate frequency histogram and then partition index of each char
    memset(hist, 0, sizeof(hist));
    readPos = 0;
    while (readPos < input_size) {
        // scan the input char vector to obtain char frequency
        symbol = static_cast<std::uint8_t> (inputCharVector[readPos++]);
        hist[symbol + 1]++;
    }
    part_idx[0] = 0;
    for (int i = 1; i <= 256; i++) {
        if (hist[i] <= 0) {
            // partition must have at least 1 space for each symbol
            part_idx[i] = part_idx[i - 1] + 1;
            continue;
        }
        // partition index is cumulate position when separate a "range"
        // into spaces for each char, space length allocated according to char frequency.
        // "aaaaccbbbbbbbb" -> [__'a'__][____'b'____][_'c'_]...
        // part_idx[i] marks the left bound of char i,
        // while (part_idx[i+1] - 1) marks the right bound.
        part_idx[i] = part_idx[i - 1] + static_cast<std::uint32_t> (hist[i]);
    }

    // rescale if range exceeds bottom_limit
    while (part_idx[256] >= bottom_limit) {
        for (int i = 1; i <= 256; i++) {
            part_idx[i] >>= 1;
            if (part_idx[i] <= part_idx[i - 1]) {
                part_idx[i] = part_idx[i - 1] + 1;
            }
        }
    }

    // Start Encoding

    // the process is to recursively partition a large range by each char,
    // which each char's range defined by part_idx[]
    // the current range is located at left index "low", and "range" record the length

    // range initialize to maximum(cast signed number "-1" to unsigned equal to numeric maximum)
    // initial range is spanned to discrete 32-bit integer that
    // mimics infinitely divisible rational number range(0..1) in theory.
    // the solution is to scale up the range(Renormalization) before
    // recursive partition hits the precision limit.
    readPos = 0;
    low = 0;
    range = static_cast<std::uint32_t> (-1);

    while (readPos < input_size) {
        // read each input symbol
        symbol = static_cast<std::uint8_t>(inputCharVector[readPos++]);

        // map to range
        // first, divide range by part_idx size to get unit length, and get low bound
        // second, get actual range length by multiply unit length with partition space
        // all under coordinate of 32-bit largest range.
        low += part_idx[symbol] * (range /= part_idx[256]);
        range *= part_idx[symbol + 1] - part_idx[symbol];

        // Renormalization
        // first case: range is completely inside a block of adjust_limit
        //      - 1. current range smaller than 2^24,
        //      - 2. further partition won't affect high 8-bit of range(don't go across two blocks)
        // second case: while first case continuously misses and range drops below bottom_limit
        //      - happens when range always coincidentally fall on the border of adjust_limit blocks
        //      - force the range to truncate inside a block of bottom_limit
        // preform resize to bottom_limit(scale up coordinate by 2^8)
        // push 8 bit to output, then scale up by 2^8 to flush them.
        while ((low ^ (low + range)) < adjust_limit ||
               ((range < bottom_limit) && ((range = -int(low) & (bottom_limit - 1)), 1))) {
            auto out = static_cast<unsigned char> (low >> 24);
            range <<= 8;
            low <<= 8;

            outputCharVector_.push_back(out);
        }

    }

    // flush remaining data
    for (int i = 0; i < 4; i++) {
        auto out = static_cast<unsigned char> (low >> 24);
        outputCharVector_.push_back(out);
        low <<= 8;
    }

    const size_t vec_len = inputCharVector.size();

    // write cumulative frequency table to output stream
    outputStream.write(reinterpret_cast<const char *> (&part_idx[0]), sizeof(part_idx));
    // write vec_size
    outputStream.write(reinterpret_cast<const char *> (&vec_len), sizeof(vec_len));
    // write encoded data to stream
    outputStream.write(reinterpret_cast<const char *> (&outputCharVector_[0]), outputCharVector_.size());


}

void EntropyCoder::decodeStreamToCharVector(
    std::istream &inputStream,
    std::vector<unsigned char> &outputCharVector)
{
    // partition of symbol ranges from cumulative frequency, define by left index
    std::uint32_t part_idx[257];

    // define range limits
    const std::uint32_t adjust_limit = static_cast<std::uint32_t> (1) << 24;
    const std::uint32_t bottom_limit = static_cast<std::uint32_t> (1) << 16;

    // decoding variables
    std::uint32_t low, range;
    std::uint32_t code;

    size_t outputPos;
    size_t output_size;

    outputPos = 0;

    // read cumulative frequency table
    inputStream.read(reinterpret_cast<char *> (&part_idx[0]), sizeof(part_idx));
    // read vec_size
    inputStream.read(reinterpret_cast<char *> (&output_size), sizeof(output_size));

    outputCharVector.clear();
    outputCharVector.resize(output_size);

    // read code
    code = 0;
    for (size_t i = 0; i < 4; i++) {
        std::uint8_t out;
        inputStream.read(reinterpret_cast<char *> (&out), sizeof(unsigned char));
        code = (code << 8) | out;
    }

    low = 0;
    range = static_cast<std::uint32_t> (-1);

    // decoding
    for (size_t i = 0; i < output_size; i++) {
        // symbol lookup in cumulative frequency table
        std::uint32_t count = (code - low) / (range /= part_idx[256]);

        // finding symbol by range using Jump search
        std::uint8_t symbol = 0;
        std::uint8_t step = 128;
        while (step > 0) {
            if (part_idx[symbol + step] <= count) {
                symbol = static_cast<std::uint8_t> (symbol + step);
            }
            step /= 2;
        }

        // write symbol to output stream
        outputCharVector[outputPos++] = symbol;

        // map to range
        low += part_idx[symbol] * range;
        range *= part_idx[symbol + 1] - part_idx[symbol];

        // check range limits, reverse Renormalization
        while ((low ^ (low + range)) < adjust_limit ||
               ((range < bottom_limit) && ((range = -int(low) & (bottom_limit - 1)), 1))) {
            std::uint8_t out;
            inputStream.read(reinterpret_cast<char *> (&out), sizeof(unsigned char));
            code = code << 8 | out;
            range <<= 8;
            low <<= 8;
        }

    }
}

void traverse(OctreeNode &root, std::vector<unsigned char> &serializedVectorOut) {
    std::queue<OctreeNode *> nodeQueue;
    nodeQueue.push(&root);
    while (!nodeQueue.empty()) {

        OctreeNode &node = *(nodeQueue.front());
        nodeQueue.pop();

        // Stop at last leaf level, no need to encode leaf node
        if (node.isLeaf) {
            break;
        }

        // Push OctreeNode occupancy code
        serializedVectorOut.push_back(OctreeKey::getBitPattern(node));

        // Further branching
        for (unsigned char i = 0; i < 8; i++) {
            if (!node.children[i].empty()) {
                nodeQueue.push(node.children[i]);
            }
        }
    }

}

void restore(OctreeNode &root,const std::vector<unsigned char> &serializedVectorIn){
    std::queue<OctreeNode *> nodeQueue;
    nodeQueue.push(&root);

    size_t index = 0;
    size_t index_bound = serializedVectorIn.size();
    // Restore tree
    while (!nodeQueue.empty()) {

        OctreeNode &node = *(nodeQueue.front());
        nodeQueue.pop();

        // Octree mode
        if (index >= index_bound) {
            // Restore Leaf level
            node.isLeaf = true;
            OctreeNode *pNode = &node;
            while (pNode != nullptr) {
                ++(pNode->pointNum);
                pNode = pNode->parent;
            }
            continue;
        }
        unsigned char mask = 1;
        unsigned char occup_code = serializedVectorIn[index++];

        for (unsigned char i = 0; i < 8; i++) {
            if (!!(occup_code & mask)) {
                node.children[i] = new OctreeNode(node.depth + 1, 0, Point3f(0, 0, 0),
                                                  Point3f(0, 0, 0), int(i), 0);
                node.children[i]->parent = &node;
                nodeQueue.push(node.children[i]);
            }
            mask = mask << 1;
        }
    }
}

void OctreeSerializeCoder::encode(const std::vector<Point3f> &pointCloud,std::vector<unsigned char> &serializedVector,
                                  double resolution,std::ostream &outputStream) {
    // create octree by pointCloud
    auto start1 = std::chrono::high_resolution_clock::now();
    this->octree->create(pointCloud, resolution);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    printf("Time taken by create: %f\n"
            ,duration1.count()/1e6);

    outputStream<<"origin "<<this->octree->p->origin.x;
    outputStream<<" "<<this->octree->p->origin.y;
    outputStream<<" "<<this->octree->p->origin.z<<"\n";

    // Encode octree by traverse its occupancy code in BFS order.
    auto start2 = std::chrono::high_resolution_clock::now();
    traverse(*(this->octree->p->rootNode),serializedVector);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    printf("Time taken by traverse: %f\n"
            ,duration2.count()/1e6);
}

void OctreeSerializeCoder::decode(const std::vector<unsigned char> &serializedVector,
                                  std::vector<Point3f> &pointCloud,Point3f &origin, double resolution){
    this->octree->clear();
    this->octree->p->origin=origin;
    this->octree->p->resolution=resolution;
    this->octree->p->rootNode=new OctreeNode();
    auto start1 = std::chrono::high_resolution_clock::now();
    restore(*this->octree->p->rootNode,serializedVector);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    printf("Time taken by restore: %f\n"
              ,duration1.count()/1e6);
    std::vector<Point3f> color;

    auto start2 = std::chrono::high_resolution_clock::now();
    this->octree->getPointCloudByOctree(pointCloud,color);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    printf("Time taken by getPointCloudByOctree: %f\n"
            ,duration2.count()/1e6);
}

void PointCloudCompression::compress(const std::vector<Point3f> &pointCloud, double resolution, std::ostream &outputStream) {
    std::vector<unsigned char> serializedVector;
    outputStream<<"resolution "<<resolution<<"\n";
    this->_coder.encode(pointCloud,serializedVector,resolution,outputStream);
    this->_entropyCoder.encodeCharVectorToStream(serializedVector,outputStream);
}

void PointCloudCompression::decompress(std::istream &inputStream, std::vector<Point3f> &pointCloud) {
    std::vector<unsigned char> outputCharVector;
    std::string res,tmp;
    std::getline(inputStream,res);
    inputStream>>tmp;
    double resolution=std::stod(res.substr(11));
    float ori_x,ori_y,ori_z;
    inputStream>>ori_x>>ori_y>>ori_z;
    Point3f origin(ori_x,ori_y,ori_z);
    inputStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    this->_entropyCoder.decodeStreamToCharVector(inputStream,outputCharVector);
    this->_coder.decode(outputCharVector,pointCloud,origin,resolution);
}

}


