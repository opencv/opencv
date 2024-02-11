// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "pcc.h"
#include "octree.hpp"
#include "zlib.h"
#include <queue>

namespace cv {

// Initialize EntropyCodingMethod(RANGE_CODING_METHOD or ZLIB_METHOD)
EntropyCodingMethod EntropyCoder::codingMethod = EntropyCodingMethod::ZLIB_METHOD;

void EntropyCoder::encodeCharVectorToStream(
        std::vector<unsigned char> &inputCharVector,
        std::ostream &outputStream) {


    if (EntropyCoder::codingMethod == EntropyCodingMethod::RANGE_CODING_METHOD) {
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
    else if (EntropyCoder::codingMethod == EntropyCodingMethod::ZLIB_METHOD){
        z_stream strm;
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;

        if (deflateInit(&strm, Z_DEFAULT_COMPRESSION) != Z_OK) {
            // Handle error
            return;
        }

        strm.avail_in = inputCharVector.size();
        strm.next_in = inputCharVector.data();

        std::vector<unsigned char> compressedBuffer(2 * strm.avail_in); // Initial estimate
        strm.avail_out = compressedBuffer.size();
        strm.next_out = compressedBuffer.data();

        int result = deflate(&strm, Z_FINISH);
        // Handle error
        if (result == Z_BUF_ERROR) {
            deflateEnd(&strm);
            CV_LOG_ERROR(NULL, "Compressed data size exceeded twice the size of the input data.");
            return;
        } else if (result != Z_STREAM_END) {
            CV_LOG_ERROR(NULL, "An error occurred during Zlib compression.");
            return;
        }

        compressedBuffer.resize(strm.total_out);
        // first write current compressed segment size and original vector length
        size_t segmentSize = compressedBuffer.size();
        size_t vec_len = inputCharVector.size();
        outputStream.write(reinterpret_cast<const char*>(&segmentSize), sizeof(size_t));
        outputStream.write(reinterpret_cast<const char *> (&vec_len), sizeof(vec_len));
        // then write the segment
        outputStream.write(reinterpret_cast<char*>(compressedBuffer.data()), compressedBuffer.size());

        deflateEnd(&strm);
    }
    else {
        CV_LOG_ERROR(NULL, "Current EntropyCodingMethod has no implementation");
    }

}

void EntropyCoder::decodeStreamToCharVector(
        std::istream &inputStream,
        std::vector<unsigned char> &outputCharVector) {

    if (EntropyCoder::codingMethod == EntropyCodingMethod::RANGE_CODING_METHOD) {
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
    else if (EntropyCoder::codingMethod == EntropyCodingMethod::ZLIB_METHOD){
        z_stream strm;
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;

        if (inflateInit(&strm) != Z_OK) {
            // Handle error
            return;
        }

        // first read current segment size and vector length
        size_t segmentSize = 0;
        size_t vec_len = 0;
        inputStream.read(reinterpret_cast<char*>(&segmentSize), sizeof(size_t));
        inputStream.read(reinterpret_cast<char *> (&vec_len), sizeof(vec_len));
        // then read the segment
        std::vector<unsigned char> compressedBuffer(segmentSize);
        inputStream.read(reinterpret_cast<char*>(compressedBuffer.data()), segmentSize);

        strm.avail_in = compressedBuffer.size();
        strm.next_in = compressedBuffer.data();

        outputCharVector.clear();
        outputCharVector.resize(vec_len);

        strm.avail_out = outputCharVector.size();
        strm.next_out = outputCharVector.data();

        int result = inflate(&strm, Z_FINISH);
        if (result != Z_STREAM_END) {
            // Handle error
            inflateEnd(&strm);
            outputCharVector.resize(strm.total_out);
            return;
        }

        outputCharVector.resize(strm.total_out);

        inflateEnd(&strm);
    }
    else {
        CV_LOG_ERROR(NULL, "Current EntropyCodingMethod has no implementation");
    }
}

void Haar3DRecursive(OctreeNode *node, std::vector<Point3f> &haarCoefficients, std::vector<OctreeNode *> &cubes,
                     size_t &N) {
    if (!node)
        return;
    if (node->isLeaf) {
        // convert RGB to YUV
        int r, g, b, y, u, v;
        r = (int)node->color.x;
        g = (int)node->color.y;
        b = (int)node->color.z;

        y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

        node->RAHTCoefficient.x = (float)(y > 235 ? 235 : y < 16 ? 16 : y);
        node->RAHTCoefficient.y = (float)(u > 240 ? 240 : u < 16 ? 16 : u);
        node->RAHTCoefficient.z = (float)(v > 240 ? 240 : v < 16 ? 16 : v);
        return;
    }

    for (const auto &child: node->children) {
        Haar3DRecursive(child, haarCoefficients, cubes, N);
    }

    std::vector<OctreeNode *> prevCube(node->children.size());
    std::vector<OctreeNode *> currCube(node->children.size());

    // use the pre-allocated object
    for (size_t idx = 0; idx < node->children.size(); ++idx) {
        prevCube[idx] = cubes[idx];
        currCube[idx] = cubes[node->children.size() + idx];
    }

    // copy node info from octree
    for (size_t idx = 0; idx < node->children.size(); ++idx) {
        if (!node->children[idx]) {
            prevCube[idx]->pointNum = 0;
            continue;
        }
        prevCube[idx]->RAHTCoefficient = node->children[idx]->RAHTCoefficient;
        prevCube[idx]->pointNum = node->children[idx]->pointNum;
    }

    size_t cubeSize = prevCube.size();
    size_t stepSize = 2;

    // start doing transform in x then y then z direction
    while (true) {
        for (size_t x = 0; x < cubeSize; x += stepSize) {
            OctreeNode *node1 = prevCube[x];
            OctreeNode *node2 = prevCube[x + 1];

            if (!node1->pointNum && !node2->pointNum) {
                currCube[x / stepSize]->pointNum = 0;
                continue;
            }

            // transform under this condition
            if (node1->pointNum && node2->pointNum) {
                currCube[x / stepSize] = new OctreeNode;
                auto w1 = (float) node1->pointNum;
                auto w2 = (float) node2->pointNum;
                float w = w1 + w2;
                float a1 = sqrt(w1 / w);
                float a2 = sqrt(w2 / w);

                currCube[x / stepSize]->pointNum = (int) w;

                // YUV
                float YLowPass = a1 * node1->RAHTCoefficient.x + a2 * node2->RAHTCoefficient.x;
                float ULowPass = a1 * node1->RAHTCoefficient.y + a2 * node2->RAHTCoefficient.y;
                float VLowPass = a1 * node1->RAHTCoefficient.z + a2 * node2->RAHTCoefficient.z;

                currCube[x / stepSize]->RAHTCoefficient = Point3f(YLowPass, ULowPass, VLowPass);

                float YHighPass = a1 * node2->RAHTCoefficient.x - a2 * node1->RAHTCoefficient.x;
                float UHighPass = a1 * node2->RAHTCoefficient.y - a2 * node1->RAHTCoefficient.y;
                float VHighPass = a1 * node2->RAHTCoefficient.z - a2 * node1->RAHTCoefficient.z;

                haarCoefficients[N++] = Point3f(YHighPass, UHighPass, VHighPass);
                continue;
            }
            // if no partner to transform, then directly use the value
            currCube[x / stepSize]->pointNum = node1->pointNum ? node1->pointNum : node2->pointNum;
            currCube[x / stepSize]->RAHTCoefficient = node1->pointNum ? node1->RAHTCoefficient
                                                                      : node2->RAHTCoefficient;
        }

        cubeSize >>= 1;
        if (cubeSize < 2)
            break;

        // swap prevCube and currCube
        for (size_t k = 0; k < prevCube.size(); ++k) {
            prevCube[k]->pointNum = currCube[k]->pointNum;
            prevCube[k]->RAHTCoefficient = currCube[k]->RAHTCoefficient;
        }
    }

    // update selected node's coefficient in the octree
    node->RAHTCoefficient = currCube[0]->RAHTCoefficient;
}

void invHaar3DRecursive(OctreeNode *node, std::vector<Point3f> &haarCoefficients, std::vector<OctreeNode *> &cubes,
                        size_t &N, std::vector<Point3f> &colorAttribute) {
    if (!node)
        return;
    if (node->isLeaf) {
        // restore leaf nodes' RGB color
        int c, d, e, r, g, b;

        c = (int)node->RAHTCoefficient.x - 16;
        d = (int)node->RAHTCoefficient.y - 128;
        e = (int)node->RAHTCoefficient.z - 128;

        r = (298 * c + 409 * e + 128) >> 8;
        g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        b = (298 * c + 516 * d + 128) >> 8;

        r = r > 255 ? 255 : r < 0 ? 0 : r;
        g = g > 255 ? 255 : g < 0 ? 0 : g;
        b = b > 255 ? 255 : b < 0 ? 0 : b;

        node->color = Point3f((float)r, (float)g, (float)b);
        colorAttribute.emplace_back((float)r,(float)g,(float)b);
        return;
    }

    // init vector with the input node
    cubes[0]->RAHTCoefficient = node->RAHTCoefficient;

    // actual size of currCube in the loop
    int stepSize, iterSize = 8;

    while (true) {
        stepSize = iterSize >> 1;

        // transformation order:  Octree is abstracted to binary tree
        // 0                      First we have the input node
        // 0       4              Then we fill in weight for its two sub-nodes
        // 0   2   4   6          After that transformation is performed
        // 0 1 2 3 4 5 6 7 8

        // sum weight for the produced temp nodes
        for (int i = 0; i < 8; i += stepSize) {
            cubes[i]->pointNum = 0;
            for (int j = i; j < i + stepSize; ++j) {
                if (node->children[j])
                    cubes[i]->pointNum += node->children[j]->pointNum;
            }
        }


        for (int i = 8 - iterSize; i >= 0; i -= iterSize) {
            Point3f lowPass = cubes[i]->RAHTCoefficient;

            OctreeNode *node1 = cubes[i];
            OctreeNode *node2 = cubes[i + stepSize];

            if (!node1->pointNum && !node2->pointNum)
                continue;
            if (node1->pointNum && node2->pointNum) {
                auto w1 = static_cast<float>(node1->pointNum);
                auto w2 = static_cast<float>(node2->pointNum);
                float w = w1 + w2;

                float a1 = sqrt(w1 / w);
                float a2 = sqrt(w2 / w);

                // get coefficients from input array
                Point3f highPassCoefficient = haarCoefficients[N--];

                // get YUV color
                node1->RAHTCoefficient = a1 * lowPass - a2 * highPassCoefficient;
                node2->RAHTCoefficient = a1 * highPassCoefficient + a2 * lowPass;
                continue;
            }
            node1->RAHTCoefficient = lowPass;
            node2->RAHTCoefficient = lowPass;
        }
        iterSize >>= 1;
        if (iterSize == 1)
            break;
    }

    for (int i = 0; i < 8; ++i)
        if (node->children[i])
            node->children[i]->RAHTCoefficient = cubes[i]->RAHTCoefficient;

    for (int i = 7; i >= 0; --i)
        invHaar3DRecursive(node->children[i], haarCoefficients, cubes, N, colorAttribute);
}


void OctreeSerializeCoder::encodeColor(float qStep, std::vector<unsigned char> &colorCode) {
    OctreeNode root = *this->octree->p->rootNode;
    std::vector<Point3f> haarCoeffs;

    size_t N = 0;

    size_t pointNum = root.pointNum;
    size_t colorNum = 3 * pointNum;

    haarCoeffs.resize(pointNum);
    colorCode.resize(colorNum << 2, '\0');

    std::vector<OctreeNode *> cubes(root.children.size() << 1);
    for (auto &cube: cubes)
        cube = new OctreeNode;

    // Obtain RAHT coefficients through 3D Haar Transform
    Haar3DRecursive(&root, haarCoeffs, cubes, N);
    haarCoeffs[N++] = root.RAHTCoefficient;

    // Init array for quantization
    std::vector<int32_t> qCoeffs(colorNum);

    // Quantization
    for (size_t i = 0; i < N; ++i) {
        qCoeffs[i] = (int32_t) std::round(haarCoeffs[i].x / qStep);
        qCoeffs[N + i] = (int32_t) std::round(haarCoeffs[i].y / qStep);
        qCoeffs[(N << 1) + i] = (int32_t) std::round(haarCoeffs[i].z / qStep);
    }

    // save coefficients to vector for further encoding
    size_t cursor = 0;

    for (auto val: qCoeffs) {
        // skip 0s
        if (!val) {
            cursor += 4;
            continue;
        }
        // signed to unsigned
        val = val > 0 ? (val << 1) : (((-val) << 1) - 1);
        colorCode[cursor++] = static_cast<unsigned char>(val & 0xFF);
        colorCode[cursor++] = static_cast<unsigned char>((val >> 8) & 0xFF);
        colorCode[cursor++] = static_cast<unsigned char>((val >> 16) & 0xFF);
        colorCode[cursor++] = static_cast<unsigned char>((val >> 24) & 0xFF);
    }

    for (auto &p: cubes)
        delete p;
}

void OctreeSerializeCoder::decodeColor(float qStep, const std::vector<unsigned char> &colorCode, std::vector<Point3f> &colorAttribute) {
    // set octree has color
    if (qStep > 0) {
        this->octree->p->hasColor = true;
    }
    OctreeNode root = *this->octree->p->rootNode;
    size_t pointNum = root.pointNum;
    size_t colorNum = 3 * pointNum;
    size_t i, j, k;

    std::vector<int32_t> qCoeffs(colorNum);
    // decode uchar vector
    for (i = 0, j = 0; i < colorNum; ++i) {
        int32_t dVal = *reinterpret_cast<const int32_t*>(&colorCode[j]);
        j += 4;

        // unsigned to signed
        qCoeffs[i] = (dVal & 0x1) ? -(dVal >> 1) - 1 : (dVal >> 1);
    }

    // de-quantization
    std::vector<Point3f> haarCoeffs(pointNum);
    for (i = 0, j = i + pointNum, k = j + pointNum; i < pointNum; ++i, ++j, ++k) {
        haarCoeffs[i].x = static_cast<float>(qCoeffs[i]) * qStep;
        haarCoeffs[i].y = static_cast<float>(qCoeffs[j]) * qStep;
        haarCoeffs[i].z = static_cast<float>(qCoeffs[k]) * qStep;
    }

    size_t N = haarCoeffs.size() - 1;
    root.RAHTCoefficient = haarCoeffs[N--];

    // temp vector for decompression
    std::vector<OctreeNode *> cubes(root.children.size());
    for (auto &cube: cubes)
        cube = new OctreeNode;

    invHaar3DRecursive(&root, haarCoeffs, cubes, N, colorAttribute);

    std::reverse(colorAttribute.begin(), colorAttribute.end());

    for (auto &cube: cubes)
        delete cube;
}

void traverse(OctreeNode &root, std::vector<unsigned char> &serializedVectorOut) {
    std::queue<OctreeNode *> nodeQueue;
    nodeQueue.push(&root);
    while (!nodeQueue.empty()) {

        OctreeNode &node = *(nodeQueue.front());
        nodeQueue.pop();

        // Push OctreeNode occupancy code
        serializedVectorOut.push_back(node.occupancy);

        // Further branching
        unsigned char mask = 1;
        for (unsigned char i = 0; i < 8; ++i, mask <<= 1) {
            if ((node.occupancy & mask) && !node.children[i]->isLeaf) {
                nodeQueue.push(node.children[i]);
            }
        }
    }
}

void restore(OctreeNode &root, const std::vector<unsigned char> &serializedVectorIn, std::vector<Point3f> &pointCloud) {
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
            // output pointcloud
            pointCloud.emplace_back(
                    (float) (node.origin.x) + (float) (node.size * 0.5),
                    (float) (node.origin.y) + (float) (node.size * 0.5),
                    (float) (node.origin.z) + (float) (node.size * 0.5)
                    );
            continue;
        }
        unsigned char mask = 1;
        unsigned char occup_code = serializedVectorIn[index++];

        double childSize = node.size * 0.5;
        for (unsigned char i = 0; i < 8; i++) {
            if (occup_code & mask) {
                // calculate the index and the origin of child.
                size_t xIndex = i&1?1:0;
                size_t yIndex = i&2?1:0;
                size_t zIndex = i&4?1:0;
                Point3f childOrigin = node.origin + Point3f(xIndex * float(childSize), yIndex * float(childSize), zIndex * float(childSize));
                node.children[i] = new OctreeNode(node.depth + 1, childSize,
                                                  childOrigin, Point3f(0,0,0), int(i), 0);
                node.children[i]->parent = &node;
                nodeQueue.push(node.children[i]);
            }
            mask = mask << 1;
        }
    }
}

void OctreeSerializeCoder::encode(const std::vector<Point3f> &pointCloud, const std::vector<Point3f> &colorAttribute,
                             std::vector<unsigned char> &serializedVector, double resolution, std::ostream &outputStream) {
    // create octree by pointCloud & colorAttribute

    this->octree->create(pointCloud, colorAttribute, resolution);

    // set file header.
    outputStream << "origin " << this->octree->p->origin.x;
    outputStream << " " << this->octree->p->origin.y;
    outputStream << " " << this->octree->p->origin.z << "\n";

    outputStream << "maxDepth " << this->octree->p->maxDepth << "\n";

    // encode octree by traverse its occupancy code in BFS order.
    traverse(*(this->octree->p->rootNode), serializedVector);
}

void OctreeSerializeCoder::decode(const std::vector<unsigned char> &serializedVector, std::vector<Point3f> &pointCloud, double resolution, Point3f &origin, size_t maxDepth) {
    this->octree->clear();
    this->octree->p->origin = origin;
    this->octree->p->resolution = resolution;
    this->octree->p->maxDepth = maxDepth;
    double rootSize = (1<<maxDepth)*resolution;
    this->octree->p->rootNode = new OctreeNode( 0, rootSize, origin, Point3f(0,0,0), -1, 0);
    restore(*this->octree->p->rootNode, serializedVector, pointCloud);
}

struct PointCloudCompression::Impl{
    Impl()
    {}

    ~Impl()
    {}

    OctreeSerializeCoder _coder = OctreeSerializeCoder();
    EntropyCoder _entropyCoder = EntropyCoder();
};

PointCloudCompression::PointCloudCompression(): p(new Impl)
{}

void PointCloudCompression::compress(const std::vector<Point3f> &pointCloud, double resolution,
                                     std::ostream &outputStream, const std::vector<Point3f> &colorAttribute, double qStep) {
    std::vector<unsigned char> serializedVector;
    serializedVector.clear();
    serializedVector.reserve(pointCloud.size());

    // refresh coder
    this->p->_coder = OctreeSerializeCoder();

    // check if color attribute exists.
    if (qStep > 0 && colorAttribute.empty()) {
        CV_LOG_WARNING(NULL, "Input pointcloud has no detected color attribute, setting QStep to -1");
        qStep = -1;
    }

    // set file header.
    outputStream << "resolution " << resolution << "\n";
    outputStream << "qstep " << qStep << "\n";

    this->p->_coder.encode(pointCloud, colorAttribute, serializedVector, resolution, outputStream);
    this->p->_entropyCoder.encodeCharVectorToStream(serializedVector, outputStream);

    // encode color if it has color attribute.
    if (qStep > 0) {
        serializedVector.clear();
        this->p->_coder.encodeColor((float) qStep, serializedVector);
        this->p->_entropyCoder.encodeCharVectorToStream(serializedVector, outputStream);
    }
}

void PointCloudCompression::decompress(std::istream &inputStream, std::vector<Point3f> &pointCloud, std::vector<Point3f> &colorAttribute) {
    std::vector<unsigned char> outputCharVector;

    // refresh coder
    this->p->_coder = OctreeSerializeCoder();

    // parse the octree parameters from the file header
    std::string tmp;
    double resolution, qStep;
    inputStream >> tmp >> resolution;
    inputStream >> tmp >> qStep;
    float ori_x, ori_y, ori_z;
    inputStream >> tmp >> ori_x >> ori_y >> ori_z;
    Point3f origin(ori_x, ori_y, ori_z);
    size_t maxDepth;
    inputStream >> tmp >> maxDepth;
    inputStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


    this->p->_entropyCoder.decodeStreamToCharVector(inputStream, outputCharVector);
    this->p->_coder.decode(outputCharVector, pointCloud, resolution, origin, maxDepth);
    outputCharVector.clear();

    // decode color if it has color attribute.
    if (qStep > 0) {
        this->p->_entropyCoder.decodeStreamToCharVector(inputStream, outputCharVector);
        this->p->_coder.decodeColor((float)qStep, outputCharVector, colorAttribute);
    }

    // this->p->_coder.getOctree()->getPointCloudByOctree(pointCloud, colorAttribute);
}
}

