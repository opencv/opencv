#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// Function to print EXIF entries in a readable format
static void dumpExif(const std::vector<std::vector<ExifEntry>>& exif_entries, const std::string& title = "EXIF DUMP")
{
    cout << "=== " << title << " ===" << endl;
    for (const auto& block : exif_entries)
    {
        cout << "----------------------" << endl;
        // Dump each entry using ExifEntry's dump method
        for (const auto& entry : block)
        {
            std::cout << entry.dumpAsString() << std::endl;
        }
    }
}

// Function to extract EXIF metadata block from metadata
static std::vector<uchar> getExifBlock(const std::vector<int>& metadata_types, const std::vector<std::vector<uchar>>& metadata)
{
    for (size_t i = 0; i < metadata_types.size(); i++)
    {
        if (metadata_types[i] == IMAGE_METADATA_EXIF)
        {
            return metadata[i];  // Return EXIF buffer
        }
    }
    return {};  // Return empty vector if no EXIF found
}

// Function to read, print, re-encode, and verify EXIF metadata
static void printExif(const std::string& path)
{
    cv::Mat img;
    std::vector<int> metadata_types;          // Holds metadata type codes
    std::vector<std::vector<uchar>> metadata; // Holds raw metadata buffers

    // Load image with metadata
    img = cv::imreadWithMetadata(path, metadata_types, metadata);
    if (img.empty())
    {
        cout << "Failed to load image: " << path << endl;
        return;
    }

    // Extract raw EXIF data
    std::vector<uchar> raw_exif = getExifBlock(metadata_types, metadata);
    if (raw_exif.empty())
    {
        cout << "No EXIF metadata found" << endl;
        return;
    }

    // Decode EXIF into structured entries
    std::vector<std::vector<ExifEntry>> exif_entries;
    bool result = cv::decodeExif(raw_exif, exif_entries);
    if (!result)
    {
        cout << "Parsing EXIF raw data failed" << endl;
        return;
    }

    // Print original EXIF
    dumpExif(exif_entries, "ORIGINAL EXIF");

    // Encode image in-memory with original metadata
    std::vector<uchar> buf;
    cv::imencodeWithMetadata(".jpg", img, metadata_types, metadata, buf);

    // Re-encode EXIF from structured entries
    std::vector<uchar> raw_exif_data;
    bool res = cv::encodeExif(exif_entries, raw_exif_data);
    if (!res)
    {
        cout << "Failed to encode EXIF" << endl;
        return;
    }

    // Create new metadata vector with re-encoded EXIF
    std::vector<std::vector<uchar>> new_metadata;
    for (size_t i = 0; i < metadata_types.size(); i++)
    {
        if (metadata_types[i] == IMAGE_METADATA_EXIF)
            new_metadata.push_back(raw_exif_data); // Replace EXIF
        else
            new_metadata.push_back(metadata[i]);   // Keep other metadata unchanged
    }

    // Encode image again with new metadata
    cv::imencodeWithMetadata(".jpg", img, metadata_types, new_metadata, buf);

    // Decode the image from memory to verify metadata
    std::vector<int> metadata_types2;
    std::vector<std::vector<uchar>> metadata2;
    cv::Mat img2 = cv::imdecodeWithMetadata(buf, metadata_types2, metadata2);

    // Extract EXIF from reloaded image
    std::vector<uchar> raw_exif2 = getExifBlock(metadata_types2, metadata2);
    if (!raw_exif2.empty())
    {
        std::vector<std::vector<ExifEntry>> exif_entries2;
        if (cv::decodeExif(raw_exif2, exif_entries2))
        {
            // Print reloaded EXIF
            dumpExif(exif_entries2, "RELOADED EXIF");
        }
    }
}

int main(int argc, const char** argv)
{
    if (argc > 1)
        printExif(argv[1]);  // Process file from command line argument
    else
        cout << "usage : " << argv[0] << " filename";  // Print usage if no file given

    return 0;
}
