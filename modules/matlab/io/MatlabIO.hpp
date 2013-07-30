#ifndef MATLAB_IO_HPP_
#define MATLAB_IO_HPP_

#include <sstream>
#include <fstream>
#include <zlib.h>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include "primitives.hpp"
#include "bridge.hpp"
#include "mxarray.hpp"
#include "map.hpp"

namespace Matlab {
  namespace IO {
    class RandomAccessRead {};
    class SequentialWrite {};
    static const int Version5  = 5;
    static const int Version73 = 73;
  }

// predeclarations
class IONode;
class IONodeIterator;
class MatlabIO;


// ----------------------------------------------------------------------------
// FILE AS A DATA STRUCTURE
// ----------------------------------------------------------------------------
class IONode {
protected:
  //! the name of the field (if associative container)
  std::string name_;
  //! the size of the field
  std::vector<size_t> size_;
  //! beginning of the data field in the file
  size_t begin_;
  //! address after the last data field
  size_t end_;
  //! Matlab stored-type
  int stored_type_;
  //! Matlab actual type (sometimes compression is used)
  int type_;
  //! is the field compressed?
  bool compressed_;
  //! are the descendents associative (mappings)
  bool associative_;
  //! is this a leaf node containing data, or an interior node
  bool leaf_;
  //! the data stream from which the file was indexed
  cv::Ptr<std::istream> stream_;
  //! valid if the container is a sequence (list)
  std::vector<IONode> sequence_;
  //! valid if the container is a mapping (associative)
  Map<std::string, IONode> mapping_;
  IONode(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, bool compressed, bool associative, bool leaf, istream& stream) :
    name_(name), size_(size), begin_(begin), end_(end), stored_type_(stored_type), type_(type),
    compressed_(compressed), associative_(associative), leaf_(leaf), stream_(stream) {}
public:
  std::string name() const { return name_; }
  std::vector<size_t> size() const { return size_; }
  size_t begin() const { return begin_; }
  size_t end() const { return end_; }
  int stored_type() const { return stored_type_; }
  int type() const { return type_; }
  bool compressed() const { return compressed_; }
  bool associative() const { return associative_; }
  bool leaf() const { return leaf_; }
  IONode() : begin_(0), end_(0), stored_type_(0), type_(0), leaf_(true) {}

#if __cplusplus >= 201103L
  // conversion operators
  template <typename T> void operator=(const T& obj) { static_assert(0, "Unimplemented specialization for given type"); }
  template <typename T> operator T() { static_assert(0, "Unimplemented specialization for given type"); }
#else
  // conversion operators
  template <typename T> void operator=(const T& obj) { T::unimplemented_specialization; }
  template <typename T> operator T() { T::unimplemented_specialization; }
#endif

  void swap(const IONode& other) {
    using std::swap;
    swap(name_, other.name_);
    swap(size_, other.size_);
    swap(begin_, other.begin_);
    swap(end_, other.end_);
    swap(stored_type_, other.stored_type_);
    swap(type_, other.type_);
    swap(compressed_, other.compressed_);
    swap(associative_, other.associative_);
    swap(leaf_, other.leaf_);
    swap(stream_, other.stream_);
    swap(sequence_, other.sequence_);
    swap(mapping_, other.mapping_);
  }
};

class SequenceIONode : public IONode {
public:
  std::vector<IONode>& sequence() { return sequence_; }
  SequenceIONode(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, const std::istream& stream) : 
    IONode(name, size, begin, end, stored_type, type, false, false, false, stream) {}
};

class MappingIONode : public IONode {
public:
  Map<std::string, IONode>& mapping() { return mapping_; }
  MappingIONode(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, const std::istream& stream) : 
    IONode(name, size, begin, end, stored_type, type, false, true, false, stream) {}
};

class LeafIONode : public IONode {
  LeafIONode(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, const std::istream& stream) : 
    IONode(name, size, begin, end, stored_type, type, false, false, true, stream) {}
};

class CompressedIONode : public IONode {
private:
  std::istringstream uncompressed_stream_;
  std::vector<char> data_;
public:
  CompressedIONode(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, const std::stream& stream) : 
    IONode(name, size, begin, end, stored_type, type, true, false, false, stream) {}
};

class Header : public IONode {
  Header(const std::string& name, const std::Vector<size_t>& size, size_t begin, size_t end, 
        int stored_type, int type, const std::stream& stream) : 
    IONode(name, size, begin, end, stored_type, type, true, false, false, stream) {}
  



// ----------------------------------------------------------------------------
// FILE NODE
// ----------------------------------------------------------------------------
class IONodeIterator : public std::iterator<std::random_access_iterator_tag, MatlabIONode> {

};



// ----------------------------------------------------------------------------
// MATLABIO
// ----------------------------------------------------------------------------
class MatlabIO {
private:
  // member variables
  static const int HEADER_LENGTH = 116;
  static const int SUBSYS_LENGTH = 8;
  static const int ENDIAN_LENGTH = 2;
  std::string header_;
  std::string subsys_;
  std::string endian_;
  int version_;
  bool byte_swap_;
  std::string filename_;
  // uses a custom stream buffer for fast memory-mapped access and endian swapping
  std::fstream stream_;
  std::ifstream::pos_type stream_pos_;
  //! the main file index. The top-level index must be associative
  IONode index_;

  // internal methods
  void getFileHeader();
  void setFileHeader();

  void getHeader();
  void setHeader();

  CompressedIONode uncompress(const IONode& node);

public:
  // construct/destruct
  MatlabIO() : header_(HEADER_LENGTH+1, '\0'), subsys_(SUBSYS_LENGTH+1, '\0'), 
               endian_(ENDIAN_LENGTH+1, '\0'), byte_swap(false), stream_pos_(0) {}
  ~MatlabIO {}

  // global read and write routines
  std::string filename(void);
  bool open(const std::string& filename, std::ios_base::openmode mode);
  bool isOpen() const;
  void close(); 
  void clear();

  // index the contents of the file
  void index();

  // print all of the top-level variables in the file
  void printRootIndex() const;
  void printFullIndex() const;

  // FileNode operations
  IONode root() const;
  IONode operator[](const String& nodename) const;
};

#endif
