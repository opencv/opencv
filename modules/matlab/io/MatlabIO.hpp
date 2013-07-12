#ifndef MATLAB_IO_HPP_
#define MATLAB_IO_HPP_

#include <opencv2/core.hpp>
#include "map.hpp"

namespace Matlab {
  namespace IO {
    static const int VERSION_5  = 5;
    static const int VERSION_73 = 73;
  }

class Index {
private:
  //! the name of the field (if associative container)
  std::string name_;
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
  //! the descendents of this node
  union {
    //! valid if the container is a sequence (list)
    std::vector<Index> sequence_;
    //! valid if the container is a mapping (associative)
    Map<std::string, Index> mapping_;
  };

};

class MatlabIONode {

};

class MatlabIO {
private:
  // member variables
  static const int HEADER_LENGTH = 116;
  static const int SUBSYS_LENGTH = 8;
  static const int ENDIAN_LENGTH = 2;
  char header_[HEADER_LENGTH+1];
  char subsys_[SUBSYS_LENGTH+1];
  char endian_[ENDIAN_LENGTH+1];
  int version_;
  bool byte_swap_;
  std::string filename_;
  // uses a custom stream buffer for fast memory-mapped access and endian swapping
  std::fstream stream_;
  //! the main file index. The top-level index must be associative
  Index index_;

  // internal methods
  void getFileHeader();
  void setFileHeader();

  void getHeader();
  void setHeader();

public:
  // construct/destruct
  MatlabIO() {}
  ~MatlabIO {}

  // global read and write routines
  std::string filename(void);
  bool open(const std::string& filename, const std::string& mode);

  // index the contents of the file
  void index();

  // print all of the top-level variables in the file
}
#endif
