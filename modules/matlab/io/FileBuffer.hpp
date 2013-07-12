#ifndef OPENCV_FILEBUFFER_HPP_
#define OPENCV_FILEBUFFER_HPP_

#include <vector>
#include <streambuf>

class EndianFileBuffer : public std::streambuf {
private:
  const int fd_;
  const size_t put_back_;
  std::vector<char> buffer_;

  // prevent copy construction
  EndianFileBuffer(const EndianFileBuffer&);
  EndianFileBuffer& operator=(const EndianFileBuffer&);

public:
  explicit EndianFileBuffer(int fd, size_t buffer_sz, size_t put_back) :
      fd_(fd), put_back_(max(put_back, 1)), buffer_(max(buffer_sz, put_back_) + put_back_) {
    char *end = &buffer_.front() + buffer_.size();
    setg(end, end, end);
  }

  std::streambuf::int_type underflow() {
    if (gptr() < egptr()) // buffer not exhausted
      return traits_type::to_int_type(*gptr());

    char *base  = &buffer_.front();
    char *start = base;

    if (eback() == base) { // true when this isn't the first fill
      std::memmove(base, egptr() - put_back_, put_back_);
      start += put_back_;
    }

    // start is now the start of the buffer
    // refill from the file
    read(fd_, start, buffer_.size() - (start - base));
    if (n == 0) return traits_type::eof();

    // set buffer pointers
    setg(base, start, start + n);
    return traits_type::to_int_type(*gptr());
  }

};

#endif
