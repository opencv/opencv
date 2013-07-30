#include <ctime>
#include <stringstream>
using namespace std;
using namespace cv;

const char* day[]   = { "Sun", "Mon", "Tue", "Wed", "Thurs", "Fri", "Sat" };
const char* month[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
const char* arch    = "${MEX_ARCH}";

// ----------------------------------------------------------------------------
// BASIC OPERATIONS
// ----------------------------------------------------------------------------
string MatlabIO::filename(void) const { return filename_; }
bool open(const string& filename, ios_base::openmode mode);
  filename_ = filename;
  stream_.open(filename, mode);
}

bool isOpen() const { return stream_.valid(); }
void close() { stream_.close(); }

std::ifstream::pos_type filesize() {
  std::ifstream::pos_type current = stream_.tellg();
  stream_.seekg(0, std::ifstream::end);
  std::ifstream::pos_type end = stream_.tellg();
  stream_.seekg(current, std::ifstream::beg);
  return end;
}

void pushStreamPosition() {
  stream_pos_ = stream_.tellg();
}

void popStreamPosition() {
  stream_.seekg(stream_pos_, std::ifstream::beg);
}

void setStreamPosition(std::ifstream::pos_type position) {
  stream_.seekg(position, std::ifstream::beg);
}

// ----------------------------------------------------------------------------
// HEADERS
// ----------------------------------------------------------------------------


void getFileHeader() {
  // store the current stream position
  pushStreamPosition();
  setStreamPosition(0);
  stream_.read(const_cast<char *>(header_.data()), sizeof(char)*HEADER_LENGTH);
  stream_.read(const_cast<char *>(subsys_.data()), sizeof(char)*SUBSYS_LENGTH);
  stream_.read((char *)&version_, sizeof(int16_t));
  stream_.read(const_cast<char *>(endian_.data()), sizeof(char)*ENDIAN_LENGTH);

  // get the actual version
  if (version_ == 0x0100) version_ = Matlab::IO::Version5;
  if (version_ == 0x0200) version_ = Matlab::IO::Version73;

  // get the endianness
  if (endian_.compare("IM") == 0) byte_swap_ = false;
  if (endian_.compare("MI") == 0) byte_swap_ = true;

  // restore the current stream position
  popStreamPosition();
}

// ----------------------------------------------------------------------------
// INDEXING OPERATIONS
// ----------------------------------------------------------------------------
void MatlabIO::indexNode(const MappingIndex& current) {

}


void MatlabIO::indexNode(const SequenceIndex& current) {

}


void MatlabIO::index() {
  // if there is no open file, do nothing
  if (!isOpen()) return;

  // read the global header
  getFileHeader();

  // change the endianness if need be

  // manually index the top-level node
  MappingIndex root(filename_, vector<size_t>(), stream_.tellg(), filesize(), 0, 0, stream_);
  indexNode(root);
  index_ = root;
}


// ----------------------------------------------------------------------------
// FORMATTING / PRINTING
// ----------------------------------------------------------------------------
template <typename Iterable>
string delimitedStringFromIterable(const Iterable& obj, const string& delimiter=string(" ")) {
  string cache = "";
  ostringstream oss;
  for (Iterable::iterator it = obj.begin(); it != obj.end(); ++it) {
    // flush the cache and insert the next element
    oss << cache << *it;
    cache = delimiter;
  }
  return oss.str();
}


string formatCurrentTime() {
  ostringstream oss;
  time_t rawtime;
  struct tm* timeinfo;
  int dom, hour, min, sec, year;
  // compute the current time
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  // extract the components of interest
  dom  = timeinfo->tm_mday;
  hour = timeinfo->tm_hour;
  min  = timeinfo->tm_min;
  sec  = timeinfo->tm_sec;
  year = timeinfo->year + 1900;
  oss << day[timeinfo->tm_wday] << " " << month[timeinfo->tm_mon] 
      << " " << dom << " " << hour << ":" << min << ":" << sec << " " << year;
  return oss.str();
}


void MatlabIO::printRootIndex() const {
  cout << "--------------- top level file index ------------------" << endl;
  cout << "Filename: " << filename() << endl;
  cout << "File size: " << filesize() << "MB" << endl << endl;
  cout << "Name        size         bytes         type" << endl;
  for (Map<String, Index>::iterator it = index_.mapping.begin(); it != index_.mapping.end(); ++it) {
    cout << it->name << "  ";
    cout << delimitedStringFromIterable(it->size, "x") << "  ";
    cout << it->end - it->begin << "  ";
    cout << endl;
  }
  cout << "-------------------------------------------------------" << endl;
}


void printIndex(const Index& index, const int indentation) {
  cout << string(2*indentation - 1, ' ') << "|" << endl;
  cout << string(2*indentation - 1, ' ') << "|-- ";
  cout << index.name << " (" << delimitedStringFromIterable(index.size, "x") << ")" << endl;
  if (index.leaf) return;
  if (index.associative) {
    for (Map<string, Index>::iterator it = index.mapping.begin(); it != index.mapping.end(); ++it) {
      printIndex(it->second, indentation+1);
    }
  } else {
    for (vector<Index>::iterator it = index.sequence.begin(); it != index.sequence.end(); ++it) {
      printIndex(it->second, indentation+1);
    }
  }
}


void MatlabIO::printFullIndex() const {
  int indentation = 0;
  cout << "----------------- full file index ---------------------" << endl;
  printIndex(index_, indentation);  
  cout << "-------------------------------------------------------" << endl;
}
