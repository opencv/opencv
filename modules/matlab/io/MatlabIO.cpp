#include <ctime>
#include <stringstream>

const char* day[]   = { "Sun", "Mon", "Tue", "Wed", "Thurs", "Fri", "Sat" };
const char* month[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
const char* arch    = "${MEX_ARCH}"

std::string formatCurrentTime() {
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

void MatlabIO::whos() {
  std::cout << "-------------------- whos --------------------" << std::endl;
  std::cout << "Filename: " << filename() << std::endl;
  std::cout << "File size: " << filesize() << "MB" << std::endl << std::endl;
  std::cout << "Name        size         bytes         type" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
