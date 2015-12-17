#include <time.h>
#include <iostream>
using namespace std; 
int main() {
  time_t ts = 0;
  struct tm t;
  char buf[16];
  ::localtime_r(&ts, &t);
  ::strftime(buf, sizeof(buf), "%z", &t);
  std::cout << "Current timezone (POSIX): " << buf << std::endl;
  ::strftime(buf, sizeof(buf), "%Z", &t);
  std::cout << "Current timezone: " << buf << std::endl;
  return 0; 
}