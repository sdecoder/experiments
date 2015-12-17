
#include "boost/date_time/posix_time/posix_time.hpp" 
#include "boost/date_time/gregorian/gregorian.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>


using namespace std; 
using namespace boost::gregorian;
using namespace boost::posix_time;

int main(int argc, char const *argv[])
{


  std::string ts1("2008-08-15 02:08:13");
  ptime t1(time_from_string(ts1));
  std::string ts2("2008-08-11 09:17:02");
  ptime t2(time_from_string(ts2));
  if(t1 > t2) cout<< "1" << endl;   
  else if(t1 == t2) cout<< "0" << endl; 
  else cout<<"-1" << endl; 
      //   printf("[printf]result is %.9Lf\n", result );

 return 0;
}





