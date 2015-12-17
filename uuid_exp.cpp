#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <uuid/uuid.h>

#include <iostream>
#include <string>
using namespace std; 

int main(int argc, char** argv) {
  //generator random uuid;
  uuid_t uuid;
  char str[1000];
  uuid_generate_random(uuid);
  uuid_unparse(uuid, str);
  cout<< "[dbg] result uuid: " << str << endl;
  cout<<"[dbg] string len: " << strlen(str) << endl;
  return 0;
}

