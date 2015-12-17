#include <stdio.h>
#include <iostream>

using namespace std; 

int main(){
	bool b[4]; 
	int i = 0xFFFFFFFF;
    *((int*) b) = i; 
    printf("%d %d %d %d\n", b[0], b[1], b[2], b[3] );

	return 0;
}
