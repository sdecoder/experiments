
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#include <iostream>
using namespace std; 


int getTimeOffset()
{
FILE *file;
int hr1, hr2, rc;
char buffer[64];
system ("date +%H >/tmp/TZ");
system ("date -u +%H >>/tmp/TZ");

file = fopen ("/tmp/TZ", "rb");
fgets (buffer, sizeof(buffer), file);
hr1 = atoi(buffer);
fgets (buffer, sizeof(buffer), file);
hr2 = atoi (buffer);
rc = fclose (file);
rc = unlink ("/tmp/TZ");

/* printf("H1=%d,H2=%d\n",hr1,hr2); */
if (hr2 > hr1)
return (hr2 - hr1);
else
hr1 = -((24 - hr1) + hr2);
return (hr1);
}

int main(int argc, char const *argv[])
{
	/* code */
	int  result = getTimeOffset();
	cout<< "find the time zone: " << result << endl; 
	return 0;
}