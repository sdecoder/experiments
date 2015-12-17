class Solution {
public:
	int nthUglyNumber(int n)
	{

	  if(n <= 0)
	      return 0;

	  int *pUglyNumbers = new int[n];
	  pUglyNumbers[0] = 1;
	  int nextUglyIndex = 1;
	  int *pMultiply2 = pUglyNumbers;
	  int *pMultiply3 = pUglyNumbers;
	  int *pMultiply5 = pUglyNumbers;
	  while(nextUglyIndex < n)
	  {

	      int min = Min(*pMultiply2 * 2, *pMultiply3 * 3, *pMultiply5 * 5);
	      pUglyNumbers[nextUglyIndex] = min;
	      while(*pMultiply2 * 2 <= pUglyNumbers[nextUglyIndex])
	          ++pMultiply2;

	      while(*pMultiply3 * 3 <= pUglyNumbers[nextUglyIndex])
	          ++pMultiply3;

	      while(*pMultiply5 * 5 <= pUglyNumbers[nextUglyIndex])
	          ++pMultiply5;
	      ++nextUglyIndex;

	  }

	  int ugly = pUglyNumbers[nextUglyIndex - 1];
	  delete[] pUglyNumbers;
	  return ugly;

	}

	int Min(int number1, int number2, int number3)
	{

	 int min = (number1 < number2) ? number1 : number2;
	 min = (min < number3) ? min : number3;
	 return min;

	}
   
};

int main(int argc, char const *argv[])
{
	
	
	return 0;
}


