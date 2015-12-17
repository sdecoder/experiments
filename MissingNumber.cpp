#include <stdio.h>
#include <vector>
#include <iostream>
using namespace std; 

class Solution {
public:
    int missingNumber(vector<int>& nums) {
    	int N = nums.size();
    	long expected_sum = (1+N)*N/2;
    	long real_sum = 0; 
    	for(int i = 0; i < N; i ++){
    		real_sum += nums[i];
    	}

    	return expected_sum - real_sum;
        
    }
};

Solution s; 

int main(int argc, char const *argv[])
{
	vector<int> input; 
	input.push_back(0);
	input.push_back(1);
	input.push_back(2);
	input.push_back(4);
	cout << "Result: " << s.missingNumber(input) << endl; 
	return 0;
}