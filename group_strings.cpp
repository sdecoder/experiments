#include <stdio.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>
using namespace std; 

class Solution {
public:

	string _convert(string para){
		int delta = para[0] - 'a'; 
		string result = para; 
		for (int i = 0; i < para.size(); ++i)
		{

            result[i] = result[i] - delta; 
            if(result[i] < 'a') result[i] = 'a' + (26  - ('a' - result[i]));
            if(result[i] > 'z') result[i] = 'z' - (26  - ( result[i] - 'z'));
		}
		return result; 


	}

    vector<vector<string> > groupStrings(vector<string>& strings) {
    	const int vec_len = strings.size();
    	map<string, set<string> > dict; 
    	map<string, set<string> >::iterator iter;


    	for (int i = 0; i < vec_len; ++i)
    	{
    		string str_obj = strings[i];
    		string converted_str = _convert(str_obj);
    		if(dict.find(converted_str) == dict.end()){
    			set<string> _str_set; 
    			_str_set.insert(str_obj);
    			dict[converted_str] = _str_set; 
    		}else{
    			set<string> _str_set = dict.find(converted_str)->second;
    			_str_set.insert(str_obj);
                dict[converted_str] = _str_set; 

    		}
    	}

    	vector<vector<string> > result; 

    	for (iter = dict.begin(); iter != dict.end(); iter ++) {
           // cout<<"[dbg] current key: " << iter->first << endl ;
    		set<string> _str_set = iter->second; 
    		set<string>::iterator _ss_iter = _str_set.begin();
    		std::vector<string> _temp;
    		for (; _ss_iter != _str_set.end(); ++_ss_iter)
    		{
              //  cout<< *_ss_iter <<" " ;
    			_temp.push_back(*_ss_iter); 
    		} //cout<< endl; 
			result.push_back(_temp);
		}

		return result; 
        
    }
};

int main(int argc, char const *argv[])
{

    std::vector<string> input_vec;
    // ["abc", "bcd","adef", "xyz", "az", "ba", "a", "z"], 
    input_vec.push_back("abc");
    input_vec.push_back("bcd");
    input_vec.push_back("adef");
    input_vec.push_back("xyz");
    input_vec.push_back("az");
    input_vec.push_back("ba");
    input_vec.push_back("a");
    input_vec.push_back("z");

    Solution s; 

    std::vector<std::vector<string> > result = s.groupStrings(input_vec);
    for (int i = 0; i < result.size(); ++i)
    {
        cout<< "Idx " << i << ":" << endl; 
        for (int j = 0; j < result[i].size(); ++j){
            cout<<result[i][j] << " "; 
        }cout<< endl; 
    }



    
    return 0;
}