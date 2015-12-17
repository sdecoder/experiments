#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <immintrin.h>
//#include <xmmintrin.h>
//#include <emmintrin.h>
//#include <pmmintrin.h>

#include <iostream>
#include <vector>



std::vector<uint64_t> buffer_;

using namespace std; 

  template<bool mod>
  inline void Get(int64_t indices[], bool results[]) {
      __m256i word_index ;

     //__m256i words = 
     _mm256_i64gather_epi64(
        reinterpret_cast<const int*>(&buffer_[0]), word_index, 8);
  }




int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}