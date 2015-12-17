#include <immintrin.h>
#include <stdint.h>
#include <vector>
using namespace std; 
std::vector<uint64_t> buffer_;

 __m256i mbuffer_size;
__m256i word_index ;
int main(int argc, char const *argv[])
{
 
    __m256i words = _mm256_i64gather_epi64(reinterpret_cast<const int*>(&buffer_[0]), word_index, 8);
	return 0;
}