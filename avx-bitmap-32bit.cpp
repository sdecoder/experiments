#include <sys/time.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

#define INDICES_LENGTH 4*2500000

class AVXBitmap{
public:
	AVXBitmap(int64_t num_bits){
		size_ = num_bits;
		//bufsize_ = (num_bits + 63) >> 6;
		bufsize_ = (num_bits + 31) >> 5;
		buf = new int[bufsize_];
		msize_ = _mm256_set1_epi32(size_);
		mbufsize_ = _mm256_set1_epi32((long long)bufsize_);
	}
 
	void set_32(int64_t bit_index,bool v){
		bit_index %= size_;
		long long word_index = bit_index >> 5;
		bit_index &= 31;
		if (v){
			buf[word_index] |= (1 << bit_index);
		}
		else{
			buf[word_index] &= ~(1 << bit_index);
		}
	}

 
	bool get_32(int64_t bit_index){
		bit_index %= size_;
		int64_t word_index = bit_index >> 5;
		if (word_index < bufsize_){
			bit_index &= 31;
			return (buf[word_index] & (1 << bit_index)) != 0;
		}
		throw invalid_argument("scalar word index out of bound");
	}

	void get_32(int64_t indices[], bool results[]) const {
		//__m256i bit_index = _mm256_lddqu_si256((const __m256i *)indices);		
 		__m256i bit_index = _mm256_setr_epi32(indices[0] % size_,
      		indices[1] % size_, 
      		indices[2] % size_, 
      		indices[3] % size_,
      		indices[4] % size_,
      		indices[5] % size_,
      		indices[6] % size_,
      		indices[7] % size_);
	
		__m256i word_index = _mm256_srli_epi32(bit_index, 5);
		__m256i check = _mm256_cmpgt_epi32(mbufsize_, word_index);
		// all the 64-bit integers should be 0xFFFFFFFFFFFFFFFF
		if (_mm256_movemask_epi8(check) == -1){
			bit_index = _mm256_and_si256(bit_index, word_mask);
			__m256i words = _mm256_i32gather_epi32(buf, word_index, 4);
			__m256i masks = _mm256_sllv_epi32(mone_, bit_index);
			words = _mm256_and_si256(words, masks);
			__m256i compare = _mm256_cmpeq_epi32(words, masks);
			//*((int *) results) = _mm256_movemask_epi8(compare);
			int mem_addr[8];
			_mm256_storeu_si256 ((__m256i *) mem_addr, compare);
			for(int i = 0; i < 8;i ++) {
				if(mem_addr[i]) results[i] = true; 
				else results[i] = false;
			}
			return;
		}
		throw invalid_argument("packed word index out of bound");
	}
 
	int64_t size(){
		return size_;
	}
private:
	int *buf;
	int64_t size_;
	uint64_t bufsize_;
	__m256i msize_;
	__m256i mbufsize_;
	static const __m256i word_mask;
	static const __m256i mone_;
};

const __m256i AVXBitmap::word_mask = _mm256_set1_epi32(31);
const __m256i AVXBitmap::mone_ = _mm256_set1_epi32(1);


void TestScalarGet_32(AVXBitmap *map, int64_t indices[], bool results[]){
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int j = 0; j < 100; j++){
		for (int i = 0; i < INDICES_LENGTH; i++){
			results[i] = map->get_32(indices[i]);
		}
	}
	gettimeofday(&end, NULL);
	double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	cout << "Scalar get takes " << elapsedTime << " ms" << endl;
}


void TestPackedGet(AVXBitmap *map, int64_t indices[], bool results[]){
	struct timeval start, end;
	int64_t size = map->size();
	gettimeofday(&start, NULL);
	for (int j = 0; j < 100; j++){
		for (int i = 0; i < INDICES_LENGTH; i += 8){
			map->get_32(indices + i, results + i);
		}
	}
	gettimeofday(&end, NULL);
	double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	cout << "Packed get takes " << elapsedTime << " ms" << endl;
}
  
/*
int main(){
	srand((unsigned)time(0));
	AVXBitmap map(32213);
	for (int i = 0; i < 10000; i++){
		map.set(rand(), true);
	}
	int64_t *indices = new int64_t[INDICES_LENGTH];
	for (int i = 0; i < INDICES_LENGTH; i++){
		indices[i] = rand() % map.size();
	}
	bool *scalar_results = new bool[INDICES_LENGTH];
	bool *packed_results = new bool[INDICES_LENGTH];
	for (int i = 0; i < 5; i++){
		cout << "Round " << i + 1 << endl;
		TestPackedGet(&map, indices, packed_results);
		TestScalarGet(&map, indices, scalar_results);
	}
	return 0;
}*/

int64_t *indices = new int64_t[INDICES_LENGTH];
bool *scalar_results = new bool[INDICES_LENGTH];
bool *packed_results = new bool[INDICES_LENGTH];

int main(){
	srand((unsigned)time(0));
	AVXBitmap map(32213);
	for (int i = 0; i < 10000; i++){
		map.set_32(rand(), true);
	}
 	
	for (int i = 0; i < INDICES_LENGTH; i++){
		indices[i] = rand() % map.size();
	}
	 

	for (int i = 0; i < 5; i++){
		cout << "Round " << i + 1 << endl;
		TestPackedGet(&map, indices, packed_results);
		TestScalarGet_32(&map, indices, scalar_results);
		fprintf(stderr, "[dbg] verifying results:\n");
		for (int i = 0; i < INDICES_LENGTH; i++){
			if(packed_results[i] && scalar_results[i] ) continue;
			if(!packed_results[i] && !scalar_results[i] ) continue;
			cout<< "[dbg] diff found at " << i 
			<<": packed_results " << packed_results[i] 
			<<" scalar_results "<< scalar_results[i] << endl;
			fprintf(stderr, "[dbg] FAILURE\n");
			return 0;			
		}
		fprintf(stderr, "[dbg] PASS\n");
	}


	return 0;
}