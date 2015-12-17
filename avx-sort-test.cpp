

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#include <immintrin.h>
#include <xmmintrin.h>
#include <sys/time.h>
  

#include <algorithm>
#include <iostream>
using namespace std; 
 

static void inline bitonic_merge_kernel(__m256 *a, __m256 *b);
static void inline bitonic_merge_kernel(__m256 *a, __m256 *b, int flag);
static void inline bitonic_merge_kernel(__m256 *a, __m256 *b, int flag, int flag2);
static void inline bitonic_sort_16elems(float *ret, float* list);
static bool verify(float* offset, int length);

static void inspect_array(float * start, int offset){
    cout<<"[dbg] inspect_array" << endl;
    for (int i = 0; i < offset; ++i)
    {
        cout<<*(start + i) << endl; 
    }

}

void inspect(__m128 v)
{
    float f[4];
    _mm_storeu_ps(f, v);
    printf("[%.1f,%.1f,%.1f,%.1f]\n", f[0], f[1], f[2], f[3]);   
}

void inspect(__m256 v)
{
    float f[8];
    _mm256_storeu_ps(f, v);
    printf("[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n", 
        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}

void inspectLH(__m128 l, __m128 h)
{
    float f1[4];
    float f2[4];
    _mm_storeu_ps(f1, l);
    _mm_storeu_ps(f2, h);    
    printf("[L0:%.1f H1:%.1f]\n", f1[0], f2[0]);
    printf("[L1:%.1f H1:%.1f]\n", f1[1], f2[1]);
    printf("[L2:%.1f H2:%.1f]\n", f1[2], f2[2]); 
    printf("[L3:%.1f H3:%.1f]\n", f1[3], f2[3]);
}


void inspectLH(__m256 l, __m256 h)
{
    float f1[8];
    float f2[8];
    _mm256_storeu_ps(f1, l);
    _mm256_storeu_ps(f2, h);    
    printf("[L0:%f H1:%f]\n", f1[0], f2[0]);
    printf("[L1:%f H1:%f]\n", f1[1], f2[1]);
    printf("[L2:%f H2:%f]\n", f1[2], f2[2]); 
    printf("[L3:%f H3:%f]\n", f1[3], f2[3]);
    printf("[L4:%f H4:%f]\n", f1[4], f2[4]);
    printf("[L5:%f H5:%f]\n", f1[5], f2[5]);
    printf("[L6:%f H6:%f]\n", f1[6], f2[6]); 
    printf("[L7:%f H7:%f]\n", f1[7], f2[7]);
}

 
inline void transpose8_ps(
    __m256 &row0, __m256 &row1, 
    __m256 &row2, __m256 &row3, 
    __m256 &row4, __m256 &row5, 
    __m256 &row6, __m256 &row7) {
__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
__t0 = _mm256_unpacklo_ps(row0, row1);
__t1 = _mm256_unpackhi_ps(row0, row1);
__t2 = _mm256_unpacklo_ps(row2, row3);
__t3 = _mm256_unpackhi_ps(row2, row3);
__t4 = _mm256_unpacklo_ps(row4, row5);
__t5 = _mm256_unpackhi_ps(row4, row5);
__t6 = _mm256_unpacklo_ps(row6, row7);
__t7 = _mm256_unpackhi_ps(row6, row7);
__tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
__tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
__tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
__tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
__tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
__tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
__tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
__tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}



/*void inline _COMPARE_TWO(__m256 &x,__m256 &y){
    __m256 min = _mm256_min_ps(x, y); 
    __m256 max = _mm256_max_ps(x, y);  
    x = min; y = max; 
} */    
   

static void
bitonic_sort_64elems(float *ret, float* list)
{
 
    __m256 x[8], min, max; 
    int i; 

 #define _COMPARE_TWO_AVX(x,y)                        \
    min = _mm256_min_ps(x, y);                     \
    max = _mm256_max_ps(x, y);                     \
    x = min; y = max;

    for(i = 0;i < 8;i++){ 
        x[i] = _mm256_loadu_ps(list + 8 * i); 
    } 

    _COMPARE_TWO_AVX(x[0], x[1]); 
    _COMPARE_TWO_AVX(x[0], x[2]); 
    _COMPARE_TWO_AVX(x[0], x[3]); 
    _COMPARE_TWO_AVX(x[0], x[4]); 
    _COMPARE_TWO_AVX(x[0], x[5]); 
    _COMPARE_TWO_AVX(x[0], x[6]); 
    _COMPARE_TWO_AVX(x[0], x[7]); 


    _COMPARE_TWO_AVX(x[1], x[2]); 
    _COMPARE_TWO_AVX(x[1], x[3]); 
    _COMPARE_TWO_AVX(x[1], x[4]); 
    _COMPARE_TWO_AVX(x[1], x[5]); 
    _COMPARE_TWO_AVX(x[1], x[6]); 
    _COMPARE_TWO_AVX(x[1], x[7]); 

    _COMPARE_TWO_AVX(x[2], x[3]); 
    _COMPARE_TWO_AVX(x[2], x[4]); 
    _COMPARE_TWO_AVX(x[2], x[5]); 
    _COMPARE_TWO_AVX(x[2], x[6]); 
    _COMPARE_TWO_AVX(x[2], x[7]); 

    _COMPARE_TWO_AVX(x[3], x[4]); 
    _COMPARE_TWO_AVX(x[3], x[5]); 
    _COMPARE_TWO_AVX(x[3], x[6]); 
    _COMPARE_TWO_AVX(x[3], x[7]); 

    _COMPARE_TWO_AVX(x[4], x[5]); 
    _COMPARE_TWO_AVX(x[4], x[6]); 
    _COMPARE_TWO_AVX(x[4], x[7]);

    _COMPARE_TWO_AVX(x[5], x[6]); 
    _COMPARE_TWO_AVX(x[5], x[7]); 

    _COMPARE_TWO_AVX(x[6], x[7]); 

  
    transpose8_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);

 
    
   // bitonic_merge_kernel(&x[0], &x[1] ,1 ,1); 
   // fprintf(stderr, "%d\n", __LINE__); inspectLH(x[0], x[1]);
    bitonic_merge_kernel(&x[0], &x[1]);  
   // fprintf(stderr, "%d\n", __LINE__); inspectLH(x[0], x[1]);
    bitonic_merge_kernel(&x[0], &x[2]); 
    bitonic_merge_kernel(&x[0], &x[3]); 
    bitonic_merge_kernel(&x[1], &x[2]); 
    bitonic_merge_kernel(&x[1], &x[3]); 
    bitonic_merge_kernel(&x[2], &x[3]); 

 
    bitonic_merge_kernel(&x[4], &x[5]); 
    bitonic_merge_kernel(&x[4], &x[6]); 
    bitonic_merge_kernel(&x[4], &x[7]); 
    bitonic_merge_kernel(&x[5], &x[6]); 
    bitonic_merge_kernel(&x[5], &x[7]); 
    bitonic_merge_kernel(&x[6], &x[7]); 
 
    
    // puts("----"); 
 
    for (int i = 0; i < 8; ++i)
    {
        _mm256_storeu_ps(list + i*8, x[i]); 
    }


}
 

static void inline
odd_merge_in_register_sort(__m128 *x0, __m128 *x1, __m128 *x2, __m128 *x3)
{
    __m128 min, max;
#define COMPARE_TWO(x,y)                        \
    min = _mm_min_ps(x, y);                     \
    max = _mm_max_ps(x, y);                     \
    x = min; y = max;

    COMPARE_TWO(*x0, *x1);
    COMPARE_TWO(*x2, *x3);

    COMPARE_TWO(*x1, *x2);

    COMPARE_TWO(*x0, *x1);
    COMPARE_TWO(*x2, *x3);

    COMPARE_TWO(*x1, *x2);

    // transpose matrix
    // see. http://download.intel.com/design/PentiumIII/sml/24504301.pdf
    __m128 tmp;
    __m128 row0, row1;
    tmp = _mm_shuffle_ps(*x0, *x1, _MM_SHUFFLE(1,0,1,0));
    row1 = _mm_shuffle_ps(*x2, *x3, _MM_SHUFFLE(1,0,1,0));

    row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0));
    row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1));

    tmp = _mm_shuffle_ps(*x0, *x1, _MM_SHUFFLE(3,2,3,2));
    *x0 = row0; *x1 = row1;
    row1 = _mm_shuffle_ps(*x2, *x3, _MM_SHUFFLE(3,2,3,2));
    
    row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0));
    row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1));
    *x2 = row0; *x3 = row1;

    // puts("odd_merge:");
    // inspect(*x0); inspect(*x1); inspect(*x2); inspect(*x3);
}



static void inline bitonic_merge_kernel(__m128 *a, __m128 *b);

static void inline bitonic_merge_permutate4x4(__m256* a, __m256* b){
	__m256 l, h, lp, hp, ol, oh;    
    *b = _mm256_shuffle_ps(*b, *b, _MM_SHUFFLE(0,1,2,3));
    
    l = _mm256_min_ps(*a, *b);
    h = _mm256_max_ps(*a, *b);

    //printf("here->%d\n", __LINE__); inspectLH(l, h);
    lp = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    hp = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
    //printf("here->%d\n", __LINE__); inspectLH(lp, hp);
    //inspectLH(lp, hp);
    
    l = _mm256_min_ps(lp, hp);
    h = _mm256_max_ps(lp, hp);
    //printf("here->%d\n", __LINE__); inspectLH(l, h);
    lp = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(2,0,2,0));
    //printf("here->%d\n", __LINE__);inspect(lp);
    lp = _mm256_shuffle_ps(lp, lp, _MM_SHUFFLE(3,1,2,0));
    //printf("here->%d\n", __LINE__);inspect(lp);

    hp = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(3,1,3,1));
    //printf("here->%d\n", __LINE__);inspect(hp);
    hp = _mm256_shuffle_ps(hp, hp, _MM_SHUFFLE(3,1,2,0));
    //printf("here->%d\n", __LINE__);inspect(hp);
    //inspectLH(lp, hp);
    l = _mm256_min_ps(lp, hp);
    h = _mm256_max_ps(lp, hp);
 	//printf("here->%d\n", __LINE__);inspectLH(l, h);

    ol = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    oh = _mm256_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
  	//printf("here->%d\n", __LINE__);inspectLH(ol, oh);
    
    *a = _mm256_shuffle_ps(ol, ol, _MM_SHUFFLE(3,1,2,0));
    *b = _mm256_shuffle_ps(oh, oh, _MM_SHUFFLE(3,1,2,0));
   	//printf("here->%d\n", __LINE__);inspectLH(*a, *b);
}

static void inline upsidedown_4n4(__m256* a){
	int _perm[8];
    _perm[0] = 4;
    _perm[1] = 5;
    _perm[2] = 6;
    _perm[3] = 7;
    _perm[4] = 0;
    _perm[5] = 1;
    _perm[6] = 2;
    _perm[7] = 3;
     __m256i idx =  _mm256_loadu_si256((__m256i *) _perm);
     *a = _mm256_permutevar8x32_ps(*a, idx);    
    

}
static void inline
bitonic_merge_kernel(__m256 *a, __m256 *b){
    bitonic_merge_permutate4x4(a,b);
    upsidedown_4n4(b);
    bitonic_merge_permutate4x4(a,b);    
    upsidedown_4n4(a);
    bitonic_merge_permutate4x4(a,b);    
    upsidedown_4n4(b);
    bitonic_merge_permutate4x4(a,b);
    upsidedown_4n4(a);
    bitonic_merge_permutate4x4(a,b);
    //printf("here->%d\n", __LINE__ );inspectLH(*a, *b);

    return; 
}

//another way is to use
//__m128 _mm256_extractf128_ps  (__m256 a, int imm8)
//__m256 c = _mm256_castps128_ps256(a);
//c = _mm256_insertf128_ps(c,b,1);
static void inline
bitonic_merge_kernel(__m256 *a, __m256 *b, int flag){
	__m128 x0 = _mm256_extractf128_ps (*a, 0);
	__m128 x1 = _mm256_extractf128_ps (*a, 1);
	__m128 x2 = _mm256_extractf128_ps (*b, 0);
	__m128 x3 = _mm256_extractf128_ps (*b, 1);
	/*fprintf(stderr, "here->%d\n", __LINE__);inspect(x0);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x1);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x2);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x3);*/

	bitonic_merge_kernel(&x0, &x1);
	bitonic_merge_kernel(&x0, &x2);
	bitonic_merge_kernel(&x0, &x3);
	bitonic_merge_kernel(&x1, &x2);
	bitonic_merge_kernel(&x1, &x3);
	bitonic_merge_kernel(&x2, &x3);


	*a = _mm256_castps128_ps256(x0);
	*a = _mm256_insertf128_ps(*a,x1,1);
	*b = _mm256_castps128_ps256(x2);
	*b = _mm256_insertf128_ps(*b,x3,1);
	//fprintf(stderr, "here->%d\n", __LINE__);inspectLH(*a, *b);


    return; 
}


static void inline
bitonic_merge_kernel(__m256 *a, __m256 *b, int flag, int flag2){
	//fprintf(stderr, "here->%d\n", __LINE__); inspectLH(*a, *b);
	//upsidedown_4n4(b);
	//fprintf(stderr, "here->%d\n", __LINE__);inspect(*b);
 	__m256 x = _mm256_blend_ps(*a, *b, _MM_SHUFFLE(3,3,0,0));;
	__m256 y = _mm256_blend_ps(*a, *b, _MM_SHUFFLE(0,0,3,3));
	//upsidedown_4n4(&y); 
	bitonic_merge_permutate4x4(&x, &y);	
	//upsidedown_4n4(&y);
	__m256 i = _mm256_blend_ps(x, y, _MM_SHUFFLE(3,3,0,0));
	__m256 j = _mm256_blend_ps(x, y, _MM_SHUFFLE(0,0,3,3));
	//upsidedown_4n4(&j);	
	bitonic_merge_permutate4x4(&i, &j);
 	
	/*{
		//__m128 x0 = _mm256_extractf128_ps (i, 0);
	__m128 x1 = _mm256_extractf128_ps (i, 1);
	__m128 x2 = _mm256_extractf128_ps (j, 0);
	//__m128 x3 = _mm256_extractf128_ps (j, 1);
	bitonic_merge_kernel(&x1, &x2);	
	/*fprintf(stderr, "here->%d\n", __LINE__);inspect(x0);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x1);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x2);
	fprintf(stderr, "here->%d\n", __LINE__);inspect(x3);*/


	/**a = _mm256_insertf128_ps(i,x1,1);
	*b = _mm256_insertf128_ps(j,x2,0);
	//fprintf(stderr, "here->%d\n", __LINE__); inspectLH(*a, *b);
	}*/


	upsidedown_4n4(&y);
 	bitonic_merge_permutate4x4(&x, &y);	
 	upsidedown_4n4(&y);	
	*a = x; *b = y; 
}


static void inline
bitonic_merge_kernel(__m128 *a, __m128 *b)
{
   // puts("[dbg] bitonic_merge_kernel(__m128)");
    //inspectLH(*a, *b);
    __m128 l, h, lp, hp, ol, oh;
    // puts("-----------------------------------------------");
    *b = _mm_shuffle_ps(*b, *b, _MM_SHUFFLE(0,1,2,3));
    //inspect(*b);
    // inspectLH(*a, *b);

    l = _mm_min_ps(*a, *b);
    h = _mm_max_ps(*a, *b);

  //  printf("here->%d\n", __LINE__); inspectLH(l, h);
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
 //  printf("here->%d\n", __LINE__); inspectLH(lp, hp);
    // inspectLH(lp, hp);
    
    l = _mm_min_ps(lp, hp);
    h = _mm_max_ps(lp, hp);
  //  printf("here->%d\n", __LINE__); inspectLH(l, h);
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(2,0,2,0));
 //   printf("here->%d\n", __LINE__);inspect(lp);
    lp = _mm_shuffle_ps(lp, lp, _MM_SHUFFLE(3,1,2,0));
 //  printf("here->%d\n", __LINE__);inspect(lp);

    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,1,3,1));
 //   printf("here->%d\n", __LINE__);inspect(hp);
    hp = _mm_shuffle_ps(hp, hp, _MM_SHUFFLE(3,1,2,0));
 //   printf("here->%d\n", __LINE__);inspect(hp);

    // inspectLH(lp, hp);

    l = _mm_min_ps(lp, hp);
    h = _mm_max_ps(lp, hp);
 //   printf("here->%d\n", __LINE__);inspectLH(l, h);

    ol = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    oh = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
  //  printf("here->%d\n", __LINE__);inspectLH(ol, oh);
    
    *a = _mm_shuffle_ps(ol, ol, _MM_SHUFFLE(3,1,2,0));
    *b = _mm_shuffle_ps(oh, oh, _MM_SHUFFLE(3,1,2,0));
    // inspectLH(l, h);
  // printf("here->%d\n", __LINE__);inspectLH(*a, *b);
}

 

/* list must be 16 aligned */
/* for now, length is assumed to be 2^n */
static void merge_sort_avx(float *buffer, float *list, uintptr_t length);
static void merge_sort_rev_avx(float *buffer, float *list, uintptr_t length);
static void inline merge_sort_merge_avx(float *output, float *input, uintptr_t length);
 
/* merge_sort is expected to sort list in place */
static void
merge_sort_avx(float *buffer, float *list, uintptr_t length)
{
    uintptr_t half = length / 2;
    if (length == 64){       
         bitonic_sort_64elems(buffer, list);       
    } else {
 
        merge_sort_rev_avx(buffer, list, half);
        merge_sort_rev_avx(buffer + half, list + half, half);
    
    }
     merge_sort_merge_avx(list, buffer, length);
        
}

/* merge_sort_rev is expected to sort list and write output to buffer */
static void 
merge_sort_rev_avx(float *buffer, float *list, uintptr_t length)
{
 
    uintptr_t half = length / 2;
    if (length == 64){        
         bitonic_sort_64elems(list, list);
    } else {
 
        merge_sort_avx(buffer, list, half);
        merge_sort_avx(buffer + half, list + half, half);
    }
    //printf("here->%d\n", __LINE__);
    merge_sort_merge_avx(buffer, list, length);
}



static void inline
merge_sort_merge_avx(float *output, float *input, uintptr_t length)
{   

 
    uintptr_t half;
    float *halfptr;
    float *sentinelptr;

    half = length / 2;
    halfptr = input + half;
    sentinelptr = input + length;


    __m256 x, y;
    float *list1 = input;
    float *list2 = halfptr;
 
    x = _mm256_loadu_ps(list1);
    y = _mm256_loadu_ps(list2);
 

    list1 += 8;
    list2 += 8;
 

    bitonic_merge_kernel(&x, &y);    
    _mm256_storeu_ps(output, x);
    output += 8;    
    while(list1 < halfptr && list2 < sentinelptr){
        if (*list1 < *list2){
            x = _mm256_load_ps(list1);
            list1 += 8;
            bitonic_merge_kernel(&x, &y);
            _mm256_storeu_ps(output, x);
            output += 8;
            if (list1 >= halfptr){
                goto nomore_in_list1;
            }
        } else {
            x = _mm256_loadu_ps(list2);
            list2 += 8;
            bitonic_merge_kernel(&x, &y);
            _mm256_storeu_ps(output, x);
            output += 8;
            if (list2 >= sentinelptr){
                goto nomore_in_list2;
            }
        }
    }
nomore_in_list1:
    while(list2 < sentinelptr){
        x = _mm256_loadu_ps(list2);
        list2 += 8;
        bitonic_merge_kernel(&x, &y);
        _mm256_storeu_ps(output, x);
        output += 8;
    }
    goto end;
nomore_in_list2:
    while(list1 < halfptr){
        x = _mm256_load_ps(list1);
        list1 += 8;
        bitonic_merge_kernel(&x, &y);
        _mm256_storeu_ps(output, x);
        output += 8;
    }
end:
     _mm256_storeu_ps(output, y);
   //printf("here->%d\n", __LINE__);
   // inspect_array(output, length);
    return;
}
 

static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}



static void merge_sort(float *buffer, float *list, uintptr_t length);
static void merge_sort_rev(float *buffer, float *list, uintptr_t length);
static void inline merge_sort_merge(float *output, float *input, uintptr_t length);


static bool verify(float* offset, int length){
    for (int i = 0; i < length-1; ++i)
    {
        if(offset[i] > offset[i+1]) {

            printf("[dbg] failed at [%d, %d] with pair [%f, %f] \n", 
                i, i+1, offset[i], offset[i+1] );
            return false; 
        }
    }
    return true; 
}


void test_routine1(){   
	puts("[dbg] Test Routine 1");
    float __mmx_a[8] = {7.0,8.0,15.0,16.0,23.0,24.0,31.0,32.0};
    float __mmx_b[8] = {6.0,14.0,22.0,30.0,38.0,46.0,54.0,62.0};
    {
    	puts("bitonic_merge_kernel/Method 1:");
    	__m256 a = _mm256_loadu_ps(__mmx_a);
    	__m256 b = _mm256_loadu_ps(__mmx_b);
    	unsigned long long _start = rdtsc();        	
    	bitonic_merge_kernel(&a, &b);
    	unsigned long long _end = rdtsc();    
    	printf("[dbg] runtime ticks: %lld\n", _end - _start);
    }

    {
    	puts("bitonic_merge_kernel/Method 2:");
    	__m256 a = _mm256_loadu_ps(__mmx_a);
    	__m256 b = _mm256_loadu_ps(__mmx_b);
    	unsigned long long _start = rdtsc();        	
    	bitonic_merge_kernel(&a, &b, 1);
    	unsigned long long _end = rdtsc();    
    	printf("[dbg] runtime ticks: %lld\n", _end - _start);
    }


    {
    	puts("bitonic_merge_kernel/Method 3:");
    	__m256 a = _mm256_loadu_ps(__mmx_a);
    	__m256 b = _mm256_loadu_ps(__mmx_b);
    	unsigned long long _start = rdtsc();        	
    	bitonic_merge_kernel(&a, &b, 1, 1);
    	unsigned long long _end = rdtsc();    
    	printf("[dbg] runtime ticks: %lld\n", _end - _start);
    }
    
    

    
    //printf("here->%d\n", __LINE__);inspectLH(a, b );
}




void test_routine3(int exp){
    const long long  arrsize = 1 << exp; 
    float *input;
    float *input_sse;
    float *output;
    float *qsort_input; 
    posix_memalign((void**)&input, 32, sizeof(float) * arrsize);
    posix_memalign((void**)&input_sse, 32, sizeof(float) * arrsize);
    posix_memalign((void**)&qsort_input, 32, sizeof(float) * arrsize);
    posix_memalign((void**)&output, 32, sizeof(float) * arrsize);

    memset(input, 0, sizeof(float) * arrsize);
    memset(input_sse, 0, sizeof(float) * arrsize);
    memset(output, 0, sizeof(float) * arrsize);


    srand(0);
    for (uintptr_t i = 0; i < arrsize ; i++)
    {
        input[i] = (float)drand48();
        input_sse[i] = input[i];
        qsort_input[i] = input[i];
    }
    

    unsigned long long avx_sort_start = rdtsc();    
    merge_sort_avx(output, input, arrsize);
    
    unsigned long long avx_sort_end = rdtsc();
    unsigned long long avx_sort_delta = avx_sort_end - avx_sort_start;
    cout<< "[dbg] AVX sort CPU ticks: " << avx_sort_delta << endl;
    if(verify(input, arrsize)) cout<< "CHECK PASS" << endl; 
    else cout<< "CHECK FAIL" << endl; 

   /* for (int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; j++){
            printf("%.1f ", input[8 *i +  j ] );
        }puts("");      
        
    }*/



    unsigned long long sse_sort_start = rdtsc();    
    merge_sort(output, input_sse, arrsize);    
    unsigned long long sse_sort_end = rdtsc();
    unsigned long long sse_sort_delta = sse_sort_end - sse_sort_start;
    cout<< "[dbg] SSE sort CPU ticks: " << sse_sort_delta << endl;
    cout<<"[dbg] Ratio: " <<(double)avx_sort_delta / sse_sort_delta << endl; 
    if(verify(input_sse, arrsize)) cout<< "CHECK PASS" << endl; 
    else cout<< "CHECK FAIL" << endl; 


    unsigned long long qsort_start = rdtsc();    
    sort(qsort_input, qsort_input +arrsize);    
    unsigned long long qsort_end = rdtsc();
    unsigned long long qsort_delta = qsort_end - qsort_start;
    cout<< "[dbg] qsort CPU ticks: " << qsort_delta << endl;
        
}


void test_routine2(){
    float input[64], input_sse[64], output[64];
    posix_memalign((void**)&input, 32, sizeof(float) * 64);
    posix_memalign((void**)&input_sse, 32, sizeof(float) * 64);
    posix_memalign((void**)&output, 32, sizeof(float) * 64);

   
    for (int i = 64; i >= 1 ; i--)
    {
        input[64-i] = i; 
    }
    memcpy(input_sse, input, sizeof(float) * 64);

    unsigned long long avx_sort_start = rdtsc();    
    merge_sort_avx(output, input, 64);
    
    unsigned long long avx_sort_end = rdtsc();
    unsigned long long avx_sort_delta = avx_sort_end - avx_sort_start;
    cout<< "[dbg] AVX sort CPU ticks: " << avx_sort_delta << endl;
    if(verify(input, 64)) cout<< "CHECK PASS" << endl; 
    else cout<< "CHECK FAIL" << endl; 

   /* for (int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; j++){
            printf("%.1f ", input[8 *i +  j ] );
        }puts("");      
        
    }*/



    unsigned long long sse_sort_start = rdtsc();    
    merge_sort(output, input_sse, 64);
    
    unsigned long long sse_sort_end = rdtsc();
    unsigned long long sse_sort_delta = sse_sort_end - sse_sort_start;
    cout<< "[dbg] SSE sort CPU ticks: " << sse_sort_delta << endl;
    cout<<"[dbg] Ratio: " <<(double)avx_sort_delta / sse_sort_delta << endl; 
    if(verify(input_sse, 64)) cout<< "CHECK PASS" << endl; 
    else cout<< "CHECK FAIL" << endl; 
   /* uintptr_t num_exp = 4;
    if (argc >= 2){
        num_exp = atoi(argv[1]);
    }
    
    merge_sort_test(num_exp);    */
    //simple_bitonic_sort_test();

   
    //bitonic_sort_64elems(output, input );

    /*for (int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; j++){
            printf("%.1f ", input[8 *i + j ] );
        }puts("");      
        
    }*/
}

void merge_intervals(float *input, float *merge_buffer, int start, int mid, int end){
  
	int length1 = mid - start; 
	int length2 = end - mid + 1; 
	int i = start, j = mid; 
	int cur = start; 


	while(i < mid && j < end ){
		if(input[i] <= input[j]){
			merge_buffer[cur++] = input[i];
			i++;
		}else{
			merge_buffer[cur++] = input[j];
			j++;
		}
	}
	while(i < mid){
		merge_buffer[cur++] = input[i++];
	}
	while(j < end){
		merge_buffer[cur++] = input[j++];
	}

  
    memcpy(input + start, merge_buffer + start, (end - start) * sizeof(float));

   

}




int test_routine4(int arrsize){   
    float *input;
    float *merge_buffer; 
    posix_memalign((void**)&input, 32, sizeof(float) * arrsize);
    posix_memalign((void**)&merge_buffer, 32, sizeof(float) * arrsize);
    memset(input, 0, sizeof(float) * arrsize);    
    memset(merge_buffer, 0, sizeof(float) * arrsize);

    srand(0);
    for (uintptr_t i = 0; i < arrsize ; i++)
    {
        input[i] = (float)drand48();
    }

    int length = arrsize;         
    int start = 0;
    int upper, interval_length;

    
    
    while(true)
    {
    	upper = int(log(length) / log(2));   	
    	interval_length = 1 << upper;
    	if(interval_length <= 8) {
    		sort(input+start, input+start+interval_length);
               	
            
        }else if(interval_length == 16){
            bitonic_sort_16elems(input+start, input+start);
            __m256 x0, x1;
            x0 = _mm256_loadu_ps(input + start  ); 
            x1 = _mm256_loadu_ps(input + start + 8 ); 
            bitonic_merge_kernel(&x0, &x1); 

            _mm256_storeu_ps(input + start, x0);
            _mm256_storeu_ps(input + start+ 8, x1);
          
            
        }else if(interval_length == 32){

            bitonic_sort_16elems(input+start, input+start);
            bitonic_sort_16elems(input+start+16, input+start +16);

            __m256 x[4];
            for(int i = 0;i < 4;i++){ 
                x[i] = _mm256_loadu_ps(input + start + 8 * i); 
            }
            bitonic_merge_kernel(&x[0], &x[1] ,1 ,1); 
            bitonic_merge_kernel(&x[0], &x[2] ,1 ,1); 
            bitonic_merge_kernel(&x[0], &x[3] ,1 ,1); 
            bitonic_merge_kernel(&x[1], &x[2] ,1 ,1); 
            bitonic_merge_kernel(&x[1], &x[3] ,1 ,1); 
            bitonic_merge_kernel(&x[2], &x[3] ,1 ,1); 

            _mm256_storeu_ps(input + start, x[0]);
            _mm256_storeu_ps(input + start+ 8, x[1]);
            _mm256_storeu_ps(input + start+ 16, x[2]);
            _mm256_storeu_ps(input + start+ 24, x[3]);

        }else if(interval_length >= 64){   
            merge_sort_avx(merge_buffer + start, input + start, interval_length);          
          
        }



    	if(start == 0) { start += interval_length;  }
    	else {
    		//start to merge    		    		
    		merge_intervals(input, merge_buffer, 0, start, start+interval_length);
    		start += interval_length;    		
          
    	} 
    	length = length - interval_length;
    	if(length == 0) break; 
    }


    if(verify(input, arrsize)) {cout<< "CHECK PASS" << endl;  return 1; }
    else{ cout<< "CHECK FAIL" << endl; return 0;}




}


int main(int argc, char **argv)
{
	if(argc >=2 ) test_routine4(atoi(argv[1]));      	
    return 0; 

}

// assume the length of list is 16
static void
bitonic_sort_16elems(float *ret, float* list)
{
    __m128 x[4];
    int i;

    for(i = 0;i < 4;i++){
        x[i] = _mm_load_ps(list + 4 * i);
    }
    odd_merge_in_register_sort(&x[0], &x[1], &x[2], &x[3]);
    bitonic_merge_kernel(&x[0], &x[1]);
    bitonic_merge_kernel(&x[2], &x[3]);
    _mm_storeu_ps(list, x[0]);
    _mm_storeu_ps(list+4, x[1]);
    _mm_storeu_ps(list+8, x[2]);
    _mm_storeu_ps(list+12, x[3]);
}




/* merge_sort is expected to sort list in place */
static void
merge_sort(float *buffer, float *list, uintptr_t length)
{
    uintptr_t half = length / 2;
    if (length == 16){
        bitonic_sort_16elems(buffer, list);
    } else {
        merge_sort_rev(buffer, list, half);
        merge_sort_rev(buffer + half, list + half, half);
    }
    merge_sort_merge(list, buffer, length);
}

/* merge_sort_rev is expected to sort list and write output to buffer */
static void 
merge_sort_rev(float *buffer, float *list, uintptr_t length)
{
    uintptr_t half = length / 2;
    if (length == 16){
        bitonic_sort_16elems(list, list);
    } else {
        merge_sort(buffer, list, half);
        merge_sort(buffer + half, list + half, half);
    }
    
    merge_sort_merge(buffer, list, length);
}



static void inline
merge_sort_merge(float *output, float *input, uintptr_t length)
{

    uintptr_t half;
    float *halfptr;
    float *sentinelptr;

    half = length / 2;
    halfptr = input + half;
    sentinelptr = input + length;

    __m128 x, y;
    float *list1 = input;
    float *list2 = halfptr;
    x = _mm_load_ps(list1);
    y = _mm_load_ps(list2);
    list1 += 4;
    list2 += 4;
    bitonic_merge_kernel(&x, &y);
    _mm_storeu_ps(output, x);
    output += 4;

    while(list1 < halfptr && list2 < sentinelptr){
        if (*list1 < *list2){
            x = _mm_load_ps(list1);
            list1 += 4;
            bitonic_merge_kernel(&x, &y);
            _mm_store_ps(output, x);
            output += 4;
            if (list1 >= halfptr){
                goto nomore_in_list1;
            }
        } else {
            x = _mm_load_ps(list2);
            list2 += 4;
            bitonic_merge_kernel(&x, &y);
            _mm_store_ps(output, x);
            output += 4;
            if (list2 >= sentinelptr){
                goto nomore_in_list2;
            }
        }
    }
nomore_in_list1:
    while(list2 < sentinelptr){
        x = _mm_load_ps(list2);
        list2 += 4;
        bitonic_merge_kernel(&x, &y);
        _mm_store_ps(output, x);
        output += 4;
    }
    goto end;
nomore_in_list2:
    while(list1 < halfptr){
        x = _mm_load_ps(list1);
        list1 += 4;
        bitonic_merge_kernel(&x, &y);
        _mm_store_ps(output, x);
        output += 4;
    }
end:
    
    _mm_store_ps(output, y);
    return;
}






