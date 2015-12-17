#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <memory.h>
#include <math.h>




#include <immintrin.h>
#include <xmmintrin.h>

#include <algorithm>
#include <iostream>
using namespace std;
//http://parallelcomp.uw.hu/ch09lev1sec2.html
//float x[] = {10.f, 5.0f, 3.0f, 12.0f, 90.0f, 60.0f, 23.0f, 95.0f};
//float y[] = {20.0f, 9.0f, 8.0f, 14.0f, 0.0f, 40.0f, 35.0f, 18.0f}; 
float x[] = {67.0f,  173.0f, 196.0f, 103.0f, 117.0f, 105.0f, 80.0f,  114.0f, 142.0f, 31.0f};
float y[] = {69.0f,  156.0f, 144.0f, 118.0f, 51.0f,  56.0f,  192.0f, 190.0f, 194.0f, 139.0f};


//float x[] = {0.000000,0.000985,0.041631,0.050832,0.091331,0.092298,0.176643,0.233178};
//float y[] = {0.364602,0.454433,0.487217,0.526750,0.556094,0.568060,0.831292,0.931731};

void _swap_reg(float &a, float &b, int idx){
	float c = a <= b ? a:b; 
	float d = a > b ? a:b;
	*(x+idx) = c; 
	*(y+idx) = d; 

}




static void
inspectLH_avx(__m256 l, __m256 h)
{
    float f1[8];
    float f2[8];
    _mm256_storeu_ps(f1, l);
    _mm256_storeu_ps(f2, h);
    

    for (int i = 0; i < 8; ++i)
    {
    	/* code */
    	printf("RegA_%d: %.1f RegB_%d: %.1f\n",  i, f1[i], i, f2[i]);
    }
    puts("=================================");

    /*printf("[L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] [L:%.1f H:%.1f] \n",
     f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7]);*/
}



static void inline
inspect_avx(__m256 v)
{
    float f[8];
    _mm256_storeu_ps(f, v);
     for (int i = 0; i < 8; ++i)
    {
    	/* code */
    	printf("Reg%d: %.1f\n", i,  f[i]);
    }
    puts("=================================");
    /*printf("[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n", 
        f[0], f[1], f[2], f[3],f[4], f[5], f[6], f[7]);*/
}


static void inline
bitonic_merge_kernel_avx(__m256 *a, __m256 *b)
{
   

  
    puts("[dbg] PHASE 1");
    __m256 a1 = _mm256_min_ps(*a, *b); 
    __m256 b1 = _mm256_max_ps(*a, *b);
    __m256 a1_out = _mm256_blend_ps (a1, b1, _MM_SHUFFLE(2,2,2,2));
    __m256 b1_out = _mm256_blend_ps (b1, a1, _MM_SHUFFLE(2,2,2,2));

    inspectLH_avx(a1_out, b1_out);
    puts("[dbg] PHASE 2");

    __m256 a1_mod = _mm256_shuffle_ps(a1_out, a1_out, _MM_SHUFFLE(2,3,0,1));
    inspectLH_avx(a1_mod, b1_out);

    __m256 b1_mod = _mm256_shuffle_ps(b1_out, b1_out, _MM_SHUFFLE(2,3,0,1));
    inspectLH_avx(a1_out, b1_mod);

    __m256 a21_blend = _mm256_blend_ps (a1_out, b1_mod, _MM_SHUFFLE(2,2,2,2));
    __m256 b21_blend = _mm256_blend_ps (a1_mod, b1_out, _MM_SHUFFLE(2,2,2,2));
    inspectLH_avx(a21_blend, b21_blend);

    __m256 a21_out = _mm256_min_ps(a21_blend, b21_blend);
    __m256 b21_out = _mm256_max_ps(a21_blend, b21_blend);
    inspectLH_avx(a21_out, b21_out);


    __m256 regx = _mm256_blend_ps (a21_out, b21_out, _MM_SHUFFLE(3,0,3,0));
    __m256 regy = _mm256_blend_ps (a21_out, b21_out, _MM_SHUFFLE(0,3,0,3));
    inspectLH_avx(regx, regy);

    __m256 regx1 = _mm256_permute_ps(regx, _MM_SHUFFLE(2,3,0,1)  );
    __m256 regy1 = _mm256_permute_ps(regy, _MM_SHUFFLE(2,3,0,1)  );
    inspectLH_avx(regx1, regy1);
    __m256 regx2 = _mm256_blend_ps (regx, regy1, _MM_SHUFFLE(2,2,2,2));
    __m256 regy2 = _mm256_blend_ps (regx1, regy, _MM_SHUFFLE(2,2,2,2));
    inspectLH_avx(regx2, regy2);

    regx = _mm256_min_ps(regx2, regy2);
    regy = _mm256_max_ps(regx2, regy2);
    inspectLH_avx(regx, regy);
    regx1 = _mm256_blend_ps (regx, regy, _MM_SHUFFLE(3,0,3,0));
    regy1 = _mm256_blend_ps (regx, regy, _MM_SHUFFLE(0,3,0,3));
    inspectLH_avx(regx1, regy1);

    regx = regx1; 
    regy = regy1; 
    //output from the second big box; 


    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;


    regx = _mm256_min_ps(regx1, regy1);
    regy = _mm256_max_ps(regx1, regy1);
    inspectLH_avx(regx, regy);


    regx1 = _mm256_blend_ps (regx, regy, _MM_SHUFFLE(3,3,0,0));
    regy1 = _mm256_blend_ps (regx, regy, _MM_SHUFFLE(0,0,3,3));
    inspectLH_avx(regx1, regy1);

 

    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy);

    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;
    inspectLH_avx(regx1, regy1);

    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy);


    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;
   // inspectLH_avx(regx1, regy1);
    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy);


    regx1 = _mm256_min_ps(regx, regy);
    regy1 = _mm256_max_ps(regx, regy);
    inspectLH_avx(regx1, regy1);
    regx = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(3,3,0,0));
    regy = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(0,0,3,3));
    inspectLH_avx(regx, regy);


    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;
   // inspectLH_avx(regx1, regy1);
    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy);


    regx1 = _mm256_min_ps(regx, regy);
    regy1 = _mm256_max_ps(regx, regy);
    inspectLH_avx(regx1, regy1);

    regx = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(3,3,0,0));
    regy = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(0,0,3,3));
    inspectLH_avx(regx, regy);

    //Box phase 2


    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;
    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
   // inspectLH_avx(regx, regy);

    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
   // inspectLH_avx(regx1, regy1);
//

    regx = _mm256_permute2f128_ps(regx1, regy1, 0x20);
    regy = _mm256_permute2f128_ps(regx1, regy1, 0x31);
    inspectLH_avx(regx, regy );


    regx1 = _mm256_min_ps(regx, regy);
    regy1 = _mm256_max_ps(regx, regy);
    inspectLH_avx(regx1, regy1); //output 

    //===========================================
    regx = (_mm256_unpacklo_ps(regx1, regy1)) ;
    regy = (_mm256_unpackhi_ps(regx1, regy1)) ;
    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx1, regy1);


    //THE FIRST LONG PHASE FINISH HERE; 
//( __m256i )0+ 2<<3 + 4 <<6 + 6<<9 + 1<<12 + 3<<15 + 5<<18 + 7<<21 
    int _perm[8];
    _perm[0] = 0;
    _perm[1] = 2;
    _perm[2] = 4;
    _perm[3] = 6;
    _perm[4] = 1;
    _perm[5] = 3;
    _perm[6] = 5;
    _perm[7] = 7;
    __m256i idx =  _mm256_load_si256((__m256i *) _perm);
    regx = _mm256_permutevar8x32_ps(regx1, idx);
    regy = _mm256_permutevar8x32_ps(regy1, idx);
    inspectLH_avx(regx, regy);



    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;


    regx = _mm256_min_ps(regx1, regy1);
    regy = _mm256_max_ps(regx1, regy1);
    inspectLH_avx(regx, regy); 
 

    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx1, regy1);
        //===========================================
    regx = (_mm256_unpacklo_ps(regx1, regy1)) ;
    regy = (_mm256_unpackhi_ps(regx1, regy1)) ;
    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx1, regy1);



    regx = (_mm256_unpacklo_ps(regx1, regy1)) ;
    regy = (_mm256_unpackhi_ps(regx1, regy1)) ;
    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
    regx = _mm256_min_ps(regx1, regy1);
    regy = _mm256_max_ps(regx1, regy1);
    inspectLH_avx(regx, regy); 

    regx1 = (_mm256_unpacklo_ps(regx, regy)) ;
    regy1 = (_mm256_unpackhi_ps(regx, regy)) ;
    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy); 


    regx1 = _mm256_min_ps(regx, regy);
    regy1 = _mm256_max_ps(regx, regy);
    inspectLH_avx(regx1, regy1); //<good result here; 


    regx = (_mm256_unpacklo_ps(regx1, regy1)) ;
    regy = (_mm256_unpackhi_ps(regx1, regy1)) ;
    regx1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(1,0,1,0));
    regy1 = _mm256_shuffle_ps (regx, regy, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx1, regy1); 


    regx = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(1,0,1,0));
    regy = _mm256_shuffle_ps (regx1, regy1, _MM_SHUFFLE(3,2,3,2));
    inspectLH_avx(regx, regy); 



    _perm[0] = 4;
    _perm[1] = 5;
    _perm[2] = 6;
    _perm[3] = 7;
    _perm[4] = 0;
    _perm[5] = 1;
    _perm[6] = 2;
    _perm[7] = 3;
    idx =  _mm256_load_si256((__m256i *) _perm);
    regx1 = _mm256_permutevar8x32_ps(regx, idx);
    regy1 = regy; 
    inspectLH_avx(regx1, regy1);


    regx = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(0,0,3,3));
    regy = _mm256_blend_ps (regx1, regy1, _MM_SHUFFLE(3,3,0,0));
    inspectLH_avx(regx, regy);


    regx1 = _mm256_permutevar8x32_ps(regx, idx);
    regy1 = regy; 
    inspectLH_avx(regx1, regy1);
    
    
    /*float ret[16];
    _mm256_storeu_ps(ret, regx1);
    _mm256_storeu_ps(ret + 8, regy1);
    puts("[dbg] output the final result");
    for(int i = 0;i < 16;i++){
        printf("%f\n", ret[i]);
    }
    puts("");*/



    

/*for(int i = 0;i < 16;i++){
        printf("%.1f ", ret[i]);
    }*/

    //__m256 a3_2_hi = _mm256_unpackhi_ps (a1, b1);
       // inspect_avx(a3_2_hi);



    return; 
   
  

}





static void
simple_bitonic_test_avx(void)
{

    //float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    //float y[] = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f};
    std::reverse(x, x + 8);
    std::reverse(y ,y + 8); 
	

    __m256 a = _mm256_set_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    __m256 b = _mm256_set_ps(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
    
	//puts("[dbg] output the xy value:");inspect_xy();
	//puts("[dbg] output the xy value");inspect_xy();
    

    bitonic_merge_kernel_avx(&a, &b);

   // inspect_avx(a);
   // inspect_avx(b);


   /* int i;     
*/
}




int main(int argc, char const *argv[])
{
	/*float test_reg_a  = 1.6;
	float test_reg_b = 1.6;
	_swap_reg(test_reg_a, test_reg_b);
	cout<< "test_reg_a: " << test_reg_a << " test_reg_b: " << test_reg_b << endl;*/
	simple_bitonic_test_avx();
	return 0;
}