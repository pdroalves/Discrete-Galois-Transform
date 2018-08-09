#include <cuda.h>
#include <iostream>
#include <dgt/dgt.h>
#include <assert.h>
#include <map>
#include <NTL/ZZ.h>
NTL_CLIENT

#define MAXDEGREE (int)8192 // REMOVE THIS

std::map<int, GaloisInteger> NTHROOT = {
  {4, (GaloisInteger){17872535116924321793, 18446744035054843905}},
  {8, (GaloisInteger){18446741870391328801, 18293621682117541889}},
  {16, (GaloisInteger){13835058050988244993, 68719460336}},
  {32, (GaloisInteger){18446664905516926977, 18446664903637878785}},
  {64, (GaloisInteger){5006145799600815370, 13182758589097755684}},
  {128, (GaloisInteger){1139268601170473317, 15214299942841048380}},
  {256, (GaloisInteger){4169533218321950981, 11340503865284752770}},
  {512, (GaloisInteger){1237460157098848423, 590072184190675415}},
  {1024, (GaloisInteger){13631489165933064639, 9250462654091849156}},
  {2048, (GaloisInteger){12452373509881738698, 10493048841432036102}},
  {4096, (GaloisInteger){12694354791372265231, 372075061476485181}},
  {8192, (GaloisInteger){9535633482534078963, 8920239275564743446}},
  {16384, (GaloisInteger){9868966171728904500, 6566969707398269596}},
  {32768, (GaloisInteger){10574165493955537594, 3637150097037633813}},
  {65536, (GaloisInteger){2132094355196445245, 12930307277289315455}}
};
std::map<int, GaloisInteger> INVNTHROOT = {
  {4, (GaloisInteger){34359736320, 17868031517297999873}},
  {8, (GaloisInteger){18311636080627023873, 18446741870391328737}},
  {16, (GaloisInteger){18446739675663041537, 18446462594505048065}},
  {32, (GaloisInteger){9223372049739972605, 9223372049739382781}},
  {64, (GaloisInteger){3985917792403544159, 10871216858344511621}},
  {128, (GaloisInteger){697250266387245748, 7269985899340929883}},
  {256, (GaloisInteger){16440350318199496391, 8259263625103887671}},
  {512, (GaloisInteger){11254465366323603399, 282547220712277683}},
  {1024, (GaloisInteger){4772545667722300316, 8077569763565898552}},
  {2048, (GaloisInteger){13028894352332048345, 9995848711590186809}},
  {4096, (GaloisInteger){11525613835860693, 17335883825168514904}},
  {8192, (GaloisInteger){17414606149056687587, 3916527805974289959}},
  {16384, (GaloisInteger){9801605401783064476, 2749242888134484347}},
  {32768, (GaloisInteger){10469048769509713349, 8715957816394874924}},
  {65536, (GaloisInteger){15132804493885713016, 7997468840100395569}}
};


__constant__ uint64_t gjk[MAXDEGREE/2];
__constant__ uint64_t invgjk[MAXDEGREE/2];

// Returns the position of the highest bit
int hob (int num)
{
    if (!num)
        return 0;

    int ret = 1;

    while (num >>= 1)
        ret++;

    return ret-1;
}
////////////////////////////////////////////////////////////////////////////////
// Special operators for arithmetic mod 0xFFFFFFFF00000001
////////////////////////////////////////////////////////////////////////////////
//
__host__ __device__ uint64_t s_rem (uint64_t a){
  // Modular reduction of a 64 bits integer
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = 0, x1 = (a >> 32), x0 = a-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  // uint64_t x3 = 0;
  // uint64_t x2 = 0;

  uint64_t x1 = (a >> 32);
  uint64_t x0 = (a & UINT32_MAX);

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = ((x1<<32) + x0);

  if(res >= PRIMEP){
    res -= PRIMEP;
    x1 = (res >> 32);
    x0 = (res & UINT32_MAX);
    res = ((x1<<32) + x0);
  }

  return res;
}

uint64_t mul64hi(uint64_t a, uint64_t b) {
     unsigned __int128 prod =  a * (unsigned __int128)b;
     return prod >> 64;
 }

__host__ __device__  uint64_t s_mul(uint64_t a, uint64_t b){
  // Multiply and reduce a and b by prime 0xFFFFFFFF00000001
  // 4294967295L == UINT64_MAX - P

#ifdef __CUDA_ARCH__
  const uint64_t cHi = __umul64hi(a,b);
#else
  const uint64_t cHi = mul64hi(a,b);
#endif
  const uint64_t cLo = a * b;


  // Split in 32 bits words
  const uint64_t x3 = (cHi >> 32);
  const uint64_t x2 = (cHi & UINT32_MAX);
  const uint64_t x1 = (cLo >> 32);
  const uint64_t x0 = (cLo & UINT32_MAX);

  const uint64_t X1 = (x1<<32);
  const uint64_t X2 = (x2<<32);

  // Reduce
  // res = ((w_2 + w_1) << 32) - w_3 -w_2 + w_0
  return s_sub(
    s_add(s_add(X1, X2), x0),
    s_add(x3, x2)
    );
}

__host__ __device__  uint64_t s_add(uint64_t a,uint64_t b){
  // Add and reduce a and b by 0xFFFFFFFF00000001
  // I assume that a < PRIMEP and b < PRIMEP
  b = PRIMEP - b;
  // if ( a>=b )
  //   return a - b;
  // else
  //   return PRIMEP - b + a;
  return PRIMEP*(a < b) - b + a;
}

__host__ __device__ uint64_t s_sub(uint64_t a,uint64_t b){
  // Computes a-b % P

  return PRIMEP*(a < b) - b + a;
}

// Computes a ^ b mod p
__host__ __device__ uint64_t s_fast_pow(uint64_t a, uint64_t b){
  uint64_t r = 1;
  uint64_t s = a;
  while(b > 0){
    if(b % 2 == 1)
      r = s_mul(r, s);
    s = s_mul(s, s);
    b /= 2;
  }
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Galois Integer operations
////////////////////////////////////////////////////////////////////////////////

// Addition
__host__ __device__ GaloisInteger GIAdd(GaloisInteger a, GaloisInteger b)
{
    GaloisInteger c;
    c.re = s_add(a.re, b.re);
    c.imag = s_add(a.imag, b.imag);
    return c;
}


__host__ __device__ GaloisInteger GISub(GaloisInteger a, GaloisInteger b)
{
    GaloisInteger c;
    c.re = s_sub(a.re, b.re);
    c.imag = s_sub(a.imag, b.imag);
    return c;
}

// GaloisInteger multiplication
__host__ __device__ GaloisInteger GIMul(GaloisInteger a, GaloisInteger b)
{
	// Karatsuba method
	// https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
	// 
	// S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
	// A=S1−S2 and B=S3−S1−S2.
	// 

    Integer s1 = s_mul(a.re, b.re);
    Integer s2 = s_mul(a.imag, b.imag);
    Integer s3 = s_mul(
    	s_add(a.re, a.imag),
    	s_add(b.re, b.imag)
    	);

    GaloisInteger c;
    c.re = s_sub(s1, s2);
    c.imag = s_sub(
    	s3,
    	s_add(s1, s2)
    	);
    
    return c;
}

// Computes a ^ b mod p
__host__ __device__ GaloisInteger GIFastPow(GaloisInteger a, int b){
  GaloisInteger r = {1, 0};
  GaloisInteger s = a;
  while(b > 0){
    if(b % 2 == 1)
      r = GIMul(r, s);
    s = GIMul(s, s);
    b /= 2;
  }
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Discrete Galois transform
////////////////////////////////////////////////////////////////////////////////

__global__ void dgt(
  GaloisInteger* data,
  const int stride,
  const int N,
  const int nresidues,
  const int direction, // Forward or Inverse
  const uint64_t scale
  ){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / (N/2);
  const int l = tid % (N/2);
  const int m = N / (2<<stride);
  const int log2k = rint(log2((double)N));

  if(tid < (N/2) * nresidues){
    const int j = l / (N/(2*m));
    const int i = j + (l % (N/(2*m)))*2*m;
    GaloisInteger xi = data[i + rid * N];
    GaloisInteger xim = data[i + m + rid * N];

    if(direction == FORWARD){
      const GaloisInteger a = {s_fast_pow(gjk[j], N >> (log2k - stride)), 0};
      // printf("m: %d, i: %d, j: %d, jk: %d, a: %llu, xi: %llu, xim: %llu\n", m, i, j, (log2k - stride - 1), a.re, xi.re, xim.re);
      
      data[i + rid * N] = GIAdd(xi, xim); 
      data[i + m + rid * N] = GIMul(a, GISub(xi, xim)); 
    }else{
      // Inverse
      const GaloisInteger a = {s_fast_pow(invgjk[j], N >> (log2k - stride)), 0};
      // printf("m: %d, i: %d, j: %d, jk: %d, a: %llu, xi: %llu, xim: %llu\n", m, i, j, stride, a.re, xi.re, xim.re);
      
      GaloisInteger new_xi = GIAdd(xi, GIMul(a, xim)); 
      GaloisInteger new_xim = GISub(xi, GIMul(a, xim)); 

      // Normalization on the last stride
      if(m == 1){
        new_xi = {s_mul(new_xi.re, scale), s_mul(new_xi.imag, scale)};
        new_xim = {s_mul(new_xim.re, scale), s_mul(new_xim.imag, scale)};
      }
      
      // Write to global memory
      data[i + rid * N] = new_xi;
      data[i + m + rid * N] = new_xim;
    }

  }

  return;
}

__host__ void execute_dgt(
  GaloisInteger* data,
  const int N,
  const int nresidues,
  const int direction // Forward or Inverse
  ){

  // To-do: Assert N is a power of 2
  const int halfsize = (N / 2) * nresidues;
  const int halfADDGRIDXDIM = (halfsize%64 == 0? halfsize/64 : halfsize/64 + 1);
  const dim3 halfgridDim(halfADDGRIDXDIM);
  const dim3 blockDim(64);

  for(int stride = 0; stride < hob(N); stride++){
    dgt<<< halfgridDim, blockDim>>>(
      data,
      (direction == FORWARD? stride : hob(N) - stride - 1),
      N,
      nresidues,
      direction,
      conv<uint64_t>(NTL::InvMod(to_ZZ(N), to_ZZ(PRIMEP)))
      );
  }
}


/**
 * @brief      Fold that as proposed by Badawi.
 *             folded_data[j] = {data[j], data[j+N/2]}.
 *
 * @param      folded_data  The folded data
 * @param      data         The data
 * @param[in]  N            { parameter_description }
 * @param[in]  nresidues    The nresidues
 */
__host__ void fold_dgt(
  GaloisInteger *folded_data,
  GaloisInteger *data,
  const int N,
  const int nresidues){
  
  /* 
   * This method isn't need on cuPoly. Such operation may be executed on
   *  poly_rns();
   */

  for(int rid = 0; rid < nresidues; rid++)
    for(int cid = 0; cid < N/2; cid++)
      folded_data[cid + rid * (N/2)] = {
        data[cid + rid * N].re,
        data[(cid + N/2) + rid * N].re 
      };

}

/**
 * @brief      Unfold that as proposed by Badawi.
 *             folded_data[j] = {data[j], data[j+N/2]}.
 *
 * @param      data         The data
 * @param      folded_data  The folded data
 * @param[in]  N            { parameter_description }
 * @param[in]  nresidues    The nresidues
 */
__host__ void unfold_dgt(
  GaloisInteger *data,
  GaloisInteger *folded_data,
  const int N,
  const int nresidues){
  
  /* 
   * This method isn't need on cuPoly. Such operation may be executed on
   *  poly_irns();
   */

  for(int rid = 0; rid < nresidues; rid++)
    for(int cid = 0; cid < N/2; cid++){
      data[cid + rid * N] = {
        folded_data[cid + rid * (N/2)].re,
        0,
      };
      data[(cid + N/2) + rid * N] = {
        folded_data[cid + rid * (N/2)].imag,
        0
      };
    }

}

__global__ void twist_dgt(
  GaloisInteger* data,
  const int N,
  const int nresidues,
  const GaloisInteger nthroot){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int cid = tid % N;

  if(tid < N * nresidues)
    data[tid] = GIMul(data[tid], GIFastPow(nthroot, cid));
}


__host__ void execute_twist_dgt(
  GaloisInteger *data,
  const int N,
  const int nresidues,
  const int direction){

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%32 == 0? size/32 : size/32 + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(32);

  twist_dgt<<< gridDim, blockDim>>>(
    data,
    N,
    nresidues,
    (direction == FORWARD? NTHROOT[N] : INVNTHROOT[N]) );
}


__global__ void add_dgt(
  GaloisInteger *c_data,
  const GaloisInteger *a_data,
  const GaloisInteger *b_data,
  const int size){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid < size)
    c_data[tid] = GIAdd(a_data[tid], b_data[tid]);
}


__host__ void execute_add_dgt(
  GaloisInteger *c_data,
  const GaloisInteger *a_data,
  const GaloisInteger *b_data,
  const int N,
  const int nresidues){

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%32 == 0? size/32 : size/32 + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(32);

  add_dgt<<< gridDim, blockDim>>>(
    c_data,
    a_data,
    b_data,
    size);
}

__global__ void mul_dgt(
  GaloisInteger *c_data,
  const GaloisInteger *a_data,
  const GaloisInteger *b_data,
  const int size){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid < size)
    c_data[tid] = GIMul(a_data[tid], b_data[tid]);
}


__host__ void execute_mul_dgt(
  GaloisInteger *c_data,
  const GaloisInteger *a_data,
  const GaloisInteger *b_data,
  const int N,
  const int nresidues){

  const int size = N * nresidues;
  const int ADDGRIDXDIM = (size%32 == 0? size/32 : size/32 + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(32);

  mul_dgt<<< gridDim, blockDim>>>(
    c_data,
    a_data,
    b_data,
    size);
}


__host__ void set_const_mem(const int k){
  if(NTHROOT.count(k) < 1 || INVNTHROOT.count(k) < 1)
    std::cerr << "WARNING! I don't know the " << k << "th-root of i" << std::endl;
  assert(k <= MAXDEGREE);
  assert(k > 0);
  assert((PRIMEP-1)%k == 0);
  const uint64_t n = (PRIMEP-1)/k;
  const uint64_t g = s_fast_pow((uint64_t)PRIMITIVE_ROOT, n);
  assert(s_fast_pow(g, k) == 1);

  // Compute the matrix
  uint64_t *h_gjk = (uint64_t*) malloc (k * sizeof(uint64_t));
  uint64_t *h_invgjk = (uint64_t*) malloc (k * sizeof(uint64_t));
  
  // Forward
  for (int j = 0; j < k/2; j++)
    h_gjk[j] = s_fast_pow(g, j);
  for (int j = 0; j < k/2; j++)
    h_invgjk[j] = s_fast_pow(g, (k - j));

  cudaMemcpyToSymbol(
    gjk,
    h_gjk,
    k/2 * sizeof(uint64_t)
    );
  cudaCheckError()

  cudaMemcpyToSymbol(
    invgjk,
    h_invgjk,
    k/2 * sizeof(uint64_t)
    );
  cudaCheckError()

  free(h_gjk);
  free(h_invgjk);
}