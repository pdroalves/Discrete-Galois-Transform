#include <cuda.h>
#include <iostream>
#include <dgt/dgt.h>
#include <assert.h>

#include <NTL/ZZ.h>
NTL_CLIENT

#define MAXDEGREE (int)8192 // REMOVE THIS
 __constant__ uint64_t gjk[MAXDEGREE/2];
 __constant__ uint64_t invgjk[MAXDEGREE/2];
 // __constant__ uint64_t PROOT; // g^k mod p


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

////////////////////////////////////////////////////////////////////////////////
// Discrete Galois transform
////////////////////////////////////////////////////////////////////////////////

__global__ void dgt(
  GaloisInteger* data,
  const int stride,
  const int N,
  const int nresidues,
  const int direction // Forward or Inverse
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
      
      data[i + rid * N] = GIAdd(xi, GIMul(a, xim)); 
      data[i + m + rid * N] = GISub(xi, GIMul(a, xim)); 

    }

  }

  return;
}

__global__ void normalize_dgt(
  GaloisInteger* data,
  const int N,
  const int nresidues,
  const uint64_t scale){

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < N * nresidues)
    data[tid] = {s_mul(data[tid].re, scale), s_mul(data[tid].imag, scale)};
}

__host__ void execute_dgt(
  GaloisInteger* data,
  const int N,
  const int nresidues,
  const int direction // Forward or Inverse
  ){

  // To-do: Assert N is a power of 2
  const int halfsize = (N / 2) * nresidues;
  const int fullsize = N * nresidues;
  const int halfADDGRIDXDIM = (halfsize%32 == 0? halfsize/32 : halfsize/32 + 1);
  const int fullADDGRIDXDIM = (fullsize%32 == 0? fullsize/32 : fullsize/32 + 1);
  const dim3 halfgridDim(halfADDGRIDXDIM);
  const dim3 fullgridDim(fullADDGRIDXDIM);
  const dim3 blockDim(32);

  for(int stride = 0; stride < hob(N); stride++)
    dgt<<< halfgridDim, blockDim>>>(
      data,
      (direction == FORWARD? stride : hob(N) - stride - 1),
      N,
      nresidues,
      direction );
  if(direction == INVERSE)
    normalize_dgt<<< fullgridDim, blockDim>>>(
      data,
      N,
      nresidues,
      conv<uint64_t>(NTL::InvMod(to_ZZ(N), to_ZZ(PRIMEP))));
}

__host__ void set_const_mem(const int k){
  assert(k <= MAXDEGREE);
  assert((PRIMEP-1)%k == 0);
  const uint64_t n = (PRIMEP-1)/k;
  const uint64_t g = s_fast_pow((uint64_t)PRIMITIVE_ROOT, n);
  assert(s_fast_pow(g, k) == 1);

  // std::cout << "k: " << k << ", n: " << n << ", g: " << g << std::endl;

  // Compute the matrix
  uint64_t *h_gjk = (uint64_t*) malloc (k * sizeof(uint64_t));
  uint64_t *h_invgjk = (uint64_t*) malloc (k * sizeof(uint64_t));
  
  // Forward
  for (int j = 0; j < k/2; j++)
    h_gjk[j] = s_fast_pow(g, j);
  for (int j = 0; j < k/2; j++)
    // h_invgjk[j] = conv<uint64_t>(NTL::InvMod(to_ZZ(s_fast_pow(g, j)), to_ZZ(PRIMEP)));
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

  // cudaMemcpyToSymbol(
  //   PROOT,
  //   &g,
  //   sizeof(uint64_t)
  //   );
  // cudaCheckError()

  free(h_gjk);
  free(h_invgjk);
}