#ifndef DGT_H
#define DGT_H
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////
typedef uint64_t Integer;
typedef struct GaloisInteger{
	Integer re;
	Integer imag;
	inline bool operator==(const GaloisInteger& b){
		return (re == b.re) * (imag == b.imag);
	};
} GaloisInteger;
enum dgt_direction{FORWARD, INVERSE};
#define PRIMEP (uint64_t)0xFFFFFFFF00000001
#define PRIMITIVE_ROOT (int)7

#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if( e == cudaErrorInvalidDevicePointer)   \
   fprintf(stderr, "Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 else if(e != cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
    exit(1);                                                                 \
 }                                                                      \
}
////////////////////////////////////////////////////////////////////////////////

__host__ __device__  uint64_t s_add(uint64_t a,uint64_t b);
__host__ __device__ uint64_t s_sub(uint64_t a,uint64_t b);
__host__ __device__  uint64_t s_mul(uint64_t a, uint64_t b);
__host__ __device__ GaloisInteger GIAdd(GaloisInteger a, GaloisInteger b);
__host__ __device__ GaloisInteger GISub(GaloisInteger a, GaloisInteger b);
__host__ __device__ GaloisInteger GIMul(GaloisInteger a, GaloisInteger b);
__host__ void execute_dgt(
  GaloisInteger* data,
  const int N,
  const int nresidues,
  const int direction // Forward or Inverse
  );
__host__ void set_const_mem(const int k);

#endif