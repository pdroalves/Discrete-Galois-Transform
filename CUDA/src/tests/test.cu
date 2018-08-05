#include <gtest/gtest.h>
#include <dgt/dgt.h>

#define NTESTS 1000

uint64_t rand64() {
    // Assuming RAND_MAX is 2^31.
    uint64_t r = rand();
    r = r<<30 | rand();
    r = r<<30 | rand();
    return r;
}

////////////////////////////////////////////////////////////////////////////////
// BasicModularArithmetic
////////////////////////////////////////////////////////////////////////////////
TEST(BasicModularArithmetic, Add) {
  for(unsigned int i = 0; i < NTESTS; i++){

    __uint128_t a = rand64();
    __uint128_t b = rand64();

    ASSERT_EQ(
      s_add((uint64_t)a, (uint64_t)b),
      (a + b) % PRIMEP
      ) << "Failure! " << (uint64_t)a << " + " << (uint64_t)b;
  }
}

TEST(BasicModularArithmetic, Sub) {
  for(unsigned int i = 0; i < NTESTS; i++){

    __uint128_t a = rand64();
    __uint128_t b = rand64();

    ASSERT_EQ(
      s_sub((uint64_t)a, (uint64_t)b),
      (a > b ? (a - b) % PRIMEP : (PRIMEP + a - b) % PRIMEP)
      ) << "Failure! " << (uint64_t)a << " - " << (uint64_t)b;
  }
}

TEST(BasicModularArithmetic, Mul) {
  for(unsigned int i = 0; i < NTESTS; i++){

    __uint128_t a = rand64();
    __uint128_t b = rand64();

    ASSERT_EQ(
      s_mul((uint64_t)a, (uint64_t)b),
      (a * b) % PRIMEP
      ) << "Failure! " << (uint64_t)a << " * " << (uint64_t)b;
  }
}

////////////////////////////////////////////////////////////////////////////////
// GaloisIntegerArithmetic
////////////////////////////////////////////////////////////////////////////////

TEST(GaloisIntegerArithmetic, Add) {
  for(unsigned int i = 0; i < NTESTS; i++){

    GaloisInteger a = {rand64(), rand64()};
    GaloisInteger b = {rand64(), rand64()};
    GaloisInteger expected = {
      s_add(a.re, b.re),
      s_add(a.imag, b.imag)
    };
    GaloisInteger received = GIAdd(a, b);

    ASSERT_EQ(
      received.re,
      expected.re
      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
    ASSERT_EQ(
      received.imag,
      expected.imag
      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
  }
}

TEST(GaloisIntegerArithmetic, Sub) {
  for(unsigned int i = 0; i < NTESTS; i++){

    GaloisInteger a = {rand64(), rand64()};
    GaloisInteger b = {rand64(), rand64()};
    GaloisInteger expected = {
      s_sub(a.re, b.re),
      s_sub(a.imag, b.imag)
    };
    GaloisInteger received = GISub(a, b);

    ASSERT_EQ(
      received.re,
      expected.re
      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
    ASSERT_EQ(
      received.imag,
      expected.imag
      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
  }
}

TEST(GaloisIntegerArithmetic, Mul) {
  // for(unsigned int i = 0; i < NTESTS; i++){

  //   GaloisInteger a = {rand64(), rand64()};
  //   GaloisInteger b = {rand64(), rand64()};
  //   GaloisInteger expected = {
  //     s_sub(
  //       s_mul(a.re, b.re),
  //       s_mul(a.imag, b.imag)
  //       ),
  //     s_add(
  //       s_mul(a.imag, b.re),
  //       s_mul(a.re, b.imag)
  //       ),
  //   };
  //   GaloisInteger received = GIMul(a, b);

  //   ASSERT_EQ(
  //     received.re,
  //     expected.re
  //     ) << "Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re << ", " << b.imag << ")";
  //   ASSERT_EQ(
  //     received.imag,
  //     expected.imag
  //     ) << "Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re << ", " << b.imag << ")";
  // }

  GaloisInteger a = {17293822564807737345, 0};
  GaloisInteger b = {18446744069414584305, 0};
  GaloisInteger c = {4294967295, 0};

  ASSERT_EQ(GIMul(a, b).re, c.re);

}

////////////////////////////////////////////////////////////////////////////////
// DGT
////////////////////////////////////////////////////////////////////////////////

TEST(DGT, Transform) {
  for(int nresidues = 1; nresidues < 20; nresidues++)
    for(int n = 32; n <= 8192; n *= 2){
      set_const_mem(n);

      // Host
      GaloisInteger *h_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      for(int i = 0; i < nresidues*n; i++)
        // h_data[i] = {i, 0};
        h_data[i] = {rand64() % PRIMEP, 0};

      // Print
      // std::cout << "Original: " << std::endl;
      // for(int i = 0; i < n; i++)
      //   std::cout << h_data[i].re << "," << std::endl;
        // std::cout << h_data[i].re << " + " << h_data[i].imag << "j," << std::endl;

      // Device
      GaloisInteger *d_data;
      cudaMalloc((void**)&d_data,nresidues * n * sizeof(GaloisInteger));
      cudaCheckError();

      // Copy
      cudaMemcpy(d_data, h_data, nresidues * n * sizeof(GaloisInteger), cudaMemcpyHostToDevice);
      cudaCheckError();

      // DGT
      execute_dgt(d_data, n, nresidues, FORWARD);
      execute_dgt(d_data, n, nresidues, INVERSE);

      // Copy
      GaloisInteger *h_result = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      cudaMemcpy(h_result, d_data, nresidues * n * sizeof(GaloisInteger), cudaMemcpyDeviceToHost);
      cudaCheckError();

      // Print
      // std::cout << "Transformed: " << std::endl;
      for(int i = 0; i < nresidues*n; i++){
        // std::cout <<h_result[i].re << " + " << h_result[i].imag  << std::endl;
        ASSERT_EQ(h_data[i].re, h_result[i].re) << "Fail at index " << i;
        ASSERT_EQ(h_data[i].imag, h_result[i].imag) << "Fail at index " << i;
      }

      // Release
      free(h_result);
      free(h_data);
      cudaFree(d_data);
    }
}

int main(int argc, char **argv) {
  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  
  srand(42);

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
