#include <gtest/gtest.h>
#include <dgt/dgt.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pE.h>

NTL_CLIENT
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
    // for(int n = 32; n <= 8192; n *= 2){
    for(int n = 256; n <= 256; n *= 2){
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


TEST(DGT, Mul) {
  ZZ_p::init(to_ZZ(PRIMEP));

  for(int nresidues = 1; nresidues < 2; nresidues++)
    for(int n = 512; n <= 512; n *= 2){

      ZZ_pX NTL_Phi;
      NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
      NTL::SetCoeff(NTL_Phi, n, conv<ZZ_p>(1));
      ZZ_pE::init(NTL_Phi);

      // Setup
      set_const_mem(n/2);

      // Host
      GaloisInteger *h_a_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      GaloisInteger *h_b_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      for(int i = 0; i < nresidues*n; i++){
        // h_a_data[i] = {i, 0};
        // h_b_data[i] = {i, 0};
        h_a_data[i] = {(rand64()/2) % PRIMEP, 0}; // NTL can't handle 64 bits
        h_b_data[i] = {(rand64()/2) % PRIMEP, 0}; // NTL can't handle 64 bits
      }

      // Fold
      GaloisInteger *h_folded_a_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      fold_dgt(h_folded_a_data, h_a_data, n, nresidues);
      GaloisInteger *h_folded_b_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      fold_dgt(h_folded_b_data, h_b_data, n, nresidues);

      // Alloc
      GaloisInteger *d_a_data, *d_b_data, *d_c_data;
      cudaMalloc((void**)&d_a_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();
      cudaMalloc((void**)&d_b_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();
      cudaMalloc((void**)&d_c_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();

      // Copy
      cudaMemcpy(d_a_data, h_folded_a_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyHostToDevice);
      cudaCheckError();
      cudaMemcpy(d_b_data, h_folded_b_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyHostToDevice);
      cudaCheckError();

      // Twist
      execute_twist_dgt(d_a_data, (n/2), nresidues, FORWARD);
      execute_twist_dgt(d_b_data, (n/2), nresidues, FORWARD);

      // DGT
      execute_dgt(d_a_data, (n/2), nresidues, FORWARD);
      execute_dgt(d_b_data, (n/2), nresidues, FORWARD);

      // Mul
      execute_mul_dgt(d_c_data, d_a_data, d_b_data, (n/2), nresidues);

      // IDGT
      execute_dgt(d_c_data, (n/2), nresidues, INVERSE);

      // Twist
      execute_twist_dgt(d_c_data, (n/2), nresidues, INVERSE);

      // Copy
      GaloisInteger *h_folded_c_data = (GaloisInteger*) malloc (nresidues * (n/2) * sizeof(GaloisInteger));
      cudaMemcpy(h_folded_c_data, d_c_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyDeviceToHost);
      cudaCheckError();

      // Unfold
      GaloisInteger *h_c_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      unfold_dgt(h_c_data, h_folded_c_data, n, nresidues);

      // Verify
      for(int rid = 0; rid < nresidues; rid++){
        ZZ_pX ntl_a, ntl_b, ntl_c;
        for(int i = 0; i < n; i++)
          SetCoeff(ntl_a, i, conv<ZZ_p>(h_a_data[i + rid * n].re));
        for(int i = 0; i < n; i++)
          SetCoeff(ntl_b, i, conv<ZZ_p>(h_b_data[i + rid * n].re));
        ntl_c = ntl_a * ntl_b % conv<ZZ_pX>(NTL_Phi) ;

        for(int i = 0; i < n; i++)
          ASSERT_EQ(h_c_data[i + rid * n].re, conv<uint64_t>(coeff(ntl_c, i))) << "Fail at index " << i << " of rid " << rid;
      }
      // Release
      free(h_a_data);
      free(h_b_data);
      free(h_c_data);
      free(h_folded_a_data);
      free(h_folded_b_data);
      free(h_folded_c_data);
      cudaFree(d_a_data);
      cudaFree(d_b_data);
      cudaFree(d_c_data);
    }
}

TEST(DGT, Add) {
  ZZ_p::init(to_ZZ(PRIMEP));

  for(int nresidues = 1; nresidues < 2; nresidues++)
    for(int n = 512; n <= 512; n *= 2){

      ZZ_pX NTL_Phi;
      NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
      NTL::SetCoeff(NTL_Phi, n, conv<ZZ_p>(1));
      ZZ_pE::init(NTL_Phi);

      // Setup
      set_const_mem(n/2);

      // Host
      GaloisInteger *h_a_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      GaloisInteger *h_b_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      for(int i = 0; i < nresidues*n; i++){
        // h_a_data[i] = {i, 0};
        // h_b_data[i] = {i, 0};
        h_a_data[i] = {(rand64()/2) % PRIMEP, 0}; // NTL can't handle 64 bits
        h_b_data[i] = {(rand64()/2) % PRIMEP, 0}; // NTL can't handle 64 bits
      }

      // Fold
      GaloisInteger *h_folded_a_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      fold_dgt(h_folded_a_data, h_a_data, n, nresidues);
      GaloisInteger *h_folded_b_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      fold_dgt(h_folded_b_data, h_b_data, n, nresidues);

      // Alloc
      GaloisInteger *d_a_data, *d_b_data, *d_c_data;
      cudaMalloc((void**)&d_a_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();
      cudaMalloc((void**)&d_b_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();
      cudaMalloc((void**)&d_c_data, nresidues * (n/2) * sizeof(GaloisInteger));
      cudaCheckError();

      // Copy
      cudaMemcpy(d_a_data, h_folded_a_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyHostToDevice);
      cudaCheckError();
      cudaMemcpy(d_b_data, h_folded_b_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyHostToDevice);
      cudaCheckError();

      // Twist
      execute_twist_dgt(d_a_data, (n/2), nresidues, FORWARD);
      execute_twist_dgt(d_b_data, (n/2), nresidues, FORWARD);

      // DGT
      execute_dgt(d_a_data, (n/2), nresidues, FORWARD);
      execute_dgt(d_b_data, (n/2), nresidues, FORWARD);

      // Mul
      execute_add_dgt(d_c_data, d_a_data, d_b_data, (n/2), nresidues);

      // IDGT
      execute_dgt(d_c_data, (n/2), nresidues, INVERSE);

      // Twist
      execute_twist_dgt(d_c_data, (n/2), nresidues, INVERSE);

      // Copy
      GaloisInteger *h_folded_c_data = (GaloisInteger*) malloc (nresidues * (n/2) * sizeof(GaloisInteger));
      cudaMemcpy(h_folded_c_data, d_c_data, nresidues * (n/2) * sizeof(GaloisInteger), cudaMemcpyDeviceToHost);
      cudaCheckError();

      // Unfold
      GaloisInteger *h_c_data = (GaloisInteger*) malloc (nresidues * n * sizeof(GaloisInteger));
      unfold_dgt(h_c_data, h_folded_c_data, n, nresidues);

      // Verify
      for(int rid = 0; rid < nresidues; rid++){
        ZZ_pX ntl_a, ntl_b, ntl_c;
        for(int i = 0; i < n; i++)
          SetCoeff(ntl_a, i, conv<ZZ_p>(h_a_data[i + rid * n].re));
        for(int i = 0; i < n; i++)
          SetCoeff(ntl_b, i, conv<ZZ_p>(h_b_data[i + rid * n].re));
        ntl_c = (ntl_a + ntl_b) % conv<ZZ_pX>(NTL_Phi) ;

        for(int i = 0; i < n; i++)
          ASSERT_EQ(h_c_data[i + rid * n].re, conv<uint64_t>(coeff(ntl_c, i))) << "Fail at index " << i << " of rid " << rid;
      }
      // Release
      free(h_a_data);
      free(h_b_data);
      free(h_c_data);
      free(h_folded_a_data);
      free(h_folded_b_data);
      free(h_folded_c_data);
      cudaFree(d_a_data);
      cudaFree(d_b_data);
      cudaFree(d_c_data);
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
