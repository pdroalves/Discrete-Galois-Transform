// cuPoly - A GPGPU-based library for doing polynomial arithmetic on RLWE-based cryptosystems
// Copyright (C) 2017, Pedro G. M. R. Alves - pedro.alves@ic.unicamp.br

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <dgt/dgt.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <cufft.h>
#include <assert.h>
#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100

__host__ double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

__host__ double runFFT(int nresidues, int n){
  // Init
  // # of RNS residues for base q
  const int batch = nresidues;
  assert(batch > 0);

  // # 1 dimensional FFT
  const int rank = 1;

  int NR[1] = {nresidues};

  // Create a plan
  cufftHandle plan;
  cufftPlanMany(
    &plan, rank, NR,
    NULL, 1, n,  //advanced data layout, NULL shuts it off
    NULL, 1, n,  //advanced data layout, NULL shuts it off
    CUFFT_Z2Z, batch);

  // alloc memory
  cufftDoubleComplex *d_data;
  cudaMalloc((void**)&d_data, nresidues * n * sizeof(cufftDoubleComplex));
  cudaCheckError();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Benchmark
  cudaEventRecord(start, 0);
  for(int i = 0; i < N; i++)
    cufftExecZ2Z( 
      plan,
      d_data,
      d_data,
      CUFFT_FORWARD
    );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float diff;
  cudaEventElapsedTime(&diff, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cufftDestroy(plan);
  cudaFree(d_data);
  return diff/N;
}

__host__ double runIFFT(int nresidues, int n){
  // Init
  // # of RNS residues for base q
  const int batch = nresidues;
  assert(batch > 0);

  // # 1 dimensional FFT
  const int rank = 1;

  int NR[1] = {nresidues};

  // Create a plan
  cufftHandle plan;
  cufftPlanMany(
    &plan, rank, NR,
    NULL, 1, n,  //advanced data layout, NULL shuts it off
    NULL, 1, n,  //advanced data layout, NULL shuts it off
    CUFFT_Z2Z, batch);

  // alloc memory
  cufftDoubleComplex *d_data;
  cudaMalloc((void**)&d_data, nresidues * n * sizeof(cufftDoubleComplex));
  cudaCheckError();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Benchmark
  cudaEventRecord(start, 0);
  for(int i = 0; i < N; i++)
    cufftExecZ2Z( 
      plan,
      d_data,
      d_data,
      CUFFT_INVERSE
    );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float diff;
  cudaEventElapsedTime(&diff, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cufftDestroy(plan);
  cudaFree(d_data);
  return diff/N;
}

__host__ double runDGT(int nresidues, int n){
  struct timespec start, stop;

  // Setup
  set_const_mem(n);
  GaloisInteger *d_data;
  cudaMalloc((void**)&d_data,nresidues * n * sizeof(GaloisInteger));
  cudaCheckError();

  // Benchmark
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N; i++)
    execute_dgt(d_data, n, nresidues, FORWARD);
  cudaDeviceSynchronize();
  clock_gettime( CLOCK_REALTIME, &stop);

  cudaFree(d_data);

  return compute_time_ms(start,stop)/N;
}

__host__ double runIDGT(int nresidues, int n){
  struct timespec start, stop;

  // Setup
  set_const_mem(n);
  GaloisInteger *d_data;
  cudaMalloc((void**)&d_data,nresidues * n * sizeof(GaloisInteger));
  cudaCheckError();

  // Benchmark
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N; i++)
    execute_dgt(d_data, n, nresidues, INVERSE);
  cudaDeviceSynchronize();
  clock_gettime( CLOCK_REALTIME, &stop);

  cudaFree(d_data);

  return compute_time_ms(start,stop)/N;
}

__host__ double runDGTMul(int nresidues, int n){
  struct timespec start, stop;

  // Setup
  set_const_mem(n/2);
  GaloisInteger *d_data;
  cudaMalloc((void**)&d_data,nresidues * (n/2) * sizeof(GaloisInteger));
  cudaCheckError();

  // Benchmark
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N; i++)
    execute_mul_dgt(d_data, d_data, d_data, (n/2), nresidues);
  cudaDeviceSynchronize();
  clock_gettime( CLOCK_REALTIME, &stop);

  cudaFree(d_data);
  return compute_time_ms(start,stop)/N;
}


__host__ double runDGTAdd(int nresidues, int n){
  struct timespec start, stop;

  // Setup
  set_const_mem(n/2);
  GaloisInteger *d_data;
  cudaMalloc((void**)&d_data,nresidues * (n/2) * sizeof(GaloisInteger));
  cudaCheckError();

  // Benchmark
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N; i++)
    execute_add_dgt(d_data, d_data, d_data, (n/2), nresidues);
  cudaDeviceSynchronize();
  clock_gettime( CLOCK_REALTIME, &stop);

  cudaFree(d_data);
  return compute_time_ms(start,stop)/N;
}

 int main(int argc, char* argv[]){
    // Output precision
    std::cout << std::fixed;
    std::cout.precision(2);

    std::vector<std::string> type_data = {
      "FFT",
      "IFFT",
      "DGT",
      "IDGT",
      "DGTMul",
    };
    std::vector<int> degrees_data = {
      2048,
      4096,
      8192
    };
    std::vector<std::vector<double>> data(type_data.size(), std::vector<double>());

    // for(int d = 16384; d <= 16384; d*=2){
    for(std::vector<int>::iterator d = degrees_data.begin(); d != degrees_data.end(); d++){
      int k;

      switch(*d){
        case 2048:
          k = 3;
        break;
        case 4096:
          k = 6;
        break;
        case 8192:
          k = 12;
        break;
        case 16384:
          k = 24;
        break;
        case 32768:
          k = 33;
        break;
        default:
          std::cout << "Which 'k' should I use?" << std::endl;
          exit(1);
      }

      for(std::vector<std::string>::iterator it = type_data.begin(); it != type_data.end(); it++){
        double diff;
        if (*it == "FFT")
          diff = runFFT(k, *d);
        else if (*it == "IFFT")
          diff = runIFFT(k, *d);
        else if (*it == "DGT")
          diff = runDGT(k, *d);
        else if (*it == "IDGT")
          diff = runIDGT(k, *d);
        else if (*it == "DGTMul")
          diff = runDGTMul(k, *d);
        else if (*it == "DGTAdd")
          diff = runDGTAdd(k, *d);
        else 
          continue;
        // else if (*it == "Reduce")
          // diff = runReduce(d, q);

        // std::cout << "Inserindo " << diff << " em " << *it << std::endl;
        data[distance(type_data.begin(), it)].push_back(diff);
      }
  
      // Release
      cudaDeviceSynchronize();
      cudaCheckError();
      cudaDeviceReset();
      cudaCheckError();
    }

    // 
    // Output
    // 
    const char separator    = ' ';
    const int nameWidth     = 10;
    const int numWidth      = 30;

    // Print degrees
    std::cout << std::left << std::setw(numWidth) << "Placeholder" << std::setw(nameWidth) << std::setfill(separator);
    for(std::vector<int>::iterator d = degrees_data.begin(); d != degrees_data.end(); d++)
      std::cout << (*d) << std::setw(nameWidth) << std::setfill(separator);
    std::cout << std::endl;

    // Print data
    for(std::vector<std::string>::iterator it = type_data.begin(); it != type_data.end(); it++){
      // Type name
      std::cout << std::left << std::setw(numWidth) << (*it) << std::setw(nameWidth) << std::setfill(separator);

      // Values
      std::vector<double> v = data[distance(type_data.begin(), it)];
      for(std::vector<double>::iterator t = v.begin(); t != v.end(); t++)
        std::cout << (*t) << std::setw(nameWidth) << std::setfill(separator);
      std::cout << std::endl;
    }

}
