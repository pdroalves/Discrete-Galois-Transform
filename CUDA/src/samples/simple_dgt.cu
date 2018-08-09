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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>

 int main(int argc, char* argv[]){

  const int n = 4096;
  const int nresidues = 5; // 5 * 64 = 320 bits?

  // Setup
  set_const_mem(n/2);
  GaloisInteger *d_data;
  cudaMalloc((void**)&d_data,nresidues * (n/2) * sizeof(GaloisInteger));
  cudaCheckError();
  
  cudaProfilerStart();
  execute_dgt(d_data, n, nresidues, FORWARD);
  execute_dgt(d_data, n, nresidues, INVERSE);
  cudaProfilerStop();

  cudaFree(d_data);
}
