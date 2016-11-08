#include <stdio.h>

#define N 4096
#define THREADS 256
#define STEPCOUNT 8
#define INT unsigned int
#define ceildiv(a, b) ((a+b-1)/b)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

INT add_host(INT* h_arr, INT size) {
    INT res =0;
    for(INT i=0; i<size; i++) {
        res += h_arr[i];
    }
    return res;
}


__global__ void add_gpu_mine(INT* d_arr, INT step, INT count, INT size) {
    INT x = threadIdx.x + blockDim.x*blockIdx.x;
    INT localres = 0;
    for(INT i=0; i<count; i++) {
        INT idx = step*i + x;
        if(idx < size)
            localres += d_arr[idx];
    }
    d_arr[x] = localres;
}

int main() {
    INT* h_arr = (INT*)malloc(N*sizeof(INT));
    for(INT i=0; i<N; i++) {
        h_arr[i] = i+1;    
    }
    
    INT h_res = add_host(h_arr, N);
    
    INT stepsize = ceildiv(N, STEPCOUNT);
    INT n=N;
    INT* d_arr;
    gpuErrchk(cudaMalloc((void**)&d_arr, N*sizeof(INT)));
    gpuErrchk(cudaMemcpy(d_arr, h_arr, N*sizeof(INT), cudaMemcpyHostToDevice));
    
    
    while(stepsize > 32) {
        dim3 threads(THREADS, 1, 1);
        dim3 blocks(ceildiv(stepsize, THREADS), 1, 1);
        add_gpu_mine<<<blocks, threads>>>(d_arr, stepsize, STEPCOUNT, n);
        n = stepsize;
        stepsize = ceildiv(n, STEPCOUNT);
    }
    
    gpuErrchk(cudaMemcpy(h_arr, d_arr, N*sizeof(INT), cudaMemcpyDeviceToHost));
    INT d_res = add_host(h_arr, n);
    
    printf("host: %d\ndevice: %d\nref: %d\n", h_res, d_res, (N*(N+1))/2);
    return 0;    
}