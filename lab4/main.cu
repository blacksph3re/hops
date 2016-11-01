 
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define NUMTHREADS 32
#define N 129

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void dyadic(int* res, int* a, int* b, int n) {
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			res[i+N*j] = a[j]*b[i];
		}
	}
}

__global__ void dyadic_gpu(int* res, int* a, int* b, int n) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if(x < n && y < n)
        res[x+n*y] = a[y]*b[x];
}

int main() {
	srand(time(NULL));
	int* a = (int*)malloc(N*sizeof(int));
 	int* b = (int*)malloc(N*sizeof(int));

 	for(int i=0; i<N; i++) {
 		a[i] = rand()%100;
 		b[i] = rand()%100;
 	}

 	int* res = (int*)malloc(N*N*sizeof(int));
 	dyadic(res, a, b, N);
	
	// CUDA stuff
        int* d_a, *d_b, *d_res;
        gpuErrchk(cudaMalloc((void**)&d_a, N*sizeof(int)));
        gpuErrchk(cudaMalloc((void**)&d_b, N*sizeof(int)));
        gpuErrchk(cudaMalloc((void**)&d_res, N*N*sizeof(int)));
        gpuErrchk(cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
        
        dim3 threads(16, 16, 1);
        dim3 grid(ceil((float)N/(float)16), ceil((float)N/(float)16), 1);
//     (N+threads.x-1)/threads.x    
        cudaMemset(d_res, 0,N*N*sizeof(int)); 
        dyadic_gpu<<<grid,threads>>>(d_res, d_a, d_b, N);
        gpuErrchk(cudaGetLastError());
        int* resgpu = (int*)malloc(N*N*sizeof(int));
        gpuErrchk(cudaMemcpy(resgpu, d_res, N*N*sizeof(int), cudaMemcpyDeviceToHost));

 	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			printf("%d ", res[i+N*j]);
		}
		printf("\n");
	}
	printf("-----\n");
        for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			printf("%d ", resgpu[i+N*j]);
		}
		printf("\n");
	}
	printf("\ngrid size: %d, %d\n", grid.x, grid.y);
}
