#include <cstdio>

#include <cuda_runtime.h>



// 转置前的矩阵存储在dev_A中，矩阵大小为M*N，转置后的数据存储在dev_B中
    // const dim3 block_size(32, 32);
    // const dim3 grid_size(N/32, M/32);
__global__ void matrix_trans_shm(int* dev_A, int M, int N, int* dev_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_data[32][32];
    if (row < M && col <N) {
        s_data[threadIdx.x][threadIdx.y^threadIdx.x] = dev_A[row * N + col];
        __syncthreads();
        int b_row = blockIdx.x * blockDim.x + threadIdx.y;
        int b_col = blockIdx.y * blockDim.y + threadIdx.x;

        if (b_row < N && b_col < M) {
            dev_B[b_row * M + b_col] = s_data[threadIdx.y][threadIdx.y^threadIdx.x];
        }

    }
    
}

// __global__ void matrix_trans_shm(int* dev_A, int M, int N, int* dev_B) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
  
//     __shared__ int s_data[32][32];

//     if (row < M && col < N) {
//       s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
//       __syncthreads();
//       int n_col = blockIdx.y * blockDim.y + threadIdx.x;   // 这里B是M列N行，所以 Block y为col, x为 row
//       int n_row = blockIdx.x * blockDim.x + threadIdx.y;
//       if (n_col < M && n_row < N) {
//         dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
//       }
//     }
    
// }

// __global__ void matrix_trans_shm_padding(int* dev_A, int M, int N, int* dev_B) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
  
//     // 每个block处理32*32的矩阵块，尾部padding来避免bank conflict
//     __shared__ int s_data[32][33];
  
//     if (row < M && col < N) {
//       s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];
//       __syncthreads();
//       int n_col = blockIdx.y * blockDim.y + threadIdx.x;
//       int n_row = blockIdx.x * blockDim.x + threadIdx.y;
//       if (n_col < M && n_row < N) {
//         dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];
//       }
//     }
//   }

// __global__ void matrix_trans_swizzling(int* dev_A, int M, int N, int* dev_B) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
  
//     __shared__ int s_data[32][32];
  
//     if (row < M && col < N) {
//       // 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)
//       // 其映射的物理存储位置位置(row=x,col=x^y)
//       s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];
//       __syncthreads();
//       int n_col = blockIdx.y * blockDim.y + threadIdx.x;
//       int n_row = blockIdx.x * blockDim.x + threadIdx.y;
//       if (n_row < N && n_col < M) {
//         // 从共享内存的逻辑坐标(row=y,col=x)读取数据
//         // 其映射的物理存储位置(row=y,col=x^y)
//         dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
//       }
//     }
//   }

int main() {
    const int M = 1024; //矩阵行
    const int N = 2048; //矩阵列


    const int size = M * N * sizeof(int);
    // 分配设备内存
    int *dev_A, *dev_B;
    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);

    // 初始化矩阵
    int *h_A = (int*)malloc(size);
    for (int i = 0; i < M * N; i++) {
        h_A[i] = i;
    }
    int *h_B = (int*)malloc(size);
    for (int i = 0; i < M * N; i++) {
        h_B[i] = 0;
    }
    cudaMemcpy(dev_A, h_A, size, cudaMemcpyHostToDevice);
    const dim3 block_size(32, 32);
    const dim3 grid_size(N/32, M/32);
    matrix_trans_shm<<<grid_size, block_size>>>(dev_A, M, N, dev_B);
    // Copy the result back from device to host
    cudaMemcpy(h_B, dev_B, size, cudaMemcpyDeviceToHost);

    // Print part of the original matrix h_A
    printf("Original Matrix (h_A) - first 32x32 elements:\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            printf("%d\t", h_A[i * N + j]);
        }
        printf("\n");
    }

    // Print part of the transposed matrix h_B
    printf("\nTransposed Matrix (h_B) - first 32x32 elements:\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d\t", h_B[i * M + j]);
        }
        printf("\n");
    }
    free(h_A);
    free(h_B);
    cudaFree(dev_A);
    cudaFree(dev_B);
}