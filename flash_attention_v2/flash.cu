#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define EPSILON 1e-10f

__global__
void flash_attention_v1_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
    }

__global__
void flash_attention_v2_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index
    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    // int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size_kv = Bc * d;  // size of Kj, Vj
    int tile_size_q = Br * d;  // size of Qi
    float* Qi = sram;
    float* Kj = &sram[tile_size_q];
    float* Vj = &sram[tile_size_q + tile_size_kv];
    float* S = &sram[tile_size_q + tile_size_kv * 2];

    for (int i = 0; i < Tr; i++) {
        int row_offset = qkv_offset + (tile_size_q * i) + (tx * d);
        // Load Qi to SRAM registers
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[row_offset + x];
            // printf("Qi: %f, tx: %d, (tx * d) + x: %d\n", Qi[(tx * d) + x], tx, (tx * d) + x);
        }
        float row_m_prev = -INFINITY;
        float row_l = 0;

        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int j = 0; j < Tc; j++)  {
            // Load Kj, Vj to SRAM
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size_kv * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size_kv * j) + (tx * d) + x];
            }
            // printf("i: %d, j: %d, tx: %d \n", i, j, tx);
            __syncthreads();
            // S = QK^T, row_m = rowmax(S)
            float row_m_new = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Br * tx) + y] = sum;
//                if (tx==0 || tx==1) {
//                    printf("S[(Br * tx) + y]: %f, tx: %d, y: %d, (Br * tx) + y: %d \n", S[(Br * tx) + y], tx, y, (Br * tx) + y);
//                }

                if (sum > row_m_new)
                    row_m_new = sum;
            }
            // Get the latest max value
            if (row_m_new < row_m_prev) {
                row_m_new = row_m_prev;
            }
//            if (tx==0 || tx==1) {
//                printf("row_m_new: %f, tx: %d. \n", row_m_new, tx);
//            }
            // P = exp(S - row_mew), row_l = rowsum(P)
            float row_l_pij = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
                row_l_pij += S[(Bc * tx) + y];
            }

            // Compute new m and l
            row_l = (__expf(row_m_prev - row_m_new) * row_l) + row_l_pij + EPSILON;
//            if (tx==0 || tx==1) {
//                printf("row_l: %f, tx: %d. \n", row_l, tx);
//            }
            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
//                if (tx==0 || tx==1) {
//                    printf("pv: %f, tx: %d. \n", pv, tx);
//                }
                O[row_offset + x] =
                        ((__expf(row_m_prev - row_m_new) *
                        O[row_offset + x]) +
                        pv);
            }
            row_m_prev = row_m_new;
        }

        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
        for (int x = 0; x < d; x++) {
            O[row_offset + x] /= row_l;
        }
    }
}


torch::Tensor flash_attention_lunch(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);
    printf("B: %d, nh: %d, N: %d, d: %d, Tc: %d, Tr: %d \n", B, nh, N, d, Tc, Tr);
    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);


    // Calculate SRAM size needed per block
    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block
    printf("forward_kernel_v2\n ");
//    auto l = torch::zeros({B, nh, N});
//    auto m = torch::full({B, nh, N}, -INFINITY);
//    torch::Device device(torch::kCUDA);
//    l = l.to(device); m = m.to(device);
    flash_attention_v2_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, O.data_ptr<float>()
    );
//    const int sram_size = (Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
//    int max_sram_size;
//    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
//    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
//
//    auto l = torch::zeros({B, nh, N});
//    auto m = torch::full({B, nh, N}, -INFINITY);
//    torch::Device device(torch::kCUDA);
//    l = l.to(device); m = m.to(device);
//    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
//            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
//            N, d, Tc, Tr, Bc, Br, softmax_scale,
//            l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
//    );
    return O;
}