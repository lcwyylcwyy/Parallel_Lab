nvcc --ptxas-options=-v -o transpose2 transpose2.cu

nvcc -O3 -o transpose2 transpose2.cu
nvcc -O3 -o transpose transpose.cu