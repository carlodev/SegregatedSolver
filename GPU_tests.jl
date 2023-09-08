using CUDA,CUDA.CUSPARSE, CUDA.CUSOLVER,SparseArrays
using Krylov
using IterativeSolvers
using IncompleteLU

N = 15
CUDA.versioninfo()

A_cpu = sprand(N, N, 0.8)
x_cpu = ones(N)
b_cpu = A_cpu*x_cpu

# GPU Arrays
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuVector(b_cpu)
L,U = CUDA.CUSOLVER.lu(A_gpu)


# Solve a square and dense system on an Nivida GPU

t_gpu = @elapsed Krylov.gmres(A_gpu,b_gpu)

t_cpu = @elapsed IterativeSolvers.gmres(A_cpu, b_cpu)

t_gpu/t_cpu




