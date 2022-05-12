#include <iostream>

//checking for errors with memory: cuda-memcheck ./lab1
//profiling of tasks and their time: nvprof ./lab1 profiling 

//global kernel
//this function - kernel, will run the same way on all the threads
__global__ void add(float *d_vec1, float *d_vec2, int N)
{
    //all threads in the block have indexes - here we are taking these indices
    //all the threads will index from (n*i), where i is the number of the block, n is the length of all blocks
    
    //block id for accessing the right memory part
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N)
    {
        //h is used for device variables (d_name)
        d_vec1[tid] = d_vec1[tid] + d_vec2[tid];
    }
}

int main(int argc , char **argsv)
{
    //a big table for that GPU would be useful
    //two tables of 1048576 (2 ^20) elements
    const int N = 1 << 20;
    float *h_vec1 = new float[N], *h_vec2 = new float[N];

    //d is used for device variables (d_name)
    float *d_vec1, *d_vec2;

    //we're allocating memory on the card
    cudaMalloc(&d_vec1, N * sizeof(float));
    cudaMalloc(&d_vec2, N * sizeof(float));

    //we're filling our two tables
    for (int i=0; i < N; i++)
    {
        h_vec1[i] = 1.0f;
        h_vec2[i] = 2.0f;
    }

    
    cudaMemcpy(d_vec1, h_vec1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, N * sizeof(float), cudaMemcpyHostToDevice);

    //the same but on CPU - iterativly
    for(int i = 0; i< N; i++)
    {
        //h_vec1[i] += h_vec2[i];
    }

    dim3 num_threads(1024);
    dim3 num_blocks(N / 1024);

    //adding on GPU by the kernel
    //max to 2048 or 1024 threads in one block
    //we copy the data to the card as the card has no access to the RAM
    add<<<num_blocks, num_threads>>>(d_vec1, d_vec2, N);

    //we're synchronizing all the threads
    //to this point all the counting must have ended - we will be taking the data from the GPU
    cudaDeviceSynchronize();

    //taking data from the card to our device
    //cudaMemcpy has cudaDeviceSynchronize inside it (it will wait on its own) - but I kept it for you to remember that it is not a given. Always check if your next function after what you're doing on GPU will synchronize.
    cudaMemcpy(h_vec1, d_vec1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec2, d_vec2, N * sizeof(float), cudaMemcpyDeviceToHost);

    //checking results
    for(int i=0; i < N; i++)
    {
        if(h_vec1[i] != 3.0f)
        {
            std::cout << "Incorrect result" << std::endl;
            return 1;
        }
        else
        {
            std::cout << "Correct result" << std::endl;
            return 1; 
        }
    }

    //cleaning the memory on host
    delete[] h_vec1;
    delete[] h_vec2;

    //cleaning the memory on device
    cudaFree(d_vec1);
    cudaFree(d_vec2);
} 