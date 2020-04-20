#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../headers/preprocess.h"

void init_data(float* pts, size_t number_of_points, size_t dimensions);
void save_data(float* pts, size_t number_of_points, size_t dimensions, const char* name);
void save_int_data(int* pts, size_t number_of_points, const char* name);
__global__ void gpu_grid_knn(float* points, float* queries, 
                int* intgr_points_per_block, int* intgr_queries_per_block, 
                int* points_per_block, int* queries_per_block, 
                float* distsances, int* neighbours,
                int num_of_points, int num_of_queries, int dimensions, int grid_d,
                int offx, int offy, int offz);



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line);

int main(int argc, char** argv){
    if(argc < 4){
        printf("Usage \n");
        return 0;
    }
    int input_num_points = atoi(argv[1]);
    int input_grid_d = atoi(argv[2]);
    int inpt_seed = atoi(argv[3]);
    /*!TODO 
        Input args
            - num points
            - num queries
            - grid_d
            - seed
    */
    gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    srand(inpt_seed);
    size_t number_of_points = 1<<input_num_points;
    size_t grid_d = 1<<input_grid_d;
    size_t dimensions = 3;
    // Init Points Array and relevant arays.
    float* ref_points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    float* ordered_ref_points = (float*)malloc(number_of_points*dimensions*sizeof(float));
    int* block_of_point = (int*)malloc(number_of_points*sizeof(int));
    int* points_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* intg_points_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* perm_points = (int*)malloc(number_of_points*sizeof(int));

    size_t number_of_queries = number_of_points;
    // Init Queries Array and relevant arays.
    float* queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    float* ordered_queries = (float*)malloc(number_of_queries*dimensions*sizeof(float));
    int* block_of_query = (int*)malloc(number_of_queries*sizeof(int));
    int* queries_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* intg_queries_per_block = (int*)malloc(pow(grid_d,dimensions)*sizeof(int));
    int* perm_queries = (int*)malloc(number_of_queries*sizeof(int));

    // --------------------- Preprocess data ---------------------
    init_data(ref_points, number_of_points, dimensions);
    init_data(queries, number_of_queries, dimensions);
    save_data(ref_points, number_of_points, dimensions, "pts.csv");
    save_data(queries, number_of_queries, dimensions, "qrs.csv");

    AssignPointsToBlocks(ref_points, number_of_points, grid_d, dimensions, block_of_point);
    CountPointsPerBlock(block_of_point, number_of_points, grid_d, dimensions, points_per_block, intg_points_per_block);
    ReorderPointsByBlock(ref_points, block_of_point, number_of_points, dimensions, ordered_ref_points, perm_points);

    AssignPointsToBlocks(queries, number_of_queries, grid_d, dimensions, block_of_query);
    CountPointsPerBlock(block_of_query, number_of_queries, grid_d, dimensions, queries_per_block, intg_queries_per_block);
    ReorderPointsByBlock(queries, block_of_query, number_of_queries, dimensions, ordered_queries, perm_queries);


    save_data(ordered_ref_points, number_of_points, dimensions, "ord_pt.csv");
    save_data(ordered_queries, number_of_queries, dimensions, "ord_qr.csv");
    save_int_data(queries_per_block, number_of_points, "bop.csv");

    // --------------------- Kernel ---------------------
	cudaProfilerStart();
    float* distances = (float*)malloc(number_of_queries*sizeof(float));
    int* neighbours = (int*)malloc(number_of_queries*sizeof(int));
    for(int i = 0; i < number_of_queries; i++){
        distances[i] = 100.0;
        neighbours[i] = -2;
    }
    

    float* dev_points; 
    float* dev_queries; 
    int *dev_intg_points_per_block, *dev_points_per_block;
    int *dev_intg_queries_per_block, *dev_queries_per_block;
    float *dev_distances;
    int* dev_neighbours;

    gpuErrchk(cudaMalloc((void**)&dev_points, number_of_points*dimensions*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_queries, number_of_queries*dimensions*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_intg_points_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_intg_queries_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_points_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_queries_per_block, pow(grid_d,dimensions)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_distances, number_of_queries*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dev_neighbours, number_of_queries*sizeof(int)));

    gpuErrchk(cudaMemcpy(dev_points, ordered_ref_points, number_of_points*dimensions*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_queries, ordered_queries, number_of_queries*dimensions*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_intg_points_per_block, intg_points_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_intg_queries_per_block, intg_queries_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_points_per_block, points_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_queries_per_block, queries_per_block, pow(grid_d,dimensions)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_distances, distances ,number_of_queries*sizeof(float), cudaMemcpyHostToDevice));   
    gpuErrchk(cudaMemcpy(dev_neighbours, neighbours ,number_of_queries*sizeof(int), cudaMemcpyHostToDevice));



    int thread_groups = 2;
    dim3 blocks(grid_d, grid_d, grid_d);
    dim3 threads(32,thread_groups,1);
    uint64_t shmem = (2*thread_groups*32*dimensions)*sizeof(float);
    gpu_grid_knn<<<blocks, threads, shmem>>>(dev_points, dev_queries,
                dev_intg_points_per_block, dev_intg_queries_per_block,
                dev_points_per_block, dev_queries_per_block,
                dev_distances, dev_neighbours,
                number_of_points, number_of_queries, dimensions, grid_d, 0,0,0);

    for(int x = -1; x < 2; x++){
        for(int y = -1; y < 2; y++){
            for(int z = -1; z < 2; z++){
                if(x == 0 && y == 0 && z == 0) continue;
                  gpu_grid_knn<<<blocks, threads, shmem>>>(dev_points, dev_queries,
                    dev_intg_points_per_block, dev_intg_queries_per_block,
                    dev_points_per_block, dev_queries_per_block,
                    dev_distances, dev_neighbours,
                    number_of_points, number_of_queries, dimensions, grid_d, x,y,z);
            }
        }
    }
    gpuErrchk(cudaMemcpy(distances, dev_distances, number_of_points*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(neighbours, dev_neighbours, number_of_points*sizeof(int), cudaMemcpyDeviceToHost));
    cudaProfilerStop();
    save_data(distances, number_of_queries, 1, "dists.csv");
    save_int_data(neighbours, number_of_queries, "nb.csv");
    // -----------------------------------------------------------
    free(ref_points);
    free(ordered_ref_points);
    free(points_per_block);
    free(intg_points_per_block);
    free(block_of_point);
    free(perm_points);


    free(queries);
    free(ordered_queries);
    free(queries_per_block);
    free(intg_queries_per_block);
    free(block_of_query);
    free(perm_queries);
    return 0;

}


void init_data(float* pts, size_t number_of_points, size_t dimensions){
    for(int i = 0; i < number_of_points; i++){
        for(int j = 0; j < dimensions; j++){
            pts[i + j*number_of_points] = (float)rand()/RAND_MAX;
        }
    }
}

void save_data(float* pts, size_t number_of_points, size_t dimensions, const char* name){
    FILE* f = fopen(name, "w+");
    fprintf(f, "%lu %lu\n", number_of_points, dimensions);
    for(int i = 0; i < number_of_points-1; i++){
        for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%f ", pts[i + j*number_of_points]);
        }
        fprintf(f, "%f\n",pts[i + (dimensions-1)*number_of_points]);
    }

    for(int j = 0; j < dimensions-1; j++){
            fprintf(f, "%f ", pts[(number_of_points-1) + j*number_of_points]);
        }
        fprintf(f, "%f\n",pts[(number_of_points-1) + (dimensions-1)*number_of_points]);
    fclose(f);
}

void save_int_data(int* pts, size_t number_of_points, const char* name){
    FILE* f = fopen(name, "w+");
    for(int i = 0; i < number_of_points-1; i++){
       
            fprintf(f, "%d\n", pts[i]);
        
    }

    fclose(f);
}

__global__ void gpu_grid_knn(float* points, float* queries, 
                int* intgr_points_per_block, int* intgr_queries_per_block, 
                int* points_per_block, int* queries_per_block, 
                float* distsances, int* neighbours,
                int num_of_points, int num_of_queries, int dimensions, int grid_d,
                int offx, int offy, int offz){

    extern __shared__ float shared_array[];
    // Check if the block is inbounds
    if( (int)blockIdx.x + offx  < 0 || offx + (int)blockIdx.x >= grid_d || 
        (int)blockIdx.y + offy  < 0 || offy + (int)blockIdx.y >= grid_d ||
        (int)blockIdx.z + offz  < 0 || offz + (int)blockIdx.z >= grid_d) return;

    // Block of queries
    int q_bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // Block of points to search
    int p_bid = blockIdx.x+offx + (blockIdx.y+offy)*gridDim.x + (blockIdx.z+offz)*gridDim.x*gridDim.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;

    // __shared__ float shared_array[(2*32*32)];
    float* sh_queries = shared_array;
    // float* query = &sh_queries[tid*dimensions];
    float* sh_points = &shared_array[stride*dimensions];
    int start_points = intgr_points_per_block[p_bid];
    int start_queries = intgr_queries_per_block[q_bid];
    int total_points = points_per_block[p_bid];
    int total_queries = queries_per_block[q_bid];
    float distance;
    int neighbour;

    // for(int i = 0; i < dimensions; i++){
    //     query[i] = 1;
    //     sh_points[tid*dimensions + i] = 1;
    // }
    for(int q = 0; q < total_queries; q += stride){
        int q_index = q + tid + start_queries;
        if(tid + q < total_queries){
            for(int d = 0; d < dimensions; d++){
                sh_queries[tid + d*stride] = queries[q_index + d*num_of_queries];
            }
            distance = distsances[q_index];
            neighbour = neighbours[q_index];
        }
        __syncthreads();
        for(int p = 0; p < total_points; p+= stride){
            int p_index = start_queries + p + tid;
            __syncthreads();
            if(p + tid < total_points){
                for(int d = 0; d < dimensions; d++){
                    sh_points[tid + d*stride] = points[p_index + d*num_of_points];
                }
            }
            __syncthreads();
            
            int bounds = stride < total_points-p ? stride : total_points-p;
            if(tid + q < total_queries){
                for(int i = 0; i < bounds; i++){
                    float tempdist = 0;
                    for(int d = 0; d < dimensions; d++){
                        float tempquery = sh_queries[tid + d*stride];
                        tempdist += powf(tempquery -  sh_points[(i+tid)%bounds + d*stride], 2);
                    }
                    tempdist = sqrtf(tempdist);

                    neighbour = tempdist < distance ? start_points+ p +(i+tid)%bounds : neighbour;
                    distance = tempdist < distance ? tempdist : distance;
                }
            }
        }
        if(tid + q < total_queries){
            distsances[q_index] = distance;
            neighbours[q_index] = neighbour;
        }
    }
}

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
