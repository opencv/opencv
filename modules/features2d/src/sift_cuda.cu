#include <stdint.h>
#include <unistd.h>

// NITHILA and RACHEL TO WORK ON
__global__ void findAndRefineExtremaKernel(
    const float* dog_prev,   // layer i-1
    const float* dog_curr,   // layer i
    const float* dog_next,   // layer i+1
    int rows, int cols, int step,
    int threshold,
    int* candidate_r,        // output: row positions
    int* candidate_c,        // output: col positions
    int* candidate_count     // output: atomic counter
) {
    // This kernel is responsible for finding the extrema and appending to a list that is shared among threads
    // It'll create a local list then move on the atmoically adding, or some sort of reduce function / gather function
    std::cout<<"Hello\n"<<std::endl;
}