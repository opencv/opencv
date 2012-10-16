#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#define MAX_FLOAT 1e7f

int bit1Count(float x)
{
    int c = 0;
    int ix = (int)x;
    for (int i = 0 ; i < 32 ; i++)
    {
        c += ix & 0x1;
        ix >>= 1;
    }
    return (float)c;
}
/* 2dim launch, global size: dim0 is (query rows + block_size - 1) / block_size * block_size, dim1 is block_size
local size: dim0 is block_size, dim1 is block_size.
*/
__kernel void BruteForceMatch_UnrollMatch(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global int *bestTrainIdx,
    __global float *bestDistance,
    __local float *sharebuffer,
    int block_size,
    int max_desc_len,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);

    __local float *s_query = sharebuffer;
    __local float *s_train = sharebuffer + block_size * max_desc_len;

    int queryIdx = groupidx * block_size + lidy;
    // load the query into local memory.
    for (int i = 0 ;  i <  max_desc_len / block_size; i ++)
    {
        int loadx = lidx + i * block_size;
        s_query[lidy * max_desc_len + loadx] = loadx < query_cols ? query[min(queryIdx, query_rows - 1)  * (step / sizeof(float)) + loadx] : 0;
    }

    float myBestDistance = MAX_FLOAT;
    int myBestTrainIdx = -1;

    // loopUnrolledCached to find the best trainIdx and best distance.
    volatile int imgIdx = 0;
    for (int t = 0 ; t < (train_rows + block_size - 1) / block_size ; t++)
    {
        float result = 0;
        for (int i = 0 ; i < max_desc_len / block_size ; i++)
        {
            //load a block_size * block_size block into local train.
            const int loadx = lidx + i * block_size;
            s_train[lidx * block_size + lidy] = loadx < train_cols ? train[min(t * block_size + lidy, train_rows - 1) * (step / sizeof(float)) + loadx] : 0;

            //synchronize to make sure each elem for reduceIteration in share memory is written already.
            barrier(CLK_LOCAL_MEM_FENCE);

            /* there are threee types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
            sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

            switch(distType)
            {
            case 0:
                for (int j = 0 ; j < block_size ; j++)
                {
                    result += fabs(s_query[lidy * max_desc_len + i * block_size + j] -  s_train[j * block_size + lidx]);
                }
                break;
            case 1:
                for (int j = 0 ; j < block_size ; j++)
                {
                    float qr = s_query[lidy * max_desc_len + i * block_size + j] -  s_train[j * block_size + lidx];
                    result += qr * qr;
                }
                break;
            case 2:
                for (int j = 0 ; j < block_size ; j++)
                {
                    //result += popcount((uint)s_query[lidy * max_desc_len + i * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                    result += bit1Count((uint)s_query[lidy * max_desc_len + i * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                }
                break;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        int trainIdx = t * block_size + lidx;

        if (queryIdx < query_rows && trainIdx < train_rows && result < myBestDistance/* && mask(queryIdx, trainIdx)*/)
        {
            //bestImgIdx = imgIdx;
            myBestDistance = result;
            myBestTrainIdx = trainIdx;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    __local float *s_distance = (__local float*)(sharebuffer);
    __local int* s_trainIdx = (__local int *)(sharebuffer + block_size * block_size);

    //find BestMatch
    s_distance += lidy * block_size;
    s_trainIdx += lidy * block_size;
    s_distance[lidx] = myBestDistance;
    s_trainIdx[lidx] = myBestTrainIdx;

    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce -- now all reduce implement in each threads.
    for (int k = 0 ; k < block_size; k++)
    {
        if (myBestDistance > s_distance[k])
        {
            myBestDistance = s_distance[k];
            myBestTrainIdx = s_trainIdx[k];
        }
    }

    if (queryIdx < query_rows && lidx == 0)
    {
        bestTrainIdx[queryIdx] = myBestTrainIdx;
        bestDistance[queryIdx] = myBestDistance;
    }
}

__kernel void BruteForceMatch_Match(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global int *bestTrainIdx,
    __global float *bestDistance,
    __local float *sharebuffer,
    int block_size,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);

    const int queryIdx = groupidx * block_size + lidy;

    float myBestDistance = MAX_FLOAT;
    int myBestTrainIdx = -1;

    __local float *s_query = sharebuffer;
    __local float *s_train = sharebuffer + block_size * block_size;

    // loop
    for (int t = 0 ;  t < (train_rows + block_size - 1) / block_size ; t++)
    {
        //Dist dist;
        float result = 0;
        for (int i = 0 ; i < (query_cols + block_size - 1) / block_size ; i++)
        {
            const int loadx = lidx + i * block_size;
            //load query and train into local memory
            s_query[lidy * block_size + lidx] = 0;
            s_train[lidx * block_size + lidy] = 0;

            if (loadx < query_cols)
            {
                s_query[lidy * block_size + lidx] = query[min(queryIdx, query_rows - 1) * (step / sizeof(float)) + loadx];
                s_train[lidx * block_size + lidy] = train[min(t * block_size + lidy, train_rows - 1) * (step / sizeof(float)) + loadx];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            /* there are threee types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
            sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

            switch(distType)
            {
            case 0:
                for (int j = 0 ; j < block_size ; j++)
                {
                    result += fabs(s_query[lidy * block_size + j] -  s_train[j * block_size + lidx]);
                }
                break;
            case 1:
                for (int j = 0 ; j < block_size ; j++)
                {
                    float qr = s_query[lidy * block_size + j] -  s_train[j * block_size + lidx];
                    result += qr * qr;
                }
                break;
            case 2:
                for (int j = 0 ; j < block_size ; j++)
                {
                    //result += popcount((uint)s_query[lidy * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                    result += bit1Count((uint)s_query[lidy * block_size + j] ^ (uint)s_train[(uint)j * block_size + lidx]);
                }
                break;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const int trainIdx = t * block_size + lidx;

        if (queryIdx < query_rows && trainIdx < train_rows && result < myBestDistance /*&& mask(queryIdx, trainIdx)*/)
        {
            //myBestImgidx = imgIdx;
            myBestDistance = result;
            myBestTrainIdx = trainIdx;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float *s_distance = (__local float *)sharebuffer;
    __local int *s_trainIdx = (__local int *)(sharebuffer + block_size * block_size);

    //findBestMatch
    s_distance += lidy * block_size;
    s_trainIdx += lidy * block_size;
    s_distance[lidx] = myBestDistance;
    s_trainIdx[lidx] = myBestTrainIdx;

    barrier(CLK_LOCAL_MEM_FENCE);

    //reduce -- now all reduce implement in each threads.
    for (int k = 0 ; k < block_size; k++)
    {
        if (myBestDistance > s_distance[k])
        {
            myBestDistance = s_distance[k];
            myBestTrainIdx = s_trainIdx[k];
        }
    }

    if (queryIdx < query_rows && lidx == 0)
    {
        bestTrainIdx[queryIdx] = myBestTrainIdx;
        bestDistance[queryIdx] = myBestDistance;
    }
}

//radius_unrollmatch
__kernel void BruteForceMatch_RadiusUnrollMatch(
    __global float *query,
    __global float *train,
    float maxDistance,
    __global float *mask,
    __global int *bestTrainIdx,
    __global float *bestDistance,
    __global int *nMatches,
    __local float *sharebuffer,
    int block_size,
    int max_desc_len,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int bestTrainIdx_cols,
    int step,
    int ostep,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);
    const int groupidy = get_group_id(1);

    const int queryIdx = groupidy * block_size + lidy;
    const int trainIdx = groupidx * block_size + lidx;

    __local float *s_query = sharebuffer;
    __local float *s_train = sharebuffer + block_size * block_size;

    float result = 0;
    for (int i = 0 ; i < max_desc_len / block_size ; ++i)
    {
        //load a block_size * block_size block into local train.
        const int loadx = lidx + i * block_size;

        s_query[lidy * block_size + lidx] = loadx < query_cols ? query[min(queryIdx, query_rows - 1)  * (step / sizeof(float)) + loadx] : 0;
        s_train[lidx * block_size + lidy] = loadx < query_cols ? train[min(groupidx * block_size + lidy, train_rows - 1)  * (step / sizeof(float)) + loadx] : 0;

        //synchronize to make sure each elem for reduceIteration in share memory is written already.
        barrier(CLK_LOCAL_MEM_FENCE);

        /* there are three types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
        sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

        switch(distType)
        {
        case 0:
            for (int j = 0 ; j < block_size ; ++j)
            {
                result += fabs(s_query[lidy * block_size + j] - s_train[j * block_size + lidx]);
            }
            break;
        case 1:
            for (int j = 0 ; j < block_size ; ++j)
            {
                float qr = s_query[lidy * block_size + j] - s_train[j * block_size + lidx];
                result += qr * qr;
            }
            break;
        case 2:
            for (int j = 0 ; j < block_size ; ++j)
            {
                result += bit1Count((uint)s_query[lidy * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
            }
            break;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (queryIdx < query_rows && trainIdx < train_rows && result < maxDistance/* && mask(queryIdx, trainIdx)*/)
    {
        unsigned int ind = atom_inc(nMatches + queryIdx/*, (unsigned int) -1*/);

        if(ind < bestTrainIdx_cols)
        {
            //bestImgIdx = imgIdx;
            bestTrainIdx[queryIdx * (ostep / sizeof(int)) + ind] = trainIdx;
            bestDistance[queryIdx * (ostep / sizeof(float)) + ind] = result;
        }
    }
}

//radius_match
__kernel void BruteForceMatch_RadiusMatch(
    __global float *query,
    __global float *train,
    float maxDistance,
    __global float *mask,
    __global int *bestTrainIdx,
    __global float *bestDistance,
    __global int *nMatches,
    __local float *sharebuffer,
    int block_size,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int bestTrainIdx_cols,
    int step,
    int ostep,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);
    const int groupidy = get_group_id(1);

    const int queryIdx = groupidy * block_size + lidy;
    const int trainIdx = groupidx * block_size + lidx;

    __local float *s_query = sharebuffer;
    __local float *s_train = sharebuffer + block_size * block_size;

    float result = 0;
    for (int i = 0 ; i < (query_cols + block_size - 1) / block_size ; ++i)
    {
        //load a block_size * block_size block into local train.
        const int loadx = lidx + i * block_size;

        s_query[lidy * block_size + lidx] = loadx < query_cols ? query[min(queryIdx, query_rows - 1)  * (step / sizeof(float)) + loadx] : 0;
        s_train[lidx * block_size + lidy] = loadx < query_cols ? train[min(groupidx * block_size + lidy, train_rows - 1)  * (step / sizeof(float)) + loadx] : 0;

        //synchronize to make sure each elem for reduceIteration in share memory is written already.
        barrier(CLK_LOCAL_MEM_FENCE);

        /* there are three types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
        sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

        switch(distType)
        {
        case 0:
            for (int j = 0 ; j < block_size ; ++j)
            {
                result += fabs(s_query[lidy * block_size + j] - s_train[j * block_size + lidx]);
            }
            break;
        case 1:
            for (int j = 0 ; j < block_size ; ++j)
            {
                float qr = s_query[lidy * block_size + j] - s_train[j * block_size + lidx];
                result += qr * qr;
            }
            break;
        case 2:
            for (int j = 0 ; j < block_size ; ++j)
            {
                result += bit1Count((uint)s_query[lidy * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
            }
            break;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (queryIdx < query_rows && trainIdx < train_rows && result < maxDistance/* && mask(queryIdx, trainIdx)*/)
    {
        unsigned int ind = atom_inc(nMatches + queryIdx/*, (unsigned int) -1*/);

        if(ind < bestTrainIdx_cols)
        {
            //bestImgIdx = imgIdx;
            bestTrainIdx[queryIdx * (ostep / sizeof(int)) + ind] = trainIdx;
            bestDistance[queryIdx * (ostep / sizeof(float)) + ind] = result;
        }
    }
}


__kernel void BruteForceMatch_knnUnrollMatch(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global int2 *bestTrainIdx,
    __global float2 *bestDistance,
    __local float *sharebuffer,
    int block_size,
    int max_desc_len,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);

    const int queryIdx = groupidx * block_size + lidy;
    local float *s_query = sharebuffer;
    local float *s_train = sharebuffer + block_size * max_desc_len;

    // load the query into local memory.
    for (int i = 0 ;  i <  max_desc_len / block_size; i ++)
    {
        int loadx = lidx + i * block_size;
        s_query[lidy * max_desc_len + loadx] = loadx < query_cols ? query[min(queryIdx, query_rows - 1)  * (step / sizeof(float)) + loadx] : 0;
    }

    float myBestDistance1 = MAX_FLOAT;
    float myBestDistance2 = MAX_FLOAT;
    int myBestTrainIdx1 = -1;
    int myBestTrainIdx2 = -1;

    //loopUnrolledCached
    volatile int imgIdx = 0;
    for (int t = 0 ; t < (train_rows + block_size - 1) / block_size ; t++)
    {
        float result = 0;
        for (int i = 0 ; i < max_desc_len / block_size ; i++)
        {
            const int loadX = lidx + i * block_size;
            //load a block_size * block_size block into local train.
            const int loadx = lidx + i * block_size;
            s_train[lidx * block_size + lidy] = loadx < train_cols ? train[min(t * block_size + lidy, train_rows - 1) * (step / sizeof(float)) + loadx] : 0;

            //synchronize to make sure each elem for reduceIteration in share memory is written already.
            barrier(CLK_LOCAL_MEM_FENCE);

            /* there are threee types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
            sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

            switch(distType)
            {
            case 0:
                for (int j = 0 ; j < block_size ; j++)
                {
                    result += fabs(s_query[lidy * max_desc_len + i * block_size + j] -  s_train[j * block_size + lidx]);
                }
                break;
            case 1:
                for (int j = 0 ; j < block_size ; j++)
                {
                    float qr = s_query[lidy * max_desc_len + i * block_size + j] -  s_train[j * block_size + lidx];
                    result += qr * qr;
                }
                break;
            case 2:
                for (int j = 0 ; j < block_size ; j++)
                {
                    //result += popcount((uint)s_query[lidy * max_desc_len + i * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                    result += bit1Count((uint)s_query[lidy * max_desc_len + i * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                }
                break;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const int trainIdx = t * block_size + lidx;

        if (queryIdx < query_rows && trainIdx < train_rows)
        {
            if (result < myBestDistance1)
            {
                myBestDistance2 = myBestDistance1;
                myBestTrainIdx2 = myBestTrainIdx1;
                myBestDistance1 = result;
                myBestTrainIdx1 = trainIdx;
            }
            else if (result < myBestDistance2)
            {
                myBestDistance2 = result;
                myBestTrainIdx2 = trainIdx;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    local float *s_distance = (local float *)sharebuffer;
    local int *s_trainIdx = (local int *)(sharebuffer + block_size * block_size);

    // find BestMatch
    s_distance += lidy * block_size;
    s_trainIdx += lidy * block_size;

    s_distance[lidx] = myBestDistance1;
    s_trainIdx[lidx] = myBestTrainIdx1;

    float bestDistance1 = MAX_FLOAT;
    float bestDistance2 = MAX_FLOAT;
    int bestTrainIdx1 = -1;
    int bestTrainIdx2 = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx == 0)
    {
        for (int i = 0 ; i < block_size ; i++)
        {
            float val = s_distance[i];
            if (val < bestDistance1)
            {
                bestDistance2 = bestDistance1;
                bestTrainIdx2 = bestTrainIdx1;

                bestDistance1 = val;
                bestTrainIdx1 = s_trainIdx[i];
            }
            else if (val < bestDistance2)
            {
                bestDistance2 = val;
                bestTrainIdx2 = s_trainIdx[i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    s_distance[lidx] = myBestDistance2;
    s_trainIdx[lidx] = myBestTrainIdx2;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx == 0)
    {
        for (int i = 0 ; i < block_size ; i++)
        {
            float val = s_distance[i];

            if (val < bestDistance2)
            {
                bestDistance2 = val;
                bestTrainIdx2 = s_trainIdx[i];
            }
        }
    }

    myBestDistance1 = bestDistance1;
    myBestDistance2 = bestDistance2;

    myBestTrainIdx1 = bestTrainIdx1;
    myBestTrainIdx2 = bestTrainIdx2;

    if (queryIdx < query_rows && lidx == 0)
    {
        bestTrainIdx[queryIdx] = (int2)(myBestTrainIdx1, myBestTrainIdx2);
        bestDistance[queryIdx] = (float2)(myBestDistance1, myBestDistance2);
    }
}

__kernel void BruteForceMatch_knnMatch(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global int2 *bestTrainIdx,
    __global float2 *bestDistance,
    __local float *sharebuffer,
    int block_size,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType
    )
{
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);
    const int groupidx = get_group_id(0);

    const int queryIdx = groupidx * block_size + lidy;
    local float *s_query = sharebuffer;
    local float *s_train = sharebuffer + block_size * block_size;

    float myBestDistance1 = MAX_FLOAT;
    float myBestDistance2 = MAX_FLOAT;
    int myBestTrainIdx1 = -1;
    int myBestTrainIdx2 = -1;

    //loop
    for (int  t = 0 ; t < (train_rows + block_size - 1) / block_size ; t++)
    {
        float result = 0.0f;
        for (int i = 0 ; i < (query_cols + block_size -1) / block_size ; i++)
        {
            const int loadx = lidx + i * block_size;
            //load query and train into local memory
            s_query[lidy * block_size + lidx] = 0;
            s_train[lidx * block_size + lidy] = 0;

            if (loadx < query_cols)
            {
                s_query[lidy * block_size + lidx] = query[min(queryIdx, query_rows - 1) * (step / sizeof(float)) + loadx];
                s_train[lidx * block_size + lidy] = train[min(t * block_size + lidy, train_rows - 1) * (step / sizeof(float)) + loadx];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            /* there are threee types in the reducer. the first is L1Dist, which to sum the abs(v1, v2), the second is L2Dist, which to
            sum the (v1 - v2) * (v1 - v2), the third is humming, which to popc(v1 ^ v2), popc is to count the bits are set to 1*/

            switch(distType)
            {
            case 0:
                for (int j = 0 ; j < block_size ; j++)
                {
                    result += fabs(s_query[lidy * block_size + j] -  s_train[j * block_size + lidx]);
                }
                break;
            case 1:
                for (int j = 0 ; j < block_size ; j++)
                {
                    float qr = s_query[lidy * block_size + j] -  s_train[j * block_size + lidx];
                    result += qr * qr;
                }
                break;
            case 2:
                for (int j = 0 ; j < block_size ; j++)
                {
                    //result += popcount((uint)s_query[lidy * block_size + j] ^ (uint)s_train[j * block_size + lidx]);
                    result += bit1Count((uint)s_query[lidy * block_size + j] ^ (uint)s_train[(uint)j * block_size + lidx]);
                }
                break;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const int trainIdx = t * block_size + lidx;

        if (queryIdx < query_rows && trainIdx < train_rows /*&& mask(queryIdx, trainIdx)*/)
        {
            if (result < myBestDistance1)
            {
                myBestDistance2 = myBestDistance1;
                myBestTrainIdx2 = myBestTrainIdx1;
                myBestDistance1 = result;
                myBestTrainIdx1 = trainIdx;
            }
            else if (result < myBestDistance2)
            {
                myBestDistance2 = result;
                myBestTrainIdx2 = trainIdx;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float *s_distance = (__local float *)sharebuffer;
    __local int *s_trainIdx = (__local int *)(sharebuffer + block_size * block_size);

    //findBestMatch
    s_distance += lidy * block_size;
    s_trainIdx += lidy * block_size;

    s_distance[lidx] = myBestDistance1;
    s_trainIdx[lidx] = myBestTrainIdx1;

    float bestDistance1 = MAX_FLOAT;
    float bestDistance2 = MAX_FLOAT;
    int bestTrainIdx1 = -1;
    int bestTrainIdx2 = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx == 0)
    {
        for (int i = 0 ; i < block_size ; i++)
        {
            float val = s_distance[i];
            if (val < bestDistance1)
            {
                bestDistance2 = bestDistance1;
                bestTrainIdx2 = bestTrainIdx1;

                bestDistance1 = val;
                bestTrainIdx1 = s_trainIdx[i];
            }
            else if (val < bestDistance2)
            {
                bestDistance2 = val;
                bestTrainIdx2 = s_trainIdx[i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    s_distance[lidx] = myBestDistance2;
    s_trainIdx[lidx] = myBestTrainIdx2;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lidx == 0)
    {
        for (int i = 0 ; i < block_size ; i++)
        {
            float val = s_distance[i];

            if (val < bestDistance2)
            {
                bestDistance2 = val;
                bestTrainIdx2 = s_trainIdx[i];
            }
        }
    }

    myBestDistance1 = bestDistance1;
    myBestDistance2 = bestDistance2;

    myBestTrainIdx1 = bestTrainIdx1;
    myBestTrainIdx2 = bestTrainIdx2;

    if (queryIdx < query_rows && lidx == 0)
    {
        bestTrainIdx[queryIdx] = (int2)(myBestTrainIdx1, myBestTrainIdx2);
        bestDistance[queryIdx] = (float2)(myBestDistance1, myBestDistance2);
    }
}

kernel void BruteForceMatch_calcDistanceUnrolled(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global float *allDist,
    __local float *sharebuffer,
    int block_size,
    int max_desc_len,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType)
{
    /* Todo */
}

kernel void BruteForceMatch_calcDistance(
    __global float *query,
    __global float *train,
    __global float *mask,
    __global float *allDist,
    __local float *sharebuffer,
    int block_size,
    int query_rows,
    int query_cols,
    int train_rows,
    int train_cols,
    int step,
    int distType)
{
    /* Todo */
}

kernel void BruteForceMatch_findBestMatch(
    __global float *allDist,
    __global int *bestTrainIdx,
    __global float *bestDistance,
     int k,
     int block_size
    )
{
    /* Todo */
}