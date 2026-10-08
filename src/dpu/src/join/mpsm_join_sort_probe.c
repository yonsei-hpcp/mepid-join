/*
 * Join with multiple tasklets
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <mutex.h>
#include <string.h>

#include "argument.h"

#define NR_TASKLETS 12

#define MUTEX_SIZE 52

#define BLOCK_LOG 11
#define BLOCK_SIZE (1 << BLOCK_LOG)
#define ELEM_PER_BLOCK (BLOCK_SIZE >> 3)
#define PART_PER_BLOCK (BLOCK_SIZE >> 2)

#define BLOCK_SIZE2 (1 << 10)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host sort_merge_sort_probe_arg param_sort_merge_sort_probe;
__host sort_merge_probe_return_arg param_sort_merge_probe_return;
__host dpu_results_t dpu_results;

// Results
int SHARED_COUNT = 0;

// The number of instructions
__host uint32_t nb_instrs;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

// Total range of elem
uint32_t ELEM_RANGE = 0;

uint32_t loops = 0;
uint32_t remains = 0;
uint32_t histogram_bucket_num = 0;

// Each relation start address
uint32_t R_key_sorted_addr;
uint32_t S_key_sorted_addr;

// Each histogram start address
uint32_t R_histogram_addr;
uint32_t S_histogram_addr;

// Join result address
uint32_t result_addr;

// Information about relation R, S
uint32_t r_total_bytes;
uint32_t s_total_bytes;
uint32_t r_total_elem;
uint32_t s_total_elem;

// Information of index R
uint32_t r_info_elem;
uint32_t *r_info_index;

// Variables for Histogram
uint32_t *r_histogram_buff;
uint32_t *s_histogram_buff;

uint32_t *r_shared_buff;

void quickSort(tuplePair_t* arr, uint32_t* stack, uint32_t size)
{
    // Initialize of the variables
    int32_t top = -1;
    uint32_t start, end, pivot;
    uint32_t pivot_val;

    // Push initial values to stack
    stack[++top] = 0;
    stack[++top] = size;

    while (top >= 0)
    {
        // Pop
        end = stack[top--];
        start = stack[top--];

        if (end - start < 2) continue;

        // Between two points
        pivot = start + (end - start) / 2;

        /* Partition Starts */

        int32_t left = start;
        int32_t right = end - 2;
        tuplePair_t temp;

        // Value of the pivot
        pivot_val = arr[pivot].lvalue;

        // Swap
        temp = arr[end - 1];
        arr[end - 1] = arr[pivot];
        arr[pivot] = temp;

        while (left < right)
        {
            if (arr[left].lvalue < pivot_val) left++;
            else if (arr[right].lvalue >= pivot_val) right--;
            else
            {
                // Swap
                temp = arr[left];
                arr[left] = arr[right];
                arr[right] = temp;
            }
        }

        uint32_t idx = right;
        if (arr[right].lvalue < pivot_val) idx++;

        // Swap
        temp = arr[end - 1];
        arr[end - 1] = arr[idx];
        arr[idx] = temp;

        pivot = idx;

        /* Partition Ends */
        stack[++top] = pivot + 1;
        stack[++top] = end;
        
        stack[++top] = start;
        stack[++top] = pivot;
    }
}

void mergeSort(tuplePair_t* arr, tuplePair_t* arr_buff, uint32_t elem_num)
{
    // Merge sort without recursion
    for (uint32_t arr_size = 1; arr_size <= elem_num; arr_size <<= 1)
    {
        // Index for sorted array
        uint32_t left = 0;
        uint32_t right = left + arr_size;

        // Where to stop
        uint32_t end = right + arr_size;
        if (end > elem_num) end = elem_num;

        // Index for new sorted array
        uint32_t sorted_idx = 0;

        uint32_t iter_loop = elem_num / (arr_size << 1);
        if (elem_num % (arr_size << 1) > arr_size) iter_loop++;

        for (uint32_t iter = 0; iter < iter_loop; iter++)
        {
            // Index to compare
            uint32_t i = left;
            uint32_t j = right;

            // When write all the elements to buffer
            while (sorted_idx < end)
            {
                if (arr[i].lvalue < arr[j].lvalue) arr_buff[sorted_idx++] = arr[i++];
                else arr_buff[sorted_idx++] = arr[j++];
                // When one of the array ends
                if ((i >= right) || (j >= end)) break;
            }

            // Copy the remains
            if (i >= right)
            {
                while (j < end) arr_buff[sorted_idx++] = arr[j++];
            }
            else if (j >= end)
            {
                while (i < right) arr_buff[sorted_idx++] = arr[i++];
            }

            left += (arr_size << 1);
            right += (arr_size << 1);

            if (right >= elem_num) 
            {
                arr_buff[sorted_idx] = arr[left];
                break;
            }
            
            end = right + arr_size;
            if (end > elem_num) end = elem_num;
        }

        // Copy the buffer to the original
        for (uint32_t idx = 0; idx < elem_num; idx++) arr[idx] = arr_buff[idx];
    }
}

int probe(tuplePair_t* r_table, tuplePair_t* s_table, uint32_t r_elem, uint32_t s_elem)
{
    uint32_t r_idx = 0;
    for (uint32_t s_idx = 0; s_idx < s_elem; s_idx++)
    {
        // Matched
        if (s_table[s_idx].lvalue == r_table[r_idx].lvalue)
        {
            s_table[s_idx].lvalue = r_table[r_idx].rvalue;
        }
        else if (s_table[s_idx].lvalue > r_table[r_idx].lvalue)
        {
            r_idx++; s_idx--;
            if (r_idx >= r_elem) return 0;
        }
        else if (s_table[s_idx].lvalue < r_table[r_idx].lvalue)
        {
            return 0;
        }
    }

    return 1;
}

int main()
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        // Reset the heap
        mem_reset();

        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;

        perfcounter_config(COUNT_CYCLES, true);
    }

    barrier_wait(&my_barrier);

    //////////////////////////////////////////////
    // Variable Setup
    //////////////////////////////////////////////

    if (tasklet_id == 0)
    {
        R_key_sorted_addr = MRAM_BASE_ADDR + param_sort_merge_sort_probe.r_sorted_start_byte;
        S_key_sorted_addr = MRAM_BASE_ADDR + param_sort_merge_sort_probe.s_sorted_start_byte;
        R_histogram_addr = MRAM_BASE_ADDR + param_sort_merge_sort_probe.r_histogram_start_byte;
        S_histogram_addr = MRAM_BASE_ADDR + param_sort_merge_sort_probe.s_histogram_start_byte;
        result_addr = MRAM_BASE_ADDR + param_sort_merge_sort_probe.result_probe_start_byte;
    }
    else if (tasklet_id == 1)
    {
        r_total_bytes = param_sort_merge_sort_probe.r_total_bytes;
        s_total_bytes = param_sort_merge_sort_probe.s_total_bytes;
        r_total_elem = r_total_bytes / sizeof(tuplePair_t);
        s_total_elem = s_total_bytes / sizeof(tuplePair_t);
    }
    else if (tasklet_id == 2)
    {
        r_histogram_buff = (uint32_t *) mem_alloc(BLOCK_SIZE);
        s_histogram_buff = (uint32_t *) mem_alloc(BLOCK_SIZE);
        r_shared_buff = (uint32_t *) mem_alloc(BLOCK_SIZE2);
    }
    else if (tasklet_id == 3)
    {
        ELEM_RANGE = param_sort_merge_sort_probe.elem_range;
        histogram_bucket_num = ELEM_RANGE / param_sort_merge_sort_probe.hist_interval + 1;

        loops = histogram_bucket_num / PART_PER_BLOCK;
        remains = histogram_bucket_num % PART_PER_BLOCK;
    }

    int r_partition_elem = 0;
    int s_partition_elem = 0;
    int return_val = 0;

    // Relation buffers to merge
    tuplePair_t *r_tuples_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE2);
    tuplePair_t *s_tuples_buff = (tuplePair_t *) mem_alloc(BLOCK_SIZE2);

    // Relation buffers to sort
    void *r_sorted_buff = (void *) mem_alloc(BLOCK_SIZE2);
    void *s_sorted_buff = (void *) mem_alloc(BLOCK_SIZE2);

    barrier_wait(&my_barrier);

    // Load histogram of partition info from MRAM
    for (uint32_t iter = 0; iter < loops; iter++)
    {
        // Histogram for R, S
        if (tasklet_id == 3)
        {
            mram_read(
                (__mram_ptr void const *)(S_histogram_addr + iter * BLOCK_SIZE),
                s_histogram_buff,
                BLOCK_SIZE);
        }
        else if (tasklet_id == 4)
        {
            mram_read(
                (__mram_ptr void const *)(R_histogram_addr + iter * BLOCK_SIZE),
                r_histogram_buff,
                BLOCK_SIZE);
        }

        barrier_wait(&my_barrier);

        for (uint32_t part = tasklet_id; part < (PART_PER_BLOCK - 1); part += NR_TASKLETS)
        {
            s_partition_elem = s_histogram_buff[part + 1] - s_histogram_buff[part];
            r_partition_elem = r_histogram_buff[part + 1] - r_histogram_buff[part];

            if (s_partition_elem == 0) continue;
            else if (s_partition_elem <= ELEM_PER_BLOCK2)
            {
                mram_read(
                    (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t)),
                    s_tuples_buff,
                    (s_partition_elem << 3));

                mram_read(
                    (__mram_ptr void const *)(R_key_sorted_addr + r_histogram_buff[part] * sizeof(tuplePair_t)),
                    r_tuples_buff,
                    (r_partition_elem << 3));

                mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, s_partition_elem);
                mergeSort(r_tuples_buff, (tuplePair_t *) r_sorted_buff, r_partition_elem);

                return_val = probe((tuplePair_t *)r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, s_partition_elem);
                mutex_lock(&(mutex_atomic[0x1F & part]));
                SHARED_COUNT += s_partition_elem;
                mutex_unlock(&(mutex_atomic[0x1F & part]));

                if (return_val == 0)
                {
                    dpu_results.ERROR_TYPE_3 = 33;
                    return 0;
                }

                mram_write(
                    (tuplePair_t *) s_sorted_buff,
                    (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t)),
                    (s_partition_elem << 3));
            }
            else 
            {
                mram_read(
                    (__mram_ptr void const *)(R_key_sorted_addr + r_histogram_buff[part] * sizeof(tuplePair_t)),
                    r_tuples_buff,
                    (r_partition_elem << 3));

                mergeSort(r_tuples_buff, (tuplePair_t *) r_sorted_buff, r_partition_elem);

                int s_loop = s_partition_elem / ELEM_PER_BLOCK2;
                int s_remains = s_partition_elem % ELEM_PER_BLOCK2;

                for (int s_iter = 0; s_iter < (s_loop + 1); s_iter++)
                {
                    if (s_iter < s_loop)
                    {
                        mram_read(
                            (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_iter),
                            s_tuples_buff,
                            BLOCK_SIZE2);

                        mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, ELEM_PER_BLOCK2);
                        return_val = probe(r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, ELEM_PER_BLOCK2);
                        mutex_lock(&(mutex_atomic[0x1F & part]));
                        SHARED_COUNT += ELEM_PER_BLOCK2;
                        mutex_unlock(&(mutex_atomic[0x1F & part]));

                        if (return_val == 0)
                        {
                            dpu_results.ERROR_TYPE_3 = 34;
                            return 0;
                        }

                        mram_write(
                            (tuplePair_t *) s_sorted_buff,
                            (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_iter),
                            BLOCK_SIZE2);
                    }
                    else
                    {
                        mram_read(
                            (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_loop),
                            s_tuples_buff,
                            (s_remains << 3));
                        
                        mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, s_remains);
                        return_val = probe(r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, s_remains);
                        mutex_lock(&(mutex_atomic[0x1F & part]));
                        SHARED_COUNT += s_remains;
                        mutex_unlock(&(mutex_atomic[0x1F & part]));

                        if (return_val == 0)
                        {
                            dpu_results.ERROR_TYPE_3 = 35;
                            return 0;
                        }

                        mram_write(
                            (tuplePair_t *) s_sorted_buff,
                            (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_loop),
                            (s_remains << 3));
                    }
                }
            }
        }

        barrier_wait(&my_barrier);
    }

    if (remains > 0)
    {
        if (tasklet_id == 5)
        {
            mram_read(
                (__mram_ptr void const *)(S_histogram_addr + loops * BLOCK_SIZE),
                s_histogram_buff,
                (remains << 2));
        }
        else if (tasklet_id == 6)
        {
            mram_read(
                (__mram_ptr void const *)(R_histogram_addr + loops * BLOCK_SIZE),
                r_histogram_buff,
                (remains << 2));
        }

        barrier_wait(&my_barrier);

        for (uint32_t part = tasklet_id; part < (remains - 1); part += NR_TASKLETS)
        {
            s_partition_elem = s_histogram_buff[part + 1] - s_histogram_buff[part];
            r_partition_elem = r_histogram_buff[part + 1] - r_histogram_buff[part];

            if (r_partition_elem > ELEM_PER_BLOCK2)
            {
                dpu_results.ERROR_TYPE_0 = r_partition_elem;
                return 0;
            }

            if (s_partition_elem == 0) continue;
            else if (s_partition_elem <= ELEM_PER_BLOCK2)
            {
                printf("s_partition_elem: %u, r_partition_elem: %u\n", s_partition_elem, r_partition_elem);
                
                mram_read(
                    (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t)),
                    s_tuples_buff,
                    (s_partition_elem << 3));
                
                mram_read(
                    (__mram_ptr void const *)(R_key_sorted_addr + r_histogram_buff[part] * sizeof(tuplePair_t)),
                    r_tuples_buff,
                    (r_partition_elem << 3));

                mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, s_partition_elem);
                mergeSort(r_tuples_buff, (tuplePair_t *) r_sorted_buff, r_partition_elem);

                return_val = probe((tuplePair_t *) r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, s_partition_elem);
                mutex_lock(&(mutex_atomic[0x1F & part]));
                SHARED_COUNT += s_partition_elem;
                mutex_unlock(&(mutex_atomic[0x1F & part]));

                if (return_val == 0)
                {
                    dpu_results.ERROR_TYPE_3 = 33;
                    return 0;
                }

                mram_write(
                    (tuplePair_t *) s_sorted_buff,
                    (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t)),
                    (s_partition_elem << 3));
            }
            else 
            {
                printf("s_partition_elem: %u, r_partition_elem: %u\n", s_partition_elem, r_partition_elem);

                mram_read(
                    (__mram_ptr void const *)(R_key_sorted_addr + r_histogram_buff[part] * sizeof(tuplePair_t)),
                    r_tuples_buff,
                    (r_partition_elem << 3));

                mergeSort(r_tuples_buff, (tuplePair_t *) r_sorted_buff, r_partition_elem);

                int s_loop = s_partition_elem / ELEM_PER_BLOCK2;
                int s_remains = s_partition_elem % ELEM_PER_BLOCK2;

                for (int s_iter = 0; s_iter < (s_loop + 1); s_iter++)
                {
                    if (s_iter < s_loop)
                    {
                        mram_read(
                            (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_iter),
                            s_tuples_buff,
                            BLOCK_SIZE2);

                        mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, ELEM_PER_BLOCK2);
                        return_val = probe((tuplePair_t *) r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, ELEM_PER_BLOCK2);
                        mutex_lock(&(mutex_atomic[0x1F & part]));
                        SHARED_COUNT += ELEM_PER_BLOCK2;
                        mutex_unlock(&(mutex_atomic[0x1F & part]));

                        if (return_val == 0)
                        {
                            dpu_results.ERROR_TYPE_3 = 34;
                            return 0;
                        }

                        mram_write(
                            (tuplePair_t *) s_sorted_buff,
                            (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_iter),
                            BLOCK_SIZE2);
                    }
                    else
                    {
                        mram_read(
                            (__mram_ptr void const *)(S_key_sorted_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_loop),
                            s_tuples_buff,
                            (s_remains << 3));
                        
                        mergeSort(s_tuples_buff, (tuplePair_t *) s_sorted_buff, s_remains);
                        return_val = probe((tuplePair_t *) r_sorted_buff, (tuplePair_t *) s_sorted_buff, r_partition_elem, s_remains);
                        mutex_lock(&(mutex_atomic[0x1F & part]));
                        SHARED_COUNT += s_remains;
                        mutex_unlock(&(mutex_atomic[0x1F & part]));

                        if (return_val == 0)
                        {
                            dpu_results.ERROR_TYPE_3 = 35;
                            return 0;
                        }

                        mram_write(
                            (tuplePair_t *) s_sorted_buff,
                            (__mram_ptr void *)(result_addr + s_histogram_buff[part] * sizeof(tuplePair_t) + BLOCK_SIZE2 * s_loop),
                            (s_remains << 3));

                    }
                }
            }
            barrier_wait(&my_barrier);
        }
    }

    return 0;
}