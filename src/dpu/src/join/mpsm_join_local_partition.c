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

#define RADIX_DIGIT 4
#define RADIX (1 << RADIX_DIGIT)

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host sort_merge_partitioning_arg param_sort_merge_partitioning;
__host sort_merge_probe_return_arg param_sort_merge_probe_return;

__host dpu_results_t dpu_results;

// The number of instructions
__host uint32_t nb_instrs;
// The number of cycles
__host uint32_t nb_cycles;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

uint32_t NR_INSTRUCTIONS = 0;
uint32_t NR_CYCLES = 0;

// Packet Value
uint32_t PACKET_BYTE = 0;
// Total number of elements in all packets
uint32_t TOTAL_ELEM = 0;
// Arg for sort-merge
uint32_t ELEM_RANGE = 0;

// Start Addr of a table
char *key_src_addr;
char *key_sorted_addr;
// Start Addr of each partition (when R)
char *partition_idx_addr;

// The number of tuples thread will handle
uint32_t tuples_per_th = 0;

// R Table or not
bool r_table = false;

// Variables for Histogram
uint32_t hist_interval;
uint32_t histogram_bucket_num = 0;
uint32_t *histogram_buff;

// Variables for Sort
void *sorted_buff1;
void *sorted_buff2;

/* Function for Histogram */
void SetHistogram(int32_t idx, uint32_t value)
{
	histogram_buff[idx] = value;
}
void IncrHistogram(int32_t idx)
{
	histogram_buff[idx] += 1;
}
uint32_t GetHistogram(int32_t idx)
{
	return histogram_buff[idx];
}
uint32_t GetIncrHistogram(int32_t idx)
{
	histogram_buff[idx] += 1;
    return (histogram_buff[idx] - 1);
}
/* Function for Histogram */

int main(void)
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
        key_src_addr = (char *) MRAM_BASE_ADDR + param_sort_merge_partitioning.r_packet_start_byte;
        key_sorted_addr = (char *) MRAM_BASE_ADDR + param_sort_merge_partitioning.r_sorted_start_byte;
        partition_idx_addr = (char *) MRAM_BASE_ADDR + param_sort_merge_partitioning.histogram_addr_start_byte;
        PACKET_BYTE = param_sort_merge_partitioning.num_packets * param_sort_merge_partitioning.packet_size;
    }
    else if (tasklet_id == 1)
    {
        ELEM_RANGE = param_sort_merge_partitioning.elem_range;
        hist_interval = param_sort_merge_partitioning.hist_interval;
        histogram_bucket_num = ELEM_RANGE / hist_interval + 1;
    }
    else if (tasklet_id == 2)
    {
        histogram_buff = (uint32_t*) mem_alloc(32 * 1024);
        tuples_per_th = BLOCK_SIZE / sizeof(tuplePair_t);
    }

    barrier_wait(&my_barrier);

    // Allocate Buffer
    void* tuples_read_buff = NULL;
    tuples_read_buff = mem_alloc(BLOCK_SIZE);

    // Buffer for each element in packet
    tuplePair_t *elem_packet;

    /*
     * Phase 1.
     *    - Build a histogram to partition each relation
     */

    if (tasklet_id == 4)
    {
        if (histogram_bucket_num == 0)
        {
            dpu_results.ERROR_TYPE_0 = 8;
            return 0;
        }

        // Memset Histogram
        for (uint32_t i = 0; i < histogram_bucket_num + 1; i++)
        {
            SetHistogram(i, 0);
        }
    }

    barrier_wait(&my_barrier);

    // Target bucket of histogram
    uint32_t dest_bucket = 0;
    // Build Histogram
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        uint32_t elems_to_validate = ((PACKET_BYTE) - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >> 3);

        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Total number of elements in packets
        uint32_t num_elem = 0;

        // Packets per each tasklet
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            elem_packet = (tuplePair_t *) tuples_read_buff;
            
            if (elem_packet[e].lvalue == 0) continue;
            else
            {
                num_elem++;

                dest_bucket = elem_packet[e].lvalue / hist_interval;
                if (dest_bucket >= histogram_bucket_num)
                {
                    dest_bucket = (histogram_bucket_num - 1);
                }

                // Histgoram build
                mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));
                IncrHistogram(dest_bucket);
                mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));
            }
        }
        
        mutex_lock(&(mutex_atomic[49]));
        // Count the total number of elements
        TOTAL_ELEM += num_elem;
        mutex_unlock(&(mutex_atomic[49]));
    }

    barrier_wait(&my_barrier);



    /*
     * Phase 3.
     *    - Build a cumulative histogram for address
     *    - Reallocate each tuples based on its range
     */

    // Buffer used for histogram build
    if (tasklet_id == 5)
    {
        uint32_t temp_hist = GetHistogram(0);
        uint32_t temp_hist2 = GetHistogram(0);

        SetHistogram(0, 0);

        for (uint32_t idx = 1; idx < (histogram_bucket_num + 1); idx++)
        {
            temp_hist2 = GetHistogram(idx);
            SetHistogram(idx, GetHistogram(idx - 1) + temp_hist);
            temp_hist = temp_hist2;
        }
        if (TOTAL_ELEM != GetHistogram(histogram_bucket_num))
        {
            printf("ERROR! TOTAL_ELEM != GetHistogram(histogram_bucket_num)\n");
        }

        // Store the local histogram
        int loops = histogram_bucket_num * sizeof(uint32_t) / BLOCK_SIZE;
        int remains = histogram_bucket_num * sizeof(uint32_t) % BLOCK_SIZE;

        for (int i = 0; i < loops; i++)
        {
            mram_write(
                &histogram_buff[PART_PER_BLOCK * i],
                (__mram_ptr void *)((uint32_t)partition_idx_addr + i * BLOCK_SIZE),
                BLOCK_SIZE);
        }
        if (remains > 0)
        {
            mram_write(
                &histogram_buff[PART_PER_BLOCK * loops],
                (__mram_ptr void *)((uint32_t)partition_idx_addr + loops * BLOCK_SIZE),
                remains);
        }
    }

    barrier_wait(&my_barrier);

    // Index of the histogram
    uint32_t hist = 0;

    // Store elements in random manner
    for (uint32_t byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < PACKET_BYTE; byte_offset += (NR_TASKLETS * BLOCK_SIZE))
    {
        uint32_t elems_to_validate = ((PACKET_BYTE) - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >> 3);

        // Read Packet
        mram_read(
            (__mram_ptr void const *)(key_src_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Tuples per each tasklet
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            elem_packet = (tuplePair_t *) tuples_read_buff;
            
            if (elem_packet[e].lvalue == 0) continue;
            else
            {
                dest_bucket = elem_packet[e].lvalue / hist_interval;
                if (dest_bucket >= histogram_bucket_num)
                {
                    dest_bucket = (histogram_bucket_num - 1);
                }

                mutex_lock(&(mutex_atomic[0x1F & dest_bucket]));
                hist = GetIncrHistogram(dest_bucket);
                mutex_unlock(&(mutex_atomic[0x1F & dest_bucket]));

                mram_write(
                    &(elem_packet[e]),
                    (__mram_ptr void *)(key_sorted_addr + hist * sizeof(tuplePair_t)),
                    sizeof(tuplePair_t));

                printf("");
            }
        }
    }

    barrier_wait(&my_barrier);

    for (uint32_t byte_offset = (TOTAL_ELEM << 3); byte_offset < PACKET_BYTE; byte_offset += BLOCK_SIZE)
    {
        uint32_t elems_to_validate = (PACKET_BYTE - byte_offset) >> 3;
        if (elems_to_validate > (BLOCK_SIZE >> 3)) elems_to_validate = (BLOCK_SIZE >> 3);

        tuplePair_t* buff = (tuplePair_t *) tuples_read_buff;
        
        // Read table
        mram_read(
            (__mram_ptr void const *)(key_sorted_addr + byte_offset),
            tuples_read_buff,
            (elems_to_validate << 3));

        // Set partition to R
        for (uint32_t e = 0; e < elems_to_validate; e++)
        {
            buff[e].lvalue = 0;
            buff[e].rvalue = 0;
        }

        // Write table
        mram_write(
            tuples_read_buff,
            (__mram_ptr void *)(key_sorted_addr + byte_offset),
            (elems_to_validate << 3));
    }

    barrier_wait(&my_barrier);

    ////////////////////////////////////////////////////////////////////////
    // Ends
    ////////////////////////////////////////////////////////////////////////

    if (tasklet_id == 0)
    {
        param_sort_merge_probe_return.result_size = TOTAL_ELEM;
    }

    barrier_wait(&my_barrier);
    
    return 0;
}
