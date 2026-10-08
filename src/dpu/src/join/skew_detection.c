/*
 * Select with multiple tasklets
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <handshake.h>
#include <barrier.h>
#include <mutex.h>
#include <string.h>

#include "argument.h"
#include "hash.h"

#define MUTEX_SIZE 52
#define NR_TASKLETS 12

// Variables from Host
__host skew_detection_arg param_skew_detection;
__host dpu_results_t dpu_results;

// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)

// Hash Table Associativity
#define NR_WAY 1
#define NR_SET (ELEM_PER_BLOCK2 / NR_WAY)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

uint32_t NR_CYCLE = 0;
uint32_t remained_elem = 0;

uint32_t Partition_ID = 0;
uint32_t Max_Partition_ID = 0;

// RR Histogram Buffer
tuplePair_t* rr_histogram_buffer;
uint32_t min_idx = 0;

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (tasklet_id == 0)
    {
        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        perfcounter_config(COUNT_CYCLES, 1);
    }

    barrier_wait(&my_barrier);

    // Set values for while loop
    if (tasklet_id == 0)
    {        
        Partition_ID = 0;
        Max_Partition_ID = (param_skew_detection.elem_num << 3) / BLOCK_SIZE2;
        
        remained_elem = param_skew_detection.elem_num % ELEM_PER_BLOCK2;
    
        if (remained_elem > 0) Max_Partition_ID++;
        else if (remained_elem == 0) remained_elem = ELEM_PER_BLOCK2;
    }
    else if (tasklet_id == 1)
    {
        // Initialize RR Histogram Buffer
        // lvalue: join key, rvalue: replication ratio
        rr_histogram_buffer = (tuplePair_t*) mem_alloc(BLOCK_SIZE2);
        memset(rr_histogram_buffer, 0, BLOCK_SIZE2);
    }

    // Allocate buffer for each tasklet
    tuplePair_t* read_buffer_payload = (tuplePair_t *) mem_alloc(BLOCK_SIZE2);

    // Address of StackNode
    uint32_t input_addr = MRAM_BASE_ADDR + param_skew_detection.input_offset;
    uint32_t rr_histogram_addr = MRAM_BASE_ADDR + param_skew_detection.rr_histogram_start_byte;

    barrier_wait(&my_barrier);

    while (1)
    {
        // Get ID
        mutex_lock(&(mutex_atomic[50]));
        // Get Partition ID
        uint32_t my_id = Partition_ID++;
        // End Condition
        if (my_id >= Max_Partition_ID)
        {
            mutex_unlock(&(mutex_atomic[50])); break;
        }
        mutex_unlock(&(mutex_atomic[50]));

        if ((my_id % 10) >= (uint32_t)param_skew_detection.sample_keep_blocks_per_10)
        {
            continue;
        }

        int read_size = BLOCK_SIZE2;
        int elem_size = ELEM_PER_BLOCK2;

        if (my_id == (Max_Partition_ID - 1))
        {
            read_size = (remained_elem << 3);
            elem_size = remained_elem;
        }

        // Read Data
        mram_read((__mram_ptr void const *)(input_addr + my_id * BLOCK_SIZE2), read_buffer_payload, read_size);
    
        for (int i = 0; i < elem_size; i++)
        {
            if (read_buffer_payload[i].lvalue != 0)
            {
                // Initialize
                bool found = false;
                uint32_t min_idx = 0; 
                uint32_t min_val = 0xFFFFFFFF;

                // Calculate Set Index
                uint32_t set = mix32_nomul(read_buffer_payload[i].lvalue) & (NR_SET - 1);

                // Update RR Histogram Buffer
                for (int j = 0; j < NR_WAY; j++)
                {
                    uint32_t idx = (set * NR_WAY) + j;

                    if (rr_histogram_buffer[idx].lvalue == read_buffer_payload[i].lvalue)
                    {
                        // Existing entry found
                        mutex_lock(&(mutex_atomic[idx]));
                        rr_histogram_buffer[idx].rvalue++;
                        mutex_unlock(&(mutex_atomic[idx]));

                        found = true;
                        break;
                    }
                    else if (rr_histogram_buffer[idx].lvalue == 0)
                    {
                        // Empty entry found
                        mutex_lock(&(mutex_atomic[idx]));
                        rr_histogram_buffer[idx].lvalue = read_buffer_payload[i].lvalue;
                        rr_histogram_buffer[idx].rvalue = 1;
                        mutex_unlock(&(mutex_atomic[idx]));

                        found = true;
                        break;
                    }
                    else if (rr_histogram_buffer[idx].rvalue < min_val)
                    {
                        // Track minimum rvalue entry for possible replacement
                        min_val = rr_histogram_buffer[idx].rvalue;
                        min_idx = idx;
                    }
                }

                if (found == false && min_val == 1)
                {
                    // Replace the entry with the minimum rvalue
                    mutex_lock(&(mutex_atomic[min_idx]));
                    rr_histogram_buffer[min_idx].lvalue = read_buffer_payload[i].lvalue;
                    rr_histogram_buffer[min_idx].rvalue = 1;
                    mutex_unlock(&(mutex_atomic[min_idx]));
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        printf("INSTR: %d\n", NR_CYCLE);
        dpu_results.cycle_count = NR_CYCLE;
    }
    else if (tasklet_id == 1)
    {
        // Write RR Histogram Buffer to MRAM
        mram_write(rr_histogram_buffer, (__mram_ptr void *)(rr_histogram_addr), BLOCK_SIZE2);
    }

    barrier_wait(&my_barrier);
    return 0;
}
