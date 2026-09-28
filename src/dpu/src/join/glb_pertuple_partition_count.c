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

#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)
#define RR_LOOKUP_SIZE 256
#define RR_LINEAR_THRESHOLD 8

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host pt_partition_count_arg param_pt_partition_count;
__host dpu_results_t dpu_results;

// Variables in IDP
uint32_t NR_CYCLE = 0;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

// Target Partition Size
char* write_buffer;
// RR Histogram Base Addr
tuplePair_t* rr_histogram_buffer;
// IDP Index per RR
uint32_t* rr_address_buffer;
// RR hot key lookup buffers
uint32_t* rr_lookup_keys;
uint16_t* rr_lookup_pos;
uint16_t* rr_buffer;
uint32_t rr_hot_count = 0;

// Global Vars for Partition Calculation
uint32_t Source_Partition_ID = 0;
uint32_t Max_Source_Partition_ID = 0;

// Global Vars for Partition
uint32_t remained_partitions = 0;

// Histgoram Operations
void SetHistogram(int32_t idx, uint32_t value)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    (*wb) = value;
}

uint32_t GetHistogram(int32_t idx)
{
    uint32_t *wb = (uint32_t *)write_buffer + idx;
    return *wb;
}

void IncrHistogram(int32_t idx)
{
    uint32_t *wb = ((uint32_t *)write_buffer + idx);
    *wb += 1;
}

void IncrHistogramVal(int32_t idx, int32_t value)
{
    uint32_t *wb = ((uint32_t *)write_buffer + idx);
    *wb += value;
}

int32_t GetIncrIndex(int32_t idx, int32_t replication_ratio)
{
    uint32_t *wb = rr_address_buffer + idx;
    uint32_t ret = *wb;
    (*wb) += 1;
    // If wb exceed the replication ratio, set to 0
    if ((*wb) >= (uint32_t)replication_ratio) (*wb) = 0;
    return ret;
}

static inline uint32_t GetReplicationRatio(tuplePair_t *rr_histogram, int32_t idx)
{
    uint32_t curr = rr_histogram[idx].rvalue;
    uint32_t next = rr_histogram[idx + 1].rvalue;

    if (next == 1) return 1;
    if (next == curr) return 1024;
    if (next > curr) return next - curr;
    return (1024 - curr) + next;
}

static inline uint32_t GetFastReplicaOffset(uint32_t tuple_id, uint32_t key_hash, uint32_t replication_ratio)
{
    uint32_t selector = tuple_id ^ key_hash;
    selector ^= selector >> 10;
    selector &= 1023u;

    if ((replication_ratio & (replication_ratio - 1)) == 0)
        return selector & (replication_ratio - 1);

    return (selector * replication_ratio) >> 10;
}

int HashPartitionCalculationArray(
    uint32_t tasklet_id,
    uint32_t mram_source_addr,
    uint32_t mram_histogram_base_addr,
    uint32_t mram_partition_info_base_addr,
    tuplePair_t *wram_read_buffer_payload,
    tuplePair_t *rr_histogram_buffer,
    uint32_t num_elem,
    int num_partition,
    int partition_type)
{
    uint32_t remained_elem = (num_elem % (ELEM_PER_BLOCK2));
    uint32_t histogram_bytes = ((uint32_t)num_partition) * sizeof(uint32_t);

    int use_range_delta = 0;
    int32_t *range_delta_buffer = NULL;

    // [DebugPoint] Use when the histogram size is small
    if (partition_type == 0 && ((histogram_bytes << 1) <= (uint32_t)(2048 * 4)))
    {
        use_range_delta = 1;
        range_delta_buffer = (int32_t *)(write_buffer + histogram_bytes);
    }

    // Build Histogram
    if (tasklet_id == 0)
    {
        Source_Partition_ID = 0;
        Max_Source_Partition_ID = ((num_elem << 3) / BLOCK_SIZE2);

        if (remained_elem > 0)
        {
            Max_Source_Partition_ID++;
        }
    }
    
    if (remained_elem == 0) remained_elem = ELEM_PER_BLOCK2;

    if (tasklet_id == 1)
    {
        // Memset Histogram
        for (int i = 0; i < num_partition; i++)
        {
            SetHistogram(i, 0);
            if (use_range_delta) range_delta_buffer[i] = 0;
        }
    }

    barrier_wait(&my_barrier);

    const uint32_t partition_mask = (uint32_t)(num_partition - 1);
    const uint32_t rr_lookup_mask = (RR_LOOKUP_SIZE - 1);
    // [DebugPoint] RR Threshold for Linear Search vs. Hash Table Lookup
    const uint32_t rr_linear_threshold = RR_LINEAR_THRESHOLD;

    // Caching keys and RR match results for R replication
    uint32_t cached_key = 0;
    int cached_rr_match_idx = -2;
    uint32_t cached_rr_address = (uint32_t)(num_partition + 1);
    uint32_t cached_replication_ratio = 1;
    // Make Histogram
    while (1)
    {
        // Get ID
        mutex_lock(&(mutex_atomic[50]));
        // Get Source partition id
        uint32_t my_id = Source_Partition_ID++;
        // End Condition
        if (my_id >= Max_Source_Partition_ID)
        {
            mutex_unlock(&(mutex_atomic[50])); break;
        }
        mutex_unlock(&(mutex_atomic[50]));

        int read_size = BLOCK_SIZE2;                                                                                                                                                      
        int elem_size = ELEM_PER_BLOCK2;

        if (my_id == (Max_Source_Partition_ID - 1))
        {
            read_size = (remained_elem << 3);
            elem_size = remained_elem;
        }
        
        // Read Data
        mram_read(
            (__mram_ptr void const *)(mram_source_addr + my_id * BLOCK_SIZE2), 
            wram_read_buffer_payload, 
            read_size);

        // Process Data
        for (int e = 0; e < elem_size; e++)
        {
            uint32_t key = (uint32_t)wram_read_buffer_payload[e].lvalue;
            uint32_t key_hash = 0;

            int replication_ratio = 1;
            int rr_address = num_partition + 1; // Default: No Replication
            int rr_match_idx = -1;

            if (key == 0) continue;
            key_hash = mix32_nomul(key);

            // [DebugPoint] Check the cached key for RR match result reuse
            if (key == cached_key && cached_rr_match_idx != -2)
            {
                rr_match_idx = cached_rr_match_idx;
                rr_address = (int)cached_rr_address;
                replication_ratio = (int)cached_replication_ratio;
            }
            else
            {
                if (rr_hot_count > 0)
                {
                    // If low skewness, linear search for RR match
                    if (rr_hot_count <= rr_linear_threshold || rr_lookup_keys == NULL || rr_lookup_pos == NULL)
                    {
                        for (uint32_t i = 0; i < rr_hot_count; i++)
                        {
                            if (rr_histogram_buffer[i].lvalue == key)
                            {
                                rr_match_idx = (int)i;
                                break;
                            }
                        }
                    }
                    // If high skewness, hash table lookup for RR match
                    else
                    {
                        uint32_t slot = key_hash & rr_lookup_mask;

                        for (int probe = 0; probe < RR_LOOKUP_SIZE; probe++)
                        {
                            uint16_t pos = rr_lookup_pos[slot];
                            if (pos == 0) break;

                            if (rr_lookup_keys[slot] == key)
                            {
                                rr_match_idx = (int)pos - 1;
                                break;
                            }

                            slot = (slot + 1) & rr_lookup_mask;
                        }
                    }
                }

                // If RR match, get RR address and replication ratio
                if (rr_match_idx >= 0)
                {
                    rr_address = rr_histogram_buffer[rr_match_idx].rvalue;
                    if (rr_buffer != NULL) replication_ratio = rr_buffer[rr_match_idx];
                    else replication_ratio = (int) GetReplicationRatio(rr_histogram_buffer, rr_match_idx);
                }

                // Cache the RR match result for potential reuse in subsequent tuples with the same key
                cached_key = key;
                cached_rr_match_idx = rr_match_idx;
                cached_rr_address = (uint32_t)rr_address;
                cached_replication_ratio = (uint32_t)replication_ratio;
            }

            // Global partitioning w/ build table replication
            if (partition_type == 0)
            {
                if (replication_ratio > 1)
                {
                    uint32_t range_begin = (uint32_t)rr_address;
                    uint32_t range_end = range_begin + (uint32_t)replication_ratio;

                    // Use range delta update when the histogram size is small to reduce locking overhead
                    if (use_range_delta)
                    {
                        if (range_begin < (uint32_t)num_partition)
                        {
                            if (range_end > (uint32_t)num_partition) range_end = (uint32_t)num_partition;

                            mutex_lock(&(mutex_atomic[range_begin & 0x1F]));
                            range_delta_buffer[range_begin] += 1;
                            mutex_unlock(&(mutex_atomic[range_begin & 0x1F]));

                            if (range_end < (uint32_t)num_partition)
                            {
                                mutex_lock(&(mutex_atomic[range_end & 0x1F]));
                                range_delta_buffer[range_end] -= 1;
                                mutex_unlock(&(mutex_atomic[range_end & 0x1F]));
                            }
                        }
                    }
                    // If the histogram size is large, directly update the histogram with fine-grained locking
                    else
                    {
                        uint32_t first_lock = range_begin & 0x1F;
                        int lock_slots = replication_ratio;
                        if (lock_slots > 32) lock_slots = 32;

                        for (int lock_offset = 0; lock_offset < lock_slots; lock_offset++)
                        {
                            uint32_t target_IDP = range_begin + (uint32_t)lock_offset;
                            uint32_t lock_idx = (first_lock + (uint32_t)lock_offset) & 0x1F;

                            mutex_lock(&(mutex_atomic[lock_idx]));
                            while (target_IDP < range_end && target_IDP < (uint32_t)num_partition)
                            {
                                IncrHistogram(target_IDP);
                                target_IDP += 32;
                            }
                            mutex_unlock(&(mutex_atomic[lock_idx]));
                        }
                    }
                }
                else
                {
                    uint32_t target_IDP = key_hash & partition_mask;
                    mutex_lock(&(mutex_atomic[target_IDP & 0x1F]));
                    IncrHistogram(target_IDP);
                    mutex_unlock(&(mutex_atomic[target_IDP & 0x1F]));
                }
            }
            // Global partitioning w/ probe table distribution
            else
            {
                uint32_t target_IDP = 0;
                if (replication_ratio > 1)
                {
                    uint32_t replica_offset = 0;
                    
                    // Use a hash-based replica offset calculation for high skewness
                    if (rr_hot_count > rr_linear_threshold)
                    {
                        replica_offset = GetFastReplicaOffset(
                            (uint32_t)wram_read_buffer_payload[e].rvalue,
                            key_hash,
                            (uint32_t)replication_ratio);
                    }
                    else
                    {
                        mutex_lock(&mutex_atomic[rr_address & 0x1F]);
                        replica_offset = (uint32_t)GetIncrIndex(rr_match_idx, replication_ratio);
                        mutex_unlock(&mutex_atomic[rr_address & 0x1F]);
                    }

                    target_IDP = ((uint32_t)rr_address + replica_offset) & partition_mask;
                }
                else
                {
                    target_IDP = key_hash & partition_mask;
                }

                mutex_lock(&(mutex_atomic[target_IDP & 0x1F]));
                IncrHistogram(target_IDP);
                mutex_unlock(&(mutex_atomic[target_IDP & 0x1F]));
            }
        }
    }

    barrier_wait(&my_barrier);

    // Make global histogram
    if (tasklet_id == 0)
    {
        if (use_range_delta)
        {
            int32_t active = 0;
            for (int i = 0; i < num_partition; i++)
            {
                active += range_delta_buffer[i];
                if (active != 0) IncrHistogramVal(i, active);
            }
        }

        int loops = (num_partition) * sizeof(uint32_t) / BLOCK_SIZE2;
        int remains = (num_partition) * sizeof(uint32_t) % BLOCK_SIZE2;

        for (int i = 0; i < loops; i++)
            mram_write((void*)(write_buffer + i * BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_histogram_base_addr + i * BLOCK_SIZE2), BLOCK_SIZE2);
        if (remains > 0)
            mram_write((void*)(write_buffer + (loops)*BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_histogram_base_addr + (loops)*BLOCK_SIZE2), remains);

        // Modify Histogram as start index
        uint32_t temp_before = GetHistogram(0);
        uint32_t temp_before_2 = GetHistogram(0);

        SetHistogram(0, 0);

        for (int i = 1; i < num_partition; i++)
        {
            temp_before_2 = GetHistogram(i);
            SetHistogram(i, GetHistogram(i - 1) + temp_before);
            temp_before = temp_before_2;
        }

        for (int i = 0; i < loops; i++)
            mram_write((void*)(write_buffer + i * BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_partition_info_base_addr + i * BLOCK_SIZE2), BLOCK_SIZE2);
        if (remains > 0)
            mram_write((void*)(write_buffer + (loops)*BLOCK_SIZE2), (__mram_ptr void *) ((uint32_t)mram_partition_info_base_addr + (loops)*BLOCK_SIZE2), remains);
    }

    barrier_wait(&my_barrier);
    return 0;
}

int main()
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
    
        write_buffer = (char*)mem_alloc(2048 * 4);
        memset(write_buffer, 0, param_pt_partition_count.partition_num * sizeof(int32_t));
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 1)
    {
        rr_histogram_buffer = (tuplePair_t*) mem_alloc(BLOCK_SIZE2);
        mram_read(
            (__mram_ptr void const *)((uint32_t)MRAM_BASE_ADDR + param_pt_partition_count.rr_histogram_start_byte),
            rr_histogram_buffer,
            BLOCK_SIZE2);

        // RR Address Buffer
        rr_address_buffer = (uint32_t*) mem_alloc(BLOCK_SIZE2);
        memset(rr_address_buffer, 0, BLOCK_SIZE2);

        // RR Buffer and RR hot count
        rr_hot_count = 0;
        rr_buffer = (uint16_t*) mem_alloc(ELEM_PER_BLOCK2 * sizeof(uint16_t));
        if (rr_buffer != NULL)
        {
            for (int i = 0; i < ELEM_PER_BLOCK2; i++) rr_buffer[i] = 1;
        }

        // Update RR hot count and replication ratio buffer
        for (int i = 0; i < 255; i++)
        {
            uint32_t replication_ratio = GetReplicationRatio(rr_histogram_buffer, i);
            int start_idx = (int) replication_ratio / param_pt_partition_count.rank_num;

            if (rr_buffer != NULL) rr_buffer[i] = (uint16_t)replication_ratio;

            if (replication_ratio == 1) break;
            rr_hot_count++;

            // Initialize RR Address Buffer
            if (start_idx > 0) rr_address_buffer[i] = start_idx * param_pt_partition_count.rank_id;
            else rr_address_buffer[i] = 0;
        }

        rr_lookup_keys = NULL;
        rr_lookup_pos = NULL;

        // Hash table setup for hot keys lookup
        if (rr_hot_count > 0 && rr_hot_count < RR_LOOKUP_SIZE)
        {
            rr_lookup_keys = (uint32_t*) mem_alloc(RR_LOOKUP_SIZE * sizeof(uint32_t));
            rr_lookup_pos = (uint16_t*) mem_alloc(RR_LOOKUP_SIZE * sizeof(uint16_t));

            if (rr_lookup_keys != NULL && rr_lookup_pos != NULL)
            {
                memset(rr_lookup_keys, 0, RR_LOOKUP_SIZE * sizeof(uint32_t));
                memset(rr_lookup_pos, 0, RR_LOOKUP_SIZE * sizeof(uint16_t));

                for (uint32_t i = 0; i < rr_hot_count; i++)
                {
                    uint32_t key = rr_histogram_buffer[i].lvalue;
                    uint32_t slot = mix32_nomul(key) & (RR_LOOKUP_SIZE - 1);

                    while (rr_lookup_pos[slot] != 0)
                        slot = (slot + 1) & (RR_LOOKUP_SIZE - 1);

                    rr_lookup_keys[slot] = key;
                    rr_lookup_pos[slot] = (uint16_t)(i + 1);
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    tuplePair_t *read_buffer_payload = (tuplePair_t*) mem_alloc(BLOCK_SIZE2);

    HashPartitionCalculationArray(
        tasklet_id,
        MRAM_BASE_ADDR + param_pt_partition_count.input_offset,
        MRAM_BASE_ADDR + param_pt_partition_count.histogram_start_byte, 
        MRAM_BASE_ADDR + param_pt_partition_count.partition_info_start_byte,
        read_buffer_payload,
        rr_histogram_buffer,
        param_pt_partition_count.elem_num,
        param_pt_partition_count.partition_num,
        param_pt_partition_count.partition_type);
    
    barrier_wait(&my_barrier);

    // Cycle Count
    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        printf("DPU Cycle Count: %d\n", NR_CYCLE);
        dpu_results.cycle_count = NR_CYCLE;
    }
    
    barrier_wait(&my_barrier);
    return 0;
}
