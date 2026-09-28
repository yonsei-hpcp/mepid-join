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

#define NUM_DPU_LOG 6
#define NUM_DPU_RANK (1 << NUM_DPU_LOG)

#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 2048
#define ELEM_PER_BLOCK1 (BLOCK_SIZE1 >> 3)
#define ELEM_PER_BLOCK2 (BLOCK_SIZE2 >> 3)
#define RR_LOOKUP_SIZE 256

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Variables from Host
__host pt_partition_packet_arg param_pt_partition_packet;
__host dpu_results_t dpu_results;

// Variables in IDP
uint32_t NR_CYCLE = 0;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t) DPU_MRAM_HEAP_POINTER;

// Histogram Address
uint32_t packet_histogram_addr;
uint32_t local_histogram_addr;
// Result Packet Address
uint32_t *addr_buffer = NULL;

// Histogram Buffer for IDPs
int32_t *histogram_buffer = NULL;
// Histogram Buffer for Ranks
int64_t *hist_accumulate = NULL;
int64_t *packet_histogram_buffer;

// RR Histogram Buffer
tuplePair_t* rr_histogram_buffer;
// IDP Index per RR
uint32_t* rr_address_buffer;
// RR hot key lookup buffers
uint32_t* rr_lookup_keys;
uint16_t* rr_lookup_pos;
uint16_t* rr_buffer;
uint32_t rr_hot_count = 0;

int32_t num_ranks = 0;
int32_t PACKET_SIZE = 0;

// Global Vars for Partition
uint32_t remained_partitions = 0;

#define SRC_NTH_OF_BANK_GROUP 8
#define DST_NTH_OF_BANK_GROUP 8

int8_t LUT[SRC_NTH_OF_BANK_GROUP][DST_NTH_OF_BANK_GROUP] =
{
    {0, 1, 2, 3, 4, 5, 6, 7},
    {7, 0, 1, 2, 3, 4, 5, 6},
    {6, 7, 0, 1, 2, 3, 4, 5},
    {5, 6, 7, 0, 1, 2, 3, 4},
    {4, 5, 6, 7, 0, 1, 2, 3},
    {3, 4, 5, 6, 7, 0, 1, 2},
    {2, 3, 4, 5, 6, 7, 0, 1},
    {1, 2, 3, 4, 5, 6, 7, 0},
};

void SetHistogram(int32_t idx, uint32_t value)
{
    int32_t *wb = histogram_buffer + idx;
    (*wb) = value;
}

int32_t GetIncrHistogram(int32_t idx)
{
    int32_t *wb = histogram_buffer + idx;
    *wb += 1;
    return (*wb - 1);
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

int32_t GetTaskIncrIndex(int idx, int replication_ratio, uint32_t* rr_address_buffer_thr)
{
    uint32_t *wb = rr_address_buffer_thr + idx;
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

int RadixPartitionArrayPacked(
    uint32_t tasklet_id,
    uint32_t num_elem,
    int num_partition,
    int partition_type,
    uint32_t mram_source_addr,
    tuplePair_t *wram_read_buffer,
    tuplePair_t *rr_histogram_buffer);

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

        PACKET_SIZE = param_pt_partition_packet.packet_size;
        num_ranks = param_pt_partition_packet.partition_num / NUM_DPU_RANK;
        // Packet Histogram Address
        packet_histogram_addr = MRAM_BASE_ADDR + param_pt_partition_packet.packet_histogram_start_byte;
        // Local Histogram Address
        local_histogram_addr = MRAM_BASE_ADDR + param_pt_partition_packet.local_histogram_start_byte;

        // Setup the address of result buffer for each IDPs
        addr_buffer = (uint32_t *) mem_alloc(2048 * sizeof(uint32_t));
        // Histogram of packets per rank (and IDPs)
        packet_histogram_buffer = (int64_t *) mem_alloc(1024 * sizeof(int64_t));
        mram_read((__mram_ptr void const *)(packet_histogram_addr), packet_histogram_buffer, num_ranks * sizeof(int64_t));

        // Setup accumulated histogram of packets per rank
        hist_accumulate = (int64_t *) mem_alloc((num_ranks + 1) * sizeof(int64_t));
        hist_accumulate[0] = 0;
        for (int r = 0; r < num_ranks; r++) hist_accumulate[r + 1] = hist_accumulate[r] + packet_histogram_buffer[r];

        // Initialize the histogram buffer for IDPs
        histogram_buffer = (int32_t *) packet_histogram_buffer;
        // for (int i = 0; i < 2048; i++) histogram_buffer[i] = 0;
        memset(histogram_buffer, 0, 2048 * sizeof(int32_t));

        int num_blocks = (hist_accumulate[num_ranks] * PACKET_SIZE * NUM_DPU_RANK) / BLOCK_SIZE2;
        int leftover_bytes = (hist_accumulate[num_ranks] * PACKET_SIZE * NUM_DPU_RANK) % BLOCK_SIZE2;

        for (int i = 0; i < num_blocks; i++)
        {
            mram_write(
                histogram_buffer,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_pt_partition_packet.result_offset + BLOCK_SIZE2 * i),
                BLOCK_SIZE2);
        }

        if (leftover_bytes > 0)
        {
            mram_write(
                histogram_buffer,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_pt_partition_packet.result_offset + BLOCK_SIZE2 * (num_blocks)),
                leftover_bytes);
        }
    }
    
    // Arguments Setting
    int32_t n_th_of_bg = param_pt_partition_packet.dpu_id / 8;

    barrier_wait(&my_barrier);

    // memset
    if (tasklet_id == 8)
    {
        memset(histogram_buffer, 0, param_pt_partition_packet.partition_num * sizeof(int32_t));

        // RR Histogram Buffer Load
        rr_histogram_buffer = (tuplePair_t*) mem_alloc(BLOCK_SIZE2);
        mram_read(
            (__mram_ptr void const *)((uint32_t)MRAM_BASE_ADDR + param_pt_partition_packet.rr_histogram_start_byte),
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
            int start_idx = (int) replication_ratio / param_pt_partition_packet.rank_num;

            if (rr_buffer != NULL) rr_buffer[i] = (uint16_t)replication_ratio;

            if (replication_ratio == 1) break;
            rr_hot_count++;

            // Initialize RR Address Buffer
            if (start_idx > 0) rr_address_buffer[i] = start_idx * param_pt_partition_packet.rank_id;
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

    // Result Address Calculation
    if (tasklet_id < 8)
    {
        for (int r = 0; r < num_ranks; r++)
        {
            uint32_t base_result_addr = MRAM_BASE_ADDR + param_pt_partition_packet.result_offset + ((int)hist_accumulate[r] * NUM_DPU_RANK * PACKET_SIZE);
            
            for (int dst_dpu = tasklet_id; dst_dpu < NUM_DPU_RANK; dst_dpu += 8)
            {
                int nth_bg_dst_dpu = (dst_dpu & 7);
                int nth_of_bg_dst_dpu = (dst_dpu / 8);

                int idx = r * NUM_DPU_RANK + dst_dpu;
                addr_buffer[idx] = base_result_addr + (LUT[n_th_of_bg][nth_of_bg_dst_dpu] + (nth_bg_dst_dpu << 3)) * PACKET_SIZE;
            }
        }
    }

    barrier_wait(&my_barrier);

    tuplePair_t *key_packet = (tuplePair_t *) mem_alloc(BLOCK_SIZE2);

    RadixPartitionArrayPacked(
        tasklet_id,
        param_pt_partition_packet.elem_num,
        param_pt_partition_packet.partition_num,
        param_pt_partition_packet.partition_type,
        MRAM_BASE_ADDR + param_pt_partition_packet.input_offset,
        key_packet,
        rr_histogram_buffer);

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        dpu_results.cycle_count = NR_CYCLE;

        // Result Verification
        int32_t *temp_buff = (int32_t *)addr_buffer;
        int read_blocks = (param_pt_partition_packet.partition_num * sizeof(int32_t)) / BLOCK_SIZE2;
        int leftovers = (param_pt_partition_packet.partition_num * sizeof(int32_t)) % BLOCK_SIZE2;

        if (leftovers == 0)
        {
            leftovers = BLOCK_SIZE2;
        }
        else
        {
            read_blocks += 1;
        }

        for (int i = 0; i < (read_blocks - 1); i++)
        {
            // Result Readback and Verification
            mram_read(
                (__mram_ptr void const *)(local_histogram_addr + i * BLOCK_SIZE2),
                temp_buff,
                BLOCK_SIZE2);

            for (uint32_t j = 0; j < (BLOCK_SIZE2 / sizeof(int32_t)); j++)
            {
                if (temp_buff[j] != histogram_buffer[j + i * (BLOCK_SIZE2 / sizeof(int32_t))])
                {
                    dpu_results.ERROR_TYPE_3 = histogram_buffer[j + i * (BLOCK_SIZE2 / sizeof(int32_t))];
                }
            }
        }

        mram_read(
            (__mram_ptr void const *)(local_histogram_addr + (read_blocks - 1) * BLOCK_SIZE2),
            temp_buff, leftovers);

        for (uint32_t j = 0; j < leftovers / sizeof(int32_t); j++)
        {
            if (temp_buff[j] != histogram_buffer[j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t))])
            {
                dpu_results.ERROR_TYPE_0 = 6;
            }
        }
        printf("\n");

        int total_ = 0;
        for (int i = 0; i < param_pt_partition_packet.partition_num; i++) total_ += histogram_buffer[i];

        printf("Total. %d/%d\n", total_, param_pt_partition_packet.elem_num);
        if (total_ > param_pt_partition_packet.elem_num && param_pt_partition_packet.partition_type == 1)
        {
            dpu_results.ERROR_TYPE_0 = total_;
            dpu_results.ERROR_TYPE_1 = param_pt_partition_packet.elem_num;
            dpu_results.ERROR_TYPE_2 = param_pt_partition_packet.partition_type;
        }
    }

    barrier_wait(&my_barrier);
    return 0;
}

int RadixPartitionArrayPacked(
    uint32_t tasklet_id,
    uint32_t num_elem,
    int num_partition,
    int partition_type,
    uint32_t mram_source_addr,
    tuplePair_t *wram_read_buffer,
    tuplePair_t *rr_histogram_buffer)
{
    if (tasklet_id == 1)
    {
        // Memset Histogram
        for (int i = 0; i < num_partition; i++)
            SetHistogram(i, 0);
    }

    barrier_wait(&my_barrier);
    
    int total_bytes = num_elem * sizeof(tuplePair_t);
    int last_elem = num_elem % ELEM_PER_BLOCK2;

    // Packet Granularity
    int elem_per_packet_shift = 0;
    int one_rnc_packet_per_rank_shift = 0;
    int elem_per_packet = PACKET_SIZE >> 3; // could be 1 2 4 8 16

    if (elem_per_packet == 1)
    {
        elem_per_packet_shift = 0;
        one_rnc_packet_per_rank_shift = 9;
    }
    else if (elem_per_packet == 2)
    {
        elem_per_packet_shift = 1;
        one_rnc_packet_per_rank_shift = 10;
    }
    else if (elem_per_packet == 4)
    {
        elem_per_packet_shift = 2;
        one_rnc_packet_per_rank_shift = 11;
    }

    if (last_elem == 0) last_elem = ELEM_PER_BLOCK2;

    const uint32_t partition_mask = (uint32_t)(num_partition - 1);
    const uint32_t rr_lookup_mask = (RR_LOOKUP_SIZE - 1);
    // [DebugPoint] RR Threshold for Linear Search vs. Hash Table Lookup
    const uint32_t rr_linear_threshold = 8;

    uint32_t cached_key = 0;
    int cached_rr_match_idx = -2;
    uint32_t cached_rr_address = (uint32_t)(num_partition + 1);
    uint32_t cached_replication_ratio = 1;

    // Build packets containing tuples
    for (int byte_addr = tasklet_id * BLOCK_SIZE2; byte_addr < total_bytes; byte_addr += (NR_TASKLETS * BLOCK_SIZE2))
    {
        int elem_size = ELEM_PER_BLOCK2;
        int read_bytes = BLOCK_SIZE2;
        if ((byte_addr + BLOCK_SIZE2) > total_bytes)
        {
            elem_size = last_elem;
            read_bytes = elem_size << 3;
        }

        // Read Data
        mram_read(
            (__mram_ptr void const *)(mram_source_addr + byte_addr),
            wram_read_buffer,
            read_bytes);

        for (int e = 0; e < elem_size; e++)
        {
            uint32_t key = (uint32_t)wram_read_buffer[e].lvalue;
            uint32_t key_hash = 0;

            int replication_ratio = 1;
            uint32_t rr_address = num_partition + 1; // Default: No Replication
            int rr_match_idx = -1;

            if (key == 0) continue;
            key_hash = mix32_nomul(key);

            // [DebugPoint] Check the cached key for RR match result reuse
            if (key == cached_key && cached_rr_match_idx != -2)
            {
                rr_match_idx = cached_rr_match_idx;
                rr_address = cached_rr_address;
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

                // If RR match found, get RR address and replication ratio
                if (rr_match_idx >= 0)
                {
                    rr_address = rr_histogram_buffer[rr_match_idx].rvalue;
                    if (rr_buffer != NULL) replication_ratio = rr_buffer[rr_match_idx];
                    else replication_ratio = (int) GetReplicationRatio(rr_histogram_buffer, rr_match_idx);
                }

                // Cache the RR match result for potential reuse in subsequent tuples with the same key
                cached_key = key;
                cached_rr_match_idx = rr_match_idx;
                cached_rr_address = rr_address;
                cached_replication_ratio = (uint32_t)replication_ratio;
            }

            tuplePair_t pair;
            pair.lvalue = (uint32_t)(wram_read_buffer[e].lvalue);
            pair.rvalue = (uint32_t)(wram_read_buffer[e].rvalue);

            uint32_t target_IDP = 0;
            // Global partitioning w/ build table replication
            if (partition_type == 0)
            {
                if (replication_ratio > 1)
                {
                    for (int r = 0; r < replication_ratio; r++)
                    {
                        target_IDP = rr_address + (uint32_t)r;
                        if (target_IDP >= (uint32_t)num_partition) break;

                        // Update histogram
                        mutex_lock(&mutex_atomic[target_IDP & 0x1F]);
                        int32_t offset_num = GetIncrHistogram(target_IDP);
                        mutex_unlock(&mutex_atomic[target_IDP & 0x1F]);

                        // Calculate tuple destination
                        uint32_t target_addr = addr_buffer[target_IDP];

                        int num_packets = (offset_num >> elem_per_packet_shift);
                        int loc_offset = (offset_num & (elem_per_packet - 1));
                        target_addr += (num_packets << one_rnc_packet_per_rank_shift) + (loc_offset << 3);

                        mram_write((&pair), (__mram_ptr void *)(target_addr), sizeof(tuplePair_t));
                    }
                }
                else
                {
                    target_IDP = key_hash & partition_mask;

                    mutex_lock(&(mutex_atomic[target_IDP & 0x1F]));
                    int32_t offset_num = GetIncrHistogram(target_IDP);
                    mutex_unlock(&(mutex_atomic[target_IDP & 0x1F]));

                    uint32_t target_addr = addr_buffer[target_IDP];
                    int num_packets = (offset_num >> elem_per_packet_shift);
                    int loc_offset = (offset_num & (elem_per_packet - 1));
                    target_addr += (num_packets << one_rnc_packet_per_rank_shift) + (loc_offset << 3);

                    mram_write((&pair), (__mram_ptr void *)(target_addr), sizeof(tuplePair_t));
                }
            }
            // Global partitioning w/ probe table distribution
            else
            {
                if (replication_ratio > 1)
                {
                    uint32_t replica_offset = 0;

                    if (rr_hot_count > rr_linear_threshold)
                    {
                        replica_offset = GetFastReplicaOffset(
                            (uint32_t)wram_read_buffer[e].rvalue,
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
                int32_t offset_num = GetIncrHistogram(target_IDP);
                mutex_unlock(&(mutex_atomic[target_IDP & 0x1F]));

                uint32_t target_addr = addr_buffer[target_IDP];
                int num_packets = (offset_num >> elem_per_packet_shift);
                int loc_offset = (offset_num & (elem_per_packet - 1));
                target_addr += (num_packets << one_rnc_packet_per_rank_shift) + (loc_offset << 3);

                mram_write((&pair), (__mram_ptr void *)(target_addr), sizeof(tuplePair_t));
            }
        }
    }

    return 0;
}
