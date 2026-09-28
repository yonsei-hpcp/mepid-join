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
#define RADIX_BL_SIZE 96

#define PARTITION_BL_LOG 10
#define PARTITION_BL_SIZE (1 << PARTITION_BL_LOG)
#define PARTITION_BL_ELEM (PARTITION_BL_SIZE >> 3)

#define NUM_DPU_LOG 6
#define NUM_DPU_RANK (1 << NUM_DPU_LOG)


#define CACHE_SIZE (1 << 13)

#define NR_TASKLETS 12

#define BLOCK_SIZE1 1800
#define BLOCK_SIZE1_ELEM (BLOCK_SIZE1 >> 3)

#define BLOCK_SIZE2 2048
#define BLOCK_SIZE2_ELEM (BLOCK_SIZE2 >> 3)
// Variables from Host
__host bg_broadcast_align_arg param_bg_broadcast_align;
__host dpu_results_t dpu_results;
// MRAM Stack Bottom
uint32_t MRAM_BASE_ADDR = (uint32_t)DPU_MRAM_HEAP_POINTER;

// Lock
uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];
// Mutex
MUTEX_INIT(my_mutex);

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t NR_CYCLE = 0;

int32_t TOTAL_PAGE = 0;
int32_t LAST_PAGE_ELEM = 0;

// TIDs Addr
uint32_t Tids_addr;
uint32_t Pair_addr;
uint32_t Tids_dest_addr;

// Payload Addr
uint32_t Payload_addr;
uint32_t Payload_dest_addr;

// Partition Infos
uint32_t global_histogram_addr;
uint32_t aggr_histogram_addr;
uint32_t result_addr;

// Initialize a local buffer to store the MRAM block
uint32_t *histogram_buff = NULL;
uint32_t *addr_buff = NULL;
uint32_t *hist_accmulate = NULL;


void SetHistogram(int32_t idx, uint32_t value)
{
    int32_t *wb = histogram_buff + idx;
    (*wb) = value;
}

void IncrHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    *wb += 1;
}

int32_t GetIncrHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    *wb += 1;
    return (*wb - 1);
}

#define GetIncrHistogram__(ret, radix)          \
    do                                          \
    {                                           \
        int32_t *wb = (histogram_buff + radix); \
        ret = *wb;                              \
        *wb += 1;                               \
    } while (0)

int32_t GetHistogram(int32_t idx)
{
    int32_t *wb = histogram_buff + idx;
    return *wb;
}

void AddHistogram(int32_t idx, uint32_t val)
{
    int32_t *wb = histogram_buff + idx;
    *wb += val;
}

void SubHistogram(int32_t idx, uint32_t val)
{
    int32_t *wb = histogram_buff + idx;
    *wb -= val;
}

inline uint32_t tid_partition(uint32_t tid)
{
    int32_t dpu_id;
    GET_RANK_DPU_FROM_TUPLE_ID(dpu_id, tid);
    return dpu_id;
}

#define SRC_NTH_OF_BANK_GROUP 8
#define DST_NTH_OF_BANK_GROUP 8

int main(void)
{
    /* Variables Setup */
    uint32_t tasklet_id = me();

    if (sizeof(tuplePair_t) != 8)
    {
        dpu_results.ERROR_TYPE_0 = 99;
        return 0;
    }

    int num_bgroup_per_rank = (NUM_DPU_RANK / param_bg_broadcast_align.bankgroup);

    if (tasklet_id == 0)
    {
        int32_t ranks_per_rgroup = 0;

        mem_reset();
        dpu_results.ERROR_TYPE_0 = 0;
        dpu_results.ERROR_TYPE_1 = 0;
        dpu_results.ERROR_TYPE_2 = 0;
        dpu_results.ERROR_TYPE_3 = 0;
        perfcounter_config(COUNT_CYCLES, 1);

        ranks_per_rgroup = (param_bg_broadcast_align.partition_num / num_bgroup_per_rank);
        printf("param_bg_broadcast_align.partition_num: %d\n", param_bg_broadcast_align.partition_num);

        // Setup Histogram (packet num) & Partiton info(of global partitioning phase)
        uint32_t *temp_histogram_buff = (uint32_t *) mem_alloc(2048 * sizeof(uint32_t));
        addr_buff = (uint32_t *) mem_alloc(2048 * sizeof(uint32_t));
        hist_accmulate = (uint32_t *) mem_alloc((ranks_per_rgroup + 1) * sizeof(uint32_t)); // Last elem for the total amount of data

        // make accumed histogram
        global_histogram_addr = MRAM_BASE_ADDR + param_bg_broadcast_align.global_histogram_start_byte;
        aggr_histogram_addr = MRAM_BASE_ADDR + param_bg_broadcast_align.aggr_histogram_start_byte;
        mram_read((__mram_ptr void const *)(aggr_histogram_addr), temp_histogram_buff, ranks_per_rgroup * sizeof(uint32_t));

        // Hist accumulated for each rank based on tuples
        hist_accmulate[0] = 0;
        for (int p = 0; p < ranks_per_rgroup; p++)
        {
            hist_accmulate[p + 1] = temp_histogram_buff[p] + hist_accmulate[p];
        }
        histogram_buff = (uint32_t *) temp_histogram_buff;

        ////////////////////////////////////////////////////////////////////////

        for (uint32_t i = 0; i < param_bg_broadcast_align.partition_num; i++)
        {
            uint32_t rank_id = i / num_bgroup_per_rank;

            if (rank_id == 0)
            {
                addr_buff[i] = i * temp_histogram_buff[rank_id];
            }
            else
            {
                addr_buff[i] = 0;

                for (int r = 0; r < rank_id; r++)
                    addr_buff[i] += temp_histogram_buff[r] * num_bgroup_per_rank;
                
                addr_buff[i] += (i - num_bgroup_per_rank * rank_id) * temp_histogram_buff[rank_id];
            }
        }

        ///////////////////////////////////////////////////////////////////////
        
        // Calculate the iterations
        uint32_t num_blocks = ((hist_accmulate[ranks_per_rgroup] * num_bgroup_per_rank) / BLOCK_SIZE2_ELEM);
        uint32_t last_block_size = ((hist_accmulate[ranks_per_rgroup] * num_bgroup_per_rank) % BLOCK_SIZE2_ELEM) * sizeof(tuplePair_t);
        printf("num_blocks: %u, last_block_size: %u\n", num_blocks, last_block_size);

        // Clear histogram_buff
        for (uint32_t i = 0; i < 2048; i++)
        {
            histogram_buff[i] = 0;
        }

        // Fill 0 to the result buffer
        for (uint32_t i = 0; i < (num_blocks); i++)
        {
            mram_write(
                histogram_buff,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_bg_broadcast_align.result_offset + BLOCK_SIZE2 * i),
                BLOCK_SIZE2);
        }

        if (last_block_size > 0)
        {
            mram_write(
                histogram_buff,
                (__mram_ptr void *)(MRAM_BASE_ADDR + param_bg_broadcast_align.result_offset + BLOCK_SIZE2 * (num_blocks)),
                last_block_size);
        }
    }

    barrier_wait(&my_barrier);
    uint32_t partition_num = param_bg_broadcast_align.partition_num;
    
    if (tasklet_id == 1)
    {
        // Address pointers of the current processing block in MRAM
        Payload_addr = MRAM_BASE_ADDR + param_bg_broadcast_align.payload_offset;
        // Result address
        result_addr = MRAM_BASE_ADDR + param_bg_broadcast_align.result_offset;
    }

    if (tasklet_id == 2)
    {
        memset(histogram_buff, 0, partition_num * sizeof(int32_t));
    }

    barrier_wait(&my_barrier);

    tuplePair_t *key_packet = (tuplePair_t *)mem_alloc(BLOCK_SIZE2);

    RadixPartitionArray(
        tasklet_id,
        Payload_addr,
        param_bg_broadcast_align.elem_num,
        key_packet,
        param_bg_broadcast_align.partition_num);

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        int total_ = 0;
        NR_CYCLE = perfcounter_get();

        // #ifdef VALIDATION
        int32_t *temp_buff = (int32_t *)addr_buff;

        int read_blocks = (param_bg_broadcast_align.partition_num * sizeof(int32_t)) / BLOCK_SIZE2;
        int leftovers = (param_bg_broadcast_align.partition_num * sizeof(int32_t)) % BLOCK_SIZE2;

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
            mram_read(
                (__mram_ptr void const *)(global_histogram_addr + i * BLOCK_SIZE2),
                temp_buff,
                BLOCK_SIZE2);

            for (uint32_t j = 0; j < (BLOCK_SIZE2 / sizeof(int32_t)); j++)
            {
                if (temp_buff[j] != histogram_buff[j + i * (BLOCK_SIZE2 / sizeof(int32_t))])
                {
                    dpu_results.ERROR_TYPE_3 = 5;
                }
            }
        }

        mram_read(
            (__mram_ptr void const *)(global_histogram_addr + (read_blocks - 1) * BLOCK_SIZE2),
            temp_buff, leftovers);

        for (uint32_t j = 0; j < leftovers / sizeof(int32_t); j++)
        {
            if (temp_buff[j] != histogram_buff[j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t))])
            {
                printf("[%2d/%2d]: %3d vs %3d |",
                       j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t)),
                       (leftovers / sizeof(int32_t)) + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t)),
                       temp_buff[j],
                       histogram_buff[j + (read_blocks - 1) * (BLOCK_SIZE2 / sizeof(int32_t))]);
                dpu_results.ERROR_TYPE_0 = 6;
            }
        }
        printf("\n");

        for (int i = 0; i < partition_num; i++)
        {
            total_ += histogram_buff[i];
        }

        printf("Total. %d/%d\n", total_, param_bg_broadcast_align.elem_num);
        if (total_ > param_bg_broadcast_align.elem_num)
        {
            dpu_results.ERROR_TYPE_0 = total_;
            dpu_results.ERROR_TYPE_1 = param_bg_broadcast_align.elem_num;
        }
    }

    barrier_wait(&my_barrier);

    return 0;
}

int RadixPartitionArray(
    uint32_t tasklet_id,
    uint32_t mram_source_addr,
    uint32_t num_elem,
    tuplePair_t *wram_buff,
    int partition_num)
{
    int RADIX = (partition_num - 1);
    // int RADIX = ((partition_num - 1) << 3) & 0xFFFFFFF8;
    int tot_bytes = num_elem * sizeof(tuplePair_t);
    int last_num = num_elem % BLOCK_SIZE2_ELEM;

    if (last_num == 0)
        last_num = BLOCK_SIZE2_ELEM;

    for (int byte_addr = tasklet_id * BLOCK_SIZE2; byte_addr < tot_bytes; byte_addr += (NR_TASKLETS * BLOCK_SIZE2))
    {
        int elem_n = BLOCK_SIZE2_ELEM;

        if ((byte_addr + BLOCK_SIZE2) >= tot_bytes)
        {
            elem_n = last_num;
        }

        // Read Data
        mram_read(
            (__mram_ptr void const *)(mram_source_addr + byte_addr),
            wram_buff,
            BLOCK_SIZE2);

        for (int e = 0; e < elem_n; e++)
        {
            if (((int32_t)(wram_buff[e].lvalue)) < 0)
            {
                dpu_results.ERROR_TYPE_3 = 1;
            }

            // Join Key exists
            if (wram_buff[e].lvalue != 0)
            {
                // Hash
                tuplePair_t pair;
                pair.lvalue = (uint32_t)(wram_buff[e].lvalue);
                pair.rvalue = (uint32_t)(wram_buff[e].rvalue);
                uint32_t hash_val = (RADIX & glb_partition_hash(wram_buff[e].lvalue));
                // uint32_t hash_val = (RADIX & glb_partition_hash(wram_buff[e].lvalue)) >> 3;

                // Get Lock
                mutex_lock(&(mutex_atomic[(hash_val & 31)]));
                uint32_t target_offset = addr_buff[hash_val] + GetIncrHistogram(hash_val);
                mutex_unlock(&(mutex_atomic[(hash_val & 31)]));

                mram_write(
                    (&pair),
                    (__mram_ptr void *)(result_addr + (target_offset * sizeof(tuplePair_t))),
                    sizeof(tuplePair_t));
            }
            else
            {   
                // Join Key not exists
                printf("wram_buff[%d/%d].lvalue: %u %u\n", e, elem_n, wram_buff[e].lvalue, wram_buff[e].rvalue);
            }
        }
    }

    if (tasklet_id == 0)
    {
        NR_CYCLE = perfcounter_get();
        dpu_results.cycle_count = NR_CYCLE;
    }

    // barrier_wait(&my_barrier);
    return 0;
}
