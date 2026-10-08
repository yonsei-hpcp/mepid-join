/* 
no partitioned hash join that gets the packetwise input
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

#include "common.h"
#include "argument.h"
#include "hash.h"

#define MUTEX_SIZE 32
#define NR_TASKLETS 12

#define BLOCK_SIZE 2048

uint8_t __atomic_bit mutex_atomic[MUTEX_SIZE];

__host packetwise_nphj_build_arg param_packetwise_nphj_build;
__host dpu_results_t dpu_results;
__host uint64_t NB_INSTR;

uintptr_t MRAM_BASE_ADDR = DPU_MRAM_HEAP_POINTER;
uintptr_t hash_table_start;

BARRIER_INIT(my_barrier, NR_TASKLETS);

int miss_count = 0;
int SHARED_COUNT = 0;

int main(void)
{
    int tasklet_id = me();

    if (tasklet_id == 0)
    {
        // Reset the heap
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
    }

    barrier_wait(&my_barrier);

    // relation R start addr
    tuplePair_t *wram_buff = (tuplePair_t *)mem_alloc(BLOCK_SIZE);

    // if (tasklet_id == 0)
    // {
    memset(wram_buff, 0, BLOCK_SIZE);
    // }

    barrier_wait(&my_barrier);

    //////////////////////////////////////////////////////////////////////////
    // memset the hash table    
    //////////////////////////////////////////////////////////////////////////

    uint32_t hash_table_addr = MRAM_BASE_ADDR + param_packetwise_nphj_build.hash_table_start_byte;
    uint32_t hash_table_byte_size = param_packetwise_nphj_build.hash_table_byte_size;
    uint32_t hash_table_byte_size_leftover = hash_table_byte_size % BLOCK_SIZE;
    if (hash_table_byte_size_leftover == 0)
        hash_table_byte_size_leftover = BLOCK_SIZE;

    for (int byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < hash_table_byte_size; byte_offset += NR_TASKLETS * BLOCK_SIZE)
    {
        int memset_size = (BLOCK_SIZE);

        if ((byte_offset + BLOCK_SIZE) > hash_table_byte_size)
        {
            memset_size = (hash_table_byte_size_leftover);
        }

        mram_write(
            wram_buff, 
            (__mram_ptr void *)(hash_table_addr + byte_offset), 
            memset_size);   
    }

    barrier_wait(&my_barrier);

    //////////////////////////////////////////////////////////////////////////
    // build hash table    
    //////////////////////////////////////////////////////////////////////////

    uint32_t input_start_addr = MRAM_BASE_ADDR + param_packetwise_nphj_build.R_packet_start_byte;
    int total_bytes = param_packetwise_nphj_build.num_input_bytes;
    int leftover_bytes = total_bytes % BLOCK_SIZE;
    if (leftover_bytes == 0)
        leftover_bytes = BLOCK_SIZE;

    int hash_table_bucket_size = param_packetwise_nphj_build.hash_table_byte_size / sizeof(tuplePair_t);

    int elem_in_buff = BLOCK_SIZE / sizeof(int64_t);

    for (int byte_offset = tasklet_id * BLOCK_SIZE; byte_offset < total_bytes; byte_offset += NR_TASKLETS * BLOCK_SIZE)
    {
        // Read Page
		mram_read(
			(__mram_ptr void const *)(input_start_addr + byte_offset),
			wram_buff,
			BLOCK_SIZE);

        ////////////////////////////////////////////
        
        int elem_num = (BLOCK_SIZE >> 3);

        if ((byte_offset + BLOCK_SIZE) > total_bytes)
        {
            elem_num = (leftover_bytes >> 3);
        }

        ////////////////////////////////////////////

        for (int e = 0; e < elem_num; e++)
        {
            uint32_t jk = wram_buff[e].lvalue;
            if (jk != 0)
            {
                uint32_t hashed = join_hash2(jk);

                int local_count = 0;

                tuplePair_t hash_table_entry;
                
                while (1)
                {
                    uint32_t radix = hashed & 0x1f;
                    uint32_t bucket_index = hashed & (hash_table_bucket_size-1);

                    mutex_lock(&(mutex_atomic[radix]));
                        
                    mram_read(
                        (__mram_ptr const void*)(hash_table_addr + (bucket_index << 3)), 
                        &hash_table_entry, 
                        sizeof(tuplePair_t));

                    // hash table hit
                    if (hash_table_entry.lvalue == 0)
                    {
                        hash_table_entry.lvalue = jk;
                        hash_table_entry.rvalue = wram_buff[e].rvalue;

                        mram_write(
                            &hash_table_entry, 
                            (__mram_ptr void *)(hash_table_addr + (bucket_index << 3)), 
                            sizeof(tuplePair_t));

                        mutex_unlock(&(mutex_atomic[radix]));
                        break;
                    }
                    else
                    {
                        mutex_unlock(&(mutex_atomic[radix]));

                        mutex_lock(&(mutex_atomic[50]));
                        miss_count++;
                        mutex_unlock(&(mutex_atomic[50]));
                        
                        hashed += 1;

                        if (hashed >= (hash_table_bucket_size))
                        {
                            hashed = 0;
                        }
                        continue;
                    }
                }
            }
        }
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0)
    {
        NB_INSTR = perfcounter_get();
        printf("Build NB_INSTR: %lu hash_table_bucket_size: %d miss_count: %d hash_table_addr:%d total_bytes: %d\n", NB_INSTR, hash_table_bucket_size, miss_count, hash_table_addr, total_bytes);
    }
    return 0;
}
