#ifndef _ARGUMENT_H_
#define _ARGUMENT_H_

#define NUM_MAX_RANKS 40

#define ERROR_TYPE_NULL 1
#define ERROR_TYPE_NOT_ACCEPTED 2

typedef int64_t Key64_t;
typedef uint64_t TupleID64_t;


#define JOIN_TYPE_EQUI 1

#define GLB_PART_ARR_L0 1

#define GLB_PART_TUPLE_ID_IDX0_2 2
#define GLB_PART_TUPLE_ID_IDX1_2 3

#define GLB_PART_TUPLE_ID_IDX0_4 4
#define GLB_PART_TUPLE_ID_IDX1_4 5
#define GLB_PART_TUPLE_ID_IDX2_4 6
#define GLB_PART_TUPLE_ID_IDX3_4 7

#define GLB_PART_JK_IDX0_2 12
#define GLB_PART_JK_IDX1_2 13

#define GLB_PART_JK_IDX0_4 14
#define GLB_PART_JK_IDX1_4 15
#define GLB_PART_JK_IDX2_4 16
#define GLB_PART_JK_IDX3_4 17

#define GLB_PART_JK_IDX0_8 18



#define PACKET_PROJ_TYPE_DOUBLE 1
#define PACKET_PROJ_TYPE_INT64 2
#define PACKET_PROJ_TYPE_INT8 3


typedef struct
{
    uint32_t lvalue;
    uint32_t rvalue;
} tuplePair_t;

typedef struct
{
    uint32_t bucket[2];
} tuplePairBucket_t;


#define DIVIDE_TID_KEY_FROM_TK_PAIR64(dst_tid, dst_key, tkpair64)		\
do{		\
	dst_tid = (tkpair64 >> 32);		\
	dst_key = (tkpair64 & 0xFFFFFFFF);		\
}while(0)

#define COMBINE_TID_KEY(tuple_id, join_key, result)		\
do{		\
	result = (((int64_t)tuple_id << 32) | (int64_t)join_key);		\
}while(0)


typedef struct
{
    int32_t ERROR_TYPE_0;
    int32_t ERROR_TYPE_1;
    int32_t ERROR_TYPE_2;
    int32_t ERROR_TYPE_3;
    uint64_t cycle_count;
    // hash join
    uint64_t miss_count;
    uint64_t total_count;
} dpu_results_t;


#define GET_RANK_DPU_FROM_TUPLE_ID(dpu_id, tuple_id)                        \
do {                                                                        \
    int32_t rank_id = ((uint32_t)tuple_id & (0xF8000000)) >> 27;            \
    int32_t local_dpu_id = (((uint32_t)tuple_id & (0x7e00000))) >> 21;      \
    dpu_id = ((rank_id << 6) + local_dpu_id);                               \
} while (0)

#define GET_LOCAL_ID_FROM_TUPLE_ID(local_id, tuple_id)     \
do{                                                        \
    local_id = (uint32_t)(tuple_id & (0x1fffff));          \
}while(0)

#define CREATE_TUPLE_ID(rank_id, dpu_id, tuple_idx, ret)      \
do {                                                          \
    ret = ((tuple_idx | (rank_id << 27)) | (dpu_id << 21));   \
} while(0)                                                            

typedef struct
{
    int32_t debugger_op;
    int32_t check_target_addr1;
    int32_t check_target_elem_num1;
    int32_t check_target_addr2;
    int32_t check_target_elem_num2;
    int32_t num_max_tuples;
    int32_t rank_id;
    int32_t dpu_id;
    int32_t total_rank;
} debugger_arg;

typedef struct
{
} debugger_return_arg;

typedef struct
{
    /* data */
    int32_t elem_range;
    int32_t num_packets;
    int32_t packet_size;
    int32_t hist_interval;
    int32_t r_packet_start_byte;
    int32_t r_sorted_start_byte;
    int32_t histogram_addr_start_byte;
    int32_t data_skewness;
} sort_merge_partitioning_arg;

typedef struct
{
    /* data */
    int32_t elem_range;
    int32_t hist_interval;
    int32_t r_sorted_start_byte;
    int32_t s_sorted_start_byte;
    int32_t r_histogram_start_byte;
    int32_t s_histogram_start_byte;
    int32_t result_probe_start_byte;
    int32_t r_total_bytes;
    int32_t s_total_bytes;
} sort_merge_sort_probe_arg;

typedef struct
{
    /* data */
    int32_t r_total_elem;
} sort_merge_partitioning_return_arg;

typedef struct
{
    /* data */
    int64_t result_size;    // Size in Byte
} sort_merge_probe_return_arg;


typedef struct
{
    int32_t table_r_start_byte;
    int32_t partitioned_table_r_start_byte;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
    int32_t partition_num;
    int32_t table_r_num_elem;
} hash_global_partitioning_arg;


typedef struct
{
    int32_t input_type; // 0: Compressed Array, 1: 2 Arrays. 
    int32_t partition_type; // 0: tid partitioning , 1: hash partitioning, 
    int32_t input_offset;
    int32_t elem_num;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
    int32_t partition_num;
    int32_t bankgroup;
    int32_t num_rankgroup;
    int32_t num_rank_in_rankgroup;
    // sort-merge
    int32_t join_type;
    int32_t hist_interval;
} glb_partition_count_arg;

typedef struct
{
    int32_t input_offset;
    int32_t elem_num;
    int32_t partition_info_start_byte;
    int32_t histogram_start_byte;
    int32_t partition_num;
    int32_t bankgroup;
    int32_t rankgroup;
    int32_t padding;
} bg_broadcast_count_arg;

// [AddPoint] Per-Tuple Partitioning Arguments for SEPID-Join
typedef struct
{
    /* data */
    int32_t rank_id;
    int32_t rank_num;
    int32_t elem_num;
    int32_t zipf_factor;
    int32_t input_offset;
    int32_t partition_num;
    int32_t partition_type;
    int32_t histogram_start_byte;
    int32_t rr_histogram_start_byte;
    int32_t partition_info_start_byte;
} pt_partition_count_arg;

// [AddPoint] Per-Tuple Partitioning Arguments for SEPID-Join
typedef struct
{
    /* data */
    int32_t dpu_id;
    int32_t rank_id;
    int32_t rank_num;
    int32_t elem_num;
    int32_t zipf_factor;
    int32_t packet_size;
    int32_t input_offset;
    int32_t result_offset;
    int32_t partition_num;
    int32_t partition_type;
    int32_t rr_histogram_start_byte;
    int32_t local_histogram_start_byte;
    int32_t packet_histogram_start_byte;
} pt_partition_packet_arg;

typedef struct
{
    uint32_t elem_num;
    uint32_t partition_num;
    uint32_t tid_offset;
    uint32_t payload_offset;
    uint32_t result_offset;
    uint32_t global_histogram_start_byte;
    uint32_t aggr_histogram_start_byte;
    int32_t bankgroup;
    int32_t rankgroup;
    int32_t padding;
} bg_broadcast_align_arg;

typedef struct
{
    int32_t join_result_start_byte;
    int32_t max_bytes;
    int32_t effective_bytes;
    int32_t dummy;
} finish_join_arg;

typedef struct
{
    int32_t elem_num;
    int32_t input_offset;
    int32_t rr_histogram_start_byte;
    int32_t sample_keep_blocks_per_10;
} skew_detection_arg;

typedef struct
{
    int32_t dpu_id;
    int32_t rank_id;
    int32_t input_type;         // 0: Compressed Array, 1: 2 Arrays. 
    int32_t partition_type;     // 0: tid partitioning , 1: hash partitioning, 
    int32_t input_offset1;
    int32_t input_offset2;
    int32_t elem_num;
    int32_t histogram_start_byte;
    int32_t packet_histogram_start_byte;
    int32_t partition_num;
    int32_t result_offset;
    int32_t packet_size;
    // opt
    int32_t partition_start;
    int32_t bankgroup;
    int32_t num_rankgroup;
    int32_t num_rank_in_rankgroup;
    // sort-merge
    int32_t join_type;
    int32_t hist_interval;
} glb_partition_packet_arg;

typedef struct
{
    int32_t input_arr_start_byte;
    int32_t input_data_bytes;
    int32_t partitioned_input_arr_start_byte;
    int32_t result_partition_info_start_byte;
    // if 0, do calculate partitioning
    int32_t do_calculate_partition_num; 
    int32_t shift_val;
    int32_t total_rank;
    int32_t tuple_size;
} hash_local_partitioning_arg;

typedef struct
{
    int32_t packet_start_byte;
    int32_t generated_local_tid_start_byte;
    int32_t partitioned_result_start_byte;
    int32_t result_partition_info_start_byte;
    // if 0, do calculate partitioning
    int32_t do_calculate_partition_num; 
    int32_t shift_val;
    int32_t total_rank;
    int32_t num_packets;
    int32_t tuple_size;
    int32_t packet_size;
    // to identify the rank group
    int32_t rank_id;
    int32_t dpu_id;
} packetwise_hash_local_partitioning_arg;

typedef struct
{
    int64_t elem_num;
    int64_t partition_num;
} packetwise_hash_local_partitioning_return_arg;

typedef struct
{
    int64_t elem_num;
    int64_t partition_num;
} hash_local_partitioning_return_arg;

typedef struct
{
	int32_t dpu_id;
	int32_t rank_id;
	int32_t histogram_start_byte;
	int32_t partition_info_start_byte;
	int32_t tid_start_byte; 
	int32_t payload_start_byte;
	int32_t result_tid_start_byte; // NOT USED
	int32_t result_payload_start_byte;
	int32_t table_num_elem;
	int32_t partition_num;
} packetwise_hash_global_partitioning_arg;

typedef struct
{
    int32_t packet_size;
    int32_t packet_type;
    int32_t dpu_total_packet_num;
} packetwise_hash_global_partitioning_return_arg;

#define TID_MASK 0x7FF
#define TID_SHIFT 23

typedef struct
{
    int32_t R_packet_start_byte;
    int32_t S_packet_start_byte;
    int32_t R_num_packets;
    int32_t S_num_packets;
    int32_t packet_size;
    int32_t result_start_byte;
} nested_loop_join_arg;

typedef struct
{
    int32_t R_packet_start_byte;
    int32_t num_input_bytes;
    int32_t hash_table_start_byte;
    int32_t hash_table_byte_size;
} packetwise_nphj_build_arg;

typedef struct
{
    int32_t S_packet_start_byte;
    int32_t num_input_bytes;
    int32_t hash_table_start_byte;
    int32_t hash_table_byte_size;
    int32_t result_start_byte;
} packetwise_nphj_probe_arg;

typedef struct
{
    uint32_t parted_R_offset;
    uint32_t parted_Tid_R_offset;
    uint32_t HT_offset;
    uint32_t parted_R_info_offset;
    uint32_t partition_num;
    uint32_t R_num;
    uint32_t compressed_key;
} hash_phj_build_arg;

typedef struct
{
    int32_t parted_S_offset;
    int32_t parted_Tid_S_offset;
    int32_t HT_offset;
    int32_t parted_S_info_offset;
    int32_t Result_offset;
    int32_t partition_num;
    int32_t S_num;
} packetwise_hash_phj_probe_arg;

typedef struct
{
    uint32_t parted_R_offset;
    uint32_t parted_Tid_R_offset;
    uint32_t HT_offset;
    uint32_t parted_R_info_offset;
    uint32_t partition_num;
    uint32_t R_num;
} packetwise_hash_phj_build_arg;

typedef struct
{
    int32_t parted_S_offset;
    int32_t parted_Tid_S_offset;
    int32_t HT_offset;
    int32_t parted_S_info_offset;
    int32_t Result_offset;
    int32_t partition_num;
    int32_t S_num;
    int32_t rank_id;
    int32_t dpu_id;
    int32_t key_table_type;
    int32_t probe_type;
} hash_phj_probe_arg;

typedef struct
{
    int32_t hash_table_offset;
    int32_t R_offset;
    int32_t R_byte_size;
} no_partitioned_join_build_hash_table_arg;

typedef struct
{
    int32_t hash_table_offset;
    int32_t hash_table_byte_size;
    int32_t S_offset;
    int32_t S_byte_size;
    int32_t result_offset;
} no_partitioned_join_probe_hash_table_arg;

typedef struct
{
	int64_t result_size;	// Size in Byte
    int32_t miss_count;
} hash_phj_probe_return_arg;

typedef struct
{
	int64_t result_size;	// Size in Byte
    int32_t miss_count;
} packetwise_nphj_probe_return_arg;


#define ADD_NODE_MASK       0xF0000000
#define SUB_NODE_MASK       0xE0000000
#define MUL_NODE_MASK       0xD0000000
#define DIV_NODE_MASK       0xC0000000

#define MRAM_REF_NODE_MASK  0xA0000000
#define CONSTANT_NODE_MASK  0x90000000
#define WRAM_REF_NODE_MASK  DPUWISE_TUPLEID_OFFSET00

#define STACK_END           0xFFFFFFFF

#define ARITH_TYPE_MASK     0x08000000
#define ARITH_OP_MASK       0xF0000000
#define ARITH_VALUE_MASK    0x03FFFFFF

#define STACK_NODE_DEPTH    10

#define ARITH_TYPE_MM 0x0
#define ARITH_TYPE_MW 0x1
#define ARITH_TYPE_MC 0x2
#define ARITH_TYPE_CM 0x3
#define ARITH_TYPE_CW 0x4
#define ARITH_TYPE_CC 0x5
#define ARITH_TYPE_WM 0x6
#define ARITH_TYPE_WW 0x7
#define ARITH_TYPE_WC 0x8

union ArithNode
{
    int32_t Node_Type;
    int32_t value;
    int32_t operand_offset;
};

typedef struct
{
    // TID
    int32_t tid_offset;
    int tids_size;
    int32_t output_offset;
    union ArithNode arith_node_stack[STACK_NODE_DEPTH];
} arithmetic_arg;

#define OP_TYPE_INT32 0x0
#define OP_TYPE_INT64 0x1
#define OP_TYPE_CHAR 0x2
#define OP_TYPE_DATE 0x3

// 8 bit

/***
 * 1024B Key based Pages
 * */

// 64 bit
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define RNS_PAGE_SIZE_128B 128
#define RNS_PAGE_SIZE_64B 64

#define PAGE_ELEM_NUM_1024B_8B 127

#define CALCULATE_LOCAL_PAGE_TUPLE_ID(my_packet_id, packet_idx, tuple_id) \
    do\
    {\
        tuple_id = (my_packet_id) * sizeof(data_packet_u64_128_t) + 8 + (packet_idx << 3);\
    } \
    while (0)


typedef struct
{
    int8_t num;
    int8_t src_rank;
    int8_t dst_rank;
    uint8_t packet_offset;
} packet_id_t;

/***
 * 128B Key based Pages
 * */
#define PAGE_ELEM_NUM_128B_8B 16

typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_i64_128_t;


typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_u64_128_t;

typedef struct
{
    tuplePair_t tup_elem[PAGE_ELEM_NUM_128B_8B];  
} data_packet_t64_128_t;


/***
 * 64B Key based Pages
 * */

#define PAGE_ELEM_NUM_64B_8B 8

typedef struct
{
    uint64_t element[PAGE_ELEM_NUM_128B_8B];
} data_packet_u64_64_t;

#endif
