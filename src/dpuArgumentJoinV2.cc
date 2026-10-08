#include "idpHandler.hpp"


#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <thread>
#include <string.h>



namespace pidjoin
{
    void *IDPArgumentHandler::ConfigureGlobalPartitioningPacketArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        RankwiseMemoryBankBuffers_t * imm_hist_buffers,
        const char *input_buffer_name1, // tid
        const char *local_histogram_name,
        const char *packet_histogram_name,
        const char *input_buffer_name2, // Payload
        const char *result_buffer_name)
    {
        arg_rank.clear();
        glb_partition_packet_arg *arg = (glb_partition_packet_arg *)malloc(sizeof(glb_partition_packet_arg) * NUM_DPU_RANK);

        int div_val = 8;
        int packet_div = 1;

        StackNode *input_buffer_node1;
        input_buffer_node1 = idp_handler->FindNode(rank_id, input_buffer_name2);
        StackNode *local_histogram_node = idp_handler->FindNode(rank_id, local_histogram_name);
        StackNode *packet_histogram_node = idp_handler->FindNode(rank_id, packet_histogram_name);
        StackNode *input_buffer_node2 = NULL;

        auto& histogram_buff = imm_hist_buffers->at(rank_id);
        char* hist_ptr = histogram_buff.at(0);

        int total_packets = 0;

        for (int p = 0; p < (partition_num) / NUM_DPU_RANK; p++)
        {
            total_packets += ((int64_t*)hist_ptr)[p];
        }
        total_packets *= NUM_DPU_RANK;
        StackNode *result_buffer_node = idp_handler->PushStackNodeAligned(
            rank_id, result_buffer_name, NULL, total_packets * packet_size, 8192);

        arg[0].dpu_id = 0;
        arg[0].rank_id = rank_id;
        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset1 = input_buffer_node1->start_byte;
        arg[0].elem_num = input_buffer_node1->data_bytes[0] / div_val;
        arg[0].histogram_start_byte = local_histogram_node->start_byte;
        arg[0].packet_histogram_start_byte = packet_histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].result_offset = result_buffer_node->start_byte;
        arg[0].packet_size = packet_size;
        arg[0].bankgroup = idp_handler->join_design.bank_set;
        
        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].elem_num = input_buffer_node1->data_bytes[i] / div_val;
            // arg[i].elem_num = input_buffer_node1->data_bytes[0] / div_val;

            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureGlobalPartitioningPacketArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        RankwiseMemoryBankBuffers_t * imm_hist_buffers,
        const char *input_buffer_name1, // tid
        const char *local_histogram_name,
        const char *packet_histogram_name,
        const char *input_buffer_name2, // Payload
        const char *result_buffer_name,
        int num_rankgroup,
        int num_rank_in_rankgroup)
    {
        arg_rank.clear();
        glb_partition_packet_arg *arg = (glb_partition_packet_arg *)malloc(sizeof(glb_partition_packet_arg) * NUM_DPU_RANK);

        int div_val = 8;
        int packet_div = 1;

        StackNode *input_buffer_node1;

        input_buffer_node1 = idp_handler->FindNode(rank_id, input_buffer_name2);

        StackNode *local_histogram_node = idp_handler->FindNode(rank_id, local_histogram_name);
        StackNode *packet_histogram_node = idp_handler->FindNode(rank_id, packet_histogram_name);
        StackNode *input_buffer_node2 = NULL;

        auto& histogram_buff = imm_hist_buffers->at(rank_id);
        char* hist_ptr = histogram_buff.at(0);

        int total_packets = 0;

        for (int p = 0; p < (partition_num) / NUM_DPU_RANK; p++)
        {
            total_packets += ((int64_t*)hist_ptr)[p];
        }
        total_packets *= NUM_DPU_RANK;
        
        StackNode *result_buffer_node = idp_handler->PushStackNodeAligned(
            rank_id, result_buffer_name, NULL, total_packets * packet_size, 8192);

        arg[0].dpu_id = 0;
        arg[0].rank_id = rank_id;
        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset1 = input_buffer_node1->start_byte;
        arg[0].elem_num = input_buffer_node1->data_bytes[0] / div_val;
        arg[0].histogram_start_byte = local_histogram_node->start_byte;
        arg[0].packet_histogram_start_byte = packet_histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].result_offset = result_buffer_node->start_byte;
        arg[0].packet_size = packet_size;
        arg[0].bankgroup = idp_handler->join_design.bank_set;
        arg[0].num_rankgroup = num_rankgroup;
        arg[0].num_rank_in_rankgroup = num_rank_in_rankgroup;

        // Sort-merge
        arg[0].join_type = idp_handler->join_design.join_alg;
        arg[0].hist_interval = idp_handler->join_design.input_size / partition_num + 1;
        
        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].elem_num = input_buffer_node1->data_bytes[i] / div_val;
            // arg[i].elem_num = input_buffer_node1->data_bytes[0] / div_val;

            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureBGBroadcastingAlignArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int partition_num,
        RankwiseMemoryBankBuffers_t *imm_hist_buffers,
        const char *input_buffer_tid,
        const char *input_buffer_payload,
        const char *global_histogram_name,
        const char *aggr_histogram_name,
        const char *result_buffer_name,
        int bank_group)
    {
        arg_rank.clear();
        bg_broadcast_align_arg *arg = (bg_broadcast_align_arg *)malloc(sizeof(bg_broadcast_align_arg) * NUM_DPU_RANK);

        int div_val = 8;

        StackNode *input_payload_node = idp_handler->FindNode(rank_id, input_buffer_payload);
        StackNode *global_histogram_node = idp_handler->FindNode(rank_id, global_histogram_name);
        StackNode *aggr_histogram_node = idp_handler->FindNode(rank_id, aggr_histogram_name);

        auto& histogram_buff = imm_hist_buffers->at(rank_id);
        char* hist_ptr = histogram_buff.at(0);

        int max_elems = 0;

        int num_bankgroup_in_rank = NUM_DPU_RANK / bank_group;

        for (int p = 0; p < (partition_num) / num_bankgroup_in_rank; p++)
        {
            max_elems += ((uint32_t*)hist_ptr)[p];
        }
        max_elems *= num_bankgroup_in_rank;
        StackNode *result_buffer_node = idp_handler->PushStackNodeAligned(
            rank_id, result_buffer_name, NULL, max_elems * sizeof(tuplePair_t), 8192);

        arg[0].payload_offset = input_payload_node->start_byte;
        arg[0].elem_num = input_payload_node->data_bytes[0] / div_val;
        arg[0].aggr_histogram_start_byte = aggr_histogram_node->start_byte;
        arg[0].global_histogram_start_byte = global_histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].result_offset = result_buffer_node->start_byte;
        arg[0].bankgroup = bank_group;
        
        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_payload_node->data_bytes[i] / div_val;
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }
    
    void *IDPArgumentHandler::ConfigureGlobalPartitioningCountArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        const char *input_buffer_name,
        const char *partition_info_name,
        const char *histogram_name)
    {
        arg_rank.clear();
        glb_partition_count_arg *arg = (glb_partition_count_arg *)malloc(sizeof(glb_partition_count_arg) * NUM_DPU_RANK);

        int div_val = 8;

        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Push partition info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(
            rank_id, partition_info_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(
            rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));

        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / div_val;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].bankgroup = idp_handler->join_design.bank_set;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_buffer_node->data_bytes[i] / div_val;
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureGlobalPartitioningCountArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int packet_size,
        int partition_num,
        int partition_type,
        const char *input_buffer_name,
        const char *partition_info_name,
        const char *histogram_name,
        int num_rankgroup,
        int num_rank_in_rankgroup)
    {
        arg_rank.clear();
        glb_partition_count_arg *arg = (glb_partition_count_arg *)malloc(sizeof(glb_partition_count_arg) * NUM_DPU_RANK);

        int div_val = 8;

        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Partition info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(
            rank_id, partition_info_name, partition_sizes, partition_num * sizeof(int32_t));
        // Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(
            rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));

        arg[0].input_type = partition_type;
        arg[0].partition_type = partition_type;
        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / div_val;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].bankgroup = idp_handler->join_design.bank_set;
        arg[0].num_rankgroup = num_rankgroup;
        arg[0].num_rank_in_rankgroup = num_rank_in_rankgroup;

        // Sort-merge join
        arg[0].join_type = idp_handler->join_design.join_alg;
        arg[0].hist_interval = idp_handler->join_design.input_size / partition_num + 1;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_buffer_node->data_bytes[i] / div_val;
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void *IDPArgumentHandler::ConfigureBGBroadcastingCountArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int partition_num,
        const char *input_buffer_name,
        const char *partition_info_name,        // gp_partition_info
        const char *histogram_name,
        int bank_group)                         // gp_histogram
    {
        arg_rank.clear();
        bg_broadcast_count_arg *arg = (bg_broadcast_count_arg *)malloc(sizeof(bg_broadcast_count_arg) * NUM_DPU_RANK);

        int div_val = 8;
        int partition_sizes[NUM_DPU_RANK] = {};
        
        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);

        // Push partition info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(
            rank_id, partition_info_name, partition_sizes, ((partition_num == 1) ? 8 : partition_num * sizeof(int32_t)));
        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(
            rank_id, histogram_name, partition_sizes, ((partition_num == 1) ? 8 : partition_num * sizeof(int32_t)));

        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / div_val;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].partition_num = partition_num;
        arg[0].bankgroup = bank_group;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_buffer_node->data_bytes[i] / div_val;
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    
    //////////////////////////////////////////////////////////////////////////////
    // [AddPoint] Per-Tuple Partition Arguments for SEPID-Join
    //////////////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler:: ConfigurePerTuplePartitionCountArgRank(
        int rank_id, IDPHandler *idp_handler, DPUKernelParams_t &arg_rank,
        int rank_num,
        int packet_size,
        int partition_num,
        int partition_type,
        const char *input_buffer_name,
        const char *rr_histogram_name,
        const char *partition_info_name,
        const char *histogram_name)
    {
        arg_rank.clear();
        pt_partition_count_arg *arg = (pt_partition_count_arg *)malloc(sizeof(pt_partition_count_arg) * NUM_DPU_RANK);

        int partition_sizes[NUM_DPU_RANK] = {};

        for (int i = 0; i < NUM_DPU_RANK; i++)
        {
            partition_sizes[i] = partition_num * sizeof(int32_t);
        }

        // Input Buffer Node
        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);
        // Per-Tuple Replication Ratio Histogram Node
        StackNode *rr_histogram_node = idp_handler->FindNode(rank_id, rr_histogram_name);

        // Push Histogram Node
        StackNode *histogram_node = idp_handler->PushStackNode(
            rank_id, histogram_name, partition_sizes, partition_num * sizeof(int32_t));
        // Push Partition Info Node
        StackNode *partition_info_node = idp_handler->PushStackNode(
            rank_id, partition_info_name, partition_sizes, partition_num * sizeof(int32_t));

        arg[0].rank_id = rank_id;
        arg[0].rank_num = rank_num;
        arg[0].partition_num = partition_num;
        arg[0].partition_type = partition_type;
        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].histogram_start_byte = histogram_node->start_byte;
        arg[0].zipf_factor = idp_handler->join_design.zipf_factor;
        arg[0].rr_histogram_start_byte = rr_histogram_node->start_byte;
        arg[0].partition_info_start_byte = partition_info_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / sizeof(tuplePair_t);

        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_buffer_node->data_bytes[i] / sizeof(tuplePair_t);
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    //////////////////////////////////////////////////////////////////////////////
    // [AddPoint] Per-Tuple Partition Arguments for SEPID-Join
    //////////////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler:: ConfigurePerTuplePartitionPacketArgRank(
        int rank_id, IDPHandler *idp_handler, DPUKernelParams_t &arg_rank,
        int rank_num,
        int packet_size,
        int partition_num,
        int partition_type,
        const char *input_buffer_name,
        const char *rr_histogram_name,
        const char *local_histogram_name,
        const char *packet_histogram_name,
        const char *result_buffer_name,
        RankwiseMemoryBankBuffers_t *imm_hist_buffers)
    {
        arg_rank.clear();
        pt_partition_packet_arg *arg = (pt_partition_packet_arg *)malloc(sizeof(pt_partition_packet_arg) * NUM_DPU_RANK);

        // Input Buffer Node
        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);
        // Per-Tuple Replication Ratio Histogram Node
        StackNode *rr_histogram_node = idp_handler->FindNode(rank_id, rr_histogram_name);
        // Histogram of the number of packets per partition
        StackNode *packet_histogram_node = idp_handler->FindNode(rank_id, packet_histogram_name);
        // Histogram of the number of elements per partition
        StackNode *local_histogram_node = idp_handler->FindNode(rank_id, local_histogram_name);

        auto& histogram_buffer = imm_hist_buffers->at(rank_id);
        char* hist_ptr = histogram_buffer.at(0);
        
        // Calculate the number of total packets
        int total_packets = 0;
        for (int p = 0; p < (partition_num / NUM_DPU_RANK); p++) total_packets += ((int64_t*)hist_ptr)[p];
        total_packets *= NUM_DPU_RANK;

        // Push Result Buffer Node
        StackNode *result_buffer_node = idp_handler->PushStackNodeAligned(
            rank_id, result_buffer_name, NULL, total_packets * packet_size, 8192);

        arg[0].dpu_id = 0;
        arg[0].rank_id = rank_id;
        arg[0].rank_num = rank_num;
        arg[0].packet_size = packet_size;
        arg[0].partition_num = partition_num;
        arg[0].partition_type = partition_type;
        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].result_offset = result_buffer_node->start_byte;
        arg[0].zipf_factor = idp_handler->join_design.zipf_factor;
        arg[0].rr_histogram_start_byte = rr_histogram_node->start_byte;
        arg[0].local_histogram_start_byte = local_histogram_node->start_byte;
        arg[0].packet_histogram_start_byte = packet_histogram_node->start_byte;

        arg[0].elem_num = input_buffer_node->data_bytes[0] / sizeof(tuplePair_t);

        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].elem_num = input_buffer_node->data_bytes[i] / sizeof(tuplePair_t);
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigureDebuggerArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int total_rank,
        int debugger_op_type,
        const char *target_addr1,
        const char *target_addr2)
    {
        arg_rank.clear();
        debugger_arg *arg = (debugger_arg *)malloc(sizeof(debugger_arg) * NUM_DPU_RANK);

        StackNode *target_addr1_node;
        StackNode *target_addr2_node;

        if (target_addr1 != NULL)
        {
            target_addr1_node = idp_handler->FindNode(rank_id, target_addr1);
        }

        if (target_addr2 != NULL)
        {
            target_addr2_node = idp_handler->FindNode(rank_id, target_addr2);
        }

        arg[0].debugger_op = debugger_op_type;
        arg[0].check_target_addr1 = target_addr1_node->start_byte;
        arg[0].check_target_elem_num1 = target_addr1_node->data_bytes[0] / sizeof(int64_t);
        arg[0].check_target_addr2 = target_addr2_node->start_byte;
        arg[0].check_target_elem_num2 = target_addr2_node->data_bytes[0] / sizeof(int64_t);
        arg[0].rank_id = rank_id;
        arg[0].dpu_id = 0;
        arg[0].total_rank = total_rank;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].dpu_id = i;
            arg[i].check_target_elem_num1 = target_addr1_node->data_bytes[i] / sizeof(int64_t);
            arg[i].check_target_elem_num2 = target_addr2_node->data_bytes[i] / sizeof(int64_t);
            
            arg_rank.push_back((char*)(arg + i));
            
        }

        return NULL;
    }


    void *IDPArgumentHandler::ConfigureNestedLoopJoinArgRank(
            int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
            int total_rank,
            int packet_size,
            const char *R_packets_name,
            const char *S_packets_name,
            const char *result_node_name)
    {
        arg_rank.clear();
        nested_loop_join_arg* arg = (nested_loop_join_arg *)malloc(sizeof(nested_loop_join_arg) * NUM_DPU_RANK);

        StackNode *R_stacknode = idp_handler->FindNode(rank_id, R_packets_name);
        StackNode *S_stacknode = idp_handler->FindNode(rank_id, S_packets_name);

        StackNode *result_node = idp_handler->PushStackNodeAligned(
            rank_id, result_node_name, S_stacknode->data_bytes, S_stacknode->block_byte, 8192);
        
        // exit(-1);
        arg[0].packet_size = packet_size;
        arg[0].R_num_packets = R_stacknode->data_bytes[0] / packet_size;
        arg[0].R_packet_start_byte = R_stacknode->start_byte;
        arg[0].result_start_byte = result_node->start_byte;
        arg[0].S_num_packets = S_stacknode->data_bytes[0] / packet_size;
        arg[0].S_packet_start_byte = S_stacknode->start_byte;

        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].R_num_packets = R_stacknode->data_bytes[i] / packet_size;
            arg[i].S_num_packets = S_stacknode->data_bytes[i] / packet_size;
            
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    //////////////////////////////////////////////////////////////////////////////
    // [AddPoint] Skew Detection Arguments for SEPID-Join
    //////////////////////////////////////////////////////////////////////////////

    void *IDPArgumentHandler:: ConfigureSkewDetectionArgRank(
        int rank_id, IDPHandler *idp_handler, DPUKernelParams_t &arg_rank,
        float sample_rate,
        const char *input_buffer_name,
        const char *rr_histogram_name)
    {
        arg_rank.clear();
        skew_detection_arg *arg = (skew_detection_arg *)malloc(sizeof(skew_detection_arg) * NUM_DPU_RANK);

        StackNode *input_buffer_node = idp_handler->FindNode(rank_id, input_buffer_name);

        StackNode *rr_histogram_node = idp_handler->PushStackNodeAligned(
            rank_id, rr_histogram_name, NULL, 256 * sizeof(tuplePair_t), 8192);

        arg[0].input_offset = input_buffer_node->start_byte;
        arg[0].rr_histogram_start_byte = rr_histogram_node->start_byte;
        arg[0].elem_num = input_buffer_node->data_bytes[0] / sizeof(tuplePair_t);
        int sample_keep_blocks_per_10 = (int)(sample_rate * 10.0f + 0.5f);
        if (sample_keep_blocks_per_10 < 0) sample_keep_blocks_per_10 = 0;
        if (sample_keep_blocks_per_10 > 10) sample_keep_blocks_per_10 = 10;
        arg[0].sample_keep_blocks_per_10 = sample_keep_blocks_per_10;

        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].elem_num = input_buffer_node->data_bytes[i] / sizeof(tuplePair_t);

            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }

    void* IDPArgumentHandler::ConfigureFinishJoinArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        const char * join_result_name)
    {
        arg_rank.clear();
        finish_join_arg* arg = (finish_join_arg *)malloc(sizeof(finish_join_arg) * NUM_DPU_RANK);

        StackNode *join_result_node = idp_handler->FindNode(rank_id, join_result_name);

        int64_t leftover_bytes = 0;

        arg[0].effective_bytes = join_result_node->data_bytes[0];
        arg[0].join_result_start_byte = join_result_node->start_byte;
        arg[0].max_bytes = join_result_node->block_byte;
        
        leftover_bytes += (arg[0].max_bytes - arg[0].effective_bytes);
        arg_rank.push_back((char*)(arg + 0));
        
        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            // Fill in Args
            arg[i] = arg[0];
            arg[i].effective_bytes = join_result_node->data_bytes[i];
            
            leftover_bytes += (arg[i].max_bytes - arg[i].effective_bytes);
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }
}
