#include "join_internals.hpp"

#include <algorithm>
#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>


namespace pidjoin
{
    std::unordered_map<std::string, int> operator_name_map =
    {
        // DPU Functions
        {"DPU_FUNC_LOCAL_HASH_PARTITIONING", DPU_FUNC_LOCAL_HASH_PARTITIONING},
        {"DPU_FUNC_GLOBAL_HASH_PARTITIONING", DPU_FUNC_GLOBAL_HASH_PARTITIONING},
        {"DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE", DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE},
        {"DPU_FUNC_PHJ_PROBE_HASH_TABLE", DPU_FUNC_PHJ_PROBE_HASH_TABLE},
        {"DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING", DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING},
        {"DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING", DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING},
        {"DPU_FUNC_PACKETWISE_NPHJ_BUILD", DPU_FUNC_PACKETWISE_NPHJ_BUILD},
        {"DPU_FUNC_PACKETWISE_NPHJ_PROBE", DPU_FUNC_PACKETWISE_NPHJ_PROBE},
        {"DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE", DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE},
        {"DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE", DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE},
        {"DPU_FUNC_NPHJ_BUILD_HASH_TABLE", DPU_FUNC_NPHJ_BUILD_HASH_TABLE},
        {"DPU_FUNC_NPHJ_PROBE_HASH_TABLE", DPU_FUNC_NPHJ_PROBE_HASH_TABLE},
	    {"DPU_FUNC_GLB_PARTITION_COUNT", DPU_FUNC_GLB_PARTITION_COUNT},
        {"DPU_FUNC_GLB_PARTITION_PACKET", DPU_FUNC_GLB_PARTITION_PACKET},
	    {"DPU_FUNC_GLB_CHIPWISE_PARTITION_COUNT", DPU_FUNC_GLB_CHIPWISE_PARTITION_COUNT},
        {"DPU_FUNC_GLB_CHIPWISE_PARTITION_PACKET", DPU_FUNC_GLB_CHIPWISE_PARTITION_PACKET},
        {"DPU_FUNC_BG_BROADCAST_COUNT", DPU_FUNC_BG_BROADCAST_COUNT},
        {"DPU_FUNC_BG_BROADCAST_ALIGN", DPU_FUNC_BG_BROADCAST_ALIGN},
        {"DPU_FUNC_NESTED_LOOP_JOIN", DPU_FUNC_NESTED_LOOP_JOIN},
        {"DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING", DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING},
        {"DPU_FUNC_MPSM_JOIN_SORT_PROBE", DPU_FUNC_MPSM_JOIN_SORT_PROBE},
        {"DPU_FUNC_FINISH_JOIN", DPU_FUNC_FINISH_JOIN},
        // Host Functions
        {"HOST_FUNC_INVALIDATE_STACKNODE", HOST_FUNC_INVALIDATE_STACKNODE}, 
        {"HOST_FUNC_ROTATE_AND_STREAM", HOST_FUNC_ROTATE_AND_STREAM},
        {"HOST_FUNC_CALCULATE_PAGE_HISTOGRAM", HOST_FUNC_CALCULATE_PAGE_HISTOGRAM},
        {"HOST_FUNC_LOAD_COLUMN", HOST_FUNC_LOAD_COLUMN},
        {"HOST_FUNC_SEND_DATA_OPT", HOST_FUNC_SEND_DATA_OPT},
        {"HOST_FUNC_RECV_DATA_OPT", HOST_FUNC_RECV_DATA_OPT},
        {"HOST_FUNC_BROADCAST_DATA", HOST_FUNC_BROADCAST_DATA},
        {"HOST_FUNC_CHIPWISE_BROADCAST", HOST_FUNC_CHIPWISE_BROADCAST},
        // Compound Functions
        {"COMPOUND_FUNC_RNS_JOIN", COMPOUND_FUNC_RNS_JOIN},
        {"COMPOUND_FUNC_IDP_JOIN", COMPOUND_FUNC_IDP_JOIN},
        {"COMPOUND_FUNC_GLB_PARTITION", COMPOUND_FUNC_GLB_PARTITION},
        {"COMPOUND_FUNC_GLB_CHIPWISE_PARTITION", COMPOUND_FUNC_GLB_CHIPWISE_PARTITION},
        {"COMPOUND_FUNC_BG_BROADCAST", COMPOUND_FUNC_BG_BROADCAST},
        // Control functions
        {"CONTROL_FUNC_SYNC_THREADS", CONTROL_FUNC_SYNC_THREADS},
        // [AddPoint] DPU Functions for SEPID-Join
        {"DPU_FUNC_SKEW_DETECTION", DPU_FUNC_SKEW_DETECTION},
        {"DPU_FUNC_GLB_PERTUPLE_PARTITION_COUNT", DPU_FUNC_GLB_PERTUPLE_PARTITION_COUNT},
        {"DPU_FUNC_GLB_PERTUPLE_PARTITION_PACKET", DPU_FUNC_GLB_PERTUPLE_PARTITION_PACKET},
        // [AddPoint] Compound Functions for SEPID-Join
        {"COMPOUND_FUNC_SKEW_DETECTION", COMPOUND_FUNC_SKEW_DETECTION},
        {"COMPOUND_FUNC_GLB_PERTUPLE_PARTITION", COMPOUND_FUNC_GLB_PERTUPLE_PARTITION},
    };

    std::unordered_map<int, std::string> param_binary_name_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, "src/dpu/bin/a2a_srj_local_partition"},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, "src/dpu/bin/a2a_srj_global_partition"},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, "src/dpu/bin/a2a_srj_build_linear_probe"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, "src/dpu/bin/a2a_srj_probe"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "src/dpu/bin/a2a_srj_probe_inner_mt_linear_probe"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "src/dpu/bin/a2a_packetwise_srj_local_partition"},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "src/dpu/bin/a2a_packetwise_srj_global_partition"},
        {DPU_FUNC_PACKETWISE_NPHJ_BUILD, "src/dpu/bin/packetwise_no_partitioned_hash_join_build"},
        {DPU_FUNC_PACKETWISE_NPHJ_PROBE, "src/dpu/bin/packetwise_no_partitioned_hash_join_probe"},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, "src/dpu/bin/a2a_packetwise_srj_build"},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, "src/dpu/bin/no_partitioned_hash_join_build"},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, "src/dpu/bin/no_partitioned_hash_join_probe"},
	    {DPU_FUNC_GLB_PARTITION_COUNT, "src/dpu/bin/glb_partition_count"},
        {DPU_FUNC_GLB_PARTITION_PACKET, "src/dpu/bin/glb_partition_packet"},
        {DPU_FUNC_GLB_CHIPWISE_PARTITION_COUNT, "src/dpu/bin/glb_chipwise_partition_count"},
        {DPU_FUNC_GLB_CHIPWISE_PARTITION_PACKET, "src/dpu/bin/glb_chipwise_partition_packet"},
        {DPU_FUNC_BG_BROADCAST_COUNT, "src/dpu/bin/bg_broadcast_count"},
        {DPU_FUNC_BG_BROADCAST_ALIGN, "src/dpu/bin/bg_broadcast_align"},
        {DPU_FUNC_NESTED_LOOP_JOIN, "src/dpu/bin/nested_loop_join"},
        {DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING, "src/dpu/bin/mpsm_join_local_partition"},
        {DPU_FUNC_MPSM_JOIN_SORT_PROBE, "src/dpu/bin/mpsm_join_sort_probe"},
        {DPU_FUNC_FINISH_JOIN, "src/dpu/bin/finish_join"},
        // [AddPoint] SEPID-Join Functions
        {DPU_FUNC_SKEW_DETECTION, "src/dpu/bin/skew_detection"},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_COUNT, "src/dpu/bin/glb_pertuple_partition_count"},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_PACKET, "src/dpu/bin/glb_pertuple_partition_packet"},
    };

    std::unordered_map<int, std::string> param_name_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, "param_local_hash_partitioning"},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, "param_global_hash_partitioning"},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, "param_phj_build_hash_table"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, "param_phj_probe_hash_table"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "param_phj_probe_hash_table_inner"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "param_packetwise_local_hash_partitioning"},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "param_packetwise_global_hash_partitioning"},
        {DPU_FUNC_PACKETWISE_NPHJ_BUILD, "param_packetwise_nphj_build"},
        {DPU_FUNC_PACKETWISE_NPHJ_PROBE, "param_packetwise_nphj_probe"},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, "param_packetwise_phj_build_hash_table"},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, "param_no_partitioned_join_build_hash_table"},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, "param_no_partitioned_join_probe_hash_table"},
	    {DPU_FUNC_GLB_PARTITION_COUNT, "param_glb_partition_count"},
        {DPU_FUNC_GLB_PARTITION_PACKET, "param_glb_partition_packet"},
	    {DPU_FUNC_GLB_CHIPWISE_PARTITION_COUNT, "param_glb_partition_count"},
        {DPU_FUNC_GLB_CHIPWISE_PARTITION_PACKET, "param_glb_partition_packet"},
        {DPU_FUNC_BG_BROADCAST_COUNT, "param_bg_broadcast_count"},
        {DPU_FUNC_BG_BROADCAST_ALIGN, "param_bg_broadcast_align"},
        {DPU_FUNC_NESTED_LOOP_JOIN, "param_nested_loop_join"},
        {DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING, "param_sort_merge_partitioning"},
        {DPU_FUNC_MPSM_JOIN_SORT_PROBE, "param_sort_merge_sort_probe"},
        {DPU_FUNC_FINISH_JOIN, "param_finish_join"},
        // [AddPoint] SEPID-Join Functions
        {DPU_FUNC_SKEW_DETECTION, "param_skew_detection"},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_COUNT, "param_pt_partition_count"},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_PACKET, "param_pt_partition_packet"},
    };

    std::unordered_map<int, int> param_size_map =
    {
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, sizeof(hash_local_partitioning_arg)},
        {DPU_FUNC_GLOBAL_HASH_PARTITIONING, sizeof(hash_global_partitioning_arg)},
        {DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE, sizeof(hash_phj_build_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE, sizeof(hash_phj_probe_arg)},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, sizeof(packetwise_hash_local_partitioning_arg)},
        {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, sizeof(packetwise_hash_global_partitioning_arg)},
        {DPU_FUNC_PACKETWISE_NPHJ_BUILD, sizeof(packetwise_nphj_build_arg)},
        {DPU_FUNC_PACKETWISE_NPHJ_PROBE, sizeof(packetwise_nphj_probe_arg)},
        {DPU_FUNC_PACKETWISE_PHJ_BUILD_HASH_TABLE, sizeof(packetwise_hash_phj_build_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, sizeof(hash_phj_probe_arg)},
        {DPU_FUNC_NPHJ_BUILD_HASH_TABLE, sizeof(no_partitioned_join_build_hash_table_arg)},
        {DPU_FUNC_NPHJ_PROBE_HASH_TABLE, sizeof(no_partitioned_join_build_hash_table_arg)},
        {DPU_FUNC_GLB_PARTITION_COUNT, sizeof(glb_partition_count_arg)},
        {DPU_FUNC_BG_BROADCAST_COUNT, sizeof(bg_broadcast_count_arg)},
        {DPU_FUNC_BG_BROADCAST_ALIGN, sizeof(bg_broadcast_align_arg)},
        {DPU_FUNC_FINISH_JOIN,sizeof(finish_join_arg)},
        {DPU_FUNC_GLB_PARTITION_PACKET, sizeof(glb_partition_packet_arg)},
        {DPU_FUNC_GLB_CHIPWISE_PARTITION_COUNT, sizeof(glb_partition_count_arg)},
        {DPU_FUNC_GLB_CHIPWISE_PARTITION_PACKET, sizeof(glb_partition_packet_arg)},
        {DPU_FUNC_FINISH_JOIN,sizeof(finish_join_arg)},
        {DPU_FUNC_NESTED_LOOP_JOIN, sizeof(nested_loop_join_arg)},
        {DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING, sizeof(sort_merge_partitioning_arg)},
        {DPU_FUNC_MPSM_JOIN_SORT_PROBE, sizeof(sort_merge_sort_probe_arg)},
        // [AddPoint] SEPID-Join Functions
        {DPU_FUNC_SKEW_DETECTION, sizeof(skew_detection_arg)},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_COUNT, sizeof(pt_partition_count_arg)},
        {DPU_FUNC_GLB_PERTUPLE_PARTITION_PACKET, sizeof(pt_partition_packet_arg)},
    };

    std::unordered_map<int, std::string> param_return_var_map = 
    {
	    {DPU_FUNC_LOCAL_HASH_PARTITIONING, "param_local_hash_partitioning_return"},
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, "param_packetwise_local_hash_partitioning_return"},
	    {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, "param_packetwise_global_hash_partitioning_return"},
        {DPU_FUNC_PACKETWISE_NPHJ_PROBE, "param_packetwise_nphj_probe_return"},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, "param_hash_phj_probe_return"},
        {DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING, "param_sort_merge_probe_return"},
        {DPU_FUNC_MPSM_JOIN_SORT_PROBE, "param_sort_merge_probe_return"},
    };
    
    std::unordered_map<int, int> param_return_size_map = 
    {
        {DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING, sizeof(packetwise_hash_local_partitioning_return_arg)},
	    {DPU_FUNC_PACKETWISE_GLOBAL_HASH_PARTITIONING, sizeof(packetwise_hash_global_partitioning_return_arg)},
        {DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE, sizeof(hash_phj_probe_return_arg)},
        {DPU_FUNC_PACKETWISE_NPHJ_PROBE, sizeof(packetwise_nphj_probe_return_arg)},
        {DPU_FUNC_LOCAL_HASH_PARTITIONING, sizeof(hash_local_partitioning_return_arg)},
        {DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING, sizeof(sort_merge_probe_return_arg)},
        {DPU_FUNC_MPSM_JOIN_SORT_PROBE, sizeof(sort_merge_probe_return_arg)},
    };
}
