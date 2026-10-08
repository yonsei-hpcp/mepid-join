#include "join_internals.hpp"

#include "iostream"
#include <time.h>
#include "typeinfo"
#include <cmath>
#include <assert.h>

using namespace pidjoin;

void JoinOperator::Execute_COMPOUND_FUNC_RNS_JOIN(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node,
    std::vector<TimeStamp_t> *perThreadTimeStamps)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto key = (*op_node)["key"];
    auto type = (*op_node)["join_type"];
    auto debug_key = (*op_node)["debug_key"];
    auto packet_size = (*op_node)["packet_size"];

    int tuple_size_ht = sizeof(int64_t);
    int tuple_size_probe = sizeof(int64_t);
    
    if (this->join_instance->join_algorithm != "")
    {
        join_alg = this->join_instance->join_algorithm;
    }

    /**
     * Strings Setup
     **/

    // HOST_FUNC_ROTATE_AND_STREAM
    std::string R_rnc_join_key("R_rnc_join_key");
    // HOST_FUNC_ROTATE_AND_STREAM
    std::string S_rnc_join_key("S_rnc_join_key");

    // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
    std::string R_packetwise_local_partitioned_join_key("R_packetwise_local_partitioned_join_key");
    std::string R_packetwise_local_partitioned_partition_info("R_packetwise_local_partitioned_partition_info");
    std::string R_packetwise_local_partition_num("R_packetwise_local_partition_num");

    // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
    std::string S_packetwise_local_partitioned_join_key("S_packetwise_local_partitioned_join_key");
    std::string S_packetwise_local_partitioned_partition_info("S_packetwise_local_partitioned_partition_info");

    // DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE
    std::string hash_table("hash_table");
    std::string join_result(outputs[0].asCString());

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    Json::Value HOST_FUNC_ROTATE_AND_STREAM_var_r;

    HOST_FUNC_ROTATE_AND_STREAM_var_r["input"] = inputs[0].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_r["output"] = R_rnc_join_key;
    HOST_FUNC_ROTATE_AND_STREAM_var_r["histogram"] = inputs[1].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_r["packet_size"] = packet_size;
    HOST_FUNC_ROTATE_AND_STREAM_var_r["rankgroup_aware"] = 1;

    // Execute RNS of Record R
    perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "HOST_FUNC_ROTATE_AND_STREAM"});
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
    Execute_HOST_FUNC_ROTATE_AND_STREAM(idp_handler, rank_id, params, &HOST_FUNC_ROTATE_AND_STREAM_var_r);
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    Json::Value HOST_FUNC_ROTATE_AND_STREAM_var_s;

    HOST_FUNC_ROTATE_AND_STREAM_var_s["input"] = inputs[2].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_s["output"] = S_rnc_join_key;
    HOST_FUNC_ROTATE_AND_STREAM_var_s["histogram"] = inputs[3].asString();
    HOST_FUNC_ROTATE_AND_STREAM_var_s["packet_size"] = packet_size;
    HOST_FUNC_ROTATE_AND_STREAM_var_s["rankgroup_aware"] = 0;
    
    // Execute RNS of Record S
    perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "HOST_FUNC_ROTATE_AND_STREAM"});
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
    Execute_HOST_FUNC_ROTATE_AND_STREAM(idp_handler, rank_id, params, &HOST_FUNC_ROTATE_AND_STREAM_var_s);
    clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

    // /////////////////////////////////////////////////////////////////////////////////////////////////////////

    // sync
    Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

    if (join_alg == "phj")
    {
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs.append(R_rnc_join_key);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_partition_info);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(true);

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["tuple_size"] = tuple_size_ht;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["packet_size"] = packet_size;

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs.append(S_rnc_join_key);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_partition_info);
        
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(false);

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["tuple_size"] = tuple_size_probe;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["packet_size"] = packet_size;

        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_partition_info);

        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs.append(hash_table);
        
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;

        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(hash_table);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
        
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs.append(join_result);
        
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["join_type"] = type;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Local_partitioning_R"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // Invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Local_partitioning_S"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // Invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        // Sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Build"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_BUILD_HASH_TABLE(idp_handler, rank_id, params, &DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
        
        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_partition_info);

            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Probe"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_PROBE_HASH_TABLE_INNER(idp_handler, rank_id, params, &DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(hash_table);

            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }
    }
    else if (join_alg == "nl")
    {
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var;
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(R_rnc_join_key);
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(S_rnc_join_key);
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_outputs.append(outputs[0]);

        DPU_FUNC_NESTED_LOOP_JOIN_var["inputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["outputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["packet_size"] = packet_size;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "JOIN"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_NESTED_LOOP_JOIN(idp_handler, rank_id, params, &DPU_FUNC_NESTED_LOOP_JOIN_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }
    else if (join_alg == "smj")
    {
        auto algorithm = (*op_node)["algorithm"];
        
        // Default for Merge Sort
        algorithm[0] = "merge";
        algorithm[1] = 0;

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R */
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs;

        // input: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs.append(R_rnc_join_key);

        // outputs: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R
        std::string R_sorted_join_key("R_sorted_join_key");
        std::string R_local_histogram("R_local_histogram");
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs.append(R_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs.append(R_local_histogram);

        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["inputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["outputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_LOCAL_PARTITIONING
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_LOCAL_PARTITIONING_R"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_LOCAL_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // Invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S */
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs;

        // input: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs.append(S_rnc_join_key);

        // outputs: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S
        std::string S_sorted_join_key("S_sorted_join_key");
        std::string S_local_histogram("S_local_histogram");
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs.append(S_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs.append(S_local_histogram);

        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["inputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["outputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_LOCAL_PARTITIONING
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_LOCAL_PARTITIONING_S"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_LOCAL_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // Invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        /* DPU_FUNC_MPSM_JOIN_SORT_PROBE */

        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs;

        // inputs: DPU_FUNC_MPSM_JOIN_SORT_PROBE
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(R_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(S_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(R_local_histogram);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(S_local_histogram);

        // outputs: DPU_FUNC_MPSM_JOIN_SORT_PROBE
        std::string join_result(outputs[0].asString());
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs.append(join_result);

        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var["inputs"] = DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs;
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var["outputs"] = DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs;
        
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_SORT_MERGE"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_SORT_PROBE(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_SORT_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // Invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_sorted_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_sorted_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_local_histogram);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_local_histogram);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);
    }
    else
    {
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var;
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs.append(R_rnc_join_key.c_str());
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs.append(hash_table);
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var["inputs"] = DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var["outputs"] = DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs;

        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var;
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs.append(S_rnc_join_key.c_str());
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs.append(hash_table);
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs.append(outputs[0].asString());
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["inputs"] = DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["outputs"] = DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["join_type"] = type;

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Build"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_NPHJ_BUILD(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_NPHJ_BUILD_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Probe"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_NPHJ_PROBE(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_NPHJ_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
        Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(hash_table);
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
        Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
    }

    // Json::Value DPU_FUNC_FINISH_JOIN_var;
    // DPU_FUNC_FINISH_JOIN_var["join_output"] = outputs[0];
    // perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "FINISH_J"});
    // clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
    // Execute_FINISH_JOIN(idp_handler, rank_id, params, &DPU_FUNC_FINISH_JOIN_var);
    // clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
}

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8) +(uint32_t)(((const uint8_t *)(d))[0]) )

static uint32_t glb_partition_hash(uint32_t key)
{
  int len = 4;
  uint8_t* data_;
  uint8_t data[4];
  data_ = data;
  *(uint32_t*)(data) = key;
  uint32_t hash = 4, tmp;
	int rem;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) 
    {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data_  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
			hash ^= hash << 16;
			hash ^= data[sizeof (uint16_t)] << 18;
			hash += hash >> 11;
			break;
        case 2: hash += get16bits (data);
			hash ^= hash << 11;
			hash += hash >> 17;
			break;
        case 1: hash += *data;
			hash ^= hash << 10;
			hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;

}



void JoinOperator::Execute_COMPOUND_FUNC_IDP_JOIN(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node,
    std::vector<TimeStamp_t> *perThreadTimeStamps)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto key = (*op_node)["key"];
    auto type = (*op_node)["join_type"];
    auto debug_key = (*op_node)["debug_key"];
    auto packet_size = (*op_node)["packet_size"];

    int tuple_size_ht = sizeof(int64_t);
    int tuple_size_probe = sizeof(int64_t);
    std::string hash_table("hash_table");

    ///////////////////////////////////////////////////////////////////////////////

    if (idp_handler->join_design.join_alg == JOIN_DESIGN_JOIN_ALGORITHM_PHJ)
    {
        // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
        std::string R_packetwise_local_partitioned_join_key("R_packetwise_local_partitioned_join_key");
        std::string R_packetwise_local_partitioned_partition_info("R_packetwise_local_partitioned_partition_info");
        std::string R_packetwise_local_partition_num("R_packetwise_local_partition_num");

        // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING
        std::string S_packetwise_local_partitioned_join_key("S_packetwise_local_partitioned_join_key");
        std::string S_packetwise_local_partitioned_partition_info("S_packetwise_local_partitioned_partition_info");

        // DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE
        
        // DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE
        std::string join_result(outputs[0].asCString());

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs.append(inputs[0].asCString());

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs.append(R_packetwise_local_partitioned_partition_info);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup.append(true);

        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s;
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs.append(inputs[1].asCString());
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        // DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_Tuple_ID);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs.append(S_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(R_packetwise_local_partition_num);
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup.append(false);

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["tuple_size"] = tuple_size_ht;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r["packet_size"] = packet_size;

        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["inputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_inputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["outputs"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_outputs;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["setup"] = DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s_setup;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["tuple_size"] = tuple_size_probe;
        DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s["packet_size"] = packet_size;

        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs.append(R_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs.append(hash_table);
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var_outputs;

        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var;
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(hash_table);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_join_key);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
        Json::Value DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs.append(join_result);
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["inputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_inputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["outputs"] = DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var_outputs;
        DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var["join_type"] = type;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Local_partitioning_R"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        
        Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_r);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
        
        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(inputs[0].asCString());
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Local_partitioning_S"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);

        if (packet_size == 0)
            Execute_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s);
        else
            Execute_PACKETWISE_LOCAL_HASH_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_LOCAL_HASH_PARTITIONING_var_s);
        
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(inputs[1].asCString());
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Build"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_BUILD_HASH_TABLE(idp_handler, rank_id, params, &DPU_FUNC_PHJ_BUILD_HASH_TABLE_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(R_packetwise_local_partitioned_partition_info);

            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Probe"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PHJ_PROBE_HASH_TABLE_INNER(idp_handler, rank_id, params, &DPU_FUNC_PHJ_PROBE_HASH_TABLE_INNER_LINEAR_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(S_packetwise_local_partitioned_partition_info);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs.append(hash_table);

            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"] = DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var_inputs;
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        Json::Value DPU_FUNC_FINISH_JOIN_var;
        DPU_FUNC_FINISH_JOIN_var["join_output"] = outputs[0];
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "FINISH_J"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_FINISH_JOIN(idp_handler, rank_id, params, &DPU_FUNC_FINISH_JOIN_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }
    else if (idp_handler->join_design.join_alg == JOIN_DESIGN_JOIN_ALGORITHM_NPHJ)
    {
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var;
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs.append(inputs[0].asString());
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs.append(hash_table);
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var["inputs"] = DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_BUILD_var["outputs"] = DPU_FUNC_PACKETWISE_NPHJ_BUILD_var_outputs;

        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var;
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs.append(inputs[1].asString());
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs.append(hash_table);
        Json::Value DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs.append(outputs[0].asString());
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["inputs"] = DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_inputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["outputs"] = DPU_FUNC_PACKETWISE_NPHJ_PROBE_var_outputs;
        DPU_FUNC_PACKETWISE_NPHJ_PROBE_var["join_type"] = type;

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Build"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_NPHJ_BUILD(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_NPHJ_BUILD_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "Probe"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_PACKETWISE_NPHJ_PROBE(idp_handler, rank_id, params, &DPU_FUNC_PACKETWISE_NPHJ_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        Json::Value DPU_FUNC_FINISH_JOIN_var;
        DPU_FUNC_FINISH_JOIN_var["join_output"] = outputs[0];
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "FINISH_J"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_FINISH_JOIN(idp_handler, rank_id, params, &DPU_FUNC_FINISH_JOIN_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }
    else if (idp_handler->join_design.join_alg == JOIN_DESIGN_JOIN_ALGORITHM_SMJ)
    {
        auto algorithm = (*op_node)["algorithm"];
        // To be fixed
        algorithm[0] = "merge";
        algorithm[1] = 0;

        // DPU_FUNC_MPSM_JOIN_SORT_R
        std::string R_rnc_join_key("Replicated_R_Tuples");
        // DPU_FUNC_MPSM_JOIN_SORT_S
        std::string S_rnc_join_key("S_rnc_join_key");

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R */

        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs;

        // input: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs.append(R_rnc_join_key);

        // outputs: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R
        std::string R_sorted_join_key("R_sorted_join_key");
        std::string R_local_histogram("R_local_histogram");
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs.append(R_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs.append(R_local_histogram);

        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["inputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_inputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["outputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var_outputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_LOCAL_PARTITIONING
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_LOCAL_PARTITIONING_R"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_LOCAL_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_R */

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S */

        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs;

        // input: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs.append(S_rnc_join_key);

        // outputs: DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S
        std::string S_sorted_join_key("S_sorted_join_key");
        std::string S_local_histogram("S_local_histogram");
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs.append(S_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs.append(S_local_histogram);

        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["inputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_inputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["outputs"] = DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var_outputs;
        DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var["packet_size"] = packet_size;

        // Execute_MPSM_JOIN_LOCAL_PARTITIONING
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_LOCAL_PARTITIONING_S"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_LOCAL_PARTITIONING(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_rnc_join_key);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        /* DPU_FUNC_MPSM_JOIN_LOCAL_PARTITIONING_S */

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);

        /* DPU_FUNC_MPSM_JOIN_SORT_PROBE */

        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs;
        Json::Value DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs;

        // inputs: DPU_FUNC_MPSM_JOIN_SORT_PROBE
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(R_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(S_sorted_join_key);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(R_local_histogram);
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs.append(S_local_histogram);

        // outputs: DPU_FUNC_MPSM_JOIN_SORT_PROBE
        std::string join_result(outputs[0].asString());
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs.append(join_result);

        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var["inputs"] = DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_inputs;
        DPU_FUNC_MPSM_JOIN_SORT_PROBE_var["outputs"] = DPU_FUNC_MPSM_JOIN_SORT_PROBE_var_outputs;
        
        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "MPSM_JOIN_SORT_MERGE"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_MPSM_JOIN_SORT_PROBE(idp_handler, rank_id, params, &DPU_FUNC_MPSM_JOIN_SORT_PROBE_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);

        // invalidate Nodes
        {
            Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_sorted_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_sorted_join_key);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(R_local_histogram);
            DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(S_local_histogram);
            Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
        }

        /* DPU_FUNC_MPSM_JOIN_SORT_PROBE */

        // sync
        Execute_CONTROL_FUNC_SYNC_THREADS(idp_handler, rank_id, params, op_node);
    }
    else if (idp_handler->join_design.join_alg == JOIN_DESIGN_JOIN_ALGORITHM_NL)
    {
        // DPU_FUNC_NESTED_LOOP_JOIN
        std::string R_rnc_join_key("R_rnc_join_key");
        // DPU_FUNC_NESTED_LOOP_JOIN
        std::string S_rnc_join_key("S_rnc_join_key");

        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var;
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(R_rnc_join_key);
        DPU_FUNC_NESTED_LOOP_JOIN_var_inputs.append(S_rnc_join_key);
        Json::Value DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var_outputs.append(outputs[0]);

        DPU_FUNC_NESTED_LOOP_JOIN_var["inputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_inputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["outputs"] = DPU_FUNC_NESTED_LOOP_JOIN_var_outputs;
        DPU_FUNC_NESTED_LOOP_JOIN_var["packet_size"] = packet_size;

        perThreadTimeStamps->push_back(TimeStamp_t{rank_id, "JOIN"});
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().start_time);
        Execute_NESTED_LOOP_JOIN(idp_handler, rank_id, params, &DPU_FUNC_NESTED_LOOP_JOIN_var);
        clock_gettime(CLOCK_MONOTONIC, &perThreadTimeStamps->back().end_time);
    }
}

void JoinOperator::Execute_COMPOUND_FUNC_GLB_PARTITION(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    std::string Tuple_ID = inputs[0].asCString();
    std::string Join_Key = inputs[1].asCString();

    // Execute_GLB_PARTITION_COUNT
    std::string gp_partition_info("gp_partition_info");
    gp_partition_info += inputs[1].asCString();
    std::string gp_histogram("gp_histogram");
    gp_histogram += inputs[1].asCString();

    // Execute_GLB_PARTITION_PACKET
    std::string packetwise_global_partitioned_tid("packetwise_partitioned_tid");

    Json::Value DPU_FUNC_GLB_PARTITION_COUNT_var;
    DPU_FUNC_GLB_PARTITION_COUNT_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_partition_info);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_COUNT_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_COUNT_var["partition_type"] = (*op_node)["partition_type"];

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["inputs"].append(gp_histogram);
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["outputs"].append(outputs[1].asCString());
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["partition_type"] = partition_type;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["packet_size"] = packet_size;

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_GLB_PARTITION_PACKET_var;
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Tuple_ID);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(outputs[1].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_PACKET_var["outputs"].append(outputs[0].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_PACKET_var["partition_type"] = (*op_node)["partition_type"];

    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_partition_info);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_histogram);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);

    // Count the element to go each partitions(DPUs)
    Execute_GLB_PARTITION_COUNT(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_COUNT_var);
    // Count and calculate appropriate packet size and number
    Execute_MT_SAFE_HOST_FUNC_CALCULATE_PAGE_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var);
    Execute_GLB_PARTITION_PACKET(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_PACKET_var);
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
}

void JoinOperator::Execute_COMPOUND_FUNC_BG_BROADCAST(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];

    std::string Tuple_ID = inputs[0].asCString();
    std::string Join_Key = inputs[1].asCString();

    // Execute_BG_BROADCAST_COUNT
    std::string gp_partition_info("bg_broadcast_gp_partition_info");
    std::string gp_histogram("bg_broadcast_gp_histogram");

    Json::Value DPU_FUNC_BG_BROADCAST_COUNT_var;
    DPU_FUNC_BG_BROADCAST_COUNT_var["inputs"].append(Join_Key);
    DPU_FUNC_BG_BROADCAST_COUNT_var["outputs"].append(gp_partition_info);
    DPU_FUNC_BG_BROADCAST_COUNT_var["outputs"].append(gp_histogram);
    
    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_BANKGROUP_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_BANKGROUP_HISTOGRAM_var["inputs"].append(gp_histogram);
    HOST_FUNC_CALCULATE_BANKGROUP_HISTOGRAM_var["outputs"].append(outputs[1].asCString());

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_BG_BROADCAST_ALIGN_var;
    DPU_FUNC_BG_BROADCAST_ALIGN_var["inputs"].append(Tuple_ID);
    DPU_FUNC_BG_BROADCAST_ALIGN_var["inputs"].append(Join_Key);
    DPU_FUNC_BG_BROADCAST_ALIGN_var["inputs"].append(gp_histogram);
    DPU_FUNC_BG_BROADCAST_ALIGN_var["inputs"].append(outputs[1].asCString());
    DPU_FUNC_BG_BROADCAST_ALIGN_var["outputs"].append(outputs[0].asCString());

    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_partition_info);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_histogram);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);

    // Count the element to go each partitions(DPUs)
    Execute_BG_BROADCAST_COUNT(idp_handler, rank_id, params, &DPU_FUNC_BG_BROADCAST_COUNT_var);
    // Count and calculate appropriate packet size and number
    Execute_MT_SAFE_HOST_FUNC_CALCULATE_BANKGROUP_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_BANKGROUP_HISTOGRAM_var);
    Execute_BG_BROADCAST_ALIGN(idp_handler, rank_id, params, &DPU_FUNC_BG_BROADCAST_ALIGN_var);
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
}

void JoinOperator::Execute_COMPOUND_FUNC_GLB_CHIPWISE_PARTITION(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    if (partition_type.asInt() == 0)
    {
        printf("%sError: Set partition_type\n", KRED);
        exit(-1);
    }

    std::string Tuple_ID = inputs[0].asCString();
    std::string Join_Key = inputs[1].asCString();

    // Execute_GLB_PARTITION_COUNT
    std::string gp_partition_info("gp_chipwise_partition_info");
    std::string gp_histogram("gp_chipwise_histogram");

    // Execute_GLB_PARTITION_PACKET
    std::string packetwise_global_partitioned_tid("packetwise_partitioned_tid");

    Json::Value DPU_FUNC_GLB_PARTITION_COUNT_var;
    DPU_FUNC_GLB_PARTITION_COUNT_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_partition_info);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_COUNT_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_COUNT_var["partition_type"] = (*op_node)["partition_type"];

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["inputs"].append(gp_histogram);
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["outputs"].append(outputs[1].asCString());
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["partition_type"] = partition_type;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["packet_size"] = packet_size;

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_GLB_PARTITION_PACKET_var;
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Tuple_ID);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(outputs[1].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_PACKET_var["outputs"].append(outputs[0].asCString());
    DPU_FUNC_GLB_PARTITION_PACKET_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_PACKET_var["partition_type"] = (*op_node)["partition_type"];

    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_partition_info);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_histogram);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);

    // Count the element to go each partitions(DPUs)
    Execute_GLB_CHIPWISE_PARTITION_COUNT(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_COUNT_var);
    // Count and calculate appropriate packet size and number
    Execute_MT_SAFE_HOST_FUNC_CHIPWISE_CALCULATE_PAGE_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var);
    Execute_GLB_CHIPWISE_PARTITION_PACKET(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_PACKET_var);
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
}

// [AddPoint] Function for per-tuple partition
void JoinOperator::Execute_COMPOUND_FUNC_GLB_PERTUPLE_PARTITION(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto packet_size = this->join_instance->packet_size;
    auto partition_type = (*op_node)["partition_type"];

    std::string Tuple_ID = inputs[0].asCString();                                       // "unused"
    std::string Join_Key = inputs[1].asCString();                                       // "right"

    // Execute_GLB_PARTITION_COUNT
    std::string rr_histogram("shared_rr_histogram");
    std::string gp_histogram("gp_pertuple_histogram");
    std::string gp_partition_info("gp_pertuple_partition_info");

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_GLB_PARTITION_COUNT_var;
    DPU_FUNC_GLB_PARTITION_COUNT_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_COUNT_var["inputs"].append(rr_histogram);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_partition_info);
    DPU_FUNC_GLB_PARTITION_COUNT_var["outputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_COUNT_var["packet_size"] = packet_size;
    DPU_FUNC_GLB_PARTITION_COUNT_var["partition_type"] = (*op_node)["partition_type"];

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["inputs"].append(gp_histogram);
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["outputs"].append(outputs[1].asCString());   // "Probe_hist"
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["partition_type"] = partition_type;
    HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var["packet_size"] = packet_size;

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_GLB_PARTITION_PACKET_var;
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(Join_Key);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(rr_histogram);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(gp_histogram);
    DPU_FUNC_GLB_PARTITION_PACKET_var["inputs"].append(outputs[1].asCString());         // "Probe_hist"
    DPU_FUNC_GLB_PARTITION_PACKET_var["outputs"].append(outputs[0].asCString());        // "Probe_arr"
    DPU_FUNC_GLB_PARTITION_PACKET_var["partition_type"] = (*op_node)["partition_type"];

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_partition_info);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(gp_histogram);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);

    // For per-tuple partition, rr_histogram is also needed in probe phase
    if (partition_type.asInt() == 1)
    {
        DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(rr_histogram);
    }

    // Count the element to go each partitions (DPUs)
    Execute_GLB_PERTUPLE_PARTITION_COUNT(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_COUNT_var);
    // Count and calculate appropriate packet size and number
    Execute_MT_SAFE_HOST_FUNC_CALCULATE_PAGE_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_PAGE_HISTOGRAM_var);
    // Build packets for each partitions (DPUs)
    Execute_GLB_PERTUPLE_PARTITION_PACKET(idp_handler, rank_id, params, &DPU_FUNC_GLB_PARTITION_PACKET_var);
    // Execute_HOST_FUNC_INVALIDATE_STACK
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
}

// [AddPoint] Function for skew detection
void JoinOperator::Execute_COMPOUND_FUNC_SKEW_DETECTION(
    IDPHandler *idp_handler,
    int rank_id,
    DPUKernelParams_t *params,
    Json::Value *op_node)
{
    auto inputs = (*op_node)["inputs"];
    auto outputs = (*op_node)["outputs"];
    auto sample_rate = idp_handler->join_design.sample_rate;

    auto sample_size = idp_handler->join_design.input_size * idp_handler->join_design.rs_ratio * sample_rate;

    std::string Join_Key = inputs[0].asCString();
    std::string rr_histogram("rr_pertuple_histogram");

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_SKEW_DETECTION_var;
    DPU_FUNC_SKEW_DETECTION_var["inputs"].append(Join_Key);
    DPU_FUNC_SKEW_DETECTION_var["outputs"].append(rr_histogram);
    DPU_FUNC_SKEW_DETECTION_var["sample_rate"] = sample_rate;

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_CALCULATE_RR_HISTOGRAM_var;
    HOST_FUNC_CALCULATE_RR_HISTOGRAM_var["inputs"].append(rr_histogram);
    HOST_FUNC_CALCULATE_RR_HISTOGRAM_var["outputs"].append(outputs[0].asCString());
    HOST_FUNC_CALCULATE_RR_HISTOGRAM_var["sample_size"] = sample_size;

    //////////////////////////////////////////////
    Json::Value DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var;
    // DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(Join_Key);
    DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var["inputs"].append(rr_histogram);

    //////////////////////////////////////////////
    Json::Value HOST_FUNC_BROADCAST_DATA_var;
    HOST_FUNC_BROADCAST_DATA_var["buffer_name"] = outputs[0].asCString();

    // Detect skew of each partition and store the result in rr_histogram (DPUs)
    Execute_SKEW_DETECTION(idp_handler, rank_id, params, &DPU_FUNC_SKEW_DETECTION_var);
    // Calcuate the accumulated RR histogram for skew-aware partitioning (Host)
    Execute_MT_SAFE_HOST_FUNC_CALCULATE_RR_HISTOGRAM(idp_handler, rank_id, params, &HOST_FUNC_CALCULATE_RR_HISTOGRAM_var);
    // Invalidate the stack not needed anymore
    Execute_HOST_FUNC_INVALIDATE_STACKNODE(idp_handler, rank_id, params, &DPU_FUNC_HOST_FUNC_INVALIDATE_STACKNODE_var);
    // Broadcast the accumulated RR histogram to all IDPs
    Execute_HOST_FUNC_BROADCAST_DATA(idp_handler, rank_id, params, &HOST_FUNC_BROADCAST_DATA_var);
}