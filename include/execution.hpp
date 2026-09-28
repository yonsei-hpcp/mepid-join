#include <algorithm>
#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "json/json.h"

//////////////////////////////////////////
// PID- / SPID-Join
//////////////////////////////////////////

#define join_json "{                                                            \
    \"query_name\": \"\",                                                       \
    \"tables\": [],                                                             \
    \"query_tree\": [                                                           \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"left\"                                           \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"right\"                                          \
        },                                                                      \
        {                                                                       \
            \"operator\": \"CONTROL_FUNC_SYNC_THREADS\"                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_GLB_PARTITION\",                      \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"left\"                                                        \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Build_arr\",                                                  \
                \"Build_hist\"                                                  \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"partition_type\": 0                                               \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_GLB_PARTITION\",                      \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"right\"                                                       \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Probe_arr\",                                                  \
                \"Probe_hist\"                                                  \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"partition_type\": 1                                               \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_RNS_JOIN\",                           \
            \"inputs\": [                                                       \
                \"Build_arr\",                                                  \
                \"Build_hist\",                                                 \
                \"Probe_arr\",                                                  \
                \"Probe_hist\"                                                  \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"join_result\"                                                 \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"join_type\": \"\",                                                \
            \"comment\": \"first join\"                                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_RECV_DATA_OPT\",                          \
            \"node_name\": \"join_result\"                                      \
        }                                                                       \
    ],                                                                          \
    \"query_string\": []                                                        \
}";

#define bg_join_json "{                                                         \
    \"query_name\": \"\",                                                       \
    \"tables\": [],                                                             \
    \"query_tree\": [                                                           \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"left\"                                           \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"right\"                                          \
        },                                                                      \
        {                                                                       \
            \"operator\": \"CONTROL_FUNC_SYNC_THREADS\"                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_BG_BROADCAST\",                       \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"left\"                                                        \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Build_arr\",                                                  \
                \"build_side_r2r_xfer_cnt\"                                     \
            ]                                                                   \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_CHIPWISE_BROADCAST\",                     \
            \"inputs\": [                                                       \
                \"Build_arr\",                                                  \
                \"build_side_r2r_xfer_cnt\"                                     \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Replicated_R_Tuples\"                                         \
            ]                                                                   \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_GLB_CHIPWISE_PARTITION\",             \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"right\"                                                       \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Probe_arr\",                                                  \
                \"Probe_hist\"                                                  \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"partition_type\": 1                                               \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_ROTATE_AND_STREAM\",                      \
            \"input\": \"Probe_arr\",                                           \
            \"output\": \"S_rnc_join_key\",                                     \
            \"histogram\": \"Probe_hist\",                                      \
            \"packet_size\": 8,                                                 \
            \"comment\": \"Hash table probe\",                                  \
            \"rankgroup_aware\": 0                                              \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_IDP_JOIN\",                           \
            \"inputs\": [                                                       \
                \"Replicated_R_Tuples\",                                        \
                \"S_rnc_join_key\"                                              \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"join_result\"                                                 \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"join_type\": \"\",                                                \
            \"comment\": \"first join\"                                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_RECV_DATA_OPT\",                          \
            \"node_name\": \"join_result\"                                      \
        }                                                                       \
    ],                                                                          \
    \"query_string\": []                                                        \
}";

//////////////////////////////////////////
// MEPID-Join
//////////////////////////////////////////

#define skew_join_json "{                                                       \
    \"query_name\": \"\",                                                       \
    \"tables\": [],                                                             \
    \"query_tree\": [                                                           \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"left\"                                           \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_SEND_DATA_OPT\",                          \
            \"buffer_name\": \"right\"                                          \
        },                                                                      \
        {                                                                       \
            \"operator\": \"CONTROL_FUNC_SYNC_THREADS\"                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_SKEW_DETECTION\",                     \
            \"inputs\": [                                                       \
                \"right\"                                                       \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"shared_rr_histogram\"                                         \
            ]                                                                   \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_GLB_PERTUPLE_PARTITION\",             \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"left\"                                                        \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Build_arr\",                                                  \
                \"Build_hist\"                                                  \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"partition_type\": 0                                               \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_GLB_PERTUPLE_PARTITION\",             \
            \"inputs\": [                                                       \
                \"unused\",                                                     \
                \"right\"                                                       \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"Probe_arr\",                                                  \
                \"Probe_hist\"                                                  \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"partition_type\": 1                                               \
        },                                                                      \
        {                                                                       \
            \"operator\": \"COMPOUND_FUNC_RNS_JOIN\",                           \
            \"inputs\": [                                                       \
                \"Build_arr\",                                                  \
                \"Build_hist\",                                                 \
                \"Probe_arr\",                                                  \
                \"Probe_hist\"                                                  \
            ],                                                                  \
            \"outputs\": [                                                      \
                \"join_result\"                                                 \
            ],                                                                  \
            \"packet_size\": 8,                                                 \
            \"join_type\": \"\",                                                \
            \"comment\": \"first join\"                                         \
        },                                                                      \
        {                                                                       \
            \"operator\": \"HOST_FUNC_RECV_DATA_OPT\",                          \
            \"node_name\": \"join_result\"                                      \
        }                                                                       \
    ],                                                                          \
    \"query_string\": []                                                        \
}";
        