#ifndef _HASH_H_
#define _HASH_H_

#include <stdio.h>
#include <defs.h>

uint32_t join_hash(uint32_t key);
uint32_t join_hash2(uint32_t key);
extern uint32_t crctab[256];
uint32_t glb_partition_hash(uint32_t key);
uint32_t double_hash1(uint32_t key);
uint32_t double_hash2(uint32_t key);
uint32_t local_partition_hash(uint32_t key);
uint32_t bit_reverse_10(uint32_t x);
uint32_t mix32(uint32_t x);
uint32_t mix32_nomul(uint32_t x);
uint32_t fastRange32(uint32_t val, uint32_t range);
uint32_t floor_pow2_u32(uint32_t val);
uint32_t ceil_pow2_u32(uint32_t val);

#endif