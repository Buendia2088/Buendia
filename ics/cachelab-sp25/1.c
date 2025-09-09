#include "cachelab.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

// 全局计数器
int l1d_hits = 0;
int l1d_misses = 0;
int l1d_evictions = 0;
int l1i_hits = 0;
int l1i_misses = 0;
int l1i_evictions = 0;
int l2_hits = 0;
int l2_misses = 0;
int l2_evictions = 0;
int l3_hits = 0;
int l3_misses = 0;
int l3_evictions = 0;

uint64_t timeCount = 0;

// 辅助函数声明
void divide(uint64_t addr, int level, uint64_t *tag, uint64_t *set_index);
int load_to_cache(uint64_t addr, int level, char op_type, bool is_write);
void invalidate_upper_caches(uint64_t addr, int start_level);

/*----- 地址划分函数 -----*/
void divide(uint64_t addr, int level, uint64_t *tag, uint64_t *set_index) {
    int offset_bits, set_bits;
    switch(level) {
        case 1: // L1D/I: 8B line, 4 sets
            offset_bits = 3; // log2(8)
            set_bits = 2;    // log2(4)
            break;
        case 2: // L2: 8B line, 8 sets
            offset_bits = 3;
            set_bits = 3;   // log2(8)
            break;
        case 3: // L3: 16B line, 16 sets
            offset_bits = 4; // log2(16)
            set_bits = 4;   // log2(16)
            break;
        default:
            *tag = 0; *set_index = 0; return;
    }
    *set_index = (addr >> offset_bits) & ((1ULL << set_bits) - 1);
    *tag = addr >> (offset_bits + set_bits);
}

/*----- 缓存加载与替换 -----*/
int load_to_cache(uint64_t addr, int level, char op_type, bool is_write) {
    uint64_t tag, set_index;
    divide(addr, level, &tag, &set_index);
    //CacheLine (*cache)[16][8]; // 通用指针，实际维度根据level调整
    int set_num, line_num, evictions = 0;
    int replace_idx = -1;
    uint64_t min_used = UINT64_MAX;
    switch(level) {
        case 1:
            set_num = L1_SET_NUM;
            line_num = L1_LINE_NUM;
            if(op_type = 'I'){
                // 查找可替换行（LRU）
                replace_idx = -1;
                min_used = UINT64_MAX;
                for(int i = 0; i < line_num; i++) {
                    if (!l3ucache[set_index][i].valid) {
                        replace_idx = i;
                        break;
                    }
                    if (l3ucache[set_index][i].latest_used < min_used) {
                        min_used = l3ucache[set_index][i].latest_used;
                        replace_idx = i;
                    }
                }

                // 处理替换
                if (l3ucache[set_index][replace_idx].valid) {
                    evictions = 1;
                    // 更新统计
                    switch(level) {
                        case 1: (op_type == 'I') ? l1i_evictions++ : l1d_evictions++; break;
                        case 2: l2_evictions++; break;
                        case 3: l3_evictions++; break;
                    }
                    // 写回dirty块
                    if (l3ucache[set_index][replace_idx].dirty && level < 3) {
                        // 递归写回下层（简化处理，实际需计算地址）
                        uint64_t evicted_addr = (l3ucache[set_index][replace_idx].tag << (int)(log2(set_num) + log2((level==1)?8:(level==2)?8:16))) | (set_index << (int)log2((level==1)?8:(level==2)?8:16));
                        load_to_cache(evicted_addr, level+1, 'S', true);
                    }
                    // 包含性策略：如果下层被替换，上层需无效化
                    if (level == 3) invalidate_upper_caches(addr, 2);
                    else if (level == 2) invalidate_upper_caches(addr, 1);
                }

                // 加载新块
                l3ucache[set_index][replace_idx].valid = 1;
                l3ucache[set_index][replace_idx].tag = tag;
                l3ucache[set_index][replace_idx].latest_used = ++timeCount;
                l3ucache[set_index][replace_idx].dirty = is_write;

                return evictions;
            }
            else{
                // 查找可替换行（LRU）
                replace_idx = -1;
                min_used = UINT64_MAX;
                for(int i = 0; i < line_num; i++) {
                    if (!l1dcache[set_index][i].valid) {
                        replace_idx = i;
                        break;
                    }
                    if (l1dcache[set_index][i].latest_used < min_used) {
                        min_used = l1dcache[set_index][i].latest_used;
                        replace_idx = i;
                    }
                }

                // 处理替换
                if (l1dcache[set_index][replace_idx].valid) {
                    evictions = 1;
                    // 更新统计
                    switch(level) {
                        case 1: (op_type == 'I') ? l1i_evictions++ : l1d_evictions++; break;
                        case 2: l2_evictions++; break;
                        case 3: l3_evictions++; break;
                    }
                    // 写回dirty块
                    if (l1dcache[set_index][replace_idx].dirty && level < 3) {
                        // 递归写回下层（简化处理，实际需计算地址）
                        uint64_t evicted_addr = (l1dcache[set_index][replace_idx].tag << (int)(log2(set_num) + log2((level==1)?8:(level==2)?8:16))) | (set_index << (int)log2((level==1)?8:(level==2)?8:16));
                        load_to_cache(evicted_addr, level+1, 'S', true);
                    }
                    // 包含性策略：如果下层被替换，上层需无效化
                    if (level == 3) invalidate_upper_caches(addr, 2);
                    else if (level == 2) invalidate_upper_caches(addr, 1);
                }

                // 加载新块
                l1dcache[set_index][replace_idx].valid = 1;
                l1dcache[set_index][replace_idx].tag = tag;
                l1dcache[set_index][replace_idx].latest_used = ++timeCount;
                l1dcache[set_index][replace_idx].dirty = is_write;

                return evictions;
            }
            break;
        case 2:
            set_num = L2_SET_NUM;
            line_num = L2_LINE_NUM;
            // 查找可替换行（LRU）
            replace_idx = -1;
            min_used = UINT64_MAX;
            for(int i = 0; i < line_num; i++) {
                if (!l2ucache[set_index][i].valid) {
                    replace_idx = i;
                    break;
                }
                if (l2ucache[set_index][i].latest_used < min_used) {
                    min_used = l2ucache[set_index][i].latest_used;
                    replace_idx = i;
                }
            }

            // 处理替换
            if (l2ucache[set_index][replace_idx].valid) {
                evictions = 1;
                // 更新统计
                switch(level) {
                    case 1: (op_type == 'I') ? l1i_evictions++ : l1d_evictions++; break;
                    case 2: l2_evictions++; break;
                    case 3: l3_evictions++; break;
                }
                // 写回dirty块
                if (l2ucache[set_index][replace_idx].dirty && level < 3) {
                    // 递归写回下层（简化处理，实际需计算地址）
                    uint64_t evicted_addr = (l2ucache[set_index][replace_idx].tag << (int)(log2(set_num) + log2((level==1)?8:(level==2)?8:16))) | (set_index << (int)log2((level==1)?8:(level==2)?8:16));
                    load_to_cache(evicted_addr, level+1, 'S', true);
                }
                // 包含性策略：如果下层被替换，上层需无效化
                if (level == 3) invalidate_upper_caches(addr, 2);
                else if (level == 2) invalidate_upper_caches(addr, 1);
            }

            // 加载新块
            l2ucache[set_index][replace_idx].valid = 1;
            l2ucache[set_index][replace_idx].tag = tag;
            l2ucache[set_index][replace_idx].latest_used = ++timeCount;
            l2ucache[set_index][replace_idx].dirty = is_write;

            return evictions;
            break;
        case 3:
            set_num = L1_SET_NUM;
            line_num = L1_LINE_NUM;
            // 查找可替换行（LRU）
            replace_idx = -1;
            min_used = UINT64_MAX;
            for(int i = 0; i < line_num; i++) {
                if (!l3ucache[set_index][i].valid) {
                    replace_idx = i;
                    break;
                }
                if (l3ucache[set_index][i].latest_used < min_used) {
                    min_used = l3ucache[set_index][i].latest_used;
                    replace_idx = i;
                }
            }

            // 处理替换
            if (l3ucache[set_index][replace_idx].valid) {
                evictions = 1;
                // 更新统计
                switch(level) {
                    case 1: (op_type == 'I') ? l1i_evictions++ : l1d_evictions++; break;
                    case 2: l2_evictions++; break;
                    case 3: l3_evictions++; break;
                }
                // 写回dirty块
                if (l3ucache[set_index][replace_idx].dirty && level < 3) {
                    // 递归写回下层（简化处理，实际需计算地址）
                    uint64_t evicted_addr = (l3ucache[set_index][replace_idx].tag << (int)(log2(set_num) + log2((level==1)?8:(level==2)?8:16))) | (set_index << (int)log2((level==1)?8:(level==2)?8:16));
                    load_to_cache(evicted_addr, level+1, 'S', true);
                }
                // 包含性策略：如果下层被替换，上层需无效化
                if (level == 3) invalidate_upper_caches(addr, 2);
                else if (level == 2) invalidate_upper_caches(addr, 1);
            }

            // 加载新块
            l3ucache[set_index][replace_idx].valid = 1;
            l3ucache[set_index][replace_idx].tag = tag;
            l3ucache[set_index][replace_idx].latest_used = ++timeCount;
            l3ucache[set_index][replace_idx].dirty = is_write;

            return evictions;
            break;
        default: return 0;
    }

    
}

/*----- 包含性策略：下层替换时无效上层缓存 -----*/
void invalidate_upper_caches(uint64_t addr, int start_level) {
    uint64_t tag, set_index;
    for (int level = start_level; level >= 1; level--) {
        divide(addr, level, &tag, &set_index);
        CacheLine (*cache)[16][8];
        int line_num;
        switch(level) {
            case 1: cache = (CacheLine (*)[16][8])l1dcache; line_num = L1_LINE_NUM; break;
            case 2: cache = (CacheLine (*)[16][8])l2ucache; line_num = L2_LINE_NUM; break;
            default: continue;
        }
        for (int i = 0; i < line_num; i++) {
            if ((*cache)[set_index][i].valid && (*cache)[set_index][i].tag == tag) {
                (*cache)[set_index][i].valid = 0;
                if ((*cache)[set_index][i].dirty) {
                    // 写回下层（简化处理）
                    load_to_cache(addr, level+1, 'S', true);
                }
                break;
            }
        }
    }
}

/*----- 缓存访问核心逻辑 -----*/
void cacheAccess(char op, uint64_t addr, uint32_t len) {
    bool is_instruction = (op == 'I');
    int current_level = 1;
    bool hit = false;
    uint64_t tag, set_index;

    // 处理M操作：先Load后Store
    if (op == 'M') {
        cacheAccess('L', addr, len);
        op = 'S'; // 后续处理Store
    }

    // 逐级查询缓存
    while (current_level <= 3 && !hit) {
        divide(addr, current_level, &tag, &set_index);
        CacheLine *cache;
        int line_num;

        switch(current_level) {
            case 1:
                cache = is_instruction ? l3ucache[set_index] : l1dcache[set_index];
                line_num = L1_LINE_NUM;
                break;
            case 2:
                cache = l2ucache[set_index];
                line_num = L2_LINE_NUM;
                break;
            case 3:
                cache = l3ucache[set_index];
                line_num = L3_LINE_NUM;
                break;
            default: return;
        }

        // 检查当前层级是否命中
        for (int i = 0; i < line_num; i++) {
            if (cache[i].valid && cache[i].tag == tag) {
                hit = true;
                cache[i].latest_used = ++timeCount;
                if (op == 'S' || op == 'M') cache[i].dirty = 1;
                break;
            }
        }

        if (hit) {
            // 更新命中统计
            switch(current_level) {
                case 1: is_instruction ? l1i_hits++ : l1d_hits++; break;
                case 2: l2_hits++; break;
                case 3: l3_hits++; break;
            }
            return;
        } else {
            // 更新未命中统计
            switch(current_level) {
                case 1: is_instruction ? l1i_misses++ : l1d_misses++; break;
                case 2: l2_misses++; break;
                case 3: l3_misses++; break;
            }
            current_level++;
        }
    }

    // 未命中所有缓存，从内存加载
    for (int level = 3; level >= 1; level--) {
        if (level == 3 || (level == 2 && current_level > 2) || (level == 1 && current_level > 1)) {
            load_to_cache(addr, level, op, (op == 'S' || op == 'M'));
        }
    }
}

/*----- 缓存初始化 -----*/
void cacheInit() {
    // 初始化L1D和L1I
    for (int i = 0; i < L1_SET_NUM; i++) {
        for (int j = 0; j < L1_LINE_NUM; j++) {
            l1dcache[i][j] = (CacheLine){0};
            l3ucache[i][j] = (CacheLine){0};
        }
    }
    // 初始化L2
    for (int i = 0; i < L2_SET_NUM; i++) {
        for (int j = 0; j < L2_LINE_NUM; j++) {
            l2ucache[i][j] = (CacheLine){0};
        }
    }
    // 初始化L3
    for (int i = 0; i < L3_SET_NUM; i++) {
        for (int j = 0; j < L3_LINE_NUM; j++) {
            l3ucache[i][j] = (CacheLine){0};
        }
    }
}