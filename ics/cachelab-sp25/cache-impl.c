// 注：此文件包含ai生成的注释，其余部分为手搓
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
void invalidate_upper_caches(uint64_t addr, int start_level, char op_type);

void hit_count(int current_level, bool is_instruction){
    switch (current_level) {
        case 1: is_instruction ? l1i_hits++ : l1d_hits++; break;
        case 2: l2_hits++; break;
        case 3: l3_hits++; break;
    }
}

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

uint64_t get_address(int level, uint64_t tag, uint64_t set_index){
    uint64_t addr = 0;
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
            addr; return 0;
    }
    addr = tag << (offset_bits + set_bits) | set_index << offset_bits;
    return addr;
}

int load_to_cache(uint64_t addr, int level, char op_type, bool is_write) {
    uint64_t tag, set_index;
    divide(addr, level, &tag, &set_index);
    void* cache_ptr;
    int set_num, line_num, evictions = 0;
    bool is_ecivted = false;
    bool is_fetching = false;
    if(op_type == 'A'){
        is_fetching = true;
        op_type = 'I';
    }
    if(op_type == 'B'){
        is_fetching = true;
        op_type = 'L';
    }

    switch(level) {
        case 1:
            cache_ptr = (op_type == 'I') ? (void*)l1icache : (void*)l1dcache;
            set_num = L1_SET_NUM;
            line_num = L1_LINE_NUM;
            break;
        case 2:
            cache_ptr = (void*)l2ucache;
            set_num = L2_SET_NUM;
            line_num = L2_LINE_NUM;
            break;
        case 3:
            cache_ptr = (void*)l3ucache;
            set_num = L3_SET_NUM;
            line_num = L3_LINE_NUM;
            break;
        default: return 0;
    }

    // 通过正确的指针类型访问缓存
    CacheLine (*cache)[set_num][line_num] = (CacheLine (*)[set_num][line_num])cache_ptr;

    // 查找可替换行（LRU）
    int replace_idx = -1;
    uint64_t min_used = UINT64_MAX;
    for(int i = 0; i < line_num; i++) {
        if (!(*cache)[set_index][i].valid) {
            replace_idx = i;
            break;
        }
        if(((*cache)[set_index][i].valid == 1) && (*cache)[set_index][i].tag == tag){
            replace_idx = i;
            hit_count(level, (op_type == 'I') ? true : false);
            break;
        }
        if ((*cache)[set_index][i].latest_used < min_used) {
            min_used = (*cache)[set_index][i].latest_used;
            replace_idx = i;
        }
    }

    // 处理替换
    if ((*cache)[set_index][replace_idx].valid && (*cache)[set_index][replace_idx].tag != tag) {
        evictions = 1;
        uint64_t target_addr;
        // 更新统计
        switch(level) {
            case 1: (op_type == 'I') ? l1i_evictions++ : l1d_evictions++; break;
            case 2: l2_evictions++; break;
            case 3: l3_evictions++; break;
        }
        // 写回dirty块

        // 包含性策略：如果下层被替换，上层需无效化
        if (level == 3 && op_type != 'E') {
            target_addr = get_address(level, (*cache)[set_index][replace_idx].tag, set_index);
            invalidate_upper_caches(target_addr, 2, op_type);
        }
        else if (level == 2 && op_type != 'E') {
            target_addr = get_address(level, (*cache)[set_index][replace_idx].tag, set_index);
            invalidate_upper_caches(target_addr, 1, op_type);
        }
        if ((*cache)[set_index][replace_idx].dirty && level < 3) {
            // 递归写回下层（简化处理，实际需计算地址）
            target_addr = get_address(level, (*cache)[set_index][replace_idx].tag, set_index);
            load_to_cache(target_addr, level+1, 'E', true);
        }
        (*cache)[set_index][replace_idx].valid = 1; // 立即进行替换
        (*cache)[set_index][replace_idx].tag = tag;
        (*cache)[set_index][replace_idx].latest_used = ++timeCount;
        is_ecivted = true;

    }

    // 控制 dirty 位
    // if(evictions == 0 && level >= 2){
    //     is_write = 0;  // 对于 L2 和 L3 缓存，默认不标记为 dirty
    // }

    if(op_type == 'E') is_write = 1;
    if(is_fetching == 1 && level >= 2) is_write = 0;
    // 加载新块：严格按照 inclusive policy 设置
    if(is_ecivted == false){
        (*cache)[set_index][replace_idx].valid = 1;
        (*cache)[set_index][replace_idx].tag = tag;
        (*cache)[set_index][replace_idx].latest_used = ++timeCount;
    }


    (*cache)[set_index][replace_idx].dirty = is_write; // L1 才是 dirty
    return evictions;
}


/*----- 包含性策略：下层替换时无效上层缓存 -----*/
void invalidate_upper_caches(uint64_t addr, int start_level, char op_type) {
    uint64_t tag, set_index;
    for (int level = start_level; level >= 1; level--) {
        divide(addr, level, &tag, &set_index);
        void* cache_ptr;
        int set_num, line_num;

        switch(level) {
            case 1:
                cache_ptr = (op_type == 'I') ? (void*)l1icache : (void*)l1dcache;
                set_num = L1_SET_NUM;
                line_num = L1_LINE_NUM;
                break;
            case 2:
                cache_ptr = (void*)l2ucache;
                set_num = L2_SET_NUM;
                line_num = L2_LINE_NUM;
                break;
            default: continue;
        }

        CacheLine (*cache)[set_num][line_num] = (CacheLine (*)[set_num][line_num])cache_ptr;
        for (int i = 0; i < line_num; i++) {
            if ((*cache)[set_index][i].valid && (*cache)[set_index][i].tag == tag) {
                (*cache)[set_index][i].valid = 0;
                if ((*cache)[set_index][i].dirty) {
                    load_to_cache(addr, level+1, 'S', true);
                }
                break;
            }
        }
    }
}

/*----- 更新更低级缓存函数 -----*/
void update_lower_caches(uint64_t addr, int from_level, char op_type) {
    uint64_t tag, set_index;
    // 对于比 from_level 小的所有缓存级别（例如，若 hit 在 L2，则检查 L1；若 hit 在 L3，则检查 L2 和 L1）
    for (int level = from_level - 1; level >= 1; level--) {
        divide(addr, level, &tag, &set_index);
        void *cache_ptr;
        int set_num, line_num;
        switch (level) {
            case 1:
                cache_ptr = (op_type == 'I') ? (void*)l1icache : (void*)l1dcache; // 更新数据缓存（也可以根据情况同时更新指令缓存）
                set_num = L1_SET_NUM;
                line_num = L1_LINE_NUM;
                break;
            case 2:
                cache_ptr = (void*)l2ucache;
                set_num = L2_SET_NUM;
                line_num = L2_LINE_NUM;
                break;
            default:
                continue;
        }
        // 以二维数组方式访问缓存：cache_ptr 的类型为 CacheLine (*)[set_num][line_num]
        CacheLine (*cache)[set_num][line_num] = (CacheLine (*)[set_num][line_num])cache_ptr;
        bool found = false;
        // 遍历该 set 中所有行，查找 tag 匹配且有效的 cacheline
        for (int i = 0; i < line_num; i++) {
            if ((*cache)[set_index][i].valid && (*cache)[set_index][i].tag == tag) {
                found = true;
                break;
            }
        }
        if (!found) {
            // 较低级缓存缺失该数据，按照包含性策略，写入新块，
            // 注意这里调用 load_to_cache() 时，采用 load 模式（op 'L'），并且设为 clean（is_write = false）
            load_to_cache(addr, level, (op_type == 'I') ? 'A' : 'B', (op_type == 'S') ? true : false);
        }
    }
}

/*----- 缓存访问核心逻辑 -----*/
void cacheAccess(char op, uint64_t addr, uint32_t len) {
    bool is_instruction = (op == 'I');
    int current_level = 1;
    bool hit = false;
    uint64_t tag, set_index;

    // 处理 M 操作：先 Load 后 Store
    if (op == 'M') {
        cacheAccess('L', addr, len);
        op = 'S'; // 后续处理 Store
    }

    // 逐级查询缓存
    while (current_level <= 3 && !hit) {
        divide(addr, current_level, &tag, &set_index);
        CacheLine *cache;
        int line_num;

        switch (current_level) {
            case 1:
                cache = is_instruction ? l1icache[set_index] : l1dcache[set_index];
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
            default:
                return;
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
            hit_count(current_level, (op == 'I' ? true : false));
            // 命中后，检查更低级别缓存（级别 < current_level），如果缺失，则写入
            update_lower_caches(addr, current_level, op);
            return;
        } else {
            // 更新未命中统计
            switch (current_level) {
                case 1: is_instruction ? l1i_misses++ : l1d_misses++; break;
                case 2: l2_misses++; break;
                case 3: l3_misses++; break;
            }
            current_level++;
        }
    }

    // 如果所有层级均未命中，从内存加载：对未命中层逐级加载（这里加载策略简化处理）
    for (int level = 3; level >= 1; level--) {
        if (level == 3 || (level == 2 && current_level > 2) || (level == 1 && current_level > 1)) {
            load_to_cache(addr, level, (op == 'I') ? 'A' : 'B', (op == 'S') ? true : false);
        }
    }
}


/*----- 缓存初始化 -----*/
void cacheInit() {
    // 初始化L1D和L1I
    for (int i = 0; i < L1_SET_NUM; i++) {
        for (int j = 0; j < L1_LINE_NUM; j++) {
            l1dcache[i][j] = (CacheLine){0};
            l1icache[i][j] = (CacheLine){0};
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