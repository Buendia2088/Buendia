#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "mm.h"
#include "memlib.h"

team_t team = {
    "XJTU ICS",
    "Your name please",
    "Your email address please",
    "",
    ""
};

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define WORD_SIZE (sizeof(unsigned int))
#define VOID_SIZE (sizeof(void *))
#define READ(PTR) (*(unsigned int *)(PTR))
#define WRITE(PTR, VALUE) ((*(unsigned int *)(PTR)) = (VALUE))
#define PACK(SIZE, IS_ALLOC) ((SIZE) | (IS_ALLOC))
#define GET_SIZE(PTR) (unsigned int)((READ(PTR) >> 3) << 3)
#define IS_ALLOC(PTR) (READ(PTR) & (unsigned int)1)
#define HEAD_PTR(PTR) ((void *)(PTR) - WORD_SIZE)
#define TAIL_PTR(PTR) ((void *)(PTR) + GET_SIZE(HEAD_PTR(PTR)) - (WORD_SIZE << 1))
#define NEXT_PTR(PTR) ((void *)(PTR) + VOID_SIZE)
#define PREV_PTR(PTR) ((void *)(PTR))
#define SET_NEXT(NEXT, ADDR) *(void **)(NEXT) = ADDR
#define SET_PREV(PREV, ADDR) *(void **)(PREV) = ADDR
#define NEXT_BLOCK(PTR) ((void *)(PTR) + GET_SIZE(HEAD_PTR(PTR)))
#define PREV_BLOCK(PTR) ((void *)(PTR) - GET_SIZE(PTR - (WORD_SIZE << 1)))
#define HEADER_TO_FREE_BLOCK(PTR) (void *)(PTR) + WORD_SIZE + (VOID_SIZE << 1)
#define PAGE_SIZE (1 << 12)
#define THRESHOLD 100
#define SEG_LIST_COUNT 43
#define MIN_BLOCK_SIZE (2 * (WORD_SIZE + VOID_SIZE))
#define MIN_FREE_SIZE (MIN_BLOCK_SIZE + (WORD_SIZE << 1)) // 最小空闲块大小

void *HeapList = NULL;
void *extendHeap(size_t size);

int getSegListIndex(size_t size)
{
    if (size <= 16)      return  0;
    if (size <= 18)      return  1;
    if (size <= 20)      return  2;
    if (size <= 22)      return  3;
    if (size <= 24)      return  4;
    if (size <= 28)      return  5;
    if (size <= 32)      return  6;
    if (size <= 40)      return  7;
    if (size <= 48)      return  8;
    if (size <= 56)      return  9;
    if (size <= 80)      return 10;
    if (size <= 96)      return 11;
    if (size <= 112)     return 12;
    if (size <= 128)     return 13;
    if (size <= 160)     return 14;
    if (size <= 192)     return 15;
    if (size <= 224)     return 16;
    if (size <= 256)     return 17;
    if (size <= 320)     return 18;
    if (size <= 384)     return 19;
    if (size <= 448)     return 20;
    if (size <= 512)     return 21;
    if (size <= 640)     return 22;
    if (size <= 768)     return 23;
    if (size <= 896)     return 24;
    if (size <= 1024)    return 25;
    if (size <= 1280)    return 26;
    if (size <= 1536)    return 27;
    if (size <= 1792)    return 28;
    if (size <= 2048)    return 29;
    if (size <= 2560)    return 30;
    if (size <= 3072)    return 31;
    if (size <= 3584)    return 32;
    if (size <= 4096)    return 33;
    if (size <= 5120)    return 34;
    if (size <= 6144)    return 35;
    if (size <= 7168)    return 36;
    if (size <= 8192)    return 37;
    if (size <= 12288)   return 38;
    if (size <= 16384)   return 39;
    if (size <= 24576)   return 40;
    if (size <= 32768)   return 41;
    return 42;
}

void **getListHead(int index) {
    return (void **)(HeapList + index * VOID_SIZE);
}

void removeFromFreeList(void *ptr, int listIndex) {
    void **listHead = getListHead(listIndex);
    void **prev = PREV_PTR(ptr);
    void **next = NEXT_PTR(ptr);
    if (*prev) {
        SET_NEXT(NEXT_PTR(*prev), *next);
    } else {
        *listHead = *next;
    }
    if (*next) {
        SET_PREV(PREV_PTR(*next), *prev);
    }
}

void addToFreeList(void *ptr, int listIndex) {
    void **listHead = getListHead(listIndex);
    SET_NEXT(NEXT_PTR(ptr), *listHead);
    SET_PREV(PREV_PTR(ptr), NULL);
    if (*listHead) {
        SET_PREV(PREV_PTR(*listHead), ptr);
    }
    *listHead = ptr;
}

void removeBlock(void *ptr) {
    size_t size = GET_SIZE(HEAD_PTR(ptr));
    int listIndex = getSegListIndex(size);
    removeFromFreeList(ptr, listIndex);
}

void insertBlock(void *ptr) {
    size_t size = GET_SIZE(HEAD_PTR(ptr));
    int listIndex = getSegListIndex(size);
    addToFreeList(ptr, listIndex);
}

void *Merge(void *ptr) {
    void *header = HEAD_PTR(ptr);
    unsigned int size = GET_SIZE(header);
    void *prevBlock = PREV_BLOCK(ptr);
    void *nextBlock = NEXT_BLOCK(ptr);
    void *prevFooter = header - WORD_SIZE;
    void *nextHeader = header + size;
    int prevAlloc = (READ(prevFooter) & 1) || (prevBlock == ptr);
    unsigned int prevSize = 0;
    if (!prevAlloc) {
        prevSize = GET_SIZE(HEAD_PTR(prevBlock));
    }
    int nextAlloc = (READ(nextHeader) & 1);
    unsigned int nextSize = 0;
    if (!nextAlloc) {
        nextSize = GET_SIZE(HEAD_PTR(nextBlock));
    }
    if (prevAlloc && nextAlloc) {
        insertBlock(ptr);
        return ptr;
    }
    if (prevAlloc && !nextAlloc) {
        removeBlock(nextBlock);
        size += nextSize;
        WRITE(header, PACK(size, 0));
        WRITE(TAIL_PTR(ptr), PACK(size, 0));
        insertBlock(ptr);
        return ptr;
    }
    if (!prevAlloc && nextAlloc) {
        removeBlock(prevBlock);
        size += prevSize;
        WRITE(HEAD_PTR(prevBlock), PACK(size, 0));
        WRITE(TAIL_PTR(prevBlock), PACK(size, 0));
        insertBlock(prevBlock);
        return prevBlock;
    }
    removeBlock(prevBlock);
    removeBlock(nextBlock);
    size += nextSize + prevSize;
    WRITE(HEAD_PTR(prevBlock), PACK(size, 0));
    WRITE(TAIL_PTR(prevBlock), PACK(size, 0));
    insertBlock(prevBlock);
    return prevBlock;
}

void *Place(void *ptr, unsigned int Size) {
    void *header = HEAD_PTR(ptr);
    unsigned int totalSize = GET_SIZE(header);
    removeBlock(ptr);
    void *allocPtr = ptr;
    unsigned int minSplitSize = Size + MIN_FREE_SIZE;
    
    if (totalSize >= minSplitSize) {
        if (Size > THRESHOLD) {
            unsigned int remainSize = totalSize - Size;
            if (remainSize >= MIN_FREE_SIZE) {
                WRITE(header, PACK(remainSize, 0));
                WRITE(header + remainSize - WORD_SIZE, PACK(remainSize, 0));
                insertBlock(ptr);
                void *allocHeader = header + remainSize;
                allocPtr = allocHeader + WORD_SIZE;
                WRITE(allocHeader, PACK(Size, 1));
                WRITE(allocHeader + Size - WORD_SIZE, PACK(Size, 1));
            } else {
                WRITE(header, PACK(totalSize, 1));
                WRITE(TAIL_PTR(ptr), PACK(totalSize, 1));
            }
        } else {
            unsigned int remainSize = totalSize - Size;
            if (remainSize >= MIN_FREE_SIZE) {
                WRITE(header, PACK(Size, 1));
                WRITE(TAIL_PTR(ptr), PACK(Size, 1));               
                void *nextBlock = NEXT_BLOCK(ptr);
                WRITE(HEAD_PTR(nextBlock), PACK(remainSize, 0));
                WRITE(TAIL_PTR(nextBlock), PACK(remainSize, 0));
                insertBlock(nextBlock);
            } else {
                WRITE(header, PACK(totalSize, 1));
                WRITE(TAIL_PTR(ptr), PACK(totalSize, 1));
            }
        }
    } else {
        WRITE(header, PACK(totalSize, 1));
        WRITE(TAIL_PTR(ptr), PACK(totalSize, 1));
    }
    return allocPtr;
}

void *SegFit(size_t size){
    int start = getSegListIndex(size);
    void *best = NULL;
    size_t bestSize = 0;
    for(int i = start; i < SEG_LIST_COUNT; i++) {
        void *ptr = *getListHead(i);
        while(ptr) {
            size_t blkSize = GET_SIZE(HEAD_PTR(ptr));
            if(blkSize >= size){
                if(!best || blkSize < bestSize) {
                    best = ptr;
                    bestSize = blkSize;
                    if(bestSize == size) return best;
                }
            }
            ptr = *(void **)NEXT_PTR(ptr);
        }
    }
    return best;
}

int mm_init() {
    size_t listSpace = SEG_LIST_COUNT * VOID_SIZE;
    size_t initSize = listSpace + (WORD_SIZE << 3);
    HeapList = mem_sbrk(initSize);
    if (HeapList == (void *)-1) return -1;
    for (int i = 0; i < SEG_LIST_COUNT; i++) {
        *getListHead(i) = NULL;
    }
    void *heapStart = HeapList + listSpace;
    WRITE(heapStart, 0);
    WRITE(heapStart + WORD_SIZE, PACK(8, 1));
    WRITE(heapStart + WORD_SIZE * 2, PACK(8, 1));
    WRITE(heapStart + WORD_SIZE * 3, PACK(0, 1));
    extendHeap(24);
    return 0;
}

void *mm_malloc(size_t size) {
    if (size == 0) return NULL;
    size += (WORD_SIZE << 1);
    if ((size & (unsigned int)7) > 0)
        size += (1 << 3) - (size & 7);
    if (size < MIN_BLOCK_SIZE)
        size = MIN_BLOCK_SIZE;
    void *Ptr = SegFit(size);
    if (Ptr != NULL) {
        return Place(Ptr, size);
    }
    unsigned int minSize = MAX(size, PAGE_SIZE);
    unsigned int maxSize = MAX(size * 2, PAGE_SIZE * 2);
    unsigned int SbrkSize = (minSize < maxSize) ? maxSize : minSize;
    void *NewPtr = extendHeap(SbrkSize);
    return Place(NewPtr, size);
}

void mm_free(void *ptr) {
    void *Header = HEAD_PTR(ptr), *Tail = TAIL_PTR(ptr);
    unsigned int Size = GET_SIZE(Header);
    WRITE(Header, PACK(Size, 0));
    WRITE(Tail, PACK(Size, 0));
    Merge(ptr);
}
void *mm_realloc(void *ptr, size_t size) {
    unsigned int BlkSize = GET_SIZE(HEAD_PTR(ptr));
    if ((size & (unsigned int)7) > 0) size += (1 << 3) - (size & 7);
    if (BlkSize >= size + WORD_SIZE * 2) return ptr;
    void *NewPtr = mm_malloc(size);
    if (NewPtr == NULL) return NULL;
    memmove(NewPtr, ptr, size);
    mm_free(ptr);
    return NewPtr;
}

void *extendHeap(size_t size) {
    if (size == 0) return NULL;
    if ((size & (unsigned int)7) > 0)
        size += (1 << 3) - (size & 7);
    if (size < MIN_FREE_SIZE) size = MIN_FREE_SIZE;
    void *NewPtr = mem_sbrk(size);
    if (NewPtr == (void *)-1) return NULL;
    void *newBlock = NewPtr - WORD_SIZE;
    WRITE(newBlock, PACK(size, 0));
    WRITE(newBlock + size - WORD_SIZE, PACK(size, 0));
    WRITE(newBlock + size, PACK(0, 1));
    void *payload = newBlock + WORD_SIZE;
    return Merge(payload);
}