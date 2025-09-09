/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 * 
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include "mm.h"
#include "memlib.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "XJTU ICS",
    /* First member's full name */
    "Your name please",
    /* First member's email address */
    "Your email address please",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""
};

void *extendHeap(size_t size);

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
#define LIMIT 100

void *HeapList = NULL;
void *listHead = NULL;
int mallocTimes = 0;
long a[10000];

void removeBlock(void *ptr) {
    void **prev = PREV_PTR(ptr);
    void **next = NEXT_PTR(ptr);
    if (*prev) { // ptr is not head of free list
        SET_NEXT(NEXT_PTR(*prev), *next);
    }
    else { // ptr is head of list
        listHead = *next; 
    }
    SET_PREV(PREV_PTR(*next), *prev);
}

void insertBlock(void *ptr) {
    // since listHead is initialized with an allocated block, it can't be NULL
    SET_NEXT(NEXT_PTR(ptr), listHead);
    SET_PREV(PREV_PTR(ptr), NULL);
    SET_PREV(PREV_PTR(listHead), ptr);
    listHead = ptr;
}

void *Merge(void *ptr) {
    void *header = HEAD_PTR(ptr);
    unsigned int size = GET_SIZE(header);
    void *prevBlock = PREV_BLOCK(ptr);
    void *nextBlock = NEXT_BLOCK(ptr);
    void *prevFooter = header - WORD_SIZE;
    void *nextHeader = header + size;
    int prevAlloc = (READ(prevFooter) & 1) || (prevBlock == ptr);
    int nextAlloc = (READ(nextHeader) & 1);
    unsigned int prevSize = GET_SIZE(prevFooter);
    unsigned int nextSize = GET_SIZE(nextHeader);
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
    else {
        removeBlock(prevBlock);
        removeBlock(nextBlock);
        size += nextSize + prevSize;
        WRITE(HEAD_PTR(prevBlock), PACK(size, 0));
        WRITE(TAIL_PTR(prevBlock), PACK(size, 0));
        insertBlock(prevBlock);
        return prevBlock;
    }
}

void* Place(void *ptr, unsigned int Size) {
    void *header = HEAD_PTR(ptr);
    unsigned int totalSize = GET_SIZE(header);
    removeBlock(ptr);
    if (totalSize >= Size + (WORD_SIZE << 1) + (VOID_SIZE << 1)) {
        // Split the block
        if (Size < LIMIT) {
            WRITE(header, PACK(Size, 1));
            WRITE(TAIL_PTR(ptr), *(unsigned int *)header);
            void *nextBlock = NEXT_BLOCK(ptr);
            WRITE(HEAD_PTR(nextBlock), PACK(totalSize - Size, 0));
            WRITE(TAIL_PTR(nextBlock), *(unsigned int *)HEAD_PTR(nextBlock));
            Merge(nextBlock);
        }
        else {
            WRITE(header, PACK(totalSize - Size, 0));
            WRITE(TAIL_PTR(ptr), *(unsigned int *)header);
            void *tail = TAIL_PTR(ptr);
            void *nextBlock = NEXT_BLOCK(ptr);
            WRITE(HEAD_PTR(nextBlock), PACK(Size, 1));
            WRITE(TAIL_PTR(nextBlock), *(unsigned int *)HEAD_PTR(nextBlock));
            Merge(ptr);
            ptr = nextBlock;
        }

    } else {
        // Allocate the entire block
        WRITE(header, PACK(totalSize, 1));
        WRITE(TAIL_PTR(ptr), *(unsigned int *)header);
    }
    return ptr;
}

void *FirstFit(size_t size) {
    void *ptr = listHead;
    while (ptr) {
        if (!IS_ALLOC(HEAD_PTR(ptr)) && GET_SIZE(HEAD_PTR(ptr)) >= size) {
            return ptr;
        }
        void **next = NEXT_PTR(ptr);
        ptr = *next;
    }
    return NULL;
}

int mm_init() {
    // Request for 16 bytes space
    HeapList = mem_sbrk(WORD_SIZE << 3);
    if (HeapList == (void *)-1) return -1;
    // Fill in metadata as initial space
    WRITE(HeapList, 0);
    // Prologue block
    WRITE(HeapList + WORD_SIZE * 1, PACK(8, 1));
    WRITE(HeapList + WORD_SIZE * 2, PACK(8, 1));
    // Epilogue block
    WRITE(HeapList + WORD_SIZE * 3, PACK(0, 1));
    listHead = HeapList + (WORD_SIZE << 1); // free list head initialization
    extendHeap(24);
    a[0] = HeapList;
    return 0;
}

void *mm_malloc(size_t size) {
    mallocTimes++;
    // If size equals zero, which means we don't need to execute malloc
    if (size == 0) return NULL;
    // Add header size and tailer size to block size
    size += (WORD_SIZE << 1);
    // Round up size to mutiple of 8
    if ((size & (unsigned int)7) > 0) size += (1 << 3) - (size & 7);
    // We call first fit function to find a space with size greater than argument 'size'
    void *Ptr = FirstFit(size);
    // If first fit function return NULL, which means there's no suitable space.
    // Else we find it. The all things to do is to place it.
    if (Ptr != NULL) {
        Ptr = Place(Ptr, size);
        a[mallocTimes] = Ptr;
        return Ptr;
    }
    // We call sbrk to extend heap size
    unsigned int SbrkSize = MAX(size, PAGE_SIZE + (WORD_SIZE << 5));
    void *NewPtr = extendHeap(SbrkSize);
    NewPtr = Place(NewPtr, size);
    a[mallocTimes] = NewPtr;
    return NewPtr;
}

void mm_free(void *ptr) {
    // We just fill in the header and tailer with PACK(Size, 0)
    void *Header = HEAD_PTR(ptr), *Tail = TAIL_PTR(ptr);
    unsigned int Size = GET_SIZE(Header);
    WRITE(Header, PACK(Size, 0));
    WRITE(Tail, PACK(Size, 0));
    // Then merge it with adjacent free blocks
    Merge(ptr);
}

void *mm_realloc(void *ptr, size_t size) {
    // We get block's original size
    unsigned int BlkSize = GET_SIZE(HEAD_PTR(ptr));
    // Round up size to mutiple of 8
    if ((size & (unsigned int)7) > 0) size += (1 << 3) - (size & 7);
    // If original size is greater than requested size, we don't do any.
    if (BlkSize >= size + WORD_SIZE * 2) return ptr;
    // Else, we call malloc to get a new space for it.
    void *NewPtr = mm_malloc(size);
    mallocTimes--;
    if (NewPtr == NULL) return NULL;
    // Move the data to new space
    memmove(NewPtr, ptr, size);
    // Free old block
    mm_free(ptr);
    return NewPtr;
}


void *extendHeap(size_t size) {
    if (size == 0) return NULL;
    // Round up size to mutiple of 8
    if ((size & (unsigned int)7) > 0) size += (1 << 3) - (size & 7);
    if (size < 24) size = 24;
    void *NewPtr = mem_sbrk(size);
    if (NewPtr == (void *)-1) return NULL;
    // Write metadata in newly requested space
    WRITE(NewPtr - WORD_SIZE, PACK(size, 0));
    WRITE(TAIL_PTR(NewPtr), PACK(size, 0));
    WRITE(TAIL_PTR(NewPtr) + WORD_SIZE, PACK(0, 1));
    // Execute function merge to merge new space and free block in front of it
    return Merge(NewPtr);
}