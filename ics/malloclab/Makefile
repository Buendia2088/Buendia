#
# Students' Makefile for the Malloc Lab
#
TEAM = bovik
VERSION = 1
HANDINDIR = /afs/cs.cmu.edu/academic/class/15213-f01/malloclab/handin

CC = gcc
# CFLAGS = -Wall -O2 -m32
CFLAGS = -Wall -g -O2

ifeq ($(shell uname -m), x86_64)
    CFLAGS += -m64
else ifeq ($(shell uname -m), aarch64)
    CFLAGS += -march=armv8-a
endif

OBJS = mdriver.o mm.o memlib.o fsecs.o fcyc.o clock.o ftimer.o

LDFLAGS = -lm -lrt

mdriver: $(OBJS)
	$(CC) $(CFLAGS) -o mdriver $(OBJS) $(LDFLAGS)

mdriver.o: mdriver.c fsecs.h fcyc.h clock.h memlib.h config.h mm.h
memlib.o: memlib.c memlib.h
mm.o: mm.c mm.h memlib.h
fsecs.o: fsecs.c fsecs.h config.h
fcyc.o: fcyc.c fcyc.h
ftimer.o: ftimer.c ftimer.h config.h
clock.o: clock.c clock.h

clean:
	rm -f *~ *.o mdriver

submit:
	zip ${USER}-handin.zip mm.c


