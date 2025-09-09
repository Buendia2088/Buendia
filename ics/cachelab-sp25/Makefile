CC = gcc
CFLAGS = -g -Wall -Werror -m64

all: csim test-trans tracegen

csim: csim.c cachelab.c cachelab.h cache-impl.c
	$(CC) $(CFLAGS) -o csim csim.c cachelab.c cache-impl.c -lm

test-trans: test-trans.c trans.o cachelab.c cachelab.h
	$(CC) $(CFLAGS) -o test-trans test-trans.c cachelab.c trans.o

tracegen: tracegen.c trans.o cachelab.c
	$(CC) $(CFLAGS) -O0 -o tracegen tracegen.c trans.o cachelab.c

trans.o: trans.c
	$(CC) $(CFLAGS) -O0 -c trans.c

grade:
	python3 ./driver.py

submit:
	zip ${USER}-handin.zip cache-impl.c

clean:
	rm -rf *.o
	rm -f *.zip
	rm -f csim
	rm -f test-trans tracegen
	rm -f trace.all trace.f* trace.tmp
	rm -f .csim_results .marker

