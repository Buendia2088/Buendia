#include "cachelab.h"
#include <assert.h>
#include <bits/getopt_core.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE 1024
#define MAX_LEVEL 3

int snapshot = 0;
int verbose = 0;
int level = 0;
int set = -1;
int data = 0;
int instruction = 0;

long current_line = 0;
long breakpoint = -1;

char *trace_file = NULL;
char *output_file = NULL;

FILE *out;

extern int l1d_hits;
extern int l1d_misses;
extern int l1d_evictions;
extern int l1i_hits;
extern int l1i_misses;
extern int l1i_evictions;
extern int l2_hits;
extern int l2_misses;
extern int l2_evictions;
extern int l3_hits;
extern int l3_misses;
extern int l3_evictions;

CacheLine l1dcache[L1_SET_NUM][L1_LINE_NUM];
// L1 Instruction Cache
CacheLine l1icache[L1_SET_NUM][L1_LINE_NUM];
// L2 Unified Cache
CacheLine l2ucache[L2_SET_NUM][L2_LINE_NUM];
// L2 Unified Cache
CacheLine l3ucache[L3_SET_NUM][L3_LINE_NUM];

static int cache_sets[] = {L1_SET_NUM, L2_SET_NUM, L3_SET_NUM};

static struct option long_options[] = {
    {"help", no_argument, NULL, 'h'},
    {"snapshot", no_argument, NULL, 'n'},
    {"level", required_argument, NULL, 'l'},
    {"set", required_argument, NULL, 's'},
    {"data", no_argument, NULL, 'd'},
    {"instruction", no_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"trace", required_argument, NULL, 't'},
    {"verbose", required_argument, NULL, 'v'},
    {"breakpoint", required_argument, NULL, 'b'}};

void printSummary(int l1d_hits, int l1d_misses, int l1d_evictions, int l1i_hits,
                  int l1i_misses, int l1i_evictions, int l2_hits, int l2_misses,
                  int l2_evictions, int l3_hits, int l3_misses,
                  int l3_evictions) {
  FILE *fp = fopen(".csim_results", "a");
  assert(fp);
  fprintf(out, "L1-d cache hits:%d misses:%d evictions:%d\n", l1d_hits,
          l1d_misses, l1d_evictions);
  fprintf(fp, "%d %d %d\n", l1d_hits, l1d_misses, l1d_evictions);
  fprintf(out, "L1-i cache hits:%d misses:%d evictions:%d\n", l1i_hits,
          l1i_misses, l1i_evictions);
  fprintf(fp, "%d %d %d\n", l1i_hits, l1i_misses, l1i_evictions);
  fprintf(out, "L2 cache hits:%d misses:%d evictions:%d\n", l2_hits, l2_misses,
          l2_evictions);
  fprintf(fp, "%d %d %d\n", l2_hits, l2_misses, l2_evictions);
  fprintf(out, "L3 cache hits:%d misses:%d evictions:%d\n", l3_hits, l3_misses,
          l3_evictions);
  fprintf(fp, "%d %d %d\n", l3_hits, l3_misses, l3_evictions);
  fclose(fp);
}

void printL1D() {
  int set_start = 0;
  int set_end = L1_SET_NUM;
  int line_start = 0;
  int line_end = L1_LINE_NUM;
  if (set != -1) {
    set_start = set;
    set_end = set + 1;
  }
  for (int i = set_start; i < set_end; i++) {
    for (int j = line_start; j < line_end; j++) {
      fprintf(out, "l1-d cache[%d][%d]: valid=%d, dirty=%d, tag=%lu\n", i, j,
              l1dcache[i][j].valid, l1dcache[i][j].dirty, l1dcache[i][j].tag);
    }
  }
}

void printL1I() {
  int set_start = 0;
  int set_end = L1_SET_NUM;
  int line_start = 0;
  int line_end = L1_LINE_NUM;
  if (set != -1) {
    set_start = set;
    set_end = set + 1;
  }
  for (int i = set_start; i < set_end; i++) {
    for (int j = line_start; j < line_end; j++) {
      fprintf(out, "l1-i cache[%d][%d]: valid=%d, dirty=%d, tag=%lu\n", i, j,
              l1icache[i][j].valid, l1icache[i][j].dirty, l1icache[i][j].tag);
    }
  }
}

void printL2() {
  int set_start = 0;
  int set_end = L2_SET_NUM;
  int line_start = 0;
  int line_end = L2_LINE_NUM;
  if (set != -1) {
    set_start = set;
    set_end = set + 1;
  }
  for (int i = set_start; i < set_end; i++) {
    for (int j = line_start; j < line_end; j++) {
      fprintf(out, "l2 cache[%d][%d]: valid=%d, dirty=%d, tag=%lu\n", i, j,
              l2ucache[i][j].valid, l2ucache[i][j].dirty, l2ucache[i][j].tag);
    }
  }
}

void printL3() {
  int set_start = 0;
  int set_end = L3_SET_NUM;
  int line_start = 0;
  int line_end = L3_LINE_NUM;
  if (set != -1) {
    set_start = set;
    set_end = set + 1;
  }
  for (int i = set_start; i < set_end; i++) {
    for (int j = line_start; j < line_end; j++) {
      fprintf(out, "l3 cache[%d][%d]: valid=%d, dirty=%d, tag=%lu\n", i, j,
              l3ucache[i][j].valid, l3ucache[i][j].dirty, l3ucache[i][j].tag);
    }
  }
}

void printSnapshot() {
  switch (level) {
  case (0): {
    printL1D();
    printL1I();
    printL2();
    printL3();
    break;
  }
  case (1): {
    if (data == 1) {
      printL1D();
    }
    if (instruction == 1) {
      printL1I();
    }
    break;
  }
  case (2): {
    printL2();
    break;
  }
  case (3): {
    printL3();
    break;
  }
  }
}

void cacheInit();
void cacheAccess(char op, uint64_t addr, uint32_t len);

void checkArguments() {
  if (level != 0) {
    if (level < 0 || level > MAX_LEVEL) {
      fprintf(stderr, "fatal error: level must be in range: [1, 2, 3]\n");
      exit(1);
    }
    if (snapshot == 0) {
      fprintf(stderr, "fatal error: [-n, --snapshot] must be enabled\n");
      exit(1);
    }
    if (level == 1) {
      if (data == 0 && instruction == 0) {
        fprintf(stderr,
                "fatal error: data or instruction cache must be specified\n");
        exit(1);
      }
    }
  }
  if (data == 1) {
    if (level != 1) {
      fprintf(stderr, "fatal error: level must be specified to 1 when "
                      "specified data cache\n");
      exit(1);
    }
  }
  if (instruction == 1) {
    if (level != 1) {
      fprintf(stderr, "fatal error: level must be specified to 1 when "
                      "specified instruction cache\n");
      exit(1);
    }
  }
  if (set != -1) {
    if (level == 0) {
      fprintf(stderr, "fatal error: level must be specified first\n");
      exit(1);
    }
    if (set < 0 || set >= cache_sets[level - 1]) {
      fprintf(stderr, "fatal error: set overflow\n");
      exit(1);
    }
  }
  if (breakpoint != -1) {
    if (verbose == 1) {
      fprintf(stderr, "fatal error: you can not specify verbose mode and "
                      "breakpoint simutaneously");
      exit(1);
    }
  }
}

void usage(char *name) {
  fprintf(stderr,
          "usage: %s -t trace_file [-h] [-n] [-v] [-l level] [-d] [-i] [-s "
          "set] [-b breakpoint] \n",
          name);
  fprintf(stderr, "options: \n");
  fprintf(stderr, "-h, --help         print this\n");
  fprintf(stderr, "-o, --output       output file\n");
  fprintf(stderr, "-t, --trace        input trace file (required)\n");
  fprintf(stderr, "-n, --snapshot     print cache snapshot\n");
  fprintf(stderr, "-b, --breakpoint   print cache info after one specified "
                  "access, note that it can not be enabled in verbose mode\n");
  fprintf(stderr,
          "-v, --verbose      print information after each cache access, "
          "instead of print snapshot at the end\n");
  fprintf(stderr, "-l, --level        which cache level [l1--1, l2--2, l3--3] "
                  "to print (only can be "
                  "specified when [-n] is enabled)\n");
  fprintf(stderr,
          "-d, --data         data cache (must be specified if level == 1)\n");
  fprintf(stderr, "-i, --instruction  instruction cache (must be specified if "
                  "level == 1 )\n");
  fprintf(stderr, "-s, --set          which cache set (index start from 0) to "
                  "print (only can be "
                  "specified when [-l] is enabled)\n");
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    usage(argv[0]);
    exit(1);
  }

  int opt;
  while ((opt = getopt_long(argc, argv, "ht:o:nl:dis:vb:", long_options,
                            NULL)) != -1) {
    switch (opt) {
    case 'h':
      usage(argv[0]);
      exit(0);
    case 't':
      trace_file = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'n':
      snapshot = 1;
      break;
    case 'l':
      level = atoi(optarg);
      break;
    case 'd':
      data = 1;
      break;
    case 'i':
      instruction = 1;
      break;
    case 's': {
      set = atoi(optarg);
      break;
    }
    case 'v': {
      verbose = 1;
      break;
    }
    case 'b': {
      breakpoint = atol(optarg);
      break;
    }
    case '?':
      usage(argv[0]);
    default:
      exit(1);
    }
  }

  checkArguments();

  FILE *trace_fp = fopen(trace_file, "r");
  if (!trace_fp) {
    fprintf(stderr, "fatal error: can not open trace files %s: %s", trace_file,
            strerror(errno));
    exit(1);
  }

  out = stdout;
  if (output_file != NULL) {
    out = fopen(output_file, "w");
    if (!out) {
      fprintf(stderr, "fatal error: can not open output files %s: %s\n",
              output_file, strerror(errno));
      exit(1);
    }
  }

  cacheInit();

  char buf[BUF_SIZE];
  uint64_t addr;
  uint32_t len;
  char op;
  while (fgets(buf, BUF_SIZE, trace_fp)) {
    if (buf[0] == 'I') {
      op = buf[0];
    } else if (buf[0] == ' ') {
      op = buf[1];
    } else {
      continue; // ignore empty lines and lines begin with '='
    }
    sscanf(buf + 3, "%lx,%d", &addr, &len);
    // to be implemented in csim.c
    if (verbose == 1 || (current_line == breakpoint)) {
      fprintf(out, "%ld: %c %lx %d\n", current_line, op, addr, len);
    }
    cacheAccess(op, addr, len);
    if (verbose == 1 || (current_line == breakpoint)) {
      printSummary(l1d_hits, l1d_misses, l1d_evictions, l1i_hits, l1i_misses,
                   l1i_evictions, l2_hits, l2_misses, l2_evictions, l3_hits,
                   l3_misses, l3_evictions);
      if (snapshot == 1) {
        printSnapshot();
      }
    }
    current_line++;
  }

  if (verbose == 0) {
    printSummary(l1d_hits, l1d_misses, l1d_evictions, l1i_hits, l1i_misses,
                 l1i_evictions, l2_hits, l2_misses, l2_evictions, l3_hits,
                 l3_misses, l3_evictions);
    if (snapshot == 1) {
      printSnapshot();
    }
  }

  fclose(trace_fp);
  if (out != stdout) {
    fclose(out);
  }

  return 0;
}
