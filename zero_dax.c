// zero_dax.c
// Zero a /dev/dax device and persist with CPU cache-line flushes (no libpmem).
// Build: cc -O2 -Wall -Wextra -march=native -o zero_dax zero_dax.c
// Usage: sudo ./zero_dax /dev/dax2.0
//
// This maps the device in chunks, memset()s to zero, then flushes with CLWB
// (or CLFLUSHOPT / CLFLUSH) and issues an SFENCE. No msync, no libpmem.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <cpuid.h>
#include <emmintrin.h>   // _mm_clflush
#include <immintrin.h>   // _mm_clflushopt, _mm_sfence (and CLWB on newer toolchains)

#ifndef MAP_SHARED_VALIDATE
#define MAP_SHARED_VALIDATE MAP_SHARED
#endif
#ifndef MAP_SYNC
#define MAP_SYNC 0
#endif

static volatile sig_atomic_t got_sigbus = 0;
static void sigbus_handler(int signo) { (void)signo; got_sigbus = 1; }

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

static uint64_t read_dax_size_from_sysfs(const char *devpath) {
    const char *base = strrchr(devpath, '/');
    base = base ? base + 1 : devpath; // e.g. "dax2.0"

    char sysfs_path[PATH_MAX];
    struct stat st;

    // Prefer modern path
    snprintf(sysfs_path, sizeof(sysfs_path),
             "/sys/bus/dax/devices/%s/size", base);
    if (stat(sysfs_path, &st) != 0) {
        // Fallback
        snprintf(sysfs_path, sizeof(sysfs_path),
                 "/sys/class/dax/%s/size", base);
        if (stat(sysfs_path, &st) != 0)
            die("Cannot find size for %s under /sys/bus/dax/devices or /sys/class/dax", base);
    }

    FILE *f = fopen(sysfs_path, "r");
    if (!f) die("Failed to open %s: %s", sysfs_path, strerror(errno));

    unsigned long long size = 0ULL;
    if (fscanf(f, "%llu", &size) != 1) {
        fclose(f);
        die("Failed to parse size from %s", sysfs_path);
    }
    fclose(f);
    return (uint64_t)size;
}

// Feature detection
static bool have_clflushopt = false;
static bool have_clwb = false;

static void detect_flush_features(void) {
    unsigned eax, ebx, ecx, edx;
    // CPUID leaf 7, subleaf 0
    if (__get_cpuid_max(0, 0) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        // EBX bits: 23=CLFLUSHOPT, 24=CLWB (on many toolchains; on some, CLWB is bit 24)
        have_clflushopt = (ebx & (1u << 23)) != 0;
        have_clwb       = (ebx & (1u << 24)) != 0;
    }
}

// Portable wrappers for flush ops
static inline void flush_clwb(const void *p) {
#if defined(__clang__) || defined(__GNUC__)
    // Use asm if intrinsic not available
    asm volatile(".byte 0x66,0x0f,0xae,0x30" :: "m"(*(const char *)p));
#else
    // Fallback to CLFLUSH if compiler lacks CLWB form
    _mm_clflush((void *)p);
#endif
}

static inline void flush_clflushopt(const void *p) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile(".byte 0x66,0x0f,0xae,0x38" :: "m"(*(const char *)p));
#else
    _mm_clflush((void *)p);
#endif
}

// Persist a range: choose CLWB > CLFLUSHOPT > CLFLUSH, then SFENCE
static void persist_range(void *addr, size_t len) {
    const size_t cacheline = 64; // x86-64 common line size
    uintptr_t p = (uintptr_t)addr;
    uintptr_t start = p & ~(cacheline - 1);
    uintptr_t end   = (uintptr_t)addr + len;

    for (uintptr_t q = start; q < end; q += cacheline) {
        if (have_clwb) {
            flush_clwb((const void *)q);
        } else if (have_clflushopt) {
            flush_clflushopt((const void *)q);
        } else {
            _mm_clflush((const void *)q);
        }
    }
    _mm_sfence();
}

int main(int argc, char **argv) {
    const char *devpath = "/dev/dax2.0";
    if (argc > 1) devpath = argv[1];

    detect_flush_features();

    struct sigaction sa = {0};
    sa.sa_handler = sigbus_handler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGBUS, &sa, NULL) != 0)
        die("sigaction(SIGBUS) failed: %s", strerror(errno));

    int fd = open(devpath, O_RDWR | O_CLOEXEC);
    if (fd < 0) die("Failed to open %s: %s", devpath, strerror(errno));

    uint64_t total = read_dax_size_from_sysfs(devpath);
    if (total == 0) die("Device size reported as 0; refusing to proceed.");

    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;

    const uint64_t chunk = (uint64_t)1 << 30; // 1 GiB
    uint64_t done = 0;

    // MAP_SYNC is irrelevant for /dev/dax; use plain MAP_SHARED
    int map_flags = MAP_SHARED;

    fprintf(stderr, "Zeroing %s (%" PRIu64 " bytes)\n", devpath, total);

    while (done < total) {
        if (got_sigbus) {
            close(fd);
            die("Received SIGBUS: mapping hit an invalid range. Aborting.");
        }

        uint64_t remain = total - done;
        size_t len = (size_t)((remain > chunk) ? chunk : remain);
        size_t align = (size_t)pagesz;
        size_t len_aligned = (len + (align - 1)) & ~(align - 1);

        void *addr = mmap(NULL, len_aligned, PROT_READ | PROT_WRITE, map_flags, fd, (off_t)done);
        if (addr == MAP_FAILED) {
            int e = errno;
            close(fd);
            die("mmap failed at offset=%" PRIu64 " len=%zu: %s", done, len_aligned, strerror(e));
        }

        memset(addr, 0, len_aligned);
        // Persist via cache-line flushes + SFENCE
        persist_range(addr, len_aligned);

        if (munmap(addr, len_aligned) != 0) {
            int e = errno;
            close(fd);
            die("munmap failed at offset=%" PRIu64 " len=%zu: %s", done, len_aligned, strerror(e));
        }

        done += len_aligned;
        fprintf(stderr, "\rProgress: %.2f%%", (100.0 * (double)done) / (double)total);
        fflush(stderr);
    }

    fprintf(stderr, "\nCompleted zeroing %s.\n", devpath);
    close(fd);
    return 0;
}
