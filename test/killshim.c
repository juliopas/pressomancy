/* LD_PRELOAD shim for the SWMR crash tests (test/swmr_kill_harness.py): after arm(k),
 * the process SIGKILLs itself right before its k-th pwrite/pwrite64 (HDF5's sec2 driver
 * writes through pwrite). arm(0) disarms. Build: cc -shared -fPIC -O2 -o killshim.so killshim.c
 *
 * Why a C shim: one Python statement (a flush, a resize, a close) makes many pwrites inside
 * HDF5's C code, and Python can only act between bytecodes, so it cannot place a kill between
 * them. Random external kills are not reproducible and need thousands of runs to cover the
 * same points (SCRIPTS' kill_restart.py does that, at script level). `strace -e inject` needs
 * ptrace and one fresh process per kill point. h5py's Python file-object driver would expose
 * every write to Python, but every file open in pressomancy would have to be monkeypatched. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>

static long armed = 0, count = 0;

void arm(long k) { armed = k; count = 0; }

static void check(void) {
    if (armed > 0 && ++count >= armed) { armed = 0; kill(getpid(), SIGKILL); }
}

ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
    static ssize_t (*real)(int, const void *, size_t, off_t) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "pwrite");
    check();
    return real(fd, buf, n, off);
}

ssize_t pwrite64(int fd, const void *buf, size_t n, off_t off) {
    static ssize_t (*real)(int, const void *, size_t, off_t) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "pwrite64");
    check();
    return real(fd, buf, n, off);
}
