#include <sys/syscall.h>
#include <pthread.h>

typedef unsigned long cpu_set_t;
#define __NCPUBITS      (8 * sizeof (unsigned long))


#define CPU_SET(cpu, cpusetp) \
	        ((*(cpusetp)) |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ISSET(cpu, cpusetp) \
	        ((*(cpusetp)) & (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
	        memset((cpusetp), 0, sizeof(cpu_set_t))

inline static int
sched_setaffinity(pid_t pid, size_t len, cpu_set_t const * cpusetp)
{
	return syscall(__NR_sched_setaffinity, pid, len, cpusetp);
}

inline static int
sched_getaffinity(pid_t pid, size_t len, cpu_set_t const * cpusetp)
{
	return syscall(__NR_sched_getaffinity, pid, len, cpusetp);
}
