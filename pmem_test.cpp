 #include <sys/types.h>
  #include <sys/stat.h>
              #include <fcntl.h>
              #include <stdio.h>
              #include <errno.h>
              #include <stdlib.h>
              #include <unistd.h>
              #include <string.h>
              #include <libpmem.h>

              /* using 4k of pmem for this example */
              #define PMEM_LEN 4096

              int
              main(int argc, char *argv[])
              {
                  int fd;
                  char *pmemaddr;
                  int is_pmem;

                  /* create a pmem file */
                  if ((fd = open("/pmem-fs/myfile",
                                       O_CREAT|O_RDWR, 0666)) < 0) {
                      perror("open");
                      exit(1);
                  }

                  /* allocate the pmem */
                  if ((errno = posix_fallocate(fd, 0, PMEM_LEN)) != 0) {
                         perror("posix_fallocate");
                      exit(1);
                  }

                  /* memory map it */
                  if ((pmemaddr = (char*) pmem_map(fd)) == NULL) {
                      perror("pmem_map");
                      exit(1);
                  }
                  close(fd);

                  /* determine if range is true pmem */
                  is_pmem = pmem_is_pmem(pmemaddr, PMEM_LEN);

                  /* store a string to the persistent memory */
                  strcpy(pmemaddr, "hello, persistent memory");

                  /* flush above strcpy to persistence */
                  if (is_pmem)
                      pmem_persist(pmemaddr, PMEM_LEN);
                  else
                      pmem_msync(pmemaddr, PMEM_LEN);
              }


