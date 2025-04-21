#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include "sock.h"

void report(const char *msg, int terminate)
{
  perror(msg);
  if (terminate)
    exit(-1); /* failure */
}

struct args_for_thread
{
  int server_fd;
} args_for_thread;

void *handleCommunicationWithClient(void *args)
{
  struct args_for_thread *args_for_thread = (struct args_for_thread *)args;
  int server_fd = args_for_thread->server_fd;
  /* a server traditionally listens indefinitely */
  struct sockaddr_in caddr; /* client address */
  int len = sizeof(caddr);  /* address length could change */

  int client_fd = accept(server_fd, (struct sockaddr *)&caddr, &len); /* accept blocks */
  if (client_fd < 0)
  {
    report("accept", 0); /* don't terminated, though there's a problem */
    // continue;
  }
  while (1)
  {

    /* read from client */
    // START CODE HERE
    char buffer[256];
    int n = read(client_fd, buffer, 255);
    printf("Client: %s\n", buffer);
    write(client_fd, buffer, n);

    if (strcmp(buffer, "exit") == 0)
    {
      close(client_fd);
      break;
    }
  }
}

int main()
{
  int fd = socket(AF_INET,     /* network versus AF_LOCAL */
                  SOCK_STREAM, /* reliable, bidirectional: TCP */
                  0);          /* system picks underlying protocol */
  if (fd < 0)
    report("socket", 1); /* terminate */

  /* bind the server's local address in memory */
  struct sockaddr_in saddr;
  memset(&saddr, 0, sizeof(saddr));          /* clear the bytes */
  saddr.sin_family = AF_INET;                /* versus AF_LOCAL */
  saddr.sin_addr.s_addr = htonl(INADDR_ANY); /* host-to-network endian */
  saddr.sin_port = htons(PortNumber);        /* for listening */

  if (bind(fd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
    report("bind", 1); /* terminate */

  /* listen to the socket */
  if (listen(fd, MaxConnects) < 0) /* listen for clients, up to MaxConnects */
    report("listen", 1);           /* terminate */

  fprintf(stderr, "Listening on port %i for clients...\n", PortNumber);

  while (1)
  {
    /* a server traditionally listens indefinitely */
    struct sockaddr_in caddr; /* client address */
    int len = sizeof(caddr);  /* address length could change */

    int client_fd = accept(fd, (struct sockaddr *)&caddr, &len); /* accept blocks */
  }
  return 0;
}
