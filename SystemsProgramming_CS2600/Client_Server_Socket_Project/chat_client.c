#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <pthread.h>

#define MaxConnects 5
#define BuffSize 256
#define ConversationLen 5
#define Host "localhost"

int portNumber;
char username[BuffSize];

/*
Prompt client for username, save username locally to this program but send it to server
so that it can be stored in the linkedlist and displayed in chat history and messages to other clients.
*/

struct args_for_thread
{
    // int fd_Of_ChatHistory;
    int sockfd;
    // struct flock lock;
    struct Node *head;
} args_for_thread;

void *listenToServer(void *arg)
{
    int sockfd = *(int *)arg;
    char buffer[BuffSize];

    while (1)
    {
        ssize_t bytes_read = read(sockfd, buffer, BuffSize - 1);
        if (bytes_read < 0)
        {
            perror("Error on read");
            break;
        }
        buffer[bytes_read] = '\0';
        printf("\n%s\n", buffer);
        printf("%s: ", username);
        fflush(stdout); // Ensure that the prompt is shown immediately
    }
    return NULL;
}

int main()
{
    portNumber = 9876;

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        perror("socket");
        return 1;
    }

    struct hostent *hptr = gethostbyname(Host);
    if (!hptr)
    {
        perror("gethostbyname");
        return 1;
    }

    if (hptr->h_addrtype != AF_INET)
    {
        perror("bad address family");
        return 1;
    }

    struct sockaddr_in saddr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = ((struct in_addr *)hptr->h_addr)->s_addr;
    saddr.sin_port = htons(portNumber);

    if (connect(sockfd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
    {
        perror("connect");
        return 1;
    }

    printf("Enter your username: ");

    fgets(username, BuffSize, stdin);
    username[strcspn(username, "\n")] = 0;
    // write(sockfd, username, strlen(username) + 1);

    pthread_t listenThread;
    struct args_for_thread *args = malloc(sizeof(struct args_for_thread));
    args->sockfd = sockfd;
    pthread_create(&listenThread, NULL, listenToServer, (void *)args);

    while (1)
    {
        char buffer[BuffSize];
        printf("%s: ", username);
        fgets(buffer, BuffSize, stdin);
        buffer[strcspn(buffer, "\n")] = 0;
        write(sockfd, buffer, strlen(buffer) + 1);

        if (strcmp(buffer, "exit") == 0)
        {
            break;
        }
    }

    return 0;
}