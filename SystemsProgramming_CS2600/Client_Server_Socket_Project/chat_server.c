#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>

#define MaxConnects 5
#define BuffSize 256
#define ConversationLen 5
#define Host "localhost"

// FILE *chat_history;

/*
Print a message when a client disconnects from chat server.
end program when there are no more clients connected.
*/

struct Node
{
    int fd;
    struct Node *next;
    // char *username;
};

struct args_for_thread
{
    // int fd_Of_ChatHistory;
    int client_fd;
    // struct flock lock;
    struct Node *head;
} args_for_thread;

void report(const char *msg, int terminate)
{
    perror(msg);
    if (terminate)
        exit(EXIT_FAILURE);
}

void *handleCommunicationWithClient(void *args)
{
    struct args_for_thread *argsFor = args;
    char buffer[BuffSize];
    // char finalBuffToSend[BuffSize];
    struct Node *head = argsFor->head;
    int client_fd = argsFor->client_fd;

    while (1)
    {
        ssize_t bytes_read = read(client_fd, buffer, BuffSize - 1);
        if (bytes_read <= 0)
        {
            if (bytes_read < 0)
                perror("Error on read");
            break;
        }
        buffer[bytes_read] = '\0';

        if (strcmp(buffer, "exit") == 0)
        {
            struct Node *prev = head;
            struct Node *current = head->next;
            while (current != NULL)
            {
                if (current->fd == client_fd)
                {
                    prev->next = current->next;
                    free(current);
                    break;
                }
                prev = current;
                current = current->next;
            }
            printf("Client disconnected\n");
            break;
        }

        struct Node *temp = head->next;
        while (temp != NULL)
        {
            if (temp->fd != client_fd)
            {
                // strcat(finalBuffToSend, temp->username);
                // strcat(finalBuffToSend, ": ");
                // strcat(finalBuffToSend, buffer);
                // printf("%s", finalBuffToSend);
                write(temp->fd, buffer, bytes_read + 1);
            }

            temp = temp->next;
        }
    }

    close(client_fd);
    free(argsFor);
    return NULL;
}

int main()
{
    int portNo = 9876;
    // printf("Enter port number: ");
    // scanf("%d", &portNo);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        report("socket", 1);

    struct sockaddr_in saddr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = htonl(INADDR_ANY);
    saddr.sin_port = htons(portNo);

    if (bind(fd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0)
        report("bind", 1);

    if (listen(fd, MaxConnects) < 0)
        report("listen", 1);

    fprintf(stderr, "Listening on port %i\n", portNo);

    // chat_history = fopen("chat_history", "w");

    // struct flock lockThis;
    // lockThis.l_type = F_WRLCK;
    // lockThis.l_whence = SEEK_SET;
    // lockThis.l_start = 0;
    // lockThis.l_len = 0;
    // lockThis.l_pid = getpid();

    // int fd_chat_history;
    // if ((fd_chat_history = open("chat_history", O_RDWR, 0666)) < 0)
    // {
    //     perror("open");
    //     exit(1);
    // }
    struct Node *head = (struct Node *)malloc(sizeof(struct Node));
    int numClients = 0;

    while (1)
    {
        struct sockaddr_in caddr;
        int len = sizeof(caddr);

        int client_fd = accept(fd, (struct sockaddr *)&caddr, (socklen_t *)&len);
        if (client_fd < 0)
        {
            // report("accept", 0);
            // continue;
        }

        // char *clientUsername = malloc(50);
        // ssize_t username_len = read(client_fd, clientUsername, BuffSize - 1);
        // if (username_len <= 0)
        // {
        //     perror("Error reading username");
        //     close(client_fd);
        //     continue;
        // }
        // clientUsername[username_len] = '\0'; // Ensure null termination

        if (numClients == 0)
        {
            struct Node *newNode = (struct Node *)malloc(sizeof(struct Node));
            newNode->fd = client_fd;
            // newNode->username = clientUsername;
            head->next = newNode;
            numClients++;
        }
        else
        {
            struct Node *temp = head;
            while (temp->next != NULL)
            {
                temp = temp->next;
            }
            struct Node *newNode = (struct Node *)malloc(sizeof(struct Node));
            newNode->fd = client_fd;
            // newNode->username = clientUsername;
            temp->next = newNode;
        }

        struct args_for_thread *args = malloc(sizeof(struct args_for_thread));
        args->head = head;
        args->client_fd = client_fd;
        // args->fd_Of_ChatHistory = fd;
        // args->lock = lockThis;

        pthread_t thread;
        pthread_create(&thread, NULL, handleCommunicationWithClient, (void *)args);
        pthread_detach(thread);

        printf("Client connected\n");
    }

    return 0;
}