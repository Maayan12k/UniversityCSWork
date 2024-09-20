#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_THREADS 2
#define ARRAY_SIZE 5000000

typedef struct Range
{
  int start;
  int end;
} Range;

double double_array[ARRAY_SIZE];

void *calculate_average(void *args)
{
  Range *range = (Range *)args;
  int start = range->start;
  int end = range->end;
  double *sum = malloc(sizeof(double)); // Allocate memory for sum dynamically
  *sum = 0.0;
  for (int i = start; i < end; i++)
  {
    *sum += double_array[i];
  }
  pthread_exit((void *)sum);
}

double calculate_average_single_thread()
{
  double sum = 0.0;
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    sum += double_array[i];
  }
  return sum / ARRAY_SIZE;
}

int main()
{
  double time_taken, average;
  clock_t t;
  Range ranges[NUM_THREADS];
  pthread_t threads[NUM_THREADS];

  // Create Random array
  srand(time(0));
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    double_array[i] = rand() % 1000;
  }

  // Linear Calculation
  t = clock();
  average = calculate_average_single_thread();
  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC;
  printf("Average: %f, Single Thread Time: %f\n", average, time_taken);

  t = clock();
  // Create the threads and assign ranges
  for (int i = 0; i < NUM_THREADS; i++)
  {
    ranges[i].start = i * (ARRAY_SIZE / NUM_THREADS);
    ranges[i].end = (i + 1) * (ARRAY_SIZE / NUM_THREADS);

    pthread_create(&threads[i], NULL, calculate_average, (void *)&ranges[i]);
  }

  // Join and collect averages
  average = 0.0;
  for (int i = 0; i < NUM_THREADS; i++)
  {
    void *thread_sum = malloc(sizeof(double));
    pthread_join(threads[i], &thread_sum);
    average += *(double *)thread_sum;
    // printf("Thread %d: %f\n", i, *(double *)thread_sum);
    free(thread_sum);
  }
  average = average / ARRAY_SIZE;
  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC;
  printf("Average: %f, Multithread Time: %f\n", average, time_taken);
}
