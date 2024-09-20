// modem.c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define num_networks 10

typedef enum
{
  kData,
  kWifi,
  kUnknown
} MEDIUM;

typedef struct
{
  char *network_name;
  int signal_strength;
  MEDIUM connection_medium;
  bool password_saved;
} network;

network *cached_networks;
int num_networks;

void ChooseNetwork(char *network)
{
  printf("Network chosen: %s\n", network);
}

MEDIUM ConvertIntToMedium(int int_medium)
{
  if (int_medium == 1)
  {
    return kData;
  }
  else if (int_medium == 2)
  {
    return kWifi;
  }
  else
  {
    return kUnknown;
  }
}

int countNumNetworks()
{
  FILE *fp = fopen("experiment_data", "r");
  int count = 0;
  char c = getc(fp);
  while (c != EOF)
  {
    if (c == '\n')
    {
      count++;
    }
    c = fgetc(fp);
  }
  fclose(fp);
  return (count + 1) / 2;
}

int countNetworksPasswordSaved(int numberOfNeworks)
{
  FILE *fp = fopen("experiment_data", "r");
  int temp = numberOfNeworks;
  char network_name[20];
  int medium;
  int signal_strength;
  char password_saved[5];
  for (int i = 0; i < numberOfNeworks; i++)
  {
    fscanf(fp, "%s", network_name);
    fscanf(fp, "%d %d %s", &medium, &signal_strength, password_saved);
    if (strcmp(password_saved, "false") == 0)
    {
      temp--;
    }
  }
  fclose(fp);
  return temp;
}

/**
 * TODO: This function is buggy and not complete
 *
 * We should first fix this function and then work on fixing ScanNetworksV2().
 * The fixes found in this function will help determine how to fix V2.
 */
void ScanNetworks()
{
  num_networks = countNumNetworks();
  cached_networks = (network *)malloc(num_networks * sizeof(network));

  FILE *fp = fopen("experiment_data", "r");
  char temp_name[20];
  int medium;
  int signal_strength;
  char password_saved[5];

  for (int i = 0; i < num_networks; i++)
  {
    fscanf(fp, "%s", temp_name);
    fscanf(fp, "%d %d %s", &medium, &signal_strength, password_saved);
    cached_networks[i].network_name = strdup(temp_name);
    cached_networks[i].signal_strength = signal_strength;
    cached_networks[i].connection_medium = ConvertIntToMedium(medium);

    if (strcmp(password_saved, "false") == 0)
    {
      cached_networks[i].password_saved = false;
    }
    else
    {
      cached_networks[i].password_saved = true;
    }
    printf("%s\n", cached_networks[i].network_name);
  }
  fclose(fp);
}

/**
 * This function early-exits from networks that we don't already have access
 * to. This way we can still scan for 5 networks, but we won't display/attempt
 * to make a decision vs networks we don't have the password for.
 *
 * TODO: This function is buggy and not complete
 */
void ScanNetworksV2()
{
  num_networks = countNumNetworks();
  int sizeOfPasswordSavedCachedNetworks = countNetworksPasswordSaved(num_networks);
  cached_networks = (network *)malloc(sizeOfPasswordSavedCachedNetworks * sizeof(network));
  FILE *fp = fopen("experiment_data", "r");
  char network_name[20];
  int signal_strength;
  int medium;
  char password_saved[5];
  bool password = false;
  int i = 0;
  int k = 0;

  while (i < num_networks)
  {
    fscanf(fp, "%s", network_name);
    fscanf(fp, "%d %d %s", &medium, &signal_strength, password_saved);

    if (strcmp(password_saved, "false") == 0)
    {
      password = false;
    }
    else
    {
      password = true;
    }

    // Only cache networks we can connect to
    if (password)
    {
      cached_networks[k].network_name = strdup(network_name);
      cached_networks[k].connection_medium = ConvertIntToMedium(medium);
      cached_networks[k].signal_strength = signal_strength;
      cached_networks[k].password_saved = password;
      printf("%s\n", cached_networks[k].network_name);
      k++;
    }
    i++;
  }
  num_networks = sizeOfPasswordSavedCachedNetworks;
  fclose(fp);
}

void ScanNetworksV3()
{
  // TO make ScanNetworksV2 even better I would cache the networks we have the password for and meet our search criteria(i.e., Wi-Fi, Data, either)
  // but also sort them by signal strength. This way we can easily choose the best network to connect to, quickly.
}

/**
 * This function selects what network we should connect to based on certain
 * criteria.
 *
 * 1. We should only choose networks that we can connect to
 * 2. We should only connect to networks based on what criteria we want
 *    (i.e., Wi-Fi, Data, either).
 * 3. We should pick the network with the highest signal strength
 *
 * criteria    String denoting "wifi", "data", or "greedy"
 * return      String of best network_name
 */
char *DetermineNetwork(char *criteria)
{
  // Iterate through cached_networks to choose the best network
  int best_signal_strength = 0;
  int best_network_index;
  int criteria_int = strcmp(criteria, "data") == 0 ? 0 : strcmp(criteria, "wifi") == 0 ? 1
                                                                                       : 2;

  for (int i = 0; i < num_networks; i++)
  {
    // printf("Network: %s\n", cached_networks[i].network_name);
    if (cached_networks[i].connection_medium == 2 || cached_networks[i].password_saved == false || (cached_networks[i].connection_medium != criteria_int && criteria_int != 2))
    {
      continue;
    }
    else if (cached_networks[i].signal_strength > best_signal_strength)
    {
      best_signal_strength = cached_networks[i].signal_strength;
      best_network_index = i;
    }
  }
  return cached_networks[best_network_index].network_name;
}

int main(int argc, char *argv[])
{
  if (argc != 2 || (strcmp(argv[1], "wifi") != 0 && strcmp(argv[1], "data") != 0 && strcmp(argv[1], "greedy") != 0))
  {
    printf("Incorrect command argument. Please pass in wifi, data, or greedy");
    return 1;
  }

  printf("Starting up modem...\n");
  printf("Scanning nearby network connections...\n");
  ScanNetworks();

  printf("Networks cached. Now determining network to connect...\n");
  printf("Connection Medium Criteria: %s\n", argv[1]);
  ChooseNetwork(DetermineNetwork(argv[1]));

  return 0;
}
