# Graph Algorithms - Shortest Wait Time

This project solves the **shortest wait time** problem using **Dijkstra's Algorithm** to find the shortest path in a graph.
It also uses **permutation logic** to calculate the shortest wait time between two nodes, 
factoring in different possible edge combinations along the path.

## Problem Description

- The goal is to find the shortest path from node 1 to node n in a graph.
- After finding the shortest path, the program calculates the wait time using permutation logic:
  - If the path has 2 edges, it returns the **maximum** wait time between them.
  - If the path has 3 edges, it calculates and returns the **minimum** of 4 possible wait times.
  - This logic extends to larger paths.

## Features

- Reads input graph data from standard input or a file.
- Implements Dijkstra's algorithm to compute the shortest path.
- Computes the shortest wait time by considering all possible wait time permutations along the path.
- Unit tests can be activated by uncommenting the `runTests();` line in the `main` method.

## Files

- `ProgrammingProjectGraphAlgorithms.java`: The main Java file implementing the algorithm.
- `tests/`: Contains test cases for verifying the program.

## Instructions

1. **To Run the Program:**
   ```bash
   java ProgrammingProjectGraphAlgorithms
   ```
   Then input the graph data when prompted.

2. **To Run Tests:**
   Uncomment the line `runTests();` in the `main` method and run the program.

3. **Graph Input Format:**
   The program expects the following input format:
   - The first line contains two integers: the number of vertices and the number of edges.
   - Each subsequent line defines an edge with three integers: the source vertex, the destination vertex, and the weight of the edge.

4. **Example Input:**
   ```
   5 6
   1 2 10
   1 3 20
   2 4 5
   3 4 15
   4 5 10
   3 5 30
   ```