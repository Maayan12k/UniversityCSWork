import java.util.Scanner;

import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;

/**
 * Solution to the shortest wait time problem. Use dijkstra's algorithm to
 * find the shortest path between node 1 and node n.
 * and then use permutation logic to find the shortest wait time between node
 * 1 and node n. if there are 2 edges, then return max
 * of the two wait times. if there are 3 edges than return the min of the 4
 * possbile wait times. and so on
 */

public class ProgrammingProjectGraphAlgorithms {

    public static void main(String[] args) throws FileNotFoundException {
        // runTests();
        ProgrammingProjectGraphAlgorithms proj = new ProgrammingProjectGraphAlgorithms();
        System.out.println("\nShortest waiting time: " +
                proj.calculateShortestWaitingTime());

    }

    public static void runTests() throws FileNotFoundException {
        System.out.println("\nNegative files: ");
        String[] negativeOneFiles = { "./tests/false1.txt", "./tests/false2.txt", "./tests/false3.txt",
                "./tests/false4.txt", "./tests/false5.txt", "./tests/false6.txt", "./tests/false7.txt" };

        for (String file : negativeOneFiles) {
            ProgrammingProjectGraphAlgorithms proj = new ProgrammingProjectGraphAlgorithms(file);
            System.out.println("\nShortest waiting time: " +
                    proj.calculateShortestWaitingTime());
        }

        System.out.println("\n\n\n\n\n Positive files: \n\n");
        String[] positiveFiles = { "./tests/input1.txt", "./tests/input2.txt", "./tests/input3.txt",
                "./tests/input4.txt", "./tests/input5.txt" };

        for (String file : positiveFiles) {
            ProgrammingProjectGraphAlgorithms proj = new ProgrammingProjectGraphAlgorithms(file);
            System.out.println("\nShortest waiting time: " +
                    proj.calculateShortestWaitingTime());
        }
    }

    AdjacencyMapGraph<Integer, Integer> graph;
    int size;

    public ProgrammingProjectGraphAlgorithms() {
        graph = new AdjacencyMapGraph<>(false);
        Scanner scanner = new Scanner(System.in);

        System.out.println("Please enter the Graph information: ");

        String[] firstLine = scanner.nextLine().split(" ");
        size = Integer.parseInt(firstLine[0]);
        int numEdges = Integer.parseInt(firstLine[1]);

        for (int i = 1; i <= size; i++) {
            graph.insertVertex(i);
        }

        for (int i = 0; i < numEdges; i++) {
            String[] line = scanner.nextLine().split(" ");
            int source = Integer.parseInt(line[0]);
            Vertex<Integer> sourceVertex = findVertex(source);

            if (sourceVertex == null) {
                System.out.println("Source vertex not found. Not adding edge");
                continue;
            }

            int destination = Integer.parseInt(line[1]);
            Vertex<Integer> destinationVertex = findVertex(destination);

            if (destinationVertex == null) {
                System.out.println("Destination vertex not found. Not adding edge");
                continue;
            }

            int weight = Integer.parseInt(line[2]);
            graph.insertEdge(sourceVertex, destinationVertex, weight);
        }

        scanner.close();
    }

    /**
     * Constructor for ProgrammingProjectGraphAlgorithms.
     * Creates a graph from a file and handles boosted nodes input and non boosted
     * nodes input.
     *
     * @param filename
     * @throws FileNotFoundException
     */
    public ProgrammingProjectGraphAlgorithms(String filename) throws FileNotFoundException {
        graph = new AdjacencyMapGraph<>(false);
        File file = new File(filename);
        Scanner scanner = new Scanner(file);
        String[] firstLine = scanner.nextLine().split(" ");
        size = Integer.parseInt(firstLine[0]);

        for (int i = 1; i <= size; i++) {
            graph.insertVertex(i);
        }

        while (scanner.hasNextLine()) {
            String[] line = scanner.nextLine().split(" ");
            int source = Integer.parseInt(line[0]);
            Vertex<Integer> sourceVertex = findVertex(source);
            if (sourceVertex == null)
                continue;
            int destination = Integer.parseInt(line[1]);
            Vertex<Integer> destinationVertex = findVertex(destination);
            if (destinationVertex == null)
                continue;
            int weight = Integer.parseInt(line[2]);
            graph.insertEdge(sourceVertex, destinationVertex, weight);
        }

        scanner.close();
    }

    private Vertex<Integer> findVertex(int vertex) {
        for (Vertex<Integer> v : graph.vertices()) {
            if (v.getElement() == vertex) {
                return v;
            }
        }
        return null;
    }

    public int calculateShortestWaitingTime() {

        if (size == 1) {
            return 0;
        }

        Vertex<Integer> srcVertex = findVertex(1);
        Vertex<Integer> destVertex = findVertex(size);

        if (!graph.outgoingEdges(srcVertex).iterator().hasNext() ||
                !graph.outgoingEdges(destVertex).iterator().hasNext()) {
            return -1;
        }

        Map<Vertex<Integer>, List<Edge<Integer>>> shortestPathLengthsSrc = shortestPathLengthsWithPaths(srcVertex);
        Map<Vertex<Integer>, List<Edge<Integer>>> shortestPathLengthsDest = shortestPathLengthsWithPaths(destVertex);

        if (shortestPathLengthsSrc.get(destVertex) == null ||
                shortestPathLengthsDest.get(srcVertex) == null) {
            return -1;
        }

        Map<Integer, ArrayList<Integer>> mapSrc = new ProbeHashMap<>();
        Map<Integer, ArrayList<Integer>> mapDest = new ProbeHashMap<>();

        for (Vertex<Integer> v : shortestPathLengthsSrc.keySet()) {
            mapSrc.put(v.getElement(), new ArrayList<>());
            List<Edge<Integer>> path = shortestPathLengthsSrc.get(v);

            for (Edge<Integer> e : path) {
                mapSrc.get(v.getElement()).add(e.getElement());
            }

        }

        for (Vertex<Integer> v : shortestPathLengthsDest.keySet()) {
            mapDest.put(v.getElement(), new ArrayList<>());
            List<Edge<Integer>> path = shortestPathLengthsDest.get(v);

            for (Edge<Integer> e : path) {
                mapDest.get(v.getElement()).add(e.getElement());
            }

        }

        int min = Integer.MAX_VALUE;

        int index = 1;
        while (index <= size) {
            int max = 0;
            ArrayList<Integer> src = mapSrc.get(index);
            ArrayList<Integer> dest = mapDest.get(index);

            int srcSum = 0;
            for (int entry : src)
                srcSum += entry;

            int destSum = 0;
            for (int entry : dest)
                destSum += entry;

            max = Math.max(srcSum, destSum);

            if (max < min)
                min = max;

            index++;
        }

        return min;
    }

    /**
     * Computes shortest-path distances from src vertex to all reachable vertices of
     * g and stores the path to that node respectively via back pointer.
     *
     * This implementation uses a modified Dijkstra's algorithm.
     *
     * The edge's element is assumed to be its integral weight.
     */
    private Map<Vertex<Integer>, List<Edge<Integer>>> shortestPathLengthsWithPaths(Vertex<Integer> src) {
        // d.get(v) is upper bound on distance from src to v
        Map<Vertex<Integer>, Integer> d = new ProbeHashMap<>();

        // map reachable v to its d value
        Map<Vertex<Integer>, Integer> cloud = new ProbeHashMap<>();

        // pq will have vertices as elements, with d.get(v) as key
        AdaptablePriorityQueue<Integer, Vertex<Integer>> pq = new HeapAdaptablePriorityQueue<>();

        // maps from vertex to its pq locator
        Map<Vertex<Integer>, Entry<Integer, Vertex<Integer>>> pqTokens = new ProbeHashMap<>();

        // store the predecessor of each vertex to reconstruct the path
        Map<Vertex<Integer>, Vertex<Integer>> predecessor = new ProbeHashMap<>();

        // store the edge weights of the path
        Map<Vertex<Integer>, List<Edge<Integer>>> pathEdges = new ProbeHashMap<>();

        // for each vertex v of the graph, add an entry to the priority queue, with
        // the source having distance 0 and all others having infinite distance
        for (Vertex<Integer> v : graph.vertices()) {
            if (v == src) {
                d.put(v, 0);
                pathEdges.put(v, new ArrayList<>()); // start with an empty path for the source
            } else {
                d.put(v, Integer.MAX_VALUE);
            }
            pqTokens.put(v, pq.insert(d.get(v), v)); // save entry for future updates
        }

        // now begin adding reachable vertices to the cloud
        while (!pq.isEmpty()) {
            Entry<Integer, Vertex<Integer>> entry = pq.removeMin();
            int key = entry.getKey();
            Vertex<Integer> u = entry.getValue();
            cloud.put(u, key); // this is actual distance to u
            pqTokens.remove(u); // u is no longer in pq

            // check outgoing edges from u
            for (Edge<Integer> e : graph.outgoingEdges(u)) {
                Vertex<Integer> v = graph.opposite(u, e);
                if (cloud.get(v) == null) {
                    // perform relaxation step on edge (u,v)
                    int edgeWeight = e.getElement();
                    if (d.get(u) + edgeWeight < d.get(v) && pathEdges.get(u) != null) { // better path to v?
                        d.put(v, d.get(u) + edgeWeight); // update the distance
                        pq.replaceKey(pqTokens.get(v), d.get(v)); // update the pq entry

                        // update the predecessor and path edges
                        predecessor.put(v, u);
                        List<Edge<Integer>> newPath = new ArrayList<>(pathEdges.get(u)); // copy the current path
                        newPath.add(e); // add the new edge
                        pathEdges.put(v, newPath); // update pathEdges for v
                    }
                }
            }
        }

        return pathEdges; // return paths with the actual edge weights
    }

}
