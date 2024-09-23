//Method class inside of HeapPriorityQueue.java class
public static void main(String[] args) throws FileNotFoundException {
    HeapPriorityQueue<Integer, Integer> list1 = new HeapPriorityQueue<>();
    HeapPriorityQueue<Integer, Integer> list2 = new HeapPriorityQueue<>();
    File file = new File("small1k.txt");
    Scanner scanny = new Scanner(file);
    long startTime = System.nanoTime ();
    while (scanny.hasNextInt()) {
        int x = scanny.nextInt();
        list1.insert(x, null);
    }
    long endTime = System.nanoTime ();
    System.out .println("Runtime(Milliseconds) to sort 1,000 element text file: " +
    ((endTime - startTime) / 1000000));
    for (int i = 0; i < 995; i++) {
        if (i < 5) {
            System.out .print(list1.removeMin().getKey() + " ");
        } else
            list1.removeMin();
    }
    System.out .println();
    for (int i = 0; i < 5; i++) {
        System.out .print(list1.removeMin().getKey() + " ");
    }
    System.out .println();
    File file1 = new File("large100k.txt");
    Scanner scannyier = new Scanner(file1);
    startTime = System.nanoTime ();
    while (scannyier.hasNextInt()) {
        int x = scannyier.nextInt();
        list2.insert(x, null);
    }
    endTime = System.nanoTime ();
    System.out .println("Runtime(Milliseconds) to sort 100,000 element text file: " +
    ((endTime - startTime) / 1000000));
    for (int i = 0; i < 99995; i++) {
        if (i < 5) {
        System.out .print(list2.removeMin().getKey() + " ");
        } else
        list2.removeMin();
    }
    System.out .println();
    for (int i = 0; i < 5; i++) {
        System.out .print(list2.removeMin().getKey() + " ");
    }
}