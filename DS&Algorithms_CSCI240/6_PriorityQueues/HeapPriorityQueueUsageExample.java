public static void main(String[] args) {
    HeapPriorityQueue<Integer, Integer> list1 = new HeapPriorityQueue<>();
    HeapPriorityQueue<Integer, Integer> list2 = new HeapPriorityQueue<>(new
    DescendingIntegerComparator());
    System.out .println("minOfList1 minOflist2(Descending)");

    list1.insert(5, null); list2.insert(5, null);
    list1.insert(4, null); list2.insert(4, null); list1.insert(7, null);
    list2.insert(7, null); list1.insert(1, null); list2.insert(1, null);

    System.out .print(list1.min().getKey() + " "); list1.removeMin();
    System.out .println(list2.min().getKey()); list2.removeMin();
    list1.insert(3, null); list2.insert(3, null); list1.insert(6, null);
    list2.insert(6, null);
    System.out .print(list1.min().getKey()+ " "); list1.removeMin();
    System.out .println(list2.min().getKey()); list2.removeMin();
    System.out .print(list1.min().getKey()+ " "); list1.removeMin();
    System.out .println(list2.min().getKey()); list2.removeMin();
    list1.insert(8, null); list2.insert(8, null);
    System.out .print(list1.min().getKey()+ " "); 
    list1.removeMin();
    System.out .println(list2.min().getKey()); 
    list2.removeMin();

    list1.insert(2, null); list2.insert(2, null);

    System.out .print(list1.min().getKey()+ " "); list1.removeMin();
    System.out .println(list2.min().getKey()); list2.removeMin();
    System.out .print(list1.min().getKey()+ " "); list1.removeMin();
    System.out .println(list2.min().getKey()); list2.removeMin();
}