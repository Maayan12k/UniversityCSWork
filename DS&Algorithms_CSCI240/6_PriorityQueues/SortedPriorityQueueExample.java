public class SortedPriorityQueueExample {

    public static void main(String[] args) throws IOException{
        SortedPriorityQueue<Integer,Integer> list1 = new SortedPriorityQueue<>(new DescendingIntegerComparator());
        SortedPriorityQueue<Integer,Integer> list2 = new SortedPriorityQueue<>();
        System.out .println("minOfList minOflist2");
        list1.insert(5, null); list2.insert(5, null);
        list1.insert(4, null); list2.insert(4, null);
        list1.insert(7, null); list2.insert(7, null);
        list1.insert(1, null); list2.insert(1, null);
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin(); list2.removeMin();
        list1.insert(3, null); list2.insert(3, null);
        list1.insert(6, null); list2.insert(6, null);
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin(); list2.removeMin();
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin();
        list2.removeMin();
        list1.insert(8, null); list2.insert(8, null);
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin();
        list2.removeMin();
        list1.insert(2, null); list2.insert(2, null);
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin();
        list2.removeMin();
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        list1.removeMin();
        list2.removeMin();
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
        System.out .print(list1.min().getKey() + " ");
        System.out .println(list2.min().getKey());
    }
}