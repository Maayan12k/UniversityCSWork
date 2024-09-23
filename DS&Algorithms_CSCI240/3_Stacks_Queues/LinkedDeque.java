/*  Java Class: LinkedDeque
    Author: Zachariah Cano
    Class: CSCI 240
    Date:9/21/2022
    Description: A Deque with doubly linked Nodes.

    I certify that the code below is my own work.
*/

public class LinkedDeque {

    public static void main(String[] args) {
        LinkedDeque linked = new LinkedDeque();
        linked.addFirst("James");
        linked.addFirst("Paul");
        linked.addFirst("Gerardo");
        linked.addFirst("Jorge");
        linked.removeFirst();
        linked.removeLast();
        linked.print();
        System.out.println("# of Items: " + linked.getSize());

        linked.addLast("Maayan");
        linked.addLast("Justus");
        linked.addFirst("Juan");
        linked.print();
        System.out.println("# of Items: " + linked.getSize());
    }

    static private class Node {
        Node prev;
        Node next;
        String data;

        public Node(String data, Node prev, Node next) {
            this.data = data;
            this.prev = prev;
            this.next = next;
        }

        public Node getPrev() { return prev; }
        public Node getNext() { return next; }
        public String getData() { return data; }
        public void setPrev(Node newPrev) { prev = newPrev; }
        public void setNext(Node newNext) { next = newNext; }
    }

    Node head;
    Node tail;
    int count = 0;

    public LinkedDeque() {
        head = new Node(null, null, null);
        tail = new Node(null, head, null);
        head.setNext(tail);
    }

    public void addFirst(String data) {
        Node newNode = new Node(data, head, head.getNext());
        Node before = newNode.getPrev();
        Node after = newNode.getNext();
        before.setNext(newNode);
        after.setPrev(newNode);
        count++;
    }

    public void addLast(String data) {
        Node newNode = new Node(data, tail.getPrev(), tail);
        Node before = tail.getPrev();
        Node after = tail;
        before.setNext(newNode);
        after.setPrev(newNode);
        count++;
    }

    public void removeFirst() {
        Node before = head;
        Node after = head.getNext().getNext();
        before.setNext(after);
        after.setPrev(before);
        count--;
    }

    public void removeLast() {
        Node before = tail.getPrev().getPrev();
        Node after = tail;
        before.setNext(after);
        after.setPrev(before);
        count--;
    }

    private void print() {
        Node curr = head.next;
        while (curr != tail) {
            System.out.println(curr.data);
            curr = curr.getNext();
        }
    }

    private int getSize() {
        return count;
    }
}
