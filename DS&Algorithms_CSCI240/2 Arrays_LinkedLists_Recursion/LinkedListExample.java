package DataStructures_Algorithms_CSCI240.copy;

/*  
Java Class: LinkedListExample
Author: Maayan Israel
Class: CSCI240
Date: 9/7/2022
Description: Singly Linked List with function of insertFront, insertRear, print, and remove Nodes.
I certify that the code below is my own work.
Exception(s): N/A
*/

public class LinkedListExample {
static private Node head = null;
static private Node tail = null;

public static void main(String[] args) {
    System.out.println("First three.");
    insertFront("Jonah");
    insertRear("Jimmy");
    insertRear("David");
    print();
    System.out.println();
    System.out.println("List of Ten");
    insertFront("Jonahthan");
    insertFront("Tuan");
    insertFront("Chang");
    insertFront("Baruch");
    insertRear("Gerardo");
    remove("Jonah");
    insertRear("Julia");
    insertRear("David");
    insertRear("Benjamin");
    print();
}

static private class Node {
    private String name;
    private Node next = null;

    public Node(String name) {
        this.name = name;
    }
}

static private void insertFront(String name) {
    Node newNode = new Node(name);
    if (head == null) {
        newNode.next = null;
        head = newNode;
        tail = newNode;
    } else {
        newNode.next = head;
        head = newNode;
    }
}

static private void insertRear(String name) {
    Node newNode = new Node(name);
    if (head == null) {
        head = newNode;
        tail = newNode;
    } else {
        tail.next = newNode;
        tail = newNode;
    }
}

static private void remove(String name) {
    Node current = head;
    Node prev = null;
    // If first node matches the name
    if (current.name.equalsIgnoreCase(name)) {
        head = current.next;
        return;
    }
    // Searches for the node with matching name
    while (current != null && !current.name.equalsIgnoreCase(name)) {
        prev = current;
        current = current.next;
    }
    // Removes the node with matching name
    if (current != null) {
        prev.next = current.next;
    }
}

static private void print() {
    Node current = head;
    while (current != null) {
        System.out.println(current.name);
        current = current.next;
    }
}
}
