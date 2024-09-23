package DataStructures_Algorithms_CSCI240.copy;

/*  
Java Class: TestArray
Author: Maayan Israel
Class: CSCI240
Date: 8/31/2022
Description: An array of strings of names with functions to add, remove, and print.
I certify that the code below is my own work.
Exception(s): N/A
*/

public class TestArray {
static int numOfItems = 0;
static String[] array = new String[10];

public static void main(String[] args) {
    // Inserts at the front of the array
    insertFront("Jonny");
    insertFront("Jorge");
    insertFront("Tuan");
    System.out.println("first three");
    print();

    System.out.println("ten list");
    insertFront("Desmond");
    insertRear("Maayan");
    insertFront("duke");
    insertFront("Barry");
    insertFront("Jacob");
    remove("duke");
    insertFront("Issac");
    insertFront("kent");
    insertRear("james");
    remove("Maayan");
    print();
}

public static void insertFront(String name) {
    // Check if array is empty, set first element
    if (numOfItems == 0) {
        array[0] = name;
        numOfItems++;
        return;
    }

    // If array is full, removes last element and shifts all to the right
    if (numOfItems == 10) {
        for (int i = 9; i > 0; i--) {
            array[i] = array[i - 1];
        }
        array[0] = name;
        return;
    }

    // For a partially filled array
    if (numOfItems < 10 && numOfItems > 0) {
        for (int i = numOfItems; i > 0; i--) {
            array[i] = array[i - 1];
        }
        array[0] = name;
        numOfItems++;
    }
}

public static void insertRear(String name) {
    // Inserts first element if array is empty
    if (numOfItems == 0) {
        array[0] = name;
        numOfItems++;
    }
    // If array is full, removes first element and adds new element to the end
    else if (numOfItems == 10) {
        for (int i = 0; i < 9; i++) {
            array[i] = array[i + 1];
        }
        array[9] = name;
    } 
    // Inserts at the rear if there's space
    else if (numOfItems > 0 && numOfItems < 10) {
        array[numOfItems] = name;
        numOfItems++;
    }
}

public static void remove(String name) {
    // Prints a message if the list is empty
    if (numOfItems == 0) {
        System.out.println("List is empty");
    } else {
        for (int i = 0; i < numOfItems; i++) {
            // Checks for a match
            if (array[i].equalsIgnoreCase(name)) {
                // If last element is matched, sets it to null
                if (i == 9) {
                    array[i] = null;
                    numOfItems--;
                    return;
                }
                // Shifts elements left to remove the matched one
                for (int j = i; j < numOfItems - 1; j++) {
                    array[j] = array[j + 1];
                    array[j + 1] = null;
                }
                numOfItems--;
                break;
            }
        }
    }
}

public static void print() {
    // Loops through array and prints each element
    for (int i = 0; i < numOfItems; i++) {
        System.out.println("Element " + (i + 1) + ": " + array[i]);
    }
}
}

