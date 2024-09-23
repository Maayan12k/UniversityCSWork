package DataStructures_Algorithms_CSCI240.copy;

/*  
Java Class: LinearFibonacci
Author: Maayan Israel
Class: CSCI240
Date: 9/7/2022
Description: Function that returns Fibonacci number and its previous value.
I certify that the code below is my own work.
Exception(s): N/A
*/

public class LinearFibonacci {
public static void main(String[] args) {
    int[] list = fibonacci(5);
    System.out.println("[" + list[0] + "]");
    int[] list1 = fibonacci(14);
    System.out.println("[" + list1[0] + "]");
}

public static int[] fibonacci(int n) {
    int[] result = new int[2];
    if (n == 0) {
        result[0] = 0;
        result[1] = 0;
        return result;
    } else if (n == 1) {
        result[0] = 1;
        result[1] = 0;
        return result;
    } else {
        int a = 0, b = 1, c;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        result[0] = b;
        result[1] = a;
        return result;
    }
}
}

