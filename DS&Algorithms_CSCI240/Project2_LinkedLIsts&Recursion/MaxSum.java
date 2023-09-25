package DataStructures_Algorithms_CSCI240.copy;

//Import the Random class to generate random numbers.
import java.util.Random;

public class MaxSum {
 // Static integer array initialized with some test data.
 static int[] list = {31,-41,59,26,-53,58,97,-93,-23,84};

 public static void main(String[] args){
     // Generate an array of random numbers of size 100.
     int[] list1 = generateRandomArray(100);

     // Record the start time to calculate elapsed time later.
     long startTime = System.nanoTime();

     // Calculate the maximum sum of the subarray using the provided algorithm.
     int sum = maxSum(list1);

     // Calculate time taken for the algorithm to complete.
     long timeElapsed = (System.nanoTime() - startTime)/1000000;
     System.out.println(sum);
     System.out.println("Program took " + timeElapsed + " milliseconds.");

     // Repeating the same process for an array of size 1000.
     int[] list2 = generateRandomArray(1000);
     long startTime1 = System.nanoTime();
     int sum1 = maxSum(list2);
     long timeElapsed1 = (System.nanoTime() - startTime1)/1000000;
     System.out.println(sum1);
     System.out.println("Program took " + timeElapsed1 + " milliseconds.");

     // Repeating the same process for an array of size 10000.
     int[] list3 = generateRandomArray(10000);
     long startTime2 = System.nanoTime();
     int sum2 = maxSum(list3);
     long timeElapsed2 = (System.nanoTime() - startTime2)/1000000;
     System.out.println(sum2);
     System.out.println("Program took " + timeElapsed2 + " milliseconds.");

     // Repeating the same process for an array of size 100000.
     int[] list4 = generateRandomArray(100000);
     long startTime3 = System.nanoTime();
     int sum3 = maxSum(list4);
     long timeElapsed3 = (System.nanoTime() - startTime3)/1000000;
     System.out.println(sum3);
     System.out.println("Program took " + timeElapsed3 + " milliseconds.");
 }

 // Algorithm to calculate the maximum sum of a subarray.
 static int maxSum(int[] array){
     // Initialize the maximum sum to the smallest integer value.
     int maxSum = Integer.MIN_VALUE;

     // Variable to keep track of the running sum of the current subarray.
     int testSum = 0;

     // Loop through each element in the array.
     for(int i = 0; i< array.length; i++){
         // Add the current element to the running sum.
         testSum += array[i];

         // If the running sum is greater than the maxSum, update maxSum.
         if(maxSum < testSum){
             maxSum = testSum;
         }

         // If running sum goes negative, reset it to 0.
         if(testSum<0){
             testSum = 0;
         }
     }

     // Return the maximum sum found.
     return maxSum;
 }

 // Helper function to generate an array of random integers of a given size.
 public static int[] generateRandomArray(int size){
     // Initialize an integer array of the given size.
     int[] list = new int[size];

     // Create a Random object to generate random numbers.
     Random rnd = new Random();

     // Populate the array with random integers between -1000 and 999.
     for(int i = 0; i< size; i++){
         list[i] = rnd.nextInt(2000) -1000;
     }

     // Return the populated array.
     return list;
 }
}

