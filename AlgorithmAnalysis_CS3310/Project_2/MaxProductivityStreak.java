import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MaxProductivityStreak {

    public MaxProductivityStreak() {
        Scanner scanny = new Scanner(System.in);

        while (true) {
            System.out.println("Enter the size of the array: ");
            int size = scanny.nextInt();
            int[] productivity = new int[size];

            System.out.println("Enter the elements of the array: ");
            for (int i = 0; i < size; i++) {
                productivity[i] = scanny.nextInt();
            }

            int[] result = maxSubarray(productivity, 0, productivity.length - 1);

            System.out.print("Max Productivity Streak(Divide & Conquer): [");
            for (int i = result[0]; i <= result[1]; i++) {
                System.out.print(productivity[i]);
                if (i < result[1]) {
                    System.out.print(", ");
                }
            }
            System.out.println(']');

            System.out.println("Max Productivity Streak(Dynamic Programming): " + maxProductivityStreak(productivity));
            System.out.println("Do you want to continue? (Y/N)");
            String response = scanny.next();
            if (response.equalsIgnoreCase("N")) {
                break;
            }
        }

        scanny.close();
    }

    public static List<Integer> maxProductivityStreak(int[] productivity) {
        int maxSum = productivity[0];
        int currentSum = productivity[0];
        int start = 0, end = 0, tempStart = 0;

        for (int i = 1; i < productivity.length; i++) {
            if (currentSum + productivity[i] < productivity[i]) {
                currentSum = productivity[i];
                tempStart = i;
            } else {
                currentSum += productivity[i];
            }

            if (currentSum > maxSum) {
                maxSum = currentSum;
                start = tempStart;
                end = i;
            }
        }

        List<Integer> result = new ArrayList<>();
        for (int i = start; i <= end; i++) {
            result.add(productivity[i]);
        }
        return result;
    }

    private static int[] maxCrossingSubarray(int[] arr, int left, int mid, int right) {
        int leftSum = Integer.MIN_VALUE;
        int sum = 0;
        int maxLeft = mid;
        for (int i = mid; i >= left; i--) {
            sum += arr[i];
            if (sum > leftSum) {
                leftSum = sum;
                maxLeft = i;
            }
        }

        int rightSum = Integer.MIN_VALUE;
        sum = 0;
        int maxRight = mid;
        for (int i = mid + 1; i <= right; i++) {
            sum += arr[i];
            if (sum > rightSum) {
                rightSum = sum;
                maxRight = i;
            }
        }

        return new int[] { maxLeft, maxRight, leftSum + rightSum };
    }

    private static int[] maxSubarray(int[] arr, int left, int right) {
        if (left == right) {
            return new int[] { left, right, arr[left] };
        }

        int mid = (left + right) / 2;

        int[] leftResult = maxSubarray(arr, left, mid);
        int[] rightResult = maxSubarray(arr, mid + 1, right);
        int[] crossResult = maxCrossingSubarray(arr, left, mid, right);

        if (leftResult[2] >= rightResult[2] && leftResult[2] >= crossResult[2]) {
            return leftResult;
        } else if (rightResult[2] >= leftResult[2] && rightResult[2] >= crossResult[2]) {
            return rightResult;
        } else {
            return crossResult;
        }
    }

    public static void main(String[] args) {
        new MaxProductivityStreak();
    }
}
