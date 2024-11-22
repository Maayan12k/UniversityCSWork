import java.util.Scanner;

public class DivideAndConquerStreak {

    public DivideAndConquerStreak() {
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

            System.out.println("Do you want to continue? (Y/N)");
            String response = scanny.next();
            if (response.equalsIgnoreCase("N")) {
                break;
            }
        }

        scanny.close();
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
        new DivideAndConquerStreak();
    }
}
