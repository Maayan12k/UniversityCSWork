import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DynamicProgrammingStreak {

    public DynamicProgrammingStreak() {
        Scanner scanny = new Scanner(System.in);

        while (true) {
            System.out.println("Enter the size of the array: ");
            int size = scanny.nextInt();
            int[] productivity = new int[size];

            System.out.println("Enter the elements of the array: ");
            for (int i = 0; i < size; i++) {
                productivity[i] = scanny.nextInt();
            }

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

    public static void main(String[] args) {
        new DynamicProgrammingStreak();
    }
}
