package DataStructures_Algorithms_CSCI240;

/*
 * This class is designed to demonstrate the Birthday Paradox by simulating
 * random birthdays and checking for collisions (same birthdays) among people.
 */
public class BirthdayParadox {

    public static void main(String args[]) {
        birthdayParadox();
    }

    /**
     * Checks if all values in the array are distinct.
     *
     * @param array Array of integer values
     * @return true if all elements are unique, otherwise false
     */
    public static boolean isDistinct(int[] array) {
        for (int i = 0; i < array.length; i++) {
            for (int n = i + 1; n < array.length; n++) {
                if (array[i] == array[n]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Simulates the Birthday Paradox for different group sizes and prints the results.
     * It increments the group size in steps of 5, from 5 to 100.
     * For each size, it simulates random birthdays and checks for collisions.
     */
    public static void birthdayParadox() {
        for (int n = 5; n <= 100; n += 5) {
            int count = 0;
            int[] array = new int[n];

            for (int i = 1; i <= 10; i++) {
                for (int k = 1; k < n; k++) {
                    array[k - 1] = (int) (Math.random() * 365) + 1;
                }
                if (!isDistinct(array)) {
                    count++;
                }
            }
            System.out.println(n + "    " + count);
        }
    }
}
