import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.File;
/*
 * Name: Maayan Israel
 * Date: 2/28/2024
 * Class: CS 3010
 * Assignment: Project 1
 * Description: This class will implement the Guassian Elimination algorithm to solve a system of linear equations.
 */
import java.io.FileNotFoundException;

public class GuassianElimination {
    private double[][] coefficientMatrix;
    private double[] constantVector;
    private double[] solutionVector;
    private int size;
    double[] scaleVector;
    private ArrayList<Integer> rowsNotUsedAsPivotsYet = new ArrayList<>();
    private ArrayList<Integer> pivotsUsedInOrder = new ArrayList<>();
    private File file;

    public static void main(String[] args) throws FileNotFoundException {
        GuassianElimination ge = new GuassianElimination("input.txt");

    }

    // Constuctor that invoked the coefficient matrix from a stdin
    public GuassianElimination() {
        System.out.println("Enter the size of the matrix: ");
        Scanner scanner = new Scanner(System.in);
        size = scanner.nextInt();
        coefficientMatrix = new double[size][size];
        solutionVector = new double[size];
        constantVector = new double[size];

        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                System.out.println("Enter the value for the coefficient matrix at position (" + i + ", " + k + "): ");
                coefficientMatrix[i][k] = scanner.nextDouble();
            }
        }

        for (int i = 0; i < size; i++) {
            System.out.println("Enter the value for the constant vector at position (" + i + "): ");
            constantVector[i] = scanner.nextDouble();
        }

        for (int i = 0; i < size; i++)
            rowsNotUsedAsPivotsYet.add(i);

        scanner.close();

        System.out.println("\nCoefficient Matrix: " + Arrays.deepToString(coefficientMatrix));
        System.out.println("Constant Vector: " + Arrays.toString(constantVector) + "\n");

        solve();

        System.out.println("\nSolution Vector: " + Arrays.toString(solutionVector));
    }

    // Constuctor that invoked the coefficient matrix from a file.
    public GuassianElimination(String fileName) throws FileNotFoundException {
        file = new File(fileName);
        Scanner scanner = new Scanner(file);

        setSize(file);

        coefficientMatrix = new double[size][size];
        solutionVector = new double[size];
        constantVector = new double[size];
        int vectorIteration = 0;
        int currentIteration = 0;

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine().trim();
            int rowIteration = 0;
            String regex = "-*[0-9]+";
            Pattern patty = Pattern.compile(regex);
            Matcher matcher = patty.matcher(line);
            double value;

            for (int i = 0; i <= size; i++) {
                if (matcher.find()) {
                    value = Double.valueOf(matcher.group());
                    if (i == size) {
                        constantVector[vectorIteration] = value;
                        vectorIteration++;
                    } else {
                        coefficientMatrix[currentIteration][rowIteration] = value;
                        rowIteration++;
                    }
                }
            }
            currentIteration++;
        }
        scanner.close();
        scaleVector = calculateScaleVector();

        for (int i = 0; i < size; i++)
            rowsNotUsedAsPivotsYet.add(i);

        System.out.println();
        for (int l = 0; l < size; l++) {
            System.out.println(Arrays.toString(coefficientMatrix[l]));
        }
        System.out.println("CV: " + Arrays.toString(constantVector) + "\n");

        solve();
    }

    public void solve() {
        int pivotRow = -1;

        for (int i = 0; i < size; i++) {
            pivotRow = findPivotRow(i);
            pivotsUsedInOrder.add(pivotRow);
            rowsNotUsedAsPivotsYet.remove((Integer) pivotRow);

            for (int rowToBeAltered : rowsNotUsedAsPivotsYet) {
                double firstElementInRowToBeAltered = coefficientMatrix[rowToBeAltered][i];
                double firstElementInPivotRow = coefficientMatrix[pivotRow][i];
                for (int k = i; k <= size; k++) {
                    if (k < size) {

                        coefficientMatrix[rowToBeAltered][k] += -firstElementInRowToBeAltered
                                * coefficientMatrix[pivotRow][k] / firstElementInPivotRow;

                    } else if (k == size) {// constant vector alteration performed here.

                        constantVector[rowToBeAltered] += -firstElementInRowToBeAltered * constantVector[pivotRow]
                                / firstElementInPivotRow;

                    }

                }

            }
            System.out.println("Iteration: " + i);
            for (int l = 0; l < size; l++) {
                System.out.println(Arrays.toString(coefficientMatrix[l]));
            }
            System.out.println("Constant Vector: " + Arrays.toString(constantVector) + "\n");

        }

        solutionVector[pivotRow] = constantVector[pivotRow] / coefficientMatrix[pivotRow][size - 1];
        double tempSolution = 0;
        double divider = 0;
        for (int i = size - 2; i >= 0; i--) {
            pivotRow = pivotsUsedInOrder.get(i);
            tempSolution = constantVector[pivotRow];
            for (int k = i; k < size; k++) {
                if (k == i) {
                    divider = coefficientMatrix[pivotRow][k];
                } else {
                    tempSolution -= coefficientMatrix[pivotRow][k] * solutionVector[k];
                }

            }
            solutionVector[i] = tempSolution / divider;

        }
        System.out.println("Solution Vector: " + Arrays.toString(solutionVector));

    }

    public void setSize(File file) throws FileNotFoundException {
        Scanner scanner = new Scanner(file);
        String firstLine = scanner.nextLine();
        String regex = "-*[0-9]+";
        Pattern patty = Pattern.compile(regex);
        Matcher matcher = patty.matcher(firstLine);

        while (matcher.find())
            size++;

        size--;
        scanner.close();
    }

    public int findPivotRow(int currentColumn) {
        double max = 0;
        int pivotRow = 0;
        double ratio = 0;
        for (int row : rowsNotUsedAsPivotsYet) {
            ratio = Math.abs(coefficientMatrix[row][currentColumn] / scaleVector[row]);
            if (ratio > max) {
                max = ratio;
                pivotRow = row;
            }
        }
        return pivotRow;
    }

    public double[] calculateScaleVector() {
        double[] scaleVector = new double[size];
        double max = 0;
        for (int i = 0; i < size; i++) {
            max = Double.MIN_VALUE;
            for (int k = 0; k < size; k++) {
                if (Math.abs(coefficientMatrix[i][k]) > max) {
                    max = Math.abs(coefficientMatrix[i][k]);
                }
            }
            scaleVector[i] = max;
        }
        return scaleVector;
    }

}