import java.io.File;
import java.io.FileNotFoundException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class JacobiGaussSeidel {

    private double[][] coefficientMatrix;
    private double[] constantVector;
    double[] previousIterationBuff;
    private int size;
    private double error;
    private File file;

    public static void main(String[] args) throws FileNotFoundException {
        Scanner scanny = new Scanner(System.in);
        int option;
        System.out.println("\nJacobi/Gauss-Seidel Calculator");
        System.out.println("Please make a selection, enter '1' for option 1 or '2' for option 2.");
        System.out.println("1) Manual input");
        System.out.println("2) Input file");
        System.out.print("Enter your choice: ");
        option = scanny.nextInt();

        while (!(option == 1) && !(option == 2)) {
            System.out.println("Please make a correct selection: ");
            option = scanny.nextInt();
        }

        if (option == 1) {
            new JacobiGaussSeidel();
        } else if (option == 2) {
            System.out.print("Please enter the name of a file in the current directory or absolute Pathname: ");
            String name = scanny.next();
            System.out.println();

            Pattern patty = Pattern.compile(".*\\..*");
            Matcher matcher = patty.matcher(name);

            if (matcher.find()) {
                new JacobiGaussSeidel(matcher.group());
            } else {
                while (!matcher.find()) {
                    System.out.println("Please enter a valid fileName:   ");
                    name = scanny.next();
                    matcher = patty.matcher(name);
                }
                new JacobiGaussSeidel(matcher.group());
            }

        }
        scanny.close();
    }

    // Constuctor that invoked the coefficient matrix from a stdin
    public JacobiGaussSeidel() {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the size of the matrix: ");
        size = scanner.nextInt();
        previousIterationBuff = new double[size];
        System.out.print("Enter the error of the solution: ");
        error = scanner.nextDouble();

        coefficientMatrix = new double[size][size];
        constantVector = new double[size];

        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                System.out.print("Enter the value for the coefficient matrix at position (" + i + ", " + k + "): ");
                coefficientMatrix[i][k] = scanner.nextDouble();
            }
        }

        for (int i = 0; i < size; i++) {
            System.out.print("Enter the value for the constant vector at position (" + i + "): ");
            constantVector[i] = scanner.nextDouble();
        }

        System.out.println("\nPlease enter the initial guess for the solution vector: ");
        for (int i = 0; i < size; i++) {
            System.out.print("Enter the value for the solution vector at position (" + i + "): ");
            previousIterationBuff[i] = scanner.nextDouble();
        }

        scanner.close();

        System.out.println("\nInput Coefficient Matrix:");
        for (int l = 0; l < size; l++) {
            System.out.println(Arrays.toString(coefficientMatrix[l]));
        }
        System.out.println("CV: " + Arrays.toString(constantVector) + "\n");

        System.out.println("\nJacobi:");
        solveJacobi();
        System.out.println("\nGauss-Seidel:");
        solveGaussSeidel();
        System.out.println();
    }

    // Constuctor that invoked the coefficient matrix from a file.
    public JacobiGaussSeidel(String fileName) throws FileNotFoundException {
        file = new File(fileName);
        Scanner scanner = new Scanner(file);
        Scanner scanner2 = new Scanner(System.in);

        System.out.print("Enter the error of the solution: ");
        error = scanner2.nextDouble();
        System.out.println();

        setSize(file);
        previousIterationBuff = new double[size];

        coefficientMatrix = new double[size][size];
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

        System.out.println("\nInput Coefficient Matrix:");
        for (int l = 0; l < size; l++) {
            System.out.println(Arrays.toString(coefficientMatrix[l]));
        }
        System.out.println("CV: " + Arrays.toString(constantVector) + "\n");

        System.out.println("Please enter the initial guess for the solution vector: ");
        for (int i = 0; i < size; i++) {
            System.out.print("Enter the value for the solution vector at position (" + i + "): ");
            previousIterationBuff[i] = scanner2.nextDouble();
        }
        scanner2.close();

        System.out.println("\nJacobi:");
        solveJacobi();
        System.out.println("\nGauss-Seidel:");
        solveGaussSeidel();
        System.out.println();

    }

    public void solveJacobi() {

        double[] currentBuff = new double[size];
        double tempSum = 0;
        double currentError = Double.MAX_VALUE;
        int numberOfIterations = 0;

        while (currentError > error) {
            if (numberOfIterations == 50) {
                System.out.println("The solution did not converge after 50 iterations.");
                break;
            }
            for (int i = 0; i < size; i++) { // this loop is for the currentBuff[i]
                for (int j = 0; j < size; j++) // this loop is for the row to calculate tempSum
                    if (j != i)
                        tempSum = new BigDecimal(tempSum - (coefficientMatrix[i][j] * previousIterationBuff[j]))
                                .setScale(15, RoundingMode.HALF_UP).doubleValue();

                tempSum = new BigDecimal((tempSum + constantVector[i]) / coefficientMatrix[i][i])
                        .setScale(15, RoundingMode.HALF_UP).doubleValue();
                currentBuff[i] = tempSum;
                tempSum = 0;
            }

            System.out.print("Iteration " + (numberOfIterations + 1) + ": " + "[");
            for (int i = 0; i < size; i++) {
                if (i == size - 1)
                    System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue());
                else
                    System.out.print(
                            new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue() + ", ");
            }
            System.out.print("]");

            currentError = new BigDecimal(calculateError(previousIterationBuff, currentBuff))
                    .setScale(15, RoundingMode.HALF_UP).doubleValue();

            System.out
                    .println(" Error: " + new BigDecimal(currentError).setScale(7, RoundingMode.HALF_UP).doubleValue());

            previousIterationBuff = Arrays.copyOf(currentBuff, size);
            numberOfIterations++;
        }

        System.out.print("Solution Vector: [");
        for (int i = 0; i < size; i++) {
            if (i == size - 1)
                System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue());
            else
                System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue() + ", ");
        }
        System.out.println("]");
        System.out.println("Number of Iterations: " + numberOfIterations);
    }

    public void solveGaussSeidel() {
        double[] previousIterationBuff = new double[size];
        double[] currentBuff = new double[size];
        double tempSum = 0;
        double currentError = Double.MAX_VALUE;
        int numberOfIterations = 0;

        while (currentError > error) {
            if (numberOfIterations == 50) {
                System.out.println("The solution did not converge after 50 iterations.");
                break;
            }
            for (int i = 0; i < size; i++) { // this loop is for the currentBuff[i]
                for (int j = 0; j < size; j++) // this loop is for the row to calculate tempSum
                    if (j != i) {
                        if (j < i)
                            tempSum = new BigDecimal(tempSum - (coefficientMatrix[i][j] * currentBuff[j]))
                                    .setScale(15, RoundingMode.HALF_UP).doubleValue();
                        else
                            tempSum = new BigDecimal(tempSum - (coefficientMatrix[i][j] * previousIterationBuff[j]))
                                    .setScale(15, RoundingMode.HALF_UP).doubleValue();
                    }

                tempSum = new BigDecimal((tempSum + constantVector[i]) / coefficientMatrix[i][i])
                        .setScale(15, RoundingMode.HALF_UP).doubleValue();
                currentBuff[i] = tempSum;
                tempSum = 0;
            }

            System.out.print("Iteration " + (numberOfIterations + 1) + ": " + "[");
            for (int i = 0; i < size; i++) {
                if (i == size - 1)
                    System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue());
                else
                    System.out.print(
                            new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue() + ", ");
            }
            System.out.print("]");

            currentError = new BigDecimal(calculateError(previousIterationBuff, currentBuff))
                    .setScale(15, RoundingMode.HALF_UP).doubleValue();

            System.out
                    .println(" Error: " + new BigDecimal(currentError).setScale(7, RoundingMode.HALF_UP).doubleValue());

            previousIterationBuff = Arrays.copyOf(currentBuff, size);
            numberOfIterations++;
        }

        System.out.print("Solution Vector: [");
        for (int i = 0; i < size; i++) {
            if (i == size - 1)
                System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue());
            else
                System.out.print(new BigDecimal(currentBuff[i]).setScale(7, RoundingMode.HALF_UP).doubleValue() + ", ");
        }
        System.out.println("]");
        System.out.println("Number of Iterations: " + numberOfIterations);
    }

    public double calculateError(double[] previousIterationBuff, double[] currentBuff) {
        double differenceBuff[] = new double[size];
        for (int i = 0; i < size; i++)
            differenceBuff[i] = new BigDecimal(currentBuff[i] - previousIterationBuff[i])
                    .setScale(15, RoundingMode.HALF_UP).doubleValue();
        return calculateL2Norm(differenceBuff) / calculateL2Norm(currentBuff);
    }

    public double calculateL2Norm(double[] buff) {
        double sum = 0;
        for (int i = 0; i < size; i++)
            sum += Math.pow(buff[i], 2);
        return Math.sqrt(sum);
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

}
