import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class InterpolationEngine {

    public static void main(String[] args) throws FileNotFoundException {

        System.out.println("Welcome to the Interpolation Program!");
        System.out.println(
                "This program can calculate the Newton Polynomial, Simplified Polynomial, Lagrange Polynomial, and print Netwon's divided difference table for a set of data points.");
        System.out.print("Please enter 1 for manual input or 2 for input file: ");
        Scanner scanner = new Scanner(System.in);
        int choice;

        while ((choice = scanner.nextInt()) != 1 && choice != 2) {
            System.out.print("Invalid input. Please enter 1 for manual input or 2 for input file: ");
        }

        System.out.print("Enter the number of data points: ");

        int n;
        while ((n = scanner.nextInt()) < 2 || n > 50) {
            System.out.print(
                    "The number of data points must be greater than 2 but less than 50. Enter the number of data points: ");
        }
        double[] x = new double[n];
        double[] y = new double[n];

        if (choice == 1) {

            System.out.println("Enter the x data points:");
            for (int i = 0; i < n; i++) {
                x[i] = scanner.nextDouble();
            }

            System.out.println("Enter the y data points:");
            for (int i = 0; i < n; i++) {
                y[i] = scanner.nextDouble();
            }

            System.out.println("Data points entered successfully!");
            Interpolation interpolator = new Interpolation();
            while (true) {
                System.out.println(
                        "Make a selection. \nEnter 1 for Newton Polynomial. \n2 for Simplified Polynomial \n3 for Lagrange Polynomial \n4 for Newton's divided difference table \n5 for all:");
                int selection;
                while ((selection = scanner.nextInt()) < 1 || selection > 5) {
                    System.out.println(
                            "Invalid input. Make a selection. \nEnter 1 for Newton Polynomial. \n2 for Simplified Polynomial \n3 for Lagrange Polynomial \n4 for Newton's divided difference table \n5 for all:");
                }

                switch (selection) {
                    case 1:
                        interpolator.printNewtonPolynomial(x, y);
                        break;
                    case 2:
                        interpolator.printSimplifiedPolynomial(x, y);
                        break;
                    case 3:
                        interpolator.printLagrangePolynomial(x, y);
                        break;
                    case 4:
                        interpolator.printDividedDifferenceTable(x, y);
                        break;
                    case 5:
                        interpolator.printNewtonPolynomial(x, y);
                        interpolator.printSimplifiedPolynomial(x, y);
                        interpolator.printLagrangePolynomial(x, y);
                        interpolator.printDividedDifferenceTable(x, y);
                        break;
                }
            }
        } else {
            System.out.print("Enter the file name: ");
            String fileName = scanner.next();
            File file = new File(fileName);

            while (file.exists() == false) {
                System.out.print("File not found. Please enter a valid file name: ");
                fileName = scanner.next();
                file = new File(fileName);
            }

            int inputStyle;
            System.out.println("Enter 1 if the file is in the form of x1 x2 x3 ... xn 'newline' y1 y2 y3 ... yn");
            System.out.println("Enter 2 if the file is in the form of x1 y1 'newline' x2 y2 'newline' x3 y3 ... xn yn");
            while ((inputStyle = scanner.nextInt()) != 1 && inputStyle != 2) {
                System.out.print("Invalid input. Enter 1 or 2: ");
            }
            Scanner scanner2 = new Scanner(file);
            Pattern pattern = Pattern.compile("-*[0-9]+[0-9]*\\.*[0-9]*");
            Matcher match = pattern.matcher(scanner2.nextLine());

            if (inputStyle == 1) {
                int nx = 0;
                while (match.find()) {
                    x[nx++] = Double.parseDouble(match.group());
                }

                match = pattern.matcher(scanner2.nextLine());
                nx = 0;

                while (match.find()) {
                    y[nx++] = Double.parseDouble(match.group());
                }
            } else {
                int nx = 0;
                while (scanner2.hasNextLine()) {
                    x[nx] = Double.parseDouble(match.group());
                    y[nx] = Double.parseDouble(match.group());
                    nx++;
                }
            }

            System.out.println("Data points entered successfully!");
            Interpolation interpolator = new Interpolation();
            while (true) {
                System.out.println(
                        "Make a selection. \nEnter 1 for Newton Polynomial. \n2 for Simplified Polynomial \n3 for Lagrange Polynomial \n4 for Newton's divided difference table \n5 for all:");
                int selection;
                while ((selection = scanner.nextInt()) < 1 || selection > 5) {
                    System.out.println(
                            "Invalid input. Make a selection. \nEnter 1 for Newton Polynomial. \n2 for Simplified Polynomial \n3 for Lagrange Polynomial \n4 for Newton's divided difference table \n5 for all:");
                }

                switch (selection) {
                    case 1:
                        interpolator.printNewtonPolynomial(x, y);
                        break;
                    case 2:
                        interpolator.printSimplifiedPolynomial(x, y);
                        break;
                    case 3:
                        interpolator.printLagrangePolynomial(x, y);
                        break;
                    case 4:
                        interpolator.printDividedDifferenceTable(x, y);
                        break;
                    case 5:
                        interpolator.printNewtonPolynomial(x, y);
                        interpolator.printSimplifiedPolynomial(x, y);
                        interpolator.printLagrangePolynomial(x, y);
                        interpolator.printDividedDifferenceTable(x, y);
                        return;
                }
            }
        }
    }
}
