import java.io.IOException;
import java.util.Scanner;

public class RootFinderEngine {

    public static void main(String[] args) throws IOException {
        System.out.println("Welcome to the RootFinder Calculator.");
        System.out.println("This program will find the roots of a polynomial.");
        System.out.println("Polynomial A  = 3x^2 - 11.7x + 17.7");
        System.out.println("Polynomial B = x + 10 - xcosh(50/x)");
        System.out.println("Please enter the desired tolerance.");
        System.out.print("Tolerance: ");
        Scanner scanny = new Scanner(System.in);
        double tolerance;

        while ((tolerance = scanny.nextDouble()) < 0 || tolerance >= 1) {
            System.out.println("Please enter a valid number.");
            System.out.print("Tolerance: ");
        }

        Polynomial polyA = new Polynomial();
        polyA.addTerm(3, 2);
        polyA.addTerm(2, -11.7);
        polyA.addTerm(1, 17.7);
        polyA.addTerm(0, -5);

        Polynomial polyB = new Polynomial(); // Taylor Series Expansion of polynomial B
        polyB.addTerm(0, 10);
        polyB.addTerm(-1, -(50 * 50) / 2);
        polyB.addTerm(-3, -(50 * 50 * 50 * 50) / (2 * 3 * 4));
        polyB.addTerm(-5, -(50 * 50 * 50 * 50 * 50 * 50) / (2 * 3 * 4 * 5 * 6));
        polyB.addTerm(-7, -(50 * 50 * 50 * 50 * 50 * 50 * 50 * 50) / (2 * 3 * 4 * 5 *
                6 * 7 * 8));
        polyB.addTerm(-9, -(50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 * 50) / (2 * 3
                * 4 * 5 * 6 * 7 * 8 * 9 * 10));
        polyB.addTerm(-11, -(50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 * 50 *
                50)
                / (2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12));

        System.out.println("Enter the polynomial you would like to find the roots of.");
        System.out.println("1. Polynomial A");
        System.out.println("2. Polynomial B");
        System.out.print("Choice: ");
        int choice;
        while ((choice = scanny.nextInt()) != 1 && choice != 2) {
            System.out.println("Please enter a valid choice.");
            System.out.print("Choice: ");
        }
        RootFinder rootFinder = new RootFinder(tolerance);
        if (choice == 1) {
            System.out.println("Please enter which method you would like to use.");
            System.out.println("1. Bisection");
            System.out.println("2. False Position");
            System.out.println("3. Newton-Raphson");
            System.out.println("4. Secant");
            System.out.println("5. Modified Secant");
            System.out.print("Choice: ");
            int method;

            while ((method = scanny.nextInt()) != 1 && method != 2 && method != 3
                    && method != 4 && method != 5) {
                System.out.println("Please enter a valid choice.");
                System.out.print("Choice: ");
            }

            switch (method) {
                case 1:
                    System.out.println("Bisection Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double a = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double b = scanny.nextDouble();
                    rootFinder.bisection(polyA, a, b);
                    break;
                case 2:
                    System.out.println("False Position Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double c = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double d = scanny.nextDouble();
                    rootFinder.falsePosition(polyA, c, d);
                    break;
                case 3:
                    System.out.println("Newton-Raphson Method:");
                    System.out.println("Enter the initial guess.");
                    System.out.print("Initial Guess: ");
                    double guess = scanny.nextDouble();
                    rootFinder.newtonRaphson(polyA, guess);
                    break;
                case 4:
                    System.out.println("Secant Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double e = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double f = scanny.nextDouble();
                    rootFinder.secant(polyA, e, f);
                    break;
                case 5:
                    System.out.println("Modified Secant Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double g = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double h = scanny.nextDouble();
                    System.out.println("Enter the delta value.");
                    System.out.print("Delta: ");
                    double delta = scanny.nextDouble();
                    rootFinder.modifiedSecant(polyA, g, h, delta);
                    break;
            }
        } else {
            System.out.println("Please enter which method you would like to use.");
            System.out.println("1. Bisection");
            System.out.println("2. False Position");
            System.out.println("3. Newton-Raphson");
            System.out.println("4. Secant");
            System.out.println("5. Modified Secant");
            System.out.print("Choice: ");
            int method;

            while ((method = scanny.nextInt()) != 1 && method != 2 && method != 3
                    && method != 4 && method != 5) {
                System.out.println("Please enter a valid choice.");
                System.out.print("Choice: ");
            }

            switch (method) {
                case 1:
                    System.out.println("Bisection Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double a = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double b = scanny.nextDouble();
                    rootFinder.bisection(polyB, a, b);
                    break;
                case 2:
                    System.out.println("False Position Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double c = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double d = scanny.nextDouble();
                    rootFinder.falsePosition(polyB, c, d);
                    break;
                case 3:
                    System.out.println("Newton-Raphson Method:");
                    System.out.println("Enter the initial guess.");
                    System.out.print("Initial Guess: ");
                    double guess = scanny.nextDouble();
                    rootFinder.newtonRaphson(polyB, guess);
                    break;
                case 4:
                    System.out.println("Secant Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double e = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double f = scanny.nextDouble();
                    rootFinder.secant(polyB, e, f);
                    break;
                case 5:
                    System.out.println("Modified Secant Method:");
                    System.out.println(
                            "Enter the interval you would like to search for a root in.");
                    System.out.print("Lower Bound: ");
                    double g = scanny.nextDouble();
                    System.out.print("Upper Bound: ");
                    double h = scanny.nextDouble();
                    System.out.println("Enter the delta value.");
                    System.out.print("Delta: ");
                    double delta = scanny.nextDouble();
                    rootFinder.modifiedSecant(polyB, g, h, delta);
                    break;
            }

        }

        scanny.close();
    }
}
