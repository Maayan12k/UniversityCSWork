import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedWriter;

/*
 * This class contains methods to find the root of a polynomial using the
 * bisection, false position, Newton-Raphson, secant, and modified secant
 * methods.
 * 
 * The stopping error is the error that the root must be within to be considered
 * accurate.
 */
public class RootFinder {

    double stoppingError;

    public RootFinder(double stoppingError) {
        this.stoppingError = stoppingError;
    }

    /*
     * This method finds the root of a polynomial using the bisection method.
     * The upper and lower bounds of the interval must bracket the root.
     * 
     * @param polynomial the polynomial to find the root of
     * 
     * @param a the first bound of the bracketing interval
     * 
     * @param b the second bound of the bracketing interval
     * 
     * @return the root of the polynomial
     */
    public double bisection(Polynomial polynomial, double a, double b) throws IOException {
        if (polynomial.evaluateAt(a) * polynomial.evaluateAt(b) > 0 || a == b) {
            System.out.println("Error! a and b must bracket the root");
            return 0;
        }

        FileWriter fileWriter = new FileWriter("Bisection.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        double root = 0;
        double postiveX = polynomial.evaluateAt(a) > 0 ? a : b;
        double negativeX = polynomial.evaluateAt(a) > 0 ? b : a;
        double midX = (postiveX + negativeX) / 2;
        double error = Double.MAX_VALUE;
        double previousMidX = midX;
        int numberOfIterations = 0;

        while (error > stoppingError && numberOfIterations < 100) {
            numberOfIterations++;

            if (numberOfIterations == 100) {
                System.out.println("Error! Bisection method failed to converge");
                bufferedWriter.write("Error! Bisection method failed to converge\n");
                bufferedWriter.close();
                fileWriter.close();
                return 0;
            }

            if (polynomial.evaluateAt(midX) == 0) {
                root = midX;
                break;
            } else {
                if (polynomial.evaluateAt(midX) > 0)
                    postiveX = midX;
                else
                    negativeX = midX;
            }
            midX = (postiveX + negativeX) / 2;
            error = Math.abs((midX - previousMidX) / midX);
            previousMidX = midX;
            root = midX;
            bufferedWriter.write(error + "\n");
        }
        System.out.println("\nNumber of iterations: " + numberOfIterations);
        System.out.println("Root: " + root);
        bufferedWriter.close();
        fileWriter.close();
        return root;
    }

    /*
     * This method finds the root of a polynomial using the false position method.
     * The upper and lower bounds of the interval must bracket the root.
     * 
     * @param polynomial the polynomial to find the root of
     * 
     * @param a the first bound of the bracketing interval
     * 
     * @param b the second bound of the bracketing interval
     * 
     * @return the root of the polynomial
     */
    public double falsePosition(Polynomial polynomial, double a, double b) throws IOException {
        if (polynomial.evaluateAt(a) * polynomial.evaluateAt(b) > 0 || a == b) {
            System.out.println("Error! a and b must bracket the root");
            return 0;
        }

        FileWriter fileWriter = new FileWriter("FalsePosition.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        double root = 0;
        double positiveX = polynomial.evaluateAt(a) > 0 ? a : b;
        double negativeX = polynomial.evaluateAt(a) > 0 ? b : a;
        double c = (positiveX + negativeX) / 2;
        double error = Double.MAX_VALUE;
        double previousC = c;
        int numberOfIterations = 0;

        while (error > stoppingError && numberOfIterations < 100) {
            numberOfIterations++;

            if (numberOfIterations == 100) {
                System.out.println("Error! False Position method failed to converge");
                bufferedWriter.write("Error! False Position method failed to converge\n");
                bufferedWriter.close();
                fileWriter.close();
                return 0;
            }

            if (polynomial.evaluateAt(c) == 0) {
                root = c;
                break;
            } else {
                if (polynomial.evaluateAt(c) > 0)
                    positiveX = c;
                else
                    negativeX = c;
            }
            c = positiveX - polynomial.evaluateAt(positiveX)
                    * ((negativeX - positiveX) / (polynomial.evaluateAt(negativeX) - polynomial.evaluateAt(positiveX)));
            error = Math.abs((c - previousC) / c);
            previousC = c;
            root = c;
            bufferedWriter.write(error + "\n");
        }

        System.out.println("\nNumber of iterations: " + numberOfIterations);
        System.out.println("Root: " + root);
        bufferedWriter.close();
        fileWriter.close();
        return root;
    }

    /*
     * This method finds the root of a polynomial using the Newton-Raphson method.
     * 
     * @param polynomial the polynomial to find the root of
     * 
     * @param a the initial guess of the root
     */
    public double newtonRaphson(Polynomial polynomial, double a) throws IOException {

        FileWriter fileWriter = new FileWriter("NewtonRaphson.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        double root = Double.MAX_VALUE;
        double Xi = a;
        double XiMinusOne = Xi;
        double error = Double.MAX_VALUE;
        int numberOfIterations = 0;

        while (error > stoppingError && numberOfIterations < 100) {
            numberOfIterations++;

            if (numberOfIterations == 100 || polynomial.derivative().evaluateAt(Xi) == 0) {
                System.out.println("Error! Newton-Raphson method failed to converge");
                bufferedWriter.write("Error! Newton-Raphson method failed to converge\n");
                bufferedWriter.close();
                fileWriter.close();
                return 0;
            }
            if (polynomial.evaluateAt(Xi) == 0) {
                root = Xi;
                break;
            }

            double f = polynomial.evaluateAt(Xi);
            double fPrime = polynomial.derivative().evaluateAt(Xi);
            XiMinusOne = Xi;
            Xi = Xi - f / fPrime;
            error = Math.abs((Xi - XiMinusOne) / Xi);
            root = Xi;
            bufferedWriter.write(error + "\n");
        }

        System.out.println("\nNumber of iterations: " + numberOfIterations);
        System.out.println("Root: " + root);
        bufferedWriter.close();
        fileWriter.close();

        return root;
    }

    /*
     * This method finds the root of a polynomial using the secant method.
     * This method is similar to the false position method, but params first and
     * second do not necessarily bracket the root.
     * 
     * @param polynomial the polynomial to find the root of
     * 
     * @param first the first initial guess of the root
     * 
     * @param second the second initial guess of the root
     */
    public double secant(Polynomial polynomial, double first, double second) throws IOException {

        if (first == second) {
            System.out.println("Error! first and second must be different");
            return 0;
        }

        FileWriter fileWriter = new FileWriter("secant.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        double root = 0;
        double a = first;
        double b = second;
        double c = a - polynomial.evaluateAt(a) * ((b - a) / (polynomial.evaluateAt(b) - polynomial.evaluateAt(a)));
        double error = Double.MAX_VALUE;
        double previousC = c;
        int numberOfIterations = 0;

        while (error > stoppingError && numberOfIterations < 100) {
            numberOfIterations++;

            if (numberOfIterations == 100) {
                System.out.println("Error! Secant method failed to converge");
                bufferedWriter.write("Error! Secant method failed to converge\n");
                bufferedWriter.close();
                fileWriter.close();
                return 0;
            }

            if (polynomial.evaluateAt(c) == 0) {
                root = c;
                break;
            } else {
                a = b;
                b = c;
            }
            c = a - polynomial.evaluateAt(a) * ((b - a) / (polynomial.evaluateAt(b) - polynomial.evaluateAt(a)));
            error = Math.abs((c - previousC) / c);
            previousC = c;
            root = c;
            bufferedWriter.write(error + "\n");
        }

        System.out.println("\nNumber of iterations: " + numberOfIterations);
        System.out.println("Root: " + root);
        bufferedWriter.close();
        fileWriter.close();
        return root;
    }

    /*
     * This method finds the root of a polynomial using the modified secant method.
     * This method is similar to the secant method, but contains an additional
     * parameter delta.
     * 
     * @param polynomial the polynomial to find the root of
     * 
     * @param first the first initial guess of the root
     * 
     * @param second the second initial guess of the root
     * 
     * @param delta the delta value used in the calculation
     */
    public double modifiedSecant(Polynomial polynomial, double first, double second, double delta) throws IOException {

        if (first == second || delta == 0) {
            System.out.println("Error! first and second must be different && delta must not be 0");
            return 0;
        }

        FileWriter fileWriter = new FileWriter("ModifiedSecant.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        double root = 0;
        double a = first;
        double b = second;
        double c = (a + b) / 2;
        double error = Double.MAX_VALUE;
        double previousC = c;
        int numberOfIterations = 0;

        while (error > stoppingError && numberOfIterations < 100) {
            numberOfIterations++;

            if (numberOfIterations == 100) {
                System.out.println("Error! Modified Secant method failed to converge");
                bufferedWriter.write("Error! Modified Secant method failed to converge\n");
                bufferedWriter.close();
                fileWriter.close();
                return 0;
            }

            if (polynomial.evaluateAt(c) == 0) {
                root = c;
                break;
            } else {
                a = b;
                b = c;
            }
            c = b - polynomial.evaluateAt(b)
                    * ((delta * b) / (polynomial.evaluateAt(b + delta * b) - polynomial.evaluateAt(b)));
            error = Math.abs((c - previousC) / c);
            previousC = c;
            root = c;
            bufferedWriter.write(error + "\n");
        }

        System.out.println("\nNumber of iterations: " + numberOfIterations);
        System.out.println("Root: " + root);
        bufferedWriter.close();
        fileWriter.close();
        return root;
    }

}
