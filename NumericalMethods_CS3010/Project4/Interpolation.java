import java.math.BigDecimal;
import java.math.RoundingMode;

public class Interpolation {

    private double[] constructDividedDifferenceTable(double x[], double y[]) {
        int n = x.length;
        if (n != y.length) {
            throw new IllegalArgumentException("Error: x and y must have the same number of elements");
        } else if (n < 2) {
            throw new IllegalArgumentException("Error: x and y must have at least 2 elements");
        }

        int numOfDividedDifferences = n * (n - 1) / 2;
        double dividedDifferences[] = new double[numOfDividedDifferences];
        int index = 0;

        for (int i = 0; i < n - 1; i++) {

            dividedDifferences[index] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
            index++;
        }

        return getDividedDifferenceArray(dividedDifferences, x, index, n - 2, 2);
    }

    private double[] getDividedDifferenceArray(double table[], double x[], int index, int numOfDDForThisCall,
            int layerNumber) {
        if (numOfDDForThisCall == 0) {
            return table;
        }

        int indexOfPreviousDD = index - numOfDDForThisCall - 1;
        for (int i = 0; i < numOfDDForThisCall; i++) {

            table[index] = (table[indexOfPreviousDD + i + 1] - table[indexOfPreviousDD + i])
                    / (x[i + layerNumber] - x[i]);
            index++;
        }
        return getDividedDifferenceArray(table, x, index, numOfDDForThisCall - 1, ++layerNumber);
    }

    public void printDividedDifferenceTable(double[] x, double y[]) {
        double[] dividedDifferenceTable = constructDividedDifferenceTable(x, y);
        int n = x.length;

        System.out.println("\n\tDivided Difference Table");
        System.out.print("x:\t");
        for (int i = 0; i < n; i++) {
            System.out.print(new BigDecimal(x[i]).setScale(2, RoundingMode.HALF_UP).doubleValue() + "\t");
        }
        System.out.println();
        System.out.print("y:\t");
        for (int i = 0; i < n; i++) {
            System.out.print(new BigDecimal(y[i]).setScale(2, RoundingMode.HALF_UP).doubleValue() + "\t");
        }
        System.out.println();

        int index = 0;
        for (int i = 0; i < n - 1; i++) {

            for (int j = 0; j < n - i - 1; j++) {
                System.out
                        .print("\t" + new BigDecimal(dividedDifferenceTable[index]).setScale(2, RoundingMode.HALF_UP)
                                .doubleValue());
                index++;
            }
            System.out.println();
        }
        System.out.println();
    }

    public void printSimplifiedPolynomial(double x[], double y[]) {
        Polynomial p = new Polynomial().interpolate(x, y);
        System.out.print("\nSimplified Polynomial\nP(x) = ");
        p.print();
    }

    public void printNewtonPolynomial(double x[], double y[]) {
        double[] dividedDifferenceTable = constructDividedDifferenceTable(x, y);
        double[] coefficients = getCoefficients(dividedDifferenceTable, x.length);
        coefficients[0] = y[0];

        System.out.print("\nNewton Polynomial\nP(x) = ");

        for (int i = 0; i < coefficients.length; i++) {
            System.out.print(new BigDecimal(coefficients[i]).setScale(2, RoundingMode.HALF_UP).doubleValue());
            for (int j = 0; j < i; j++) {
                if (x[j] > 0)
                    System.out.print("(x - " + x[j] + ")");
                else if (x[j] < 0)
                    System.out.print("(x + " + Math.abs(x[j]) + ")");
                else
                    System.out.print("x");
            }
            if (i != coefficients.length - 1) {
                System.out.print(" + ");
            }
        }
        System.out.println();

    }

    private double[] getCoefficients(double values[], int n) {
        int index = values.length - 1;
        int subtracter = 2;
        double[] result = new double[n];

        for (int i = n - 1; i > 0; i--) {
            result[i] = values[index];
            index -= subtracter;
            subtracter++;
        }

        return result;
    }

    public void printLagrangePolynomial(double x[], double y[]) {

        System.out.print("Lagrange Polynomial\nP(x) = ");
        for (int i = 0; i < x.length; i++) {
            System.out.print("((");
            for (int j = 0; j < x.length; j++) {
                if (j != i) {
                    System.out.print("(x - " + x[j] + ")");
                }
            }
            System.out.print(")");

            System.out.print("/(");
            for (int j = 0; j < x.length; j++) {
                if (j != i) {
                    System.out.print("(" + x[i] + " - " + x[j] + ")");
                }
            }
            System.out.println("))" + new BigDecimal(y[i]).setScale(3, RoundingMode.HALF_UP).doubleValue());

            if (i != x.length - 1) {
                System.out.print(" + ");
            }

        }
        System.out.println();

    }

}
