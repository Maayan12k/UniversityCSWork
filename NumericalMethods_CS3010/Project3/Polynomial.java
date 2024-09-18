import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;

/*
 * Polynomial class represents a polynomial with integer coefficients.
 * The polynomial is represented as a hash map where the key is the exponent and the value is the coefficient.
 * The class provides methods to add, subtract, multiply, and take the derivative of polynomials.
 * The class also provides a method to print the polynomial.
 * The class does not allow zero coefficients to be stored in the polynomial.
 * This version does not support division of polynomials.
 * This version supports evaluation of the polynomial at a given value of x.
 * This version contains precision of a double, however, print() method will only display 3 values after the decimal.
 */
public class Polynomial {
    HashMap<Double, Double> polynomial;

    public Polynomial() {
        polynomial = new HashMap<>();
    }

    /*
     * This method evaluates the polynomial at the given value of x.
     * The method returns the value of the polynomial at x.
     * The method uses Horner's method to evaluate the polynomial.
     * 
     * @param x the value at which the polynomial is to be evaluated
     * 
     */
    public double evaluateAt(double x) {
        double result = 0;
        for (double exponent : polynomial.keySet()) {
            result += polynomial.get(exponent) * Math.pow(x, exponent);
        }
        return result;
    }

    /*
     * This method interpolates a polynomial that passes through the given points.
     * The xValues array contains the x-coordinates of the points.
     * The yValues array contains the y-coordinates of the points.
     * The method returns the polynomial that passes through the given points.
     * The method uses Lagrange interpolation to find the polynomial.
     * 
     * @param xValues the x-coordinates of the points
     * 
     * @param yValues the y-coordinates of the points
     * 
     * @throws IllegalArgumentException if the number of x-values and y-values are
     * not the same
     * 
     * @throws IllegalArgumentException if the x-values are not unique
     */
    public Polynomial interpolate(double[] xValues, double[] yValues) {

        if (xValues.length != yValues.length) {
            throw new IllegalArgumentException("The number of x-values and y-values must be the same.");
        }

        HashMap<Double, Double> uniqueCheck = new HashMap<>();
        for (int i = 0; xValues.length > i; i++) {
            if (uniqueCheck.containsKey(xValues[i])) {
                throw new IllegalArgumentException("The x-values must be unique.");
            }
            uniqueCheck.put(xValues[i], yValues[i]);
        }

        Polynomial result = new Polynomial();
        int size = xValues.length;
        double constant = 0;
        double productConstant = 1;

        for (int i = 0; i < size; i++) {
            constant = yValues[i];
            if (constant == 0)
                continue;
            productConstant = 1;
            for (int j = 0; j < size; j++) {
                if (j != i) {
                    productConstant *= (xValues[i] - xValues[j]);
                }
            }
            constant /= productConstant;
            Polynomial temp = evaluateLaGrangeNumerator(xValues, i);
            temp = temp.multiply(constant);
            result = result.add(temp);
        }
        Polynomial newResult = new Polynomial();
        for (double exponent : result.polynomial.keySet()) {
            if (result.polynomial.get(exponent) != 0) {
                newResult.addTerm(exponent, result.polynomial.get(exponent));
            }
        }
        return newResult;
    }

    private Polynomial evaluateLaGrangeNumerator(double[] xValues, int index) {
        Polynomial result = new Polynomial();
        for (int i = 0; i < xValues.length; i++) {
            if (i != index) {
                Polynomial temp = new Polynomial();
                temp.addTerm(1, 1);
                temp.addTerm(0, -xValues[i]);

                if (result.polynomial.isEmpty()) {
                    result = temp;
                } else {
                    result = result.multiply(temp);
                }
            }
        }
        Polynomial newResult = new Polynomial();
        for (double exponent : result.polynomial.keySet()) {
            if (result.polynomial.get(exponent) != 0) {
                newResult.addTerm(exponent, result.polynomial.get(exponent));
            }
        }
        return newResult;
    }

    /*
     * This method adds a term to the polynomial.
     * The exponent is the exponent of the term.
     * The coefficient is the coefficient of the term.
     * If the coefficient is zero, the term is not added to the polynomial.
     * If the term already exists in the polynomial, the coefficient is added to the
     * existing coefficient.
     * 
     * @param exponent the exponent of the term
     * 
     * @param coefficient the coefficient of the term
     */
    public void addTerm(double exponent, double coefficient) {
        if (coefficient == 0)
            return;
        if (polynomial.containsKey(exponent)) {
            polynomial.put(exponent, polynomial.get(exponent) + coefficient);
        } else {
            polynomial.put(exponent, coefficient);
        }
    }

    /*
     * This method adds two polynomials.
     * The method returns the sum of the two polynomials.
     * The method does not modify the original polynomials.
     * 
     * @param other the polynomial to be added to the polynomial
     */
    public Polynomial add(Polynomial other) {
        Polynomial result = new Polynomial();
        for (double exponent : polynomial.keySet()) {
            result.addTerm(exponent, polynomial.get(exponent));
        }
        for (double exponent : other.polynomial.keySet()) {
            result.addTerm(exponent, other.polynomial.get(exponent));
        }
        Polynomial newResult = new Polynomial();
        for (double exponent : result.polynomial.keySet())
            if (result.polynomial.get(exponent) != 0)
                newResult.addTerm(exponent, result.polynomial.get(exponent));

        return newResult;
    }

    /*
     * This method subtracts two polynomials.
     * The method returns the difference of the two polynomials.
     * The method does not modify the original polynomials.
     * 
     * @param other the polynomial to be subtracted from the polynomial
     */
    public Polynomial subtract(Polynomial other) {
        Polynomial result = new Polynomial();
        for (double exponent : polynomial.keySet())
            result.addTerm(exponent, polynomial.get(exponent));

        for (double exponent : other.polynomial.keySet())
            result.addTerm(exponent, -other.polynomial.get(exponent));

        Polynomial newResult = new Polynomial();
        for (double exponent : result.polynomial.keySet())
            if (result.polynomial.get(exponent) != 0)
                newResult.addTerm(exponent, result.polynomial.get(exponent));

        return newResult;
    }

    /*
     * This method multiplies two polynomials.
     * The method returns the product of the two polynomials.
     * The method does not modify the original polynomials.
     * 
     * @param other the polynomial to be multiplied with the polynomial
     */
    public Polynomial multiply(Polynomial other) {
        if (polynomial.isEmpty() || other.polynomial.isEmpty())
            return new Polynomial();

        Polynomial result = new Polynomial();
        for (double exponent1 : polynomial.keySet())
            for (double exponent2 : other.polynomial.keySet()) {
                double coefficient1 = polynomial.get(exponent1);
                double coefficient2 = other.polynomial.get(exponent2);
                result.addTerm(exponent1 + exponent2, coefficient1 * coefficient2);
            }

        return result;
    }

    /*
     * This method multiplies the polynomial by a scalar.
     * The method returns the product of the polynomial and the scalar.
     * The method does not modify the original polynomial.
     * 
     * @param scalar the scalar to multiply the polynomial by
     */
    public Polynomial multiply(double scalar) {
        Polynomial result = new Polynomial();
        for (double exponent : polynomial.keySet())
            result.addTerm(exponent, polynomial.get(exponent) * scalar);
        return result;
    }

    /*
     * This method returns the derivative of the polynomial.
     * The method returns the derivative of the polynomial.
     * The method does not modify the original polynomial.
     */
    public Polynomial derivative() {
        Polynomial result = new Polynomial();
        for (double exponent : polynomial.keySet()) {
            double coefficient = polynomial.get(exponent);
            if (exponent != 0)
                result.addTerm(exponent - 1, coefficient * exponent);
        }
        return result;
    }

    /*
     * This method prints the polynomial.
     * The method prints the polynomial in the form of a string.
     * The method does not modify the original polynomial.
     */
    public void print() {
        if (polynomial.isEmpty()) {
            System.out.println("0\n");
            return;
        }
        int size = polynomial.size();
        int index = 0;
        for (double exponent : polynomial.keySet()) {
            double coefficient = polynomial.get(exponent);
            if (exponent == 0) {
                if (coefficient == 1.0) {
                    if (++index == size)
                        System.out.print("1");
                    else
                        System.out.print("1 + ");
                } else {
                    if (++index == size)
                        System.out.print(new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue());
                    else
                        System.out.print(
                                new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue() + " + ");
                }
            } else if (exponent == 1) {
                if (coefficient == 1.0) {
                    if (++index == size)
                        System.out.print("x");
                    else
                        System.out.print("x + ");
                } else {
                    if (++index == size)
                        System.out.print(
                                new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue() + "x");
                    else
                        System.out.print(
                                new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue() + "x + ");
                }
            } else {
                if (coefficient == 1.0) {
                    if (++index == size)
                        System.out.print("x^" + (int) exponent);
                    else
                        System.out.print("x^" + (int) exponent + " + ");
                } else if (coefficient == -1.0) {
                    if (++index == size)
                        System.out.print("-x^" + (int) exponent);
                    else
                        System.out.print("-x^" + (int) exponent + " + ");
                } else {
                    if (++index == size)
                        System.out.print(new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue()
                                + "x^" + (int) exponent);
                    else
                        System.out.print(new BigDecimal(coefficient).setScale(3, RoundingMode.HALF_UP).doubleValue()
                                + "x^" + (int) exponent + " + ");
                }

            }

        }
        System.out.println("\n");
    }

}