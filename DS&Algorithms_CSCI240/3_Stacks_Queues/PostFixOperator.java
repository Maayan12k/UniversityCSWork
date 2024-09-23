/*  Java Class: PostFixOperator
    Author: Zachariah Cano
    Class: CSCI 240
    Date:9/14/2022
    Description: Stack example with computation and postfix operators.

    I certify that the code below is my own work.
	Exception(s): N/A
*/
import java.util.Stack;

public class PostFixOperator {

    static String[] items = {"17","2","3","+","/","13","-"};
    static String[] items1 = {"5","2","3","^","*"};
    static String[] items2 = {"2","3","2","^","^"};

    public static void main(String[] args) {
        System.out.println(computePostFix(items));
        System.out.println(computePostFix(items1));
        System.out.println(computePostFix(items2));
    }

    public static int computePostFix(String[] statement) {
        Stack<Integer> stacky = new Stack<Integer>();
        if (statement == null) {
            System.out.println("Null");
            return -1;
        }
        for (int i = 0; i < statement.length; i++) {
             if (statement[i].equalsIgnoreCase("+")) {
                int y = stacky.pop();
                int x = stacky.pop();
                stacky.push(x + y);

            } else if (statement[i].equalsIgnoreCase("/")) {
                int y = stacky.pop();
                int x = stacky.pop();
                stacky.push(x / y);

            } else if (statement[i].equalsIgnoreCase("-")) {
                int y = stacky.pop();
                int x = stacky.pop();
                stacky.push(x - y);

            } else if (statement[i].equalsIgnoreCase("*")) {
                int y = stacky.pop();
                int x = stacky.pop();
                stacky.push(x * y);

            }else if (statement[i].equalsIgnoreCase("^")) {
                 int y = stacky.pop();
                 int x = stacky.pop();
                 stacky.push((int) Math.pow(x,y));
             }
             else{
                 int x = Integer.parseInt(statement[i]);
                 stacky.push(x);
             }
        }
        return stacky.pop();
    }
}
