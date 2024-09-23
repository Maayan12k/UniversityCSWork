/*  Java Class: BalancingSymbols
    Author: Zachariah Cano
    Class: CSCI 240
    Date:9/14/2022
    Description: Stack example to check for balanced parenthetical.

    I certify that the code below is my own work.
	Exception(s): N/A
*/
import java.util.Stack;

public class BalancingSymbols {

    public static void main(String[] args){
        System.out.println(stackExample("{( a + b )* c1}"));
        System.out.println(stackExample("{( a + b )* c1]"));
        System.out.println(stackExample("(( a + b) )* c1} /15)"));
        }

    public static boolean stackExample(String s) {
        Stack<Character> stack  = new Stack<Character>();
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c == '[' || c == '(' || c == '{' ) {
                stack.push(c);
            } else if(c == ']') {
                if(stack.isEmpty() || stack.pop() != '[') {
                    return false;
                }
            } else if(c == ')') {
                if(stack.isEmpty() || stack.pop() != '(') {
                    return false;
                }
            } else if(c == '}') {
                if(stack.isEmpty() || stack.pop() != '{') {
                    return false;
                }
            }

        }
        return stack.isEmpty();
    }
}
