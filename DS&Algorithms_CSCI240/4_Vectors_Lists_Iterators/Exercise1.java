/*  Java Class:Exercise1
   Author: Maayan Israel
   Class:CSCI240
   Date:9/26/2022
   Description:driver test for ArrayList
*/
import java.util.ArrayList;

public class Exercise1 {

    public static void main(String[] args) {
        ArrayList<String> items = new ArrayList<>();
        items.add(0,"Two");
        items.add(1,"Three");
        items.add(0,"One");
        items.add(3,"Three");
        for (String x: items) {
            System.out.println(x);
        }
        System.out.println();
        items.remove(0);
        items.remove(2);
        items.add(0,"Zachariah");
        for (String x: items) {
            System.out.println(x);
        }
    }
}
