/* Java Class:TextEditor
Author: Maayan Israel
Class:CSCI240
Date:9/26/2022
Description: TextEditor class for PA#4
I certify that the code below is my own work.
*/
import java.util.Iterator;
public class TextEditor {
    static LinkedPositionalList<Character> listOfChars = new LinkedPositionalList<>();
    static int indexOfCursor;
    public static void main(String[] args) {
        TextEditor te = new TextEditor("HHello Word");
        te.display();
        te.right();
        te.left();
        te.insert('l');
        te.display();
        te.move(0);
        te.delete();
        te.display();
    }
    public TextEditor(String words) {
        for(int i =0; i<words.length(); i++) {
            listOfChars.addLast(words.charAt(i));
        }
        this.indexOfCursor = listOfChars .size();
    }
    public void left() {
        if(indexOfCursor == 0) return;
        else indexOfCursor --;
    }
    public void right() {
        if(indexOfCursor == listOfChars .size()) {return;}
        else {indexOfCursor ++;}
    }
    public void insert(char letter) {
        Iterable<Position<Character>> itra = listOfChars .positions();
        Iterator<Position<Character>> iter = itra.iterator();
        Position<Character> x = null;
        for(int i = 0; i<= indexOfCursor ;i++) {
            x = iter.next();
        }
        listOfChars.addBefore(x, letter);
        right();
    }
    public void delete() {
        Iterable<Position<Character>> itra = listOfChars .positions();
        Iterator<Position<Character>> iter = itra.iterator();
        Position<Character> x = null;
        for(int i = 0; i<= indexOfCursor ;i++) {
            x = iter.next();
        }
        listOfChars .remove(x);
        left();
    }
    public int current() {
        return this.indexOfCursor ;
    }
    public void move(int index) {
        if(index >=0 && index<= listOfChars .size())
        this.indexOfCursor = index;
    }
    public void display() {
        Iterator<Character> x = listOfChars .iterator();
        for(int i = 0; i < listOfChars .size(); i++) {
            if(indexOfCursor == i) 
            System.out.print(">");
            System.out.print(x.next());
        }
        if(indexOfCursor == listOfChars .size()) {
            System.out.print(">");
        }
        System.out.println();
    }
}