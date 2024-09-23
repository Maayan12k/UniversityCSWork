/* Java Class:LinkedPositionalList
    Author: Maayan Israel
    Class:CSCI240
    Date:9/26/2022
    Description:LinkedPositonalList driver method for PA#4
    I certify that the code below is my own work.
*/
public static void main(String[] args) {
    LinkedPositionalList<String> items = new LinkedPositionalList<String>();;
    items.addFirst("Two");
    items.addLast("Three");
    items.addFirst("One");
    items.addLast("Four");
    Iterator<String> x = items.iterator();
    while(x.hasNext()) {
        System.out .println(x.next());
    }
    items.remove(items.first());
    items.remove(items.last());
    System.out .println();
    Iterable<Position<String>> y = items.positions();
    Iterator<Position<String>> z = y.iterator();
    items.addBefore(z.next(),"Israel");
    x = items.iterator();
    items.addFirst("James");
    while(x.hasNext()) {
        System.out .println(x.next());
    }
}