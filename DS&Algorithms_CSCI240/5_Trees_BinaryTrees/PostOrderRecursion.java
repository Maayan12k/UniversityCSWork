//Method inside of class LinkedBinaryTree
public static void main(String[] args) {
    LinkedBinaryTree<String> tree = new LinkedBinaryTree<>();
    Position<String> root = tree.addRoot("A");
    Node<String> b = tree.createNode("B", tree.root, null, null);
    tree.root.setLeft(b);
    Node<String> c = tree.createNode("C", tree.root, null, null);
    tree.root.setRight(c);
    Node<String> d = tree.createNode("D", b, null, null);
    b.setLeft(d);
    Node<String> e = tree.createNode("E", b, null, null);
    b.setRight(e);
    tree.printPostOrder();
}
void printPostOrder() {printPostOrder(root);}

public void printPostOrder(Node<E> node){
    if(node == null)
    return;
    printPostOrder(node.getLeft());
    printPostOrder(node.getRight());
    System.out .print(node.getElement() + " ");
}