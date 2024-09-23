//Method inside of class LinkedBinaryTree
void printPreOrder() {printPreOrder(root);}
public void printPreOrder(Node<E> node){
    if(node == null)
        return;
    System.out .print(node.getElement() + " ");
    printPreOrder(node.getLeft());
    printPreOrder(node.getRight());
}
void printInOrder() {printInOrder(root);}

public void printInOrder(Node<E> node){
    if(node == null)
        return;
    printInOrder(node.getLeft());
    System.out.print(node.getElement() + " ");
    printInOrder(node.getRight());
} 