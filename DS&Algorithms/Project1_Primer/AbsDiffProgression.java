package DataStructures_Algorithms_CSCI240;

/**
 * AbsDiffProgression represents a sequence of numbers where the next term
 * is the absolute difference of the previous two terms.
 */
public class AbsDiffProgression extends Progression {
    protected long prev;

    /**
     * Default constructor; initializes the progression with 0 and 1.
     */
    public AbsDiffProgression() {
        this(0, 1);
    }

    /**
     * Construct the progression with specific first and second values.
     *
     * @param first  The first term in the progression.
     * @param second The second term in the progression.
     */
    public AbsDiffProgression(long first, long second) {
        super(first);
        prev = second + first;
    }

    /**
     * Advances the progression to the next number in the sequence.
     */
    protected void advance() {
        long temp = prev;
        prev = current;
        current = Math.abs(current - temp);
    }

    public static void main(String args[]) {
        AbsDiffProgression prog1 = new AbsDiffProgression(2, 200);
        prog1.printProgression(25);
    }
}
