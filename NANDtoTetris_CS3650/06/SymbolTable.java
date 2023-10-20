import java.util.HashMap;

public class SymbolTable {

    HashMap<String, Integer> hashy;

    public SymbolTable() {
        hashy = new HashMap<>();
        hashy.put("SP", 0);
        hashy.put("LCL", 1);
        hashy.put("ARG", 2);
        hashy.put("THIS", 3);
        hashy.put("THAT", 4);
        hashy.put("R0", 0);
        hashy.put("R1", 1);
        hashy.put("R2", 2);
        hashy.put("R3", 3);
        hashy.put("R4", 4);
        hashy.put("R5", 5);
        hashy.put("R6", 6);
        hashy.put("R7", 7);
        hashy.put("R8", 8);
        hashy.put("R9", 9);
        hashy.put("R10", 10);
        hashy.put("R11", 11);
        hashy.put("R12", 12);
        hashy.put("R13", 13);
        hashy.put("R14", 14);
        hashy.put("R15", 15);
        hashy.put("SCREEN", 16384);
        hashy.put("KBD", 24576);
    }

    public void addEntry(String symbol, int address) {
        if (hashy.get(symbol) == null)
            hashy.put(symbol, address);
    }

    public boolean contains(String symbol) {
        if (hashy.get(symbol) != null)
            return true;
        return false;
    }

    public int getAddress(String symbol) {
        if (contains(symbol))
            return hashy.get(symbol);
        return -1;
    }
}
