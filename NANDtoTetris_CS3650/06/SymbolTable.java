import java.util.HashMap;

public class SymbolTable {

    // HashMap to store assembly symbols and their corresponding addresses
    HashMap<String, Integer> hashy;

    // Constructor: Initializes the symbol table with predefined symbols
    public SymbolTable() {
        hashy = new HashMap<>();

        // Predefined symbols for assembly language
        hashy.put("SP", 0);
        hashy.put("LCL", 1);
        hashy.put("ARG", 2);
        hashy.put("THIS", 3);
        hashy.put("THAT", 4);

        // Predefined register symbols
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

        // Memory-mapped I/O addresses
        hashy.put("SCREEN", 16384);
        hashy.put("KBD", 24576);
    }

    // Method to add a new entry to the symbol table
    // Only adds if the symbol doesn't already exist
    public void addEntry(String symbol, int address) {
        if (hashy.get(symbol) == null)
            hashy.put(symbol, address);
    }

    // Method to check if a symbol exists in the symbol table
    public boolean contains(String symbol) {
        if (hashy.get(symbol) != null)
            return true;
        return false;
    }

    // Method to retrieve the address of a symbol from the symbol table
    // Returns -1 if the symbol doesn't exist
    public int getAddress(String symbol) {
        if (contains(symbol))
            return hashy.get(symbol);
        return -1;
    }
}
