import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;

public class Parser {

    Code decoder = new Code(); // Helper object for decoding mnemonics into binary

    File file; // Input file to be parsed

    String outputFileName;

    File fileOut = new File(outputFileName); // Output file for binary results

    Scanner scanny; // Scanner to read the input file

    SymbolTable simbo = new SymbolTable(); // Symbol table for handling labels and variables

    String currentCommand; // Holds the current command being parsed

    FileWriter outputFile = new FileWriter(fileOut); // FileWriter to write to the output file

    int numOfVariables = 16; // Number of variables encountered so far (starts from address 16)

    // Constructor: Takes a fileName as input, initializes the scanner, and
    // processes the file
    public Parser(String fileName, String outFileName) throws FileNotFoundException, IOException {
        this.outputFileName = outFileName;
        file = new File(fileName);
        scanny = new Scanner(file);

        // First pass: Count commands and populate the symbol table with labels
        String currentCommandType;
        int ROMCounter = 0;
        while (hasMoreCommands() == true) {
            currentCommandType = commandType();
            if (currentCommandType.equals("C_COMMAND"))
                ROMCounter++;
            if (currentCommandType.equals("A_COMMAND"))
                ROMCounter++;
            if (currentCommandType.equals("L_COMMAND")) {
                int indexOfOpenParenthetical = currentCommand.indexOf("(");
                int indexOfCloseParenthetical = currentCommand.indexOf(")");
                String symbol = currentCommand.substring(indexOfOpenParenthetical + 1, indexOfCloseParenthetical);
                if (!simbo.contains(symbol))
                    simbo.addEntry(symbol, ROMCounter);
            }
        }
        scanny.close();
        scanny = new Scanner(file); // Resetting the scanner to re-process the file

        // Second pass: Translate commands to binary and handle symbols
        while (hasMoreCommands() == true) {
            currentCommandType = commandType();
            if (currentCommandType != "COMMENT") {

                if (currentCommandType.equals("C_COMMAND")) {
                    if (currentCommand.contains("=")) {
                        String destMnemonic = dest("=");
                        String compMnemonic = comp();
                        destMnemonic = destMnemonic.trim();
                        compMnemonic = compMnemonic.trim();
                        String destMnemonicBinary = decoder.dest(destMnemonic);
                        String compMnemonicBinary = decoder.comp(compMnemonic);

                        outputFile.write("111" + compMnemonicBinary + destMnemonicBinary + "000" + "\n");
                    } else {
                        String jumpMnemonic = jump();
                        String compMnemonic = dest(";");
                        jumpMnemonic = jumpMnemonic.trim();
                        compMnemonic = compMnemonic.trim();
                        String compMnemonicBinary = decoder.comp(compMnemonic);
                        String jumpMnemonicBinary = decoder.jump(jumpMnemonic);
                        outputFile.write("111" + compMnemonicBinary + "000" + jumpMnemonicBinary + "\n");
                    }
                } else if (currentCommandType.equals("A_COMMAND")) {
                    int address = symbol();
                    String finalBinary = convertToUnsignedBinaryString(address, 15);
                    outputFile.write("0" + finalBinary + "\n");

                }
            }
        }

        outputFile.close();

    }

    // Converts an integer to its binary representation with a fixed bit count
    public String convertToUnsignedBinaryString(int number, int bitCount) {

        StringBuilder binaryStringBuilder = new StringBuilder();

        for (int i = bitCount - 1; i >= 0; i--) {
            // Check the bit at the i-th position (from the most significant bit)
            int bitValue = (number >> i) & 1;
            // Append the bit to the binary string
            binaryStringBuilder.append(bitValue);
        }

        return binaryStringBuilder.toString();
    }

    // Checks if there are more commands in the input file to process
    public boolean hasMoreCommands() {
        return scanny.hasNextLine();
    }

    // Determines the type of the current command
    public String commandType() {
        currentCommand = scanny.nextLine().trim();
        if (currentCommand.startsWith("/"))
            return "COMMENT";
        else if (currentCommand.startsWith("@"))
            return "A_COMMAND";
        else if (currentCommand.contains(";") || currentCommand.contains("="))
            return "C_COMMAND";
        else if (currentCommand.contains("("))
            return "L_COMMAND";
        else
            return "EMPTY_LINE";

    }

    // Retrieves the address or symbol from an A_COMMAND
    public int symbol() {
        int indexOfAtSign = currentCommand.indexOf("@");
        int aCommandButNumbers;

        if (currentCommand.contains("/")) {
            int indexOfComment = currentCommand.indexOf("/");

            if (simbo.contains(currentCommand.substring(indexOfAtSign + 1, indexOfComment).trim()))
                return simbo.getAddress(currentCommand.substring(indexOfAtSign + 1, indexOfComment).trim());

            try {
                aCommandButNumbers = Integer
                        .parseInt(currentCommand.substring(indexOfAtSign + 1, indexOfComment).trim());
                return aCommandButNumbers;
            } catch (NumberFormatException e) {
                numOfVariables++;
                simbo.addEntry(currentCommand.substring(indexOfAtSign + 1, indexOfComment), numOfVariables);
                return numOfVariables;
            }
        } else {

            if (simbo.contains(currentCommand.substring(indexOfAtSign + 1).trim()))
                return simbo.getAddress(currentCommand.substring(indexOfAtSign + 1).trim());
            try {
                aCommandButNumbers = Integer.parseInt(currentCommand.substring(indexOfAtSign + 1).trim());
                return aCommandButNumbers;
            } catch (NumberFormatException e) {
                numOfVariables++;
                simbo.addEntry(currentCommand.substring(indexOfAtSign + 1).trim(), numOfVariables);
                return numOfVariables;
            }
        }
    }

    // This will serve as a divider even though it is called dest
    // additionally Extracts the dest mnemonic from a C_COMMAND
    public String dest(String divider) {
        int indexOfDivider = currentCommand.indexOf(divider);
        return currentCommand.substring(0, indexOfDivider);
    }

    // Extracts the comp mnemonic from a C_COMMAND
    public String comp() {
        int indexOfEqualSign = currentCommand.indexOf("=");
        if (currentCommand.contains("/")) {
            int indexOfComment = currentCommand.indexOf("/");
            return currentCommand.substring(indexOfEqualSign + 1, indexOfComment);
        }
        return currentCommand.substring(indexOfEqualSign + 1).trim();
    }

    // Extracts the jump mnemonic from a C_COMMAND
    public String jump() {
        int indexOfSemiColon = currentCommand.indexOf(";");
        if (currentCommand.contains("/")) {
            int indexOfComment = currentCommand.indexOf("/");
            return currentCommand.substring(indexOfSemiColon + 1, indexOfComment - 1);
        }
        return currentCommand.substring(indexOfSemiColon + 1).trim();
    }

}