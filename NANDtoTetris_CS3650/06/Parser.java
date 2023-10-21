import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;

public class Parser {

    Code decoder = new Code();
    File file;

    File fileOut = new File("RectL.hack");
    Scanner scanny;
    SymbolTable simbo = new SymbolTable();

    String currentCommand;
    FileWriter outputFile = new FileWriter(fileOut);
    int numOfVariables = 16;

    public static void main(String[] args) throws FileNotFoundException, IOException {
        Parser parsy = new Parser("RectL.asm");
    }

    public Parser(String fileName) throws FileNotFoundException, IOException {
        file = new File(fileName);
        scanny = new Scanner(file);
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
        scanny = new Scanner(file);

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

    public boolean hasMoreCommands() {
        return scanny.hasNextLine();
    }

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
    public String dest(String divider) {
        int indexOfDivider = currentCommand.indexOf(divider);
        return currentCommand.substring(0, indexOfDivider);
    }

    public String comp() {
        int indexOfEqualSign = currentCommand.indexOf("=");
        if (currentCommand.contains("/")) {
            int indexOfComment = currentCommand.indexOf("/");
            return currentCommand.substring(indexOfEqualSign + 1, indexOfComment);
        }
        return currentCommand.substring(indexOfEqualSign + 1).trim();
    }

    public String jump() {
        int indexOfSemiColon = currentCommand.indexOf(";");
        if (currentCommand.contains("/")) {
            int indexOfComment = currentCommand.indexOf("/");
            return currentCommand.substring(indexOfSemiColon + 1, indexOfComment - 1);
        }
        return currentCommand.substring(indexOfSemiColon + 1).trim();
    }

}