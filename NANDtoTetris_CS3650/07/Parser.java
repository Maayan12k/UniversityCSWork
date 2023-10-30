import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * The Parser class is responsible for parsing Hack Assembly Language
 * instructions
 * from an input source. It reads each line of the source code and provides
 * methods
 * to access the various components of the instruction, such as the command
 * type,
 * destination, computation, and jump fields.
 *
 * Usage:
 * 1. Construct a Parser object with an input source (e.g., a FileReader or
 * BufferedReader).
 * 2. Use the `hasMoreCommands` method to check if there are more commands to
 * parse.
 * 3. Use the `advance` method to read the next command.
 * 4. Use the `getCommandType` method to determine the type of the current
 * command
 * (A_COMMAND, C_COMMAND, or L_COMMAND).
 * 5. Use the appropriate methods to retrieve specific components of the
 * command:
 * - For A_COMMAND: use `getSymbol` to retrieve the symbol.
 * - For C_COMMAND: use `getDest`, `getComp`, and `getJump` to retrieve the
 * destination, computation, and jump fields, respectively.
 * 6. Repeat steps 2-5 until there are no more commands to parse.
 * 7. Close the input source when done using the `close` method.
 *
 * @author Maayan Israel
 * @version 1.0
 * @since 10/26/2023
 */
public class Parser {

    File file; // Input file to be parsed
    Scanner scanny; // Scanner to read the input file
    String currentCommand; // Holds the current command being parsed
    String currentCommandType;

    public Parser(String fileName) throws FileNotFoundException {
        file = new File(fileName);
        scanny = new Scanner(file);
    }

    // Checks if there are more commands in the input file to process
    public boolean hasMoreCommands() {
        return scanny.hasNextLine();
    }

    // update currentCommand global variable with next vm instruction
    public void advance() {
        currentCommand = scanny.nextLine().trim();
        currentCommandType = commandType();
    }

    // Determines the type of the current command or detects comment or empty line
    public String commandType() {
        if (currentCommand.startsWith("/"))
            return "COMMENT";
        else if (currentCommand.startsWith("pop"))
            return "C_POP";
        else if (currentCommand.startsWith("push"))
            return "C_PUSH";
        else if (currentCommand.startsWith("add") || currentCommand.startsWith("sub")
                || currentCommand.startsWith("neg") || currentCommand.startsWith("eq")
                || currentCommand.startsWith("gt") || currentCommand.startsWith("lt")
                || currentCommand.startsWith("and") || currentCommand.startsWith("or")
                || currentCommand.startsWith("not"))
            return "C_ARITHMETIC";
        else if (currentCommand.startsWith("label"))
            return "C_LABEL";
        else if (currentCommand.startsWith("function"))
            return "C_FUNCTION";
        else if (currentCommand.startsWith("return"))
            return "C_RETURN";
        else if (currentCommand.startsWith("if"))
            return "C_IF";
        else if (currentCommand.startsWith("goto"))
            return "C_GOTO";
        else if (currentCommand.startsWith("call"))
            return "C_CALL";
        else
            return "EMPTY_LINE";
    }

    // returns the first argument, output will depend upon what type of command
    // currentCommand is.
    public String arg1() {
        if (currentCommandType == "C_POP")
            return currentCommand.substring(4, currentCommand.indexOf(" ", 4));

        else if (currentCommandType == "C_PUSH")
            return currentCommand.substring(5, currentCommand.indexOf(" ", 5));

        else if (currentCommandType == "C_ARITHMETIC")
            return currentCommand;

        else if (currentCommandType == "C_LABEL" || currentCommandType == "C_GOTO" || currentCommandType == "C_IF")
            return currentCommand.substring(currentCommand.indexOf(" ") + 1);

        else
            return currentCommand.substring(currentCommand.indexOf(" ") + 1,
                    currentCommand.indexOf(" ", currentCommand.indexOf(" ") + 1));

    }

    // returns the offset for the basepointers(local, this, that, static, etc)
    public int arg2() {
        int firstSpace = currentCommand.indexOf(" ");
        int secondSpace = currentCommand.indexOf(" ", firstSpace + 1);
        return Integer.parseInt(currentCommand.substring(secondSpace).trim());
    }

}
