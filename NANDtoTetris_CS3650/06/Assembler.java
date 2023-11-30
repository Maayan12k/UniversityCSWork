import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * The Assembler class provides the main entry point for the assembly-to-binary
 * code translation.
 * It leverages the Parser class to handle the actual parsing and code
 * generation process.
 * 
 * Usage:
 * Run the Assembler class, which will initialize the assembly process using the
 * specified
 * input file ("PongL.asm") and produce a binary code output file
 * ("PongL.hack").
 * 
 * Example:
 * To assemble a program, simply run the Assembler class.
 * 
 * Note:
 * This is a simplified representation and assumes the input and output
 * filenames are hardcoded.
 * For a more flexible version, consider passing filenames as command line
 * arguments.
 * 
 * @author Maayan Israel
 * @version 1.0
 * @since 10/26/2023
 */

public class Assembler {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        new Assembler("PongL.asm", "PongL.hack");
    }

    public Assembler(String fileName, String binaryCodeFileName) throws FileNotFoundException, IOException {
        // new Parser(fileName, binaryCodeFileName);
    }

}
