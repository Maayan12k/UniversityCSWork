# Nand to Tetris Project

Welcome to the Nand to Tetris project, inspired by the "Elements of Computing Systems" textbook. This project encompasses various subfolders, each containing work related to different aspects of building a computer system from the ground up.

## Project Overview

This project includes the following components:

- **Hardware Simulation**: `.hdl` files where hardware devices like logic gates, multiplexers, and the ALU are designed and simulated.
- **Assembler**: An assembler written in Java that translates assembly code into machine code. Key files include `Assembler.java` (the main assembler), and helper classes `Parser.java`, `Code.java`, and `SymbolTable.java`. Output is provided in `.hack` files.
- **VM Translator**: A program that translates VM code into assembly code, which is then converted to binary. Key files include `VMTranslator.java` (the main engine), and helper classes `CodeWriter.java` and `Parser.java`.
- **Jack Compiler**: A simple compiler that translates Jack code into VM Translator code. The project includes `CompilationEngine.java` (the main engine), and helper classes `JackAnalyzer.java` and `JackTokenizer.java`.

Each subfolder contains a `README.md` file that provides detailed information about the specific work performed in that folder and the purpose of the project.

This comprehensive project aims to build a complete computer system from the fundamental components, providing a deep understanding of computing systems.
