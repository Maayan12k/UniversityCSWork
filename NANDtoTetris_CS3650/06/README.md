# Assembler Project

This folder contains an assembler written in Java, designed to translate assembly code into machine code (binary code) that the CPU can execute.

## Main Components:

- **Assembler.java**: The main file that drives the assembly process, reading assembly instructions and translating them into machine-readable code.
- **Parser.java**: A helper class that handles parsing the assembly code, breaking down each instruction into its components.
- **Code.java**: Another helper class responsible for converting parsed instructions into binary code.
- **SymbolTable.java**: A helper class that manages symbols and labels in the assembly code, ensuring they are appropriately translated into memory addresses.

## Output:
- **.hack Files**: The assembler generates `.hack` files as output. These are the binary files containing the machine code that the CPU can read and execute.

Together, these files form a functional assembler, capable of converting assembly language into binary instructions, which are saved in `.hack` files.
