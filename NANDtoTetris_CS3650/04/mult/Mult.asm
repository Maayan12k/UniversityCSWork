// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/04/Mult.asm

// Multiplies R0 and R1 and stores the result in R2.
// (R0, R1, R2 refer to RAM[0], RAM[1], and RAM[2], respectively.)
//
// This program only needs to handle arguments that satisfy
// R0 >= 0, R1 >= 0, and R0*R1 < 32768.

// MULTIPLICAND X MULTIPLIER = PRODUCT

@R2             //Load R2 into Reg A
M = 0           //Product = 0 by default

@R1             //Load multiplier into Reg A
D = M           //Register D equals to multiplier
@R3             //Load R3 into Reg A
M = D           // R4 = multiplicand (R4 = program counter)

@R0
D=M
@ENDWHILE
D; JEQ          //if(multiplicand = 0) END WHILE LOOP

@R1
D=M
@ENDWHILE
D; JEQ          //if(multiplier = 0)  END WHILE LOOP


//BEGIN WHILE
(WHILE)

@R3
D=M
@ENDWHILE
D; JEQ          //if(counter = 0) end while loop

@R0             //Load multiplicand into Reg A
D = M           // REG D = MULTIPLICAND
@R2             // Load Product into Reg A
M = M + D       // Product = product + multiplicand

@R3             //Load counter into Reg A
M = M-1         //COunter = counter -1;

@WHILE          // Jump back up to beginning of while loop
0; JMP

//END WHILE LOOP
(ENDWHILE)
@ENDWHILE
0; JMP