// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/04/Fill.asm

// Runs an infinite loop that listens to the keyboard input.
// When a key is pressed (any key), the program blackens the screen,
// i.e. writes "black" in every pixel;
// the screen should remain fully black as long as the key is pressed. 
// When no key is pressed, the program clears the screen, i.e. writes
// "white" in every pixel;
// the screen should remain fully clear as long as no key is pressed.


(Beginn)
@SCREEN
D=A
@0
M=D  // Store starting screen address in RAM[0].

(CheckKeyStroke)

@KBD
D=M
@ScreenBlack
D;JGT  // Go to ScreenBlack if a key is pressed.
@ScreenWhite
D;JEQ  // Otherwise, go to ScreenWhite.

@CheckKeyStroke
0;JMP  // Continue checking for key strokes.

(ScreenBlack)
@1
M=-1  // Set color to black.
@Switch
0;JMP

(ScreenWhite)
@1
M=0  // Set color to white.
@Switch
0;JMP

(Switch)
@1
D=M  // Retrieve color value.

@0
A=M  // Point to screen pixel.
M=D  // Update pixel color.

@0
D=M+1  // Point to next pixel.
@KBD
D=A-D  // Calculate offset from KBD.

@0
M=M+1  // Move to next screen pixel.
A=M

@Switch
D;JGT  // Repeat until screen is filled.

@Beginn
0;JMP  // Loop back to beginning.
