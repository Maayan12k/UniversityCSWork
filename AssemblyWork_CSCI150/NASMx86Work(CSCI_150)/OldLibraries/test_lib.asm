%include "lib.inc"

; who: Maayan Israel
; what: file used for testing lib.asm
; why: because I am an engineer!
; when: Spring 2023

section .text

global _start

_start:

    ; Display the prompt for user input
    mov     eax, string       ; Load the address of the string into eax
    mov     ebx, str_sz       ; Load the size of the string into ebx
    mov     ecx, buffer       ; Load the address of the input buffer into ecx
    mov     edx, buffer_sz    ; Load the size of the input buffer into edx
    call    get_input         ; Call the get_input function to receive user input

    ; Push the address of the input buffer
    push    eax

    ; Reverse the string in the input buffer
    mov     ebx, eax           ; Copy the address of the input buffer into ebx
    mov     eax, buffer        ; Load the address of the input buffer into eax
    call    rev_str            ; Call the rev_str function to reverse the string

    ; Pop the address of the input buffer
    pop     ebx

    ; Display the reversed string
    mov     ebx, eax           ; Copy the address of the reversed string into ebx
    mov     eax, buffer        ; Load the address of the input buffer into eax
    call    print_str          ; Call the print_str function to print the string

    ; Print a newline character
    mov     eax, 0x0a           ; Load the newline character into eax
    call    print_char          ; Call the print_char function to print the character

exit:  
    ; Exit the program
    mov     ebx, 0              ; Return 0 status on exit (no errors)
    mov     eax, 1              ; Invoke SYS_EXIT (kernel opcode 1)
    int     80h

section .bss
    buffer_sz: equ 255
    buffer:     resb buffer_sz   ; Reserve space for the input buffer

section .data
    string: db "Enter String: "   ; Define the prompt string
    str_sz: equ $ - string        ; Calculate the size of the prompt string
