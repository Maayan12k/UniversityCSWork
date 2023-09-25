; who: Maayan Israel
; what: read and store student information, including ID, name, and major, from user input
; why: because I am an engineer!
; when: 4/28/2023


%include "structs.inc"                          ; Include a file with structure definitions
%include "lib.inc"                              ; Include a file with library functions

section .text

global _start

_start:

call read_student  ; Call the read_student function to get student information

exit:  
    mov ebx, 0                                  ; Set ebx to 0 (return 0 status on exit - 'No Errors')
    mov eax, 1                                  ; Invoke SYS_EXIT (kernel opcode 1)
    int 80h                                     ; Trigger the kernel interrupt to exit the program

section .bss
students: resb (student_size * 5)               ; Reserve space for student records (assuming up to 5 records)

section .data
id_prompt: db "Enter Student's ID: ", 0         ; Prompt for entering student ID
id_sz: equ student.name                         ; Size of the ID prompt

name_prompt: db "Enter Student's Name: ", 0     ; Prompt for entering student name
name_sz: equ 150                                ; Size of the name prompt

major_prompt: db "Enter Student's Major: ", 0   ; Prompt for entering student major
major_sz: equ 100                               ; Size of the major prompt

section .text 

read_student:
; Description: Gets student record from user input
; Receives: Address of student record

    push ebp                                    ; Save the base pointer
    mov ebp, esp                                ; Set up a new base pointer

    push DWORD id_prompt_sz
    call print_str                              ; Call a function to print the ID prompt and null-terminate it

    leave                                       ; Restore the previous base pointer
    ret                                         ; Return to the caller

