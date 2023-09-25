; who: Maayan Israel
; what: LinkedList in Assembly
; why: because I am an engineer!
; when: 4/31/2023


%include "structs.inc"
%include "lib.inc"

section     .text

global      _start

_start:

    ; Call the 'read_student' function to get student records from user input
    call    read_student

exit:  
    ; Exit the program with status 0
    mov     ebx, 0      ; return 0 status on exit - 'No Errors'
    mov     eax, 1      ; invoke SYS_EXIT (kernel opcode 1)
    int     80h

section     .bss
students:   resb (student_size * 5)  ; Reserve memory for student records

section     .data
    id_prompt:      db "Enter Student's ID: ", 0  ; Prompt for student ID
    id_sz:         equ student.name  ; Size of the ID field

    name_prompt:    db "Enter Student's Name: ", 0  ; Prompt for student name
    name_sz:   equ 150  ; Size of the name field

    major_prompt:   db "Enter Student's Major: ", 0  ; Prompt for student major
    major_sz:   equ 100  ; Size of the major field

section     .text 

read_student:
; Description: Gets student records from user input.
; Receives: Address of the student record
; Returns: Nothing

    push ebp 
    mov ebp, esp 

    ; Push the size of the ID prompt and call 'print_str' to print the prompt
    push DWORD id_prompt_sz
    call print_str      ; Update to null-terminate the string

    leave  ; Restore the stack frame
    ret  ; Return from the function

    