
; who: Maayan Israel
; what: Bubble sort in ASSEMBLY!  
; why: Because I am an engineer
; when: 4/23/2023

%include "bubble.inc"   ; Include the bubble sort function from an external file

section .text

global _start

_start:

    push    DWORD array_sz     ; Push the size of the array onto the stack
    push    array             ; Push the address of the array onto the stack
    call    bubble_sort       ; Call the bubble_sort function to sort the array
    add     esp, 8            ; Adjust the stack pointer to remove the arguments

exit:  
    mov     ebx, 0            ; Set ebx to 0 (return 0 status on exit - 'No Errors')
    mov     eax, 1            ; Invoke SYS_EXIT (kernel opcode 1)
    int     80h               ; Trigger the kernel interrupt to exit the program

section .bss

section .data

    array: dd 6,3,7,4,5,2,8,9,1,10   ; Initialize an array with unsorted values
    array_sz: equ $ - array          ; Calculate the size of the array
