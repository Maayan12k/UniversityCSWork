; who: Maayan Israel 
; what: copy array into another array
; why: Assembly 
; when: 2/28/2023

section .text

global _start

_start:
    mov ecx, array_sz    ; Initialize loop counter to 'array_sz'
    mov esi, array1      ; Load the source array 'array1' into esi
    mov edi, array2      ; Load the destination array 'array2' into edi
    
.loop:
    mov eax, [esi]       ; Load an integer from the source array 'array1' into eax
    mov [edi], eax       ; Store the integer from eax into the destination array 'array2'
    add esi, 4           ; Move the source pointer forward by 4 bytes (to the next integer)
    add edi, 4           ; Move the destination pointer forward by 4 bytes (to the next integer)

    loop .loop           ; Continue looping until ecx becomes 0

exit:
    mov ebx, 0           ; Set ebx to 0 (return status on exit - 'No Errors')
    mov eax, 1           ; Invoke SYS_EXIT (kernel opcode 1)
    int 80h              ; Trigger the kernel interrupt to exit the program

section .bss
    array_sz: equ 5      ; Define a constant 'array_sz' with the value 5
    array2: resd 5       ; Reserve space for an integer array 'array2' with 5 elements

section .data
    array1: dd 1,2,3,4,5 ; Initialize 'array1' with 5 integers: 1, 2, 3, 4, 5


