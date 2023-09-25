; who: Maayan Israel
; what: demo quad value as an array
; why: why not?
; when: 03/15/2023

section .text

global _start

_start:
    ; Load the values of five integers into registers and calculate their sum
    mov eax, [firstInt]   ; Load the value of 'firstInt' into eax
    add eax, [secondInt]  ; Add 'secondInt' to eax
    add eax, [thirdInt]   ; Add 'thirdInt' to eax
    add eax, [fourthInt]  ; Add 'fourthInt' to eax
    add eax, [fifthInt]   ; Add 'fifthInt' to eax
    mov [sumOfAll], eax   ; Store the sum in 'sumOfAll'

exit:
    mov ebx, 0            ; Return 0 status on exit - 'No Errors'
    mov eax, 1            ; Invoke SYS_EXIT (kernel opcode 1)
    int 80h               ; Trigger the kernel interrupt to exit the program

section .data
    firstInt:  dd 0x09    ; Initialize 'firstInt' with the value 0x09
    secondInt: dd 0x19    ; Initialize 'secondInt' with the value 0x19
    thirdInt:  dd 0x05    ; Initialize 'thirdInt' with the value 0x05
    fourthInt: dd 0x14    ; Initialize 'fourthInt' with the value 0x14
    fifthInt:  dd 0x22    ; Initialize 'fifthInt' with the value 0x22

section .bss
    sumOfAll: resb 1      ; Reserve space for a single byte variable 'sumOfAll'



   