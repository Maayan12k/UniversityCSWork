; who: Maayan Israel
; what: tests strlen in lib.asm
; why: homework
; when: 5/9/2023

%include "lib(1).inc"

section .text
  
global _start

_start:

;returns    EAX = inputted string
;recevies   [ebp + 8] = prompt address
;           [ebp + 12] = prompt size
;           [ebp + 16] = input buffer address
;           [ebp + 20] = input buffer size

    mov  eax, string                ;pass arg1
    mov  ebx, buffer                ; PASS ARG2
    mov  ecx, inputbuffer           ; PASS ARG3
    mov  edx, input_sz_buffer       ; pass arg4

    push edx                        ;preserve
    push ecx                        ;preserve
    push ebx                        ;preserve
    push eax                        ;preserve
    call get_input

    push  input_sz_buffer        ; buffer size
    push  inputbuffer    ; buffer address size

    call strlen                 ; call function

exit:
    mov   eax, 1                ; Set syscall number for exit
    xor   ebx, ebx              ; Set exit status to zero
    int   0x80                  ; Call the exit syscall

section .bss 

    
    inputbuffer: resb 255
    input_sz_buffer: equ $ - inputbuffer

section .data

    string: db "Give me string", 0x0a,0
    buffer: equ $ - string

    finalString: db "Length of String was "
    finalString_sz: equ - finalString

    stringLenVal: dd 1