; who: Maayan Israel
; what: calculate first 20 values of the fibonacci sequence and store in an array
; why: because I am a dilligent computer science student
; when: 3/29/2023

section .text

global _start

_start:
    mov ecx, 18              ; Initialize loop counter to 18
    mov ebx, [prev]          ; Load the value of 'prev' into ebx
    mov edx, [next]          ; Load the value of 'next' into edx
    mov [fiboArray], ebx     ; Store 'prev' into the first element of fiboArray
    mov [fiboArray + 4], edx ; Store 'next' into the second element of fiboArray
    xor ebx, ebx             ; Clear ebx (used for temporary storage)
    mov edx, 0               ; Initialize edx to 0

.loop:
    mov eax, [prev]         ; Load 'prev' into eax
    add eax, [next]         ; Add 'next' to eax (calculating the next Fibonacci number)
    mov ebx, [next]         ; Load 'next' into ebx (used to update 'prev' in the next iteration)
    mov [next], eax         ; Update 'next' with the newly calculated Fibonacci number
    mov [prev], ebx         ; Update 'prev' with the previous 'next' value
    mov [fiboArray + 8 + edx], eax ; Store the Fibonacci number in fiboArray
    add edx, 4              ; Increment the offset in fiboArray by 4 bytes
    loop .loop              ; Continue looping until ecx becomes 0

exit:
    mov ebx, 0              ; Set ebx to 0 (return status on exit - 'No Errors')
    mov eax, 1              ; Invoke SYS_EXIT (kernel opcode 1)
    int 0x80                ; Trigger the kernel interrupt to exit the program

section .bss

fiboArray: resd 20          ; Reserve space for an array of 20 integers

section .data

prev: dd 0                  ; Initialize 'prev' to 0
next: dd 1                  ; Initialize 'next' to 1
