; who: Maayan Israel
; what: Bubble sort in ASSEMBLY!  
; why: Because I am an engineer
; when: 4/23/2023

section .text

global bubble_sort
global swap

; Bubble Sort function
bubble_sort:
; Description: Sorts an array of signed double words in ascending order using the bubble sort algorithm.
; Receives:
;   arg1: array address
;   arg2: array size (in bytes)
; Returns: N/A

    push    ebp                 ; Preserve the current base pointer
    mov     ebp, esp            ; Set up a new base pointer

    ; Preserve registers
    push    ebx
    push    esi 
    push    edi

    ; Set edi to the last element of the array 
    mov     edi, [ebp + 8]      ; edi = array address
    mov     ecx, [ebp + 12]     ; ecx = array size in bytes
    lea     edi, [edi + ecx - 4] ; edi = address of the last element

.outloop:
    mov     esi, [ebp + 8]      ; esi = beginning of the array 
    cmp     edi, esi            ; Compare edi with esi (end with start)
    jbe     .break_out          ; If true, exit the loop

.inloop:
    cmp     esi, edi            ; Compare esi with edi (start with end)
    jae     .break_in           ; If true, exit the inner loop
    lea     ebx, [esi + 4]      ; ebx = esi + 4 (address of the next element)
    mov     eax, [ebx]          ; eax = value of the next element

.if:
    cmp     [esi], eax          ; Compare array[esi] with eax
    jle     .endif              ; If true, skip swapping

    push    esi                 ; Pass arg1 (address of the first element)
    push    ebx                 ; Pass arg2 (address of the second element)
    call    swap                ; Call the swap function
    add     esp, 8              ; Deallocate the arguments

.endif:
    add     esi, 4              ; Increment esi by 4 to move to the next element
    jmp     .inloop             ; Jump back to the inner loop

.break_in:                      ; Exit the inner loop
    sub     edi, 4              ; Decrement edi to move one step back
    jmp     .outloop            ; Jump back to the outer loop

.break_out:                     ; Exit the outer loop

; Restore registers
    pop edi 
    pop esi 
    pop ebx

    leave                       ; Restore the previous base pointer and stack frame
    ret                         ; Return to the calling function

; Swap function
swap:
; Receives:
;   arg1: address of the first value
;   arg2: address of the second value
;---------------------------
    push    ebp                 ; Preserve the current base pointer
    mov     ebp, esp            ; Set up a new base pointer
    push    esi
    push    edi

    mov     esi, [ebp + 8]      ; esi = address of the first value
    mov     edi, [ebp + 12]     ; edi = address of the second value
    mov     eax, [esi]          ; eax = value at address esi
    xchg    [edi], eax          ; Exchange values at addresses esi and edi
    mov     [esi], eax          ; Store the updated value back to esi

    pop     edi
    pop     esi
    leave                       ; Restore the previous base pointer and stack frame
    ret                         ; Return to the caller

; END of swap-------------------------------
