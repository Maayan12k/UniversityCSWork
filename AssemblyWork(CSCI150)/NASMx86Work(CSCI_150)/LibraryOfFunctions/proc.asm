; who: Maayan Israel
; what: LinkedList in Assembly
; why: because I am an engineer!
; when: 4/31/2023

section     .text

global      _start

_start:

    ; Allocate 4 bytes of memory using 'malloc' and store the address in EAX
    push    DWORD 4
    call    malloc 
    add     esp, 4

    ; Push the address in EAX onto the stack
    push    eax

    ; Push the address of 'name' and call the 'create_node' function
    push    DWORD name
    call    create_node
    add     esp, 8


create_node:
; Description: Creates a new node with a 32-bit value and a reference to the next node.
; Receives: arg1 = 32-bit value
;           arg2 = address to the next node
; Returns: address of the new node in EAX

    push    ebp
    mov     ebp, esp

    ; Allocate memory for a new node structure using 'malloc'
    push DWORD node_size
    call malloc

    ; Preserve EDI and store the address of the new node in EDI
    push    edi
    mov     edi, eax

    ; Copy the value from arg1 to the 'node.value' field of the new node
    mov     eax, [ebp + 8]
    mov     [edi + node.value], eax

    ; Copy the address from arg2 to the 'node.next' field of the new node
    mov     eax, [ebp + 12]
    mov     [edi + node.next], eax

    ; Set EAX to the address of the new node as the return value
    mov     eax, edi

    ; Restore EDI and clean up the stack frame
    pop     edi
    leave 
    ret

exit:  
    ; Exit the program with status 0
    mov     ebx, 0      ; return 0 status on exit - 'No Errors'
    mov     eax, 1      ; invoke SYS_EXIT (kernel opcode 1)
    int     80h

section     .bss

section     .data

    ; Define a string 'name'
    name: db "name"

section     .text 

; Define a structure 'node' with two fields: 'value' and 'next'
struc node:
    .value: resd 1
    .next:  resd 1
endstruc
