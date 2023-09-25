; who: Maayan Israel
; what: calculate first 20 values of the fibonacci sequence and store in an array
; why: because we did it in class
; when: 3/29/2023

section     .text

global      _start

_start:
    mov                 eax,    [prev]              ; EAX = PREV 
    mov                 ebx,    [next]              ; EBX = NEXT
    mov                 ecx,    seq_sz              ; ECX = COUNTER
    mov                 edi,    fib_seq             ; EDI = ITERATOR

    .loop:
    mov                 [edi],  eax                 ; write prev to sequence
    mov                 edx,    ebx                 ; temp (EDX) = next
    add                 ebx,    eax                 ; calculate new next
    mov                 eax,    edx                 ; prev (EAX) = temp (EDX)
    add                 edi,    4                   ; iterator++
    loop                .loop


exit:  
    mov                 ebx,    0                   ; return 0 status on exit - 'No Errors'
    mov                 eax,    1                   ; invoke SYS_EXIT (kernel opcode 1)
    int                 80h

section     .bss

    seq_sz:             equ     20                  ; reserved space size
    fib_seq:            resd    seq_sz              ; reserved space in data

section     .data

    prev:               dd      0                   ;default first value of sequence
    next:               dd      1                   ;default second value of sequence
