section .text

;this version of lib.asm passes argument via stack

global atoi
global sum_array
global print_char
global rev_str
global get_input
global print_str
global errno
global current_time
global seed_random
global gen_random
global print_uint
global print_nl
global strlen
global swap

;----------------------------
strlen:
; Description:  Compute length of a null-terminated string
; Receives:     arg1: string address
;               arg2: max buffer size
; Returns:      eax = length of the string
; Requires:     Nothing
;----------------------------
    push    ebp
    mov     ebp, esp
    
    mov     edi, [ebp +8]       ;set destination
    xor     eax, eax            ; clear eax
    mov     ecx, [ebp +12]      ; set counter
    cld                         ; set direct to forward

    repne   scasb
    cmp     ecx, 0              ; check for string of length of the buffer
    jz      .default

    dec     edi                 ; set edi to point to char right before 0
    mov     eax, edi            ; set for subtraction
    sub     eax, [ebp +8]       ; calculate length unto eax
    jmp     .done

    .default:
    mov     eax, [ebp +12]      ; default is to set length to magnitude of buffer
    dec     eax                 
    
    .done:
    pop     ebp 
    ret


print_nl:
;description: prints a new line 
;receives nothing
; returns nothing
; requires print_char

    push        ebp
    mov         ebp, esp

    push        0x0a
    call        print_char
    leave 
    ret

print_uint:
; description: prints an unsigned integer
;receives:      arg1 = uint value
;requires: character buffer(char_buff)

    push    ebp
    mov     ebp, esp

    push    char_buff
    push    DWORD [ebp + 8]
    call    itoa
    add     esp, 8

    push    char_buff_sz
    push    char_buff
    call    strlen
    add     esp, 4

    push    eax
    push    char_buff
    call    print_str

    pop     ebx


    leave 
    ret


seed_random:
;description:  gets input for random generator
; receives [ebp + 8] = furnisher

    push    ebp                     ; preserve
    mov     ebp, esp                ; set new stack frame 

    mov     eax, [ebp + 8]
    mov     [next], eax             ;furnish next var

    pop     ebp
    ret

gen_random:
; description: generates random number 
; returns: on EAX a random value
; receives: N/A

    mov     eax, [next]     
    mov     ecx, const1
    mul     ecx             ;next * 1103515245 

    add     eax, const2     ;next + 12345
    mov     [next], eax     ; eax = next
    xor     edx, edx        ; clear edx

    mov     ecx, const3
    div     ecx             ; next/65536

    mov     ecx, const4
    xor     edx, edx
    div     ecx             ; next % 32768
    mov     eax, edx        ; return eax as the rand value

    ret

current_time:
; description: gets the number of seconds from 1/1/1970

    push        ebx
    mov         eax, 13
    xor         ebx, ebx
    int         80h     

    pop         ebx

    ret



atoi: 
;description: converts a null terminated string of digits to an integer
;receives     [ebp + 8]= address of the string 
;requirments: preserve ebp             
; returns:  EAX = the integer valuer
;Algoritithm: horner's method


; EAX is accumulator
; ECX will hold our character code

.factor:    equ 10
.offset:    equ 48

    push        ebp                 ; preserve
    mov         ebp, esp            ; set new stack frame

    push        esi
    mov         ESI, [ebp+8]        ;ESI = source
    xor         eax, eax            ; eax = 0
    xor         ecx, ecx            ; ecx = 0
    mov         ebx, .factor

    .while:
    mov         cl, [esi]           ; grab first char
    test        cl, cl              ; check cl to see if zero
    jz          .wend               ; exit if zero
    sub         cl, .offset         ;calculate integer from ASCII
    mul         ebx                 ; eax = eax *10
    add         eax, ecx            ; add next character to eax
    
    inc         esi
    jmp         .while
    .wend:

    pop         esi                 ; restore
    pop         ebp                 ; restore

    ret


sum_array:

;Description: Sums the values of an double word array
;Receives:       [ebp + 8] = Address of an array 8 bytes from esp
;                [ebp + 12] =  number of elements 12 bytes from stack
;                  
;returns:           EAX = sum
;requirment         preserve ebp

    push        ebp                 ; preserve
    mov         ebp, esp            ; set new stack frame

    push        esi 
   
    mov         esi, [ebp+8]        ;esi = array source address
    xor         eax,eax             ; eax = sum 
    mov         ecx, [ebp+12]       ; ECX = counter

    .loop:
    add         eax, [esi]
    add         esi, 4
    loop    .loop

    pop         esi                 ; restore
    pop         ebp                 ; restore

    ret


print_char:

;description: prints a single character
; receives:     [ebp + 8] = char value
; Requires:     1) preserve ebp
; Requires:    a buffer provided below

    push        ebp                 ; preserve
    mov         ebp, esp            ; set new stack frame

    mov         al,  [ebp +11]
    mov         [char_buff], al
    mov         eax, 4
    mov         ebx, 1
    mov         ecx, char_buff
    mov         edx, 1
    int         80h

    pop         ebp                 ;restore

    ret

rev_str:
;description: reverses a string 
;receives:    [ebp + 8] = address on string
;             [ebp + 12] = the sizes in bytes of ths string
;returns:     n/a

    push        ebp                 ; preserve
    mov         ebp, esp            ; set new stack frame

    push        esi                 ;preserve esi
    push        edi                 ;preserve edi

    mov         eax, [ebp +12]
    test        [ebp + 12], eax      
    jnz         .start

    mov         dword [errno], -1
    jmp         .exit

    .start:
    mov         ecx, [ebp + 12]     ;set counter
    mov         esi, [ebp + 8]      ;SOURCE
    mov         edi, [ebp + 8]      ;DESTination

    .pushloop:                      ;push loop
    movzx       edx, byte [esi]     ;mov into register char
    push        edx                 ;push unto stack a char
    inc         esi                 ;calculate effective address
    loop    .pushloop

    mov         ecx, [ebp + 12]     ;reset counter

    .poploop:                       ;push loop
    pop         edx                 ;mov into register char
    mov         [edi], dl
    inc         edi                 ;calculate effective address
    loop    .poploop

    .exit:
    pop         edi             ;restore
    pop         esi             ;restore
    pop         ebp             ;restore

    ret

get_input:

;returns    EAX = inputted string
;recevies   [ebp + 8] = prompt address
;           [ebp + 12] = prompt size
;           [ebp + 16] = input buffer address
;           [ebp + 20] = input buffer size

;get string from user

    push        ebp                 ; preserve
    mov         ebp, esp            ; set new stack frame


    push        dword [ebp + 16]    ;preserve 
    push        dword [ebp + 20]    ;preserve 
    push        dword [ebp + 12]    ;preserve 
    push        dword [ebp + 8]     ;preserve edi
    
    mov         edi, [ebp + 16]     ; move buffer address into edi

    call        print_str           ;print prompt

    mov         eax, 3              ;read op
    mov         ebx, 0              ;read from stdin
    pop         edx                 ; edx = input buffer size
    pop         ecx                 ; ecx = input buffer 
    int         80h

    mov         BYTE [edi + eax], 0

    pop         edi                 ;restore
    pop         ebx                 ;restore
    pop         ebp                 ;restore

    ret 

print_str:
;receives arg1 = address of string
;         arg2 = size of str

    push        ebp                     ; preserve
    mov         ebp, esp                ; set new stack frame

    mov         ecx, [ebp + 8]          ; mov address into ecx
    mov         edx, [ebp + 12]         ; mov size into edx
    mov         eax, 4                  ; write op
    mov         ebx, 1                  ; std out
    int         80h
    
    ;eax holds the quantity of the characters input
    dec         eax                     ; chuck newline character
    mov         byte [edi + eax], 0

    pop         ebp                     ;restore
    ret
    
swap:

    push        ebp             ;preserve base pointer
    mov         ebp, esp        ;setup frame

    push        esi
    push        edi
    mov         esi, [ebp +8]   ;esi holds first address
    mov         edi, [ebp +12]  ;edi holds second address
    mov         eax, [esi]      ; eax -> temp = val at first address
    xchg        eax, [edi]      ; swap eax with val at second address
    mov         [esi], eax      ; store temp at first address

    pop         edi             ;restore
    pop         esi             ;restore

    mov         esp, ebp
    pop         ebp             ; restore caller's base pointer

    ret

itoa:
;description: convert integer to string
; receives: arg1: integer
;           arg2: string buffer address (assume its big enough)
; returns: nothing
;requires: nothing

    push        ebp             ;preserve caller's base 
    mov         ebp, esp        ; set base of frame
    push        edi             ; preserve reg
    push        ebx             ; preserve reg

    mov         ebx, 10
    mov         eax, [ebp + 8]  ; store integer in eax
    mov         edi, [ebp +12]  ;store address in edi

    .do:
    xor         edx, edx        ; clear edx
    div         ebx             ; divide by 10
    add         dl, 48          ; 
    mov         [edi], dl       ; store char in buffer
    inc         edi




    test        eax, eax
    jnz         .do
    .wend:
;repeatedly divide by 10 and the remainder +48 as the character 

    mov         byte [edi], 0
    pop         ebx
    pop         edi
    leave 
    ret



section .bss 
    char_buff_sz:   equ 255
    char_buff:      resb char_buff_sz

    input_sz: equ 255
    input: resb input_sz

    

section     .data
    errno: dd 0

    next:   dd    1
    const1: equ 1103515245
    const2: equ 12345
    const3: equ 65536
    const4: equ 32768
    

