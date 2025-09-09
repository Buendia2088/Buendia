
ctarget:     file format elf64-x86-64


Disassembly of section .init:

0000000000400cc0 <_init>:
  400cc0:	48 83 ec 08          	sub    $0x8,%rsp
  400cc4:	48 8b 05 2d 33 20 00 	mov    0x20332d(%rip),%rax        # 603ff8 <__gmon_start__>
  400ccb:	48 85 c0             	test   %rax,%rax
  400cce:	74 05                	je     400cd5 <_init+0x15>
  400cd0:	e8 3b 02 00 00       	call   400f10 <__gmon_start__@plt>
  400cd5:	48 83 c4 08          	add    $0x8,%rsp
  400cd9:	c3                   	ret    

Disassembly of section .plt:

0000000000400ce0 <.plt>:
  400ce0:	ff 35 22 33 20 00    	push   0x203322(%rip)        # 604008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400ce6:	ff 25 24 33 20 00    	jmp    *0x203324(%rip)        # 604010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400cec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cf0 <__printf_chk@plt>:
  400cf0:	ff 25 22 33 20 00    	jmp    *0x203322(%rip)        # 604018 <__printf_chk>
  400cf6:	68 00 00 00 00       	push   $0x0
  400cfb:	e9 e0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d00 <strcasecmp@plt>:
  400d00:	ff 25 1a 33 20 00    	jmp    *0x20331a(%rip)        # 604020 <strcasecmp@GLIBC_2.2.5>
  400d06:	68 01 00 00 00       	push   $0x1
  400d0b:	e9 d0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d10 <__errno_location@plt>:
  400d10:	ff 25 12 33 20 00    	jmp    *0x203312(%rip)        # 604028 <__errno_location@GLIBC_2.2.5>
  400d16:	68 02 00 00 00       	push   $0x2
  400d1b:	e9 c0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d20 <srandom@plt>:
  400d20:	ff 25 0a 33 20 00    	jmp    *0x20330a(%rip)        # 604030 <srandom@GLIBC_2.2.5>
  400d26:	68 03 00 00 00       	push   $0x3
  400d2b:	e9 b0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d30 <strncmp@plt>:
  400d30:	ff 25 02 33 20 00    	jmp    *0x203302(%rip)        # 604038 <strncmp@GLIBC_2.2.5>
  400d36:	68 04 00 00 00       	push   $0x4
  400d3b:	e9 a0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d40 <strcpy@plt>:
  400d40:	ff 25 fa 32 20 00    	jmp    *0x2032fa(%rip)        # 604040 <strcpy@GLIBC_2.2.5>
  400d46:	68 05 00 00 00       	push   $0x5
  400d4b:	e9 90 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d50 <puts@plt>:
  400d50:	ff 25 f2 32 20 00    	jmp    *0x2032f2(%rip)        # 604048 <puts@GLIBC_2.2.5>
  400d56:	68 06 00 00 00       	push   $0x6
  400d5b:	e9 80 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d60 <write@plt>:
  400d60:	ff 25 ea 32 20 00    	jmp    *0x2032ea(%rip)        # 604050 <write@GLIBC_2.2.5>
  400d66:	68 07 00 00 00       	push   $0x7
  400d6b:	e9 70 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d70 <__stack_chk_fail@plt>:
  400d70:	ff 25 e2 32 20 00    	jmp    *0x2032e2(%rip)        # 604058 <__stack_chk_fail@GLIBC_2.4>
  400d76:	68 08 00 00 00       	push   $0x8
  400d7b:	e9 60 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d80 <mmap@plt>:
  400d80:	ff 25 da 32 20 00    	jmp    *0x2032da(%rip)        # 604060 <mmap@GLIBC_2.2.5>
  400d86:	68 09 00 00 00       	push   $0x9
  400d8b:	e9 50 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d90 <memset@plt>:
  400d90:	ff 25 d2 32 20 00    	jmp    *0x2032d2(%rip)        # 604068 <memset@GLIBC_2.2.5>
  400d96:	68 0a 00 00 00       	push   $0xa
  400d9b:	e9 40 ff ff ff       	jmp    400ce0 <.plt>

0000000000400da0 <alarm@plt>:
  400da0:	ff 25 ca 32 20 00    	jmp    *0x2032ca(%rip)        # 604070 <alarm@GLIBC_2.2.5>
  400da6:	68 0b 00 00 00       	push   $0xb
  400dab:	e9 30 ff ff ff       	jmp    400ce0 <.plt>

0000000000400db0 <close@plt>:
  400db0:	ff 25 c2 32 20 00    	jmp    *0x2032c2(%rip)        # 604078 <close@GLIBC_2.2.5>
  400db6:	68 0c 00 00 00       	push   $0xc
  400dbb:	e9 20 ff ff ff       	jmp    400ce0 <.plt>

0000000000400dc0 <read@plt>:
  400dc0:	ff 25 ba 32 20 00    	jmp    *0x2032ba(%rip)        # 604080 <read@GLIBC_2.2.5>
  400dc6:	68 0d 00 00 00       	push   $0xd
  400dcb:	e9 10 ff ff ff       	jmp    400ce0 <.plt>

0000000000400dd0 <__libc_start_main@plt>:
  400dd0:	ff 25 b2 32 20 00    	jmp    *0x2032b2(%rip)        # 604088 <__libc_start_main@GLIBC_2.2.5>
  400dd6:	68 0e 00 00 00       	push   $0xe
  400ddb:	e9 00 ff ff ff       	jmp    400ce0 <.plt>

0000000000400de0 <signal@plt>:
  400de0:	ff 25 aa 32 20 00    	jmp    *0x2032aa(%rip)        # 604090 <signal@GLIBC_2.2.5>
  400de6:	68 0f 00 00 00       	push   $0xf
  400deb:	e9 f0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400df0 <gethostbyname@plt>:
  400df0:	ff 25 a2 32 20 00    	jmp    *0x2032a2(%rip)        # 604098 <gethostbyname@GLIBC_2.2.5>
  400df6:	68 10 00 00 00       	push   $0x10
  400dfb:	e9 e0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e00 <__memmove_chk@plt>:
  400e00:	ff 25 9a 32 20 00    	jmp    *0x20329a(%rip)        # 6040a0 <__memmove_chk@GLIBC_2.3.4>
  400e06:	68 11 00 00 00       	push   $0x11
  400e0b:	e9 d0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e10 <strtol@plt>:
  400e10:	ff 25 92 32 20 00    	jmp    *0x203292(%rip)        # 6040a8 <strtol@GLIBC_2.2.5>
  400e16:	68 12 00 00 00       	push   $0x12
  400e1b:	e9 c0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e20 <memcpy@plt>:
  400e20:	ff 25 8a 32 20 00    	jmp    *0x20328a(%rip)        # 6040b0 <memcpy@GLIBC_2.14>
  400e26:	68 13 00 00 00       	push   $0x13
  400e2b:	e9 b0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e30 <__sprintf_chk@plt>:
  400e30:	ff 25 82 32 20 00    	jmp    *0x203282(%rip)        # 6040b8 <__sprintf_chk>
  400e36:	68 14 00 00 00       	push   $0x14
  400e3b:	e9 a0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e40 <time@plt>:
  400e40:	ff 25 7a 32 20 00    	jmp    *0x20327a(%rip)        # 6040c0 <time@GLIBC_2.2.5>
  400e46:	68 15 00 00 00       	push   $0x15
  400e4b:	e9 90 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e50 <random@plt>:
  400e50:	ff 25 72 32 20 00    	jmp    *0x203272(%rip)        # 6040c8 <random@GLIBC_2.2.5>
  400e56:	68 16 00 00 00       	push   $0x16
  400e5b:	e9 80 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e60 <_IO_getc@plt>:
  400e60:	ff 25 6a 32 20 00    	jmp    *0x20326a(%rip)        # 6040d0 <_IO_getc@GLIBC_2.2.5>
  400e66:	68 17 00 00 00       	push   $0x17
  400e6b:	e9 70 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e70 <__isoc99_sscanf@plt>:
  400e70:	ff 25 62 32 20 00    	jmp    *0x203262(%rip)        # 6040d8 <__isoc99_sscanf@GLIBC_2.7>
  400e76:	68 18 00 00 00       	push   $0x18
  400e7b:	e9 60 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e80 <munmap@plt>:
  400e80:	ff 25 5a 32 20 00    	jmp    *0x20325a(%rip)        # 6040e0 <munmap@GLIBC_2.2.5>
  400e86:	68 19 00 00 00       	push   $0x19
  400e8b:	e9 50 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e90 <fopen@plt>:
  400e90:	ff 25 52 32 20 00    	jmp    *0x203252(%rip)        # 6040e8 <fopen@GLIBC_2.2.5>
  400e96:	68 1a 00 00 00       	push   $0x1a
  400e9b:	e9 40 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ea0 <getopt@plt>:
  400ea0:	ff 25 4a 32 20 00    	jmp    *0x20324a(%rip)        # 6040f0 <getopt@GLIBC_2.2.5>
  400ea6:	68 1b 00 00 00       	push   $0x1b
  400eab:	e9 30 fe ff ff       	jmp    400ce0 <.plt>

0000000000400eb0 <strtoul@plt>:
  400eb0:	ff 25 42 32 20 00    	jmp    *0x203242(%rip)        # 6040f8 <strtoul@GLIBC_2.2.5>
  400eb6:	68 1c 00 00 00       	push   $0x1c
  400ebb:	e9 20 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ec0 <gethostname@plt>:
  400ec0:	ff 25 3a 32 20 00    	jmp    *0x20323a(%rip)        # 604100 <gethostname@GLIBC_2.2.5>
  400ec6:	68 1d 00 00 00       	push   $0x1d
  400ecb:	e9 10 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ed0 <exit@plt>:
  400ed0:	ff 25 32 32 20 00    	jmp    *0x203232(%rip)        # 604108 <exit@GLIBC_2.2.5>
  400ed6:	68 1e 00 00 00       	push   $0x1e
  400edb:	e9 00 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ee0 <connect@plt>:
  400ee0:	ff 25 2a 32 20 00    	jmp    *0x20322a(%rip)        # 604110 <connect@GLIBC_2.2.5>
  400ee6:	68 1f 00 00 00       	push   $0x1f
  400eeb:	e9 f0 fd ff ff       	jmp    400ce0 <.plt>

0000000000400ef0 <__fprintf_chk@plt>:
  400ef0:	ff 25 22 32 20 00    	jmp    *0x203222(%rip)        # 604118 <__fprintf_chk@GLIBC_2.3.4>
  400ef6:	68 20 00 00 00       	push   $0x20
  400efb:	e9 e0 fd ff ff       	jmp    400ce0 <.plt>

0000000000400f00 <socket@plt>:
  400f00:	ff 25 1a 32 20 00    	jmp    *0x20321a(%rip)        # 604120 <socket@GLIBC_2.2.5>
  400f06:	68 21 00 00 00       	push   $0x21
  400f0b:	e9 d0 fd ff ff       	jmp    400ce0 <.plt>

Disassembly of section .plt.got:

0000000000400f10 <__gmon_start__@plt>:
  400f10:	ff 25 e2 30 20 00    	jmp    *0x2030e2(%rip)        # 603ff8 <__gmon_start__>
  400f16:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400f20 <_start>:
  400f20:	31 ed                	xor    %ebp,%ebp
  400f22:	49 89 d1             	mov    %rdx,%r9
  400f25:	5e                   	pop    %rsi
  400f26:	48 89 e2             	mov    %rsp,%rdx
  400f29:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  400f2d:	50                   	push   %rax
  400f2e:	54                   	push   %rsp
  400f2f:	49 c7 c0 60 2e 40 00 	mov    $0x402e60,%r8
  400f36:	48 c7 c1 f0 2d 40 00 	mov    $0x402df0,%rcx
  400f3d:	48 c7 c7 25 12 40 00 	mov    $0x401225,%rdi
  400f44:	e8 87 fe ff ff       	call   400dd0 <__libc_start_main@plt>
  400f49:	f4                   	hlt    
  400f4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400f50 <deregister_tm_clones>:
  400f50:	b8 b7 44 60 00       	mov    $0x6044b7,%eax
  400f55:	55                   	push   %rbp
  400f56:	48 2d b0 44 60 00    	sub    $0x6044b0,%rax
  400f5c:	48 83 f8 0e          	cmp    $0xe,%rax
  400f60:	48 89 e5             	mov    %rsp,%rbp
  400f63:	76 1b                	jbe    400f80 <deregister_tm_clones+0x30>
  400f65:	b8 00 00 00 00       	mov    $0x0,%eax
  400f6a:	48 85 c0             	test   %rax,%rax
  400f6d:	74 11                	je     400f80 <deregister_tm_clones+0x30>
  400f6f:	5d                   	pop    %rbp
  400f70:	bf b0 44 60 00       	mov    $0x6044b0,%edi
  400f75:	ff e0                	jmp    *%rax
  400f77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f7e:	00 00 
  400f80:	5d                   	pop    %rbp
  400f81:	c3                   	ret    
  400f82:	0f 1f 40 00          	nopl   0x0(%rax)
  400f86:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  400f8d:	00 00 00 

0000000000400f90 <register_tm_clones>:
  400f90:	be b0 44 60 00       	mov    $0x6044b0,%esi
  400f95:	55                   	push   %rbp
  400f96:	48 81 ee b0 44 60 00 	sub    $0x6044b0,%rsi
  400f9d:	48 c1 fe 03          	sar    $0x3,%rsi
  400fa1:	48 89 e5             	mov    %rsp,%rbp
  400fa4:	48 89 f0             	mov    %rsi,%rax
  400fa7:	48 c1 e8 3f          	shr    $0x3f,%rax
  400fab:	48 01 c6             	add    %rax,%rsi
  400fae:	48 d1 fe             	sar    %rsi
  400fb1:	74 15                	je     400fc8 <register_tm_clones+0x38>
  400fb3:	b8 00 00 00 00       	mov    $0x0,%eax
  400fb8:	48 85 c0             	test   %rax,%rax
  400fbb:	74 0b                	je     400fc8 <register_tm_clones+0x38>
  400fbd:	5d                   	pop    %rbp
  400fbe:	bf b0 44 60 00       	mov    $0x6044b0,%edi
  400fc3:	ff e0                	jmp    *%rax
  400fc5:	0f 1f 00             	nopl   (%rax)
  400fc8:	5d                   	pop    %rbp
  400fc9:	c3                   	ret    
  400fca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400fd0 <__do_global_dtors_aux>:
  400fd0:	80 3d 11 35 20 00 00 	cmpb   $0x0,0x203511(%rip)        # 6044e8 <completed.7594>
  400fd7:	75 11                	jne    400fea <__do_global_dtors_aux+0x1a>
  400fd9:	55                   	push   %rbp
  400fda:	48 89 e5             	mov    %rsp,%rbp
  400fdd:	e8 6e ff ff ff       	call   400f50 <deregister_tm_clones>
  400fe2:	5d                   	pop    %rbp
  400fe3:	c6 05 fe 34 20 00 01 	movb   $0x1,0x2034fe(%rip)        # 6044e8 <completed.7594>
  400fea:	f3 c3                	repz ret 
  400fec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ff0 <frame_dummy>:
  400ff0:	bf 10 3e 60 00       	mov    $0x603e10,%edi
  400ff5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400ff9:	75 05                	jne    401000 <frame_dummy+0x10>
  400ffb:	eb 93                	jmp    400f90 <register_tm_clones>
  400ffd:	0f 1f 00             	nopl   (%rax)
  401000:	b8 00 00 00 00       	mov    $0x0,%eax
  401005:	48 85 c0             	test   %rax,%rax
  401008:	74 f1                	je     400ffb <frame_dummy+0xb>
  40100a:	55                   	push   %rbp
  40100b:	48 89 e5             	mov    %rsp,%rbp
  40100e:	ff d0                	call   *%rax
  401010:	5d                   	pop    %rbp
  401011:	e9 7a ff ff ff       	jmp    400f90 <register_tm_clones>

0000000000401016 <usage>:
  401016:	48 83 ec 08          	sub    $0x8,%rsp
  40101a:	48 89 fa             	mov    %rdi,%rdx
  40101d:	83 3d 08 35 20 00 00 	cmpl   $0x0,0x203508(%rip)        # 60452c <is_checker>
  401024:	74 3e                	je     401064 <usage+0x4e>
  401026:	be 78 2e 40 00       	mov    $0x402e78,%esi
  40102b:	bf 01 00 00 00       	mov    $0x1,%edi
  401030:	b8 00 00 00 00       	mov    $0x0,%eax
  401035:	e8 b6 fc ff ff       	call   400cf0 <__printf_chk@plt>
  40103a:	bf b0 2e 40 00       	mov    $0x402eb0,%edi
  40103f:	e8 0c fd ff ff       	call   400d50 <puts@plt>
  401044:	bf 28 30 40 00       	mov    $0x403028,%edi
  401049:	e8 02 fd ff ff       	call   400d50 <puts@plt>
  40104e:	bf d8 2e 40 00       	mov    $0x402ed8,%edi
  401053:	e8 f8 fc ff ff       	call   400d50 <puts@plt>
  401058:	bf 42 30 40 00       	mov    $0x403042,%edi
  40105d:	e8 ee fc ff ff       	call   400d50 <puts@plt>
  401062:	eb 32                	jmp    401096 <usage+0x80>
  401064:	be 5e 30 40 00       	mov    $0x40305e,%esi
  401069:	bf 01 00 00 00       	mov    $0x1,%edi
  40106e:	b8 00 00 00 00       	mov    $0x0,%eax
  401073:	e8 78 fc ff ff       	call   400cf0 <__printf_chk@plt>
  401078:	bf 00 2f 40 00       	mov    $0x402f00,%edi
  40107d:	e8 ce fc ff ff       	call   400d50 <puts@plt>
  401082:	bf 28 2f 40 00       	mov    $0x402f28,%edi
  401087:	e8 c4 fc ff ff       	call   400d50 <puts@plt>
  40108c:	bf 7c 30 40 00       	mov    $0x40307c,%edi
  401091:	e8 ba fc ff ff       	call   400d50 <puts@plt>
  401096:	bf 00 00 00 00       	mov    $0x0,%edi
  40109b:	e8 30 fe ff ff       	call   400ed0 <exit@plt>

00000000004010a0 <initialize_target>:
  4010a0:	55                   	push   %rbp
  4010a1:	53                   	push   %rbx
  4010a2:	48 81 ec 18 21 00 00 	sub    $0x2118,%rsp
  4010a9:	89 f5                	mov    %esi,%ebp
  4010ab:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4010b2:	00 00 
  4010b4:	48 89 84 24 08 21 00 	mov    %rax,0x2108(%rsp)
  4010bb:	00 
  4010bc:	31 c0                	xor    %eax,%eax
  4010be:	89 3d 58 34 20 00    	mov    %edi,0x203458(%rip)        # 60451c <check_level>
  4010c4:	8b 3d 9e 30 20 00    	mov    0x20309e(%rip),%edi        # 604168 <target_id>
  4010ca:	e8 ff 1c 00 00       	call   402dce <gencookie>
  4010cf:	89 05 53 34 20 00    	mov    %eax,0x203453(%rip)        # 604528 <cookie>
  4010d5:	89 c7                	mov    %eax,%edi
  4010d7:	e8 f2 1c 00 00       	call   402dce <gencookie>
  4010dc:	89 05 42 34 20 00    	mov    %eax,0x203442(%rip)        # 604524 <authkey>
  4010e2:	8b 05 80 30 20 00    	mov    0x203080(%rip),%eax        # 604168 <target_id>
  4010e8:	8d 78 01             	lea    0x1(%rax),%edi
  4010eb:	e8 30 fc ff ff       	call   400d20 <srandom@plt>
  4010f0:	e8 5b fd ff ff       	call   400e50 <random@plt>
  4010f5:	89 c7                	mov    %eax,%edi
  4010f7:	e8 03 03 00 00       	call   4013ff <scramble>
  4010fc:	89 c3                	mov    %eax,%ebx
  4010fe:	85 ed                	test   %ebp,%ebp
  401100:	74 18                	je     40111a <initialize_target+0x7a>
  401102:	bf 00 00 00 00       	mov    $0x0,%edi
  401107:	e8 34 fd ff ff       	call   400e40 <time@plt>
  40110c:	89 c7                	mov    %eax,%edi
  40110e:	e8 0d fc ff ff       	call   400d20 <srandom@plt>
  401113:	e8 38 fd ff ff       	call   400e50 <random@plt>
  401118:	eb 05                	jmp    40111f <initialize_target+0x7f>
  40111a:	b8 00 00 00 00       	mov    $0x0,%eax
  40111f:	01 c3                	add    %eax,%ebx
  401121:	0f b7 db             	movzwl %bx,%ebx
  401124:	8d 04 dd 00 01 00 00 	lea    0x100(,%rbx,8),%eax
  40112b:	89 c0                	mov    %eax,%eax
  40112d:	48 89 05 74 33 20 00 	mov    %rax,0x203374(%rip)        # 6044a8 <buf_offset>
  401134:	c6 05 15 40 20 00 63 	movb   $0x63,0x204015(%rip)        # 605150 <target_prefix>
  40113b:	83 3d d6 33 20 00 00 	cmpl   $0x0,0x2033d6(%rip)        # 604518 <notify>
  401142:	0f 84 bb 00 00 00    	je     401203 <initialize_target+0x163>
  401148:	83 3d dd 33 20 00 00 	cmpl   $0x0,0x2033dd(%rip)        # 60452c <is_checker>
  40114f:	0f 85 ae 00 00 00    	jne    401203 <initialize_target+0x163>
  401155:	be 00 01 00 00       	mov    $0x100,%esi
  40115a:	48 89 e7             	mov    %rsp,%rdi
  40115d:	e8 5e fd ff ff       	call   400ec0 <gethostname@plt>
  401162:	85 c0                	test   %eax,%eax
  401164:	74 25                	je     40118b <initialize_target+0xeb>
  401166:	bf 58 2f 40 00       	mov    $0x402f58,%edi
  40116b:	e8 e0 fb ff ff       	call   400d50 <puts@plt>
  401170:	bf 08 00 00 00       	mov    $0x8,%edi
  401175:	e8 56 fd ff ff       	call   400ed0 <exit@plt>
  40117a:	48 89 e6             	mov    %rsp,%rsi
  40117d:	e8 7e fb ff ff       	call   400d00 <strcasecmp@plt>
  401182:	85 c0                	test   %eax,%eax
  401184:	74 21                	je     4011a7 <initialize_target+0x107>
  401186:	83 c3 01             	add    $0x1,%ebx
  401189:	eb 05                	jmp    401190 <initialize_target+0xf0>
  40118b:	bb 00 00 00 00       	mov    $0x0,%ebx
  401190:	48 63 c3             	movslq %ebx,%rax
  401193:	48 8b 3c c5 80 41 60 	mov    0x604180(,%rax,8),%rdi
  40119a:	00 
  40119b:	48 85 ff             	test   %rdi,%rdi
  40119e:	75 da                	jne    40117a <initialize_target+0xda>
  4011a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4011a5:	eb 05                	jmp    4011ac <initialize_target+0x10c>
  4011a7:	b8 01 00 00 00       	mov    $0x1,%eax
  4011ac:	85 c0                	test   %eax,%eax
  4011ae:	75 1c                	jne    4011cc <initialize_target+0x12c>
  4011b0:	48 89 e2             	mov    %rsp,%rdx
  4011b3:	be 90 2f 40 00       	mov    $0x402f90,%esi
  4011b8:	bf 01 00 00 00       	mov    $0x1,%edi
  4011bd:	e8 2e fb ff ff       	call   400cf0 <__printf_chk@plt>
  4011c2:	bf 08 00 00 00       	mov    $0x8,%edi
  4011c7:	e8 04 fd ff ff       	call   400ed0 <exit@plt>
  4011cc:	48 8d bc 24 00 01 00 	lea    0x100(%rsp),%rdi
  4011d3:	00 
  4011d4:	e8 5f 19 00 00       	call   402b38 <init_driver>
  4011d9:	85 c0                	test   %eax,%eax
  4011db:	79 26                	jns    401203 <initialize_target+0x163>
  4011dd:	48 8d 94 24 00 01 00 	lea    0x100(%rsp),%rdx
  4011e4:	00 
  4011e5:	be d0 2f 40 00       	mov    $0x402fd0,%esi
  4011ea:	bf 01 00 00 00       	mov    $0x1,%edi
  4011ef:	b8 00 00 00 00       	mov    $0x0,%eax
  4011f4:	e8 f7 fa ff ff       	call   400cf0 <__printf_chk@plt>
  4011f9:	bf 08 00 00 00       	mov    $0x8,%edi
  4011fe:	e8 cd fc ff ff       	call   400ed0 <exit@plt>
  401203:	48 8b 84 24 08 21 00 	mov    0x2108(%rsp),%rax
  40120a:	00 
  40120b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401212:	00 00 
  401214:	74 05                	je     40121b <initialize_target+0x17b>
  401216:	e8 55 fb ff ff       	call   400d70 <__stack_chk_fail@plt>
  40121b:	48 81 c4 18 21 00 00 	add    $0x2118,%rsp
  401222:	5b                   	pop    %rbx
  401223:	5d                   	pop    %rbp
  401224:	c3                   	ret    

0000000000401225 <main>:
  401225:	41 56                	push   %r14
  401227:	41 55                	push   %r13
  401229:	41 54                	push   %r12
  40122b:	55                   	push   %rbp
  40122c:	53                   	push   %rbx
  40122d:	41 89 fc             	mov    %edi,%r12d
  401230:	48 89 f3             	mov    %rsi,%rbx
  401233:	be 73 1e 40 00       	mov    $0x401e73,%esi
  401238:	bf 0b 00 00 00       	mov    $0xb,%edi
  40123d:	e8 9e fb ff ff       	call   400de0 <signal@plt>
  401242:	be 25 1e 40 00       	mov    $0x401e25,%esi
  401247:	bf 07 00 00 00       	mov    $0x7,%edi
  40124c:	e8 8f fb ff ff       	call   400de0 <signal@plt>
  401251:	be c1 1e 40 00       	mov    $0x401ec1,%esi
  401256:	bf 04 00 00 00       	mov    $0x4,%edi
  40125b:	e8 80 fb ff ff       	call   400de0 <signal@plt>
  401260:	83 3d c5 32 20 00 00 	cmpl   $0x0,0x2032c5(%rip)        # 60452c <is_checker>
  401267:	74 20                	je     401289 <main+0x64>
  401269:	be 0f 1f 40 00       	mov    $0x401f0f,%esi
  40126e:	bf 0e 00 00 00       	mov    $0xe,%edi
  401273:	e8 68 fb ff ff       	call   400de0 <signal@plt>
  401278:	bf 05 00 00 00       	mov    $0x5,%edi
  40127d:	e8 1e fb ff ff       	call   400da0 <alarm@plt>
  401282:	bd 9a 30 40 00       	mov    $0x40309a,%ebp
  401287:	eb 05                	jmp    40128e <main+0x69>
  401289:	bd 95 30 40 00       	mov    $0x403095,%ebp
  40128e:	48 8b 05 2b 32 20 00 	mov    0x20322b(%rip),%rax        # 6044c0 <stdin@GLIBC_2.2.5>
  401295:	48 89 05 74 32 20 00 	mov    %rax,0x203274(%rip)        # 604510 <infile>
  40129c:	41 bd 00 00 00 00    	mov    $0x0,%r13d
  4012a2:	41 be 00 00 00 00    	mov    $0x0,%r14d
  4012a8:	e9 c6 00 00 00       	jmp    401373 <main+0x14e>
  4012ad:	83 e8 61             	sub    $0x61,%eax
  4012b0:	3c 10                	cmp    $0x10,%al
  4012b2:	0f 87 9c 00 00 00    	ja     401354 <main+0x12f>
  4012b8:	0f b6 c0             	movzbl %al,%eax
  4012bb:	ff 24 c5 e0 30 40 00 	jmp    *0x4030e0(,%rax,8)
  4012c2:	48 8b 3b             	mov    (%rbx),%rdi
  4012c5:	e8 4c fd ff ff       	call   401016 <usage>
  4012ca:	be 22 33 40 00       	mov    $0x403322,%esi
  4012cf:	48 8b 3d f2 31 20 00 	mov    0x2031f2(%rip),%rdi        # 6044c8 <optarg@GLIBC_2.2.5>
  4012d6:	e8 b5 fb ff ff       	call   400e90 <fopen@plt>
  4012db:	48 89 05 2e 32 20 00 	mov    %rax,0x20322e(%rip)        # 604510 <infile>
  4012e2:	48 85 c0             	test   %rax,%rax
  4012e5:	0f 85 88 00 00 00    	jne    401373 <main+0x14e>
  4012eb:	48 8b 0d d6 31 20 00 	mov    0x2031d6(%rip),%rcx        # 6044c8 <optarg@GLIBC_2.2.5>
  4012f2:	ba a2 30 40 00       	mov    $0x4030a2,%edx
  4012f7:	be 01 00 00 00       	mov    $0x1,%esi
  4012fc:	48 8b 3d dd 31 20 00 	mov    0x2031dd(%rip),%rdi        # 6044e0 <stderr@GLIBC_2.2.5>
  401303:	e8 e8 fb ff ff       	call   400ef0 <__fprintf_chk@plt>
  401308:	b8 01 00 00 00       	mov    $0x1,%eax
  40130d:	e9 e4 00 00 00       	jmp    4013f6 <main+0x1d1>
  401312:	ba 10 00 00 00       	mov    $0x10,%edx
  401317:	be 00 00 00 00       	mov    $0x0,%esi
  40131c:	48 8b 3d a5 31 20 00 	mov    0x2031a5(%rip),%rdi        # 6044c8 <optarg@GLIBC_2.2.5>
  401323:	e8 88 fb ff ff       	call   400eb0 <strtoul@plt>
  401328:	41 89 c6             	mov    %eax,%r14d
  40132b:	eb 46                	jmp    401373 <main+0x14e>
  40132d:	ba 0a 00 00 00       	mov    $0xa,%edx
  401332:	be 00 00 00 00       	mov    $0x0,%esi
  401337:	48 8b 3d 8a 31 20 00 	mov    0x20318a(%rip),%rdi        # 6044c8 <optarg@GLIBC_2.2.5>
  40133e:	e8 cd fa ff ff       	call   400e10 <strtol@plt>
  401343:	41 89 c5             	mov    %eax,%r13d
  401346:	eb 2b                	jmp    401373 <main+0x14e>
  401348:	c7 05 c6 31 20 00 00 	movl   $0x0,0x2031c6(%rip)        # 604518 <notify>
  40134f:	00 00 00 
  401352:	eb 1f                	jmp    401373 <main+0x14e>
  401354:	0f be d2             	movsbl %dl,%edx
  401357:	be bf 30 40 00       	mov    $0x4030bf,%esi
  40135c:	bf 01 00 00 00       	mov    $0x1,%edi
  401361:	b8 00 00 00 00       	mov    $0x0,%eax
  401366:	e8 85 f9 ff ff       	call   400cf0 <__printf_chk@plt>
  40136b:	48 8b 3b             	mov    (%rbx),%rdi
  40136e:	e8 a3 fc ff ff       	call   401016 <usage>
  401373:	48 89 ea             	mov    %rbp,%rdx
  401376:	48 89 de             	mov    %rbx,%rsi
  401379:	44 89 e7             	mov    %r12d,%edi
  40137c:	e8 1f fb ff ff       	call   400ea0 <getopt@plt>
  401381:	89 c2                	mov    %eax,%edx
  401383:	3c ff                	cmp    $0xff,%al
  401385:	0f 85 22 ff ff ff    	jne    4012ad <main+0x88>
  40138b:	be 00 00 00 00       	mov    $0x0,%esi
  401390:	44 89 ef             	mov    %r13d,%edi
  401393:	e8 08 fd ff ff       	call   4010a0 <initialize_target>
  401398:	83 3d 8d 31 20 00 00 	cmpl   $0x0,0x20318d(%rip)        # 60452c <is_checker>
  40139f:	74 2a                	je     4013cb <main+0x1a6>
  4013a1:	44 3b 35 7c 31 20 00 	cmp    0x20317c(%rip),%r14d        # 604524 <authkey>
  4013a8:	74 21                	je     4013cb <main+0x1a6>
  4013aa:	44 89 f2             	mov    %r14d,%edx
  4013ad:	be f8 2f 40 00       	mov    $0x402ff8,%esi
  4013b2:	bf 01 00 00 00       	mov    $0x1,%edi
  4013b7:	b8 00 00 00 00       	mov    $0x0,%eax
  4013bc:	e8 2f f9 ff ff       	call   400cf0 <__printf_chk@plt>
  4013c1:	b8 00 00 00 00       	mov    $0x0,%eax
  4013c6:	e8 fb 07 00 00       	call   401bc6 <check_fail>
  4013cb:	8b 15 57 31 20 00    	mov    0x203157(%rip),%edx        # 604528 <cookie>
  4013d1:	be d2 30 40 00       	mov    $0x4030d2,%esi
  4013d6:	bf 01 00 00 00       	mov    $0x1,%edi
  4013db:	b8 00 00 00 00       	mov    $0x0,%eax
  4013e0:	e8 0b f9 ff ff       	call   400cf0 <__printf_chk@plt>
  4013e5:	48 8b 3d bc 30 20 00 	mov    0x2030bc(%rip),%rdi        # 6044a8 <buf_offset>
  4013ec:	e8 1e 0c 00 00       	call   40200f <stable_launch>
  4013f1:	b8 00 00 00 00       	mov    $0x0,%eax
  4013f6:	5b                   	pop    %rbx
  4013f7:	5d                   	pop    %rbp
  4013f8:	41 5c                	pop    %r12
  4013fa:	41 5d                	pop    %r13
  4013fc:	41 5e                	pop    %r14
  4013fe:	c3                   	ret    

00000000004013ff <scramble>:
  4013ff:	48 83 ec 38          	sub    $0x38,%rsp
  401403:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40140a:	00 00 
  40140c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  401411:	31 c0                	xor    %eax,%eax
  401413:	eb 10                	jmp    401425 <scramble+0x26>
  401415:	69 d0 56 a7 00 00    	imul   $0xa756,%eax,%edx
  40141b:	01 fa                	add    %edi,%edx
  40141d:	89 c1                	mov    %eax,%ecx
  40141f:	89 14 8c             	mov    %edx,(%rsp,%rcx,4)
  401422:	83 c0 01             	add    $0x1,%eax
  401425:	83 f8 09             	cmp    $0x9,%eax
  401428:	76 eb                	jbe    401415 <scramble+0x16>
  40142a:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  40142e:	69 c0 e9 79 00 00    	imul   $0x79e9,%eax,%eax
  401434:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  401438:	8b 04 24             	mov    (%rsp),%eax
  40143b:	69 c0 b6 f5 00 00    	imul   $0xf5b6,%eax,%eax
  401441:	89 04 24             	mov    %eax,(%rsp)
  401444:	8b 04 24             	mov    (%rsp),%eax
  401447:	69 c0 c1 c0 00 00    	imul   $0xc0c1,%eax,%eax
  40144d:	89 04 24             	mov    %eax,(%rsp)
  401450:	8b 44 24 10          	mov    0x10(%rsp),%eax
  401454:	69 c0 3e 0f 00 00    	imul   $0xf3e,%eax,%eax
  40145a:	89 44 24 10          	mov    %eax,0x10(%rsp)
  40145e:	8b 44 24 24          	mov    0x24(%rsp),%eax
  401462:	69 c0 26 9c 00 00    	imul   $0x9c26,%eax,%eax
  401468:	89 44 24 24          	mov    %eax,0x24(%rsp)
  40146c:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401470:	69 c0 c5 9f 00 00    	imul   $0x9fc5,%eax,%eax
  401476:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40147a:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40147e:	69 c0 3d 4c 00 00    	imul   $0x4c3d,%eax,%eax
  401484:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401488:	8b 44 24 10          	mov    0x10(%rsp),%eax
  40148c:	69 c0 c2 94 00 00    	imul   $0x94c2,%eax,%eax
  401492:	89 44 24 10          	mov    %eax,0x10(%rsp)
  401496:	8b 44 24 04          	mov    0x4(%rsp),%eax
  40149a:	69 c0 58 2d 00 00    	imul   $0x2d58,%eax,%eax
  4014a0:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4014a4:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4014a8:	69 c0 c2 f4 00 00    	imul   $0xf4c2,%eax,%eax
  4014ae:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4014b2:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4014b6:	69 c0 10 9a 00 00    	imul   $0x9a10,%eax,%eax
  4014bc:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4014c0:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4014c4:	69 c0 35 bb 00 00    	imul   $0xbb35,%eax,%eax
  4014ca:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4014ce:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4014d2:	69 c0 ca 6a 00 00    	imul   $0x6aca,%eax,%eax
  4014d8:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4014dc:	8b 44 24 14          	mov    0x14(%rsp),%eax
  4014e0:	69 c0 72 61 00 00    	imul   $0x6172,%eax,%eax
  4014e6:	89 44 24 14          	mov    %eax,0x14(%rsp)
  4014ea:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  4014ee:	69 c0 4e 3e 00 00    	imul   $0x3e4e,%eax,%eax
  4014f4:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  4014f8:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4014fc:	69 c0 e1 db 00 00    	imul   $0xdbe1,%eax,%eax
  401502:	89 44 24 04          	mov    %eax,0x4(%rsp)
  401506:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  40150a:	69 c0 80 18 00 00    	imul   $0x1880,%eax,%eax
  401510:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  401514:	8b 44 24 10          	mov    0x10(%rsp),%eax
  401518:	69 c0 c6 05 00 00    	imul   $0x5c6,%eax,%eax
  40151e:	89 44 24 10          	mov    %eax,0x10(%rsp)
  401522:	8b 44 24 08          	mov    0x8(%rsp),%eax
  401526:	69 c0 d2 f0 00 00    	imul   $0xf0d2,%eax,%eax
  40152c:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401530:	8b 44 24 20          	mov    0x20(%rsp),%eax
  401534:	69 c0 46 31 00 00    	imul   $0x3146,%eax,%eax
  40153a:	89 44 24 20          	mov    %eax,0x20(%rsp)
  40153e:	8b 44 24 18          	mov    0x18(%rsp),%eax
  401542:	69 c0 ba 27 00 00    	imul   $0x27ba,%eax,%eax
  401548:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40154c:	8b 44 24 24          	mov    0x24(%rsp),%eax
  401550:	69 c0 9f fb 00 00    	imul   $0xfb9f,%eax,%eax
  401556:	89 44 24 24          	mov    %eax,0x24(%rsp)
  40155a:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40155e:	69 c0 82 f5 00 00    	imul   $0xf582,%eax,%eax
  401564:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401568:	8b 44 24 08          	mov    0x8(%rsp),%eax
  40156c:	69 c0 18 76 00 00    	imul   $0x7618,%eax,%eax
  401572:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401576:	8b 04 24             	mov    (%rsp),%eax
  401579:	69 c0 61 97 00 00    	imul   $0x9761,%eax,%eax
  40157f:	89 04 24             	mov    %eax,(%rsp)
  401582:	8b 44 24 24          	mov    0x24(%rsp),%eax
  401586:	69 c0 96 c6 00 00    	imul   $0xc696,%eax,%eax
  40158c:	89 44 24 24          	mov    %eax,0x24(%rsp)
  401590:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401594:	69 c0 18 b2 00 00    	imul   $0xb218,%eax,%eax
  40159a:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40159e:	8b 04 24             	mov    (%rsp),%eax
  4015a1:	69 c0 ee 1c 00 00    	imul   $0x1cee,%eax,%eax
  4015a7:	89 04 24             	mov    %eax,(%rsp)
  4015aa:	8b 04 24             	mov    (%rsp),%eax
  4015ad:	69 c0 a5 28 00 00    	imul   $0x28a5,%eax,%eax
  4015b3:	89 04 24             	mov    %eax,(%rsp)
  4015b6:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4015ba:	69 c0 6d 51 00 00    	imul   $0x516d,%eax,%eax
  4015c0:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4015c4:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4015c8:	69 c0 f0 fa 00 00    	imul   $0xfaf0,%eax,%eax
  4015ce:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4015d2:	8b 44 24 08          	mov    0x8(%rsp),%eax
  4015d6:	69 c0 2a 1f 00 00    	imul   $0x1f2a,%eax,%eax
  4015dc:	89 44 24 08          	mov    %eax,0x8(%rsp)
  4015e0:	8b 44 24 10          	mov    0x10(%rsp),%eax
  4015e4:	69 c0 e6 fe 00 00    	imul   $0xfee6,%eax,%eax
  4015ea:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4015ee:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4015f2:	69 c0 b2 ec 00 00    	imul   $0xecb2,%eax,%eax
  4015f8:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4015fc:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  401600:	69 c0 f5 39 00 00    	imul   $0x39f5,%eax,%eax
  401606:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  40160a:	8b 44 24 04          	mov    0x4(%rsp),%eax
  40160e:	69 c0 b0 7c 00 00    	imul   $0x7cb0,%eax,%eax
  401614:	89 44 24 04          	mov    %eax,0x4(%rsp)
  401618:	8b 44 24 20          	mov    0x20(%rsp),%eax
  40161c:	69 c0 05 ef 00 00    	imul   $0xef05,%eax,%eax
  401622:	89 44 24 20          	mov    %eax,0x20(%rsp)
  401626:	8b 44 24 24          	mov    0x24(%rsp),%eax
  40162a:	69 c0 a6 d5 00 00    	imul   $0xd5a6,%eax,%eax
  401630:	89 44 24 24          	mov    %eax,0x24(%rsp)
  401634:	8b 44 24 18          	mov    0x18(%rsp),%eax
  401638:	69 c0 b2 08 00 00    	imul   $0x8b2,%eax,%eax
  40163e:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401642:	8b 44 24 14          	mov    0x14(%rsp),%eax
  401646:	69 c0 61 07 00 00    	imul   $0x761,%eax,%eax
  40164c:	89 44 24 14          	mov    %eax,0x14(%rsp)
  401650:	8b 44 24 14          	mov    0x14(%rsp),%eax
  401654:	69 c0 98 b6 00 00    	imul   $0xb698,%eax,%eax
  40165a:	89 44 24 14          	mov    %eax,0x14(%rsp)
  40165e:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  401662:	69 c0 85 d1 00 00    	imul   $0xd185,%eax,%eax
  401668:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  40166c:	8b 44 24 24          	mov    0x24(%rsp),%eax
  401670:	69 c0 c3 4f 00 00    	imul   $0x4fc3,%eax,%eax
  401676:	89 44 24 24          	mov    %eax,0x24(%rsp)
  40167a:	8b 44 24 20          	mov    0x20(%rsp),%eax
  40167e:	69 c0 93 85 00 00    	imul   $0x8593,%eax,%eax
  401684:	89 44 24 20          	mov    %eax,0x20(%rsp)
  401688:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  40168c:	69 c0 d4 6c 00 00    	imul   $0x6cd4,%eax,%eax
  401692:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  401696:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40169a:	69 c0 5b df 00 00    	imul   $0xdf5b,%eax,%eax
  4016a0:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4016a4:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4016a8:	69 c0 50 04 00 00    	imul   $0x450,%eax,%eax
  4016ae:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4016b2:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4016b6:	69 c0 34 7b 00 00    	imul   $0x7b34,%eax,%eax
  4016bc:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4016c0:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4016c4:	69 c0 76 23 00 00    	imul   $0x2376,%eax,%eax
  4016ca:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4016ce:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4016d2:	69 c0 ef f2 00 00    	imul   $0xf2ef,%eax,%eax
  4016d8:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4016dc:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4016e0:	69 c0 e0 85 00 00    	imul   $0x85e0,%eax,%eax
  4016e6:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4016ea:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4016ee:	69 c0 1a bf 00 00    	imul   $0xbf1a,%eax,%eax
  4016f4:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4016f8:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4016fc:	69 c0 ef e7 00 00    	imul   $0xe7ef,%eax,%eax
  401702:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401706:	8b 44 24 14          	mov    0x14(%rsp),%eax
  40170a:	69 c0 6c 59 00 00    	imul   $0x596c,%eax,%eax
  401710:	89 44 24 14          	mov    %eax,0x14(%rsp)
  401714:	8b 44 24 14          	mov    0x14(%rsp),%eax
  401718:	69 c0 7f e0 00 00    	imul   $0xe07f,%eax,%eax
  40171e:	89 44 24 14          	mov    %eax,0x14(%rsp)
  401722:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  401726:	69 c0 b5 90 00 00    	imul   $0x90b5,%eax,%eax
  40172c:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  401730:	8b 44 24 20          	mov    0x20(%rsp),%eax
  401734:	69 c0 c5 2e 00 00    	imul   $0x2ec5,%eax,%eax
  40173a:	89 44 24 20          	mov    %eax,0x20(%rsp)
  40173e:	8b 44 24 20          	mov    0x20(%rsp),%eax
  401742:	69 c0 c8 ff 00 00    	imul   $0xffc8,%eax,%eax
  401748:	89 44 24 20          	mov    %eax,0x20(%rsp)
  40174c:	8b 44 24 08          	mov    0x8(%rsp),%eax
  401750:	69 c0 ad 50 00 00    	imul   $0x50ad,%eax,%eax
  401756:	89 44 24 08          	mov    %eax,0x8(%rsp)
  40175a:	8b 44 24 24          	mov    0x24(%rsp),%eax
  40175e:	69 c0 f0 f6 00 00    	imul   $0xf6f0,%eax,%eax
  401764:	89 44 24 24          	mov    %eax,0x24(%rsp)
  401768:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  40176c:	69 c0 5d 5c 00 00    	imul   $0x5c5d,%eax,%eax
  401772:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  401776:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  40177a:	69 c0 45 25 00 00    	imul   $0x2545,%eax,%eax
  401780:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  401784:	8b 44 24 18          	mov    0x18(%rsp),%eax
  401788:	69 c0 d0 d4 00 00    	imul   $0xd4d0,%eax,%eax
  40178e:	89 44 24 18          	mov    %eax,0x18(%rsp)
  401792:	8b 44 24 10          	mov    0x10(%rsp),%eax
  401796:	69 c0 e3 63 00 00    	imul   $0x63e3,%eax,%eax
  40179c:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4017a0:	8b 04 24             	mov    (%rsp),%eax
  4017a3:	69 c0 61 08 00 00    	imul   $0x861,%eax,%eax
  4017a9:	89 04 24             	mov    %eax,(%rsp)
  4017ac:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4017b0:	69 c0 b1 55 00 00    	imul   $0x55b1,%eax,%eax
  4017b6:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4017ba:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  4017be:	69 c0 72 c6 00 00    	imul   $0xc672,%eax,%eax
  4017c4:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  4017c8:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  4017cc:	69 c0 26 03 00 00    	imul   $0x326,%eax,%eax
  4017d2:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  4017d6:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4017da:	69 c0 8c a9 00 00    	imul   $0xa98c,%eax,%eax
  4017e0:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4017e4:	8b 44 24 14          	mov    0x14(%rsp),%eax
  4017e8:	69 c0 03 9f 00 00    	imul   $0x9f03,%eax,%eax
  4017ee:	89 44 24 14          	mov    %eax,0x14(%rsp)
  4017f2:	8b 44 24 08          	mov    0x8(%rsp),%eax
  4017f6:	69 c0 5f bd 00 00    	imul   $0xbd5f,%eax,%eax
  4017fc:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401800:	8b 44 24 10          	mov    0x10(%rsp),%eax
  401804:	69 c0 22 21 00 00    	imul   $0x2122,%eax,%eax
  40180a:	89 44 24 10          	mov    %eax,0x10(%rsp)
  40180e:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401812:	69 c0 f2 91 00 00    	imul   $0x91f2,%eax,%eax
  401818:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40181c:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401820:	69 c0 ac 6a 00 00    	imul   $0x6aac,%eax,%eax
  401826:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40182a:	8b 44 24 08          	mov    0x8(%rsp),%eax
  40182e:	69 c0 47 a6 00 00    	imul   $0xa647,%eax,%eax
  401834:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401838:	8b 44 24 10          	mov    0x10(%rsp),%eax
  40183c:	69 c0 21 a0 00 00    	imul   $0xa021,%eax,%eax
  401842:	89 44 24 10          	mov    %eax,0x10(%rsp)
  401846:	8b 04 24             	mov    (%rsp),%eax
  401849:	69 c0 e7 37 00 00    	imul   $0x37e7,%eax,%eax
  40184f:	89 04 24             	mov    %eax,(%rsp)
  401852:	8b 04 24             	mov    (%rsp),%eax
  401855:	69 c0 51 63 00 00    	imul   $0x6351,%eax,%eax
  40185b:	89 04 24             	mov    %eax,(%rsp)
  40185e:	8b 44 24 18          	mov    0x18(%rsp),%eax
  401862:	69 c0 e0 65 00 00    	imul   $0x65e0,%eax,%eax
  401868:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40186c:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401870:	69 c0 fe 04 00 00    	imul   $0x4fe,%eax,%eax
  401876:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40187a:	8b 44 24 10          	mov    0x10(%rsp),%eax
  40187e:	69 c0 07 1e 00 00    	imul   $0x1e07,%eax,%eax
  401884:	89 44 24 10          	mov    %eax,0x10(%rsp)
  401888:	8b 44 24 04          	mov    0x4(%rsp),%eax
  40188c:	69 c0 82 fc 00 00    	imul   $0xfc82,%eax,%eax
  401892:	89 44 24 04          	mov    %eax,0x4(%rsp)
  401896:	8b 44 24 20          	mov    0x20(%rsp),%eax
  40189a:	69 c0 7b 6f 00 00    	imul   $0x6f7b,%eax,%eax
  4018a0:	89 44 24 20          	mov    %eax,0x20(%rsp)
  4018a4:	8b 44 24 24          	mov    0x24(%rsp),%eax
  4018a8:	69 c0 da 92 00 00    	imul   $0x92da,%eax,%eax
  4018ae:	89 44 24 24          	mov    %eax,0x24(%rsp)
  4018b2:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  4018b6:	69 c0 3a 5c 00 00    	imul   $0x5c3a,%eax,%eax
  4018bc:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  4018c0:	8b 04 24             	mov    (%rsp),%eax
  4018c3:	69 c0 12 a3 00 00    	imul   $0xa312,%eax,%eax
  4018c9:	89 04 24             	mov    %eax,(%rsp)
  4018cc:	8b 44 24 14          	mov    0x14(%rsp),%eax
  4018d0:	69 c0 95 61 00 00    	imul   $0x6195,%eax,%eax
  4018d6:	89 44 24 14          	mov    %eax,0x14(%rsp)
  4018da:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  4018de:	69 c0 be 05 00 00    	imul   $0x5be,%eax,%eax
  4018e4:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  4018e8:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4018ec:	69 c0 20 e5 00 00    	imul   $0xe520,%eax,%eax
  4018f2:	89 44 24 04          	mov    %eax,0x4(%rsp)
  4018f6:	8b 44 24 24          	mov    0x24(%rsp),%eax
  4018fa:	69 c0 27 d6 00 00    	imul   $0xd627,%eax,%eax
  401900:	89 44 24 24          	mov    %eax,0x24(%rsp)
  401904:	8b 44 24 08          	mov    0x8(%rsp),%eax
  401908:	69 c0 07 d2 00 00    	imul   $0xd207,%eax,%eax
  40190e:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401912:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  401916:	69 c0 0b 8e 00 00    	imul   $0x8e0b,%eax,%eax
  40191c:	89 44 24 0c          	mov    %eax,0xc(%rsp)
  401920:	8b 44 24 18          	mov    0x18(%rsp),%eax
  401924:	69 c0 ae 3b 00 00    	imul   $0x3bae,%eax,%eax
  40192a:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40192e:	8b 44 24 04          	mov    0x4(%rsp),%eax
  401932:	69 c0 a7 16 00 00    	imul   $0x16a7,%eax,%eax
  401938:	89 44 24 04          	mov    %eax,0x4(%rsp)
  40193c:	ba 00 00 00 00       	mov    $0x0,%edx
  401941:	b8 00 00 00 00       	mov    $0x0,%eax
  401946:	eb 0a                	jmp    401952 <scramble+0x553>
  401948:	89 d1                	mov    %edx,%ecx
  40194a:	8b 0c 8c             	mov    (%rsp,%rcx,4),%ecx
  40194d:	01 c8                	add    %ecx,%eax
  40194f:	83 c2 01             	add    $0x1,%edx
  401952:	83 fa 09             	cmp    $0x9,%edx
  401955:	76 f1                	jbe    401948 <scramble+0x549>
  401957:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  40195c:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  401963:	00 00 
  401965:	74 05                	je     40196c <scramble+0x56d>
  401967:	e8 04 f4 ff ff       	call   400d70 <__stack_chk_fail@plt>
  40196c:	48 83 c4 38          	add    $0x38,%rsp
  401970:	c3                   	ret    

0000000000401971 <getbuf>:
  401971:	48 83 ec 28          	sub    $0x28,%rsp
  401975:	48 89 e7             	mov    %rsp,%rdi
  401978:	e8 7e 02 00 00       	call   401bfb <Gets>
  40197d:	b8 01 00 00 00       	mov    $0x1,%eax
  401982:	48 83 c4 28          	add    $0x28,%rsp
  401986:	c3                   	ret    

0000000000401987 <touch1>:
  401987:	48 83 ec 08          	sub    $0x8,%rsp
  40198b:	c7 05 8b 2b 20 00 01 	movl   $0x1,0x202b8b(%rip)        # 604520 <vlevel>
  401992:	00 00 00 
  401995:	bf c2 31 40 00       	mov    $0x4031c2,%edi
  40199a:	e8 b1 f3 ff ff       	call   400d50 <puts@plt>
  40199f:	bf 01 00 00 00       	mov    $0x1,%edi
  4019a4:	e8 92 03 00 00       	call   401d3b <validate>
  4019a9:	bf 00 00 00 00       	mov    $0x0,%edi
  4019ae:	e8 1d f5 ff ff       	call   400ed0 <exit@plt>

00000000004019b3 <touch2>:
  4019b3:	48 83 ec 08          	sub    $0x8,%rsp
  4019b7:	89 fa                	mov    %edi,%edx
  4019b9:	c7 05 5d 2b 20 00 02 	movl   $0x2,0x202b5d(%rip)        # 604520 <vlevel>
  4019c0:	00 00 00 
  4019c3:	39 3d 5f 2b 20 00    	cmp    %edi,0x202b5f(%rip)        # 604528 <cookie>
  4019c9:	75 20                	jne    4019eb <touch2+0x38>
  4019cb:	be e8 31 40 00       	mov    $0x4031e8,%esi
  4019d0:	bf 01 00 00 00       	mov    $0x1,%edi
  4019d5:	b8 00 00 00 00       	mov    $0x0,%eax
  4019da:	e8 11 f3 ff ff       	call   400cf0 <__printf_chk@plt>
  4019df:	bf 02 00 00 00       	mov    $0x2,%edi
  4019e4:	e8 52 03 00 00       	call   401d3b <validate>
  4019e9:	eb 1e                	jmp    401a09 <ouch2+0x56>
  4019eb:	be 10 32 40 00       	mov    $0x403210,%esi
  4019f0:	bf 01 00 00 00       	mov    $0x1,%edi
  4019f5:	b8 00 00 00 00       	mov    $0x0,%eax
  4019fa:	e8 f1 f2 ff ff       	call   400cf0 <__printf_chk@plt>
  4019ff:	bf 02 00 00 00       	mov    $0x2,%edi
  401a04:	e8 f4 03 00 00       	call   401dfd <fail>
  401a09:	bf 00 00 00 00       	mov    $0x0,%edi
  401a0e:	e8 bd f4 ff ff       	call   400ed0 <exit@plt>

0000000000401a13 <hexmatch>:
  401a13:	41 54                	push   %r12
  401a15:	55                   	push   %rbp
  401a16:	53                   	push   %rbx
  401a17:	48 83 c4 80          	add    $0xffffffffffffff80,%rsp
  401a1b:	89 fd                	mov    %edi,%ebp
  401a1d:	48 89 f3             	mov    %rsi,%rbx
  401a20:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401a27:	00 00 
  401a29:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  401a2e:	31 c0                	xor    %eax,%eax
  401a30:	e8 1b f4 ff ff       	call   400e50 <random@plt>
  401a35:	48 89 c1             	mov    %rax,%rcx
  401a38:	48 ba 0b d7 a3 70 3d 	movabs $0xa3d70a3d70a3d70b,%rdx
  401a3f:	0a d7 a3 
  401a42:	48 f7 ea             	imul   %rdx
  401a45:	48 01 ca             	add    %rcx,%rdx
  401a48:	48 c1 fa 06          	sar    $0x6,%rdx
  401a4c:	48 89 c8             	mov    %rcx,%rax
  401a4f:	48 c1 f8 3f          	sar    $0x3f,%rax
  401a53:	48 29 c2             	sub    %rax,%rdx
  401a56:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  401a5a:	48 8d 14 80          	lea    (%rax,%rax,4),%rdx
  401a5e:	48 8d 04 95 00 00 00 	lea    0x0(,%rdx,4),%rax
  401a65:	00 
  401a66:	48 29 c1             	sub    %rax,%rcx
  401a69:	4c 8d 24 0c          	lea    (%rsp,%rcx,1),%r12
  401a6d:	41 89 e8             	mov    %ebp,%r8d
  401a70:	b9 df 31 40 00       	mov    $0x4031df,%ecx
  401a75:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  401a7c:	be 01 00 00 00       	mov    $0x1,%esi
  401a81:	4c 89 e7             	mov    %r12,%rdi
  401a84:	b8 00 00 00 00       	mov    $0x0,%eax
  401a89:	e8 a2 f3 ff ff       	call   400e30 <__sprintf_chk@plt>
  401a8e:	ba 09 00 00 00       	mov    $0x9,%edx
  401a93:	4c 89 e6             	mov    %r12,%rsi
  401a96:	48 89 df             	mov    %rbx,%rdi
  401a99:	e8 92 f2 ff ff       	call   400d30 <strncmp@plt>
  401a9e:	85 c0                	test   %eax,%eax
  401aa0:	0f 94 c0             	sete   %al
  401aa3:	48 8b 5c 24 78       	mov    0x78(%rsp),%rbx
  401aa8:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
  401aaf:	00 00 
  401ab1:	74 05                	je     401ab8 <hexmatch+0xa5>
  401ab3:	e8 b8 f2 ff ff       	call   400d70 <__stack_chk_fail@plt>
  401ab8:	0f b6 c0             	movzbl %al,%eax
  401abb:	48 83 ec 80          	sub    $0xffffffffffffff80,%rsp
  401abf:	5b                   	pop    %rbx
  401ac0:	5d                   	pop    %rbp
  401ac1:	41 5c                	pop    %r12
  401ac3:	c3                   	ret    

0000000000401ac4 <touch3>:
  401ac4:	53                   	push   %rbx
  401ac5:	48 89 fb             	mov    %rdi,%rbx
  401ac8:	c7 05 4e 2a 20 00 03 	movl   $0x3,0x202a4e(%rip)        # 604520 <vlevel>
  401acf:	00 00 00 
  401ad2:	48 89 fe             	mov    %rdi,%rsi
  401ad5:	8b 3d 4d 2a 20 00    	mov    0x202a4d(%rip),%edi        # 604528 <cookie>
  401adb:	e8 33 ff ff ff       	call   401a13 <hexmatch>
  401ae0:	85 c0                	test   %eax,%eax
  401ae2:	74 23                	je     401b07 <touch3+0x43>
  401ae4:	48 89 da             	mov    %rbx,%rdx
  401ae7:	be 38 32 40 00       	mov    $0x403238,%esi
  401aec:	bf 01 00 00 00       	mov    $0x1,%edi
  401af1:	b8 00 00 00 00       	mov    $0x0,%eax
  401af6:	e8 f5 f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401afb:	bf 03 00 00 00       	mov    $0x3,%edi
  401b00:	e8 36 02 00 00       	call   401d3b <validate>
  401b05:	eb 21                	jmp    401b28 <touch3+0x64>
  401b07:	48 89 da             	mov    %rbx,%rdx
  401b0a:	be 60 32 40 00       	mov    $0x403260,%esi
  401b0f:	bf 01 00 00 00       	mov    $0x1,%edi
  401b14:	b8 00 00 00 00       	mov    $0x0,%eax
  401b19:	e8 d2 f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401b1e:	bf 03 00 00 00       	mov    $0x3,%edi
  401b23:	e8 d5 02 00 00       	call   401dfd <fail>
  401b28:	bf 00 00 00 00       	mov    $0x0,%edi
  401b2d:	e8 9e f3 ff ff       	call   400ed0 <exit@plt>

0000000000401b32 <test>:
  401b32:	48 83 ec 08          	sub    $0x8,%rsp
  401b36:	b8 00 00 00 00       	mov    $0x0,%eax
  401b3b:	e8 31 fe ff ff       	call   401971 <getbuf>
  401b40:	89 c2                	mov    %eax,%edx
  401b42:	be 88 32 40 00       	mov    $0x403288,%esi
  401b47:	bf 01 00 00 00       	mov    $0x1,%edi
  401b4c:	b8 00 00 00 00       	mov    $0x0,%eax
  401b51:	e8 9a f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401b56:	48 83 c4 08          	add    $0x8,%rsp
  401b5a:	c3                   	ret    

0000000000401b5b <save_char>:
  401b5b:	8b 05 e3 35 20 00    	mov    0x2035e3(%rip),%eax        # 605144 <gets_cnt>
  401b61:	3d ff 03 00 00       	cmp    $0x3ff,%eax
  401b66:	7f 49                	jg     401bb1 <save_char+0x56>
  401b68:	8d 14 40             	lea    (%rax,%rax,2),%edx
  401b6b:	89 f9                	mov    %edi,%ecx
  401b6d:	c0 e9 04             	shr    $0x4,%cl
  401b70:	83 e1 0f             	and    $0xf,%ecx
  401b73:	0f b6 b1 00 35 40 00 	movzbl 0x403500(%rcx),%esi
  401b7a:	48 63 ca             	movslq %edx,%rcx
  401b7d:	40 88 b1 40 45 60 00 	mov    %sil,0x604540(%rcx)
  401b84:	8d 4a 01             	lea    0x1(%rdx),%ecx
  401b87:	83 e7 0f             	and    $0xf,%edi
  401b8a:	0f b6 b7 00 35 40 00 	movzbl 0x403500(%rdi),%esi
  401b91:	48 63 c9             	movslq %ecx,%rcx
  401b94:	40 88 b1 40 45 60 00 	mov    %sil,0x604540(%rcx)
  401b9b:	83 c2 02             	add    $0x2,%edx
  401b9e:	48 63 d2             	movslq %edx,%rdx
  401ba1:	c6 82 40 45 60 00 20 	movb   $0x20,0x604540(%rdx)
  401ba8:	83 c0 01             	add    $0x1,%eax
  401bab:	89 05 93 35 20 00    	mov    %eax,0x203593(%rip)        # 605144 <gets_cnt>
  401bb1:	f3 c3                	repz ret 

0000000000401bb3 <save_term>:
  401bb3:	8b 05 8b 35 20 00    	mov    0x20358b(%rip),%eax        # 605144 <gets_cnt>
  401bb9:	8d 04 40             	lea    (%rax,%rax,2),%eax
  401bbc:	48 98                	cltq   
  401bbe:	c6 80 40 45 60 00 00 	movb   $0x0,0x604540(%rax)
  401bc5:	c3                   	ret    

0000000000401bc6 <check_fail>:
  401bc6:	48 83 ec 08          	sub    $0x8,%rsp
  401bca:	0f be 15 7f 35 20 00 	movsbl 0x20357f(%rip),%edx        # 605150 <target_prefix>
  401bd1:	41 b8 40 45 60 00    	mov    $0x604540,%r8d
  401bd7:	8b 0d 3f 29 20 00    	mov    0x20293f(%rip),%ecx        # 60451c <check_level>
  401bdd:	be ab 32 40 00       	mov    $0x4032ab,%esi
  401be2:	bf 01 00 00 00       	mov    $0x1,%edi
  401be7:	b8 00 00 00 00       	mov    $0x0,%eax
  401bec:	e8 ff f0 ff ff       	call   400cf0 <__printf_chk@plt>
  401bf1:	bf 01 00 00 00       	mov    $0x1,%edi
  401bf6:	e8 d5 f2 ff ff       	call   400ed0 <exit@plt>

0000000000401bfb <Gets>:
  401bfb:	41 54                	push   %r12
  401bfd:	55                   	push   %rbp
  401bfe:	53                   	push   %rbx
  401bff:	49 89 fc             	mov    %rdi,%r12
  401c02:	c7 05 38 35 20 00 00 	movl   $0x0,0x203538(%rip)        # 605144 <gets_cnt>
  401c09:	00 00 00 
  401c0c:	48 89 fb             	mov    %rdi,%rbx
  401c0f:	eb 11                	jmp    401c22 <Gets+0x27>
  401c11:	48 8d 6b 01          	lea    0x1(%rbx),%rbp
  401c15:	88 03                	mov    %al,(%rbx)
  401c17:	0f b6 f8             	movzbl %al,%edi
  401c1a:	e8 3c ff ff ff       	call   401b5b <save_char>
  401c1f:	48 89 eb             	mov    %rbp,%rbx
  401c22:	48 8b 3d e7 28 20 00 	mov    0x2028e7(%rip),%rdi        # 604510 <infile>
  401c29:	e8 32 f2 ff ff       	call   400e60 <_IO_getc@plt>
  401c2e:	83 f8 ff             	cmp    $0xffffffff,%eax
  401c31:	74 05                	je     401c38 <Gets+0x3d>
  401c33:	83 f8 0a             	cmp    $0xa,%eax
  401c36:	75 d9                	jne    401c11 <Gets+0x16>
  401c38:	c6 03 00             	movb   $0x0,(%rbx)
  401c3b:	b8 00 00 00 00       	mov    $0x0,%eax
  401c40:	e8 6e ff ff ff       	call   401bb3 <save_term>
  401c45:	4c 89 e0             	mov    %r12,%rax
  401c48:	5b                   	pop    %rbx
  401c49:	5d                   	pop    %rbp
  401c4a:	41 5c                	pop    %r12
  401c4c:	c3                   	ret    

0000000000401c4d <notify_server>:
  401c4d:	53                   	push   %rbx
  401c4e:	48 81 ec 10 20 00 00 	sub    $0x2010,%rsp
  401c55:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401c5c:	00 00 
  401c5e:	48 89 84 24 08 20 00 	mov    %rax,0x2008(%rsp)
  401c65:	00 
  401c66:	31 c0                	xor    %eax,%eax
  401c68:	83 3d bd 28 20 00 00 	cmpl   $0x0,0x2028bd(%rip)        # 60452c <is_checker>
  401c6f:	0f 85 a5 00 00 00    	jne    401d1a <notify_server+0xcd>
  401c75:	89 fb                	mov    %edi,%ebx
  401c77:	8b 05 c7 34 20 00    	mov    0x2034c7(%rip),%eax        # 605144 <gets_cnt>
  401c7d:	83 c0 64             	add    $0x64,%eax
  401c80:	3d 00 20 00 00       	cmp    $0x2000,%eax
  401c85:	7e 1e                	jle    401ca5 <notify_server+0x58>
  401c87:	be 90 33 40 00       	mov    $0x403390,%esi
  401c8c:	bf 01 00 00 00       	mov    $0x1,%edi
  401c91:	b8 00 00 00 00       	mov    $0x0,%eax
  401c96:	e8 55 f0 ff ff       	call   400cf0 <__printf_chk@plt>
  401c9b:	bf 01 00 00 00       	mov    $0x1,%edi
  401ca0:	e8 2b f2 ff ff       	call   400ed0 <exit@plt>
  401ca5:	0f be 05 a4 34 20 00 	movsbl 0x2034a4(%rip),%eax        # 605150 <target_prefix>
  401cac:	83 3d 65 28 20 00 00 	cmpl   $0x0,0x202865(%rip)        # 604518 <notify>
  401cb3:	74 08                	je     401cbd <notify_server+0x70>
  401cb5:	8b 15 69 28 20 00    	mov    0x202869(%rip),%edx        # 604524 <authkey>
  401cbb:	eb 05                	jmp    401cc2 <notify_server+0x75>
  401cbd:	ba ff ff ff ff       	mov    $0xffffffff,%edx
  401cc2:	85 db                	test   %ebx,%ebx
  401cc4:	74 08                	je     401cce <notify_server+0x81>
  401cc6:	41 b9 c1 32 40 00    	mov    $0x4032c1,%r9d
  401ccc:	eb 06                	jmp    401cd4 <notify_server+0x87>
  401cce:	41 b9 c6 32 40 00    	mov    $0x4032c6,%r9d
  401cd4:	68 40 45 60 00       	push   $0x604540
  401cd9:	56                   	push   %rsi
  401cda:	50                   	push   %rax
  401cdb:	52                   	push   %rdx
  401cdc:	44 8b 05 85 24 20 00 	mov    0x202485(%rip),%r8d        # 604168 <target_id>
  401ce3:	b9 cb 32 40 00       	mov    $0x4032cb,%ecx
  401ce8:	ba 00 20 00 00       	mov    $0x2000,%edx
  401ced:	be 01 00 00 00       	mov    $0x1,%esi
  401cf2:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  401cf7:	b8 00 00 00 00       	mov    $0x0,%eax
  401cfc:	e8 2f f1 ff ff       	call   400e30 <__sprintf_chk@plt>
  401d01:	48 83 c4 20          	add    $0x20,%rsp
  401d05:	85 db                	test   %ebx,%ebx
  401d07:	74 07                	je     401d10 <notify_server+0xc3>
  401d09:	bf c1 32 40 00       	mov    $0x4032c1,%edi
  401d0e:	eb 05                	jmp    401d15 <notify_server+0xc8>
  401d10:	bf c6 32 40 00       	mov    $0x4032c6,%edi
  401d15:	e8 36 f0 ff ff       	call   400d50 <puts@plt>
  401d1a:	48 8b 84 24 08 20 00 	mov    0x2008(%rsp),%rax
  401d21:	00 
  401d22:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401d29:	00 00 
  401d2b:	74 05                	je     401d32 <notify_server+0xe5>
  401d2d:	e8 3e f0 ff ff       	call   400d70 <__stack_chk_fail@plt>
  401d32:	48 81 c4 10 20 00 00 	add    $0x2010,%rsp
  401d39:	5b                   	pop    %rbx
  401d3a:	c3                   	ret    

0000000000401d3b <validate>:
  401d3b:	53                   	push   %rbx
  401d3c:	89 fb                	mov    %edi,%ebx
  401d3e:	83 3d e7 27 20 00 00 	cmpl   $0x0,0x2027e7(%rip)        # 60452c <is_checker>
  401d45:	74 6b                	je     401db2 <validate+0x77>
  401d47:	39 3d d3 27 20 00    	cmp    %edi,0x2027d3(%rip)        # 604520 <vlevel>
  401d4d:	74 14                	je     401d63 <validate+0x28>
  401d4f:	bf e7 32 40 00       	mov    $0x4032e7,%edi
  401d54:	e8 f7 ef ff ff       	call   400d50 <puts@plt>
  401d59:	b8 00 00 00 00       	mov    $0x0,%eax
  401d5e:	e8 63 fe ff ff       	call   401bc6 <check_fail>
  401d63:	8b 15 b3 27 20 00    	mov    0x2027b3(%rip),%edx        # 60451c <check_level>
  401d69:	39 d7                	cmp    %edx,%edi
  401d6b:	74 20                	je     401d8d <validate+0x52>
  401d6d:	89 f9                	mov    %edi,%ecx
  401d6f:	be c0 33 40 00       	mov    $0x4033c0,%esi
  401d74:	bf 01 00 00 00       	mov    $0x1,%edi
  401d79:	b8 00 00 00 00       	mov    $0x0,%eax
  401d7e:	e8 6d ef ff ff       	call   400cf0 <__printf_chk@plt>
  401d83:	b8 00 00 00 00       	mov    $0x0,%eax
  401d88:	e8 39 fe ff ff       	call   401bc6 <check_fail>
  401d8d:	0f be 15 bc 33 20 00 	movsbl 0x2033bc(%rip),%edx        # 605150 <target_prefix>
  401d94:	41 b8 40 45 60 00    	mov    $0x604540,%r8d
  401d9a:	89 f9                	mov    %edi,%ecx
  401d9c:	be 05 33 40 00       	mov    $0x403305,%esi
  401da1:	bf 01 00 00 00       	mov    $0x1,%edi
  401da6:	b8 00 00 00 00       	mov    $0x0,%eax
  401dab:	e8 40 ef ff ff       	call   400cf0 <__printf_chk@plt>
  401db0:	eb 49                	jmp    401dfb <validate+0xc0>
  401db2:	3b 3d 68 27 20 00    	cmp    0x202768(%rip),%edi        # 604520 <vlevel>
  401db8:	74 18                	je     401dd2 <validate+0x97>
  401dba:	bf e7 32 40 00       	mov    $0x4032e7,%edi
  401dbf:	e8 8c ef ff ff       	call   400d50 <puts@plt>
  401dc4:	89 de                	mov    %ebx,%esi
  401dc6:	bf 00 00 00 00       	mov    $0x0,%edi
  401dcb:	e8 7d fe ff ff       	call   401c4d <notify_server>
  401dd0:	eb 29                	jmp    401dfb <validate+0xc0>
  401dd2:	0f be 0d 77 33 20 00 	movsbl 0x203377(%rip),%ecx        # 605150 <target_prefix>
  401dd9:	89 fa                	mov    %edi,%edx
  401ddb:	be e8 33 40 00       	mov    $0x4033e8,%esi
  401de0:	bf 01 00 00 00       	mov    $0x1,%edi
  401de5:	b8 00 00 00 00       	mov    $0x0,%eax
  401dea:	e8 01 ef ff ff       	call   400cf0 <__printf_chk@plt>
  401def:	89 de                	mov    %ebx,%esi
  401df1:	bf 01 00 00 00       	mov    $0x1,%edi
  401df6:	e8 52 fe ff ff       	call   401c4d <notify_server>
  401dfb:	5b                   	pop    %rbx
  401dfc:	c3                   	ret    

0000000000401dfd <fail>:
  401dfd:	48 83 ec 08          	sub    $0x8,%rsp
  401e01:	83 3d 24 27 20 00 00 	cmpl   $0x0,0x202724(%rip)        # 60452c <is_checker>
  401e08:	74 0a                	je     401e14 <fail+0x17>
  401e0a:	b8 00 00 00 00       	mov    $0x0,%eax
  401e0f:	e8 b2 fd ff ff       	call   401bc6 <check_fail>
  401e14:	89 fe                	mov    %edi,%esi
  401e16:	bf 00 00 00 00       	mov    $0x0,%edi
  401e1b:	e8 2d fe ff ff       	call   401c4d <notify_server>
  401e20:	48 83 c4 08          	add    $0x8,%rsp
  401e24:	c3                   	ret    

0000000000401e25 <bushandler>:
  401e25:	48 83 ec 08          	sub    $0x8,%rsp
  401e29:	83 3d fc 26 20 00 00 	cmpl   $0x0,0x2026fc(%rip)        # 60452c <is_checker>
  401e30:	74 14                	je     401e46 <bushandler+0x21>
  401e32:	bf 1a 33 40 00       	mov    $0x40331a,%edi
  401e37:	e8 14 ef ff ff       	call   400d50 <puts@plt>
  401e3c:	b8 00 00 00 00       	mov    $0x0,%eax
  401e41:	e8 80 fd ff ff       	call   401bc6 <check_fail>
  401e46:	bf 20 34 40 00       	mov    $0x403420,%edi
  401e4b:	e8 00 ef ff ff       	call   400d50 <puts@plt>
  401e50:	bf 24 33 40 00       	mov    $0x403324,%edi
  401e55:	e8 f6 ee ff ff       	call   400d50 <puts@plt>
  401e5a:	be 00 00 00 00       	mov    $0x0,%esi
  401e5f:	bf 00 00 00 00       	mov    $0x0,%edi
  401e64:	e8 e4 fd ff ff       	call   401c4d <notify_server>
  401e69:	bf 01 00 00 00       	mov    $0x1,%edi
  401e6e:	e8 5d f0 ff ff       	call   400ed0 <exit@plt>

0000000000401e73 <seghandler>:
  401e73:	48 83 ec 08          	sub    $0x8,%rsp
  401e77:	83 3d ae 26 20 00 00 	cmpl   $0x0,0x2026ae(%rip)        # 60452c <is_checker>
  401e7e:	74 14                	je     401e94 <seghandler+0x21>
  401e80:	bf 3a 33 40 00       	mov    $0x40333a,%edi
  401e85:	e8 c6 ee ff ff       	call   400d50 <puts@plt>
  401e8a:	b8 00 00 00 00       	mov    $0x0,%eax
  401e8f:	e8 32 fd ff ff       	call   401bc6 <check_fail>
  401e94:	bf 40 34 40 00       	mov    $0x403440,%edi
  401e99:	e8 b2 ee ff ff       	call   400d50 <puts@plt>
  401e9e:	bf 24 33 40 00       	mov    $0x403324,%edi
  401ea3:	e8 a8 ee ff ff       	call   400d50 <puts@plt>
  401ea8:	be 00 00 00 00       	mov    $0x0,%esi
  401ead:	bf 00 00 00 00       	mov    $0x0,%edi
  401eb2:	e8 96 fd ff ff       	call   401c4d <notify_server>
  401eb7:	bf 01 00 00 00       	mov    $0x1,%edi
  401ebc:	e8 0f f0 ff ff       	call   400ed0 <exit@plt>

0000000000401ec1 <illegalhandler>:
  401ec1:	48 83 ec 08          	sub    $0x8,%rsp
  401ec5:	83 3d 60 26 20 00 00 	cmpl   $0x0,0x202660(%rip)        # 60452c <is_checker>
  401ecc:	74 14                	je     401ee2 <illegalhandler+0x21>
  401ece:	bf 4d 33 40 00       	mov    $0x40334d,%edi
  401ed3:	e8 78 ee ff ff       	call   400d50 <puts@plt>
  401ed8:	b8 00 00 00 00       	mov    $0x0,%eax
  401edd:	e8 e4 fc ff ff       	call   401bc6 <check_fail>
  401ee2:	bf 68 34 40 00       	mov    $0x403468,%edi
  401ee7:	e8 64 ee ff ff       	call   400d50 <puts@plt>
  401eec:	bf 24 33 40 00       	mov    $0x403324,%edi
  401ef1:	e8 5a ee ff ff       	call   400d50 <puts@plt>
  401ef6:	be 00 00 00 00       	mov    $0x0,%esi
  401efb:	bf 00 00 00 00       	mov    $0x0,%edi
  401f00:	e8 48 fd ff ff       	call   401c4d <notify_server>
  401f05:	bf 01 00 00 00       	mov    $0x1,%edi
  401f0a:	e8 c1 ef ff ff       	call   400ed0 <exit@plt>

0000000000401f0f <sigalrmhandler>:
  401f0f:	48 83 ec 08          	sub    $0x8,%rsp
  401f13:	83 3d 12 26 20 00 00 	cmpl   $0x0,0x202612(%rip)        # 60452c <is_checker>
  401f1a:	74 14                	je     401f30 <sigalrmhandler+0x21>
  401f1c:	bf 61 33 40 00       	mov    $0x403361,%edi
  401f21:	e8 2a ee ff ff       	call   400d50 <puts@plt>
  401f26:	b8 00 00 00 00       	mov    $0x0,%eax
  401f2b:	e8 96 fc ff ff       	call   401bc6 <check_fail>
  401f30:	ba 05 00 00 00       	mov    $0x5,%edx
  401f35:	be 98 34 40 00       	mov    $0x403498,%esi
  401f3a:	bf 01 00 00 00       	mov    $0x1,%edi
  401f3f:	b8 00 00 00 00       	mov    $0x0,%eax
  401f44:	e8 a7 ed ff ff       	call   400cf0 <__printf_chk@plt>
  401f49:	be 00 00 00 00       	mov    $0x0,%esi
  401f4e:	bf 00 00 00 00       	mov    $0x0,%edi
  401f53:	e8 f5 fc ff ff       	call   401c4d <notify_server>
  401f58:	bf 01 00 00 00       	mov    $0x1,%edi
  401f5d:	e8 6e ef ff ff       	call   400ed0 <exit@plt>

0000000000401f62 <launch>:
  401f62:	55                   	push   %rbp
  401f63:	48 89 e5             	mov    %rsp,%rbp
  401f66:	48 83 ec 10          	sub    $0x10,%rsp
  401f6a:	48 89 fa             	mov    %rdi,%rdx
  401f6d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401f74:	00 00 
  401f76:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  401f7a:	31 c0                	xor    %eax,%eax
  401f7c:	48 8d 47 1e          	lea    0x1e(%rdi),%rax
  401f80:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
  401f84:	48 29 c4             	sub    %rax,%rsp
  401f87:	48 8d 7c 24 0f       	lea    0xf(%rsp),%rdi
  401f8c:	48 83 e7 f0          	and    $0xfffffffffffffff0,%rdi
  401f90:	be f4 00 00 00       	mov    $0xf4,%esi
  401f95:	e8 f6 ed ff ff       	call   400d90 <memset@plt>
  401f9a:	48 8b 05 1f 25 20 00 	mov    0x20251f(%rip),%rax        # 6044c0 <stdin@GLIBC_2.2.5>
  401fa1:	48 39 05 68 25 20 00 	cmp    %rax,0x202568(%rip)        # 604510 <infile>
  401fa8:	75 14                	jne    401fbe <launch+0x5c>
  401faa:	be 69 33 40 00       	mov    $0x403369,%esi
  401faf:	bf 01 00 00 00       	mov    $0x1,%edi
  401fb4:	b8 00 00 00 00       	mov    $0x0,%eax
  401fb9:	e8 32 ed ff ff       	call   400cf0 <__printf_chk@plt>
  401fbe:	c7 05 58 25 20 00 00 	movl   $0x0,0x202558(%rip)        # 604520 <vlevel>
  401fc5:	00 00 00 
  401fc8:	b8 00 00 00 00       	mov    $0x0,%eax
  401fcd:	e8 60 fb ff ff       	call   401b32 <test>
  401fd2:	83 3d 53 25 20 00 00 	cmpl   $0x0,0x202553(%rip)        # 60452c <is_checker>
  401fd9:	74 14                	je     401fef <launch+0x8d>
  401fdb:	bf 76 33 40 00       	mov    $0x403376,%edi
  401fe0:	e8 6b ed ff ff       	call   400d50 <puts@plt>
  401fe5:	b8 00 00 00 00       	mov    $0x0,%eax
  401fea:	e8 d7 fb ff ff       	call   401bc6 <check_fail>
  401fef:	bf 81 33 40 00       	mov    $0x403381,%edi
  401ff4:	e8 57 ed ff ff       	call   400d50 <puts@plt>
  401ff9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401ffd:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  402004:	00 00 
  402006:	74 05                	je     40200d <launch+0xab>
  402008:	e8 63 ed ff ff       	call   400d70 <__stack_chk_fail@plt>
  40200d:	c9                   	leave  
  40200e:	c3                   	ret    

000000000040200f <stable_launch>:
  40200f:	53                   	push   %rbx
  402010:	48 89 3d f1 24 20 00 	mov    %rdi,0x2024f1(%rip)        # 604508 <global_offset>
  402017:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  40201d:	41 b8 00 00 00 00    	mov    $0x0,%r8d
  402023:	b9 32 01 00 00       	mov    $0x132,%ecx
  402028:	ba 07 00 00 00       	mov    $0x7,%edx
  40202d:	be 00 00 10 00       	mov    $0x100000,%esi
  402032:	bf 00 60 58 55       	mov    $0x55586000,%edi
  402037:	e8 44 ed ff ff       	call   400d80 <mmap@plt>
  40203c:	48 89 c3             	mov    %rax,%rbx
  40203f:	48 3d 00 60 58 55    	cmp    $0x55586000,%rax
  402045:	74 37                	je     40207e <stable_launch+0x6f>
  402047:	be 00 00 10 00       	mov    $0x100000,%esi
  40204c:	48 89 c7             	mov    %rax,%rdi
  40204f:	e8 2c ee ff ff       	call   400e80 <munmap@plt>
  402054:	b9 00 60 58 55       	mov    $0x55586000,%ecx
  402059:	ba d0 34 40 00       	mov    $0x4034d0,%edx
  40205e:	be 01 00 00 00       	mov    $0x1,%esi
  402063:	48 8b 3d 76 24 20 00 	mov    0x202476(%rip),%rdi        # 6044e0 <stderr@GLIBC_2.2.5>
  40206a:	b8 00 00 00 00       	mov    $0x0,%eax
  40206f:	e8 7c ee ff ff       	call   400ef0 <__fprintf_chk@plt>
  402074:	bf 01 00 00 00       	mov    $0x1,%edi
  402079:	e8 52 ee ff ff       	call   400ed0 <exit@plt>
  40207e:	48 8d 90 f8 ff 0f 00 	lea    0xffff8(%rax),%rdx
  402085:	48 89 15 bc 30 20 00 	mov    %rdx,0x2030bc(%rip)        # 605148 <stack_top>
  40208c:	48 89 e0             	mov    %rsp,%rax
  40208f:	48 89 d4             	mov    %rdx,%rsp
  402092:	48 89 c2             	mov    %rax,%rdx
  402095:	48 89 15 64 24 20 00 	mov    %rdx,0x202464(%rip)        # 604500 <global_save_stack>
  40209c:	48 8b 3d 65 24 20 00 	mov    0x202465(%rip),%rdi        # 604508 <global_offset>
  4020a3:	e8 ba fe ff ff       	call   401f62 <launch>
  4020a8:	48 8b 05 51 24 20 00 	mov    0x202451(%rip),%rax        # 604500 <global_save_stack>
  4020af:	48 89 c4             	mov    %rax,%rsp
  4020b2:	be 00 00 10 00       	mov    $0x100000,%esi
  4020b7:	48 89 df             	mov    %rbx,%rdi
  4020ba:	e8 c1 ed ff ff       	call   400e80 <munmap@plt>
  4020bf:	5b                   	pop    %rbx
  4020c0:	c3                   	ret    

00000000004020c1 <rio_readinitb>:
  4020c1:	89 37                	mov    %esi,(%rdi)
  4020c3:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  4020ca:	48 8d 47 10          	lea    0x10(%rdi),%rax
  4020ce:	48 89 47 08          	mov    %rax,0x8(%rdi)
  4020d2:	c3                   	ret    

00000000004020d3 <sigalrm_handler>:
  4020d3:	48 83 ec 08          	sub    $0x8,%rsp
  4020d7:	b9 00 00 00 00       	mov    $0x0,%ecx
  4020dc:	ba 10 35 40 00       	mov    $0x403510,%edx
  4020e1:	be 01 00 00 00       	mov    $0x1,%esi
  4020e6:	48 8b 3d f3 23 20 00 	mov    0x2023f3(%rip),%rdi        # 6044e0 <stderr@GLIBC_2.2.5>
  4020ed:	b8 00 00 00 00       	mov    $0x0,%eax
  4020f2:	e8 f9 ed ff ff       	call   400ef0 <__fprintf_chk@plt>
  4020f7:	bf 01 00 00 00       	mov    $0x1,%edi
  4020fc:	e8 cf ed ff ff       	call   400ed0 <exit@plt>

0000000000402101 <rio_writen>:
  402101:	41 55                	push   %r13
  402103:	41 54                	push   %r12
  402105:	55                   	push   %rbp
  402106:	53                   	push   %rbx
  402107:	48 83 ec 08          	sub    $0x8,%rsp
  40210b:	41 89 fc             	mov    %edi,%r12d
  40210e:	48 89 f5             	mov    %rsi,%rbp
  402111:	49 89 d5             	mov    %rdx,%r13
  402114:	48 89 d3             	mov    %rdx,%rbx
  402117:	eb 28                	jmp    402141 <rio_writen+0x40>
  402119:	48 89 da             	mov    %rbx,%rdx
  40211c:	48 89 ee             	mov    %rbp,%rsi
  40211f:	44 89 e7             	mov    %r12d,%edi
  402122:	e8 39 ec ff ff       	call   400d60 <write@plt>
  402127:	48 85 c0             	test   %rax,%rax
  40212a:	7f 0f                	jg     40213b <rio_writen+0x3a>
  40212c:	e8 df eb ff ff       	call   400d10 <__errno_location@plt>
  402131:	83 38 04             	cmpl   $0x4,(%rax)
  402134:	75 15                	jne    40214b <rio_writen+0x4a>
  402136:	b8 00 00 00 00       	mov    $0x0,%eax
  40213b:	48 29 c3             	sub    %rax,%rbx
  40213e:	48 01 c5             	add    %rax,%rbp
  402141:	48 85 db             	test   %rbx,%rbx
  402144:	75 d3                	jne    402119 <rio_writen+0x18>
  402146:	4c 89 e8             	mov    %r13,%rax
  402149:	eb 07                	jmp    402152 <rio_writen+0x51>
  40214b:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  402152:	48 83 c4 08          	add    $0x8,%rsp
  402156:	5b                   	pop    %rbx
  402157:	5d                   	pop    %rbp
  402158:	41 5c                	pop    %r12
  40215a:	41 5d                	pop    %r13
  40215c:	c3                   	ret    

000000000040215d <rio_read>:
  40215d:	41 55                	push   %r13
  40215f:	41 54                	push   %r12
  402161:	55                   	push   %rbp
  402162:	53                   	push   %rbx
  402163:	48 83 ec 08          	sub    $0x8,%rsp
  402167:	48 89 fb             	mov    %rdi,%rbx
  40216a:	49 89 f5             	mov    %rsi,%r13
  40216d:	49 89 d4             	mov    %rdx,%r12
  402170:	eb 2e                	jmp    4021a0 <rio_read+0x43>
  402172:	48 8d 6b 10          	lea    0x10(%rbx),%rbp
  402176:	8b 3b                	mov    (%rbx),%edi
  402178:	ba 00 20 00 00       	mov    $0x2000,%edx
  40217d:	48 89 ee             	mov    %rbp,%rsi
  402180:	e8 3b ec ff ff       	call   400dc0 <read@plt>
  402185:	89 43 04             	mov    %eax,0x4(%rbx)
  402188:	85 c0                	test   %eax,%eax
  40218a:	79 0c                	jns    402198 <rio_read+0x3b>
  40218c:	e8 7f eb ff ff       	call   400d10 <__errno_location@plt>
  402191:	83 38 04             	cmpl   $0x4,(%rax)
  402194:	74 0a                	je     4021a0 <rio_read+0x43>
  402196:	eb 37                	jmp    4021cf <rio_read+0x72>
  402198:	85 c0                	test   %eax,%eax
  40219a:	74 3c                	je     4021d8 <rio_read+0x7b>
  40219c:	48 89 6b 08          	mov    %rbp,0x8(%rbx)
  4021a0:	8b 6b 04             	mov    0x4(%rbx),%ebp
  4021a3:	85 ed                	test   %ebp,%ebp
  4021a5:	7e cb                	jle    402172 <rio_read+0x15>
  4021a7:	89 e8                	mov    %ebp,%eax
  4021a9:	49 39 c4             	cmp    %rax,%r12
  4021ac:	77 03                	ja     4021b1 <rio_read+0x54>
  4021ae:	44 89 e5             	mov    %r12d,%ebp
  4021b1:	4c 63 e5             	movslq %ebp,%r12
  4021b4:	48 8b 73 08          	mov    0x8(%rbx),%rsi
  4021b8:	4c 89 e2             	mov    %r12,%rdx
  4021bb:	4c 89 ef             	mov    %r13,%rdi
  4021be:	e8 5d ec ff ff       	call   400e20 <memcpy@plt>
  4021c3:	4c 01 63 08          	add    %r12,0x8(%rbx)
  4021c7:	29 6b 04             	sub    %ebp,0x4(%rbx)
  4021ca:	4c 89 e0             	mov    %r12,%rax
  4021cd:	eb 0e                	jmp    4021dd <rio_read+0x80>
  4021cf:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4021d6:	eb 05                	jmp    4021dd <rio_read+0x80>
  4021d8:	b8 00 00 00 00       	mov    $0x0,%eax
  4021dd:	48 83 c4 08          	add    $0x8,%rsp
  4021e1:	5b                   	pop    %rbx
  4021e2:	5d                   	pop    %rbp
  4021e3:	41 5c                	pop    %r12
  4021e5:	41 5d                	pop    %r13
  4021e7:	c3                   	ret    

00000000004021e8 <rio_readlineb>:
  4021e8:	41 55                	push   %r13
  4021ea:	41 54                	push   %r12
  4021ec:	55                   	push   %rbp
  4021ed:	53                   	push   %rbx
  4021ee:	48 83 ec 18          	sub    $0x18,%rsp
  4021f2:	49 89 fd             	mov    %rdi,%r13
  4021f5:	48 89 f5             	mov    %rsi,%rbp
  4021f8:	49 89 d4             	mov    %rdx,%r12
  4021fb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  402202:	00 00 
  402204:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  402209:	31 c0                	xor    %eax,%eax
  40220b:	bb 01 00 00 00       	mov    $0x1,%ebx
  402210:	eb 3f                	jmp    402251 <rio_readlineb+0x69>
  402212:	ba 01 00 00 00       	mov    $0x1,%edx
  402217:	48 8d 74 24 07       	lea    0x7(%rsp),%rsi
  40221c:	4c 89 ef             	mov    %r13,%rdi
  40221f:	e8 39 ff ff ff       	call   40215d <rio_read>
  402224:	83 f8 01             	cmp    $0x1,%eax
  402227:	75 15                	jne    40223e <rio_readlineb+0x56>
  402229:	48 8d 45 01          	lea    0x1(%rbp),%rax
  40222d:	0f b6 54 24 07       	movzbl 0x7(%rsp),%edx
  402232:	88 55 00             	mov    %dl,0x0(%rbp)
  402235:	80 7c 24 07 0a       	cmpb   $0xa,0x7(%rsp)
  40223a:	75 0e                	jne    40224a <rio_readlineb+0x62>
  40223c:	eb 1a                	jmp    402258 <rio_readlineb+0x70>
  40223e:	85 c0                	test   %eax,%eax
  402240:	75 22                	jne    402264 <rio_readlineb+0x7c>
  402242:	48 83 fb 01          	cmp    $0x1,%rbx
  402246:	75 13                	jne    40225b <rio_readlineb+0x73>
  402248:	eb 23                	jmp    40226d <rio_readlineb+0x85>
  40224a:	48 83 c3 01          	add    $0x1,%rbx
  40224e:	48 89 c5             	mov    %rax,%rbp
  402251:	4c 39 e3             	cmp    %r12,%rbx
  402254:	72 bc                	jb     402212 <rio_readlineb+0x2a>
  402256:	eb 03                	jmp    40225b <rio_readlineb+0x73>
  402258:	48 89 c5             	mov    %rax,%rbp
  40225b:	c6 45 00 00          	movb   $0x0,0x0(%rbp)
  40225f:	48 89 d8             	mov    %rbx,%rax
  402262:	eb 0e                	jmp    402272 <rio_readlineb+0x8a>
  402264:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  40226b:	eb 05                	jmp    402272 <rio_readlineb+0x8a>
  40226d:	b8 00 00 00 00       	mov    $0x0,%eax
  402272:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  402277:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  40227e:	00 00 
  402280:	74 05                	je     402287 <rio_readlineb+0x9f>
  402282:	e8 e9 ea ff ff       	call   400d70 <__stack_chk_fail@plt>
  402287:	48 83 c4 18          	add    $0x18,%rsp
  40228b:	5b                   	pop    %rbx
  40228c:	5d                   	pop    %rbp
  40228d:	41 5c                	pop    %r12
  40228f:	41 5d                	pop    %r13
  402291:	c3                   	ret    

0000000000402292 <urlencode>:
  402292:	41 54                	push   %r12
  402294:	55                   	push   %rbp
  402295:	53                   	push   %rbx
  402296:	48 83 ec 10          	sub    $0x10,%rsp
  40229a:	48 89 fb             	mov    %rdi,%rbx
  40229d:	48 89 f5             	mov    %rsi,%rbp
  4022a0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4022a7:	00 00 
  4022a9:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4022ae:	31 c0                	xor    %eax,%eax
  4022b0:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  4022b7:	f2 ae                	repnz scas %es:(%rdi),%al
  4022b9:	48 f7 d1             	not    %rcx
  4022bc:	8d 41 ff             	lea    -0x1(%rcx),%eax
  4022bf:	e9 aa 00 00 00       	jmp    40236e <urlencode+0xdc>
  4022c4:	44 0f b6 03          	movzbl (%rbx),%r8d
  4022c8:	41 80 f8 2a          	cmp    $0x2a,%r8b
  4022cc:	0f 94 c2             	sete   %dl
  4022cf:	41 80 f8 2d          	cmp    $0x2d,%r8b
  4022d3:	0f 94 c0             	sete   %al
  4022d6:	08 c2                	or     %al,%dl
  4022d8:	75 24                	jne    4022fe <urlencode+0x6c>
  4022da:	41 80 f8 2e          	cmp    $0x2e,%r8b
  4022de:	74 1e                	je     4022fe <urlencode+0x6c>
  4022e0:	41 80 f8 5f          	cmp    $0x5f,%r8b
  4022e4:	74 18                	je     4022fe <urlencode+0x6c>
  4022e6:	41 8d 40 d0          	lea    -0x30(%r8),%eax
  4022ea:	3c 09                	cmp    $0x9,%al
  4022ec:	76 10                	jbe    4022fe <urlencode+0x6c>
  4022ee:	41 8d 40 bf          	lea    -0x41(%r8),%eax
  4022f2:	3c 19                	cmp    $0x19,%al
  4022f4:	76 08                	jbe    4022fe <urlencode+0x6c>
  4022f6:	41 8d 40 9f          	lea    -0x61(%r8),%eax
  4022fa:	3c 19                	cmp    $0x19,%al
  4022fc:	77 0a                	ja     402308 <urlencode+0x76>
  4022fe:	44 88 45 00          	mov    %r8b,0x0(%rbp)
  402302:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  402306:	eb 5f                	jmp    402367 <urlencode+0xd5>
  402308:	41 80 f8 20          	cmp    $0x20,%r8b
  40230c:	75 0a                	jne    402318 <urlencode+0x86>
  40230e:	c6 45 00 2b          	movb   $0x2b,0x0(%rbp)
  402312:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  402316:	eb 4f                	jmp    402367 <urlencode+0xd5>
  402318:	41 8d 40 e0          	lea    -0x20(%r8),%eax
  40231c:	3c 5f                	cmp    $0x5f,%al
  40231e:	0f 96 c2             	setbe  %dl
  402321:	41 80 f8 09          	cmp    $0x9,%r8b
  402325:	0f 94 c0             	sete   %al
  402328:	08 c2                	or     %al,%dl
  40232a:	74 50                	je     40237c <urlencode+0xea>
  40232c:	45 0f b6 c0          	movzbl %r8b,%r8d
  402330:	b9 a8 35 40 00       	mov    $0x4035a8,%ecx
  402335:	ba 08 00 00 00       	mov    $0x8,%edx
  40233a:	be 01 00 00 00       	mov    $0x1,%esi
  40233f:	48 89 e7             	mov    %rsp,%rdi
  402342:	b8 00 00 00 00       	mov    $0x0,%eax
  402347:	e8 e4 ea ff ff       	call   400e30 <__sprintf_chk@plt>
  40234c:	0f b6 04 24          	movzbl (%rsp),%eax
  402350:	88 45 00             	mov    %al,0x0(%rbp)
  402353:	0f b6 44 24 01       	movzbl 0x1(%rsp),%eax
  402358:	88 45 01             	mov    %al,0x1(%rbp)
  40235b:	0f b6 44 24 02       	movzbl 0x2(%rsp),%eax
  402360:	88 45 02             	mov    %al,0x2(%rbp)
  402363:	48 8d 6d 03          	lea    0x3(%rbp),%rbp
  402367:	48 83 c3 01          	add    $0x1,%rbx
  40236b:	44 89 e0             	mov    %r12d,%eax
  40236e:	44 8d 60 ff          	lea    -0x1(%rax),%r12d
  402372:	85 c0                	test   %eax,%eax
  402374:	0f 85 4a ff ff ff    	jne    4022c4 <urlencode+0x32>
  40237a:	eb 05                	jmp    402381 <urlencode+0xef>
  40237c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402381:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  402386:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  40238d:	00 00 
  40238f:	74 05                	je     402396 <urlencode+0x104>
  402391:	e8 da e9 ff ff       	call   400d70 <__stack_chk_fail@plt>
  402396:	48 83 c4 10          	add    $0x10,%rsp
  40239a:	5b                   	pop    %rbx
  40239b:	5d                   	pop    %rbp
  40239c:	41 5c                	pop    %r12
  40239e:	c3                   	ret    

000000000040239f <submitr>:
  40239f:	41 57                	push   %r15
  4023a1:	41 56                	push   %r14
  4023a3:	41 55                	push   %r13
  4023a5:	41 54                	push   %r12
  4023a7:	55                   	push   %rbp
  4023a8:	53                   	push   %rbx
  4023a9:	48 81 ec 58 a0 00 00 	sub    $0xa058,%rsp
  4023b0:	49 89 fc             	mov    %rdi,%r12
  4023b3:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4023b7:	49 89 d7             	mov    %rdx,%r15
  4023ba:	49 89 ce             	mov    %rcx,%r14
  4023bd:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  4023c2:	4d 89 cd             	mov    %r9,%r13
  4023c5:	48 8b 9c 24 90 a0 00 	mov    0xa090(%rsp),%rbx
  4023cc:	00 
  4023cd:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4023d4:	00 00 
  4023d6:	48 89 84 24 48 a0 00 	mov    %rax,0xa048(%rsp)
  4023dd:	00 
  4023de:	31 c0                	xor    %eax,%eax
  4023e0:	c7 44 24 1c 00 00 00 	movl   $0x0,0x1c(%rsp)
  4023e7:	00 
  4023e8:	ba 00 00 00 00       	mov    $0x0,%edx
  4023ed:	be 01 00 00 00       	mov    $0x1,%esi
  4023f2:	bf 02 00 00 00       	mov    $0x2,%edi
  4023f7:	e8 04 eb ff ff       	call   400f00 <socket@plt>
  4023fc:	85 c0                	test   %eax,%eax
  4023fe:	79 4e                	jns    40244e <submitr+0xaf>
  402400:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402407:	3a 20 43 
  40240a:	48 89 03             	mov    %rax,(%rbx)
  40240d:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402414:	20 75 6e 
  402417:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40241b:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402422:	74 6f 20 
  402425:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402429:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  402430:	65 20 73 
  402433:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402437:	c7 43 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbx)
  40243e:	66 c7 43 24 74 00    	movw   $0x74,0x24(%rbx)
  402444:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402449:	e9 97 06 00 00       	jmp    402ae5 <submitr+0x746>
  40244e:	89 c5                	mov    %eax,%ebp
  402450:	4c 89 e7             	mov    %r12,%rdi
  402453:	e8 98 e9 ff ff       	call   400df0 <gethostbyname@plt>
  402458:	48 85 c0             	test   %rax,%rax
  40245b:	75 67                	jne    4024c4 <submitr+0x125>
  40245d:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  402464:	3a 20 44 
  402467:	48 89 03             	mov    %rax,(%rbx)
  40246a:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  402471:	20 75 6e 
  402474:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402478:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  40247f:	74 6f 20 
  402482:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402486:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  40248d:	76 65 20 
  402490:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402494:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  40249b:	72 20 61 
  40249e:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4024a2:	c7 43 28 64 64 72 65 	movl   $0x65726464,0x28(%rbx)
  4024a9:	66 c7 43 2c 73 73    	movw   $0x7373,0x2c(%rbx)
  4024af:	c6 43 2e 00          	movb   $0x0,0x2e(%rbx)
  4024b3:	89 ef                	mov    %ebp,%edi
  4024b5:	e8 f6 e8 ff ff       	call   400db0 <close@plt>
  4024ba:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4024bf:	e9 21 06 00 00       	jmp    402ae5 <submitr+0x746>
  4024c4:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
  4024cb:	00 00 
  4024cd:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
  4024d4:	00 00 
  4024d6:	66 c7 44 24 20 02 00 	movw   $0x2,0x20(%rsp)
  4024dd:	48 63 50 14          	movslq 0x14(%rax),%rdx
  4024e1:	48 8b 40 18          	mov    0x18(%rax),%rax
  4024e5:	48 8b 30             	mov    (%rax),%rsi
  4024e8:	48 8d 7c 24 24       	lea    0x24(%rsp),%rdi
  4024ed:	b9 0c 00 00 00       	mov    $0xc,%ecx
  4024f2:	e8 09 e9 ff ff       	call   400e00 <__memmove_chk@plt>
  4024f7:	0f b7 44 24 04       	movzwl 0x4(%rsp),%eax
  4024fc:	66 c1 c8 08          	ror    $0x8,%ax
  402500:	66 89 44 24 22       	mov    %ax,0x22(%rsp)
  402505:	ba 10 00 00 00       	mov    $0x10,%edx
  40250a:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  40250f:	89 ef                	mov    %ebp,%edi
  402511:	e8 ca e9 ff ff       	call   400ee0 <connect@plt>
  402516:	85 c0                	test   %eax,%eax
  402518:	79 59                	jns    402573 <submitr+0x1d4>
  40251a:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
  402521:	3a 20 55 
  402524:	48 89 03             	mov    %rax,(%rbx)
  402527:	48 b8 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rax
  40252e:	20 74 6f 
  402531:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402535:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
  40253c:	65 63 74 
  40253f:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402543:	48 b8 20 74 6f 20 74 	movabs $0x20656874206f7420,%rax
  40254a:	68 65 20 
  40254d:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402551:	c7 43 20 73 65 72 76 	movl   $0x76726573,0x20(%rbx)
  402558:	66 c7 43 24 65 72    	movw   $0x7265,0x24(%rbx)
  40255e:	c6 43 26 00          	movb   $0x0,0x26(%rbx)
  402562:	89 ef                	mov    %ebp,%edi
  402564:	e8 47 e8 ff ff       	call   400db0 <close@plt>
  402569:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40256e:	e9 72 05 00 00       	jmp    402ae5 <submitr+0x746>
  402573:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  40257a:	b8 00 00 00 00       	mov    $0x0,%eax
  40257f:	48 89 f1             	mov    %rsi,%rcx
  402582:	4c 89 ef             	mov    %r13,%rdi
  402585:	f2 ae                	repnz scas %es:(%rdi),%al
  402587:	48 f7 d1             	not    %rcx
  40258a:	48 89 ca             	mov    %rcx,%rdx
  40258d:	48 89 f1             	mov    %rsi,%rcx
  402590:	4c 89 ff             	mov    %r15,%rdi
  402593:	f2 ae                	repnz scas %es:(%rdi),%al
  402595:	48 f7 d1             	not    %rcx
  402598:	49 89 c8             	mov    %rcx,%r8
  40259b:	48 89 f1             	mov    %rsi,%rcx
  40259e:	4c 89 f7             	mov    %r14,%rdi
  4025a1:	f2 ae                	repnz scas %es:(%rdi),%al
  4025a3:	48 f7 d1             	not    %rcx
  4025a6:	4d 8d 44 08 fe       	lea    -0x2(%r8,%rcx,1),%r8
  4025ab:	48 89 f1             	mov    %rsi,%rcx
  4025ae:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4025b3:	f2 ae                	repnz scas %es:(%rdi),%al
  4025b5:	48 89 c8             	mov    %rcx,%rax
  4025b8:	48 f7 d0             	not    %rax
  4025bb:	49 8d 4c 00 ff       	lea    -0x1(%r8,%rax,1),%rcx
  4025c0:	48 8d 44 52 fd       	lea    -0x3(%rdx,%rdx,2),%rax
  4025c5:	48 8d 84 01 80 00 00 	lea    0x80(%rcx,%rax,1),%rax
  4025cc:	00 
  4025cd:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
  4025d3:	76 72                	jbe    402647 <submitr+0x2a8>
  4025d5:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  4025dc:	3a 20 52 
  4025df:	48 89 03             	mov    %rax,(%rbx)
  4025e2:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  4025e9:	20 73 74 
  4025ec:	48 89 43 08          	mov    %rax,0x8(%rbx)
  4025f0:	48 b8 72 69 6e 67 20 	movabs $0x6f6f7420676e6972,%rax
  4025f7:	74 6f 6f 
  4025fa:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4025fe:	48 b8 20 6c 61 72 67 	movabs $0x202e656772616c20,%rax
  402605:	65 2e 20 
  402608:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40260c:	48 b8 49 6e 63 72 65 	movabs $0x6573616572636e49,%rax
  402613:	61 73 65 
  402616:	48 89 43 20          	mov    %rax,0x20(%rbx)
  40261a:	48 b8 20 53 55 42 4d 	movabs $0x5254494d42555320,%rax
  402621:	49 54 52 
  402624:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402628:	48 b8 5f 4d 41 58 42 	movabs $0x46554258414d5f,%rax
  40262f:	55 46 00 
  402632:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402636:	89 ef                	mov    %ebp,%edi
  402638:	e8 73 e7 ff ff       	call   400db0 <close@plt>
  40263d:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402642:	e9 9e 04 00 00       	jmp    402ae5 <submitr+0x746>
  402647:	48 8d b4 24 40 40 00 	lea    0x4040(%rsp),%rsi
  40264e:	00 
  40264f:	b9 00 04 00 00       	mov    $0x400,%ecx
  402654:	b8 00 00 00 00       	mov    $0x0,%eax
  402659:	48 89 f7             	mov    %rsi,%rdi
  40265c:	f3 48 ab             	rep stos %rax,%es:(%rdi)
  40265f:	4c 89 ef             	mov    %r13,%rdi
  402662:	e8 2b fc ff ff       	call   402292 <urlencode>
  402667:	85 c0                	test   %eax,%eax
  402669:	0f 89 8a 00 00 00    	jns    4026f9 <submitr+0x35a>
  40266f:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  402676:	3a 20 52 
  402679:	48 89 03             	mov    %rax,(%rbx)
  40267c:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  402683:	20 73 74 
  402686:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40268a:	48 b8 72 69 6e 67 20 	movabs $0x6e6f6320676e6972,%rax
  402691:	63 6f 6e 
  402694:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402698:	48 b8 74 61 69 6e 73 	movabs $0x6e6120736e696174,%rax
  40269f:	20 61 6e 
  4026a2:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4026a6:	48 b8 20 69 6c 6c 65 	movabs $0x6c6167656c6c6920,%rax
  4026ad:	67 61 6c 
  4026b0:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4026b4:	48 b8 20 6f 72 20 75 	movabs $0x72706e7520726f20,%rax
  4026bb:	6e 70 72 
  4026be:	48 89 43 28          	mov    %rax,0x28(%rbx)
  4026c2:	48 b8 69 6e 74 61 62 	movabs $0x20656c6261746e69,%rax
  4026c9:	6c 65 20 
  4026cc:	48 89 43 30          	mov    %rax,0x30(%rbx)
  4026d0:	48 b8 63 68 61 72 61 	movabs $0x6574636172616863,%rax
  4026d7:	63 74 65 
  4026da:	48 89 43 38          	mov    %rax,0x38(%rbx)
  4026de:	66 c7 43 40 72 2e    	movw   $0x2e72,0x40(%rbx)
  4026e4:	c6 43 42 00          	movb   $0x0,0x42(%rbx)
  4026e8:	89 ef                	mov    %ebp,%edi
  4026ea:	e8 c1 e6 ff ff       	call   400db0 <close@plt>
  4026ef:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4026f4:	e9 ec 03 00 00       	jmp    402ae5 <submitr+0x746>
  4026f9:	4c 8d ac 24 40 20 00 	lea    0x2040(%rsp),%r13
  402700:	00 
  402701:	41 54                	push   %r12
  402703:	48 8d 84 24 48 40 00 	lea    0x4048(%rsp),%rax
  40270a:	00 
  40270b:	50                   	push   %rax
  40270c:	4d 89 f9             	mov    %r15,%r9
  40270f:	4d 89 f0             	mov    %r14,%r8
  402712:	b9 38 35 40 00       	mov    $0x403538,%ecx
  402717:	ba 00 20 00 00       	mov    $0x2000,%edx
  40271c:	be 01 00 00 00       	mov    $0x1,%esi
  402721:	4c 89 ef             	mov    %r13,%rdi
  402724:	b8 00 00 00 00       	mov    $0x0,%eax
  402729:	e8 02 e7 ff ff       	call   400e30 <__sprintf_chk@plt>
  40272e:	b8 00 00 00 00       	mov    $0x0,%eax
  402733:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  40273a:	4c 89 ef             	mov    %r13,%rdi
  40273d:	f2 ae                	repnz scas %es:(%rdi),%al
  40273f:	48 f7 d1             	not    %rcx
  402742:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
  402746:	4c 89 ee             	mov    %r13,%rsi
  402749:	89 ef                	mov    %ebp,%edi
  40274b:	e8 b1 f9 ff ff       	call   402101 <rio_writen>
  402750:	48 83 c4 10          	add    $0x10,%rsp
  402754:	48 85 c0             	test   %rax,%rax
  402757:	79 6e                	jns    4027c7 <submitr+0x428>
  402759:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402760:	3a 20 43 
  402763:	48 89 03             	mov    %rax,(%rbx)
  402766:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  40276d:	20 75 6e 
  402770:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402774:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  40277b:	74 6f 20 
  40277e:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402782:	48 b8 77 72 69 74 65 	movabs $0x6f74206574697277,%rax
  402789:	20 74 6f 
  40278c:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402790:	48 b8 20 74 68 65 20 	movabs $0x7365722065687420,%rax
  402797:	72 65 73 
  40279a:	48 89 43 20          	mov    %rax,0x20(%rbx)
  40279e:	48 b8 75 6c 74 20 73 	movabs $0x7672657320746c75,%rax
  4027a5:	65 72 76 
  4027a8:	48 89 43 28          	mov    %rax,0x28(%rbx)
  4027ac:	66 c7 43 30 65 72    	movw   $0x7265,0x30(%rbx)
  4027b2:	c6 43 32 00          	movb   $0x0,0x32(%rbx)
  4027b6:	89 ef                	mov    %ebp,%edi
  4027b8:	e8 f3 e5 ff ff       	call   400db0 <close@plt>
  4027bd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4027c2:	e9 1e 03 00 00       	jmp    402ae5 <submitr+0x746>
  4027c7:	89 ee                	mov    %ebp,%esi
  4027c9:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4027ce:	e8 ee f8 ff ff       	call   4020c1 <rio_readinitb>
  4027d3:	ba 00 20 00 00       	mov    $0x2000,%edx
  4027d8:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  4027df:	00 
  4027e0:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4027e5:	e8 fe f9 ff ff       	call   4021e8 <rio_readlineb>
  4027ea:	48 85 c0             	test   %rax,%rax
  4027ed:	7f 7d                	jg     40286c <submitr+0x4cd>
  4027ef:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  4027f6:	3a 20 43 
  4027f9:	48 89 03             	mov    %rax,(%rbx)
  4027fc:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402803:	20 75 6e 
  402806:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40280a:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402811:	74 6f 20 
  402814:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402818:	48 b8 72 65 61 64 20 	movabs $0x7269662064616572,%rax
  40281f:	66 69 72 
  402822:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402826:	48 b8 73 74 20 68 65 	movabs $0x6564616568207473,%rax
  40282d:	61 64 65 
  402830:	48 89 43 20          	mov    %rax,0x20(%rbx)
  402834:	48 b8 72 20 66 72 6f 	movabs $0x72206d6f72662072,%rax
  40283b:	6d 20 72 
  40283e:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402842:	48 b8 65 73 75 6c 74 	movabs $0x657320746c757365,%rax
  402849:	20 73 65 
  40284c:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402850:	c7 43 38 72 76 65 72 	movl   $0x72657672,0x38(%rbx)
  402857:	c6 43 3c 00          	movb   $0x0,0x3c(%rbx)
  40285b:	89 ef                	mov    %ebp,%edi
  40285d:	e8 4e e5 ff ff       	call   400db0 <close@plt>
  402862:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402867:	e9 79 02 00 00       	jmp    402ae5 <submitr+0x746>
  40286c:	4c 8d 84 24 40 80 00 	lea    0x8040(%rsp),%r8
  402873:	00 
  402874:	48 8d 4c 24 1c       	lea    0x1c(%rsp),%rcx
  402879:	48 8d 94 24 40 60 00 	lea    0x6040(%rsp),%rdx
  402880:	00 
  402881:	be af 35 40 00       	mov    $0x4035af,%esi
  402886:	48 8d bc 24 40 20 00 	lea    0x2040(%rsp),%rdi
  40288d:	00 
  40288e:	b8 00 00 00 00       	mov    $0x0,%eax
  402893:	e8 d8 e5 ff ff       	call   400e70 <__isoc99_sscanf@plt>
  402898:	e9 95 00 00 00       	jmp    402932 <submitr+0x593>
  40289d:	ba 00 20 00 00       	mov    $0x2000,%edx
  4028a2:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  4028a9:	00 
  4028aa:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4028af:	e8 34 f9 ff ff       	call   4021e8 <rio_readlineb>
  4028b4:	48 85 c0             	test   %rax,%rax
  4028b7:	7f 79                	jg     402932 <submitr+0x593>
  4028b9:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  4028c0:	3a 20 43 
  4028c3:	48 89 03             	mov    %rax,(%rbx)
  4028c6:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  4028cd:	20 75 6e 
  4028d0:	48 89 43 08          	mov    %rax,0x8(%rbx)
  4028d4:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  4028db:	74 6f 20 
  4028de:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4028e2:	48 b8 72 65 61 64 20 	movabs $0x6165682064616572,%rax
  4028e9:	68 65 61 
  4028ec:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4028f0:	48 b8 64 65 72 73 20 	movabs $0x6f72662073726564,%rax
  4028f7:	66 72 6f 
  4028fa:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4028fe:	48 b8 6d 20 74 68 65 	movabs $0x657220656874206d,%rax
  402905:	20 72 65 
  402908:	48 89 43 28          	mov    %rax,0x28(%rbx)
  40290c:	48 b8 73 75 6c 74 20 	movabs $0x72657320746c7573,%rax
  402913:	73 65 72 
  402916:	48 89 43 30          	mov    %rax,0x30(%rbx)
  40291a:	c7 43 38 76 65 72 00 	movl   $0x726576,0x38(%rbx)
  402921:	89 ef                	mov    %ebp,%edi
  402923:	e8 88 e4 ff ff       	call   400db0 <close@plt>
  402928:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40292d:	e9 b3 01 00 00       	jmp    402ae5 <submitr+0x746>
  402932:	0f b6 94 24 40 20 00 	movzbl 0x2040(%rsp),%edx
  402939:	00 
  40293a:	b8 0d 00 00 00       	mov    $0xd,%eax
  40293f:	29 d0                	sub    %edx,%eax
  402941:	75 1b                	jne    40295e <submitr+0x5bf>
  402943:	0f b6 94 24 41 20 00 	movzbl 0x2041(%rsp),%edx
  40294a:	00 
  40294b:	b8 0a 00 00 00       	mov    $0xa,%eax
  402950:	29 d0                	sub    %edx,%eax
  402952:	75 0a                	jne    40295e <submitr+0x5bf>
  402954:	0f b6 84 24 42 20 00 	movzbl 0x2042(%rsp),%eax
  40295b:	00 
  40295c:	f7 d8                	neg    %eax
  40295e:	85 c0                	test   %eax,%eax
  402960:	0f 85 37 ff ff ff    	jne    40289d <submitr+0x4fe>
  402966:	ba 00 20 00 00       	mov    $0x2000,%edx
  40296b:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  402972:	00 
  402973:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402978:	e8 6b f8 ff ff       	call   4021e8 <rio_readlineb>
  40297d:	48 85 c0             	test   %rax,%rax
  402980:	0f 8f 83 00 00 00    	jg     402a09 <submitr+0x66a>
  402986:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  40298d:	3a 20 43 
  402990:	48 89 03             	mov    %rax,(%rbx)
  402993:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  40299a:	20 75 6e 
  40299d:	48 89 43 08          	mov    %rax,0x8(%rbx)
  4029a1:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  4029a8:	74 6f 20 
  4029ab:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4029af:	48 b8 72 65 61 64 20 	movabs $0x6174732064616572,%rax
  4029b6:	73 74 61 
  4029b9:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4029bd:	48 b8 74 75 73 20 6d 	movabs $0x7373656d20737574,%rax
  4029c4:	65 73 73 
  4029c7:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4029cb:	48 b8 61 67 65 20 66 	movabs $0x6d6f726620656761,%rax
  4029d2:	72 6f 6d 
  4029d5:	48 89 43 28          	mov    %rax,0x28(%rbx)
  4029d9:	48 b8 20 72 65 73 75 	movabs $0x20746c7573657220,%rax
  4029e0:	6c 74 20 
  4029e3:	48 89 43 30          	mov    %rax,0x30(%rbx)
  4029e7:	c7 43 38 73 65 72 76 	movl   $0x76726573,0x38(%rbx)
  4029ee:	66 c7 43 3c 65 72    	movw   $0x7265,0x3c(%rbx)
  4029f4:	c6 43 3e 00          	movb   $0x0,0x3e(%rbx)
  4029f8:	89 ef                	mov    %ebp,%edi
  4029fa:	e8 b1 e3 ff ff       	call   400db0 <close@plt>
  4029ff:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402a04:	e9 dc 00 00 00       	jmp    402ae5 <submitr+0x746>
  402a09:	44 8b 44 24 1c       	mov    0x1c(%rsp),%r8d
  402a0e:	41 81 f8 c8 00 00 00 	cmp    $0xc8,%r8d
  402a15:	74 37                	je     402a4e <submitr+0x6af>
  402a17:	4c 8d 8c 24 40 80 00 	lea    0x8040(%rsp),%r9
  402a1e:	00 
  402a1f:	b9 78 35 40 00       	mov    $0x403578,%ecx
  402a24:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  402a2b:	be 01 00 00 00       	mov    $0x1,%esi
  402a30:	48 89 df             	mov    %rbx,%rdi
  402a33:	b8 00 00 00 00       	mov    $0x0,%eax
  402a38:	e8 f3 e3 ff ff       	call   400e30 <__sprintf_chk@plt>
  402a3d:	89 ef                	mov    %ebp,%edi
  402a3f:	e8 6c e3 ff ff       	call   400db0 <close@plt>
  402a44:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402a49:	e9 97 00 00 00       	jmp    402ae5 <submitr+0x746>
  402a4e:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  402a55:	00 
  402a56:	48 89 df             	mov    %rbx,%rdi
  402a59:	e8 e2 e2 ff ff       	call   400d40 <strcpy@plt>
  402a5e:	89 ef                	mov    %ebp,%edi
  402a60:	e8 4b e3 ff ff       	call   400db0 <close@plt>
  402a65:	0f b6 03             	movzbl (%rbx),%eax
  402a68:	ba 4f 00 00 00       	mov    $0x4f,%edx
  402a6d:	29 c2                	sub    %eax,%edx
  402a6f:	75 22                	jne    402a93 <submitr+0x6f4>
  402a71:	0f b6 4b 01          	movzbl 0x1(%rbx),%ecx
  402a75:	b8 4b 00 00 00       	mov    $0x4b,%eax
  402a7a:	29 c8                	sub    %ecx,%eax
  402a7c:	75 17                	jne    402a95 <submitr+0x6f6>
  402a7e:	0f b6 4b 02          	movzbl 0x2(%rbx),%ecx
  402a82:	b8 0a 00 00 00       	mov    $0xa,%eax
  402a87:	29 c8                	sub    %ecx,%eax
  402a89:	75 0a                	jne    402a95 <submitr+0x6f6>
  402a8b:	0f b6 43 03          	movzbl 0x3(%rbx),%eax
  402a8f:	f7 d8                	neg    %eax
  402a91:	eb 02                	jmp    402a95 <submitr+0x6f6>
  402a93:	89 d0                	mov    %edx,%eax
  402a95:	85 c0                	test   %eax,%eax
  402a97:	74 40                	je     402ad9 <submitr+0x73a>
  402a99:	bf c0 35 40 00       	mov    $0x4035c0,%edi
  402a9e:	b9 05 00 00 00       	mov    $0x5,%ecx
  402aa3:	48 89 de             	mov    %rbx,%rsi
  402aa6:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  402aa8:	0f 97 c0             	seta   %al
  402aab:	0f 92 c1             	setb   %cl
  402aae:	29 c8                	sub    %ecx,%eax
  402ab0:	0f be c0             	movsbl %al,%eax
  402ab3:	85 c0                	test   %eax,%eax
  402ab5:	74 2e                	je     402ae5 <submitr+0x746>
  402ab7:	85 d2                	test   %edx,%edx
  402ab9:	75 13                	jne    402ace <submitr+0x72f>
  402abb:	0f b6 43 01          	movzbl 0x1(%rbx),%eax
  402abf:	ba 4b 00 00 00       	mov    $0x4b,%edx
  402ac4:	29 c2                	sub    %eax,%edx
  402ac6:	75 06                	jne    402ace <submitr+0x72f>
  402ac8:	0f b6 53 02          	movzbl 0x2(%rbx),%edx
  402acc:	f7 da                	neg    %edx
  402ace:	85 d2                	test   %edx,%edx
  402ad0:	75 0e                	jne    402ae0 <submitr+0x741>
  402ad2:	b8 00 00 00 00       	mov    $0x0,%eax
  402ad7:	eb 0c                	jmp    402ae5 <submitr+0x746>
  402ad9:	b8 00 00 00 00       	mov    $0x0,%eax
  402ade:	eb 05                	jmp    402ae5 <submitr+0x746>
  402ae0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402ae5:	48 8b 9c 24 48 a0 00 	mov    0xa048(%rsp),%rbx
  402aec:	00 
  402aed:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
  402af4:	00 00 
  402af6:	74 05                	je     402afd <submitr+0x75e>
  402af8:	e8 73 e2 ff ff       	call   400d70 <__stack_chk_fail@plt>
  402afd:	48 81 c4 58 a0 00 00 	add    $0xa058,%rsp
  402b04:	5b                   	pop    %rbx
  402b05:	5d                   	pop    %rbp
  402b06:	41 5c                	pop    %r12
  402b08:	41 5d                	pop    %r13
  402b0a:	41 5e                	pop    %r14
  402b0c:	41 5f                	pop    %r15
  402b0e:	c3                   	ret    

0000000000402b0f <init_timeout>:
  402b0f:	85 ff                	test   %edi,%edi
  402b11:	74 23                	je     402b36 <init_timeout+0x27>
  402b13:	53                   	push   %rbx
  402b14:	89 fb                	mov    %edi,%ebx
  402b16:	85 ff                	test   %edi,%edi
  402b18:	79 05                	jns    402b1f <init_timeout+0x10>
  402b1a:	bb 00 00 00 00       	mov    $0x0,%ebx
  402b1f:	be d3 20 40 00       	mov    $0x4020d3,%esi
  402b24:	bf 0e 00 00 00       	mov    $0xe,%edi
  402b29:	e8 b2 e2 ff ff       	call   400de0 <signal@plt>
  402b2e:	89 df                	mov    %ebx,%edi
  402b30:	e8 6b e2 ff ff       	call   400da0 <alarm@plt>
  402b35:	5b                   	pop    %rbx
  402b36:	f3 c3                	repz ret 

0000000000402b38 <init_driver>:
  402b38:	55                   	push   %rbp
  402b39:	53                   	push   %rbx
  402b3a:	48 83 ec 28          	sub    $0x28,%rsp
  402b3e:	48 89 fd             	mov    %rdi,%rbp
  402b41:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  402b48:	00 00 
  402b4a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  402b4f:	31 c0                	xor    %eax,%eax
  402b51:	be 01 00 00 00       	mov    $0x1,%esi
  402b56:	bf 0d 00 00 00       	mov    $0xd,%edi
  402b5b:	e8 80 e2 ff ff       	call   400de0 <signal@plt>
  402b60:	be 01 00 00 00       	mov    $0x1,%esi
  402b65:	bf 1d 00 00 00       	mov    $0x1d,%edi
  402b6a:	e8 71 e2 ff ff       	call   400de0 <signal@plt>
  402b6f:	be 01 00 00 00       	mov    $0x1,%esi
  402b74:	bf 1d 00 00 00       	mov    $0x1d,%edi
  402b79:	e8 62 e2 ff ff       	call   400de0 <signal@plt>
  402b7e:	ba 00 00 00 00       	mov    $0x0,%edx
  402b83:	be 01 00 00 00       	mov    $0x1,%esi
  402b88:	bf 02 00 00 00       	mov    $0x2,%edi
  402b8d:	e8 6e e3 ff ff       	call   400f00 <socket@plt>
  402b92:	85 c0                	test   %eax,%eax
  402b94:	79 4f                	jns    402be5 <init_driver+0xad>
  402b96:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402b9d:	3a 20 43 
  402ba0:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402ba4:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402bab:	20 75 6e 
  402bae:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402bb2:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402bb9:	74 6f 20 
  402bbc:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402bc0:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  402bc7:	65 20 73 
  402bca:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402bce:	c7 45 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbp)
  402bd5:	66 c7 45 24 74 00    	movw   $0x74,0x24(%rbp)
  402bdb:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402be0:	e9 2a 01 00 00       	jmp    402d0f <init_driver+0x1d7>
  402be5:	89 c3                	mov    %eax,%ebx
  402be7:	bf c5 35 40 00       	mov    $0x4035c5,%edi
  402bec:	e8 ff e1 ff ff       	call   400df0 <gethostbyname@plt>
  402bf1:	48 85 c0             	test   %rax,%rax
  402bf4:	75 68                	jne    402c5e <init_driver+0x126>
  402bf6:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  402bfd:	3a 20 44 
  402c00:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402c04:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  402c0b:	20 75 6e 
  402c0e:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402c12:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402c19:	74 6f 20 
  402c1c:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402c20:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  402c27:	76 65 20 
  402c2a:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402c2e:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  402c35:	72 20 61 
  402c38:	48 89 45 20          	mov    %rax,0x20(%rbp)
  402c3c:	c7 45 28 64 64 72 65 	movl   $0x65726464,0x28(%rbp)
  402c43:	66 c7 45 2c 73 73    	movw   $0x7373,0x2c(%rbp)
  402c49:	c6 45 2e 00          	movb   $0x0,0x2e(%rbp)
  402c4d:	89 df                	mov    %ebx,%edi
  402c4f:	e8 5c e1 ff ff       	call   400db0 <close@plt>
  402c54:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402c59:	e9 b1 00 00 00       	jmp    402d0f <init_driver+0x1d7>
  402c5e:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  402c65:	00 
  402c66:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
  402c6d:	00 00 
  402c6f:	66 c7 04 24 02 00    	movw   $0x2,(%rsp)
  402c75:	48 63 50 14          	movslq 0x14(%rax),%rdx
  402c79:	48 8b 40 18          	mov    0x18(%rax),%rax
  402c7d:	48 8b 30             	mov    (%rax),%rsi
  402c80:	48 8d 7c 24 04       	lea    0x4(%rsp),%rdi
  402c85:	b9 0c 00 00 00       	mov    $0xc,%ecx
  402c8a:	e8 71 e1 ff ff       	call   400e00 <__memmove_chk@plt>
  402c8f:	66 c7 44 24 02 3c 9a 	movw   $0x9a3c,0x2(%rsp)
  402c96:	ba 10 00 00 00       	mov    $0x10,%edx
  402c9b:	48 89 e6             	mov    %rsp,%rsi
  402c9e:	89 df                	mov    %ebx,%edi
  402ca0:	e8 3b e2 ff ff       	call   400ee0 <connect@plt>
  402ca5:	85 c0                	test   %eax,%eax
  402ca7:	79 50                	jns    402cf9 <init_driver+0x1c1>
  402ca9:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
  402cb0:	3a 20 55 
  402cb3:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402cb7:	48 b8 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rax
  402cbe:	20 74 6f 
  402cc1:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402cc5:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
  402ccc:	65 63 74 
  402ccf:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402cd3:	48 b8 20 74 6f 20 73 	movabs $0x76726573206f7420,%rax
  402cda:	65 72 76 
  402cdd:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402ce1:	66 c7 45 20 65 72    	movw   $0x7265,0x20(%rbp)
  402ce7:	c6 45 22 00          	movb   $0x0,0x22(%rbp)
  402ceb:	89 df                	mov    %ebx,%edi
  402ced:	e8 be e0 ff ff       	call   400db0 <close@plt>
  402cf2:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402cf7:	eb 16                	jmp    402d0f <init_driver+0x1d7>
  402cf9:	89 df                	mov    %ebx,%edi
  402cfb:	e8 b0 e0 ff ff       	call   400db0 <close@plt>
  402d00:	66 c7 45 00 4f 4b    	movw   $0x4b4f,0x0(%rbp)
  402d06:	c6 45 02 00          	movb   $0x0,0x2(%rbp)
  402d0a:	b8 00 00 00 00       	mov    $0x0,%eax
  402d0f:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  402d14:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  402d1b:	00 00 
  402d1d:	74 05                	je     402d24 <init_driver+0x1ec>
  402d1f:	e8 4c e0 ff ff       	call   400d70 <__stack_chk_fail@plt>
  402d24:	48 83 c4 28          	add    $0x28,%rsp
  402d28:	5b                   	pop    %rbx
  402d29:	5d                   	pop    %rbp
  402d2a:	c3                   	ret    

0000000000402d2b <driver_post>:
  402d2b:	53                   	push   %rbx
  402d2c:	4c 89 cb             	mov    %r9,%rbx
  402d2f:	45 85 c0             	test   %r8d,%r8d
  402d32:	74 27                	je     402d5b <driver_post+0x30>
  402d34:	48 89 ca             	mov    %rcx,%rdx
  402d37:	be dd 35 40 00       	mov    $0x4035dd,%esi
  402d3c:	bf 01 00 00 00       	mov    $0x1,%edi
  402d41:	b8 00 00 00 00       	mov    $0x0,%eax
  402d46:	e8 a5 df ff ff       	call   400cf0 <__printf_chk@plt>
  402d4b:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  402d50:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  402d54:	b8 00 00 00 00       	mov    $0x0,%eax
  402d59:	eb 3f                	jmp    402d9a <driver_post+0x6f>
  402d5b:	48 85 ff             	test   %rdi,%rdi
  402d5e:	74 2c                	je     402d8c <driver_post+0x61>
  402d60:	80 3f 00             	cmpb   $0x0,(%rdi)
  402d63:	74 27                	je     402d8c <driver_post+0x61>
  402d65:	48 83 ec 08          	sub    $0x8,%rsp
  402d69:	41 51                	push   %r9
  402d6b:	49 89 c9             	mov    %rcx,%r9
  402d6e:	49 89 d0             	mov    %rdx,%r8
  402d71:	48 89 f9             	mov    %rdi,%rcx
  402d74:	48 89 f2             	mov    %rsi,%rdx
  402d77:	be 9a 3c 00 00       	mov    $0x3c9a,%esi
  402d7c:	bf c5 35 40 00       	mov    $0x4035c5,%edi
  402d81:	e8 19 f6 ff ff       	call   40239f <submitr>
  402d86:	48 83 c4 10          	add    $0x10,%rsp
  402d8a:	eb 0e                	jmp    402d9a <driver_post+0x6f>
  402d8c:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  402d91:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  402d95:	b8 00 00 00 00       	mov    $0x0,%eax
  402d9a:	5b                   	pop    %rbx
  402d9b:	c3                   	ret    

0000000000402d9c <check>:
  402d9c:	89 f8                	mov    %edi,%eax
  402d9e:	c1 e8 1c             	shr    $0x1c,%eax
  402da1:	85 c0                	test   %eax,%eax
  402da3:	74 1d                	je     402dc2 <check+0x26>
  402da5:	b9 00 00 00 00       	mov    $0x0,%ecx
  402daa:	eb 0b                	jmp    402db7 <check+0x1b>
  402dac:	89 f8                	mov    %edi,%eax
  402dae:	d3 e8                	shr    %cl,%eax
  402db0:	3c 0a                	cmp    $0xa,%al
  402db2:	74 14                	je     402dc8 <check+0x2c>
  402db4:	83 c1 08             	add    $0x8,%ecx
  402db7:	83 f9 1f             	cmp    $0x1f,%ecx
  402dba:	7e f0                	jle    402dac <check+0x10>
  402dbc:	b8 01 00 00 00       	mov    $0x1,%eax
  402dc1:	c3                   	ret    
  402dc2:	b8 00 00 00 00       	mov    $0x0,%eax
  402dc7:	c3                   	ret    
  402dc8:	b8 00 00 00 00       	mov    $0x0,%eax
  402dcd:	c3                   	ret    

0000000000402dce <gencookie>:
  402dce:	53                   	push   %rbx
  402dcf:	83 c7 01             	add    $0x1,%edi
  402dd2:	e8 49 df ff ff       	call   400d20 <srandom@plt>
  402dd7:	e8 74 e0 ff ff       	call   400e50 <random@plt>
  402ddc:	89 c3                	mov    %eax,%ebx
  402dde:	89 c7                	mov    %eax,%edi
  402de0:	e8 b7 ff ff ff       	call   402d9c <check>
  402de5:	85 c0                	test   %eax,%eax
  402de7:	74 ee                	je     402dd7 <gencookie+0x9>
  402de9:	89 d8                	mov    %ebx,%eax
  402deb:	5b                   	pop    %rbx
  402dec:	c3                   	ret    
  402ded:	0f 1f 00             	nopl   (%rax)

0000000000402df0 <__libc_csu_init>:
  402df0:	41 57                	push   %r15
  402df2:	41 56                	push   %r14
  402df4:	41 89 ff             	mov    %edi,%r15d
  402df7:	41 55                	push   %r13
  402df9:	41 54                	push   %r12
  402dfb:	4c 8d 25 fe 0f 20 00 	lea    0x200ffe(%rip),%r12        # 603e00 <__frame_dummy_init_array_entry>
  402e02:	55                   	push   %rbp
  402e03:	48 8d 2d fe 0f 20 00 	lea    0x200ffe(%rip),%rbp        # 603e08 <__do_global_dtors_aux_fini_array_entry>
  402e0a:	53                   	push   %rbx
  402e0b:	49 89 f6             	mov    %rsi,%r14
  402e0e:	49 89 d5             	mov    %rdx,%r13
  402e11:	4c 29 e5             	sub    %r12,%rbp
  402e14:	48 83 ec 08          	sub    $0x8,%rsp
  402e18:	48 c1 fd 03          	sar    $0x3,%rbp
  402e1c:	e8 9f de ff ff       	call   400cc0 <_init>
  402e21:	48 85 ed             	test   %rbp,%rbp
  402e24:	74 20                	je     402e46 <__libc_csu_init+0x56>
  402e26:	31 db                	xor    %ebx,%ebx
  402e28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  402e2f:	00 
  402e30:	4c 89 ea             	mov    %r13,%rdx
  402e33:	4c 89 f6             	mov    %r14,%rsi
  402e36:	44 89 ff             	mov    %r15d,%edi
  402e39:	41 ff 14 dc          	call   *(%r12,%rbx,8)
  402e3d:	48 83 c3 01          	add    $0x1,%rbx
  402e41:	48 39 eb             	cmp    %rbp,%rbx
  402e44:	75 ea                	jne    402e30 <__libc_csu_init+0x40>
  402e46:	48 83 c4 08          	add    $0x8,%rsp
  402e4a:	5b                   	pop    %rbx
  402e4b:	5d                   	pop    %rbp
  402e4c:	41 5c                	pop    %r12
  402e4e:	41 5d                	pop    %r13
  402e50:	41 5e                	pop    %r14
  402e52:	41 5f                	pop    %r15
  402e54:	c3                   	ret    
  402e55:	90                   	nop
  402e56:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402e5d:	00 00 00 

0000000000402e60 <__libc_csu_fini>:
  402e60:	f3 c3                	repz ret 

Disassembly of section .fini:

0000000000402e64 <_fini>:
  402e64:	48 83 ec 08          	sub    $0x8,%rsp
  402e68:	48 83 c4 08          	add    $0x8,%rsp
  402e6c:	c3                   	ret    
