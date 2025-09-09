
rtarget:     file format elf64-x86-64


Disassembly of section .init:

0000000000400cc0 <_init>:
  400cc0:	48 83 ec 08          	sub    $0x8,%rsp
  400cc4:	48 8b 05 2d 43 20 00 	mov    0x20432d(%rip),%rax        # 604ff8 <__gmon_start__>
  400ccb:	48 85 c0             	test   %rax,%rax
  400cce:	74 05                	je     400cd5 <_init+0x15>
  400cd0:	e8 3b 02 00 00       	call   400f10 <__gmon_start__@plt>
  400cd5:	48 83 c4 08          	add    $0x8,%rsp
  400cd9:	c3                   	ret    

Disassembly of section .plt:

0000000000400ce0 <.plt>:
  400ce0:	ff 35 22 43 20 00    	push   0x204322(%rip)        # 605008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400ce6:	ff 25 24 43 20 00    	jmp    *0x204324(%rip)        # 605010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400cec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cf0 <__printf_chk@plt>:
  400cf0:	ff 25 22 43 20 00    	jmp    *0x204322(%rip)        # 605018 <__printf_chk>
  400cf6:	68 00 00 00 00       	push   $0x0
  400cfb:	e9 e0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d00 <strcasecmp@plt>:
  400d00:	ff 25 1a 43 20 00    	jmp    *0x20431a(%rip)        # 605020 <strcasecmp@GLIBC_2.2.5>
  400d06:	68 01 00 00 00       	push   $0x1
  400d0b:	e9 d0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d10 <__errno_location@plt>:
  400d10:	ff 25 12 43 20 00    	jmp    *0x204312(%rip)        # 605028 <__errno_location@GLIBC_2.2.5>
  400d16:	68 02 00 00 00       	push   $0x2
  400d1b:	e9 c0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d20 <srandom@plt>:
  400d20:	ff 25 0a 43 20 00    	jmp    *0x20430a(%rip)        # 605030 <srandom@GLIBC_2.2.5>
  400d26:	68 03 00 00 00       	push   $0x3
  400d2b:	e9 b0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d30 <strncmp@plt>:
  400d30:	ff 25 02 43 20 00    	jmp    *0x204302(%rip)        # 605038 <strncmp@GLIBC_2.2.5>
  400d36:	68 04 00 00 00       	push   $0x4
  400d3b:	e9 a0 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d40 <strcpy@plt>:
  400d40:	ff 25 fa 42 20 00    	jmp    *0x2042fa(%rip)        # 605040 <strcpy@GLIBC_2.2.5>
  400d46:	68 05 00 00 00       	push   $0x5
  400d4b:	e9 90 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d50 <puts@plt>:
  400d50:	ff 25 f2 42 20 00    	jmp    *0x2042f2(%rip)        # 605048 <puts@GLIBC_2.2.5>
  400d56:	68 06 00 00 00       	push   $0x6
  400d5b:	e9 80 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d60 <write@plt>:
  400d60:	ff 25 ea 42 20 00    	jmp    *0x2042ea(%rip)        # 605050 <write@GLIBC_2.2.5>
  400d66:	68 07 00 00 00       	push   $0x7
  400d6b:	e9 70 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d70 <__stack_chk_fail@plt>:
  400d70:	ff 25 e2 42 20 00    	jmp    *0x2042e2(%rip)        # 605058 <__stack_chk_fail@GLIBC_2.4>
  400d76:	68 08 00 00 00       	push   $0x8
  400d7b:	e9 60 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d80 <mmap@plt>:
  400d80:	ff 25 da 42 20 00    	jmp    *0x2042da(%rip)        # 605060 <mmap@GLIBC_2.2.5>
  400d86:	68 09 00 00 00       	push   $0x9
  400d8b:	e9 50 ff ff ff       	jmp    400ce0 <.plt>

0000000000400d90 <memset@plt>:
  400d90:	ff 25 d2 42 20 00    	jmp    *0x2042d2(%rip)        # 605068 <memset@GLIBC_2.2.5>
  400d96:	68 0a 00 00 00       	push   $0xa
  400d9b:	e9 40 ff ff ff       	jmp    400ce0 <.plt>

0000000000400da0 <alarm@plt>:
  400da0:	ff 25 ca 42 20 00    	jmp    *0x2042ca(%rip)        # 605070 <alarm@GLIBC_2.2.5>
  400da6:	68 0b 00 00 00       	push   $0xb
  400dab:	e9 30 ff ff ff       	jmp    400ce0 <.plt>

0000000000400db0 <close@plt>:
  400db0:	ff 25 c2 42 20 00    	jmp    *0x2042c2(%rip)        # 605078 <close@GLIBC_2.2.5>
  400db6:	68 0c 00 00 00       	push   $0xc
  400dbb:	e9 20 ff ff ff       	jmp    400ce0 <.plt>

0000000000400dc0 <read@plt>:
  400dc0:	ff 25 ba 42 20 00    	jmp    *0x2042ba(%rip)        # 605080 <read@GLIBC_2.2.5>
  400dc6:	68 0d 00 00 00       	push   $0xd
  400dcb:	e9 10 ff ff ff       	jmp    400ce0 <.plt>

0000000000400dd0 <__libc_start_main@plt>:
  400dd0:	ff 25 b2 42 20 00    	jmp    *0x2042b2(%rip)        # 605088 <__libc_start_main@GLIBC_2.2.5>
  400dd6:	68 0e 00 00 00       	push   $0xe
  400ddb:	e9 00 ff ff ff       	jmp    400ce0 <.plt>

0000000000400de0 <signal@plt>:
  400de0:	ff 25 aa 42 20 00    	jmp    *0x2042aa(%rip)        # 605090 <signal@GLIBC_2.2.5>
  400de6:	68 0f 00 00 00       	push   $0xf
  400deb:	e9 f0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400df0 <gethostbyname@plt>:
  400df0:	ff 25 a2 42 20 00    	jmp    *0x2042a2(%rip)        # 605098 <gethostbyname@GLIBC_2.2.5>
  400df6:	68 10 00 00 00       	push   $0x10
  400dfb:	e9 e0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e00 <__memmove_chk@plt>:
  400e00:	ff 25 9a 42 20 00    	jmp    *0x20429a(%rip)        # 6050a0 <__memmove_chk@GLIBC_2.3.4>
  400e06:	68 11 00 00 00       	push   $0x11
  400e0b:	e9 d0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e10 <strtol@plt>:
  400e10:	ff 25 92 42 20 00    	jmp    *0x204292(%rip)        # 6050a8 <strtol@GLIBC_2.2.5>
  400e16:	68 12 00 00 00       	push   $0x12
  400e1b:	e9 c0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e20 <memcpy@plt>:
  400e20:	ff 25 8a 42 20 00    	jmp    *0x20428a(%rip)        # 6050b0 <memcpy@GLIBC_2.14>
  400e26:	68 13 00 00 00       	push   $0x13
  400e2b:	e9 b0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e30 <__sprintf_chk@plt>:
  400e30:	ff 25 82 42 20 00    	jmp    *0x204282(%rip)        # 6050b8 <__sprintf_chk>
  400e36:	68 14 00 00 00       	push   $0x14
  400e3b:	e9 a0 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e40 <time@plt>:
  400e40:	ff 25 7a 42 20 00    	jmp    *0x20427a(%rip)        # 6050c0 <time@GLIBC_2.2.5>
  400e46:	68 15 00 00 00       	push   $0x15
  400e4b:	e9 90 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e50 <random@plt>:
  400e50:	ff 25 72 42 20 00    	jmp    *0x204272(%rip)        # 6050c8 <random@GLIBC_2.2.5>
  400e56:	68 16 00 00 00       	push   $0x16
  400e5b:	e9 80 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e60 <_IO_getc@plt>:
  400e60:	ff 25 6a 42 20 00    	jmp    *0x20426a(%rip)        # 6050d0 <_IO_getc@GLIBC_2.2.5>
  400e66:	68 17 00 00 00       	push   $0x17
  400e6b:	e9 70 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e70 <__isoc99_sscanf@plt>:
  400e70:	ff 25 62 42 20 00    	jmp    *0x204262(%rip)        # 6050d8 <__isoc99_sscanf@GLIBC_2.7>
  400e76:	68 18 00 00 00       	push   $0x18
  400e7b:	e9 60 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e80 <munmap@plt>:
  400e80:	ff 25 5a 42 20 00    	jmp    *0x20425a(%rip)        # 6050e0 <munmap@GLIBC_2.2.5>
  400e86:	68 19 00 00 00       	push   $0x19
  400e8b:	e9 50 fe ff ff       	jmp    400ce0 <.plt>

0000000000400e90 <fopen@plt>:
  400e90:	ff 25 52 42 20 00    	jmp    *0x204252(%rip)        # 6050e8 <fopen@GLIBC_2.2.5>
  400e96:	68 1a 00 00 00       	push   $0x1a
  400e9b:	e9 40 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ea0 <getopt@plt>:
  400ea0:	ff 25 4a 42 20 00    	jmp    *0x20424a(%rip)        # 6050f0 <getopt@GLIBC_2.2.5>
  400ea6:	68 1b 00 00 00       	push   $0x1b
  400eab:	e9 30 fe ff ff       	jmp    400ce0 <.plt>

0000000000400eb0 <strtoul@plt>:
  400eb0:	ff 25 42 42 20 00    	jmp    *0x204242(%rip)        # 6050f8 <strtoul@GLIBC_2.2.5>
  400eb6:	68 1c 00 00 00       	push   $0x1c
  400ebb:	e9 20 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ec0 <gethostname@plt>:
  400ec0:	ff 25 3a 42 20 00    	jmp    *0x20423a(%rip)        # 605100 <gethostname@GLIBC_2.2.5>
  400ec6:	68 1d 00 00 00       	push   $0x1d
  400ecb:	e9 10 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ed0 <exit@plt>:
  400ed0:	ff 25 32 42 20 00    	jmp    *0x204232(%rip)        # 605108 <exit@GLIBC_2.2.5>
  400ed6:	68 1e 00 00 00       	push   $0x1e
  400edb:	e9 00 fe ff ff       	jmp    400ce0 <.plt>

0000000000400ee0 <connect@plt>:
  400ee0:	ff 25 2a 42 20 00    	jmp    *0x20422a(%rip)        # 605110 <connect@GLIBC_2.2.5>
  400ee6:	68 1f 00 00 00       	push   $0x1f
  400eeb:	e9 f0 fd ff ff       	jmp    400ce0 <.plt>

0000000000400ef0 <__fprintf_chk@plt>:
  400ef0:	ff 25 22 42 20 00    	jmp    *0x204222(%rip)        # 605118 <__fprintf_chk@GLIBC_2.3.4>
  400ef6:	68 20 00 00 00       	push   $0x20
  400efb:	e9 e0 fd ff ff       	jmp    400ce0 <.plt>

0000000000400f00 <socket@plt>:
  400f00:	ff 25 1a 42 20 00    	jmp    *0x20421a(%rip)        # 605120 <socket@GLIBC_2.2.5>
  400f06:	68 21 00 00 00       	push   $0x21
  400f0b:	e9 d0 fd ff ff       	jmp    400ce0 <.plt>

Disassembly of section .plt.got:

0000000000400f10 <__gmon_start__@plt>:
  400f10:	ff 25 e2 40 20 00    	jmp    *0x2040e2(%rip)        # 604ff8 <__gmon_start__>
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
  400f2f:	49 c7 c0 90 2f 40 00 	mov    $0x402f90,%r8
  400f36:	48 c7 c1 20 2f 40 00 	mov    $0x402f20,%rcx
  400f3d:	48 c7 c7 25 12 40 00 	mov    $0x401225,%rdi
  400f44:	e8 87 fe ff ff       	call   400dd0 <__libc_start_main@plt>
  400f49:	f4                   	hlt    
  400f4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400f50 <deregister_tm_clones>:
  400f50:	b8 b7 54 60 00       	mov    $0x6054b7,%eax
  400f55:	55                   	push   %rbp
  400f56:	48 2d b0 54 60 00    	sub    $0x6054b0,%rax
  400f5c:	48 83 f8 0e          	cmp    $0xe,%rax
  400f60:	48 89 e5             	mov    %rsp,%rbp
  400f63:	76 1b                	jbe    400f80 <deregister_tm_clones+0x30>
  400f65:	b8 00 00 00 00       	mov    $0x0,%eax
  400f6a:	48 85 c0             	test   %rax,%rax
  400f6d:	74 11                	je     400f80 <deregister_tm_clones+0x30>
  400f6f:	5d                   	pop    %rbp
  400f70:	bf b0 54 60 00       	mov    $0x6054b0,%edi
  400f75:	ff e0                	jmp    *%rax
  400f77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f7e:	00 00 
  400f80:	5d                   	pop    %rbp
  400f81:	c3                   	ret    
  400f82:	0f 1f 40 00          	nopl   0x0(%rax)
  400f86:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  400f8d:	00 00 00 

0000000000400f90 <register_tm_clones>:
  400f90:	be b0 54 60 00       	mov    $0x6054b0,%esi
  400f95:	55                   	push   %rbp
  400f96:	48 81 ee b0 54 60 00 	sub    $0x6054b0,%rsi
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
  400fbe:	bf b0 54 60 00       	mov    $0x6054b0,%edi
  400fc3:	ff e0                	jmp    *%rax
  400fc5:	0f 1f 00             	nopl   (%rax)
  400fc8:	5d                   	pop    %rbp
  400fc9:	c3                   	ret    
  400fca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400fd0 <__do_global_dtors_aux>:
  400fd0:	80 3d 11 45 20 00 00 	cmpb   $0x0,0x204511(%rip)        # 6054e8 <completed.7594>
  400fd7:	75 11                	jne    400fea <__do_global_dtors_aux+0x1a>
  400fd9:	55                   	push   %rbp
  400fda:	48 89 e5             	mov    %rsp,%rbp
  400fdd:	e8 6e ff ff ff       	call   400f50 <deregister_tm_clones>
  400fe2:	5d                   	pop    %rbp
  400fe3:	c6 05 fe 44 20 00 01 	movb   $0x1,0x2044fe(%rip)        # 6054e8 <completed.7594>
  400fea:	f3 c3                	repz ret 
  400fec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ff0 <frame_dummy>:
  400ff0:	bf 10 4e 60 00       	mov    $0x604e10,%edi
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
  40101d:	83 3d 08 45 20 00 00 	cmpl   $0x0,0x204508(%rip)        # 60552c <is_checker>
  401024:	74 3e                	je     401064 <usage+0x4e>
  401026:	be a8 2f 40 00       	mov    $0x402fa8,%esi
  40102b:	bf 01 00 00 00       	mov    $0x1,%edi
  401030:	b8 00 00 00 00       	mov    $0x0,%eax
  401035:	e8 b6 fc ff ff       	call   400cf0 <__printf_chk@plt>
  40103a:	bf e0 2f 40 00       	mov    $0x402fe0,%edi
  40103f:	e8 0c fd ff ff       	call   400d50 <puts@plt>
  401044:	bf 58 31 40 00       	mov    $0x403158,%edi
  401049:	e8 02 fd ff ff       	call   400d50 <puts@plt>
  40104e:	bf 08 30 40 00       	mov    $0x403008,%edi
  401053:	e8 f8 fc ff ff       	call   400d50 <puts@plt>
  401058:	bf 72 31 40 00       	mov    $0x403172,%edi
  40105d:	e8 ee fc ff ff       	call   400d50 <puts@plt>
  401062:	eb 32                	jmp    401096 <usage+0x80>
  401064:	be 8e 31 40 00       	mov    $0x40318e,%esi
  401069:	bf 01 00 00 00       	mov    $0x1,%edi
  40106e:	b8 00 00 00 00       	mov    $0x0,%eax
  401073:	e8 78 fc ff ff       	call   400cf0 <__printf_chk@plt>
  401078:	bf 30 30 40 00       	mov    $0x403030,%edi
  40107d:	e8 ce fc ff ff       	call   400d50 <puts@plt>
  401082:	bf 58 30 40 00       	mov    $0x403058,%edi
  401087:	e8 c4 fc ff ff       	call   400d50 <puts@plt>
  40108c:	bf ac 31 40 00       	mov    $0x4031ac,%edi
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
  4010be:	89 3d 58 44 20 00    	mov    %edi,0x204458(%rip)        # 60551c <check_level>
  4010c4:	8b 3d 9e 40 20 00    	mov    0x20409e(%rip),%edi        # 605168 <target_id>
  4010ca:	e8 26 1e 00 00       	call   402ef5 <gencookie>
  4010cf:	89 05 53 44 20 00    	mov    %eax,0x204453(%rip)        # 605528 <cookie>
  4010d5:	89 c7                	mov    %eax,%edi
  4010d7:	e8 19 1e 00 00       	call   402ef5 <gencookie>
  4010dc:	89 05 42 44 20 00    	mov    %eax,0x204442(%rip)        # 605524 <authkey>
  4010e2:	8b 05 80 40 20 00    	mov    0x204080(%rip),%eax        # 605168 <target_id>
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
  40112d:	48 89 05 74 43 20 00 	mov    %rax,0x204374(%rip)        # 6054a8 <buf_offset>
  401134:	c6 05 15 50 20 00 72 	movb   $0x72,0x205015(%rip)        # 606150 <target_prefix>
  40113b:	83 3d d6 43 20 00 00 	cmpl   $0x0,0x2043d6(%rip)        # 605518 <notify>
  401142:	0f 84 bb 00 00 00    	je     401203 <initialize_target+0x163>
  401148:	83 3d dd 43 20 00 00 	cmpl   $0x0,0x2043dd(%rip)        # 60552c <is_checker>
  40114f:	0f 85 ae 00 00 00    	jne    401203 <initialize_target+0x163>
  401155:	be 00 01 00 00       	mov    $0x100,%esi
  40115a:	48 89 e7             	mov    %rsp,%rdi
  40115d:	e8 5e fd ff ff       	call   400ec0 <gethostname@plt>
  401162:	85 c0                	test   %eax,%eax
  401164:	74 25                	je     40118b <initialize_target+0xeb>
  401166:	bf 88 30 40 00       	mov    $0x403088,%edi
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
  401193:	48 8b 3c c5 80 51 60 	mov    0x605180(,%rax,8),%rdi
  40119a:	00 
  40119b:	48 85 ff             	test   %rdi,%rdi
  40119e:	75 da                	jne    40117a <initialize_target+0xda>
  4011a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4011a5:	eb 05                	jmp    4011ac <initialize_target+0x10c>
  4011a7:	b8 01 00 00 00       	mov    $0x1,%eax
  4011ac:	85 c0                	test   %eax,%eax
  4011ae:	75 1c                	jne    4011cc <initialize_target+0x12c>
  4011b0:	48 89 e2             	mov    %rsp,%rdx
  4011b3:	be c0 30 40 00       	mov    $0x4030c0,%esi
  4011b8:	bf 01 00 00 00       	mov    $0x1,%edi
  4011bd:	e8 2e fb ff ff       	call   400cf0 <__printf_chk@plt>
  4011c2:	bf 08 00 00 00       	mov    $0x8,%edi
  4011c7:	e8 04 fd ff ff       	call   400ed0 <exit@plt>
  4011cc:	48 8d bc 24 00 01 00 	lea    0x100(%rsp),%rdi
  4011d3:	00 
  4011d4:	e8 86 1a 00 00       	call   402c5f <init_driver>
  4011d9:	85 c0                	test   %eax,%eax
  4011db:	79 26                	jns    401203 <initialize_target+0x163>
  4011dd:	48 8d 94 24 00 01 00 	lea    0x100(%rsp),%rdx
  4011e4:	00 
  4011e5:	be 00 31 40 00       	mov    $0x403100,%esi
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
  401233:	be 9a 1f 40 00       	mov    $0x401f9a,%esi
  401238:	bf 0b 00 00 00       	mov    $0xb,%edi
  40123d:	e8 9e fb ff ff       	call   400de0 <signal@plt>
  401242:	be 4c 1f 40 00       	mov    $0x401f4c,%esi
  401247:	bf 07 00 00 00       	mov    $0x7,%edi
  40124c:	e8 8f fb ff ff       	call   400de0 <signal@plt>
  401251:	be e8 1f 40 00       	mov    $0x401fe8,%esi
  401256:	bf 04 00 00 00       	mov    $0x4,%edi
  40125b:	e8 80 fb ff ff       	call   400de0 <signal@plt>
  401260:	83 3d c5 42 20 00 00 	cmpl   $0x0,0x2042c5(%rip)        # 60552c <is_checker>
  401267:	74 20                	je     401289 <main+0x64>
  401269:	be 36 20 40 00       	mov    $0x402036,%esi
  40126e:	bf 0e 00 00 00       	mov    $0xe,%edi
  401273:	e8 68 fb ff ff       	call   400de0 <signal@plt>
  401278:	bf 05 00 00 00       	mov    $0x5,%edi
  40127d:	e8 1e fb ff ff       	call   400da0 <alarm@plt>
  401282:	bd ca 31 40 00       	mov    $0x4031ca,%ebp
  401287:	eb 05                	jmp    40128e <main+0x69>
  401289:	bd c5 31 40 00       	mov    $0x4031c5,%ebp
  40128e:	48 8b 05 2b 42 20 00 	mov    0x20422b(%rip),%rax        # 6054c0 <stdin@GLIBC_2.2.5>
  401295:	48 89 05 74 42 20 00 	mov    %rax,0x204274(%rip)        # 605510 <infile>
  40129c:	41 bd 00 00 00 00    	mov    $0x0,%r13d
  4012a2:	41 be 00 00 00 00    	mov    $0x0,%r14d
  4012a8:	e9 c6 00 00 00       	jmp    401373 <main+0x14e>
  4012ad:	83 e8 61             	sub    $0x61,%eax
  4012b0:	3c 10                	cmp    $0x10,%al
  4012b2:	0f 87 9c 00 00 00    	ja     401354 <main+0x12f>
  4012b8:	0f b6 c0             	movzbl %al,%eax
  4012bb:	ff 24 c5 10 32 40 00 	jmp    *0x403210(,%rax,8)
  4012c2:	48 8b 3b             	mov    (%rbx),%rdi
  4012c5:	e8 4c fd ff ff       	call   401016 <usage>
  4012ca:	be 52 34 40 00       	mov    $0x403452,%esi
  4012cf:	48 8b 3d f2 41 20 00 	mov    0x2041f2(%rip),%rdi        # 6054c8 <optarg@GLIBC_2.2.5>
  4012d6:	e8 b5 fb ff ff       	call   400e90 <fopen@plt>
  4012db:	48 89 05 2e 42 20 00 	mov    %rax,0x20422e(%rip)        # 605510 <infile>
  4012e2:	48 85 c0             	test   %rax,%rax
  4012e5:	0f 85 88 00 00 00    	jne    401373 <main+0x14e>
  4012eb:	48 8b 0d d6 41 20 00 	mov    0x2041d6(%rip),%rcx        # 6054c8 <optarg@GLIBC_2.2.5>
  4012f2:	ba d2 31 40 00       	mov    $0x4031d2,%edx
  4012f7:	be 01 00 00 00       	mov    $0x1,%esi
  4012fc:	48 8b 3d dd 41 20 00 	mov    0x2041dd(%rip),%rdi        # 6054e0 <stderr@GLIBC_2.2.5>
  401303:	e8 e8 fb ff ff       	call   400ef0 <__fprintf_chk@plt>
  401308:	b8 01 00 00 00       	mov    $0x1,%eax
  40130d:	e9 e4 00 00 00       	jmp    4013f6 <main+0x1d1>
  401312:	ba 10 00 00 00       	mov    $0x10,%edx
  401317:	be 00 00 00 00       	mov    $0x0,%esi
  40131c:	48 8b 3d a5 41 20 00 	mov    0x2041a5(%rip),%rdi        # 6054c8 <optarg@GLIBC_2.2.5>
  401323:	e8 88 fb ff ff       	call   400eb0 <strtoul@plt>
  401328:	41 89 c6             	mov    %eax,%r14d
  40132b:	eb 46                	jmp    401373 <main+0x14e>
  40132d:	ba 0a 00 00 00       	mov    $0xa,%edx
  401332:	be 00 00 00 00       	mov    $0x0,%esi
  401337:	48 8b 3d 8a 41 20 00 	mov    0x20418a(%rip),%rdi        # 6054c8 <optarg@GLIBC_2.2.5>
  40133e:	e8 cd fa ff ff       	call   400e10 <strtol@plt>
  401343:	41 89 c5             	mov    %eax,%r13d
  401346:	eb 2b                	jmp    401373 <main+0x14e>
  401348:	c7 05 c6 41 20 00 00 	movl   $0x0,0x2041c6(%rip)        # 605518 <notify>
  40134f:	00 00 00 
  401352:	eb 1f                	jmp    401373 <main+0x14e>
  401354:	0f be d2             	movsbl %dl,%edx
  401357:	be ef 31 40 00       	mov    $0x4031ef,%esi
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
  40138b:	be 01 00 00 00       	mov    $0x1,%esi
  401390:	44 89 ef             	mov    %r13d,%edi
  401393:	e8 08 fd ff ff       	call   4010a0 <initialize_target>
  401398:	83 3d 8d 41 20 00 00 	cmpl   $0x0,0x20418d(%rip)        # 60552c <is_checker>
  40139f:	74 2a                	je     4013cb <main+0x1a6>
  4013a1:	44 3b 35 7c 41 20 00 	cmp    0x20417c(%rip),%r14d        # 605524 <authkey>
  4013a8:	74 21                	je     4013cb <main+0x1a6>
  4013aa:	44 89 f2             	mov    %r14d,%edx
  4013ad:	be 28 31 40 00       	mov    $0x403128,%esi
  4013b2:	bf 01 00 00 00       	mov    $0x1,%edi
  4013b7:	b8 00 00 00 00       	mov    $0x0,%eax
  4013bc:	e8 2f f9 ff ff       	call   400cf0 <__printf_chk@plt>
  4013c1:	b8 00 00 00 00       	mov    $0x0,%eax
  4013c6:	e8 22 09 00 00       	call   401ced <check_fail>
  4013cb:	8b 15 57 41 20 00    	mov    0x204157(%rip),%edx        # 605528 <cookie>
  4013d1:	be 02 32 40 00       	mov    $0x403202,%esi
  4013d6:	bf 01 00 00 00       	mov    $0x1,%edi
  4013db:	b8 00 00 00 00       	mov    $0x0,%eax
  4013e0:	e8 0b f9 ff ff       	call   400cf0 <__printf_chk@plt>
  4013e5:	48 8b 3d bc 40 20 00 	mov    0x2040bc(%rip),%rdi        # 6054a8 <buf_offset>
  4013ec:	e8 98 0c 00 00       	call   402089 <launch>
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
  401978:	e8 a5 03 00 00       	call   401d22 <Gets>
  40197d:	b8 01 00 00 00       	mov    $0x1,%eax
  401982:	48 83 c4 28          	add    $0x28,%rsp
  401986:	c3                   	ret    

0000000000401987 <touch1>:
  401987:	48 83 ec 08          	sub    $0x8,%rsp
  40198b:	c7 05 8b 3b 20 00 01 	movl   $0x1,0x203b8b(%rip)        # 605520 <vlevel>
  401992:	00 00 00 
  401995:	bf f2 32 40 00       	mov    $0x4032f2,%edi
  40199a:	e8 b1 f3 ff ff       	call   400d50 <puts@plt>
  40199f:	bf 01 00 00 00       	mov    $0x1,%edi
  4019a4:	e8 b9 04 00 00       	call   401e62 <validate>
  4019a9:	bf 00 00 00 00       	mov    $0x0,%edi
  4019ae:	e8 1d f5 ff ff       	call   400ed0 <exit@plt>

00000000004019b3 <touch2>:
  4019b3:	48 83 ec 08          	sub    $0x8,%rsp
  4019b7:	89 fa                	mov    %edi,%edx
  4019b9:	c7 05 5d 3b 20 00 02 	movl   $0x2,0x203b5d(%rip)        # 605520 <vlevel>
  4019c0:	00 00 00 
  4019c3:	39 3d 5f 3b 20 00    	cmp    %edi,0x203b5f(%rip)        # 605528 <cookie>
  4019c9:	75 20                	jne    4019eb <touch2+0x38>
  4019cb:	be 18 33 40 00       	mov    $0x403318,%esi
  4019d0:	bf 01 00 00 00       	mov    $0x1,%edi
  4019d5:	b8 00 00 00 00       	mov    $0x0,%eax
  4019da:	e8 11 f3 ff ff       	call   400cf0 <__printf_chk@plt>
  4019df:	bf 02 00 00 00       	mov    $0x2,%edi
  4019e4:	e8 79 04 00 00       	call   401e62 <validate>
  4019e9:	eb 1e                	jmp    401a09 <touch2+0x56>
  4019eb:	be 40 33 40 00       	mov    $0x403340,%esi
  4019f0:	bf 01 00 00 00       	mov    $0x1,%edi
  4019f5:	b8 00 00 00 00       	mov    $0x0,%eax
  4019fa:	e8 f1 f2 ff ff       	call   400cf0 <__printf_chk@plt>
  4019ff:	bf 02 00 00 00       	mov    $0x2,%edi
  401a04:	e8 1b 05 00 00       	call   401f24 <fail>
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
  401a70:	b9 0f 33 40 00       	mov    $0x40330f,%ecx
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
  401ac8:	c7 05 4e 3a 20 00 03 	movl   $0x3,0x203a4e(%rip)        # 605520 <vlevel>
  401acf:	00 00 00 
  401ad2:	48 89 fe             	mov    %rdi,%rsi
  401ad5:	8b 3d 4d 3a 20 00    	mov    0x203a4d(%rip),%edi        # 605528 <cookie>
  401adb:	e8 33 ff ff ff       	call   401a13 <hexmatch>
  401ae0:	85 c0                	test   %eax,%eax
  401ae2:	74 23                	je     401b07 <touch3+0x43>
  401ae4:	48 89 da             	mov    %rbx,%rdx
  401ae7:	be 68 33 40 00       	mov    $0x403368,%esi
  401aec:	bf 01 00 00 00       	mov    $0x1,%edi
  401af1:	b8 00 00 00 00       	mov    $0x0,%eax
  401af6:	e8 f5 f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401afb:	bf 03 00 00 00       	mov    $0x3,%edi
  401b00:	e8 5d 03 00 00       	call   401e62 <validate>
  401b05:	eb 21                	jmp    401b28 <touch3+0x64>
  401b07:	48 89 da             	mov    %rbx,%rdx
  401b0a:	be 90 33 40 00       	mov    $0x403390,%esi
  401b0f:	bf 01 00 00 00       	mov    $0x1,%edi
  401b14:	b8 00 00 00 00       	mov    $0x0,%eax
  401b19:	e8 d2 f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401b1e:	bf 03 00 00 00       	mov    $0x3,%edi
  401b23:	e8 fc 03 00 00       	call   401f24 <fail>
  401b28:	bf 00 00 00 00       	mov    $0x0,%edi
  401b2d:	e8 9e f3 ff ff       	call   400ed0 <exit@plt>

0000000000401b32 <test>:
  401b32:	48 83 ec 08          	sub    $0x8,%rsp
  401b36:	b8 00 00 00 00       	mov    $0x0,%eax
  401b3b:	e8 31 fe ff ff       	call   401971 <getbuf>
  401b40:	89 c2                	mov    %eax,%edx
  401b42:	be b8 33 40 00       	mov    $0x4033b8,%esi
  401b47:	bf 01 00 00 00       	mov    $0x1,%edi
  401b4c:	b8 00 00 00 00       	mov    $0x0,%eax
  401b51:	e8 9a f1 ff ff       	call   400cf0 <__printf_chk@plt>
  401b56:	48 83 c4 08          	add    $0x8,%rsp
  401b5a:	c3                   	ret    

0000000000401b5b <start_farm>:
  401b5b:	b8 01 00 00 00       	mov    $0x1,%eax
  401b60:	c3                   	ret    

0000000000401b61 <getval_307>:
  401b61:	b8 48 89 c7 91       	mov    $0x91c78948,%eax
  401b66:	c3                   	ret    

0000000000401b67 <addval_470>:
  401b67:	8d 87 48 89 c7 90    	lea    -0x6f3876b8(%rdi),%eax
  401b6d:	c3                   	ret    

0000000000401b6e <setval_391>:
  401b6e:	c7 07 48 89 c7 c3    	movl   $0xc3c78948,(%rdi)
  401b74:	c3                   	ret    

0000000000401b75 <setval_243>:
  401b75:	c7 07 a3 58 90 c3    	movl   $0xc39058a3,(%rdi)
  401b7b:	c3                   	ret    

0000000000401b7c <addval_268>:
  401b7c:	8d 87 d8 c3 40 0a    	lea    0xa40c3d8(%rdi),%eax
  401b82:	c3                   	ret    

0000000000401b83 <getval_443>:
  401b83:	b8 2c 25 58 94       	mov    $0x9458252c,%eax
  401b88:	c3                   	ret    

0000000000401b89 <setval_122>:
  401b89:	c7 07 58 90 90 c3    	movl   $0xc3909058,(%rdi)
  401b8f:	c3                   	ret    

0000000000401b90 <setval_489>:
  401b90:	c7 07 48 89 c7 92    	movl   $0x92c78948,(%rdi)
  401b96:	c3                   	ret    

0000000000401b97 <mid_farm>:
  401b97:	b8 01 00 00 00       	mov    $0x1,%eax
  401b9c:	c3                   	ret    

0000000000401b9d <add_xy>:
  401b9d:	48 8d 04 37          	lea    (%rdi,%rsi,1),%rax
  401ba1:	c3                   	ret    

0000000000401ba2 <addval_298>:
  401ba2:	8d 87 89 ca 18 d2    	lea    -0x2de73577(%rdi),%eax
  401ba8:	c3                   	ret    

0000000000401ba9 <getval_247>:
  401ba9:	b8 99 c1 38 db       	mov    $0xdb38c199,%eax
  401bae:	c3                   	ret    

0000000000401baf <setval_310>:
  401baf:	c7 07 88 ca 08 db    	movl   $0xdb08ca88,(%rdi)
  401bb5:	c3                   	ret    

0000000000401bb6 <setval_417>:
  401bb6:	c7 07 89 ca 20 d2    	movl   $0xd220ca89,(%rdi)
  401bbc:	c3                   	ret    

0000000000401bbd <getval_137>:
  401bbd:	b8 99 d6 90 90       	mov    $0x9090d699,%eax
  401bc2:	c3                   	ret    

0000000000401bc3 <addval_198>:
  401bc3:	8d 87 88 c1 38 d2    	lea    -0x2dc73e78(%rdi),%eax
  401bc9:	c3                   	ret    

0000000000401bca <addval_257>:
  401bca:	8d 87 89 c1 78 db    	lea    -0x24873e77(%rdi),%eax
  401bd0:	c3                   	ret    

0000000000401bd1 <addval_406>:
  401bd1:	8d 87 48 89 e0 c3    	lea    -0x3c1f76b8(%rdi),%eax
  401bd7:	c3                   	ret    

0000000000401bd8 <setval_337>:
  401bd8:	c7 07 89 ca 38 d2    	movl   $0xd238ca89,(%rdi)
  401bde:	c3                   	ret    

0000000000401bdf <addval_194>:
  401bdf:	8d 87 89 d6 94 c9    	lea    -0x366b2977(%rdi),%eax
  401be5:	c3                   	ret    

0000000000401be6 <getval_467>:
  401be6:	b8 09 ca 84 db       	mov    $0xdb84ca09,%eax
  401beb:	c3                   	ret    

0000000000401bec <addval_347>:
  401bec:	8d 87 48 89 e0 c1    	lea    -0x3e1f76b8(%rdi),%eax
  401bf2:	c3                   	ret    

0000000000401bf3 <setval_476>:
  401bf3:	c7 07 48 89 e0 92    	movl   $0x92e08948,(%rdi)
  401bf9:	c3                   	ret    

0000000000401bfa <setval_321>:
  401bfa:	c7 07 f0 89 d6 94    	movl   $0x94d689f0,(%rdi)
  401c00:	c3                   	ret    

0000000000401c01 <setval_485>:
  401c01:	c7 07 89 c1 94 d2    	movl   $0xd294c189,(%rdi)
  401c07:	c3                   	ret    

0000000000401c08 <getval_378>:
  401c08:	b8 89 d6 38 c0       	mov    $0xc038d689,%eax
  401c0d:	c3                   	ret    

0000000000401c0e <addval_460>:
  401c0e:	8d 87 89 c1 08 d2    	lea    -0x2df73e77(%rdi),%eax
  401c14:	c3                   	ret    

0000000000401c15 <addval_190>:
  401c15:	8d 87 89 d6 91 c3    	lea    -0x3c6e2977(%rdi),%eax
  401c1b:	c3                   	ret    

0000000000401c1c <addval_377>:
  401c1c:	8d 87 c8 89 e0 c3    	lea    -0x3c1f7638(%rdi),%eax
  401c22:	c3                   	ret    

0000000000401c23 <setval_176>:
  401c23:	c7 07 89 c1 94 90    	movl   $0x9094c189,(%rdi)
  401c29:	c3                   	ret    

0000000000401c2a <setval_465>:
  401c2a:	c7 07 89 c1 90 90    	movl   $0x9090c189,(%rdi)
  401c30:	c3                   	ret    

0000000000401c31 <addval_118>:
  401c31:	8d 87 89 ca 28 c9    	lea    -0x36d73577(%rdi),%eax
  401c37:	c3                   	ret    

0000000000401c38 <addval_357>:
  401c38:	8d 87 89 d6 90 90    	lea    -0x6f6f2977(%rdi),%eax
  401c3e:	c3                   	ret    

0000000000401c3f <addval_365>:
  401c3f:	8d 87 8b ca 90 c3    	lea    -0x3c6f3575(%rdi),%eax
  401c45:	c3                   	ret    

0000000000401c46 <setval_383>:
  401c46:	c7 07 48 99 e0 c3    	movl   $0xc3e09948,(%rdi)
  401c4c:	c3                   	ret    

0000000000401c4d <getval_430>:
  401c4d:	b8 c7 49 89 e0       	mov    $0xe08949c7,%eax
  401c52:	c3                   	ret    

0000000000401c53 <setval_140>:
  401c53:	c7 07 89 d6 90 c1    	movl   $0xc190d689,(%rdi)
  401c59:	c3                   	ret    

0000000000401c5a <getval_492>:
  401c5a:	b8 89 c1 28 c0       	mov    $0xc028c189,%eax
  401c5f:	c3                   	ret    

0000000000401c60 <setval_359>:
  401c60:	c7 07 48 89 e0 c3    	movl   $0xc3e08948,(%rdi)
  401c66:	c3                   	ret    

0000000000401c67 <setval_154>:
  401c67:	c7 07 81 d6 90 c3    	movl   $0xc390d681,(%rdi)
  401c6d:	c3                   	ret    

0000000000401c6e <setval_336>:
  401c6e:	c7 07 48 8d e0 c3    	movl   $0xc3e08d48,(%rdi)
  401c74:	c3                   	ret    

0000000000401c75 <addval_481>:
  401c75:	8d 87 a9 ca 38 db    	lea    -0x24c73557(%rdi),%eax
  401c7b:	c3                   	ret    

0000000000401c7c <end_farm>:
  401c7c:	b8 01 00 00 00       	mov    $0x1,%eax
  401c81:	c3                   	ret    

0000000000401c82 <save_char>:
  401c82:	8b 05 bc 44 20 00    	mov    0x2044bc(%rip),%eax        # 606144 <gets_cnt>
  401c88:	3d ff 03 00 00       	cmp    $0x3ff,%eax
  401c8d:	7f 49                	jg     401cd8 <save_char+0x56>
  401c8f:	8d 14 40             	lea    (%rax,%rax,2),%edx
  401c92:	89 f9                	mov    %edi,%ecx
  401c94:	c0 e9 04             	shr    $0x4,%cl
  401c97:	83 e1 0f             	and    $0xf,%ecx
  401c9a:	0f b6 b1 30 36 40 00 	movzbl 0x403630(%rcx),%esi
  401ca1:	48 63 ca             	movslq %edx,%rcx
  401ca4:	40 88 b1 40 55 60 00 	mov    %sil,0x605540(%rcx)
  401cab:	8d 4a 01             	lea    0x1(%rdx),%ecx
  401cae:	83 e7 0f             	and    $0xf,%edi
  401cb1:	0f b6 b7 30 36 40 00 	movzbl 0x403630(%rdi),%esi
  401cb8:	48 63 c9             	movslq %ecx,%rcx
  401cbb:	40 88 b1 40 55 60 00 	mov    %sil,0x605540(%rcx)
  401cc2:	83 c2 02             	add    $0x2,%edx
  401cc5:	48 63 d2             	movslq %edx,%rdx
  401cc8:	c6 82 40 55 60 00 20 	movb   $0x20,0x605540(%rdx)
  401ccf:	83 c0 01             	add    $0x1,%eax
  401cd2:	89 05 6c 44 20 00    	mov    %eax,0x20446c(%rip)        # 606144 <gets_cnt>
  401cd8:	f3 c3                	repz ret 

0000000000401cda <save_term>:
  401cda:	8b 05 64 44 20 00    	mov    0x204464(%rip),%eax        # 606144 <gets_cnt>
  401ce0:	8d 04 40             	lea    (%rax,%rax,2),%eax
  401ce3:	48 98                	cltq   
  401ce5:	c6 80 40 55 60 00 00 	movb   $0x0,0x605540(%rax)
  401cec:	c3                   	ret    

0000000000401ced <check_fail>:
  401ced:	48 83 ec 08          	sub    $0x8,%rsp
  401cf1:	0f be 15 58 44 20 00 	movsbl 0x204458(%rip),%edx        # 606150 <target_prefix>
  401cf8:	41 b8 40 55 60 00    	mov    $0x605540,%r8d
  401cfe:	8b 0d 18 38 20 00    	mov    0x203818(%rip),%ecx        # 60551c <check_level>
  401d04:	be db 33 40 00       	mov    $0x4033db,%esi
  401d09:	bf 01 00 00 00       	mov    $0x1,%edi
  401d0e:	b8 00 00 00 00       	mov    $0x0,%eax
  401d13:	e8 d8 ef ff ff       	call   400cf0 <__printf_chk@plt>
  401d18:	bf 01 00 00 00       	mov    $0x1,%edi
  401d1d:	e8 ae f1 ff ff       	call   400ed0 <exit@plt>

0000000000401d22 <Gets>:
  401d22:	41 54                	push   %r12
  401d24:	55                   	push   %rbp
  401d25:	53                   	push   %rbx
  401d26:	49 89 fc             	mov    %rdi,%r12
  401d29:	c7 05 11 44 20 00 00 	movl   $0x0,0x204411(%rip)        # 606144 <gets_cnt>
  401d30:	00 00 00 
  401d33:	48 89 fb             	mov    %rdi,%rbx
  401d36:	eb 11                	jmp    401d49 <Gets+0x27>
  401d38:	48 8d 6b 01          	lea    0x1(%rbx),%rbp
  401d3c:	88 03                	mov    %al,(%rbx)
  401d3e:	0f b6 f8             	movzbl %al,%edi
  401d41:	e8 3c ff ff ff       	call   401c82 <save_char>
  401d46:	48 89 eb             	mov    %rbp,%rbx
  401d49:	48 8b 3d c0 37 20 00 	mov    0x2037c0(%rip),%rdi        # 605510 <infile>
  401d50:	e8 0b f1 ff ff       	call   400e60 <_IO_getc@plt>
  401d55:	83 f8 ff             	cmp    $0xffffffff,%eax
  401d58:	74 05                	je     401d5f <Gets+0x3d>
  401d5a:	83 f8 0a             	cmp    $0xa,%eax
  401d5d:	75 d9                	jne    401d38 <Gets+0x16>
  401d5f:	c6 03 00             	movb   $0x0,(%rbx)
  401d62:	b8 00 00 00 00       	mov    $0x0,%eax
  401d67:	e8 6e ff ff ff       	call   401cda <save_term>
  401d6c:	4c 89 e0             	mov    %r12,%rax
  401d6f:	5b                   	pop    %rbx
  401d70:	5d                   	pop    %rbp
  401d71:	41 5c                	pop    %r12
  401d73:	c3                   	ret    

0000000000401d74 <notify_server>:
  401d74:	53                   	push   %rbx
  401d75:	48 81 ec 10 20 00 00 	sub    $0x2010,%rsp
  401d7c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401d83:	00 00 
  401d85:	48 89 84 24 08 20 00 	mov    %rax,0x2008(%rsp)
  401d8c:	00 
  401d8d:	31 c0                	xor    %eax,%eax
  401d8f:	83 3d 96 37 20 00 00 	cmpl   $0x0,0x203796(%rip)        # 60552c <is_checker>
  401d96:	0f 85 a5 00 00 00    	jne    401e41 <notify_server+0xcd>
  401d9c:	89 fb                	mov    %edi,%ebx
  401d9e:	8b 05 a0 43 20 00    	mov    0x2043a0(%rip),%eax        # 606144 <gets_cnt>
  401da4:	83 c0 64             	add    $0x64,%eax
  401da7:	3d 00 20 00 00       	cmp    $0x2000,%eax
  401dac:	7e 1e                	jle    401dcc <notify_server+0x58>
  401dae:	be c0 34 40 00       	mov    $0x4034c0,%esi
  401db3:	bf 01 00 00 00       	mov    $0x1,%edi
  401db8:	b8 00 00 00 00       	mov    $0x0,%eax
  401dbd:	e8 2e ef ff ff       	call   400cf0 <__printf_chk@plt>
  401dc2:	bf 01 00 00 00       	mov    $0x1,%edi
  401dc7:	e8 04 f1 ff ff       	call   400ed0 <exit@plt>
  401dcc:	0f be 05 7d 43 20 00 	movsbl 0x20437d(%rip),%eax        # 606150 <target_prefix>
  401dd3:	83 3d 3e 37 20 00 00 	cmpl   $0x0,0x20373e(%rip)        # 605518 <notify>
  401dda:	74 08                	je     401de4 <notify_server+0x70>
  401ddc:	8b 15 42 37 20 00    	mov    0x203742(%rip),%edx        # 605524 <authkey>
  401de2:	eb 05                	jmp    401de9 <notify_server+0x75>
  401de4:	ba ff ff ff ff       	mov    $0xffffffff,%edx
  401de9:	85 db                	test   %ebx,%ebx
  401deb:	74 08                	je     401df5 <notify_server+0x81>
  401ded:	41 b9 f1 33 40 00    	mov    $0x4033f1,%r9d
  401df3:	eb 06                	jmp    401dfb <notify_server+0x87>
  401df5:	41 b9 f6 33 40 00    	mov    $0x4033f6,%r9d
  401dfb:	68 40 55 60 00       	push   $0x605540
  401e00:	56                   	push   %rsi
  401e01:	50                   	push   %rax
  401e02:	52                   	push   %rdx
  401e03:	44 8b 05 5e 33 20 00 	mov    0x20335e(%rip),%r8d        # 605168 <target_id>
  401e0a:	b9 fb 33 40 00       	mov    $0x4033fb,%ecx
  401e0f:	ba 00 20 00 00       	mov    $0x2000,%edx
  401e14:	be 01 00 00 00       	mov    $0x1,%esi
  401e19:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  401e1e:	b8 00 00 00 00       	mov    $0x0,%eax
  401e23:	e8 08 f0 ff ff       	call   400e30 <__sprintf_chk@plt>
  401e28:	48 83 c4 20          	add    $0x20,%rsp
  401e2c:	85 db                	test   %ebx,%ebx
  401e2e:	74 07                	je     401e37 <notify_server+0xc3>
  401e30:	bf f1 33 40 00       	mov    $0x4033f1,%edi
  401e35:	eb 05                	jmp    401e3c <notify_server+0xc8>
  401e37:	bf f6 33 40 00       	mov    $0x4033f6,%edi
  401e3c:	e8 0f ef ff ff       	call   400d50 <puts@plt>
  401e41:	48 8b 84 24 08 20 00 	mov    0x2008(%rsp),%rax
  401e48:	00 
  401e49:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401e50:	00 00 
  401e52:	74 05                	je     401e59 <notify_server+0xe5>
  401e54:	e8 17 ef ff ff       	call   400d70 <__stack_chk_fail@plt>
  401e59:	48 81 c4 10 20 00 00 	add    $0x2010,%rsp
  401e60:	5b                   	pop    %rbx
  401e61:	c3                   	ret    

0000000000401e62 <validate>:
  401e62:	53                   	push   %rbx
  401e63:	89 fb                	mov    %edi,%ebx
  401e65:	83 3d c0 36 20 00 00 	cmpl   $0x0,0x2036c0(%rip)        # 60552c <is_checker>
  401e6c:	74 6b                	je     401ed9 <validate+0x77>
  401e6e:	39 3d ac 36 20 00    	cmp    %edi,0x2036ac(%rip)        # 605520 <vlevel>
  401e74:	74 14                	je     401e8a <validate+0x28>
  401e76:	bf 17 34 40 00       	mov    $0x403417,%edi
  401e7b:	e8 d0 ee ff ff       	call   400d50 <puts@plt>
  401e80:	b8 00 00 00 00       	mov    $0x0,%eax
  401e85:	e8 63 fe ff ff       	call   401ced <check_fail>
  401e8a:	8b 15 8c 36 20 00    	mov    0x20368c(%rip),%edx        # 60551c <check_level>
  401e90:	39 d7                	cmp    %edx,%edi
  401e92:	74 20                	je     401eb4 <validate+0x52>
  401e94:	89 f9                	mov    %edi,%ecx
  401e96:	be f0 34 40 00       	mov    $0x4034f0,%esi
  401e9b:	bf 01 00 00 00       	mov    $0x1,%edi
  401ea0:	b8 00 00 00 00       	mov    $0x0,%eax
  401ea5:	e8 46 ee ff ff       	call   400cf0 <__printf_chk@plt>
  401eaa:	b8 00 00 00 00       	mov    $0x0,%eax
  401eaf:	e8 39 fe ff ff       	call   401ced <check_fail>
  401eb4:	0f be 15 95 42 20 00 	movsbl 0x204295(%rip),%edx        # 606150 <target_prefix>
  401ebb:	41 b8 40 55 60 00    	mov    $0x605540,%r8d
  401ec1:	89 f9                	mov    %edi,%ecx
  401ec3:	be 35 34 40 00       	mov    $0x403435,%esi
  401ec8:	bf 01 00 00 00       	mov    $0x1,%edi
  401ecd:	b8 00 00 00 00       	mov    $0x0,%eax
  401ed2:	e8 19 ee ff ff       	call   400cf0 <__printf_chk@plt>
  401ed7:	eb 49                	jmp    401f22 <validate+0xc0>
  401ed9:	3b 3d 41 36 20 00    	cmp    0x203641(%rip),%edi        # 605520 <vlevel>
  401edf:	74 18                	je     401ef9 <validate+0x97>
  401ee1:	bf 17 34 40 00       	mov    $0x403417,%edi
  401ee6:	e8 65 ee ff ff       	call   400d50 <puts@plt>
  401eeb:	89 de                	mov    %ebx,%esi
  401eed:	bf 00 00 00 00       	mov    $0x0,%edi
  401ef2:	e8 7d fe ff ff       	call   401d74 <notify_server>
  401ef7:	eb 29                	jmp    401f22 <validate+0xc0>
  401ef9:	0f be 0d 50 42 20 00 	movsbl 0x204250(%rip),%ecx        # 606150 <target_prefix>
  401f00:	89 fa                	mov    %edi,%edx
  401f02:	be 18 35 40 00       	mov    $0x403518,%esi
  401f07:	bf 01 00 00 00       	mov    $0x1,%edi
  401f0c:	b8 00 00 00 00       	mov    $0x0,%eax
  401f11:	e8 da ed ff ff       	call   400cf0 <__printf_chk@plt>
  401f16:	89 de                	mov    %ebx,%esi
  401f18:	bf 01 00 00 00       	mov    $0x1,%edi
  401f1d:	e8 52 fe ff ff       	call   401d74 <notify_server>
  401f22:	5b                   	pop    %rbx
  401f23:	c3                   	ret    

0000000000401f24 <fail>:
  401f24:	48 83 ec 08          	sub    $0x8,%rsp
  401f28:	83 3d fd 35 20 00 00 	cmpl   $0x0,0x2035fd(%rip)        # 60552c <is_checker>
  401f2f:	74 0a                	je     401f3b <fail+0x17>
  401f31:	b8 00 00 00 00       	mov    $0x0,%eax
  401f36:	e8 b2 fd ff ff       	call   401ced <check_fail>
  401f3b:	89 fe                	mov    %edi,%esi
  401f3d:	bf 00 00 00 00       	mov    $0x0,%edi
  401f42:	e8 2d fe ff ff       	call   401d74 <notify_server>
  401f47:	48 83 c4 08          	add    $0x8,%rsp
  401f4b:	c3                   	ret    

0000000000401f4c <bushandler>:
  401f4c:	48 83 ec 08          	sub    $0x8,%rsp
  401f50:	83 3d d5 35 20 00 00 	cmpl   $0x0,0x2035d5(%rip)        # 60552c <is_checker>
  401f57:	74 14                	je     401f6d <bushandler+0x21>
  401f59:	bf 4a 34 40 00       	mov    $0x40344a,%edi
  401f5e:	e8 ed ed ff ff       	call   400d50 <puts@plt>
  401f63:	b8 00 00 00 00       	mov    $0x0,%eax
  401f68:	e8 80 fd ff ff       	call   401ced <check_fail>
  401f6d:	bf 50 35 40 00       	mov    $0x403550,%edi
  401f72:	e8 d9 ed ff ff       	call   400d50 <puts@plt>
  401f77:	bf 54 34 40 00       	mov    $0x403454,%edi
  401f7c:	e8 cf ed ff ff       	call   400d50 <puts@plt>
  401f81:	be 00 00 00 00       	mov    $0x0,%esi
  401f86:	bf 00 00 00 00       	mov    $0x0,%edi
  401f8b:	e8 e4 fd ff ff       	call   401d74 <notify_server>
  401f90:	bf 01 00 00 00       	mov    $0x1,%edi
  401f95:	e8 36 ef ff ff       	call   400ed0 <exit@plt>

0000000000401f9a <seghandler>:
  401f9a:	48 83 ec 08          	sub    $0x8,%rsp
  401f9e:	83 3d 87 35 20 00 00 	cmpl   $0x0,0x203587(%rip)        # 60552c <is_checker>
  401fa5:	74 14                	je     401fbb <seghandler+0x21>
  401fa7:	bf 6a 34 40 00       	mov    $0x40346a,%edi
  401fac:	e8 9f ed ff ff       	call   400d50 <puts@plt>
  401fb1:	b8 00 00 00 00       	mov    $0x0,%eax
  401fb6:	e8 32 fd ff ff       	call   401ced <check_fail>
  401fbb:	bf 70 35 40 00       	mov    $0x403570,%edi
  401fc0:	e8 8b ed ff ff       	call   400d50 <puts@plt>
  401fc5:	bf 54 34 40 00       	mov    $0x403454,%edi
  401fca:	e8 81 ed ff ff       	call   400d50 <puts@plt>
  401fcf:	be 00 00 00 00       	mov    $0x0,%esi
  401fd4:	bf 00 00 00 00       	mov    $0x0,%edi
  401fd9:	e8 96 fd ff ff       	call   401d74 <notify_server>
  401fde:	bf 01 00 00 00       	mov    $0x1,%edi
  401fe3:	e8 e8 ee ff ff       	call   400ed0 <exit@plt>

0000000000401fe8 <illegalhandler>:
  401fe8:	48 83 ec 08          	sub    $0x8,%rsp
  401fec:	83 3d 39 35 20 00 00 	cmpl   $0x0,0x203539(%rip)        # 60552c <is_checker>
  401ff3:	74 14                	je     402009 <illegalhandler+0x21>
  401ff5:	bf 7d 34 40 00       	mov    $0x40347d,%edi
  401ffa:	e8 51 ed ff ff       	call   400d50 <puts@plt>
  401fff:	b8 00 00 00 00       	mov    $0x0,%eax
  402004:	e8 e4 fc ff ff       	call   401ced <check_fail>
  402009:	bf 98 35 40 00       	mov    $0x403598,%edi
  40200e:	e8 3d ed ff ff       	call   400d50 <puts@plt>
  402013:	bf 54 34 40 00       	mov    $0x403454,%edi
  402018:	e8 33 ed ff ff       	call   400d50 <puts@plt>
  40201d:	be 00 00 00 00       	mov    $0x0,%esi
  402022:	bf 00 00 00 00       	mov    $0x0,%edi
  402027:	e8 48 fd ff ff       	call   401d74 <notify_server>
  40202c:	bf 01 00 00 00       	mov    $0x1,%edi
  402031:	e8 9a ee ff ff       	call   400ed0 <exit@plt>

0000000000402036 <sigalrmhandler>:
  402036:	48 83 ec 08          	sub    $0x8,%rsp
  40203a:	83 3d eb 34 20 00 00 	cmpl   $0x0,0x2034eb(%rip)        # 60552c <is_checker>
  402041:	74 14                	je     402057 <sigalrmhandler+0x21>
  402043:	bf 91 34 40 00       	mov    $0x403491,%edi
  402048:	e8 03 ed ff ff       	call   400d50 <puts@plt>
  40204d:	b8 00 00 00 00       	mov    $0x0,%eax
  402052:	e8 96 fc ff ff       	call   401ced <check_fail>
  402057:	ba 05 00 00 00       	mov    $0x5,%edx
  40205c:	be c8 35 40 00       	mov    $0x4035c8,%esi
  402061:	bf 01 00 00 00       	mov    $0x1,%edi
  402066:	b8 00 00 00 00       	mov    $0x0,%eax
  40206b:	e8 80 ec ff ff       	call   400cf0 <__printf_chk@plt>
  402070:	be 00 00 00 00       	mov    $0x0,%esi
  402075:	bf 00 00 00 00       	mov    $0x0,%edi
  40207a:	e8 f5 fc ff ff       	call   401d74 <notify_server>
  40207f:	bf 01 00 00 00       	mov    $0x1,%edi
  402084:	e8 47 ee ff ff       	call   400ed0 <exit@plt>

0000000000402089 <launch>:
  402089:	55                   	push   %rbp
  40208a:	48 89 e5             	mov    %rsp,%rbp
  40208d:	48 83 ec 10          	sub    $0x10,%rsp
  402091:	48 89 fa             	mov    %rdi,%rdx
  402094:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40209b:	00 00 
  40209d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4020a1:	31 c0                	xor    %eax,%eax
  4020a3:	48 8d 47 1e          	lea    0x1e(%rdi),%rax
  4020a7:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
  4020ab:	48 29 c4             	sub    %rax,%rsp
  4020ae:	48 8d 7c 24 0f       	lea    0xf(%rsp),%rdi
  4020b3:	48 83 e7 f0          	and    $0xfffffffffffffff0,%rdi
  4020b7:	be f4 00 00 00       	mov    $0xf4,%esi
  4020bc:	e8 cf ec ff ff       	call   400d90 <memset@plt>
  4020c1:	48 8b 05 f8 33 20 00 	mov    0x2033f8(%rip),%rax        # 6054c0 <stdin@GLIBC_2.2.5>
  4020c8:	48 39 05 41 34 20 00 	cmp    %rax,0x203441(%rip)        # 605510 <infile>
  4020cf:	75 14                	jne    4020e5 <launch+0x5c>
  4020d1:	be 99 34 40 00       	mov    $0x403499,%esi
  4020d6:	bf 01 00 00 00       	mov    $0x1,%edi
  4020db:	b8 00 00 00 00       	mov    $0x0,%eax
  4020e0:	e8 0b ec ff ff       	call   400cf0 <__printf_chk@plt>
  4020e5:	c7 05 31 34 20 00 00 	movl   $0x0,0x203431(%rip)        # 605520 <vlevel>
  4020ec:	00 00 00 
  4020ef:	b8 00 00 00 00       	mov    $0x0,%eax
  4020f4:	e8 39 fa ff ff       	call   401b32 <test>
  4020f9:	83 3d 2c 34 20 00 00 	cmpl   $0x0,0x20342c(%rip)        # 60552c <is_checker>
  402100:	74 14                	je     402116 <launch+0x8d>
  402102:	bf a6 34 40 00       	mov    $0x4034a6,%edi
  402107:	e8 44 ec ff ff       	call   400d50 <puts@plt>
  40210c:	b8 00 00 00 00       	mov    $0x0,%eax
  402111:	e8 d7 fb ff ff       	call   401ced <check_fail>
  402116:	bf b1 34 40 00       	mov    $0x4034b1,%edi
  40211b:	e8 30 ec ff ff       	call   400d50 <puts@plt>
  402120:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  402124:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40212b:	00 00 
  40212d:	74 05                	je     402134 <launch+0xab>
  40212f:	e8 3c ec ff ff       	call   400d70 <__stack_chk_fail@plt>
  402134:	c9                   	leave  
  402135:	c3                   	ret    

0000000000402136 <stable_launch>:
  402136:	53                   	push   %rbx
  402137:	48 89 3d ca 33 20 00 	mov    %rdi,0x2033ca(%rip)        # 605508 <global_offset>
  40213e:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  402144:	41 b8 00 00 00 00    	mov    $0x0,%r8d
  40214a:	b9 32 01 00 00       	mov    $0x132,%ecx
  40214f:	ba 07 00 00 00       	mov    $0x7,%edx
  402154:	be 00 00 10 00       	mov    $0x100000,%esi
  402159:	bf 00 60 58 55       	mov    $0x55586000,%edi
  40215e:	e8 1d ec ff ff       	call   400d80 <mmap@plt>
  402163:	48 89 c3             	mov    %rax,%rbx
  402166:	48 3d 00 60 58 55    	cmp    $0x55586000,%rax
  40216c:	74 37                	je     4021a5 <stable_launch+0x6f>
  40216e:	be 00 00 10 00       	mov    $0x100000,%esi
  402173:	48 89 c7             	mov    %rax,%rdi
  402176:	e8 05 ed ff ff       	call   400e80 <munmap@plt>
  40217b:	b9 00 60 58 55       	mov    $0x55586000,%ecx
  402180:	ba 00 36 40 00       	mov    $0x403600,%edx
  402185:	be 01 00 00 00       	mov    $0x1,%esi
  40218a:	48 8b 3d 4f 33 20 00 	mov    0x20334f(%rip),%rdi        # 6054e0 <stderr@GLIBC_2.2.5>
  402191:	b8 00 00 00 00       	mov    $0x0,%eax
  402196:	e8 55 ed ff ff       	call   400ef0 <__fprintf_chk@plt>
  40219b:	bf 01 00 00 00       	mov    $0x1,%edi
  4021a0:	e8 2b ed ff ff       	call   400ed0 <exit@plt>
  4021a5:	48 8d 90 f8 ff 0f 00 	lea    0xffff8(%rax),%rdx
  4021ac:	48 89 15 95 3f 20 00 	mov    %rdx,0x203f95(%rip)        # 606148 <stack_top>
  4021b3:	48 89 e0             	mov    %rsp,%rax
  4021b6:	48 89 d4             	mov    %rdx,%rsp
  4021b9:	48 89 c2             	mov    %rax,%rdx
  4021bc:	48 89 15 3d 33 20 00 	mov    %rdx,0x20333d(%rip)        # 605500 <global_save_stack>
  4021c3:	48 8b 3d 3e 33 20 00 	mov    0x20333e(%rip),%rdi        # 605508 <global_offset>
  4021ca:	e8 ba fe ff ff       	call   402089 <launch>
  4021cf:	48 8b 05 2a 33 20 00 	mov    0x20332a(%rip),%rax        # 605500 <global_save_stack>
  4021d6:	48 89 c4             	mov    %rax,%rsp
  4021d9:	be 00 00 10 00       	mov    $0x100000,%esi
  4021de:	48 89 df             	mov    %rbx,%rdi
  4021e1:	e8 9a ec ff ff       	call   400e80 <munmap@plt>
  4021e6:	5b                   	pop    %rbx
  4021e7:	c3                   	ret    

00000000004021e8 <rio_readinitb>:
  4021e8:	89 37                	mov    %esi,(%rdi)
  4021ea:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  4021f1:	48 8d 47 10          	lea    0x10(%rdi),%rax
  4021f5:	48 89 47 08          	mov    %rax,0x8(%rdi)
  4021f9:	c3                   	ret    

00000000004021fa <sigalrm_handler>:
  4021fa:	48 83 ec 08          	sub    $0x8,%rsp
  4021fe:	b9 00 00 00 00       	mov    $0x0,%ecx
  402203:	ba 40 36 40 00       	mov    $0x403640,%edx
  402208:	be 01 00 00 00       	mov    $0x1,%esi
  40220d:	48 8b 3d cc 32 20 00 	mov    0x2032cc(%rip),%rdi        # 6054e0 <stderr@GLIBC_2.2.5>
  402214:	b8 00 00 00 00       	mov    $0x0,%eax
  402219:	e8 d2 ec ff ff       	call   400ef0 <__fprintf_chk@plt>
  40221e:	bf 01 00 00 00       	mov    $0x1,%edi
  402223:	e8 a8 ec ff ff       	call   400ed0 <exit@plt>

0000000000402228 <rio_writen>:
  402228:	41 55                	push   %r13
  40222a:	41 54                	push   %r12
  40222c:	55                   	push   %rbp
  40222d:	53                   	push   %rbx
  40222e:	48 83 ec 08          	sub    $0x8,%rsp
  402232:	41 89 fc             	mov    %edi,%r12d
  402235:	48 89 f5             	mov    %rsi,%rbp
  402238:	49 89 d5             	mov    %rdx,%r13
  40223b:	48 89 d3             	mov    %rdx,%rbx
  40223e:	eb 28                	jmp    402268 <rio_writen+0x40>
  402240:	48 89 da             	mov    %rbx,%rdx
  402243:	48 89 ee             	mov    %rbp,%rsi
  402246:	44 89 e7             	mov    %r12d,%edi
  402249:	e8 12 eb ff ff       	call   400d60 <write@plt>
  40224e:	48 85 c0             	test   %rax,%rax
  402251:	7f 0f                	jg     402262 <rio_writen+0x3a>
  402253:	e8 b8 ea ff ff       	call   400d10 <__errno_location@plt>
  402258:	83 38 04             	cmpl   $0x4,(%rax)
  40225b:	75 15                	jne    402272 <rio_writen+0x4a>
  40225d:	b8 00 00 00 00       	mov    $0x0,%eax
  402262:	48 29 c3             	sub    %rax,%rbx
  402265:	48 01 c5             	add    %rax,%rbp
  402268:	48 85 db             	test   %rbx,%rbx
  40226b:	75 d3                	jne    402240 <rio_writen+0x18>
  40226d:	4c 89 e8             	mov    %r13,%rax
  402270:	eb 07                	jmp    402279 <rio_writen+0x51>
  402272:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  402279:	48 83 c4 08          	add    $0x8,%rsp
  40227d:	5b                   	pop    %rbx
  40227e:	5d                   	pop    %rbp
  40227f:	41 5c                	pop    %r12
  402281:	41 5d                	pop    %r13
  402283:	c3                   	ret    

0000000000402284 <rio_read>:
  402284:	41 55                	push   %r13
  402286:	41 54                	push   %r12
  402288:	55                   	push   %rbp
  402289:	53                   	push   %rbx
  40228a:	48 83 ec 08          	sub    $0x8,%rsp
  40228e:	48 89 fb             	mov    %rdi,%rbx
  402291:	49 89 f5             	mov    %rsi,%r13
  402294:	49 89 d4             	mov    %rdx,%r12
  402297:	eb 2e                	jmp    4022c7 <rio_read+0x43>
  402299:	48 8d 6b 10          	lea    0x10(%rbx),%rbp
  40229d:	8b 3b                	mov    (%rbx),%edi
  40229f:	ba 00 20 00 00       	mov    $0x2000,%edx
  4022a4:	48 89 ee             	mov    %rbp,%rsi
  4022a7:	e8 14 eb ff ff       	call   400dc0 <read@plt>
  4022ac:	89 43 04             	mov    %eax,0x4(%rbx)
  4022af:	85 c0                	test   %eax,%eax
  4022b1:	79 0c                	jns    4022bf <rio_read+0x3b>
  4022b3:	e8 58 ea ff ff       	call   400d10 <__errno_location@plt>
  4022b8:	83 38 04             	cmpl   $0x4,(%rax)
  4022bb:	74 0a                	je     4022c7 <rio_read+0x43>
  4022bd:	eb 37                	jmp    4022f6 <rio_read+0x72>
  4022bf:	85 c0                	test   %eax,%eax
  4022c1:	74 3c                	je     4022ff <rio_read+0x7b>
  4022c3:	48 89 6b 08          	mov    %rbp,0x8(%rbx)
  4022c7:	8b 6b 04             	mov    0x4(%rbx),%ebp
  4022ca:	85 ed                	test   %ebp,%ebp
  4022cc:	7e cb                	jle    402299 <rio_read+0x15>
  4022ce:	89 e8                	mov    %ebp,%eax
  4022d0:	49 39 c4             	cmp    %rax,%r12
  4022d3:	77 03                	ja     4022d8 <rio_read+0x54>
  4022d5:	44 89 e5             	mov    %r12d,%ebp
  4022d8:	4c 63 e5             	movslq %ebp,%r12
  4022db:	48 8b 73 08          	mov    0x8(%rbx),%rsi
  4022df:	4c 89 e2             	mov    %r12,%rdx
  4022e2:	4c 89 ef             	mov    %r13,%rdi
  4022e5:	e8 36 eb ff ff       	call   400e20 <memcpy@plt>
  4022ea:	4c 01 63 08          	add    %r12,0x8(%rbx)
  4022ee:	29 6b 04             	sub    %ebp,0x4(%rbx)
  4022f1:	4c 89 e0             	mov    %r12,%rax
  4022f4:	eb 0e                	jmp    402304 <rio_read+0x80>
  4022f6:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4022fd:	eb 05                	jmp    402304 <rio_read+0x80>
  4022ff:	b8 00 00 00 00       	mov    $0x0,%eax
  402304:	48 83 c4 08          	add    $0x8,%rsp
  402308:	5b                   	pop    %rbx
  402309:	5d                   	pop    %rbp
  40230a:	41 5c                	pop    %r12
  40230c:	41 5d                	pop    %r13
  40230e:	c3                   	ret    

000000000040230f <rio_readlineb>:
  40230f:	41 55                	push   %r13
  402311:	41 54                	push   %r12
  402313:	55                   	push   %rbp
  402314:	53                   	push   %rbx
  402315:	48 83 ec 18          	sub    $0x18,%rsp
  402319:	49 89 fd             	mov    %rdi,%r13
  40231c:	48 89 f5             	mov    %rsi,%rbp
  40231f:	49 89 d4             	mov    %rdx,%r12
  402322:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  402329:	00 00 
  40232b:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  402330:	31 c0                	xor    %eax,%eax
  402332:	bb 01 00 00 00       	mov    $0x1,%ebx
  402337:	eb 3f                	jmp    402378 <rio_readlineb+0x69>
  402339:	ba 01 00 00 00       	mov    $0x1,%edx
  40233e:	48 8d 74 24 07       	lea    0x7(%rsp),%rsi
  402343:	4c 89 ef             	mov    %r13,%rdi
  402346:	e8 39 ff ff ff       	call   402284 <rio_read>
  40234b:	83 f8 01             	cmp    $0x1,%eax
  40234e:	75 15                	jne    402365 <rio_readlineb+0x56>
  402350:	48 8d 45 01          	lea    0x1(%rbp),%rax
  402354:	0f b6 54 24 07       	movzbl 0x7(%rsp),%edx
  402359:	88 55 00             	mov    %dl,0x0(%rbp)
  40235c:	80 7c 24 07 0a       	cmpb   $0xa,0x7(%rsp)
  402361:	75 0e                	jne    402371 <rio_readlineb+0x62>
  402363:	eb 1a                	jmp    40237f <rio_readlineb+0x70>
  402365:	85 c0                	test   %eax,%eax
  402367:	75 22                	jne    40238b <rio_readlineb+0x7c>
  402369:	48 83 fb 01          	cmp    $0x1,%rbx
  40236d:	75 13                	jne    402382 <rio_readlineb+0x73>
  40236f:	eb 23                	jmp    402394 <rio_readlineb+0x85>
  402371:	48 83 c3 01          	add    $0x1,%rbx
  402375:	48 89 c5             	mov    %rax,%rbp
  402378:	4c 39 e3             	cmp    %r12,%rbx
  40237b:	72 bc                	jb     402339 <rio_readlineb+0x2a>
  40237d:	eb 03                	jmp    402382 <rio_readlineb+0x73>
  40237f:	48 89 c5             	mov    %rax,%rbp
  402382:	c6 45 00 00          	movb   $0x0,0x0(%rbp)
  402386:	48 89 d8             	mov    %rbx,%rax
  402389:	eb 0e                	jmp    402399 <rio_readlineb+0x8a>
  40238b:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  402392:	eb 05                	jmp    402399 <rio_readlineb+0x8a>
  402394:	b8 00 00 00 00       	mov    $0x0,%eax
  402399:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  40239e:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  4023a5:	00 00 
  4023a7:	74 05                	je     4023ae <rio_readlineb+0x9f>
  4023a9:	e8 c2 e9 ff ff       	call   400d70 <__stack_chk_fail@plt>
  4023ae:	48 83 c4 18          	add    $0x18,%rsp
  4023b2:	5b                   	pop    %rbx
  4023b3:	5d                   	pop    %rbp
  4023b4:	41 5c                	pop    %r12
  4023b6:	41 5d                	pop    %r13
  4023b8:	c3                   	ret    

00000000004023b9 <urlencode>:
  4023b9:	41 54                	push   %r12
  4023bb:	55                   	push   %rbp
  4023bc:	53                   	push   %rbx
  4023bd:	48 83 ec 10          	sub    $0x10,%rsp
  4023c1:	48 89 fb             	mov    %rdi,%rbx
  4023c4:	48 89 f5             	mov    %rsi,%rbp
  4023c7:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4023ce:	00 00 
  4023d0:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4023d5:	31 c0                	xor    %eax,%eax
  4023d7:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  4023de:	f2 ae                	repnz scas %es:(%rdi),%al
  4023e0:	48 f7 d1             	not    %rcx
  4023e3:	8d 41 ff             	lea    -0x1(%rcx),%eax
  4023e6:	e9 aa 00 00 00       	jmp    402495 <urlencode+0xdc>
  4023eb:	44 0f b6 03          	movzbl (%rbx),%r8d
  4023ef:	41 80 f8 2a          	cmp    $0x2a,%r8b
  4023f3:	0f 94 c2             	sete   %dl
  4023f6:	41 80 f8 2d          	cmp    $0x2d,%r8b
  4023fa:	0f 94 c0             	sete   %al
  4023fd:	08 c2                	or     %al,%dl
  4023ff:	75 24                	jne    402425 <urlencode+0x6c>
  402401:	41 80 f8 2e          	cmp    $0x2e,%r8b
  402405:	74 1e                	je     402425 <urlencode+0x6c>
  402407:	41 80 f8 5f          	cmp    $0x5f,%r8b
  40240b:	74 18                	je     402425 <urlencode+0x6c>
  40240d:	41 8d 40 d0          	lea    -0x30(%r8),%eax
  402411:	3c 09                	cmp    $0x9,%al
  402413:	76 10                	jbe    402425 <urlencode+0x6c>
  402415:	41 8d 40 bf          	lea    -0x41(%r8),%eax
  402419:	3c 19                	cmp    $0x19,%al
  40241b:	76 08                	jbe    402425 <urlencode+0x6c>
  40241d:	41 8d 40 9f          	lea    -0x61(%r8),%eax
  402421:	3c 19                	cmp    $0x19,%al
  402423:	77 0a                	ja     40242f <urlencode+0x76>
  402425:	44 88 45 00          	mov    %r8b,0x0(%rbp)
  402429:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  40242d:	eb 5f                	jmp    40248e <urlencode+0xd5>
  40242f:	41 80 f8 20          	cmp    $0x20,%r8b
  402433:	75 0a                	jne    40243f <urlencode+0x86>
  402435:	c6 45 00 2b          	movb   $0x2b,0x0(%rbp)
  402439:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  40243d:	eb 4f                	jmp    40248e <urlencode+0xd5>
  40243f:	41 8d 40 e0          	lea    -0x20(%r8),%eax
  402443:	3c 5f                	cmp    $0x5f,%al
  402445:	0f 96 c2             	setbe  %dl
  402448:	41 80 f8 09          	cmp    $0x9,%r8b
  40244c:	0f 94 c0             	sete   %al
  40244f:	08 c2                	or     %al,%dl
  402451:	74 50                	je     4024a3 <urlencode+0xea>
  402453:	45 0f b6 c0          	movzbl %r8b,%r8d
  402457:	b9 d8 36 40 00       	mov    $0x4036d8,%ecx
  40245c:	ba 08 00 00 00       	mov    $0x8,%edx
  402461:	be 01 00 00 00       	mov    $0x1,%esi
  402466:	48 89 e7             	mov    %rsp,%rdi
  402469:	b8 00 00 00 00       	mov    $0x0,%eax
  40246e:	e8 bd e9 ff ff       	call   400e30 <__sprintf_chk@plt>
  402473:	0f b6 04 24          	movzbl (%rsp),%eax
  402477:	88 45 00             	mov    %al,0x0(%rbp)
  40247a:	0f b6 44 24 01       	movzbl 0x1(%rsp),%eax
  40247f:	88 45 01             	mov    %al,0x1(%rbp)
  402482:	0f b6 44 24 02       	movzbl 0x2(%rsp),%eax
  402487:	88 45 02             	mov    %al,0x2(%rbp)
  40248a:	48 8d 6d 03          	lea    0x3(%rbp),%rbp
  40248e:	48 83 c3 01          	add    $0x1,%rbx
  402492:	44 89 e0             	mov    %r12d,%eax
  402495:	44 8d 60 ff          	lea    -0x1(%rax),%r12d
  402499:	85 c0                	test   %eax,%eax
  40249b:	0f 85 4a ff ff ff    	jne    4023eb <urlencode+0x32>
  4024a1:	eb 05                	jmp    4024a8 <urlencode+0xef>
  4024a3:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4024a8:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  4024ad:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  4024b4:	00 00 
  4024b6:	74 05                	je     4024bd <urlencode+0x104>
  4024b8:	e8 b3 e8 ff ff       	call   400d70 <__stack_chk_fail@plt>
  4024bd:	48 83 c4 10          	add    $0x10,%rsp
  4024c1:	5b                   	pop    %rbx
  4024c2:	5d                   	pop    %rbp
  4024c3:	41 5c                	pop    %r12
  4024c5:	c3                   	ret    

00000000004024c6 <submitr>:
  4024c6:	41 57                	push   %r15
  4024c8:	41 56                	push   %r14
  4024ca:	41 55                	push   %r13
  4024cc:	41 54                	push   %r12
  4024ce:	55                   	push   %rbp
  4024cf:	53                   	push   %rbx
  4024d0:	48 81 ec 58 a0 00 00 	sub    $0xa058,%rsp
  4024d7:	49 89 fc             	mov    %rdi,%r12
  4024da:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4024de:	49 89 d7             	mov    %rdx,%r15
  4024e1:	49 89 ce             	mov    %rcx,%r14
  4024e4:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
  4024e9:	4d 89 cd             	mov    %r9,%r13
  4024ec:	48 8b 9c 24 90 a0 00 	mov    0xa090(%rsp),%rbx
  4024f3:	00 
  4024f4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4024fb:	00 00 
  4024fd:	48 89 84 24 48 a0 00 	mov    %rax,0xa048(%rsp)
  402504:	00 
  402505:	31 c0                	xor    %eax,%eax
  402507:	c7 44 24 1c 00 00 00 	movl   $0x0,0x1c(%rsp)
  40250e:	00 
  40250f:	ba 00 00 00 00       	mov    $0x0,%edx
  402514:	be 01 00 00 00       	mov    $0x1,%esi
  402519:	bf 02 00 00 00       	mov    $0x2,%edi
  40251e:	e8 dd e9 ff ff       	call   400f00 <socket@plt>
  402523:	85 c0                	test   %eax,%eax
  402525:	79 4e                	jns    402575 <submitr+0xaf>
  402527:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  40252e:	3a 20 43 
  402531:	48 89 03             	mov    %rax,(%rbx)
  402534:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  40253b:	20 75 6e 
  40253e:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402542:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402549:	74 6f 20 
  40254c:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402550:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  402557:	65 20 73 
  40255a:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40255e:	c7 43 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbx)
  402565:	66 c7 43 24 74 00    	movw   $0x74,0x24(%rbx)
  40256b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402570:	e9 97 06 00 00       	jmp    402c0c <submitr+0x746>
  402575:	89 c5                	mov    %eax,%ebp
  402577:	4c 89 e7             	mov    %r12,%rdi
  40257a:	e8 71 e8 ff ff       	call   400df0 <gethostbyname@plt>
  40257f:	48 85 c0             	test   %rax,%rax
  402582:	75 67                	jne    4025eb <submitr+0x125>
  402584:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  40258b:	3a 20 44 
  40258e:	48 89 03             	mov    %rax,(%rbx)
  402591:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  402598:	20 75 6e 
  40259b:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40259f:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  4025a6:	74 6f 20 
  4025a9:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4025ad:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  4025b4:	76 65 20 
  4025b7:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4025bb:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  4025c2:	72 20 61 
  4025c5:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4025c9:	c7 43 28 64 64 72 65 	movl   $0x65726464,0x28(%rbx)
  4025d0:	66 c7 43 2c 73 73    	movw   $0x7373,0x2c(%rbx)
  4025d6:	c6 43 2e 00          	movb   $0x0,0x2e(%rbx)
  4025da:	89 ef                	mov    %ebp,%edi
  4025dc:	e8 cf e7 ff ff       	call   400db0 <close@plt>
  4025e1:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4025e6:	e9 21 06 00 00       	jmp    402c0c <submitr+0x746>
  4025eb:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
  4025f2:	00 00 
  4025f4:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
  4025fb:	00 00 
  4025fd:	66 c7 44 24 20 02 00 	movw   $0x2,0x20(%rsp)
  402604:	48 63 50 14          	movslq 0x14(%rax),%rdx
  402608:	48 8b 40 18          	mov    0x18(%rax),%rax
  40260c:	48 8b 30             	mov    (%rax),%rsi
  40260f:	48 8d 7c 24 24       	lea    0x24(%rsp),%rdi
  402614:	b9 0c 00 00 00       	mov    $0xc,%ecx
  402619:	e8 e2 e7 ff ff       	call   400e00 <__memmove_chk@plt>
  40261e:	0f b7 44 24 04       	movzwl 0x4(%rsp),%eax
  402623:	66 c1 c8 08          	ror    $0x8,%ax
  402627:	66 89 44 24 22       	mov    %ax,0x22(%rsp)
  40262c:	ba 10 00 00 00       	mov    $0x10,%edx
  402631:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  402636:	89 ef                	mov    %ebp,%edi
  402638:	e8 a3 e8 ff ff       	call   400ee0 <connect@plt>
  40263d:	85 c0                	test   %eax,%eax
  40263f:	79 59                	jns    40269a <submitr+0x1d4>
  402641:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
  402648:	3a 20 55 
  40264b:	48 89 03             	mov    %rax,(%rbx)
  40264e:	48 b8 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rax
  402655:	20 74 6f 
  402658:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40265c:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
  402663:	65 63 74 
  402666:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40266a:	48 b8 20 74 6f 20 74 	movabs $0x20656874206f7420,%rax
  402671:	68 65 20 
  402674:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402678:	c7 43 20 73 65 72 76 	movl   $0x76726573,0x20(%rbx)
  40267f:	66 c7 43 24 65 72    	movw   $0x7265,0x24(%rbx)
  402685:	c6 43 26 00          	movb   $0x0,0x26(%rbx)
  402689:	89 ef                	mov    %ebp,%edi
  40268b:	e8 20 e7 ff ff       	call   400db0 <close@plt>
  402690:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402695:	e9 72 05 00 00       	jmp    402c0c <submitr+0x746>
  40269a:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  4026a1:	b8 00 00 00 00       	mov    $0x0,%eax
  4026a6:	48 89 f1             	mov    %rsi,%rcx
  4026a9:	4c 89 ef             	mov    %r13,%rdi
  4026ac:	f2 ae                	repnz scas %es:(%rdi),%al
  4026ae:	48 f7 d1             	not    %rcx
  4026b1:	48 89 ca             	mov    %rcx,%rdx
  4026b4:	48 89 f1             	mov    %rsi,%rcx
  4026b7:	4c 89 ff             	mov    %r15,%rdi
  4026ba:	f2 ae                	repnz scas %es:(%rdi),%al
  4026bc:	48 f7 d1             	not    %rcx
  4026bf:	49 89 c8             	mov    %rcx,%r8
  4026c2:	48 89 f1             	mov    %rsi,%rcx
  4026c5:	4c 89 f7             	mov    %r14,%rdi
  4026c8:	f2 ae                	repnz scas %es:(%rdi),%al
  4026ca:	48 f7 d1             	not    %rcx
  4026cd:	4d 8d 44 08 fe       	lea    -0x2(%r8,%rcx,1),%r8
  4026d2:	48 89 f1             	mov    %rsi,%rcx
  4026d5:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4026da:	f2 ae                	repnz scas %es:(%rdi),%al
  4026dc:	48 89 c8             	mov    %rcx,%rax
  4026df:	48 f7 d0             	not    %rax
  4026e2:	49 8d 4c 00 ff       	lea    -0x1(%r8,%rax,1),%rcx
  4026e7:	48 8d 44 52 fd       	lea    -0x3(%rdx,%rdx,2),%rax
  4026ec:	48 8d 84 01 80 00 00 	lea    0x80(%rcx,%rax,1),%rax
  4026f3:	00 
  4026f4:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
  4026fa:	76 72                	jbe    40276e <submitr+0x2a8>
  4026fc:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  402703:	3a 20 52 
  402706:	48 89 03             	mov    %rax,(%rbx)
  402709:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  402710:	20 73 74 
  402713:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402717:	48 b8 72 69 6e 67 20 	movabs $0x6f6f7420676e6972,%rax
  40271e:	74 6f 6f 
  402721:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402725:	48 b8 20 6c 61 72 67 	movabs $0x202e656772616c20,%rax
  40272c:	65 2e 20 
  40272f:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402733:	48 b8 49 6e 63 72 65 	movabs $0x6573616572636e49,%rax
  40273a:	61 73 65 
  40273d:	48 89 43 20          	mov    %rax,0x20(%rbx)
  402741:	48 b8 20 53 55 42 4d 	movabs $0x5254494d42555320,%rax
  402748:	49 54 52 
  40274b:	48 89 43 28          	mov    %rax,0x28(%rbx)
  40274f:	48 b8 5f 4d 41 58 42 	movabs $0x46554258414d5f,%rax
  402756:	55 46 00 
  402759:	48 89 43 30          	mov    %rax,0x30(%rbx)
  40275d:	89 ef                	mov    %ebp,%edi
  40275f:	e8 4c e6 ff ff       	call   400db0 <close@plt>
  402764:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402769:	e9 9e 04 00 00       	jmp    402c0c <submitr+0x746>
  40276e:	48 8d b4 24 40 40 00 	lea    0x4040(%rsp),%rsi
  402775:	00 
  402776:	b9 00 04 00 00       	mov    $0x400,%ecx
  40277b:	b8 00 00 00 00       	mov    $0x0,%eax
  402780:	48 89 f7             	mov    %rsi,%rdi
  402783:	f3 48 ab             	rep stos %rax,%es:(%rdi)
  402786:	4c 89 ef             	mov    %r13,%rdi
  402789:	e8 2b fc ff ff       	call   4023b9 <urlencode>
  40278e:	85 c0                	test   %eax,%eax
  402790:	0f 89 8a 00 00 00    	jns    402820 <submitr+0x35a>
  402796:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  40279d:	3a 20 52 
  4027a0:	48 89 03             	mov    %rax,(%rbx)
  4027a3:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  4027aa:	20 73 74 
  4027ad:	48 89 43 08          	mov    %rax,0x8(%rbx)
  4027b1:	48 b8 72 69 6e 67 20 	movabs $0x6e6f6320676e6972,%rax
  4027b8:	63 6f 6e 
  4027bb:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4027bf:	48 b8 74 61 69 6e 73 	movabs $0x6e6120736e696174,%rax
  4027c6:	20 61 6e 
  4027c9:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4027cd:	48 b8 20 69 6c 6c 65 	movabs $0x6c6167656c6c6920,%rax
  4027d4:	67 61 6c 
  4027d7:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4027db:	48 b8 20 6f 72 20 75 	movabs $0x72706e7520726f20,%rax
  4027e2:	6e 70 72 
  4027e5:	48 89 43 28          	mov    %rax,0x28(%rbx)
  4027e9:	48 b8 69 6e 74 61 62 	movabs $0x20656c6261746e69,%rax
  4027f0:	6c 65 20 
  4027f3:	48 89 43 30          	mov    %rax,0x30(%rbx)
  4027f7:	48 b8 63 68 61 72 61 	movabs $0x6574636172616863,%rax
  4027fe:	63 74 65 
  402801:	48 89 43 38          	mov    %rax,0x38(%rbx)
  402805:	66 c7 43 40 72 2e    	movw   $0x2e72,0x40(%rbx)
  40280b:	c6 43 42 00          	movb   $0x0,0x42(%rbx)
  40280f:	89 ef                	mov    %ebp,%edi
  402811:	e8 9a e5 ff ff       	call   400db0 <close@plt>
  402816:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40281b:	e9 ec 03 00 00       	jmp    402c0c <submitr+0x746>
  402820:	4c 8d ac 24 40 20 00 	lea    0x2040(%rsp),%r13
  402827:	00 
  402828:	41 54                	push   %r12
  40282a:	48 8d 84 24 48 40 00 	lea    0x4048(%rsp),%rax
  402831:	00 
  402832:	50                   	push   %rax
  402833:	4d 89 f9             	mov    %r15,%r9
  402836:	4d 89 f0             	mov    %r14,%r8
  402839:	b9 68 36 40 00       	mov    $0x403668,%ecx
  40283e:	ba 00 20 00 00       	mov    $0x2000,%edx
  402843:	be 01 00 00 00       	mov    $0x1,%esi
  402848:	4c 89 ef             	mov    %r13,%rdi
  40284b:	b8 00 00 00 00       	mov    $0x0,%eax
  402850:	e8 db e5 ff ff       	call   400e30 <__sprintf_chk@plt>
  402855:	b8 00 00 00 00       	mov    $0x0,%eax
  40285a:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  402861:	4c 89 ef             	mov    %r13,%rdi
  402864:	f2 ae                	repnz scas %es:(%rdi),%al
  402866:	48 f7 d1             	not    %rcx
  402869:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
  40286d:	4c 89 ee             	mov    %r13,%rsi
  402870:	89 ef                	mov    %ebp,%edi
  402872:	e8 b1 f9 ff ff       	call   402228 <rio_writen>
  402877:	48 83 c4 10          	add    $0x10,%rsp
  40287b:	48 85 c0             	test   %rax,%rax
  40287e:	79 6e                	jns    4028ee <submitr+0x428>
  402880:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402887:	3a 20 43 
  40288a:	48 89 03             	mov    %rax,(%rbx)
  40288d:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402894:	20 75 6e 
  402897:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40289b:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  4028a2:	74 6f 20 
  4028a5:	48 89 43 10          	mov    %rax,0x10(%rbx)
  4028a9:	48 b8 77 72 69 74 65 	movabs $0x6f74206574697277,%rax
  4028b0:	20 74 6f 
  4028b3:	48 89 43 18          	mov    %rax,0x18(%rbx)
  4028b7:	48 b8 20 74 68 65 20 	movabs $0x7365722065687420,%rax
  4028be:	72 65 73 
  4028c1:	48 89 43 20          	mov    %rax,0x20(%rbx)
  4028c5:	48 b8 75 6c 74 20 73 	movabs $0x7672657320746c75,%rax
  4028cc:	65 72 76 
  4028cf:	48 89 43 28          	mov    %rax,0x28(%rbx)
  4028d3:	66 c7 43 30 65 72    	movw   $0x7265,0x30(%rbx)
  4028d9:	c6 43 32 00          	movb   $0x0,0x32(%rbx)
  4028dd:	89 ef                	mov    %ebp,%edi
  4028df:	e8 cc e4 ff ff       	call   400db0 <close@plt>
  4028e4:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4028e9:	e9 1e 03 00 00       	jmp    402c0c <submitr+0x746>
  4028ee:	89 ee                	mov    %ebp,%esi
  4028f0:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4028f5:	e8 ee f8 ff ff       	call   4021e8 <rio_readinitb>
  4028fa:	ba 00 20 00 00       	mov    $0x2000,%edx
  4028ff:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  402906:	00 
  402907:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  40290c:	e8 fe f9 ff ff       	call   40230f <rio_readlineb>
  402911:	48 85 c0             	test   %rax,%rax
  402914:	7f 7d                	jg     402993 <submitr+0x4cd>
  402916:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  40291d:	3a 20 43 
  402920:	48 89 03             	mov    %rax,(%rbx)
  402923:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  40292a:	20 75 6e 
  40292d:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402931:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402938:	74 6f 20 
  40293b:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40293f:	48 b8 72 65 61 64 20 	movabs $0x7269662064616572,%rax
  402946:	66 69 72 
  402949:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40294d:	48 b8 73 74 20 68 65 	movabs $0x6564616568207473,%rax
  402954:	61 64 65 
  402957:	48 89 43 20          	mov    %rax,0x20(%rbx)
  40295b:	48 b8 72 20 66 72 6f 	movabs $0x72206d6f72662072,%rax
  402962:	6d 20 72 
  402965:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402969:	48 b8 65 73 75 6c 74 	movabs $0x657320746c757365,%rax
  402970:	20 73 65 
  402973:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402977:	c7 43 38 72 76 65 72 	movl   $0x72657672,0x38(%rbx)
  40297e:	c6 43 3c 00          	movb   $0x0,0x3c(%rbx)
  402982:	89 ef                	mov    %ebp,%edi
  402984:	e8 27 e4 ff ff       	call   400db0 <close@plt>
  402989:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40298e:	e9 79 02 00 00       	jmp    402c0c <submitr+0x746>
  402993:	4c 8d 84 24 40 80 00 	lea    0x8040(%rsp),%r8
  40299a:	00 
  40299b:	48 8d 4c 24 1c       	lea    0x1c(%rsp),%rcx
  4029a0:	48 8d 94 24 40 60 00 	lea    0x6040(%rsp),%rdx
  4029a7:	00 
  4029a8:	be df 36 40 00       	mov    $0x4036df,%esi
  4029ad:	48 8d bc 24 40 20 00 	lea    0x2040(%rsp),%rdi
  4029b4:	00 
  4029b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4029ba:	e8 b1 e4 ff ff       	call   400e70 <__isoc99_sscanf@plt>
  4029bf:	e9 95 00 00 00       	jmp    402a59 <submitr+0x593>
  4029c4:	ba 00 20 00 00       	mov    $0x2000,%edx
  4029c9:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  4029d0:	00 
  4029d1:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4029d6:	e8 34 f9 ff ff       	call   40230f <rio_readlineb>
  4029db:	48 85 c0             	test   %rax,%rax
  4029de:	7f 79                	jg     402a59 <submitr+0x593>
  4029e0:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  4029e7:	3a 20 43 
  4029ea:	48 89 03             	mov    %rax,(%rbx)
  4029ed:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  4029f4:	20 75 6e 
  4029f7:	48 89 43 08          	mov    %rax,0x8(%rbx)
  4029fb:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402a02:	74 6f 20 
  402a05:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402a09:	48 b8 72 65 61 64 20 	movabs $0x6165682064616572,%rax
  402a10:	68 65 61 
  402a13:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402a17:	48 b8 64 65 72 73 20 	movabs $0x6f72662073726564,%rax
  402a1e:	66 72 6f 
  402a21:	48 89 43 20          	mov    %rax,0x20(%rbx)
  402a25:	48 b8 6d 20 74 68 65 	movabs $0x657220656874206d,%rax
  402a2c:	20 72 65 
  402a2f:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402a33:	48 b8 73 75 6c 74 20 	movabs $0x72657320746c7573,%rax
  402a3a:	73 65 72 
  402a3d:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402a41:	c7 43 38 76 65 72 00 	movl   $0x726576,0x38(%rbx)
  402a48:	89 ef                	mov    %ebp,%edi
  402a4a:	e8 61 e3 ff ff       	call   400db0 <close@plt>
  402a4f:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402a54:	e9 b3 01 00 00       	jmp    402c0c <submitr+0x746>
  402a59:	0f b6 94 24 40 20 00 	movzbl 0x2040(%rsp),%edx
  402a60:	00 
  402a61:	b8 0d 00 00 00       	mov    $0xd,%eax
  402a66:	29 d0                	sub    %edx,%eax
  402a68:	75 1b                	jne    402a85 <submitr+0x5bf>
  402a6a:	0f b6 94 24 41 20 00 	movzbl 0x2041(%rsp),%edx
  402a71:	00 
  402a72:	b8 0a 00 00 00       	mov    $0xa,%eax
  402a77:	29 d0                	sub    %edx,%eax
  402a79:	75 0a                	jne    402a85 <submitr+0x5bf>
  402a7b:	0f b6 84 24 42 20 00 	movzbl 0x2042(%rsp),%eax
  402a82:	00 
  402a83:	f7 d8                	neg    %eax
  402a85:	85 c0                	test   %eax,%eax
  402a87:	0f 85 37 ff ff ff    	jne    4029c4 <submitr+0x4fe>
  402a8d:	ba 00 20 00 00       	mov    $0x2000,%edx
  402a92:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  402a99:	00 
  402a9a:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402a9f:	e8 6b f8 ff ff       	call   40230f <rio_readlineb>
  402aa4:	48 85 c0             	test   %rax,%rax
  402aa7:	0f 8f 83 00 00 00    	jg     402b30 <submitr+0x66a>
  402aad:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402ab4:	3a 20 43 
  402ab7:	48 89 03             	mov    %rax,(%rbx)
  402aba:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402ac1:	20 75 6e 
  402ac4:	48 89 43 08          	mov    %rax,0x8(%rbx)
  402ac8:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402acf:	74 6f 20 
  402ad2:	48 89 43 10          	mov    %rax,0x10(%rbx)
  402ad6:	48 b8 72 65 61 64 20 	movabs $0x6174732064616572,%rax
  402add:	73 74 61 
  402ae0:	48 89 43 18          	mov    %rax,0x18(%rbx)
  402ae4:	48 b8 74 75 73 20 6d 	movabs $0x7373656d20737574,%rax
  402aeb:	65 73 73 
  402aee:	48 89 43 20          	mov    %rax,0x20(%rbx)
  402af2:	48 b8 61 67 65 20 66 	movabs $0x6d6f726620656761,%rax
  402af9:	72 6f 6d 
  402afc:	48 89 43 28          	mov    %rax,0x28(%rbx)
  402b00:	48 b8 20 72 65 73 75 	movabs $0x20746c7573657220,%rax
  402b07:	6c 74 20 
  402b0a:	48 89 43 30          	mov    %rax,0x30(%rbx)
  402b0e:	c7 43 38 73 65 72 76 	movl   $0x76726573,0x38(%rbx)
  402b15:	66 c7 43 3c 65 72    	movw   $0x7265,0x3c(%rbx)
  402b1b:	c6 43 3e 00          	movb   $0x0,0x3e(%rbx)
  402b1f:	89 ef                	mov    %ebp,%edi
  402b21:	e8 8a e2 ff ff       	call   400db0 <close@plt>
  402b26:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402b2b:	e9 dc 00 00 00       	jmp    402c0c <submitr+0x746>
  402b30:	44 8b 44 24 1c       	mov    0x1c(%rsp),%r8d
  402b35:	41 81 f8 c8 00 00 00 	cmp    $0xc8,%r8d
  402b3c:	74 37                	je     402b75 <submitr+0x6af>
  402b3e:	4c 8d 8c 24 40 80 00 	lea    0x8040(%rsp),%r9
  402b45:	00 
  402b46:	b9 a8 36 40 00       	mov    $0x4036a8,%ecx
  402b4b:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  402b52:	be 01 00 00 00       	mov    $0x1,%esi
  402b57:	48 89 df             	mov    %rbx,%rdi
  402b5a:	b8 00 00 00 00       	mov    $0x0,%eax
  402b5f:	e8 cc e2 ff ff       	call   400e30 <__sprintf_chk@plt>
  402b64:	89 ef                	mov    %ebp,%edi
  402b66:	e8 45 e2 ff ff       	call   400db0 <close@plt>
  402b6b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402b70:	e9 97 00 00 00       	jmp    402c0c <submitr+0x746>
  402b75:	48 8d b4 24 40 20 00 	lea    0x2040(%rsp),%rsi
  402b7c:	00 
  402b7d:	48 89 df             	mov    %rbx,%rdi
  402b80:	e8 bb e1 ff ff       	call   400d40 <strcpy@plt>
  402b85:	89 ef                	mov    %ebp,%edi
  402b87:	e8 24 e2 ff ff       	call   400db0 <close@plt>
  402b8c:	0f b6 03             	movzbl (%rbx),%eax
  402b8f:	ba 4f 00 00 00       	mov    $0x4f,%edx
  402b94:	29 c2                	sub    %eax,%edx
  402b96:	75 22                	jne    402bba <submitr+0x6f4>
  402b98:	0f b6 4b 01          	movzbl 0x1(%rbx),%ecx
  402b9c:	b8 4b 00 00 00       	mov    $0x4b,%eax
  402ba1:	29 c8                	sub    %ecx,%eax
  402ba3:	75 17                	jne    402bbc <submitr+0x6f6>
  402ba5:	0f b6 4b 02          	movzbl 0x2(%rbx),%ecx
  402ba9:	b8 0a 00 00 00       	mov    $0xa,%eax
  402bae:	29 c8                	sub    %ecx,%eax
  402bb0:	75 0a                	jne    402bbc <submitr+0x6f6>
  402bb2:	0f b6 43 03          	movzbl 0x3(%rbx),%eax
  402bb6:	f7 d8                	neg    %eax
  402bb8:	eb 02                	jmp    402bbc <submitr+0x6f6>
  402bba:	89 d0                	mov    %edx,%eax
  402bbc:	85 c0                	test   %eax,%eax
  402bbe:	74 40                	je     402c00 <submitr+0x73a>
  402bc0:	bf f0 36 40 00       	mov    $0x4036f0,%edi
  402bc5:	b9 05 00 00 00       	mov    $0x5,%ecx
  402bca:	48 89 de             	mov    %rbx,%rsi
  402bcd:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  402bcf:	0f 97 c0             	seta   %al
  402bd2:	0f 92 c1             	setb   %cl
  402bd5:	29 c8                	sub    %ecx,%eax
  402bd7:	0f be c0             	movsbl %al,%eax
  402bda:	85 c0                	test   %eax,%eax
  402bdc:	74 2e                	je     402c0c <submitr+0x746>
  402bde:	85 d2                	test   %edx,%edx
  402be0:	75 13                	jne    402bf5 <submitr+0x72f>
  402be2:	0f b6 43 01          	movzbl 0x1(%rbx),%eax
  402be6:	ba 4b 00 00 00       	mov    $0x4b,%edx
  402beb:	29 c2                	sub    %eax,%edx
  402bed:	75 06                	jne    402bf5 <submitr+0x72f>
  402bef:	0f b6 53 02          	movzbl 0x2(%rbx),%edx
  402bf3:	f7 da                	neg    %edx
  402bf5:	85 d2                	test   %edx,%edx
  402bf7:	75 0e                	jne    402c07 <submitr+0x741>
  402bf9:	b8 00 00 00 00       	mov    $0x0,%eax
  402bfe:	eb 0c                	jmp    402c0c <submitr+0x746>
  402c00:	b8 00 00 00 00       	mov    $0x0,%eax
  402c05:	eb 05                	jmp    402c0c <submitr+0x746>
  402c07:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402c0c:	48 8b 9c 24 48 a0 00 	mov    0xa048(%rsp),%rbx
  402c13:	00 
  402c14:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
  402c1b:	00 00 
  402c1d:	74 05                	je     402c24 <submitr+0x75e>
  402c1f:	e8 4c e1 ff ff       	call   400d70 <__stack_chk_fail@plt>
  402c24:	48 81 c4 58 a0 00 00 	add    $0xa058,%rsp
  402c2b:	5b                   	pop    %rbx
  402c2c:	5d                   	pop    %rbp
  402c2d:	41 5c                	pop    %r12
  402c2f:	41 5d                	pop    %r13
  402c31:	41 5e                	pop    %r14
  402c33:	41 5f                	pop    %r15
  402c35:	c3                   	ret    

0000000000402c36 <init_timeout>:
  402c36:	85 ff                	test   %edi,%edi
  402c38:	74 23                	je     402c5d <init_timeout+0x27>
  402c3a:	53                   	push   %rbx
  402c3b:	89 fb                	mov    %edi,%ebx
  402c3d:	85 ff                	test   %edi,%edi
  402c3f:	79 05                	jns    402c46 <init_timeout+0x10>
  402c41:	bb 00 00 00 00       	mov    $0x0,%ebx
  402c46:	be fa 21 40 00       	mov    $0x4021fa,%esi
  402c4b:	bf 0e 00 00 00       	mov    $0xe,%edi
  402c50:	e8 8b e1 ff ff       	call   400de0 <signal@plt>
  402c55:	89 df                	mov    %ebx,%edi
  402c57:	e8 44 e1 ff ff       	call   400da0 <alarm@plt>
  402c5c:	5b                   	pop    %rbx
  402c5d:	f3 c3                	repz ret 

0000000000402c5f <init_driver>:
  402c5f:	55                   	push   %rbp
  402c60:	53                   	push   %rbx
  402c61:	48 83 ec 28          	sub    $0x28,%rsp
  402c65:	48 89 fd             	mov    %rdi,%rbp
  402c68:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  402c6f:	00 00 
  402c71:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  402c76:	31 c0                	xor    %eax,%eax
  402c78:	be 01 00 00 00       	mov    $0x1,%esi
  402c7d:	bf 0d 00 00 00       	mov    $0xd,%edi
  402c82:	e8 59 e1 ff ff       	call   400de0 <signal@plt>
  402c87:	be 01 00 00 00       	mov    $0x1,%esi
  402c8c:	bf 1d 00 00 00       	mov    $0x1d,%edi
  402c91:	e8 4a e1 ff ff       	call   400de0 <signal@plt>
  402c96:	be 01 00 00 00       	mov    $0x1,%esi
  402c9b:	bf 1d 00 00 00       	mov    $0x1d,%edi
  402ca0:	e8 3b e1 ff ff       	call   400de0 <signal@plt>
  402ca5:	ba 00 00 00 00       	mov    $0x0,%edx
  402caa:	be 01 00 00 00       	mov    $0x1,%esi
  402caf:	bf 02 00 00 00       	mov    $0x2,%edi
  402cb4:	e8 47 e2 ff ff       	call   400f00 <socket@plt>
  402cb9:	85 c0                	test   %eax,%eax
  402cbb:	79 4f                	jns    402d0c <init_driver+0xad>
  402cbd:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  402cc4:	3a 20 43 
  402cc7:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402ccb:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402cd2:	20 75 6e 
  402cd5:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402cd9:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402ce0:	74 6f 20 
  402ce3:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402ce7:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  402cee:	65 20 73 
  402cf1:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402cf5:	c7 45 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbp)
  402cfc:	66 c7 45 24 74 00    	movw   $0x74,0x24(%rbp)
  402d02:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402d07:	e9 2a 01 00 00       	jmp    402e36 <init_driver+0x1d7>
  402d0c:	89 c3                	mov    %eax,%ebx
  402d0e:	bf f5 36 40 00       	mov    $0x4036f5,%edi
  402d13:	e8 d8 e0 ff ff       	call   400df0 <gethostbyname@plt>
  402d18:	48 85 c0             	test   %rax,%rax
  402d1b:	75 68                	jne    402d85 <init_driver+0x126>
  402d1d:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  402d24:	3a 20 44 
  402d27:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402d2b:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  402d32:	20 75 6e 
  402d35:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402d39:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402d40:	74 6f 20 
  402d43:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402d47:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  402d4e:	76 65 20 
  402d51:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402d55:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  402d5c:	72 20 61 
  402d5f:	48 89 45 20          	mov    %rax,0x20(%rbp)
  402d63:	c7 45 28 64 64 72 65 	movl   $0x65726464,0x28(%rbp)
  402d6a:	66 c7 45 2c 73 73    	movw   $0x7373,0x2c(%rbp)
  402d70:	c6 45 2e 00          	movb   $0x0,0x2e(%rbp)
  402d74:	89 df                	mov    %ebx,%edi
  402d76:	e8 35 e0 ff ff       	call   400db0 <close@plt>
  402d7b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402d80:	e9 b1 00 00 00       	jmp    402e36 <init_driver+0x1d7>
  402d85:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  402d8c:	00 
  402d8d:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
  402d94:	00 00 
  402d96:	66 c7 04 24 02 00    	movw   $0x2,(%rsp)
  402d9c:	48 63 50 14          	movslq 0x14(%rax),%rdx
  402da0:	48 8b 40 18          	mov    0x18(%rax),%rax
  402da4:	48 8b 30             	mov    (%rax),%rsi
  402da7:	48 8d 7c 24 04       	lea    0x4(%rsp),%rdi
  402dac:	b9 0c 00 00 00       	mov    $0xc,%ecx
  402db1:	e8 4a e0 ff ff       	call   400e00 <__memmove_chk@plt>
  402db6:	66 c7 44 24 02 3c 9a 	movw   $0x9a3c,0x2(%rsp)
  402dbd:	ba 10 00 00 00       	mov    $0x10,%edx
  402dc2:	48 89 e6             	mov    %rsp,%rsi
  402dc5:	89 df                	mov    %ebx,%edi
  402dc7:	e8 14 e1 ff ff       	call   400ee0 <connect@plt>
  402dcc:	85 c0                	test   %eax,%eax
  402dce:	79 50                	jns    402e20 <init_driver+0x1c1>
  402dd0:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
  402dd7:	3a 20 55 
  402dda:	48 89 45 00          	mov    %rax,0x0(%rbp)
  402dde:	48 b8 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rax
  402de5:	20 74 6f 
  402de8:	48 89 45 08          	mov    %rax,0x8(%rbp)
  402dec:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
  402df3:	65 63 74 
  402df6:	48 89 45 10          	mov    %rax,0x10(%rbp)
  402dfa:	48 b8 20 74 6f 20 73 	movabs $0x76726573206f7420,%rax
  402e01:	65 72 76 
  402e04:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402e08:	66 c7 45 20 65 72    	movw   $0x7265,0x20(%rbp)
  402e0e:	c6 45 22 00          	movb   $0x0,0x22(%rbp)
  402e12:	89 df                	mov    %ebx,%edi
  402e14:	e8 97 df ff ff       	call   400db0 <close@plt>
  402e19:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402e1e:	eb 16                	jmp    402e36 <init_driver+0x1d7>
  402e20:	89 df                	mov    %ebx,%edi
  402e22:	e8 89 df ff ff       	call   400db0 <close@plt>
  402e27:	66 c7 45 00 4f 4b    	movw   $0x4b4f,0x0(%rbp)
  402e2d:	c6 45 02 00          	movb   $0x0,0x2(%rbp)
  402e31:	b8 00 00 00 00       	mov    $0x0,%eax
  402e36:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  402e3b:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  402e42:	00 00 
  402e44:	74 05                	je     402e4b <init_driver+0x1ec>
  402e46:	e8 25 df ff ff       	call   400d70 <__stack_chk_fail@plt>
  402e4b:	48 83 c4 28          	add    $0x28,%rsp
  402e4f:	5b                   	pop    %rbx
  402e50:	5d                   	pop    %rbp
  402e51:	c3                   	ret    

0000000000402e52 <driver_post>:
  402e52:	53                   	push   %rbx
  402e53:	4c 89 cb             	mov    %r9,%rbx
  402e56:	45 85 c0             	test   %r8d,%r8d
  402e59:	74 27                	je     402e82 <driver_post+0x30>
  402e5b:	48 89 ca             	mov    %rcx,%rdx
  402e5e:	be 0d 37 40 00       	mov    $0x40370d,%esi
  402e63:	bf 01 00 00 00       	mov    $0x1,%edi
  402e68:	b8 00 00 00 00       	mov    $0x0,%eax
  402e6d:	e8 7e de ff ff       	call   400cf0 <__printf_chk@plt>
  402e72:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  402e77:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  402e7b:	b8 00 00 00 00       	mov    $0x0,%eax
  402e80:	eb 3f                	jmp    402ec1 <driver_post+0x6f>
  402e82:	48 85 ff             	test   %rdi,%rdi
  402e85:	74 2c                	je     402eb3 <driver_post+0x61>
  402e87:	80 3f 00             	cmpb   $0x0,(%rdi)
  402e8a:	74 27                	je     402eb3 <driver_post+0x61>
  402e8c:	48 83 ec 08          	sub    $0x8,%rsp
  402e90:	41 51                	push   %r9
  402e92:	49 89 c9             	mov    %rcx,%r9
  402e95:	49 89 d0             	mov    %rdx,%r8
  402e98:	48 89 f9             	mov    %rdi,%rcx
  402e9b:	48 89 f2             	mov    %rsi,%rdx
  402e9e:	be 9a 3c 00 00       	mov    $0x3c9a,%esi
  402ea3:	bf f5 36 40 00       	mov    $0x4036f5,%edi
  402ea8:	e8 19 f6 ff ff       	call   4024c6 <submitr>
  402ead:	48 83 c4 10          	add    $0x10,%rsp
  402eb1:	eb 0e                	jmp    402ec1 <driver_post+0x6f>
  402eb3:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  402eb8:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  402ebc:	b8 00 00 00 00       	mov    $0x0,%eax
  402ec1:	5b                   	pop    %rbx
  402ec2:	c3                   	ret    

0000000000402ec3 <check>:
  402ec3:	89 f8                	mov    %edi,%eax
  402ec5:	c1 e8 1c             	shr    $0x1c,%eax
  402ec8:	85 c0                	test   %eax,%eax
  402eca:	74 1d                	je     402ee9 <check+0x26>
  402ecc:	b9 00 00 00 00       	mov    $0x0,%ecx
  402ed1:	eb 0b                	jmp    402ede <check+0x1b>
  402ed3:	89 f8                	mov    %edi,%eax
  402ed5:	d3 e8                	shr    %cl,%eax
  402ed7:	3c 0a                	cmp    $0xa,%al
  402ed9:	74 14                	je     402eef <check+0x2c>
  402edb:	83 c1 08             	add    $0x8,%ecx
  402ede:	83 f9 1f             	cmp    $0x1f,%ecx
  402ee1:	7e f0                	jle    402ed3 <check+0x10>
  402ee3:	b8 01 00 00 00       	mov    $0x1,%eax
  402ee8:	c3                   	ret    
  402ee9:	b8 00 00 00 00       	mov    $0x0,%eax
  402eee:	c3                   	ret    
  402eef:	b8 00 00 00 00       	mov    $0x0,%eax
  402ef4:	c3                   	ret    

0000000000402ef5 <gencookie>:
  402ef5:	53                   	push   %rbx
  402ef6:	83 c7 01             	add    $0x1,%edi
  402ef9:	e8 22 de ff ff       	call   400d20 <srandom@plt>
  402efe:	e8 4d df ff ff       	call   400e50 <random@plt>
  402f03:	89 c3                	mov    %eax,%ebx
  402f05:	89 c7                	mov    %eax,%edi
  402f07:	e8 b7 ff ff ff       	call   402ec3 <check>
  402f0c:	85 c0                	test   %eax,%eax
  402f0e:	74 ee                	je     402efe <gencookie+0x9>
  402f10:	89 d8                	mov    %ebx,%eax
  402f12:	5b                   	pop    %rbx
  402f13:	c3                   	ret    
  402f14:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402f1b:	00 00 00 
  402f1e:	66 90                	xchg   %ax,%ax

0000000000402f20 <__libc_csu_init>:
  402f20:	41 57                	push   %r15
  402f22:	41 56                	push   %r14
  402f24:	41 89 ff             	mov    %edi,%r15d
  402f27:	41 55                	push   %r13
  402f29:	41 54                	push   %r12
  402f2b:	4c 8d 25 ce 1e 20 00 	lea    0x201ece(%rip),%r12        # 604e00 <__frame_dummy_init_array_entry>
  402f32:	55                   	push   %rbp
  402f33:	48 8d 2d ce 1e 20 00 	lea    0x201ece(%rip),%rbp        # 604e08 <__do_global_dtors_aux_fini_array_entry>
  402f3a:	53                   	push   %rbx
  402f3b:	49 89 f6             	mov    %rsi,%r14
  402f3e:	49 89 d5             	mov    %rdx,%r13
  402f41:	4c 29 e5             	sub    %r12,%rbp
  402f44:	48 83 ec 08          	sub    $0x8,%rsp
  402f48:	48 c1 fd 03          	sar    $0x3,%rbp
  402f4c:	e8 6f dd ff ff       	call   400cc0 <_init>
  402f51:	48 85 ed             	test   %rbp,%rbp
  402f54:	74 20                	je     402f76 <__libc_csu_init+0x56>
  402f56:	31 db                	xor    %ebx,%ebx
  402f58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  402f5f:	00 
  402f60:	4c 89 ea             	mov    %r13,%rdx
  402f63:	4c 89 f6             	mov    %r14,%rsi
  402f66:	44 89 ff             	mov    %r15d,%edi
  402f69:	41 ff 14 dc          	call   *(%r12,%rbx,8)
  402f6d:	48 83 c3 01          	add    $0x1,%rbx
  402f71:	48 39 eb             	cmp    %rbp,%rbx
  402f74:	75 ea                	jne    402f60 <__libc_csu_init+0x40>
  402f76:	48 83 c4 08          	add    $0x8,%rsp
  402f7a:	5b                   	pop    %rbx
  402f7b:	5d                   	pop    %rbp
  402f7c:	41 5c                	pop    %r12
  402f7e:	41 5d                	pop    %r13
  402f80:	41 5e                	pop    %r14
  402f82:	41 5f                	pop    %r15
  402f84:	c3                   	ret    
  402f85:	90                   	nop
  402f86:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402f8d:	00 00 00 

0000000000402f90 <__libc_csu_fini>:
  402f90:	f3 c3                	repz ret 

Disassembly of section .fini:

0000000000402f94 <_fini>:
  402f94:	48 83 ec 08          	sub    $0x8,%rsp
  402f98:	48 83 c4 08          	add    $0x8,%rsp
  402f9c:	c3                   	ret    
