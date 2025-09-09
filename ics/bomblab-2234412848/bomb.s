bomb:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64 
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 d9 3f 00 00 	mov    0x3fd9(%rip),%rax        # 4fe8 <__gmon_start__@Base>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	call   *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	ret    

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 ca 3e 00 00    	push   0x3eca(%rip)        # 4ef0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	f2 ff 25 cb 3e 00 00 	bnd jmp *0x3ecb(%rip)        # 4ef8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102d:	0f 1f 00             	nopl   (%rax)
    1030:	f3 0f 1e fa          	endbr64 
    1034:	68 00 00 00 00       	push   $0x0
    1039:	f2 e9 e1 ff ff ff    	bnd jmp 1020 <_init+0x20>
    103f:	90                   	nop
    1040:	f3 0f 1e fa          	endbr64 
    1044:	68 01 00 00 00       	push   $0x1
    1049:	f2 e9 d1 ff ff ff    	bnd jmp 1020 <_init+0x20>
    104f:	90                   	nop
    1050:	f3 0f 1e fa          	endbr64 
    1054:	68 02 00 00 00       	push   $0x2
    1059:	f2 e9 c1 ff ff ff    	bnd jmp 1020 <_init+0x20>
    105f:	90                   	nop
    1060:	f3 0f 1e fa          	endbr64 
    1064:	68 03 00 00 00       	push   $0x3
    1069:	f2 e9 b1 ff ff ff    	bnd jmp 1020 <_init+0x20>
    106f:	90                   	nop
    1070:	f3 0f 1e fa          	endbr64 
    1074:	68 04 00 00 00       	push   $0x4
    1079:	f2 e9 a1 ff ff ff    	bnd jmp 1020 <_init+0x20>
    107f:	90                   	nop
    1080:	f3 0f 1e fa          	endbr64 
    1084:	68 05 00 00 00       	push   $0x5
    1089:	f2 e9 91 ff ff ff    	bnd jmp 1020 <_init+0x20>
    108f:	90                   	nop
    1090:	f3 0f 1e fa          	endbr64 
    1094:	68 06 00 00 00       	push   $0x6
    1099:	f2 e9 81 ff ff ff    	bnd jmp 1020 <_init+0x20>
    109f:	90                   	nop
    10a0:	f3 0f 1e fa          	endbr64 
    10a4:	68 07 00 00 00       	push   $0x7
    10a9:	f2 e9 71 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10af:	90                   	nop
    10b0:	f3 0f 1e fa          	endbr64 
    10b4:	68 08 00 00 00       	push   $0x8
    10b9:	f2 e9 61 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10bf:	90                   	nop
    10c0:	f3 0f 1e fa          	endbr64 
    10c4:	68 09 00 00 00       	push   $0x9
    10c9:	f2 e9 51 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10cf:	90                   	nop
    10d0:	f3 0f 1e fa          	endbr64 
    10d4:	68 0a 00 00 00       	push   $0xa
    10d9:	f2 e9 41 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10df:	90                   	nop
    10e0:	f3 0f 1e fa          	endbr64 
    10e4:	68 0b 00 00 00       	push   $0xb
    10e9:	f2 e9 31 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10ef:	90                   	nop
    10f0:	f3 0f 1e fa          	endbr64 
    10f4:	68 0c 00 00 00       	push   $0xc
    10f9:	f2 e9 21 ff ff ff    	bnd jmp 1020 <_init+0x20>
    10ff:	90                   	nop
    1100:	f3 0f 1e fa          	endbr64 
    1104:	68 0d 00 00 00       	push   $0xd
    1109:	f2 e9 11 ff ff ff    	bnd jmp 1020 <_init+0x20>
    110f:	90                   	nop
    1110:	f3 0f 1e fa          	endbr64 
    1114:	68 0e 00 00 00       	push   $0xe
    1119:	f2 e9 01 ff ff ff    	bnd jmp 1020 <_init+0x20>
    111f:	90                   	nop
    1120:	f3 0f 1e fa          	endbr64 
    1124:	68 0f 00 00 00       	push   $0xf
    1129:	f2 e9 f1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    112f:	90                   	nop
    1130:	f3 0f 1e fa          	endbr64 
    1134:	68 10 00 00 00       	push   $0x10
    1139:	f2 e9 e1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    113f:	90                   	nop
    1140:	f3 0f 1e fa          	endbr64 
    1144:	68 11 00 00 00       	push   $0x11
    1149:	f2 e9 d1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    114f:	90                   	nop
    1150:	f3 0f 1e fa          	endbr64 
    1154:	68 12 00 00 00       	push   $0x12
    1159:	f2 e9 c1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    115f:	90                   	nop
    1160:	f3 0f 1e fa          	endbr64 
    1164:	68 13 00 00 00       	push   $0x13
    1169:	f2 e9 b1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    116f:	90                   	nop
    1170:	f3 0f 1e fa          	endbr64 
    1174:	68 14 00 00 00       	push   $0x14
    1179:	f2 e9 a1 fe ff ff    	bnd jmp 1020 <_init+0x20>
    117f:	90                   	nop
    1180:	f3 0f 1e fa          	endbr64 
    1184:	68 15 00 00 00       	push   $0x15
    1189:	f2 e9 91 fe ff ff    	bnd jmp 1020 <_init+0x20>
    118f:	90                   	nop
    1190:	f3 0f 1e fa          	endbr64 
    1194:	68 16 00 00 00       	push   $0x16
    1199:	f2 e9 81 fe ff ff    	bnd jmp 1020 <_init+0x20>
    119f:	90                   	nop
    11a0:	f3 0f 1e fa          	endbr64 
    11a4:	68 17 00 00 00       	push   $0x17
    11a9:	f2 e9 71 fe ff ff    	bnd jmp 1020 <_init+0x20>
    11af:	90                   	nop
    11b0:	f3 0f 1e fa          	endbr64 
    11b4:	68 18 00 00 00       	push   $0x18
    11b9:	f2 e9 61 fe ff ff    	bnd jmp 1020 <_init+0x20>
    11bf:	90                   	nop
    11c0:	f3 0f 1e fa          	endbr64 
    11c4:	68 19 00 00 00       	push   $0x19
    11c9:	f2 e9 51 fe ff ff    	bnd jmp 1020 <_init+0x20>
    11cf:	90                   	nop
    11d0:	f3 0f 1e fa          	endbr64 
    11d4:	68 1a 00 00 00       	push   $0x1a
    11d9:	f2 e9 41 fe ff ff    	bnd jmp 1020 <_init+0x20>
    11df:	90                   	nop

Disassembly of section .plt.got:

00000000000011e0 <__cxa_finalize@plt>:
    11e0:	f3 0f 1e fa          	endbr64 
    11e4:	f2 ff 25 0d 3e 00 00 	bnd jmp *0x3e0d(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    11eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000011f0 <getenv@plt>:
    11f0:	f3 0f 1e fa          	endbr64 
    11f4:	f2 ff 25 05 3d 00 00 	bnd jmp *0x3d05(%rip)        # 4f00 <getenv@GLIBC_2.2.5>
    11fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001200 <__errno_location@plt>:
    1200:	f3 0f 1e fa          	endbr64 
    1204:	f2 ff 25 fd 3c 00 00 	bnd jmp *0x3cfd(%rip)        # 4f08 <__errno_location@GLIBC_2.2.5>
    120b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001210 <strcpy@plt>:
    1210:	f3 0f 1e fa          	endbr64 
    1214:	f2 ff 25 f5 3c 00 00 	bnd jmp *0x3cf5(%rip)        # 4f10 <strcpy@GLIBC_2.2.5>
    121b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001220 <puts@plt>:
    1220:	f3 0f 1e fa          	endbr64 
    1224:	f2 ff 25 ed 3c 00 00 	bnd jmp *0x3ced(%rip)        # 4f18 <puts@GLIBC_2.2.5>
    122b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001230 <write@plt>:
    1230:	f3 0f 1e fa          	endbr64 
    1234:	f2 ff 25 e5 3c 00 00 	bnd jmp *0x3ce5(%rip)        # 4f20 <write@GLIBC_2.2.5>
    123b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001240 <strlen@plt>:
    1240:	f3 0f 1e fa          	endbr64 
    1244:	f2 ff 25 dd 3c 00 00 	bnd jmp *0x3cdd(%rip)        # 4f28 <strlen@GLIBC_2.2.5>
    124b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001250 <__stack_chk_fail@plt>:
    1250:	f3 0f 1e fa          	endbr64 
    1254:	f2 ff 25 d5 3c 00 00 	bnd jmp *0x3cd5(%rip)        # 4f30 <__stack_chk_fail@GLIBC_2.4>
    125b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001260 <alarm@plt>:
    1260:	f3 0f 1e fa          	endbr64 
    1264:	f2 ff 25 cd 3c 00 00 	bnd jmp *0x3ccd(%rip)        # 4f38 <alarm@GLIBC_2.2.5>
    126b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001270 <close@plt>:
    1270:	f3 0f 1e fa          	endbr64 
    1274:	f2 ff 25 c5 3c 00 00 	bnd jmp *0x3cc5(%rip)        # 4f40 <close@GLIBC_2.2.5>
    127b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001280 <read@plt>:
    1280:	f3 0f 1e fa          	endbr64 
    1284:	f2 ff 25 bd 3c 00 00 	bnd jmp *0x3cbd(%rip)        # 4f48 <read@GLIBC_2.2.5>
    128b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001290 <fgets@plt>:
    1290:	f3 0f 1e fa          	endbr64 
    1294:	f2 ff 25 b5 3c 00 00 	bnd jmp *0x3cb5(%rip)        # 4f50 <fgets@GLIBC_2.2.5>
    129b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012a0 <strcmp@plt>:
    12a0:	f3 0f 1e fa          	endbr64 
    12a4:	f2 ff 25 ad 3c 00 00 	bnd jmp *0x3cad(%rip)        # 4f58 <strcmp@GLIBC_2.2.5>
    12ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012b0 <signal@plt>:
    12b0:	f3 0f 1e fa          	endbr64 
    12b4:	f2 ff 25 a5 3c 00 00 	bnd jmp *0x3ca5(%rip)        # 4f60 <signal@GLIBC_2.2.5>
    12bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012c0 <gethostbyname@plt>:
    12c0:	f3 0f 1e fa          	endbr64 
    12c4:	f2 ff 25 9d 3c 00 00 	bnd jmp *0x3c9d(%rip)        # 4f68 <gethostbyname@GLIBC_2.2.5>
    12cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012d0 <__memmove_chk@plt>:
    12d0:	f3 0f 1e fa          	endbr64 
    12d4:	f2 ff 25 95 3c 00 00 	bnd jmp *0x3c95(%rip)        # 4f70 <__memmove_chk@GLIBC_2.3.4>
    12db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012e0 <strtol@plt>:
    12e0:	f3 0f 1e fa          	endbr64 
    12e4:	f2 ff 25 8d 3c 00 00 	bnd jmp *0x3c8d(%rip)        # 4f78 <strtol@GLIBC_2.2.5>
    12eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012f0 <fflush@plt>:
    12f0:	f3 0f 1e fa          	endbr64 
    12f4:	f2 ff 25 85 3c 00 00 	bnd jmp *0x3c85(%rip)        # 4f80 <fflush@GLIBC_2.2.5>
    12fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001300 <__isoc99_sscanf@plt>:
    1300:	f3 0f 1e fa          	endbr64 
    1304:	f2 ff 25 7d 3c 00 00 	bnd jmp *0x3c7d(%rip)        # 4f88 <__isoc99_sscanf@GLIBC_2.7>
    130b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001310 <__printf_chk@plt>:
    1310:	f3 0f 1e fa          	endbr64 
    1314:	f2 ff 25 75 3c 00 00 	bnd jmp *0x3c75(%rip)        # 4f90 <__printf_chk@GLIBC_2.3.4>
    131b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001320 <fopen@plt>:
    1320:	f3 0f 1e fa          	endbr64 
    1324:	f2 ff 25 6d 3c 00 00 	bnd jmp *0x3c6d(%rip)        # 4f98 <fopen@GLIBC_2.2.5>
    132b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001330 <exit@plt>:
    1330:	f3 0f 1e fa          	endbr64 
    1334:	f2 ff 25 65 3c 00 00 	bnd jmp *0x3c65(%rip)        # 4fa0 <exit@GLIBC_2.2.5>
    133b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001340 <connect@plt>:
    1340:	f3 0f 1e fa          	endbr64 
    1344:	f2 ff 25 5d 3c 00 00 	bnd jmp *0x3c5d(%rip)        # 4fa8 <connect@GLIBC_2.2.5>
    134b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001350 <__fprintf_chk@plt>:
    1350:	f3 0f 1e fa          	endbr64 
    1354:	f2 ff 25 55 3c 00 00 	bnd jmp *0x3c55(%rip)        # 4fb0 <__fprintf_chk@GLIBC_2.3.4>
    135b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001360 <sleep@plt>:
    1360:	f3 0f 1e fa          	endbr64 
    1364:	f2 ff 25 4d 3c 00 00 	bnd jmp *0x3c4d(%rip)        # 4fb8 <sleep@GLIBC_2.2.5>
    136b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001370 <__ctype_b_loc@plt>:
    1370:	f3 0f 1e fa          	endbr64 
    1374:	f2 ff 25 45 3c 00 00 	bnd jmp *0x3c45(%rip)        # 4fc0 <__ctype_b_loc@GLIBC_2.3>
    137b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001380 <__sprintf_chk@plt>:
    1380:	f3 0f 1e fa          	endbr64 
    1384:	f2 ff 25 3d 3c 00 00 	bnd jmp *0x3c3d(%rip)        # 4fc8 <__sprintf_chk@GLIBC_2.3.4>
    138b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001390 <socket@plt>:
    1390:	f3 0f 1e fa          	endbr64 
    1394:	f2 ff 25 35 3c 00 00 	bnd jmp *0x3c35(%rip)        # 4fd0 <socket@GLIBC_2.2.5>
    139b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

00000000000013a0 <_start>:
    13a0:	f3 0f 1e fa          	endbr64 
    13a4:	31 ed                	xor    %ebp,%ebp
    13a6:	49 89 d1             	mov    %rdx,%r9
    13a9:	5e                   	pop    %rsi
    13aa:	48 89 e2             	mov    %rsp,%rdx
    13ad:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    13b1:	50                   	push   %rax
    13b2:	54                   	push   %rsp
    13b3:	45 31 c0             	xor    %r8d,%r8d
    13b6:	31 c9                	xor    %ecx,%ecx
    13b8:	48 8d 3d ca 00 00 00 	lea    0xca(%rip),%rdi        # 1489 <main>
    13bf:	ff 15 13 3c 00 00    	call   *0x3c13(%rip)        # 4fd8 <__libc_start_main@GLIBC_2.34>
    13c5:	f4                   	hlt    
    13c6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    13cd:	00 00 00 

00000000000013d0 <deregister_tm_clones>:
    13d0:	48 8d 3d 89 42 00 00 	lea    0x4289(%rip),%rdi        # 5660 <stdout@GLIBC_2.2.5>
    13d7:	48 8d 05 82 42 00 00 	lea    0x4282(%rip),%rax        # 5660 <stdout@GLIBC_2.2.5>
    13de:	48 39 f8             	cmp    %rdi,%rax
    13e1:	74 15                	je     13f8 <deregister_tm_clones+0x28>
    13e3:	48 8b 05 f6 3b 00 00 	mov    0x3bf6(%rip),%rax        # 4fe0 <_ITM_deregisterTMCloneTable@Base>
    13ea:	48 85 c0             	test   %rax,%rax
    13ed:	74 09                	je     13f8 <deregister_tm_clones+0x28>
    13ef:	ff e0                	jmp    *%rax
    13f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    13f8:	c3                   	ret    
    13f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001400 <register_tm_clones>:
    1400:	48 8d 3d 59 42 00 00 	lea    0x4259(%rip),%rdi        # 5660 <stdout@GLIBC_2.2.5>
    1407:	48 8d 35 52 42 00 00 	lea    0x4252(%rip),%rsi        # 5660 <stdout@GLIBC_2.2.5>
    140e:	48 29 fe             	sub    %rdi,%rsi
    1411:	48 89 f0             	mov    %rsi,%rax
    1414:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1418:	48 c1 f8 03          	sar    $0x3,%rax
    141c:	48 01 c6             	add    %rax,%rsi
    141f:	48 d1 fe             	sar    %rsi
    1422:	74 14                	je     1438 <register_tm_clones+0x38>
    1424:	48 8b 05 c5 3b 00 00 	mov    0x3bc5(%rip),%rax        # 4ff0 <_ITM_registerTMCloneTable@Base>
    142b:	48 85 c0             	test   %rax,%rax
    142e:	74 08                	je     1438 <register_tm_clones+0x38>
    1430:	ff e0                	jmp    *%rax
    1432:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1438:	c3                   	ret    
    1439:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001440 <__do_global_dtors_aux>:
    1440:	f3 0f 1e fa          	endbr64 
    1444:	80 3d 3d 42 00 00 00 	cmpb   $0x0,0x423d(%rip)        # 5688 <completed.0>
    144b:	75 2b                	jne    1478 <__do_global_dtors_aux+0x38>
    144d:	55                   	push   %rbp
    144e:	48 83 3d a2 3b 00 00 	cmpq   $0x0,0x3ba2(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    1455:	00 
    1456:	48 89 e5             	mov    %rsp,%rbp
    1459:	74 0c                	je     1467 <__do_global_dtors_aux+0x27>
    145b:	48 8b 3d a6 3b 00 00 	mov    0x3ba6(%rip),%rdi        # 5008 <__dso_handle>
    1462:	e8 79 fd ff ff       	call   11e0 <__cxa_finalize@plt>
    1467:	e8 64 ff ff ff       	call   13d0 <deregister_tm_clones>
    146c:	c6 05 15 42 00 00 01 	movb   $0x1,0x4215(%rip)        # 5688 <completed.0>
    1473:	5d                   	pop    %rbp
    1474:	c3                   	ret    
    1475:	0f 1f 00             	nopl   (%rax)
    1478:	c3                   	ret    
    1479:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001480 <frame_dummy>:
    1480:	f3 0f 1e fa          	endbr64 
    1484:	e9 77 ff ff ff       	jmp    1400 <register_tm_clones>

0000000000001489 <main>:
    1489:	f3 0f 1e fa          	endbr64 
    148d:	53                   	push   %rbx
    148e:	83 ff 01             	cmp    $0x1,%edi
    1491:	0f 84 bc 00 00 00    	je     1553 <main+0xca>
    1497:	48 89 f3             	mov    %rsi,%rbx
    149a:	83 ff 02             	cmp    $0x2,%edi
    149d:	0f 85 e5 00 00 00    	jne    1588 <main+0xff>
    14a3:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
    14a7:	48 8d 35 56 1b 00 00 	lea    0x1b56(%rip),%rsi        # 3004 <_IO_stdin_used+0x4>
    14ae:	e8 6d fe ff ff       	call   1320 <fopen@plt>
    14b3:	48 89 05 d6 41 00 00 	mov    %rax,0x41d6(%rip)        # 5690 <infile>
    14ba:	48 85 c0             	test   %rax,%rax
    14bd:	0f 84 a3 00 00 00    	je     1566 <main+0xdd>
    14c3:	e8 31 06 00 00       	call   1af9 <initialize_bomb>
    14c8:	48 8d 3d 71 1b 00 00 	lea    0x1b71(%rip),%rdi        # 3040 <_IO_stdin_used+0x40>
    14cf:	e8 4c fd ff ff       	call   1220 <puts@plt>
    14d4:	48 8d 3d a5 1b 00 00 	lea    0x1ba5(%rip),%rdi        # 3080 <_IO_stdin_used+0x80>
    14db:	e8 40 fd ff ff       	call   1220 <puts@plt>
    14e0:	e8 65 07 00 00       	call   1c4a <read_line>
    14e5:	48 89 c7             	mov    %rax,%rdi
    14e8:	e8 be 00 00 00       	call   15ab <phase_1>
    14ed:	e8 ac 08 00 00       	call   1d9e <phase_defused>
    14f2:	e8 53 07 00 00       	call   1c4a <read_line>
    14f7:	48 89 c7             	mov    %rax,%rdi
    14fa:	e8 d0 00 00 00       	call   15cf <phase_2>
    14ff:	e8 9a 08 00 00       	call   1d9e <phase_defused>
    1504:	e8 41 07 00 00       	call   1c4a <read_line>
    1509:	48 89 c7             	mov    %rax,%rdi
    150c:	e8 32 01 00 00       	call   1643 <phase_3>
    1511:	e8 88 08 00 00       	call   1d9e <phase_defused>
    1516:	e8 2f 07 00 00       	call   1c4a <read_line>
    151b:	48 89 c7             	mov    %rax,%rdi
    151e:	e8 41 02 00 00       	call   1764 <phase_4>
    1523:	e8 76 08 00 00       	call   1d9e <phase_defused>
    1528:	e8 1d 07 00 00       	call   1c4a <read_line>
    152d:	48 89 c7             	mov    %rax,%rdi
    1530:	e8 a4 02 00 00       	call   17d9 <phase_5>
    1535:	e8 64 08 00 00       	call   1d9e <phase_defused>
    153a:	e8 0b 07 00 00       	call   1c4a <read_line>
    153f:	48 89 c7             	mov    %rax,%rdi
    1542:	e8 de 02 00 00       	call   1825 <phase_6>
    1547:	e8 52 08 00 00       	call   1d9e <phase_defused>
    154c:	b8 00 00 00 00       	mov    $0x0,%eax
    1551:	5b                   	pop    %rbx
    1552:	c3                   	ret    
    1553:	48 8b 05 16 41 00 00 	mov    0x4116(%rip),%rax        # 5670 <stdin@GLIBC_2.2.5>
    155a:	48 89 05 2f 41 00 00 	mov    %rax,0x412f(%rip)        # 5690 <infile>
    1561:	e9 5d ff ff ff       	jmp    14c3 <main+0x3a>
    1566:	48 8b 4b 08          	mov    0x8(%rbx),%rcx
    156a:	48 8b 13             	mov    (%rbx),%rdx
    156d:	48 8d 35 92 1a 00 00 	lea    0x1a92(%rip),%rsi        # 3006 <_IO_stdin_used+0x6>
    1574:	bf 01 00 00 00       	mov    $0x1,%edi
    1579:	e8 92 fd ff ff       	call   1310 <__printf_chk@plt>
    157e:	bf 08 00 00 00       	mov    $0x8,%edi
    1583:	e8 a8 fd ff ff       	call   1330 <exit@plt>
    1588:	48 8b 16             	mov    (%rsi),%rdx
    158b:	48 8d 35 91 1a 00 00 	lea    0x1a91(%rip),%rsi        # 3023 <_IO_stdin_used+0x23>
    1592:	bf 01 00 00 00       	mov    $0x1,%edi
    1597:	b8 00 00 00 00       	mov    $0x0,%eax
    159c:	e8 6f fd ff ff       	call   1310 <__printf_chk@plt>
    15a1:	bf 08 00 00 00       	mov    $0x8,%edi
    15a6:	e8 85 fd ff ff       	call   1330 <exit@plt>

00000000000015ab <phase_1>:
    15ab:	f3 0f 1e fa          	endbr64 
    15af:	48 83 ec 08          	sub    $0x8,%rsp
    15b3:	48 8d 35 f6 1a 00 00 	lea    0x1af6(%rip),%rsi        # 30b0 <_IO_stdin_used+0xb0>
    15ba:	e8 da 04 00 00       	call   1a99 <strings_not_equal>
    15bf:	85 c0                	test   %eax,%eax
    15c1:	75 05                	jne    15c8 <phase_1+0x1d>
    15c3:	48 83 c4 08          	add    $0x8,%rsp
    15c7:	c3                   	ret    
    15c8:	e8 e0 05 00 00       	call   1bad <explode_bomb>
    15cd:	eb f4                	jmp    15c3 <phase_1+0x18>

00000000000015cf <phase_2>:
    15cf:	f3 0f 1e fa          	endbr64 
    15d3:	55                   	push   %rbp
    15d4:	53                   	push   %rbx
    15d5:	48 83 ec 28          	sub    $0x28,%rsp
    15d9:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    15e0:	00 00 
    15e2:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    15e7:	31 c0                	xor    %eax,%eax
    15e9:	48 89 e6             	mov    %rsp,%rsi
    15ec:	e8 14 06 00 00       	call   1c05 <read_six_numbers>
    15f1:	83 3c 24 00          	cmpl   $0x0,(%rsp)
    15f5:	75 07                	jne    15fe <phase_2+0x2f>
    15f7:	83 7c 24 04 01       	cmpl   $0x1,0x4(%rsp)
    15fc:	74 05                	je     1603 <phase_2+0x34>
    15fe:	e8 aa 05 00 00       	call   1bad <explode_bomb>
    1603:	48 89 e3             	mov    %rsp,%rbx
    1606:	48 8d 6c 24 10       	lea    0x10(%rsp),%rbp
    160b:	eb 09                	jmp    1616 <phase_2+0x47>
    160d:	48 83 c3 04          	add    $0x4,%rbx
    1611:	48 39 eb             	cmp    %rbp,%rbx
    1614:	74 11                	je     1627 <phase_2+0x58>
    1616:	8b 43 04             	mov    0x4(%rbx),%eax
    1619:	03 03                	add    (%rbx),%eax
    161b:	39 43 08             	cmp    %eax,0x8(%rbx)
    161e:	74 ed                	je     160d <phase_2+0x3e>
    1620:	e8 88 05 00 00       	call   1bad <explode_bomb>
    1625:	eb e6                	jmp    160d <phase_2+0x3e>
    1627:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    162c:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    1633:	00 00 
    1635:	75 07                	jne    163e <phase_2+0x6f>
    1637:	48 83 c4 28          	add    $0x28,%rsp
    163b:	5b                   	pop    %rbx
    163c:	5d                   	pop    %rbp
    163d:	c3                   	ret    
    163e:	e8 0d fc ff ff       	call   1250 <__stack_chk_fail@plt>

0000000000001643 <phase_3>:
    1643:	f3 0f 1e fa          	endbr64 
    1647:	48 83 ec 18          	sub    $0x18,%rsp
    164b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1652:	00 00 
    1654:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    1659:	31 c0                	xor    %eax,%eax
    165b:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
    1660:	48 89 e2             	mov    %rsp,%rdx
    1663:	48 8d 35 e6 1c 00 00 	lea    0x1ce6(%rip),%rsi        # 3350 <array.0+0x210>
    166a:	e8 91 fc ff ff       	call   1300 <__isoc99_sscanf@plt>
    166f:	83 f8 01             	cmp    $0x1,%eax
    1672:	7e 1e                	jle    1692 <phase_3+0x4f>
    1674:	83 3c 24 07          	cmpl   $0x7,(%rsp)
    1678:	0f 87 9a 00 00 00    	ja     1718 <phase_3+0xd5>
    167e:	8b 04 24             	mov    (%rsp),%eax
    1681:	48 8d 15 98 1a 00 00 	lea    0x1a98(%rip),%rdx        # 3120 <_IO_stdin_used+0x120>
    1688:	48 63 04 82          	movslq (%rdx,%rax,4),%rax
    168c:	48 01 d0             	add    %rdx,%rax
    168f:	3e ff e0             	notrack jmp *%rax
    1692:	e8 16 05 00 00       	call   1bad <explode_bomb>
    1697:	eb db                	jmp    1674 <phase_3+0x31>
    1699:	b8 13 02 00 00       	mov    $0x213,%eax
    169e:	2d 35 01 00 00       	sub    $0x135,%eax
    16a3:	05 6d 02 00 00       	add    $0x26d,%eax
    16a8:	2d ad 02 00 00       	sub    $0x2ad,%eax
    16ad:	05 ad 02 00 00       	add    $0x2ad,%eax
    16b2:	2d ad 02 00 00       	sub    $0x2ad,%eax
    16b7:	05 ad 02 00 00       	add    $0x2ad,%eax
    16bc:	2d ad 02 00 00       	sub    $0x2ad,%eax
    16c1:	83 3c 24 05          	cmpl   $0x5,(%rsp)
    16c5:	7f 06                	jg     16cd <phase_3+0x8a>
    16c7:	39 44 24 04          	cmp    %eax,0x4(%rsp)
    16cb:	74 05                	je     16d2 <phase_3+0x8f>
    16cd:	e8 db 04 00 00       	call   1bad <explode_bomb>
    16d2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    16d7:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    16de:	00 00 
    16e0:	75 42                	jne    1724 <phase_3+0xe1>
    16e2:	48 83 c4 18          	add    $0x18,%rsp
    16e6:	c3                   	ret    
    16e7:	b8 00 00 00 00       	mov    $0x0,%eax
    16ec:	eb b0                	jmp    169e <phase_3+0x5b>
    16ee:	b8 00 00 00 00       	mov    $0x0,%eax
    16f3:	eb ae                	jmp    16a3 <phase_3+0x60>
    16f5:	b8 00 00 00 00       	mov    $0x0,%eax
    16fa:	eb ac                	jmp    16a8 <phase_3+0x65>
    16fc:	b8 00 00 00 00       	mov    $0x0,%eax
    1701:	eb aa                	jmp    16ad <phase_3+0x6a>
    1703:	b8 00 00 00 00       	mov    $0x0,%eax
    1708:	eb a8                	jmp    16b2 <phase_3+0x6f>
    170a:	b8 00 00 00 00       	mov    $0x0,%eax
    170f:	eb a6                	jmp    16b7 <phase_3+0x74>
    1711:	b8 00 00 00 00       	mov    $0x0,%eax
    1716:	eb a4                	jmp    16bc <phase_3+0x79>
    1718:	e8 90 04 00 00       	call   1bad <explode_bomb>
    171d:	b8 00 00 00 00       	mov    $0x0,%eax
    1722:	eb 9d                	jmp    16c1 <phase_3+0x7e>
    1724:	e8 27 fb ff ff       	call   1250 <__stack_chk_fail@plt>

0000000000001729 <func4>:
    1729:	f3 0f 1e fa          	endbr64 
    172d:	b8 00 00 00 00       	mov    $0x0,%eax
    1732:	85 ff                	test   %edi,%edi
    1734:	7e 2d                	jle    1763 <func4+0x3a>
    1736:	41 54                	push   %r12
    1738:	55                   	push   %rbp
    1739:	53                   	push   %rbx
    173a:	89 fb                	mov    %edi,%ebx
    173c:	89 f5                	mov    %esi,%ebp
    173e:	89 f0                	mov    %esi,%eax
    1740:	83 ff 01             	cmp    $0x1,%edi
    1743:	74 19                	je     175e <func4+0x35>
    1745:	8d 7f ff             	lea    -0x1(%rdi),%edi
    1748:	e8 dc ff ff ff       	call   1729 <func4>
    174d:	44 8d 24 28          	lea    (%rax,%rbp,1),%r12d
    1751:	8d 7b fe             	lea    -0x2(%rbx),%edi
    1754:	89 ee                	mov    %ebp,%esi
    1756:	e8 ce ff ff ff       	call   1729 <func4>
    175b:	44 01 e0             	add    %r12d,%eax
    175e:	5b                   	pop    %rbx
    175f:	5d                   	pop    %rbp
    1760:	41 5c                	pop    %r12
    1762:	c3                   	ret    
    1763:	c3                   	ret    

0000000000001764 <phase_4>:
    1764:	f3 0f 1e fa          	endbr64 
    1768:	48 83 ec 18          	sub    $0x18,%rsp
    176c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1773:	00 00 
    1775:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    177a:	31 c0                	xor    %eax,%eax
    177c:	48 89 e1             	mov    %rsp,%rcx
    177f:	48 8d 54 24 04       	lea    0x4(%rsp),%rdx
    1784:	48 8d 35 c5 1b 00 00 	lea    0x1bc5(%rip),%rsi        # 3350 <array.0+0x210>
    178b:	e8 70 fb ff ff       	call   1300 <__isoc99_sscanf@plt>
    1790:	83 f8 02             	cmp    $0x2,%eax
    1793:	75 0b                	jne    17a0 <phase_4+0x3c>
    1795:	8b 04 24             	mov    (%rsp),%eax
    1798:	83 e8 02             	sub    $0x2,%eax
    179b:	83 f8 02             	cmp    $0x2,%eax
    179e:	76 05                	jbe    17a5 <phase_4+0x41>
    17a0:	e8 08 04 00 00       	call   1bad <explode_bomb>
    17a5:	8b 34 24             	mov    (%rsp),%esi
    17a8:	bf 09 00 00 00       	mov    $0x9,%edi
    17ad:	e8 77 ff ff ff       	call   1729 <func4>
    17b2:	39 44 24 04          	cmp    %eax,0x4(%rsp)
    17b6:	75 15                	jne    17cd <phase_4+0x69>
    17b8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    17bd:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    17c4:	00 00 
    17c6:	75 0c                	jne    17d4 <phase_4+0x70>
    17c8:	48 83 c4 18          	add    $0x18,%rsp
    17cc:	c3                   	ret    
    17cd:	e8 db 03 00 00       	call   1bad <explode_bomb>
    17d2:	eb e4                	jmp    17b8 <phase_4+0x54>
    17d4:	e8 77 fa ff ff       	call   1250 <__stack_chk_fail@plt>

00000000000017d9 <phase_5>:
    17d9:	f3 0f 1e fa          	endbr64 
    17dd:	53                   	push   %rbx
    17de:	48 89 fb             	mov    %rdi,%rbx
    17e1:	e8 92 02 00 00       	call   1a78 <string_length>
    17e6:	83 f8 06             	cmp    $0x6,%eax
    17e9:	75 2c                	jne    1817 <phase_5+0x3e>
    17eb:	48 89 d8             	mov    %rbx,%rax
    17ee:	48 8d 7b 06          	lea    0x6(%rbx),%rdi
    17f2:	b9 00 00 00 00       	mov    $0x0,%ecx
    17f7:	48 8d 35 42 19 00 00 	lea    0x1942(%rip),%rsi        # 3140 <array.0>
    17fe:	0f b6 10             	movzbl (%rax),%edx
    1801:	83 e2 0f             	and    $0xf,%edx
    1804:	03 0c 96             	add    (%rsi,%rdx,4),%ecx
    1807:	48 83 c0 01          	add    $0x1,%rax
    180b:	48 39 f8             	cmp    %rdi,%rax
    180e:	75 ee                	jne    17fe <phase_5+0x25>
    1810:	83 f9 34             	cmp    $0x34,%ecx
    1813:	75 09                	jne    181e <phase_5+0x45>
    1815:	5b                   	pop    %rbx
    1816:	c3                   	ret    
    1817:	e8 91 03 00 00       	call   1bad <explode_bomb>
    181c:	eb cd                	jmp    17eb <phase_5+0x12>
    181e:	e8 8a 03 00 00       	call   1bad <explode_bomb>
    1823:	eb f0                	jmp    1815 <phase_5+0x3c>

0000000000001825 <phase_6>:
    1825:	f3 0f 1e fa          	endbr64 
    1829:	41 56                	push   %r14
    182b:	41 55                	push   %r13
    182d:	41 54                	push   %r12
    182f:	55                   	push   %rbp
    1830:	53                   	push   %rbx
    1831:	48 83 ec 60          	sub    $0x60,%rsp
    1835:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    183c:	00 00 
    183e:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    1843:	31 c0                	xor    %eax,%eax
    1845:	49 89 e5             	mov    %rsp,%r13
    1848:	4c 89 ee             	mov    %r13,%rsi
    184b:	e8 b5 03 00 00       	call   1c05 <read_six_numbers>
    1850:	41 be 01 00 00 00    	mov    $0x1,%r14d
    1856:	49 89 e4             	mov    %rsp,%r12
    1859:	eb 28                	jmp    1883 <phase_6+0x5e>
    185b:	e8 4d 03 00 00       	call   1bad <explode_bomb>
    1860:	eb 30                	jmp    1892 <phase_6+0x6d>
    1862:	48 83 c3 01          	add    $0x1,%rbx
    1866:	83 fb 05             	cmp    $0x5,%ebx
    1869:	7f 10                	jg     187b <phase_6+0x56>
    186b:	41 8b 04 9c          	mov    (%r12,%rbx,4),%eax
    186f:	39 45 00             	cmp    %eax,0x0(%rbp)
    1872:	75 ee                	jne    1862 <phase_6+0x3d>
    1874:	e8 34 03 00 00       	call   1bad <explode_bomb>
    1879:	eb e7                	jmp    1862 <phase_6+0x3d>
    187b:	49 83 c6 01          	add    $0x1,%r14
    187f:	49 83 c5 04          	add    $0x4,%r13
    1883:	4c 89 ed             	mov    %r13,%rbp
    1886:	41 8b 45 00          	mov    0x0(%r13),%eax
    188a:	83 e8 01             	sub    $0x1,%eax
    188d:	83 f8 05             	cmp    $0x5,%eax
    1890:	77 c9                	ja     185b <phase_6+0x36>
    1892:	41 83 fe 05          	cmp    $0x5,%r14d
    1896:	7f 05                	jg     189d <phase_6+0x78>
    1898:	4c 89 f3             	mov    %r14,%rbx
    189b:	eb ce                	jmp    186b <phase_6+0x46>
    189d:	be 00 00 00 00       	mov    $0x0,%esi
    18a2:	8b 0c b4             	mov    (%rsp,%rsi,4),%ecx
    18a5:	b8 01 00 00 00       	mov    $0x1,%eax
    18aa:	48 8d 15 5f 39 00 00 	lea    0x395f(%rip),%rdx        # 5210 <node1>
    18b1:	83 f9 01             	cmp    $0x1,%ecx
    18b4:	7e 0b                	jle    18c1 <phase_6+0x9c>
    18b6:	48 8b 52 08          	mov    0x8(%rdx),%rdx
    18ba:	83 c0 01             	add    $0x1,%eax
    18bd:	39 c8                	cmp    %ecx,%eax
    18bf:	75 f5                	jne    18b6 <phase_6+0x91>
    18c1:	48 89 54 f4 20       	mov    %rdx,0x20(%rsp,%rsi,8)
    18c6:	48 83 c6 01          	add    $0x1,%rsi
    18ca:	48 83 fe 06          	cmp    $0x6,%rsi
    18ce:	75 d2                	jne    18a2 <phase_6+0x7d>
    18d0:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
    18d5:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    18da:	48 89 43 08          	mov    %rax,0x8(%rbx)
    18de:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
    18e3:	48 89 50 08          	mov    %rdx,0x8(%rax)
    18e7:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    18ec:	48 89 42 08          	mov    %rax,0x8(%rdx)
    18f0:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    18f5:	48 89 50 08          	mov    %rdx,0x8(%rax)
    18f9:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    18fe:	48 89 42 08          	mov    %rax,0x8(%rdx)
    1902:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
    1909:	00 
    190a:	bd 05 00 00 00       	mov    $0x5,%ebp
    190f:	eb 09                	jmp    191a <phase_6+0xf5>
    1911:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
    1915:	83 ed 01             	sub    $0x1,%ebp
    1918:	74 11                	je     192b <phase_6+0x106>
    191a:	48 8b 43 08          	mov    0x8(%rbx),%rax
    191e:	8b 00                	mov    (%rax),%eax
    1920:	39 03                	cmp    %eax,(%rbx)
    1922:	7e ed                	jle    1911 <phase_6+0xec>
    1924:	e8 84 02 00 00       	call   1bad <explode_bomb>
    1929:	eb e6                	jmp    1911 <phase_6+0xec>
    192b:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    1930:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    1937:	00 00 
    1939:	75 0d                	jne    1948 <phase_6+0x123>
    193b:	48 83 c4 60          	add    $0x60,%rsp
    193f:	5b                   	pop    %rbx
    1940:	5d                   	pop    %rbp
    1941:	41 5c                	pop    %r12
    1943:	41 5d                	pop    %r13
    1945:	41 5e                	pop    %r14
    1947:	c3                   	ret    
    1948:	e8 03 f9 ff ff       	call   1250 <__stack_chk_fail@plt>

000000000000194d <fun7>:
    194d:	f3 0f 1e fa          	endbr64 
    1951:	48 85 ff             	test   %rdi,%rdi
    1954:	74 32                	je     1988 <fun7+0x3b>
    1956:	48 83 ec 08          	sub    $0x8,%rsp
    195a:	8b 17                	mov    (%rdi),%edx
    195c:	39 f2                	cmp    %esi,%edx
    195e:	7f 0c                	jg     196c <fun7+0x1f>
    1960:	b8 00 00 00 00       	mov    $0x0,%eax
    1965:	75 12                	jne    1979 <fun7+0x2c>
    1967:	48 83 c4 08          	add    $0x8,%rsp
    196b:	c3                   	ret    
    196c:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
    1970:	e8 d8 ff ff ff       	call   194d <fun7>
    1975:	01 c0                	add    %eax,%eax
    1977:	eb ee                	jmp    1967 <fun7+0x1a>
    1979:	48 8b 7f 10          	mov    0x10(%rdi),%rdi
    197d:	e8 cb ff ff ff       	call   194d <fun7>
    1982:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax
    1986:	eb df                	jmp    1967 <fun7+0x1a>
    1988:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    198d:	c3                   	ret    

000000000000198e <secret_phase>:
    198e:	f3 0f 1e fa          	endbr64 
    1992:	53                   	push   %rbx
    1993:	e8 b2 02 00 00       	call   1c4a <read_line>
    1998:	48 89 c7             	mov    %rax,%rdi
    199b:	ba 0a 00 00 00       	mov    $0xa,%edx
    19a0:	be 00 00 00 00       	mov    $0x0,%esi
    19a5:	e8 36 f9 ff ff       	call   12e0 <strtol@plt>
    19aa:	89 c3                	mov    %eax,%ebx
    19ac:	83 e8 01             	sub    $0x1,%eax
    19af:	3d e8 03 00 00       	cmp    $0x3e8,%eax
    19b4:	77 25                	ja     19db <secret_phase+0x4d>
    19b6:	89 de                	mov    %ebx,%esi
    19b8:	48 8d 3d 71 37 00 00 	lea    0x3771(%rip),%rdi        # 5130 <n1>
    19bf:	e8 89 ff ff ff       	call   194d <fun7>
    19c4:	85 c0                	test   %eax,%eax
    19c6:	75 1a                	jne    19e2 <secret_phase+0x54>
    19c8:	48 8d 3d 11 17 00 00 	lea    0x1711(%rip),%rdi        # 30e0 <_IO_stdin_used+0xe0>
    19cf:	e8 4c f8 ff ff       	call   1220 <puts@plt>
    19d4:	e8 c5 03 00 00       	call   1d9e <phase_defused>
    19d9:	5b                   	pop    %rbx
    19da:	c3                   	ret    
    19db:	e8 cd 01 00 00       	call   1bad <explode_bomb>
    19e0:	eb d4                	jmp    19b6 <secret_phase+0x28>
    19e2:	e8 c6 01 00 00       	call   1bad <explode_bomb>
    19e7:	eb df                	jmp    19c8 <secret_phase+0x3a>

00000000000019e9 <sig_handler>:
    19e9:	f3 0f 1e fa          	endbr64 
    19ed:	50                   	push   %rax
    19ee:	58                   	pop    %rax
    19ef:	48 83 ec 08          	sub    $0x8,%rsp
    19f3:	48 8d 3d 86 17 00 00 	lea    0x1786(%rip),%rdi        # 3180 <array.0+0x40>
    19fa:	e8 21 f8 ff ff       	call   1220 <puts@plt>
    19ff:	bf 03 00 00 00       	mov    $0x3,%edi
    1a04:	e8 57 f9 ff ff       	call   1360 <sleep@plt>
    1a09:	48 8d 35 0a 19 00 00 	lea    0x190a(%rip),%rsi        # 331a <array.0+0x1da>
    1a10:	bf 01 00 00 00       	mov    $0x1,%edi
    1a15:	b8 00 00 00 00       	mov    $0x0,%eax
    1a1a:	e8 f1 f8 ff ff       	call   1310 <__printf_chk@plt>
    1a1f:	48 8b 3d 3a 3c 00 00 	mov    0x3c3a(%rip),%rdi        # 5660 <stdout@GLIBC_2.2.5>
    1a26:	e8 c5 f8 ff ff       	call   12f0 <fflush@plt>
    1a2b:	bf 01 00 00 00       	mov    $0x1,%edi
    1a30:	e8 2b f9 ff ff       	call   1360 <sleep@plt>
    1a35:	48 8d 3d e6 18 00 00 	lea    0x18e6(%rip),%rdi        # 3322 <array.0+0x1e2>
    1a3c:	e8 df f7 ff ff       	call   1220 <puts@plt>
    1a41:	bf 10 00 00 00       	mov    $0x10,%edi
    1a46:	e8 e5 f8 ff ff       	call   1330 <exit@plt>

0000000000001a4b <invalid_phase>:
    1a4b:	f3 0f 1e fa          	endbr64 
    1a4f:	50                   	push   %rax
    1a50:	58                   	pop    %rax
    1a51:	48 83 ec 08          	sub    $0x8,%rsp
    1a55:	48 89 fa             	mov    %rdi,%rdx
    1a58:	48 8d 35 cb 18 00 00 	lea    0x18cb(%rip),%rsi        # 332a <array.0+0x1ea>
    1a5f:	bf 01 00 00 00       	mov    $0x1,%edi
    1a64:	b8 00 00 00 00       	mov    $0x0,%eax
    1a69:	e8 a2 f8 ff ff       	call   1310 <__printf_chk@plt>
    1a6e:	bf 08 00 00 00       	mov    $0x8,%edi
    1a73:	e8 b8 f8 ff ff       	call   1330 <exit@plt>

0000000000001a78 <string_length>:
    1a78:	f3 0f 1e fa          	endbr64 
    1a7c:	80 3f 00             	cmpb   $0x0,(%rdi)
    1a7f:	74 12                	je     1a93 <string_length+0x1b>
    1a81:	b8 00 00 00 00       	mov    $0x0,%eax
    1a86:	48 83 c7 01          	add    $0x1,%rdi
    1a8a:	83 c0 01             	add    $0x1,%eax
    1a8d:	80 3f 00             	cmpb   $0x0,(%rdi)
    1a90:	75 f4                	jne    1a86 <string_length+0xe>
    1a92:	c3                   	ret    
    1a93:	b8 00 00 00 00       	mov    $0x0,%eax
    1a98:	c3                   	ret    

0000000000001a99 <strings_not_equal>:
    1a99:	f3 0f 1e fa          	endbr64 
    1a9d:	41 54                	push   %r12
    1a9f:	55                   	push   %rbp
    1aa0:	53                   	push   %rbx
    1aa1:	48 89 fb             	mov    %rdi,%rbx
    1aa4:	48 89 f5             	mov    %rsi,%rbp
    1aa7:	e8 cc ff ff ff       	call   1a78 <string_length>
    1aac:	41 89 c4             	mov    %eax,%r12d
    1aaf:	48 89 ef             	mov    %rbp,%rdi
    1ab2:	e8 c1 ff ff ff       	call   1a78 <string_length>
    1ab7:	89 c2                	mov    %eax,%edx
    1ab9:	b8 01 00 00 00       	mov    $0x1,%eax
    1abe:	41 39 d4             	cmp    %edx,%r12d
    1ac1:	75 31                	jne    1af4 <strings_not_equal+0x5b>
    1ac3:	0f b6 13             	movzbl (%rbx),%edx
    1ac6:	84 d2                	test   %dl,%dl
    1ac8:	74 1e                	je     1ae8 <strings_not_equal+0x4f>
    1aca:	b8 00 00 00 00       	mov    $0x0,%eax
    1acf:	38 54 05 00          	cmp    %dl,0x0(%rbp,%rax,1)
    1ad3:	75 1a                	jne    1aef <strings_not_equal+0x56>
    1ad5:	48 83 c0 01          	add    $0x1,%rax
    1ad9:	0f b6 14 03          	movzbl (%rbx,%rax,1),%edx
    1add:	84 d2                	test   %dl,%dl
    1adf:	75 ee                	jne    1acf <strings_not_equal+0x36>
    1ae1:	b8 00 00 00 00       	mov    $0x0,%eax
    1ae6:	eb 0c                	jmp    1af4 <strings_not_equal+0x5b>
    1ae8:	b8 00 00 00 00       	mov    $0x0,%eax
    1aed:	eb 05                	jmp    1af4 <strings_not_equal+0x5b>
    1aef:	b8 01 00 00 00       	mov    $0x1,%eax
    1af4:	5b                   	pop    %rbx
    1af5:	5d                   	pop    %rbp
    1af6:	41 5c                	pop    %r12
    1af8:	c3                   	ret    

0000000000001af9 <initialize_bomb>:
    1af9:	f3 0f 1e fa          	endbr64 
    1afd:	48 83 ec 08          	sub    $0x8,%rsp
    1b01:	48 8d 35 e1 fe ff ff 	lea    -0x11f(%rip),%rsi        # 19e9 <sig_handler>
    1b08:	bf 02 00 00 00       	mov    $0x2,%edi
    1b0d:	e8 9e f7 ff ff       	call   12b0 <signal@plt>
    1b12:	48 83 c4 08          	add    $0x8,%rsp
    1b16:	c3                   	ret    

0000000000001b17 <initialize_bomb_solve>:
    1b17:	f3 0f 1e fa          	endbr64 
    1b1b:	c3                   	ret    

0000000000001b1c <blank_line>:
    1b1c:	f3 0f 1e fa          	endbr64 
    1b20:	55                   	push   %rbp
    1b21:	53                   	push   %rbx
    1b22:	48 83 ec 08          	sub    $0x8,%rsp
    1b26:	48 89 fd             	mov    %rdi,%rbp
    1b29:	0f b6 5d 00          	movzbl 0x0(%rbp),%ebx
    1b2d:	84 db                	test   %bl,%bl
    1b2f:	74 1e                	je     1b4f <blank_line+0x33>
    1b31:	e8 3a f8 ff ff       	call   1370 <__ctype_b_loc@plt>
    1b36:	48 83 c5 01          	add    $0x1,%rbp
    1b3a:	48 0f be db          	movsbq %bl,%rbx
    1b3e:	48 8b 00             	mov    (%rax),%rax
    1b41:	f6 44 58 01 20       	testb  $0x20,0x1(%rax,%rbx,2)
    1b46:	75 e1                	jne    1b29 <blank_line+0xd>
    1b48:	b8 00 00 00 00       	mov    $0x0,%eax
    1b4d:	eb 05                	jmp    1b54 <blank_line+0x38>
    1b4f:	b8 01 00 00 00       	mov    $0x1,%eax
    1b54:	48 83 c4 08          	add    $0x8,%rsp
    1b58:	5b                   	pop    %rbx
    1b59:	5d                   	pop    %rbp
    1b5a:	c3                   	ret    

0000000000001b5b <skip>:
    1b5b:	f3 0f 1e fa          	endbr64 
    1b5f:	55                   	push   %rbp
    1b60:	53                   	push   %rbx
    1b61:	48 83 ec 08          	sub    $0x8,%rsp
    1b65:	48 8d 2d 94 3b 00 00 	lea    0x3b94(%rip),%rbp        # 5700 <input_strings>
    1b6c:	48 63 05 81 3b 00 00 	movslq 0x3b81(%rip),%rax        # 56f4 <num_input_strings>
    1b73:	48 8d 3c 80          	lea    (%rax,%rax,4),%rdi
    1b77:	48 c1 e7 04          	shl    $0x4,%rdi
    1b7b:	48 01 ef             	add    %rbp,%rdi
    1b7e:	48 8b 15 0b 3b 00 00 	mov    0x3b0b(%rip),%rdx        # 5690 <infile>
    1b85:	be 50 00 00 00       	mov    $0x50,%esi
    1b8a:	e8 01 f7 ff ff       	call   1290 <fgets@plt>
    1b8f:	48 89 c3             	mov    %rax,%rbx
    1b92:	48 85 c0             	test   %rax,%rax
    1b95:	74 0c                	je     1ba3 <skip+0x48>
    1b97:	48 89 c7             	mov    %rax,%rdi
    1b9a:	e8 7d ff ff ff       	call   1b1c <blank_line>
    1b9f:	85 c0                	test   %eax,%eax
    1ba1:	75 c9                	jne    1b6c <skip+0x11>
    1ba3:	48 89 d8             	mov    %rbx,%rax
    1ba6:	48 83 c4 08          	add    $0x8,%rsp
    1baa:	5b                   	pop    %rbx
    1bab:	5d                   	pop    %rbp
    1bac:	c3                   	ret    

0000000000001bad <explode_bomb>:
    1bad:	f3 0f 1e fa          	endbr64 
    1bb1:	50                   	push   %rax
    1bb2:	58                   	pop    %rax
    1bb3:	48 83 ec 08          	sub    $0x8,%rsp
    1bb7:	48 8d 3d 7d 17 00 00 	lea    0x177d(%rip),%rdi        # 333b <array.0+0x1fb>
    1bbe:	e8 5d f6 ff ff       	call   1220 <puts@plt>
    1bc3:	8b 15 2b 3b 00 00    	mov    0x3b2b(%rip),%edx        # 56f4 <num_input_strings>
    1bc9:	48 8d 35 e8 15 00 00 	lea    0x15e8(%rip),%rsi        # 31b8 <array.0+0x78>
    1bd0:	bf 01 00 00 00       	mov    $0x1,%edi
    1bd5:	b8 00 00 00 00       	mov    $0x0,%eax
    1bda:	e8 31 f7 ff ff       	call   1310 <__printf_chk@plt>
    1bdf:	8b 15 0b 3b 00 00    	mov    0x3b0b(%rip),%edx        # 56f0 <score>
    1be5:	48 8d 35 f4 15 00 00 	lea    0x15f4(%rip),%rsi        # 31e0 <array.0+0xa0>
    1bec:	bf 01 00 00 00       	mov    $0x1,%edi
    1bf1:	b8 00 00 00 00       	mov    $0x0,%eax
    1bf6:	e8 15 f7 ff ff       	call   1310 <__printf_chk@plt>
    1bfb:	bf 08 00 00 00       	mov    $0x8,%edi
    1c00:	e8 2b f7 ff ff       	call   1330 <exit@plt>

0000000000001c05 <read_six_numbers>:
    1c05:	f3 0f 1e fa          	endbr64 
    1c09:	48 83 ec 08          	sub    $0x8,%rsp
    1c0d:	48 89 f2             	mov    %rsi,%rdx
    1c10:	48 8d 4e 04          	lea    0x4(%rsi),%rcx
    1c14:	48 8d 46 14          	lea    0x14(%rsi),%rax
    1c18:	50                   	push   %rax
    1c19:	48 8d 46 10          	lea    0x10(%rsi),%rax
    1c1d:	50                   	push   %rax
    1c1e:	4c 8d 4e 0c          	lea    0xc(%rsi),%r9
    1c22:	4c 8d 46 08          	lea    0x8(%rsi),%r8
    1c26:	48 8d 35 17 17 00 00 	lea    0x1717(%rip),%rsi        # 3344 <array.0+0x204>
    1c2d:	b8 00 00 00 00       	mov    $0x0,%eax
    1c32:	e8 c9 f6 ff ff       	call   1300 <__isoc99_sscanf@plt>
    1c37:	48 83 c4 10          	add    $0x10,%rsp
    1c3b:	83 f8 05             	cmp    $0x5,%eax
    1c3e:	7e 05                	jle    1c45 <read_six_numbers+0x40>
    1c40:	48 83 c4 08          	add    $0x8,%rsp
    1c44:	c3                   	ret    
    1c45:	e8 63 ff ff ff       	call   1bad <explode_bomb>

0000000000001c4a <read_line>:
    1c4a:	f3 0f 1e fa          	endbr64 
    1c4e:	55                   	push   %rbp
    1c4f:	53                   	push   %rbx
    1c50:	48 83 ec 08          	sub    $0x8,%rsp
    1c54:	b8 00 00 00 00       	mov    $0x0,%eax
    1c59:	e8 fd fe ff ff       	call   1b5b <skip>
    1c5e:	48 85 c0             	test   %rax,%rax
    1c61:	74 5d                	je     1cc0 <read_line+0x76>
    1c63:	8b 2d 8b 3a 00 00    	mov    0x3a8b(%rip),%ebp        # 56f4 <num_input_strings>
    1c69:	48 63 c5             	movslq %ebp,%rax
    1c6c:	48 8d 1c 80          	lea    (%rax,%rax,4),%rbx
    1c70:	48 c1 e3 04          	shl    $0x4,%rbx
    1c74:	48 8d 05 85 3a 00 00 	lea    0x3a85(%rip),%rax        # 5700 <input_strings>
    1c7b:	48 01 c3             	add    %rax,%rbx
    1c7e:	48 89 df             	mov    %rbx,%rdi
    1c81:	e8 ba f5 ff ff       	call   1240 <strlen@plt>
    1c86:	83 f8 4e             	cmp    $0x4e,%eax
    1c89:	0f 8f c5 00 00 00    	jg     1d54 <read_line+0x10a>
    1c8f:	83 e8 01             	sub    $0x1,%eax
    1c92:	48 98                	cltq   
    1c94:	48 63 d5             	movslq %ebp,%rdx
    1c97:	48 8d 0c 92          	lea    (%rdx,%rdx,4),%rcx
    1c9b:	48 c1 e1 04          	shl    $0x4,%rcx
    1c9f:	48 8d 15 5a 3a 00 00 	lea    0x3a5a(%rip),%rdx        # 5700 <input_strings>
    1ca6:	48 01 ca             	add    %rcx,%rdx
    1ca9:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
    1cad:	83 c5 01             	add    $0x1,%ebp
    1cb0:	89 2d 3e 3a 00 00    	mov    %ebp,0x3a3e(%rip)        # 56f4 <num_input_strings>
    1cb6:	48 89 d8             	mov    %rbx,%rax
    1cb9:	48 83 c4 08          	add    $0x8,%rsp
    1cbd:	5b                   	pop    %rbx
    1cbe:	5d                   	pop    %rbp
    1cbf:	c3                   	ret    
    1cc0:	48 8b 05 a9 39 00 00 	mov    0x39a9(%rip),%rax        # 5670 <stdin@GLIBC_2.2.5>
    1cc7:	48 39 05 c2 39 00 00 	cmp    %rax,0x39c2(%rip)        # 5690 <infile>
    1cce:	74 1b                	je     1ceb <read_line+0xa1>
    1cd0:	48 8d 3d 9d 16 00 00 	lea    0x169d(%rip),%rdi        # 3374 <array.0+0x234>
    1cd7:	e8 14 f5 ff ff       	call   11f0 <getenv@plt>
    1cdc:	48 85 c0             	test   %rax,%rax
    1cdf:	74 3c                	je     1d1d <read_line+0xd3>
    1ce1:	bf 00 00 00 00       	mov    $0x0,%edi
    1ce6:	e8 45 f6 ff ff       	call   1330 <exit@plt>
    1ceb:	48 8d 3d 64 16 00 00 	lea    0x1664(%rip),%rdi        # 3356 <array.0+0x216>
    1cf2:	e8 29 f5 ff ff       	call   1220 <puts@plt>
    1cf7:	8b 15 f3 39 00 00    	mov    0x39f3(%rip),%edx        # 56f0 <score>
    1cfd:	48 8d 35 0c 15 00 00 	lea    0x150c(%rip),%rsi        # 3210 <array.0+0xd0>
    1d04:	bf 01 00 00 00       	mov    $0x1,%edi
    1d09:	b8 00 00 00 00       	mov    $0x0,%eax
    1d0e:	e8 fd f5 ff ff       	call   1310 <__printf_chk@plt>
    1d13:	bf 08 00 00 00       	mov    $0x8,%edi
    1d18:	e8 13 f6 ff ff       	call   1330 <exit@plt>
    1d1d:	48 8b 05 4c 39 00 00 	mov    0x394c(%rip),%rax        # 5670 <stdin@GLIBC_2.2.5>
    1d24:	48 89 05 65 39 00 00 	mov    %rax,0x3965(%rip)        # 5690 <infile>
    1d2b:	b8 00 00 00 00       	mov    $0x0,%eax
    1d30:	e8 26 fe ff ff       	call   1b5b <skip>
    1d35:	48 85 c0             	test   %rax,%rax
    1d38:	0f 85 25 ff ff ff    	jne    1c63 <read_line+0x19>
    1d3e:	48 8d 3d 11 16 00 00 	lea    0x1611(%rip),%rdi        # 3356 <array.0+0x216>
    1d45:	e8 d6 f4 ff ff       	call   1220 <puts@plt>
    1d4a:	bf 00 00 00 00       	mov    $0x0,%edi
    1d4f:	e8 dc f5 ff ff       	call   1330 <exit@plt>
    1d54:	48 8d 3d 24 16 00 00 	lea    0x1624(%rip),%rdi        # 337f <array.0+0x23f>
    1d5b:	e8 c0 f4 ff ff       	call   1220 <puts@plt>
    1d60:	8b 05 8e 39 00 00    	mov    0x398e(%rip),%eax        # 56f4 <num_input_strings>
    1d66:	8d 50 01             	lea    0x1(%rax),%edx
    1d69:	89 15 85 39 00 00    	mov    %edx,0x3985(%rip)        # 56f4 <num_input_strings>
    1d6f:	48 98                	cltq   
    1d71:	48 6b c0 50          	imul   $0x50,%rax,%rax
    1d75:	48 8d 15 84 39 00 00 	lea    0x3984(%rip),%rdx        # 5700 <input_strings>
    1d7c:	48 be 2a 2a 2a 74 72 	movabs $0x636e7572742a2a2a,%rsi
    1d83:	75 6e 63 
    1d86:	48 bf 61 74 65 64 2a 	movabs $0x2a2a2a64657461,%rdi
    1d8d:	2a 2a 00 
    1d90:	48 89 34 02          	mov    %rsi,(%rdx,%rax,1)
    1d94:	48 89 7c 02 08       	mov    %rdi,0x8(%rdx,%rax,1)
    1d99:	e8 0f fe ff ff       	call   1bad <explode_bomb>

0000000000001d9e <phase_defused>:
    1d9e:	f3 0f 1e fa          	endbr64 
    1da2:	48 83 ec 78          	sub    $0x78,%rsp
    1da6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    1dad:	00 00 
    1daf:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    1db4:	31 c0                	xor    %eax,%eax
    1db6:	8b 15 38 39 00 00    	mov    0x3938(%rip),%edx        # 56f4 <num_input_strings>
    1dbc:	83 fa 01             	cmp    $0x1,%edx
    1dbf:	74 18                	je     1dd9 <phase_defused+0x3b>
    1dc1:	83 fa 06             	cmp    $0x6,%edx
    1dc4:	77 1a                	ja     1de0 <phase_defused+0x42>
    1dc6:	89 d0                	mov    %edx,%eax
    1dc8:	48 8d 0d 6d 16 00 00 	lea    0x166d(%rip),%rcx        # 343c <array.0+0x2fc>
    1dcf:	48 63 04 81          	movslq (%rcx,%rax,4),%rax
    1dd3:	48 01 c8             	add    %rcx,%rax
    1dd6:	3e ff e0             	notrack jmp *%rax
    1dd9:	83 05 10 39 00 00 0a 	addl   $0xa,0x3910(%rip)        # 56f0 <score>
    1de0:	48 8d 35 b3 15 00 00 	lea    0x15b3(%rip),%rsi        # 339a <array.0+0x25a>
    1de7:	bf 01 00 00 00       	mov    $0x1,%edi
    1dec:	b8 00 00 00 00       	mov    $0x0,%eax
    1df1:	e8 1a f5 ff ff       	call   1310 <__printf_chk@plt>
    1df6:	8b 15 f4 38 00 00    	mov    0x38f4(%rip),%edx        # 56f0 <score>
    1dfc:	48 8d 35 0d 14 00 00 	lea    0x140d(%rip),%rsi        # 3210 <array.0+0xd0>
    1e03:	bf 01 00 00 00       	mov    $0x1,%edi
    1e08:	b8 00 00 00 00       	mov    $0x0,%eax
    1e0d:	e8 fe f4 ff ff       	call   1310 <__printf_chk@plt>
    1e12:	83 3d db 38 00 00 06 	cmpl   $0x6,0x38db(%rip)        # 56f4 <num_input_strings>
    1e19:	74 49                	je     1e64 <phase_defused+0xc6>
    1e1b:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
    1e20:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    1e27:	00 00 
    1e29:	0f 85 c3 00 00 00    	jne    1ef2 <phase_defused+0x154>
    1e2f:	48 83 c4 78          	add    $0x78,%rsp
    1e33:	c3                   	ret    
    1e34:	83 05 b5 38 00 00 0f 	addl   $0xf,0x38b5(%rip)        # 56f0 <score>
    1e3b:	eb a3                	jmp    1de0 <phase_defused+0x42>
    1e3d:	83 05 ac 38 00 00 14 	addl   $0x14,0x38ac(%rip)        # 56f0 <score>
    1e44:	eb 9a                	jmp    1de0 <phase_defused+0x42>
    1e46:	83 05 a3 38 00 00 1e 	addl   $0x1e,0x38a3(%rip)        # 56f0 <score>
    1e4d:	eb 91                	jmp    1de0 <phase_defused+0x42>
    1e4f:	83 05 9a 38 00 00 0f 	addl   $0xf,0x389a(%rip)        # 56f0 <score>
    1e56:	eb 88                	jmp    1de0 <phase_defused+0x42>
    1e58:	83 05 91 38 00 00 0a 	addl   $0xa,0x3891(%rip)        # 56f0 <score>
    1e5f:	e9 7c ff ff ff       	jmp    1de0 <phase_defused+0x42>
    1e64:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
    1e69:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
    1e6e:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
    1e73:	48 8d 35 3e 15 00 00 	lea    0x153e(%rip),%rsi        # 33b8 <array.0+0x278>
    1e7a:	48 8d 3d 6f 39 00 00 	lea    0x396f(%rip),%rdi        # 57f0 <input_strings+0xf0>
    1e81:	b8 00 00 00 00       	mov    $0x0,%eax
    1e86:	e8 75 f4 ff ff       	call   1300 <__isoc99_sscanf@plt>
    1e8b:	83 f8 03             	cmp    $0x3,%eax
    1e8e:	74 1d                	je     1ead <phase_defused+0x10f>
    1e90:	48 8d 3d 59 14 00 00 	lea    0x1459(%rip),%rdi        # 32f0 <array.0+0x1b0>
    1e97:	e8 84 f3 ff ff       	call   1220 <puts@plt>
    1e9c:	48 8d 3d 25 15 00 00 	lea    0x1525(%rip),%rdi        # 33c8 <array.0+0x288>
    1ea3:	e8 78 f3 ff ff       	call   1220 <puts@plt>
    1ea8:	e9 6e ff ff ff       	jmp    1e1b <phase_defused+0x7d>
    1ead:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
    1eb2:	48 8d 35 08 15 00 00 	lea    0x1508(%rip),%rsi        # 33c1 <array.0+0x281>
    1eb9:	e8 db fb ff ff       	call   1a99 <strings_not_equal>
    1ebe:	85 c0                	test   %eax,%eax
    1ec0:	75 ce                	jne    1e90 <phase_defused+0xf2>
    1ec2:	48 8d 3d 6f 13 00 00 	lea    0x136f(%rip),%rdi        # 3238 <array.0+0xf8>
    1ec9:	e8 52 f3 ff ff       	call   1220 <puts@plt>
    1ece:	48 8d 3d 8b 13 00 00 	lea    0x138b(%rip),%rdi        # 3260 <array.0+0x120>
    1ed5:	e8 46 f3 ff ff       	call   1220 <puts@plt>
    1eda:	b8 00 00 00 00       	mov    $0x0,%eax
    1edf:	e8 aa fa ff ff       	call   198e <secret_phase>
    1ee4:	48 8d 3d ad 13 00 00 	lea    0x13ad(%rip),%rdi        # 3298 <array.0+0x158>
    1eeb:	e8 30 f3 ff ff       	call   1220 <puts@plt>
    1ef0:	eb 9e                	jmp    1e90 <phase_defused+0xf2>
    1ef2:	e8 59 f3 ff ff       	call   1250 <__stack_chk_fail@plt>

0000000000001ef7 <sigalrm_handler>:
    1ef7:	f3 0f 1e fa          	endbr64 
    1efb:	50                   	push   %rax
    1efc:	58                   	pop    %rax
    1efd:	48 83 ec 08          	sub    $0x8,%rsp
    1f01:	b9 00 00 00 00       	mov    $0x0,%ecx
    1f06:	48 8d 15 4b 15 00 00 	lea    0x154b(%rip),%rdx        # 3458 <array.0+0x318>
    1f0d:	be 01 00 00 00       	mov    $0x1,%esi
    1f12:	48 8b 3d 67 37 00 00 	mov    0x3767(%rip),%rdi        # 5680 <stderr@GLIBC_2.2.5>
    1f19:	b8 00 00 00 00       	mov    $0x0,%eax
    1f1e:	e8 2d f4 ff ff       	call   1350 <__fprintf_chk@plt>
    1f23:	bf 01 00 00 00       	mov    $0x1,%edi
    1f28:	e8 03 f4 ff ff       	call   1330 <exit@plt>

0000000000001f2d <rio_readlineb>:
    1f2d:	41 56                	push   %r14
    1f2f:	41 55                	push   %r13
    1f31:	41 54                	push   %r12
    1f33:	55                   	push   %rbp
    1f34:	53                   	push   %rbx
    1f35:	49 89 f4             	mov    %rsi,%r12
    1f38:	48 83 fa 01          	cmp    $0x1,%rdx
    1f3c:	0f 86 92 00 00 00    	jbe    1fd4 <rio_readlineb+0xa7>
    1f42:	48 89 fb             	mov    %rdi,%rbx
    1f45:	4c 8d 74 16 ff       	lea    -0x1(%rsi,%rdx,1),%r14
    1f4a:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    1f50:	48 8d 6f 10          	lea    0x10(%rdi),%rbp
    1f54:	eb 56                	jmp    1fac <rio_readlineb+0x7f>
    1f56:	e8 a5 f2 ff ff       	call   1200 <__errno_location@plt>
    1f5b:	83 38 04             	cmpl   $0x4,(%rax)
    1f5e:	75 55                	jne    1fb5 <rio_readlineb+0x88>
    1f60:	ba 00 20 00 00       	mov    $0x2000,%edx
    1f65:	48 89 ee             	mov    %rbp,%rsi
    1f68:	8b 3b                	mov    (%rbx),%edi
    1f6a:	e8 11 f3 ff ff       	call   1280 <read@plt>
    1f6f:	89 c2                	mov    %eax,%edx
    1f71:	89 43 04             	mov    %eax,0x4(%rbx)
    1f74:	85 c0                	test   %eax,%eax
    1f76:	78 de                	js     1f56 <rio_readlineb+0x29>
    1f78:	85 c0                	test   %eax,%eax
    1f7a:	74 42                	je     1fbe <rio_readlineb+0x91>
    1f7c:	48 89 6b 08          	mov    %rbp,0x8(%rbx)
    1f80:	48 8b 43 08          	mov    0x8(%rbx),%rax
    1f84:	0f b6 08             	movzbl (%rax),%ecx
    1f87:	48 83 c0 01          	add    $0x1,%rax
    1f8b:	48 89 43 08          	mov    %rax,0x8(%rbx)
    1f8f:	83 ea 01             	sub    $0x1,%edx
    1f92:	89 53 04             	mov    %edx,0x4(%rbx)
    1f95:	49 83 c4 01          	add    $0x1,%r12
    1f99:	41 88 4c 24 ff       	mov    %cl,-0x1(%r12)
    1f9e:	80 f9 0a             	cmp    $0xa,%cl
    1fa1:	74 3c                	je     1fdf <rio_readlineb+0xb2>
    1fa3:	41 83 c5 01          	add    $0x1,%r13d
    1fa7:	4d 39 f4             	cmp    %r14,%r12
    1faa:	74 30                	je     1fdc <rio_readlineb+0xaf>
    1fac:	8b 53 04             	mov    0x4(%rbx),%edx
    1faf:	85 d2                	test   %edx,%edx
    1fb1:	7e ad                	jle    1f60 <rio_readlineb+0x33>
    1fb3:	eb cb                	jmp    1f80 <rio_readlineb+0x53>
    1fb5:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
    1fbc:	eb 05                	jmp    1fc3 <rio_readlineb+0x96>
    1fbe:	b8 00 00 00 00       	mov    $0x0,%eax
    1fc3:	85 c0                	test   %eax,%eax
    1fc5:	75 29                	jne    1ff0 <rio_readlineb+0xc3>
    1fc7:	b8 00 00 00 00       	mov    $0x0,%eax
    1fcc:	41 83 fd 01          	cmp    $0x1,%r13d
    1fd0:	75 0d                	jne    1fdf <rio_readlineb+0xb2>
    1fd2:	eb 13                	jmp    1fe7 <rio_readlineb+0xba>
    1fd4:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    1fda:	eb 03                	jmp    1fdf <rio_readlineb+0xb2>
    1fdc:	4d 89 f4             	mov    %r14,%r12
    1fdf:	41 c6 04 24 00       	movb   $0x0,(%r12)
    1fe4:	49 63 c5             	movslq %r13d,%rax
    1fe7:	5b                   	pop    %rbx
    1fe8:	5d                   	pop    %rbp
    1fe9:	41 5c                	pop    %r12
    1feb:	41 5d                	pop    %r13
    1fed:	41 5e                	pop    %r14
    1fef:	c3                   	ret    
    1ff0:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
    1ff7:	eb ee                	jmp    1fe7 <rio_readlineb+0xba>

0000000000001ff9 <submitr>:
    1ff9:	f3 0f 1e fa          	endbr64 
    1ffd:	41 57                	push   %r15
    1fff:	41 56                	push   %r14
    2001:	41 55                	push   %r13
    2003:	41 54                	push   %r12
    2005:	55                   	push   %rbp
    2006:	53                   	push   %rbx
    2007:	4c 8d 9c 24 00 60 ff 	lea    -0xa000(%rsp),%r11
    200e:	ff 
    200f:	48 81 ec 00 10 00 00 	sub    $0x1000,%rsp
    2016:	48 83 0c 24 00       	orq    $0x0,(%rsp)
    201b:	4c 39 dc             	cmp    %r11,%rsp
    201e:	75 ef                	jne    200f <submitr+0x16>
    2020:	48 83 ec 78          	sub    $0x78,%rsp
    2024:	49 89 fd             	mov    %rdi,%r13
    2027:	89 f5                	mov    %esi,%ebp
    2029:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    202e:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    2033:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    2038:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
    203d:	48 8b 9c 24 b0 a0 00 	mov    0xa0b0(%rsp),%rbx
    2044:	00 
    2045:	4c 8b bc 24 b8 a0 00 	mov    0xa0b8(%rsp),%r15
    204c:	00 
    204d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2054:	00 00 
    2056:	48 89 84 24 68 a0 00 	mov    %rax,0xa068(%rsp)
    205d:	00 
    205e:	31 c0                	xor    %eax,%eax
    2060:	c7 44 24 3c 00 00 00 	movl   $0x0,0x3c(%rsp)
    2067:	00 
    2068:	ba 00 00 00 00       	mov    $0x0,%edx
    206d:	be 01 00 00 00       	mov    $0x1,%esi
    2072:	bf 02 00 00 00       	mov    $0x2,%edi
    2077:	e8 14 f3 ff ff       	call   1390 <socket@plt>
    207c:	85 c0                	test   %eax,%eax
    207e:	0f 88 12 01 00 00    	js     2196 <submitr+0x19d>
    2084:	41 89 c4             	mov    %eax,%r12d
    2087:	4c 89 ef             	mov    %r13,%rdi
    208a:	e8 31 f2 ff ff       	call   12c0 <gethostbyname@plt>
    208f:	48 85 c0             	test   %rax,%rax
    2092:	0f 84 4e 01 00 00    	je     21e6 <submitr+0x1ed>
    2098:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
    209d:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
    20a4:	00 00 
    20a6:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    20ad:	00 00 
    20af:	66 c7 44 24 40 02 00 	movw   $0x2,0x40(%rsp)
    20b6:	48 63 50 14          	movslq 0x14(%rax),%rdx
    20ba:	48 8b 40 18          	mov    0x18(%rax),%rax
    20be:	48 8d 7c 24 44       	lea    0x44(%rsp),%rdi
    20c3:	b9 0c 00 00 00       	mov    $0xc,%ecx
    20c8:	48 8b 30             	mov    (%rax),%rsi
    20cb:	e8 00 f2 ff ff       	call   12d0 <__memmove_chk@plt>
    20d0:	66 c1 c5 08          	rol    $0x8,%bp
    20d4:	66 89 6c 24 42       	mov    %bp,0x42(%rsp)
    20d9:	ba 10 00 00 00       	mov    $0x10,%edx
    20de:	4c 89 ee             	mov    %r13,%rsi
    20e1:	44 89 e7             	mov    %r12d,%edi
    20e4:	e8 57 f2 ff ff       	call   1340 <connect@plt>
    20e9:	85 c0                	test   %eax,%eax
    20eb:	0f 88 60 01 00 00    	js     2251 <submitr+0x258>
    20f1:	48 89 df             	mov    %rbx,%rdi
    20f4:	e8 47 f1 ff ff       	call   1240 <strlen@plt>
    20f9:	48 89 c5             	mov    %rax,%rbp
    20fc:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    2101:	e8 3a f1 ff ff       	call   1240 <strlen@plt>
    2106:	49 89 c6             	mov    %rax,%r14
    2109:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    210e:	e8 2d f1 ff ff       	call   1240 <strlen@plt>
    2113:	49 89 c5             	mov    %rax,%r13
    2116:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    211b:	e8 20 f1 ff ff       	call   1240 <strlen@plt>
    2120:	48 89 c2             	mov    %rax,%rdx
    2123:	4b 8d 84 2e 80 00 00 	lea    0x80(%r14,%r13,1),%rax
    212a:	00 
    212b:	48 01 d0             	add    %rdx,%rax
    212e:	48 8d 54 6d 00       	lea    0x0(%rbp,%rbp,2),%rdx
    2133:	48 01 d0             	add    %rdx,%rax
    2136:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
    213c:	0f 87 6c 01 00 00    	ja     22ae <submitr+0x2b5>
    2142:	48 8d 94 24 60 40 00 	lea    0x4060(%rsp),%rdx
    2149:	00 
    214a:	b9 00 04 00 00       	mov    $0x400,%ecx
    214f:	b8 00 00 00 00       	mov    $0x0,%eax
    2154:	48 89 d7             	mov    %rdx,%rdi
    2157:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    215a:	48 89 df             	mov    %rbx,%rdi
    215d:	e8 de f0 ff ff       	call   1240 <strlen@plt>
    2162:	85 c0                	test   %eax,%eax
    2164:	0f 84 07 05 00 00    	je     2671 <submitr+0x678>
    216a:	8d 40 ff             	lea    -0x1(%rax),%eax
    216d:	4c 8d 6c 03 01       	lea    0x1(%rbx,%rax,1),%r13
    2172:	48 8d ac 24 60 40 00 	lea    0x4060(%rsp),%rbp
    2179:	00 
    217a:	48 8d 84 24 60 80 00 	lea    0x8060(%rsp),%rax
    2181:	00 
    2182:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    2187:	49 be d9 ff 00 00 00 	movabs $0x2000000000ffd9,%r14
    218e:	00 20 00 
    2191:	e9 a6 01 00 00       	jmp    233c <submitr+0x343>
    2196:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    219d:	3a 20 43 
    21a0:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    21a7:	20 75 6e 
    21aa:	49 89 07             	mov    %rax,(%r15)
    21ad:	49 89 57 08          	mov    %rdx,0x8(%r15)
    21b1:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    21b8:	74 6f 20 
    21bb:	48 ba 63 72 65 61 74 	movabs $0x7320657461657263,%rdx
    21c2:	65 20 73 
    21c5:	49 89 47 10          	mov    %rax,0x10(%r15)
    21c9:	49 89 57 18          	mov    %rdx,0x18(%r15)
    21cd:	41 c7 47 20 6f 63 6b 	movl   $0x656b636f,0x20(%r15)
    21d4:	65 
    21d5:	66 41 c7 47 24 74 00 	movw   $0x74,0x24(%r15)
    21dc:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    21e1:	e9 03 03 00 00       	jmp    24e9 <submitr+0x4f0>
    21e6:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
    21ed:	3a 20 44 
    21f0:	48 ba 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rdx
    21f7:	20 75 6e 
    21fa:	49 89 07             	mov    %rax,(%r15)
    21fd:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2201:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2208:	74 6f 20 
    220b:	48 ba 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rdx
    2212:	76 65 20 
    2215:	49 89 47 10          	mov    %rax,0x10(%r15)
    2219:	49 89 57 18          	mov    %rdx,0x18(%r15)
    221d:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
    2224:	72 20 61 
    2227:	49 89 47 20          	mov    %rax,0x20(%r15)
    222b:	41 c7 47 28 64 64 72 	movl   $0x65726464,0x28(%r15)
    2232:	65 
    2233:	66 41 c7 47 2c 73 73 	movw   $0x7373,0x2c(%r15)
    223a:	41 c6 47 2e 00       	movb   $0x0,0x2e(%r15)
    223f:	44 89 e7             	mov    %r12d,%edi
    2242:	e8 29 f0 ff ff       	call   1270 <close@plt>
    2247:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    224c:	e9 98 02 00 00       	jmp    24e9 <submitr+0x4f0>
    2251:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
    2258:	3a 20 55 
    225b:	48 ba 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rdx
    2262:	20 74 6f 
    2265:	49 89 07             	mov    %rax,(%r15)
    2268:	49 89 57 08          	mov    %rdx,0x8(%r15)
    226c:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
    2273:	65 63 74 
    2276:	48 ba 20 74 6f 20 74 	movabs $0x20656874206f7420,%rdx
    227d:	68 65 20 
    2280:	49 89 47 10          	mov    %rax,0x10(%r15)
    2284:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2288:	41 c7 47 20 73 65 72 	movl   $0x76726573,0x20(%r15)
    228f:	76 
    2290:	66 41 c7 47 24 65 72 	movw   $0x7265,0x24(%r15)
    2297:	41 c6 47 26 00       	movb   $0x0,0x26(%r15)
    229c:	44 89 e7             	mov    %r12d,%edi
    229f:	e8 cc ef ff ff       	call   1270 <close@plt>
    22a4:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    22a9:	e9 3b 02 00 00       	jmp    24e9 <submitr+0x4f0>
    22ae:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
    22b5:	3a 20 52 
    22b8:	48 ba 65 73 75 6c 74 	movabs $0x747320746c757365,%rdx
    22bf:	20 73 74 
    22c2:	49 89 07             	mov    %rax,(%r15)
    22c5:	49 89 57 08          	mov    %rdx,0x8(%r15)
    22c9:	48 b8 72 69 6e 67 20 	movabs $0x6f6f7420676e6972,%rax
    22d0:	74 6f 6f 
    22d3:	48 ba 20 6c 61 72 67 	movabs $0x202e656772616c20,%rdx
    22da:	65 2e 20 
    22dd:	49 89 47 10          	mov    %rax,0x10(%r15)
    22e1:	49 89 57 18          	mov    %rdx,0x18(%r15)
    22e5:	48 b8 49 6e 63 72 65 	movabs $0x6573616572636e49,%rax
    22ec:	61 73 65 
    22ef:	48 ba 20 53 55 42 4d 	movabs $0x5254494d42555320,%rdx
    22f6:	49 54 52 
    22f9:	49 89 47 20          	mov    %rax,0x20(%r15)
    22fd:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2301:	48 b8 5f 4d 41 58 42 	movabs $0x46554258414d5f,%rax
    2308:	55 46 00 
    230b:	49 89 47 30          	mov    %rax,0x30(%r15)
    230f:	44 89 e7             	mov    %r12d,%edi
    2312:	e8 59 ef ff ff       	call   1270 <close@plt>
    2317:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    231c:	e9 c8 01 00 00       	jmp    24e9 <submitr+0x4f0>
    2321:	49 0f a3 c6          	bt     %rax,%r14
    2325:	73 21                	jae    2348 <submitr+0x34f>
    2327:	44 88 45 00          	mov    %r8b,0x0(%rbp)
    232b:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
    232f:	48 83 c3 01          	add    $0x1,%rbx
    2333:	4c 39 eb             	cmp    %r13,%rbx
    2336:	0f 84 35 03 00 00    	je     2671 <submitr+0x678>
    233c:	44 0f b6 03          	movzbl (%rbx),%r8d
    2340:	41 8d 40 d6          	lea    -0x2a(%r8),%eax
    2344:	3c 35                	cmp    $0x35,%al
    2346:	76 d9                	jbe    2321 <submitr+0x328>
    2348:	44 89 c0             	mov    %r8d,%eax
    234b:	83 e0 df             	and    $0xffffffdf,%eax
    234e:	83 e8 41             	sub    $0x41,%eax
    2351:	3c 19                	cmp    $0x19,%al
    2353:	76 d2                	jbe    2327 <submitr+0x32e>
    2355:	41 80 f8 20          	cmp    $0x20,%r8b
    2359:	74 60                	je     23bb <submitr+0x3c2>
    235b:	41 8d 40 e0          	lea    -0x20(%r8),%eax
    235f:	3c 5f                	cmp    $0x5f,%al
    2361:	76 0a                	jbe    236d <submitr+0x374>
    2363:	41 80 f8 09          	cmp    $0x9,%r8b
    2367:	0f 85 77 02 00 00    	jne    25e4 <submitr+0x5eb>
    236d:	45 0f b6 c0          	movzbl %r8b,%r8d
    2371:	48 8d 0d b6 11 00 00 	lea    0x11b6(%rip),%rcx        # 352e <array.0+0x3ee>
    2378:	ba 08 00 00 00       	mov    $0x8,%edx
    237d:	be 01 00 00 00       	mov    $0x1,%esi
    2382:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2387:	b8 00 00 00 00       	mov    $0x0,%eax
    238c:	e8 ef ef ff ff       	call   1380 <__sprintf_chk@plt>
    2391:	0f b6 84 24 60 80 00 	movzbl 0x8060(%rsp),%eax
    2398:	00 
    2399:	88 45 00             	mov    %al,0x0(%rbp)
    239c:	0f b6 84 24 61 80 00 	movzbl 0x8061(%rsp),%eax
    23a3:	00 
    23a4:	88 45 01             	mov    %al,0x1(%rbp)
    23a7:	0f b6 84 24 62 80 00 	movzbl 0x8062(%rsp),%eax
    23ae:	00 
    23af:	88 45 02             	mov    %al,0x2(%rbp)
    23b2:	48 8d 6d 03          	lea    0x3(%rbp),%rbp
    23b6:	e9 74 ff ff ff       	jmp    232f <submitr+0x336>
    23bb:	c6 45 00 2b          	movb   $0x2b,0x0(%rbp)
    23bf:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
    23c3:	e9 67 ff ff ff       	jmp    232f <submitr+0x336>
    23c8:	48 01 c5             	add    %rax,%rbp
    23cb:	48 29 c3             	sub    %rax,%rbx
    23ce:	0f 84 08 03 00 00    	je     26dc <submitr+0x6e3>
    23d4:	48 89 da             	mov    %rbx,%rdx
    23d7:	48 89 ee             	mov    %rbp,%rsi
    23da:	44 89 e7             	mov    %r12d,%edi
    23dd:	e8 4e ee ff ff       	call   1230 <write@plt>
    23e2:	48 85 c0             	test   %rax,%rax
    23e5:	7f e1                	jg     23c8 <submitr+0x3cf>
    23e7:	e8 14 ee ff ff       	call   1200 <__errno_location@plt>
    23ec:	83 38 04             	cmpl   $0x4,(%rax)
    23ef:	0f 85 90 01 00 00    	jne    2585 <submitr+0x58c>
    23f5:	4c 89 e8             	mov    %r13,%rax
    23f8:	eb ce                	jmp    23c8 <submitr+0x3cf>
    23fa:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2401:	3a 20 43 
    2404:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    240b:	20 75 6e 
    240e:	49 89 07             	mov    %rax,(%r15)
    2411:	49 89 57 08          	mov    %rdx,0x8(%r15)
    2415:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    241c:	74 6f 20 
    241f:	48 ba 72 65 61 64 20 	movabs $0x7269662064616572,%rdx
    2426:	66 69 72 
    2429:	49 89 47 10          	mov    %rax,0x10(%r15)
    242d:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2431:	48 b8 73 74 20 68 65 	movabs $0x6564616568207473,%rax
    2438:	61 64 65 
    243b:	48 ba 72 20 66 72 6f 	movabs $0x73206d6f72662072,%rdx
    2442:	6d 20 73 
    2445:	49 89 47 20          	mov    %rax,0x20(%r15)
    2449:	49 89 57 28          	mov    %rdx,0x28(%r15)
    244d:	41 c7 47 30 65 72 76 	movl   $0x65767265,0x30(%r15)
    2454:	65 
    2455:	66 41 c7 47 34 72 00 	movw   $0x72,0x34(%r15)
    245c:	44 89 e7             	mov    %r12d,%edi
    245f:	e8 0c ee ff ff       	call   1270 <close@plt>
    2464:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2469:	eb 7e                	jmp    24e9 <submitr+0x4f0>
    246b:	4c 8d 8c 24 60 80 00 	lea    0x8060(%rsp),%r9
    2472:	00 
    2473:	48 8d 0d 06 10 00 00 	lea    0x1006(%rip),%rcx        # 3480 <array.0+0x340>
    247a:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
    2481:	be 01 00 00 00       	mov    $0x1,%esi
    2486:	4c 89 ff             	mov    %r15,%rdi
    2489:	b8 00 00 00 00       	mov    $0x0,%eax
    248e:	e8 ed ee ff ff       	call   1380 <__sprintf_chk@plt>
    2493:	44 89 e7             	mov    %r12d,%edi
    2496:	e8 d5 ed ff ff       	call   1270 <close@plt>
    249b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    24a0:	eb 47                	jmp    24e9 <submitr+0x4f0>
    24a2:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    24a9:	00 
    24aa:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
    24af:	ba 00 20 00 00       	mov    $0x2000,%edx
    24b4:	e8 74 fa ff ff       	call   1f2d <rio_readlineb>
    24b9:	48 85 c0             	test   %rax,%rax
    24bc:	7e 54                	jle    2512 <submitr+0x519>
    24be:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    24c5:	00 
    24c6:	4c 89 ff             	mov    %r15,%rdi
    24c9:	e8 42 ed ff ff       	call   1210 <strcpy@plt>
    24ce:	44 89 e7             	mov    %r12d,%edi
    24d1:	e8 9a ed ff ff       	call   1270 <close@plt>
    24d6:	48 8d 35 6c 10 00 00 	lea    0x106c(%rip),%rsi        # 3549 <array.0+0x409>
    24dd:	4c 89 ff             	mov    %r15,%rdi
    24e0:	e8 bb ed ff ff       	call   12a0 <strcmp@plt>
    24e5:	f7 d8                	neg    %eax
    24e7:	19 c0                	sbb    %eax,%eax
    24e9:	48 8b 94 24 68 a0 00 	mov    0xa068(%rsp),%rdx
    24f0:	00 
    24f1:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    24f8:	00 00 
    24fa:	0f 85 f8 02 00 00    	jne    27f8 <submitr+0x7ff>
    2500:	48 81 c4 78 a0 00 00 	add    $0xa078,%rsp
    2507:	5b                   	pop    %rbx
    2508:	5d                   	pop    %rbp
    2509:	41 5c                	pop    %r12
    250b:	41 5d                	pop    %r13
    250d:	41 5e                	pop    %r14
    250f:	41 5f                	pop    %r15
    2511:	c3                   	ret    
    2512:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2519:	3a 20 43 
    251c:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2523:	20 75 6e 
    2526:	49 89 07             	mov    %rax,(%r15)
    2529:	49 89 57 08          	mov    %rdx,0x8(%r15)
    252d:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2534:	74 6f 20 
    2537:	48 ba 72 65 61 64 20 	movabs $0x6174732064616572,%rdx
    253e:	73 74 61 
    2541:	49 89 47 10          	mov    %rax,0x10(%r15)
    2545:	49 89 57 18          	mov    %rdx,0x18(%r15)
    2549:	48 b8 74 75 73 20 6d 	movabs $0x7373656d20737574,%rax
    2550:	65 73 73 
    2553:	48 ba 61 67 65 20 66 	movabs $0x6d6f726620656761,%rdx
    255a:	72 6f 6d 
    255d:	49 89 47 20          	mov    %rax,0x20(%r15)
    2561:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2565:	48 b8 20 73 65 72 76 	movabs $0x72657672657320,%rax
    256c:	65 72 00 
    256f:	49 89 47 30          	mov    %rax,0x30(%r15)
    2573:	44 89 e7             	mov    %r12d,%edi
    2576:	e8 f5 ec ff ff       	call   1270 <close@plt>
    257b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2580:	e9 64 ff ff ff       	jmp    24e9 <submitr+0x4f0>
    2585:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    258c:	3a 20 43 
    258f:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2596:	20 75 6e 
    2599:	49 89 07             	mov    %rax,(%r15)
    259c:	49 89 57 08          	mov    %rdx,0x8(%r15)
    25a0:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    25a7:	74 6f 20 
    25aa:	48 ba 77 72 69 74 65 	movabs $0x6f74206574697277,%rdx
    25b1:	20 74 6f 
    25b4:	49 89 47 10          	mov    %rax,0x10(%r15)
    25b8:	49 89 57 18          	mov    %rdx,0x18(%r15)
    25bc:	48 b8 20 74 68 65 20 	movabs $0x7265732065687420,%rax
    25c3:	73 65 72 
    25c6:	49 89 47 20          	mov    %rax,0x20(%r15)
    25ca:	41 c7 47 28 76 65 72 	movl   $0x726576,0x28(%r15)
    25d1:	00 
    25d2:	44 89 e7             	mov    %r12d,%edi
    25d5:	e8 96 ec ff ff       	call   1270 <close@plt>
    25da:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    25df:	e9 05 ff ff ff       	jmp    24e9 <submitr+0x4f0>
    25e4:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
    25eb:	3a 20 52 
    25ee:	48 ba 65 73 75 6c 74 	movabs $0x747320746c757365,%rdx
    25f5:	20 73 74 
    25f8:	49 89 07             	mov    %rax,(%r15)
    25fb:	49 89 57 08          	mov    %rdx,0x8(%r15)
    25ff:	48 b8 72 69 6e 67 20 	movabs $0x6e6f6320676e6972,%rax
    2606:	63 6f 6e 
    2609:	48 ba 74 61 69 6e 73 	movabs $0x6e6120736e696174,%rdx
    2610:	20 61 6e 
    2613:	49 89 47 10          	mov    %rax,0x10(%r15)
    2617:	49 89 57 18          	mov    %rdx,0x18(%r15)
    261b:	48 b8 20 69 6c 6c 65 	movabs $0x6c6167656c6c6920,%rax
    2622:	67 61 6c 
    2625:	48 ba 20 6f 72 20 75 	movabs $0x72706e7520726f20,%rdx
    262c:	6e 70 72 
    262f:	49 89 47 20          	mov    %rax,0x20(%r15)
    2633:	49 89 57 28          	mov    %rdx,0x28(%r15)
    2637:	48 b8 69 6e 74 61 62 	movabs $0x20656c6261746e69,%rax
    263e:	6c 65 20 
    2641:	48 ba 63 68 61 72 61 	movabs $0x6574636172616863,%rdx
    2648:	63 74 65 
    264b:	49 89 47 30          	mov    %rax,0x30(%r15)
    264f:	49 89 57 38          	mov    %rdx,0x38(%r15)
    2653:	66 41 c7 47 40 72 2e 	movw   $0x2e72,0x40(%r15)
    265a:	41 c6 47 42 00       	movb   $0x0,0x42(%r15)
    265f:	44 89 e7             	mov    %r12d,%edi
    2662:	e8 09 ec ff ff       	call   1270 <close@plt>
    2667:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    266c:	e9 78 fe ff ff       	jmp    24e9 <submitr+0x4f0>
    2671:	48 8d 9c 24 60 20 00 	lea    0x2060(%rsp),%rbx
    2678:	00 
    2679:	48 83 ec 08          	sub    $0x8,%rsp
    267d:	48 8d 84 24 68 40 00 	lea    0x4068(%rsp),%rax
    2684:	00 
    2685:	50                   	push   %rax
    2686:	ff 74 24 28          	push   0x28(%rsp)
    268a:	ff 74 24 38          	push   0x38(%rsp)
    268e:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
    2693:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    2698:	48 8d 0d 11 0e 00 00 	lea    0xe11(%rip),%rcx        # 34b0 <array.0+0x370>
    269f:	ba 00 20 00 00       	mov    $0x2000,%edx
    26a4:	be 01 00 00 00       	mov    $0x1,%esi
    26a9:	48 89 df             	mov    %rbx,%rdi
    26ac:	b8 00 00 00 00       	mov    $0x0,%eax
    26b1:	e8 ca ec ff ff       	call   1380 <__sprintf_chk@plt>
    26b6:	48 83 c4 20          	add    $0x20,%rsp
    26ba:	48 89 df             	mov    %rbx,%rdi
    26bd:	e8 7e eb ff ff       	call   1240 <strlen@plt>
    26c2:	48 89 c3             	mov    %rax,%rbx
    26c5:	48 8d ac 24 60 20 00 	lea    0x2060(%rsp),%rbp
    26cc:	00 
    26cd:	41 bd 00 00 00 00    	mov    $0x0,%r13d
    26d3:	48 85 c0             	test   %rax,%rax
    26d6:	0f 85 f8 fc ff ff    	jne    23d4 <submitr+0x3db>
    26dc:	44 89 64 24 50       	mov    %r12d,0x50(%rsp)
    26e1:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
    26e8:	00 
    26e9:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
    26ee:	48 8d 44 24 60       	lea    0x60(%rsp),%rax
    26f3:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    26f8:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    26ff:	00 
    2700:	ba 00 20 00 00       	mov    $0x2000,%edx
    2705:	e8 23 f8 ff ff       	call   1f2d <rio_readlineb>
    270a:	48 85 c0             	test   %rax,%rax
    270d:	0f 8e e7 fc ff ff    	jle    23fa <submitr+0x401>
    2713:	48 8d 4c 24 3c       	lea    0x3c(%rsp),%rcx
    2718:	48 8d 94 24 60 60 00 	lea    0x6060(%rsp),%rdx
    271f:	00 
    2720:	48 8d bc 24 60 20 00 	lea    0x2060(%rsp),%rdi
    2727:	00 
    2728:	4c 8d 84 24 60 80 00 	lea    0x8060(%rsp),%r8
    272f:	00 
    2730:	48 8d 35 fe 0d 00 00 	lea    0xdfe(%rip),%rsi        # 3535 <array.0+0x3f5>
    2737:	b8 00 00 00 00       	mov    $0x0,%eax
    273c:	e8 bf eb ff ff       	call   1300 <__isoc99_sscanf@plt>
    2741:	44 8b 44 24 3c       	mov    0x3c(%rsp),%r8d
    2746:	41 81 f8 c8 00 00 00 	cmp    $0xc8,%r8d
    274d:	0f 85 18 fd ff ff    	jne    246b <submitr+0x472>
    2753:	48 8d 1d ec 0d 00 00 	lea    0xdec(%rip),%rbx        # 3546 <array.0+0x406>
    275a:	48 8d bc 24 60 20 00 	lea    0x2060(%rsp),%rdi
    2761:	00 
    2762:	48 89 de             	mov    %rbx,%rsi
    2765:	e8 36 eb ff ff       	call   12a0 <strcmp@plt>
    276a:	85 c0                	test   %eax,%eax
    276c:	0f 84 30 fd ff ff    	je     24a2 <submitr+0x4a9>
    2772:	48 8d b4 24 60 20 00 	lea    0x2060(%rsp),%rsi
    2779:	00 
    277a:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
    277f:	ba 00 20 00 00       	mov    $0x2000,%edx
    2784:	e8 a4 f7 ff ff       	call   1f2d <rio_readlineb>
    2789:	48 85 c0             	test   %rax,%rax
    278c:	7f cc                	jg     275a <submitr+0x761>
    278e:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2795:	3a 20 43 
    2798:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    279f:	20 75 6e 
    27a2:	49 89 07             	mov    %rax,(%r15)
    27a5:	49 89 57 08          	mov    %rdx,0x8(%r15)
    27a9:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    27b0:	74 6f 20 
    27b3:	48 ba 72 65 61 64 20 	movabs $0x6165682064616572,%rdx
    27ba:	68 65 61 
    27bd:	49 89 47 10          	mov    %rax,0x10(%r15)
    27c1:	49 89 57 18          	mov    %rdx,0x18(%r15)
    27c5:	48 b8 64 65 72 73 20 	movabs $0x6f72662073726564,%rax
    27cc:	66 72 6f 
    27cf:	48 ba 6d 20 73 65 72 	movabs $0x726576726573206d,%rdx
    27d6:	76 65 72 
    27d9:	49 89 47 20          	mov    %rax,0x20(%r15)
    27dd:	49 89 57 28          	mov    %rdx,0x28(%r15)
    27e1:	41 c6 47 30 00       	movb   $0x0,0x30(%r15)
    27e6:	44 89 e7             	mov    %r12d,%edi
    27e9:	e8 82 ea ff ff       	call   1270 <close@plt>
    27ee:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    27f3:	e9 f1 fc ff ff       	jmp    24e9 <submitr+0x4f0>
    27f8:	e8 53 ea ff ff       	call   1250 <__stack_chk_fail@plt>

00000000000027fd <init_timeout>:
    27fd:	f3 0f 1e fa          	endbr64 
    2801:	85 ff                	test   %edi,%edi
    2803:	75 01                	jne    2806 <init_timeout+0x9>
    2805:	c3                   	ret    
    2806:	53                   	push   %rbx
    2807:	89 fb                	mov    %edi,%ebx
    2809:	48 8d 35 e7 f6 ff ff 	lea    -0x919(%rip),%rsi        # 1ef7 <sigalrm_handler>
    2810:	bf 0e 00 00 00       	mov    $0xe,%edi
    2815:	e8 96 ea ff ff       	call   12b0 <signal@plt>
    281a:	85 db                	test   %ebx,%ebx
    281c:	b8 00 00 00 00       	mov    $0x0,%eax
    2821:	0f 49 c3             	cmovns %ebx,%eax
    2824:	89 c7                	mov    %eax,%edi
    2826:	e8 35 ea ff ff       	call   1260 <alarm@plt>
    282b:	5b                   	pop    %rbx
    282c:	c3                   	ret    

000000000000282d <init_driver>:
    282d:	f3 0f 1e fa          	endbr64 
    2831:	41 54                	push   %r12
    2833:	55                   	push   %rbp
    2834:	53                   	push   %rbx
    2835:	48 83 ec 20          	sub    $0x20,%rsp
    2839:	48 89 fd             	mov    %rdi,%rbp
    283c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2843:	00 00 
    2845:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    284a:	31 c0                	xor    %eax,%eax
    284c:	be 01 00 00 00       	mov    $0x1,%esi
    2851:	bf 0d 00 00 00       	mov    $0xd,%edi
    2856:	e8 55 ea ff ff       	call   12b0 <signal@plt>
    285b:	be 01 00 00 00       	mov    $0x1,%esi
    2860:	bf 1d 00 00 00       	mov    $0x1d,%edi
    2865:	e8 46 ea ff ff       	call   12b0 <signal@plt>
    286a:	be 01 00 00 00       	mov    $0x1,%esi
    286f:	bf 1d 00 00 00       	mov    $0x1d,%edi
    2874:	e8 37 ea ff ff       	call   12b0 <signal@plt>
    2879:	ba 00 00 00 00       	mov    $0x0,%edx
    287e:	be 01 00 00 00       	mov    $0x1,%esi
    2883:	bf 02 00 00 00       	mov    $0x2,%edi
    2888:	e8 03 eb ff ff       	call   1390 <socket@plt>
    288d:	85 c0                	test   %eax,%eax
    288f:	0f 88 9c 00 00 00    	js     2931 <init_driver+0x104>
    2895:	89 c3                	mov    %eax,%ebx
    2897:	48 8d 3d ae 0c 00 00 	lea    0xcae(%rip),%rdi        # 354c <array.0+0x40c>
    289e:	e8 1d ea ff ff       	call   12c0 <gethostbyname@plt>
    28a3:	48 85 c0             	test   %rax,%rax
    28a6:	0f 84 d1 00 00 00    	je     297d <init_driver+0x150>
    28ac:	49 89 e4             	mov    %rsp,%r12
    28af:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    28b6:	00 
    28b7:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    28be:	00 00 
    28c0:	66 c7 04 24 02 00    	movw   $0x2,(%rsp)
    28c6:	48 63 50 14          	movslq 0x14(%rax),%rdx
    28ca:	48 8b 40 18          	mov    0x18(%rax),%rax
    28ce:	48 8d 7c 24 04       	lea    0x4(%rsp),%rdi
    28d3:	b9 0c 00 00 00       	mov    $0xc,%ecx
    28d8:	48 8b 30             	mov    (%rax),%rsi
    28db:	e8 f0 e9 ff ff       	call   12d0 <__memmove_chk@plt>
    28e0:	66 c7 44 24 02 3b 6e 	movw   $0x6e3b,0x2(%rsp)
    28e7:	ba 10 00 00 00       	mov    $0x10,%edx
    28ec:	4c 89 e6             	mov    %r12,%rsi
    28ef:	89 df                	mov    %ebx,%edi
    28f1:	e8 4a ea ff ff       	call   1340 <connect@plt>
    28f6:	85 c0                	test   %eax,%eax
    28f8:	0f 88 e7 00 00 00    	js     29e5 <init_driver+0x1b8>
    28fe:	89 df                	mov    %ebx,%edi
    2900:	e8 6b e9 ff ff       	call   1270 <close@plt>
    2905:	66 c7 45 00 4f 4b    	movw   $0x4b4f,0x0(%rbp)
    290b:	c6 45 02 00          	movb   $0x0,0x2(%rbp)
    290f:	b8 00 00 00 00       	mov    $0x0,%eax
    2914:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    2919:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    2920:	00 00 
    2922:	0f 85 f5 00 00 00    	jne    2a1d <init_driver+0x1f0>
    2928:	48 83 c4 20          	add    $0x20,%rsp
    292c:	5b                   	pop    %rbx
    292d:	5d                   	pop    %rbp
    292e:	41 5c                	pop    %r12
    2930:	c3                   	ret    
    2931:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
    2938:	3a 20 43 
    293b:	48 ba 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rdx
    2942:	20 75 6e 
    2945:	48 89 45 00          	mov    %rax,0x0(%rbp)
    2949:	48 89 55 08          	mov    %rdx,0x8(%rbp)
    294d:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    2954:	74 6f 20 
    2957:	48 ba 63 72 65 61 74 	movabs $0x7320657461657263,%rdx
    295e:	65 20 73 
    2961:	48 89 45 10          	mov    %rax,0x10(%rbp)
    2965:	48 89 55 18          	mov    %rdx,0x18(%rbp)
    2969:	c7 45 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbp)
    2970:	66 c7 45 24 74 00    	movw   $0x74,0x24(%rbp)
    2976:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    297b:	eb 97                	jmp    2914 <init_driver+0xe7>
    297d:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
    2984:	3a 20 44 
    2987:	48 ba 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rdx
    298e:	20 75 6e 
    2991:	48 89 45 00          	mov    %rax,0x0(%rbp)
    2995:	48 89 55 08          	mov    %rdx,0x8(%rbp)
    2999:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
    29a0:	74 6f 20 
    29a3:	48 ba 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rdx
    29aa:	76 65 20 
    29ad:	48 89 45 10          	mov    %rax,0x10(%rbp)
    29b1:	48 89 55 18          	mov    %rdx,0x18(%rbp)
    29b5:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
    29bc:	72 20 61 
    29bf:	48 89 45 20          	mov    %rax,0x20(%rbp)
    29c3:	c7 45 28 64 64 72 65 	movl   $0x65726464,0x28(%rbp)
    29ca:	66 c7 45 2c 73 73    	movw   $0x7373,0x2c(%rbp)
    29d0:	c6 45 2e 00          	movb   $0x0,0x2e(%rbp)
    29d4:	89 df                	mov    %ebx,%edi
    29d6:	e8 95 e8 ff ff       	call   1270 <close@plt>
    29db:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    29e0:	e9 2f ff ff ff       	jmp    2914 <init_driver+0xe7>
    29e5:	4c 8d 05 60 0b 00 00 	lea    0xb60(%rip),%r8        # 354c <array.0+0x40c>
    29ec:	48 8d 0d 15 0b 00 00 	lea    0xb15(%rip),%rcx        # 3508 <array.0+0x3c8>
    29f3:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
    29fa:	be 01 00 00 00       	mov    $0x1,%esi
    29ff:	48 89 ef             	mov    %rbp,%rdi
    2a02:	b8 00 00 00 00       	mov    $0x0,%eax
    2a07:	e8 74 e9 ff ff       	call   1380 <__sprintf_chk@plt>
    2a0c:	89 df                	mov    %ebx,%edi
    2a0e:	e8 5d e8 ff ff       	call   1270 <close@plt>
    2a13:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2a18:	e9 f7 fe ff ff       	jmp    2914 <init_driver+0xe7>
    2a1d:	e8 2e e8 ff ff       	call   1250 <__stack_chk_fail@plt>

0000000000002a22 <driver_post>:
    2a22:	f3 0f 1e fa          	endbr64 
    2a26:	53                   	push   %rbx
    2a27:	4c 89 c3             	mov    %r8,%rbx
    2a2a:	85 c9                	test   %ecx,%ecx
    2a2c:	75 17                	jne    2a45 <driver_post+0x23>
    2a2e:	48 85 ff             	test   %rdi,%rdi
    2a31:	74 05                	je     2a38 <driver_post+0x16>
    2a33:	80 3f 00             	cmpb   $0x0,(%rdi)
    2a36:	75 33                	jne    2a6b <driver_post+0x49>
    2a38:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
    2a3d:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
    2a41:	89 c8                	mov    %ecx,%eax
    2a43:	5b                   	pop    %rbx
    2a44:	c3                   	ret    
    2a45:	48 8d 35 18 0b 00 00 	lea    0xb18(%rip),%rsi        # 3564 <array.0+0x424>
    2a4c:	bf 01 00 00 00       	mov    $0x1,%edi
    2a51:	b8 00 00 00 00       	mov    $0x0,%eax
    2a56:	e8 b5 e8 ff ff       	call   1310 <__printf_chk@plt>
    2a5b:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
    2a60:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
    2a64:	b8 00 00 00 00       	mov    $0x0,%eax
    2a69:	eb d8                	jmp    2a43 <driver_post+0x21>
    2a6b:	41 50                	push   %r8
    2a6d:	52                   	push   %rdx
    2a6e:	4c 8d 0d 06 0b 00 00 	lea    0xb06(%rip),%r9        # 357b <array.0+0x43b>
    2a75:	49 89 f0             	mov    %rsi,%r8
    2a78:	48 89 f9             	mov    %rdi,%rcx
    2a7b:	48 8d 15 01 0b 00 00 	lea    0xb01(%rip),%rdx        # 3583 <array.0+0x443>
    2a82:	be 6e 3b 00 00       	mov    $0x3b6e,%esi
    2a87:	48 8d 3d be 0a 00 00 	lea    0xabe(%rip),%rdi        # 354c <array.0+0x40c>
    2a8e:	e8 66 f5 ff ff       	call   1ff9 <submitr>
    2a93:	48 83 c4 10          	add    $0x10,%rsp
    2a97:	eb aa                	jmp    2a43 <driver_post+0x21>

Disassembly of section .fini:

0000000000002a9c <_fini>:
    2a9c:	f3 0f 1e fa          	endbr64 
    2aa0:	48 83 ec 08          	sub    $0x8,%rsp
    2aa4:	48 83 c4 08          	add    $0x8,%rsp
    2aa8:	c3                   	ret    
