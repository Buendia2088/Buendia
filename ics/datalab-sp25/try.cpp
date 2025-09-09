int bitReverse(int x) {
    int a, c = ~(1 << 31); // 初始化掩码 c = 0x7FFFFFFF

    // Step 1: Swap every 1 bit (mask 0xAAAAAAAA)
    a = (0xaa << 8) | 0xaa;   // 0xaaaa (合并表达式)
    a = (a << 16) | a;        // 0xAAAAAAAA (总操作符: << |)
    x = ((x & a) >> 1 | (x & ~a) << 1) & c;

    // Step 2: Swap every 2 bits (mask 0xCCCCCCCC)
    a = (0xcc << 8) | 0xcc;   // 0xcccc
    a = (a << 16) | a;        // 0xCCCCCCCC
    x = ((x & a) >> 2 | (x & ~a) << 2) & (c >>= 1); // 复用 c

    // Step 3: Swap every 4 bits (mask 0xF0F0F0F0)
    a = (0xf0 << 8) | 0xf0;   // 0xf0f0
    a = (a << 16) | a;        // 0xF0F0F0F0
    x = ((x & a) >> 4 | (x & ~a) << 4) & (c >>= 2);

    // Step 4: Swap every 8 bits (mask 0xFF00FF00)
    a = (0xff << 8) << 16;    // 0xff0000 → 左移得 0xFF00FF00
    x = ((x & a) >> 8 | (x & ~a) << 8) & (c >>= 4);

    // Step 5: Swap every 16 bits (直接掩码 0xFFFF0000)
    return (x >> 16) | (x << 16); // 无需掩码，符号位自动处理
}