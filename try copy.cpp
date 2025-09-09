#include <iostream>
using namespace std;

uint64_t get_address(int level, uint64_t tag, uint64_t set_index)
{
    uint64_t addr = 0;
    int offset_bits, set_bits;
    switch(level) {
        case 1: // L1D/I: 8B line, 4 sets
            offset_bits = 3; // log2(8)
            set_bits = 2;    // log2(4)
            break;
        case 2: // L2: 8B line, 8 sets
            offset_bits = 3;
            set_bits = 3;   // log2(8)
            break;
        case 3: // L3: 16B line, 16 sets
            offset_bits = 4; // log2(16)
            set_bits = 4;   // log2(16)
            break;
        default:
            addr; return 0;
    }
    addr = tag << (offset_bits + set_bits) | set_index << offset_bits;
    return addr;
}

int main()
{
    int a, b, c;
    cin >> a >> b >> c;
    cout << get_address(a, b, c) << endl;
    return 0;
}