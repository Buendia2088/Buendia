import struct

def hex_to_double(hex_str):
    # 将16进制字符串转换为字节串
    hex_bytes = bytes.fromhex(hex_str)
    # 使用struct.unpack()函数将字节串转换为双精度浮点数
    double_value = struct.unpack('!d', hex_bytes)[0]
    return double_value

# 示例
hex_value = "c05e280000000000"  # 16进制的双精度浮点数表示为0x1.921fb54442d18p+1
decimal_value = hex_to_double(hex_value)
print("十进制值:", decimal_value)

print(bin(120))

#40200000H0000000
#c05e280000000000