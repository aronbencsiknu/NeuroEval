import random
from itertools import combinations

ADDR_W = 5
MSG_W = 10  # Example width for MSG_W
EAST, NORTH, WEST, SOUTH, L1 = range(5)
NUM_PACKETS_P_INJ = 5

def generate_variables(direction):
    address_lists = {
        EAST: [],
        NORTH: [],
        WEST: [],
        SOUTH: [],
        L1: []
    }
    
    message_lists = {
        EAST: [],
        NORTH: [],
        WEST: [],
        SOUTH: [],
        L1: []
    }

    for i in range(NUM_PACKETS_P_INJ):
        if direction == EAST:
            zero_bit = 4
            address_lists[EAST].append(generate_address(zero_bit, valid_combinations))
            message_lists[EAST].append(generate_message())
        elif direction == NORTH:
            zero_bit = 3
            address_lists[NORTH].append(generate_address(zero_bit, valid_combinations))
            message_lists[NORTH].append(generate_message())
        elif direction == WEST:
            zero_bit = 2
            address_lists[WEST].append(generate_address(zero_bit, valid_combinations))
            message_lists[WEST].append(generate_message())
        elif direction == SOUTH:
            zero_bit = 1
            address_lists[SOUTH].append(generate_address(zero_bit, valid_combinations))
            message_lists[SOUTH].append(generate_message())
        elif direction == L1:
            zero_bit = 0
            address_lists[L1].append(generate_address(zero_bit, valid_combinations))
            message_lists[L1].append(generate_message())

    return address_lists, message_lists

def generate_message():
    # Generate a random number in the range [0, 2^MSG_W - 1]
    message = random.randint(0, (1 << MSG_W) - 1)
    
    # Convert the message to a binary string, zero-padded to MSG_W bits
    message_bits = f'{message:0{MSG_W}b}'
    
    return message_bits

def generate_valid_combinations(addr_w):
    valid_combinations = []
    for num_ones in range(2, addr_w + 1):  # from exactly two '1's to addr_w '1's
        for combo in combinations(range(addr_w), num_ones):
            bits = ['0'] * addr_w
            for bit in combo:
                bits[bit] = '1'
            valid_combinations.append(int(''.join(bits), 2))
    return valid_combinations

valid_combinations = generate_valid_combinations(ADDR_W)

def generate_address(zero_bit, valid_combinations):
    # Select a random address from valid_combinations
    address = valid_combinations[random.randint(0, len(valid_combinations) - 1)]
    print(address)
    # Convert address to binary list of bits
    address_bits = list(f'{address:0{ADDR_W}b}')
    
    # Set the specified bit to zero
    address_bits[zero_bit] = '0'

    address = ''.join(address_bits)
    
    return address

def concatenate_message_address(message_list, address_list):
    concatenated_list = []
    for i in range(NUM_PACKETS_P_INJ):
        concatenated_list.append(message_list[i] + address_list[i])
    return concatenated_list

def expand_address_list(in_a_list, in_m_list, addr_width):
    out_list = []
    idx = 0  # Initialize index

    # Iterate over each packet
    for i in range(len(in_a_list)):
        # Iterate over each bit in the address width
        for j in range(addr_width):
            if in_a_list[i][j] == "1":
                bit_value = 1 << (addr_width - 1)
                temp = bit_value >> j
                formatted_value = f"{temp:0{addr_width}b}"
                # Shift 1 to the position of each set bit, concatenate with message
                out_list.append(in_m_list[i] + formatted_value)
                idx += 1

    return out_list, idx

# Example usage
if __name__ == "__main__":
    direction = EAST  # Example direction
    # address_lists, message_lists = generate_variables(direction)
    # print("Address List for EAST:", address_lists[EAST])
    # print("Message List for EAST:", message_lists[EAST])

    # c_list = concatenate_message_address(message_lists[EAST], address_lists[EAST])

    # print(c_list)

    MSG_W = 8
    ADDR_W = 5

    in_a_list = ["10101"]
    in_m_list =  ["01011010","10101001", "01101101", "00011110"]

    out_list, idx = expand_address_list(in_a_list, in_m_list, ADDR_W)

    # Print the output
    print(out_list)
    print(idx)


