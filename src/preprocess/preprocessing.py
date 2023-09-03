"""
预处理
"""
import os
from units import path_leaf
from graph import static_graph_generate


per_block_neighbors_bids = {}
non_code_block_ids = []
opcode_list = []

register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp',
                        'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp',
                        'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']
register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp',
                        'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']
register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl',
                        'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


def node_dict_generate(src_node_list, des_node_list):
    src_node_dict = {}
    des_node_dict = {}

    for i in range(len(src_node_list)):
        src_node_dict[src_node_list[i]] = i

    for i in range(len(des_node_list)):
        j = i + len(des_node_list)
        des_node_dict[des_node_list[i]] = j

    return src_node_dict, des_node_dict


def offset_str_mapping_generate(src_cfg, des_cfg, src_binary, des_binary):

    offset_str_mapping = {}

    extern_func_names_src_binary = []
    extern_func_names_des_binary = []

    for func in src_cfg.functions.values():
        if func.binary_name == src_binary:
            for offset, str_ref in func.string_references(vex_only=True):
                offset = str(offset)
                if offset not in offset_str_mapping:
                    offset_str_mapping[offset] = ''.join(str_ref.split())
        elif func.binary_name not in extern_func_names_src_binary:
            extern_func_names_src_binary.append(func.name)

    for func in des_cfg.functions.values():
        if func.binary_name == des_binary:
            for offset, str_ref in func.string_references(vex_only=True):
                offset = str(offset)
                if offset not in offset_str_mapping:
                    offset_str_mapping[offset] = ''.join(str_ref).split()

        elif func.binary_name not in extern_func_names_des_binary:
            extern_func_names_des_binary.append(func.name)

    return offset_str_mapping, extern_func_names_src_binary, extern_func_names_des_binary


def normalization(opstr, offset_str_mapping):
    optoken = ''
    opstr_num = ''
    if opstr.startswith('0x') or opstr.startswith('0X'):
        opstr_num = str(int(opstr, 16))

    if 'ptr' in opstr:
        optoken = 'ptr'
    elif opstr_num in offset_str_mapping:
        optoken = offset_str_mapping[opstr_num]
    elif opstr.startswith('0x') or opstr.startswith('-0x') \
            or opstr.replace('.', '', 1).replace('-', '', 1).isdigit():
        optoken = 'num'
    elif opstr in register_list_1_byte:
        optoken = 'reg1'
    elif opstr in register_list_2_byte:
        optoken = 'reg2'
    elif opstr in register_list_4_byte:
        optoken = 'reg4'
    elif opstr in register_list_8_byte:
        optoken = 'reg8'
    else:
        optoken = str(opstr)

    return optoken


def node_index_to_code_generate(src_node_list, des_node_list, src_node_dict, des_node_dict,
                                offset_str_mapping, output_dir):

    # string_bid[string] = block id
    src_string_bid = {}
    des_string_bid = {}

    # block_idx_to_tokens[block id] = token list of that block
    block_idx_to_tokens = {}

    # block_idx_to_opcode_num[block id] = Number of instructions in the block
    block_idx_to_opcode_num = {}

    # instructions_to_block_counts[opcode] = Number of blocks containing opcodes
    instructions_to_block_counts = {}

    # instructions_to_block_counts[block id] = The number of occurrences of each opcode
    block_idx_to_opcode_counts = {}

    with open(output_dir + 'node_index_to_code', 'w') as node_to_index:
        node_to_index.write(str(len(src_node_list)) + ' ' + str(len(des_node_list)) + '\n')

        for node in src_node_list:
            predecessors = node.predecessors
            successors = node.successors

            predecessors_ids = []
            successors_ids = []
            for pred in predecessors:
                predecessors_ids.append(src_node_dict[pred])
            for succ in successors:
                successors_ids.append(src_node_dict[succ])
            per_block_neighbors_bids[src_node_dict[node]] = [predecessors_ids, successors_ids]

            if node.block is None:
                non_code_block_ids.append(src_node_dict[node])
                block_idx_to_tokens[str(src_node_dict[node])] = []
                block_idx_to_opcode_counts[str(src_node_dict[node])] = {}
                block_idx_to_opcode_num[str(src_node_dict[node])] = 0
                continue

            node_to_index.write(str(src_node_dict[node]) + ':\n')
            node_to_index.write(str(node.block.capstone.insns) + '\n\n')


            tokens = []
            opcode_counts = {}
            counted_insns = []
            num_insns = 0
            for insn in node.block.capstone.insns:
                num_insns += 1


                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)


                if insn.mnemonic not in counted_insns:
                    if insn.mnemonic not in instructions_to_block_counts:
                        instructions_to_block_counts[insn.mnemonic] = 1
                    else:
                        instructions_to_block_counts[insn.mnemonic] += 1

                    counted_insns.append(insn.mnemonic)


                if insn.mnemonic not in opcode_counts:
                    opcode_counts[insn.mnemonic] = 1
                else:
                    opcode_counts[insn.mnemonic] += 1

                tokens.append(str(insn.mnemonic))


                opstrs = insn.op_str.split(', ')
                for opstr in opstrs:
                    optoken = normalization(opstr, offset_str_mapping)
                    if optoken != '':
                        tokens.append(optoken)


                    opstr_num = ''
                    if opstr.startswith('0x') or opstr.startswith('0X'):
                        opstr_num = str(int(opstr, 16))
                    if opstr_num in offset_str_mapping:
                        src_string_bid[offset_str_mapping[opstr_num]] = src_node_dict[node]

            block_idx_to_tokens[str(src_node_dict[node])] = tokens
            block_idx_to_opcode_counts[str(src_node_dict[node])] = opcode_counts
            block_idx_to_opcode_num[str(src_node_dict[node])] = num_insns

        for node in des_node_list:
            predecessors = node.predecessors
            successors = node.successors

            predecessors_ids = []
            successors_ids = []
            for pred in predecessors:
                predecessors_ids.append(des_node_dict[pred])
            for succ in successors:
                successors_ids.append(des_node_dict[succ])

            per_block_neighbors_bids[des_node_dict[node]] = [predecessors_ids, successors_ids]

            if node.block is None:
                non_code_block_ids.append(des_node_dict[node])
                block_idx_to_tokens[str(des_node_dict[node])] = []
                block_idx_to_opcode_counts[str(des_node_dict[node])] = {}
                block_idx_to_opcode_num[str(des_node_dict[node])] = 0
                continue

            node_to_index.write(str(des_node_dict[node]) + ':\n')
            node_to_index.write(str(node.block.capstone.insns) + '\n\n')

            tokens = []
            opcode_counts = {}
            counted_insns = []
            num_insns = 0
            for insn in node.block.capstone.insns:
                num_insns += 1

                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)

                if insn.mnemonic not in counted_insns:
                    if insn.mnemonic not in instructions_to_block_counts:
                        instructions_to_block_counts[insn.mnemonic] = 1
                    else:
                        instructions_to_block_counts[insn.mnemonic] += 1

                    counted_insns.append(insn.mnemonic)

                if insn.mnemonic not in opcode_counts:
                    opcode_counts[insn.mnemonic] = 1
                else:
                    opcode_counts[insn.mnemonic] += 1

                tokens.append(str(insn.mnemonic))

                opstrs = insn.op_str.split(', ')
                for opstr in opstrs:
                    optoken = normalization(opstr, offset_str_mapping)
                    if optoken != '':
                        tokens.append(optoken)

                    opstr_num = ''
                    if opstr.startswith('0x') or opstr.startswith('0X'):
                        opstr_num = str(int(opstr, 16))
                    if opstr_num in offset_str_mapping:
                        des_string_bid[offset_str_mapping[opstr_num]] = des_node_dict[node]

            block_idx_to_tokens[str(des_node_dict[node])] = tokens
            block_idx_to_opcode_counts[str(des_node_dict[node])] = opcode_counts
            block_idx_to_opcode_num[str(des_node_dict[node])] = num_insns

    return block_idx_to_tokens, block_idx_to_opcode_num, block_idx_to_opcode_counts, instructions_to_block_counts, \
           src_string_bid, des_string_bid


def edge_list_generate(src_edge_list, src_node_dict, des_edge_list, des_node_dict, output_dir):

    with open(output_dir + 'edge_list', 'w') as edge_list_file:
        for (src, target) in src_edge_list:
            edge_list_file.write(str(src_node_dict[src]) + ' ' + str(src_node_dict[target]) + '\n')
        for (src, target) in des_edge_list:
            edge_list_file.write(str(des_node_dict[src]) + ' ' + str(des_node_dict[target]) + '\n')

