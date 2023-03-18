"""
预处理
"""
import os
from units import path_leaf
from graph import static_graph_generate

# 存储节点的前身和后继，
# per_block_neighbors_bids[block_id] = [[block ids of predecessors ],[block ids of successors]]
per_block_neighbors_bids = {}
# 没有代码的块
non_code_block_ids = []
# 存储两个二进制文件中的所有操作码(不重复)
opcode_list = []

# 8位寄存器
register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp',
                        'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
# 4位寄存器
register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp',
                        'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']
# 2位寄存器
register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp',
                        'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']
# 1位寄存器
register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl',
                        'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


def node_dict_generate(src_node_list, des_node_list):
    """
    对节点从0开始进行编号，并使用字典进行保存，k=节点名称，v=编号
    :param src_node_list:
    :param des_node_list:
    :return:
    """
    src_node_dict = {}
    des_node_dict = {}

    for i in range(len(src_node_list)):
        src_node_dict[src_node_list[i]] = i

    for i in range(len(des_node_list)):
        j = i + len(des_node_list)
        des_node_dict[des_node_list[i]] = j

    return src_node_dict, des_node_dict


def offset_str_mapping_generate(src_cfg, des_cfg, src_binary, des_binary):
    """
    记录函数中的常量字符串的位置和值，k=字符串在内存中的位置，v=字符串的内容
    记录非so文件中函数的名称
    :param src_cfg:
    :param des_cfg:
    :param src_binary:
    :param des_binary:
    :param mnemonic_list:
    :return:
    """
    # 记录函数中字符串的在内容中的位置和值
    # offset_str_mapping[offset] = strRef.strip()
    offset_str_mapping = {}

    # 列出不存储在 src_binary 和 des_binary 二进制文件中所有的二进制函数
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

    print("两个二进制文件中总共有{}种类型的字符串".format(len(offset_str_mapping)))
    return offset_str_mapping, extern_func_names_src_binary, extern_func_names_des_binary


def normalization(opstr, offset_str_mapping):
    """
    获取指令中的操作地址（操作数）序列化的字符串
    :param opstr:
    :param offset_str_mapping:
    :return:
    """
    optoken = ''
    opstr_num = ''
    if opstr.startswith('0x') or opstr.startswith('0X'):
        opstr_num = str(int(opstr, 16))

    if 'ptr' in opstr:
        # 使用ptr代替指针
        optoken = 'ptr'
    elif opstr_num in offset_str_mapping:
        # 使用字符串代替偏移量
        optoken = offset_str_mapping[opstr_num]
    elif opstr.startswith('0x') or opstr.startswith('-0x') \
            or opstr.replace('.', '', 1).replace('-', '', 1).isdigit():
        # 使用imme代替数字
        optoken = 'imme'
    elif opstr in register_list_1_byte:
        # 使用reg1代替大小为1的寄存器
        optoken = 'reg1'
    elif opstr in register_list_2_byte:
        # 使用reg2代替大小为2的寄存器
        optoken = 'reg2'
    elif opstr in register_list_4_byte:
        # 使用reg4代替大小为4的寄存器
        optoken = 'reg4'
    elif opstr in register_list_8_byte:
        # 使用reg8代替大小为8的寄存器
        optoken = 'reg8'
    else:
        optoken = str(opstr)

    return optoken


def node_index_to_code_generate(src_node_list, des_node_list, src_node_dict, des_node_dict,
                                offset_str_mapping, output_dir):
    """
    对节点进行处理
    1.对操作地址进行序列化
    2.对指令进行统计，以计算TF-IDF
    3.记录字符串出现的块
    :param src_node_list:
    :param des_node_list:
    :param src_node_dict:
    :param des_node_dict:
    :param offset_str_mapping:
    :param output_dir:
    :return:
    """
    # 记录字符串出现的块
    # string_bid[string] = block id
    src_string_bid = {}
    des_string_bid = {}

    # 记录块的索引和序列化后的Token
    # block_idx_to_tokens[block id] = token list of that block
    block_idx_to_tokens = {}

    # 记录每个块中的指令的数量，用于计算 TF-IDF 中的 TF
    # block_idx_to_opcode_num[block id] = Number of instructions in the block
    block_idx_to_opcode_num = {}

    # 记录包含某个操作码的块的数量，用于计算 TF-IDF 中的 IDF 部分
    # instructions_to_block_counts[opcode] = Number of blocks containing opcodes
    instructions_to_block_counts = {}

    # 记录在每个块中各个操作码出现的次数
    # instructions_to_block_counts[block id] = The number of occurrences of each opcode
    block_idx_to_opcode_counts = {}

    # 存储节点索引到代码映射以供参考
    with open(output_dir + 'node_index_to_code', 'w') as node_to_index:
        # 写入两个二进制文件中节点的数量
        node_to_index.write(str(len(src_node_list)) + ' ' + str(len(des_node_list)) + '\n')

        # 处理 src 的节点
        for node in src_node_list:
            predecessors = node.predecessors  # 当前节点的前身节点
            successors = node.successors  # 当前节点的后继节点

            # 前身节点和后继节点的编号
            predecessors_ids = []
            successors_ids = []
            for pred in predecessors:
                predecessors_ids.append(src_node_dict[pred])
            for succ in successors:
                successors_ids.append(src_node_dict[succ])

            # 该节点的相邻节点
            per_block_neighbors_bids[src_node_dict[node]] = [predecessors_ids, successors_ids]

            # 空节点的统计信息
            if node.block is None:
                non_code_block_ids.append(src_node_dict[node])
                block_idx_to_tokens[str(src_node_dict[node])] = []
                block_idx_to_opcode_counts[str(src_node_dict[node])] = {}
                block_idx_to_opcode_num[str(src_node_dict[node])] = 0
                continue

            node_to_index.write(str(src_node_dict[node]) + ':\n')
            node_to_index.write(str(node.block.capstone.insns) + '\n\n')

            # 遍历该块中所有的指令，并进行统计
            tokens = []  # 序列化的指令
            opcode_counts = {}  # 操作码出现的次数
            counted_insns = []  # 记录在该块中至少计数一次的操作码，用于统计某操作码出现的块的数量
            num_insns = 0  # 指令的数量
            for insn in node.block.capstone.insns:
                num_insns += 1

                # 记录出现的操作码
                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)

                # 统计操作码出现的块的数量
                if insn.mnemonic not in counted_insns:
                    if insn.mnemonic not in instructions_to_block_counts:
                        instructions_to_block_counts[insn.mnemonic] = 1
                    else:
                        instructions_to_block_counts[insn.mnemonic] += 1

                    counted_insns.append(insn.mnemonic)

                # 统计该块中操作码出现的次数
                if insn.mnemonic not in opcode_counts:
                    opcode_counts[insn.mnemonic] = 1
                else:
                    opcode_counts[insn.mnemonic] += 1

                tokens.append(str(insn.mnemonic))

                # 遍历操作地址，并进行序列化
                opstrs = insn.op_str.split(', ')
                for opstr in opstrs:
                    optoken = normalization(opstr, offset_str_mapping)
                    if optoken != '':
                        tokens.append(optoken)

                    # 记录字符串出现的块
                    opstr_num = ''
                    if opstr.startswith('0x') or opstr.startswith('0X'):
                        opstr_num = str(int(opstr, 16))
                    if opstr_num in offset_str_mapping:
                        src_string_bid[offset_str_mapping[opstr_num]] = src_node_dict[node]

            block_idx_to_tokens[str(src_node_dict[node])] = tokens
            block_idx_to_opcode_counts[str(src_node_dict[node])] = opcode_counts
            block_idx_to_opcode_num[str(src_node_dict[node])] = num_insns

        # 处理 des 的节点
        for node in des_node_list:
            predecessors = node.predecessors  # 当前节点的前身节点
            successors = node.successors  # 当前节点的后继节点

            # 前身节点和后继节点的编号
            predecessors_ids = []
            successors_ids = []
            for pred in predecessors:
                predecessors_ids.append(des_node_dict[pred])
            for succ in successors:
                successors_ids.append(des_node_dict[succ])

            # 相邻节点
            per_block_neighbors_bids[des_node_dict[node]] = [predecessors_ids, successors_ids]

            # 空节点的统计信息
            if node.block is None:
                non_code_block_ids.append(des_node_dict[node])
                block_idx_to_tokens[str(des_node_dict[node])] = []
                block_idx_to_opcode_counts[str(des_node_dict[node])] = {}
                block_idx_to_opcode_num[str(des_node_dict[node])] = 0
                continue

            node_to_index.write(str(des_node_dict[node]) + ':\n')
            node_to_index.write(str(node.block.capstone.insns) + '\n\n')

            # 遍历块中的每条指令，并进行统计
            tokens = []  # 序列化的指令
            opcode_counts = {}  # 操作码出现的次数
            counted_insns = []  # 记录在该块中至少计数一次的操作码，用于统计某操作码出现的块的数量
            num_insns = 0  # 指令的数量
            for insn in node.block.capstone.insns:
                num_insns += 1

                # 记录出现的操作码
                if insn.mnemonic not in opcode_list:
                    opcode_list.append(insn.mnemonic)

                # 统计操作码出现的块的数量
                if insn.mnemonic not in counted_insns:
                    if insn.mnemonic not in instructions_to_block_counts:
                        instructions_to_block_counts[insn.mnemonic] = 1
                    else:
                        instructions_to_block_counts[insn.mnemonic] += 1

                    counted_insns.append(insn.mnemonic)

                # 统计该块中操作码出现的次数
                if insn.mnemonic not in opcode_counts:
                    opcode_counts[insn.mnemonic] = 1
                else:
                    opcode_counts[insn.mnemonic] += 1

                tokens.append(str(insn.mnemonic))

                # 遍历操作地址，并进行序列化
                opstrs = insn.op_str.split(', ')
                for opstr in opstrs:
                    optoken = normalization(opstr, offset_str_mapping)
                    if optoken != '':
                        tokens.append(optoken)

                    # 记录字符串出现的块
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
    """
    记录 CFG 的边
    :param src_edge_list:
    :param src_node_dict:
    :param des_edge_list:
    :param des_node_dict:
    :param output_dir:
    :return:
    """
    with open(output_dir + 'edge_list', 'w') as edge_list_file:
        for (src, target) in src_edge_list:
            edge_list_file.write(str(src_node_dict[src]) + ' ' + str(src_node_dict[target]) + '\n')
        for (src, target) in des_edge_list:
            edge_list_file.write(str(des_node_dict[src]) + ' ' + str(des_node_dict[target]) + '\n')

