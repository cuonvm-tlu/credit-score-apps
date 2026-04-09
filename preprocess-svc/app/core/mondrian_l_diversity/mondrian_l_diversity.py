"""
main module of mondrian_l_diversity
"""

import pdb
from functools import cmp_to_key
import time

from app.core.anonymization_shared.gentree import GenTree
from app.core.anonymization_shared.numrange import NumRange
from app.core.anonymization_shared.utility import cmp_str

__DEBUG = False
QI_LEN = 10
GL_L = 0
RESULT = []
ATT_TREES = []
QI_RANGE = []
IS_CAT = []


class Partition(object):
    def __init__(self, data, width, middle):
        self.member = data[:]
        self.width = list(width)
        self.middle = list(middle)
        self.allow = [1] * QI_LEN

    def __len__(self):
        return len(self.member)


def check_l_diversity(partition):
    sa_dict = {}
    if len(partition) < GL_L:
        return False
    records_set = partition.member if isinstance(partition, Partition) else partition
    num_record = len(records_set)
    for record in records_set:
        sa_value = record[-1]
        sa_dict[sa_value] = sa_dict.get(sa_value, 0) + 1
    if len(sa_dict.keys()) < GL_L:
        return False
    for sa in sa_dict.keys():
        if sa_dict[sa] > 1.0 * num_record / GL_L:
            return False
    return True


def get_normalized_width(partition, index):
    if IS_CAT[index] is False:
        low = partition.width[index][0]
        high = partition.width[index][1]
        width = float(ATT_TREES[index].sort_value[high]) - float(ATT_TREES[index].sort_value[low])
    else:
        width = partition.width[index]
    return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    max_width = -1
    max_dim = -1
    for i in range(QI_LEN):
        if partition.allow[i] == 0:
            continue
        norm_width = get_normalized_width(partition, i)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = i
    if max_width > 1:
        print("Error: max_witdh > 1")
        pdb.set_trace()
    if max_dim == -1:
        print("cannot find the max dim")
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    frequency = {}
    for record in partition.member:
        frequency[record[dim]] = frequency.get(record[dim], 0) + 1
    return frequency


def find_median(partition, dim):
    frequency = frequency_set(partition, dim)
    split_val = ""
    value_list = list(frequency.keys())
    value_list.sort(key=cmp_to_key(cmp_str))
    total = sum(frequency.values())
    middle = total // 2
    if middle < GL_L or len(value_list) <= 1:
        return ("", "", value_list[0], value_list[-1])
    index = 0
    split_index = 0
    for i, t in enumerate(value_list):
        index += frequency[t]
        if index >= middle:
            split_val = t
            split_index = i
            break
    else:
        print("Error: cannot find splitVal")
    try:
        next_val = value_list[split_index + 1]
    except IndexError:
        next_val = split_val
    return (split_val, next_val, value_list[0], value_list[-1])


def split_numerical_value(numeric_value, split_val):
    split_num = numeric_value.split(",")
    if len(split_num) <= 1:
        return split_num[0], split_num[0]
    low = split_num[0]
    high = split_num[1]
    lvalue = low if low == split_val else low + "," + split_val
    rvalue = high if high == split_val else split_val + "," + high
    return lvalue, rvalue


def split_numerical(partition, dim, pwidth, pmiddle):
    sub_partitions = []
    split_val, next_val, low, high = find_median(partition, dim)
    p_low = ATT_TREES[dim].dict[low]
    p_high = ATT_TREES[dim].dict[high]
    pmiddle[dim] = low if low == high else low + "," + high
    pwidth[dim] = (p_low, p_high)
    if split_val == "" or split_val == next_val:
        return []
    middle_pos = ATT_TREES[dim].dict[split_val]
    lmiddle = pmiddle[:]
    rmiddle = pmiddle[:]
    lmiddle[dim], rmiddle[dim] = split_numerical_value(pmiddle[dim], split_val)
    lhs = []
    rhs = []
    for temp in partition.member:
        pos = ATT_TREES[dim].dict[temp[dim]]
        if pos <= middle_pos:
            lhs.append(temp)
        else:
            rhs.append(temp)
    lwidth = pwidth[:]
    rwidth = pwidth[:]
    lwidth[dim] = (pwidth[dim][0], middle_pos)
    rwidth[dim] = (ATT_TREES[dim].dict[next_val], pwidth[dim][1])
    if not check_l_diversity(lhs) or not check_l_diversity(rhs):
        return []
    sub_partitions.append(Partition(lhs, lwidth, lmiddle))
    sub_partitions.append(Partition(rhs, rwidth, rmiddle))
    return sub_partitions


def split_categorical(partition, dim, pwidth, pmiddle):
    sub_partitions = []
    split_val = ATT_TREES[dim][partition.middle[dim]]
    sub_node = [t for t in split_val.child]
    sub_groups = [[] for _ in range(len(sub_node))]
    if len(sub_groups) == 0:
        return []
    for temp in partition.member:
        qid_value = temp[dim]
        for i, node in enumerate(sub_node):
            try:
                node.cover[qid_value]
                sub_groups[i].append(temp)
                break
            except KeyError:
                continue
    flag = True
    for sub_group in sub_groups:
        if len(sub_group) == 0:
            continue
        if not check_l_diversity(sub_group):
            flag = False
            break
    if flag:
        for i, sub_group in enumerate(sub_groups):
            if len(sub_group) == 0:
                continue
            wtemp = pwidth[:]
            mtemp = pmiddle[:]
            wtemp[dim] = len(sub_node[i])
            mtemp[dim] = sub_node[i].value
            sub_partitions.append(Partition(sub_group, wtemp, mtemp))
    return sub_partitions


def split_partition(partition, dim):
    pwidth = partition.width
    pmiddle = partition.middle
    if IS_CAT[dim] is False:
        return split_numerical(partition, dim, pwidth, pmiddle)
    return split_categorical(partition, dim, pwidth, pmiddle)


def anonymize(partition):
    if not check_splitable(partition):
        RESULT.append(partition)
        return
    dim = choose_dimension(partition)
    if dim == -1:
        print("Error: dim=-1")
        pdb.set_trace()
    sub_partitions = split_partition(partition, dim)
    if len(sub_partitions) == 0:
        partition.allow[dim] = 0
        anonymize(partition)
    else:
        for sub_p in sub_partitions:
            anonymize(sub_p)


def check_splitable(partition):
    return sum(partition.allow) != 0


def init(att_trees, data, l_value):
    global GL_L, RESULT, QI_LEN, ATT_TREES, QI_RANGE, IS_CAT
    ATT_TREES = att_trees
    IS_CAT = []
    for gen_tree in att_trees:
        if isinstance(gen_tree, NumRange):
            IS_CAT.append(False)
        else:
            IS_CAT.append(True)
    QI_LEN = len(data[0]) - 1
    GL_L = l_value
    RESULT = []
    QI_RANGE = []


def mondrian_l_diversity(att_trees, data, l_value):
    init(att_trees, data, l_value)
    middle = []
    result = []
    wtemp = []
    for i in range(QI_LEN):
        if IS_CAT[i] is False:
            QI_RANGE.append(ATT_TREES[i].range)
            wtemp.append((0, len(ATT_TREES[i].sort_value) - 1))
            middle.append(ATT_TREES[i].value)
        else:
            QI_RANGE.append(len(ATT_TREES[i]["*"]))
            wtemp.append(len(ATT_TREES[i]["*"]))
            middle.append("*")
    whole_partition = Partition(data, wtemp, middle)
    start_time = time.time()
    anonymize(whole_partition)
    rtime = float(time.time() - start_time)
    ncp = 0.0
    for partition in RESULT:
        rncp = 0.0
        for i in range(QI_LEN):
            rncp += get_normalized_width(partition, i)
        for i in range(len(partition)):
            gen_result = partition.middle + [partition.member[i][-1]]
            result.append(gen_result[:])
        rncp *= len(partition)
        ncp += rncp
    ncp /= QI_LEN
    ncp /= len(data)
    ncp *= 100
    return (result, (ncp, rtime))

