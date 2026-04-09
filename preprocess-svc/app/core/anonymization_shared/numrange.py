#!/usr/bin/env python
# coding=utf-8


class NumRange(object):
    """Numeric range representation used by Mondrian variants."""

    def __init__(self, sort_value, support):
        self.sort_value = list(sort_value)
        self.support = support.copy()
        self.range = float(sort_value[-1]) - float(sort_value[0])
        self.dict = {}
        for i, v in enumerate(sort_value):
            self.dict[v] = i
        self.value = sort_value[0] + "," + sort_value[-1]

