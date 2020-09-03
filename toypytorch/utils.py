import collections
from typing import List, Tuple


def subvals(args, vals: List[tuple]):
    args = list(args)
    for argnum, argval in vals:
        args[argnum] = argval

    return tuple(args)
    

def upper_first_letter(s: str):
    if s:
        return s[0].upper() + s[1:]
    else:
        return ''


def topo_sort(end_tensor):
    fn_dict = collections.defaultdict(int)
    stack = [end_tensor.get_grad_fn()]
    while stack:
        grad_fn = stack.pop()
        fn_dict[grad_fn] += 1

        stack.extend(grad_fn.get_all_next_fn())

    entry_list = [end_tensor]
    while entry_list:
        entry_tensor = entry_list.pop()
        yield entry_tensor

        for next_fn, next_tensor in (
                entry_tensor.get_grad_fn().get_all_next_fn_and_tensor()):
            fn_dict[next_fn] -= 1
            if fn_dict[next_fn] == 0:
                entry_list.append(next_tensor)

