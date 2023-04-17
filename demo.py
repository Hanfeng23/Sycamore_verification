from os import makedirs
import torch
import numpy as np
from os.path import exists, dirname, abspath
import time
import sys
from traceback import print_exc

from cutensor.torch import einsum_contraction as EinsumOpt
import torch.cuda.nvtx as nvtx

import os

new_equation_list = []

result_ref = []
result_new = []

acc_type = torch.complex64


def print_step(step, tensors, eq, eq_count):
    print(f'Equation count = {eq_count}, Length of step = {len(step)}')
    i, j = step[0]
    print('  step[0]:', i, j)
    # print('  step[1]:', step[1])
    print('  step[1]:', eq)
    batch_i, batch_j = step[2]
    if len(batch_i) > 0:
        print('  step[2]:')
        print("    batched len = ", len(batch_i))
        for ii in range(len(batch_i)):
            print("    k = ", ii)
            print("      batch_i: ", batch_i[ii].shape, batch_i[ii])
            if len(batch_j) > 0:
                print("      batch_j: ", batch_j[ii].shape, batch_j[ii])
    if len(step) > 3:
        for ii in range(3, len(step)):
            print('  step[%d]:' % ii, step[ii])
    print(
        "tensors[i] shape:",
        tensors[i].shape,
        "\ntensors[j] shape:",
        tensors[j].shape,
        "\n",
    )


# @profile
def tensor_contraction_einsum(
    tensors,
    contraction_scheme,
    opt_plans=None,
    use_reorder=False,
    timing_kernels=False,
):
    '''
    step[0]: locations of tensors to be contracted
    step[1]: contraction equation
    step[2]: batch dimension of the contraction
    step[3]: reshape sequence
    step[4]: result's shape if it exists
    '''

    nvtx.range_push("tensor_contraction_einsum")
    device = tensors[0].device
    eq_count = 0

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]

        eq = ''
        if use_reorder:
            eq = new_equation_list[eq_count]
        else:
            eq = step[1]

        try:
            if timing_kernels:
                torch.cuda.synchronize(device)
                tt0 = time.time()
            nvtx.range_push(str(i) + "-" + str(j))
            temp_opt = (
                torch.IntTensor([opt_plans[eq_count]])
                if opt_plans is not None
                else None
            )
            if len(batch_i) > 1:
                tensors[i] = [tensors[i]]
                for k in range(len(batch_i) - 1, -1, -1):
                    nvtx.range_push(str(eq_count) + ": batched " + str(k))
                    if k != 0:
                        tensors[i].insert(
                            1,
                            EinsumOpt(
                                eq,
                                tensors[i][0],
                                tensors[j],
                                batch_i=batch_i[k],
                                batch_j=batch_j[k],
                                opt_plan=temp_opt,
                                use_cutlass=0,
                            ),
                        )
                    else:
                        tensors[i][0] = EinsumOpt(
                            eq,
                            tensors[i][0],
                            tensors[j],
                            batch_i=batch_i[k],
                            batch_j=batch_j[k],
                            opt_plan=temp_opt,
                            use_cutlass=0,
                        )
                    nvtx.range_pop()
                tensors[j] = []
                nvtx.range_push('torch.cat')
                if step[3]:
                    tensors[i] = torch.cat(tensors[i], dim=0).reshape(step[3])
                else:
                    tensors[i] = torch.cat(tensors[i], dim=0)
                nvtx.range_pop()
            elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
                nvtx.range_push(str(eq_count) + ": simple(batched)")
                tensors[i] = EinsumOpt(
                    eq,
                    tensors[i],
                    tensors[j],
                    batch_i=batch_i[0],
                    batch_j=batch_j[0],
                    opt_plan=temp_opt,
                    use_cutlass=0,
                )
                tensors[j] = []
                nvtx.range_pop()
            elif len(step) > 3:  # here step[4] = False
                nvtx.range_push(str(eq_count) + ": simple(reshape)")
                tensors[i] = EinsumOpt(
                    eq,
                    tensors[i],
                    tensors[j],
                    opt_plan=temp_opt,
                    use_cutlass=0,
                ).reshape(step[3])
                if len(batch_i) == 1:
                    tensors[i] = tensors[i][batch_i[0]]
                nvtx.range_pop()
                tensors[j] = []
            else:  # here len(step) <= 3
                nvtx.range_push(str(eq_count) + ": simple")
                tensors[i] = EinsumOpt(
                    eq,
                    tensors[i],
                    tensors[j],
                    opt_plan=temp_opt,
                    use_cutlass=0,
                )
                nvtx.range_pop()
                tensors[j] = []
            if opt_plans is not None:
                opt_plans[eq_count] = temp_opt.item()
            eq_count += 1
            nvtx.range_pop()
            if timing_kernels:
                torch.cuda.synchronize(device)
                tt1 = time.time()
                print(f'equation id {eq_count - 1}: {(tt1-tt0)*1000} ms\n')
        except:
            print(torch.cuda.memory_summary())
            print_step(step, tensors, eq, eq_count)
            print_exc()
            sys.exit(1)

    nvtx.range_pop()
    return tensors[i]


def get_kernels_number(schemes):
    kernels_count = 0
    equation_count = 0
    for step in schemes:
        equation_count += 1
        batch_i, batch_j = step[2]
        if len(batch_i) > 1:
            kernels_count += len(batch_i)
        else:
            kernels_count += 1
    return kernels_count, equation_count


def check_results(r_ref, r_new):
    assert len(r_ref) == len(r_new)
    print('comparing results with norm2')
    for i in range(len(r_ref)):
        norm2_ref = torch.linalg.norm(r_ref[i])
        norm2_new = torch.linalg.norm(r_new[i])
        rel_diff = abs((norm2_ref - norm2_new) / norm2_ref)
        print(
            f'i = {i}, norm2_ref = {norm2_ref}, norm2_new = {norm2_new}, rel_diff = {rel_diff} '
        )


def contraction(
    tensors: list,
    scheme: list,
    slicing_indices: dict,
    num_bitstrings: int,
    task_num: int,
    device='cuda:0',
    get_time=False,
    timing_kernels=False,
    use_reorder=False,
    opt_plans=None,
    check_results=False,
    start_task=0,
):
    slicing_edges = list(slicing_indices.keys())
    tensors_gpu = [tensor.to(device) for tensor in tensors]
    collect_tensor = torch.zeros(num_bitstrings, dtype=acc_type, device=device)

    total_tasks = 2**23
    assert start_task < total_tasks
    end_task = (
        start_task + task_num if start_task + task_num < total_tasks else total_tasks
    )
    for s in range(start_task, end_task):
        if get_time:
            torch.cuda.synchronize(device)
            t0 = time.time()
        nvtx.range_push("start_task" + str(s))
        configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
        sliced_tensors = tensors_gpu.copy()
        for x in range(len(slicing_edges)):
            m, n = slicing_edges[x]
            idxm_n, idxn_m = slicing_indices[(m, n)]
            sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
            sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()

        if timing_kernels:
            print('Warming starts')
            sliced_tensors_copy = sliced_tensors.copy()
            tensor_contraction_einsum(
                sliced_tensors_copy,
                scheme,
                opt_plans=opt_plans,
                use_reorder=use_reorder,
                timing_kernels=False,
            )
            print('Warming ends')

        result_tensor = tensor_contraction_einsum(
            sliced_tensors,
            scheme,
            opt_plans=opt_plans,
            use_reorder=use_reorder,
            timing_kernels=timing_kernels,
        )
        collect_tensor += result_tensor
        nvtx.range_pop()
        if get_time:
            torch.cuda.synchronize(device)
            t1 = time.time()
            print(f'task_id {s}: {t1-t0} s')
        if check_results:
            if use_reorder:
                result_new.append(collect_tensor.clone())
                fname = 'result_tensor_opt_' + str(s) + '.pt'
                torch.save(
                    result_tensor, abspath(dirname(__file__)) + '/results/' + fname
                )
            else:
                result_ref.append(collect_tensor.clone())
                fname = 'result_tensor_ref_' + str(s) + '.pt'
                torch.save(
                    result_tensor, abspath(dirname(__file__)) + '/results/' + fname
                )
    if use_reorder:
        fname = (
            'collect_tensor_opt_' + str(start_task) + '_' + str(end_task - 1) + '.pt'
        )
        torch.save(collect_tensor, abspath(dirname(__file__)) + '/results/' + fname)
    else:
        fname = (
            'collect_tensor_ref_' + str(start_task) + '_' + str(end_task - 1) + '.pt'
        )
        torch.save(collect_tensor, abspath(dirname(__file__)) + '/results/' + fname)

    print(
        f'Sub tasks from {start_task} to {end_task - 1} have been finished successfully.'
    )
    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "-cuda", type=int, default=0, help="cuda device to use, -1 for cpu"
    )
    parser.add_argument(
        "-get_time", action='store_true', help="report the simulaiton time or not"
    )
    parser.add_argument(
        "-task_num", type=int, default=10, help="# of subtasks to be calculated"
    )
    parser.add_argument("-start_task", type=int, default=0, help="starting subtask id")
    parser.add_argument(
        "-get_timing_kernels", default=False, help="report time of each kernel or not"
    )
    parser.add_argument(
        "-bitstrings",
        type=int,
        default=20,
        help="log2(#) of bitstrings to be calculated, if > 100 then will be # of bitstrings",
    )
    parser.add_argument(
        "-check_results", default=False, help="check optimized results with referenced"
    )
    parser.add_argument(
        "-path_file", type=str, default='not_defined', help='path file name .pt file'
    )
    args = parser.parse_args()

    num_bitstrings = 2**args.bitstrings if args.bitstrings < 100 else args.bitstrings

    contraction_file = args.path_file
    contraction_filename = (
        abspath(dirname(__file__)) + f'/data/' + contraction_file + '.pt'
    )
    if not exists(contraction_filename):
        assert ValueError('No contraction data!')

    print('contraction file:', contraction_filename)
    tensors, scheme, slicing_indices, bitstrings = torch.load(contraction_filename)

    device = f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu'

    n_layer = contraction_file[-3:]
    with open(
        abspath(dirname(__file__)) + '/data/new_equations_' + f'{n_layer}' + '.txt',
        'r',
    ) as f:
        lines = f.readlines()
        for line in lines:
            new_equation_list.append(''.join(line.split()))

    kernel_count, equation_count = get_kernels_number(scheme)
    print(f"# equations = {equation_count}, # kernels = {kernel_count}")
    print(f'Will run {args.task_num} tasks.')
    print(f'accumulated type is {acc_type}')

    if args.check_results:
        print('reference:')
        contraction(
            tensors,
            scheme,
            slicing_indices,
            num_bitstrings,
            args.task_num,
            device,
            args.get_time,
            check_results=args.check_results,
            timing_kernels=args.get_timing_kernels,
            start_task=args.start_task,
        )

    print('optimized:')
    opt_plans = [-1] * equation_count
    contraction(
        tensors,
        scheme,
        slicing_indices,
        num_bitstrings,
        args.task_num,
        device,
        args.get_time,
        use_reorder=True,
        opt_plans=opt_plans,
        check_results=args.check_results,
        timing_kernels=args.get_timing_kernels,
        start_task=args.start_task,
    )

    # check results
    if args.check_results:
        check_results(result_ref, result_new)
