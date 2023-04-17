#! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
import torch.autograd
import numpy as np
from .binding import einsum, einsum_gemm
from ..common import normalize_subscript


class EinsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, equation, input_0, input_1=None, opt_plan=None):
        # print("opt_plan in forward", opt_plan)
        equation, isBinary = normalize_subscript(equation)
        if isBinary and input_1 is None:
            raise RuntimeError(
                'The subscript indicates two inputs, but only one was passed'
            )
        if not isBinary and input_1 is not None:
            raise RuntimeError('The subscript indicates one input, but two were passed')
        if input_1 is None:
            input_1 = input_0.new_empty((1,))

        # default
        my_opt = torch.IntTensor([-1])
        if opt_plan is not None:
            my_opt = opt_plan

        output = einsum(equation, my_opt, input_0, input_1, False, False)

        if isBinary:
            ctx.save_for_backward(input_0, input_1)

        ctx.equation = equation
        ctx.isBinary = isBinary

        return output

    @staticmethod
    def gemm(
        output,
        tensor_i,
        tensor_j,
        m,
        n,
        k,
        l=1,
        batch_i=torch.empty(0),
        batch_j=torch.empty(0),
        use_cutlass=0,
    ):
        einsum_gemm(
            output, tensor_i, tensor_j, m, n, k, l, batch_i, batch_j, use_cutlass
        )


class Einsum(torch.nn.Module):
    def __init__(self, equation):
        super(Einsum, self).__init__()
        self.equation = equation
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input_0, input_1):
        return EinsumFunction.apply(self.equation, input_0, input_1)


def _compute_target_tensor(in0, in1, target):
    result = ""
    for m in in0[:-1] + in1[:-1] + in1[-1] + in0[-1]:
        if m in target and not m in result:
            result += m
    # reorder target modes like target
    result = list(result)
    for i in range(len(result)):
        if result[i] not in target:
            continue
        for j in range(i):
            if result[j] not in target:
                continue
            if target.index(result[j]) > target.index(result[i]):
                result[i], result[j] = result[j], result[i]
    return ''.join(result)


def EinsumGeneral_new(equation, opt_plan, *tensors, **kwargs):
    tensors = list(tensors)
    equation, isBinary = normalize_subscript(equation)
    path = np.einsum_path(
        equation, *[np.broadcast_to(np.nan, t.shape) for t in tensors], **kwargs
    )
    path = path[0][1:]
    equation = equation.split('->')
    eqs = equation[0].split(',')
    target = equation[1]
    for step in path:
        if len(step) == 1:
            result = EinsumFunction.apply(eqs[0] + '->' + target, tensors[0])
            continue
        assert step[0] < step[1]
        in0 = tensors[step[0]]
        in1 = tensors[step[1]]

        tensors.pop(step[1])
        tensors.pop(step[0])
        tgt = _compute_target_tensor(eqs[step[0]], eqs[step[1]], target)
        assert tgt != ""
        eq = eqs[step[0]] + ',' + eqs[step[1]] + '->' + tgt
        eqs.pop(step[1])
        eqs.pop(step[0])
        eqs.append(tgt)
        result = EinsumFunction.apply(eq, in0, in1, opt_plan)
        tensors.append(result)
    return result


def einsum_contraction(
    equation,
    tensor_i,
    tensor_j,
    opt_plan=None,
    batch_i=None,
    batch_j=None,
    use_cutlass=0,
):
    import re

    left_eq_i, left_eq_j, right_eq = re.split(',|->', equation)
    reduced_letter_i = []
    remained_letter_i = []
    for letter in left_eq_i:
        if letter in left_eq_j and letter not in right_eq:
            reduced_letter_i.append(letter)
        else:
            remained_letter_i.append(letter)
    reduced_letter_j = []
    remained_letter_j = []
    for letter in left_eq_j:
        if letter in left_eq_i and letter not in right_eq:
            reduced_letter_j.append(letter)
        else:
            remained_letter_j.append(letter)

    is_gemm = (
        True
        if ''.join(remained_letter_i) + ''.join(reduced_letter_i) == left_eq_i
        and ''.join(remained_letter_j) + ''.join(reduced_letter_j) == left_eq_j
        and ''.join(reduced_letter_i) == ''.join(reduced_letter_j)
        and len(reduced_letter_i) > 0
        and len(right_eq) > 6  # ignore small matrices
        and ''.join(remained_letter_i) + ''.join(remained_letter_j) == right_eq
        else False
    )
    is_batched_gemm = (
        True
        if ''.join(reduced_letter_i) == ''.join(reduced_letter_j)
        and ''.join(remained_letter_i) + ''.join(remained_letter_j[1:]) == right_eq
        else False
    )

    if is_batched_gemm:
        assert batch_i.shape[0] > 0 and batch_i.shape[0] == batch_j.shape[0]

        l = batch_i.shape[0]
        m = 1
        n = 1
        k = 1
        reshape_size = [batch_i.shape[0]]
        for i in range(1, len(remained_letter_i)):
            m *= tensor_i.shape[i]
            reshape_size.append(tensor_i.shape[i])
        for i in range(1, len(remained_letter_j)):
            n *= tensor_j.shape[i]
            reshape_size.append(tensor_j.shape[i])
        for i in range(len(remained_letter_i), len(left_eq_i)):
            k *= tensor_i.shape[i]

        result = torch.zeros(l, m, n, dtype=tensor_i.dtype, device=tensor_i.device)

        if use_cutlass:
            EinsumFunction.gemm(
                result, tensor_j, tensor_i, n, m, k, l, batch_j, batch_i
            )
        else:
            EinsumFunction.gemm(
                result,
                tensor_i,
                tensor_j,
                m,
                n,
                k,
                l,
                batch_i,
                batch_j,
                use_cutlass=use_cutlass,
            )
        return result.reshape(reshape_size)

    if is_gemm:
        l = 1
        m = 1
        n = 1
        k = 1
        reshape_size = []
        for i in range(0, len(remained_letter_i)):
            m *= tensor_i.shape[i]
            reshape_size.append(tensor_i.shape[i])
        for i in range(0, len(remained_letter_j)):
            n *= tensor_j.shape[i]
            reshape_size.append(tensor_j.shape[i])
        for i in range(len(remained_letter_i), len(left_eq_i)):
            k *= tensor_i.shape[i]

        # print(f'm = {m}, n = {n}, k = {k}, l = {l}')
        # print(f'reshape_size = {reshape_size}')
        result = torch.zeros(l, m, n, dtype=tensor_i.dtype, device=tensor_i.device)
        # print(f'output gemm shape: {result.shape}')
        if use_cutlass:
            EinsumFunction.gemm(result, tensor_j, tensor_i, n, m, k, l)
        else:
            EinsumFunction.gemm(
                result, tensor_i, tensor_j, m, n, k, l, use_cutlass=use_cutlass
            )
        return result.reshape(reshape_size)

    if batch_i is None:
        result = EinsumFunction.apply(equation, tensor_i, tensor_j, opt_plan)
        return result
    else:
        assert batch_i.shape[0] > 0 and batch_i.shape[0] == batch_j.shape[0]
        result = EinsumFunction.apply(
            equation, tensor_i[batch_i], tensor_j[batch_j], opt_plan
        )
        return result
