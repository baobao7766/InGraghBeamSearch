# -*- coding:utf-8 -*-
"""
GatherTree 方法的实现
作用：从最后一层的叶子节点回溯，推出 beam_width 个路径上的 id 的过程。
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf

__all__ = [
    "gather_tree",      # Numpy --> Tensorflow
    "gather_tree_tf",   # Tensorflow 实现
]

###################################################
## Numpy --> Tensorflow
###################################################
def gather_tree_py(step_ids, parent_ids, sequence_lengths):
    """
    从最后一层的叶子节点回溯，推出 beam_width 个路径上的 id 的过程。
    :param step_ids:            Tensor<int32>, (max_time, batch_size, beam_width)
    :param parent_ids:          Tensor<int32>, (max_time, batch_size, beam_width)
    :param sequence_lengths:    Tensor<int32>, (batch_size, beam_width)
    :return: (max_time, batch_size, beam_width)
    :return:
    """
    max_time      = step_ids.shape[0]
    batch_size    = step_ids.shape[1]
    beam_width    = step_ids.shape[2]

    res = np.zeros_like(step_ids)
    for beam_id in range(beam_width):
        for batch in range(batch_size):
            max_len = sequence_lengths[batch][beam_id]

            # res 的 max_len 到 max_time 之间的内容应该和 step_ids 保持不变
            res[max_len-1:, batch, beam_id] = step_ids[max_len-1:, batch, beam_id]
            parent = parent_ids[max_len-1][batch][beam_id]
            for level in reversed(range(max_len - 1)):
                res[level, batch, beam_id] = step_ids[level][batch][parent]     # 注意前面是 beam_id 而不是 parent
                parent = parent_ids[level][batch][parent]
    return np.array(res).astype(step_ids.dtype)


def gather_tree(step_ids, parent_ids, sequence_lengths):
    """
    Tensor version of gather_tree_py
    以python为后端实现，调用 PyFunc 节点，不支持 tf-serving
    :param step_ids:
    :param parent_ids:
    :param sequence_lengths:
    :return:
    """
    res = tf.py_func(
        func=gather_tree_py,
        inp=[step_ids, parent_ids, sequence_lengths],
        Tout=step_ids.dtype)
    res.set_shape(step_ids.get_shape().as_list())
    return res


###################################################
## Tensorflow 实现的 GatherTree
###################################################
def gather_tree_tf(step_ids, parent_ids, sequence_lengths):
    with tf.variable_scope(name_or_scope="GatherTree", reuse=None):
        idx_shape = tf.shape(step_ids)      # 4, 2, 3
        idx_dtype = step_ids.dtype

        max_time    = idx_shape[0]  # 4
        batch_size  = idx_shape[1]  # 2
        beam_width  = idx_shape[2]  # 3

        # res 需要实现为变量
        init_res = tf.zeros(idx_shape, dtype=idx_dtype)
        init_cnt = tf.constant(0)

        # 定义 index 矩阵 (beam_width*batch_size, 2)
        a = tf.reshape(tf.tile(tf.expand_dims(tf.range(beam_width), axis=1), [1, batch_size]), [-1, ])
        b = tf.tile(tf.range(batch_size), [beam_width])
        idx_arr = tf.concat(
            [
                tf.expand_dims(a, axis=1),  # beam_width
                tf.expand_dims(b, axis=1),  # batch_size
            ], axis=-1)
        max_cnt = tf.shape(idx_arr)[0]  # 8

        def condition(cnt, res_arr):
            return tf.less(cnt, max_cnt)

        def body(cnt, res_arr):
            beam_id = idx_arr[cnt][0]   # 0
            batch   = idx_arr[cnt][1]   # 0

            max_len = sequence_lengths[batch][beam_id]  # 4
            parent = parent_ids[max_len - 1][batch][beam_id]    # 2

            # res 的 max_len 到 max_time 之间的内容应该和 step_ids 保持不变
            temp_idx = tf.concat([tf.expand_dims(batch, axis=0), tf.expand_dims(beam_id, axis=0)], axis=-1)
            temp_a = tf.expand_dims(tf.range(start=max_len-1, limit=max_time), axis=1)
            temp_b = tf.tile(tf.expand_dims(temp_idx, axis=0), [max_time-max_len+1, 1])
            mask_idx = tf.concat([temp_a, temp_b], axis=1)
            mask = indice2mask3D(idxs=mask_idx, shape=idx_shape)
            new_res_arr = tf.where(mask, step_ids, res_arr)

            def condition2(cnt2, parent2, res_arr2):
                return tf.greater(cnt2, 0)

            def body2(cnt2, parent2, res_arr2):
                level = cnt2 - 1    # 2
                ## 实现对 Tensor 两次赋值（Loop内不要使用变量）
                # temp_var2 = tf.Variable(name="TempVar", initial_value=init_res, trainable=True, validate_shape=False)
                # new_res_arr2 = tf.assign(temp_var2[level, batch, beam_id], step_ids[level][batch][parent2])
                new_res_arr2 = tf.reshape(
                    tf.tile(tf.expand_dims(step_ids[level][batch][parent2], axis=0), [tf.reduce_prod(idx_shape)]),
                    idx_shape)
                temp_idx2 = tf.expand_dims(tf.concat(
                    [
                        tf.expand_dims(level, axis=0),  # time = 2
                        tf.expand_dims(batch, axis=0),  # batch = 0
                        tf.expand_dims(beam_id, axis=0),  # parent_beam = 2, beam_id = 0
                    ], axis=0), axis=0)
                mask2 = indice2mask3D(idxs=temp_idx2, shape=idx_shape)
                # 之前这里写错了，产生新的 array 时，坐标要用 beam_id 而不用 parent
                new_res_arr2 = tf.where(mask2, new_res_arr2, res_arr2)

                new_parent = parent_ids[level][batch][parent2]
                return tf.subtract(cnt2, 1), new_parent, new_res_arr2

            out_cnt2, out_new_parent2, out_res_arr2 = tf.while_loop(
                cond=condition2,
                body=body2,
                loop_vars=[max_len-1, parent, new_res_arr],
                name="SubLoop2")
            new_cnt = tf.add(cnt, 1, name="newCount")
            return new_cnt, out_res_arr2

        final_cnt, final_res_arr = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[init_cnt, init_res],
            name="MainLoop")
        return final_res_arr


def indice2mask3D(idxs, shape):
    """
    用于生成指定位置的 mask
    只支持三维Tensor
    :param idxs: nested_list_of_indices
    :param shape: (x, y, z)
    :return:
    """
    with tf.variable_scope("Mask3D"):
        t = tf.range(tf.reduce_prod(shape))
        max_cnt = tf.shape(idxs)[0]
        init_mask = tf.zeros(shape=shape, dtype=tf.bool)

        def cond(cnt, mask):
            return tf.less(cnt, max_cnt)

        def body(cnt, mask):
            idx = idxs[cnt]
            pos = idx[0] * shape[1] * shape[2] + idx[1] * shape[2] + idx[2]
            cur_mask = tf.reshape(tf.equal(t, pos), shape)
            return tf.add(cnt, 1), tf.logical_or(mask, cur_mask)

        final_cnt, final_mask = tf.while_loop(
            cond=cond, body=body, loop_vars=[0, init_mask],
            name="ResultLoop"
        )
        return final_mask


###################################################
## 测试自己实现的 gather_tree
###################################################
def test_mask():
    idxs = tf.constant([(2, 3, 1), (0, 3, 2)])
    cur_shape = tf.constant((3, 5, 7))
    mask = indice2mask3D(idxs, shape=cur_shape)
    choose_op = tf.where(mask, tf.ones_like(mask, dtype=tf.int32), tf.zeros_like(mask, dtype=tf.int32))
    with tf.Session() as sess:
        res = sess.run(mask)
        print(res)

        res = sess.run(choose_op)
        print(res)


def test_gather_tree_tf(predict_ids, parent_ids, seq_lens):
    # shape = (time_step, batch_size, beam_width)
    predict_ids = tf.constant(predict_ids, dtype=tf.int32)
    parent_ids = tf.constant(parent_ids, dtype=tf.int32)
    seq_lens = tf.constant(seq_lens, dtype=tf.int32)

    # gather_tree_op = gather_tree(
    #     step_ids=predict_ids, parent_ids=parent_ids, sequence_lengths=seq_lens)

    gather_tree_op = gather_tree_tf(
        step_ids=predict_ids, parent_ids=parent_ids, sequence_lengths=seq_lens)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        graph = sess.graph
        mask_op = graph.get_tensor_by_name("GatherTree/concat:0")
        print(sess.run(mask_op))

        # mask_op = graph.get_tensor_by_name("GatherTree/MainLoop/newCount:0")
        # print(sess.run(mask_op))

        res = sess.run(gather_tree_op)
        print("\n\n[tf results]")
        print(res)


def test_gather_tree():
    predict_ids = np.array(
        [
            [[1, 2, 3], [5, 2, 1]],
            [[4, 6, 9], [6, 7, 4]],
            [[4, 5, 7], [6, 7, 4]],
            [[7, 8, 9], [9, 8, 7]]
        ]
    )

    parent_ids = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 1, 1], [1, 1, 0]],
            [[1, 2, 0], [1, 2, 0]],
            [[2, 1, 1], [2, 1, 2]]
        ]
    )

    seq_lens = np.array(
        [[2, 3, 4], [3, 3, 3]]
    )

    print("predict_ids\n", predict_ids)
    print("\nparent_ids\n", parent_ids)
    print("\nseq_lens\n", seq_lens)
    print("=" * 30)
    test_gather_tree_tf(predict_ids, parent_ids, seq_lens)
    print("=" * 30)
    res = gather_tree_py(
        step_ids=predict_ids, parent_ids=parent_ids, sequence_lengths=seq_lens)
    print("\n\n[np results]")
    print(res)

###################################################
## 测试 numpy 转 tensorflow
###################################################
def test_numpy(arr):
    brr = arr + 1
    return brr


def test_tf(arr):
    res = tf.py_func(func=test_numpy, inp=[arr], Tout=arr.dtype)
    return res


def test_numpy2tf():
    predict_ids = tf.constant(
        [
            [[1, 2, 3]],
            [[4, 5, 6]],
            [[7, 8, 9]]
        ]
    )
    with tf.Session() as sess:
        res = sess.run(test_tf(predict_ids))
        print(res)



if __name__ == "__main__":
    # test_numpy2tf()
    test_gather_tree()
    # test_mask()




