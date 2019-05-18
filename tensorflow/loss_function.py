'''
    Softmax
    Softmax + Sigmoid(D)
    Softmax + Hinge_like(D)
    Softmax + Cauchy(D)
    Softmax + Gaussian(D)
    Softmax + Center Loss
'''

import tensorflow as tf
import config

import torch



def euclidean_dist_all(mat_ab, label_raw):
    dim_v = tf.size(label_raw)
    mat_a = tf.strided_slice(mat_ab, begin=[0], end=[dim_v], strides=[2])
    mat_b = tf.strided_slice(mat_ab, begin=[1], end=[dim_v], strides=[2])
    dist_ab_eula = tf.sqrt(tf.reduce_sum(tf.square(mat_a-mat_b), axis=1, keepdims=True))
    dist_ab_eula = tf.clip_by_value(dist_ab_eula, 1e-8, 1e8)
    return dist_ab_eula


def get_dense_labels(labels):
    dim_v = tf.size(labels)
    sparse_labels = tf.reshape(labels, [dim_v, 1])
    indices = tf.reshape(tf.range(dim_v), [dim_v, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated,
                                      [dim_v, n_classes],
                                      1.0, 0.0)
    return dense_labels

def loss_dist_norm_all(dist_mat, labels_raw, beta=0.005):
    dense_label = get_dense_labels(labels_raw)
    # 将数据切分成两块，按照块进行计算，one-hot形式
    dim_v = tf.size(labels_raw)
    labels_1 = tf.strided_slice(dense_label, begin=[0], end=[dim_v], strides=[2])
    labels_2 = tf.strided_slice(dense_label, begin=[1], end=[dim_v], strides=[2])
    label_ip = tf.reduce_max(tf.multiply(labels_1, labels_2), axis=1, keepdims=True)

    dist_mat_sq = tf.multiply(tf.constant(beta, dtype=tf.float32), tf.square(dist_mat))  # dist_mat 太小了
    dist_mat_sq = tf.cast(dist_mat_sq, dtype=tf.float64)
    label_ip = tf.cast(label_ip, dtype=tf.float64)
    loss_norm = tf.add(dist_mat_sq, tf.multiply(tf.reshape((label_ip - 1.0), [-1, 1]),
                                                tf.log(tf.reshape((tf.exp(dist_mat_sq) - 1.0), [-1, 1]))))
    loss_norm = tf.cast(loss_norm, dtype=tf.float32)
    loss_return = tf.reduce_mean(loss_norm, name='L_gaussian_loss')
    tf.summary.scalar('L_gaussian_loss', loss_return)
    return loss_return

if __name__ == '__main__':

    # BETA = 0.05
    # fc_margin = torch.randn(128, 2)
    # y = [ 9,  5,  4, 3, 8,  9,  5,  7,  1,  6,  1,  3,  0,  9,
    #      8,  6,  9,  1,  5,  4,  7,  8,  4,  6,  0,  2,  3,  5,
    #      0,  0,  5,  1,  7,  0,  2,  5,  3,  3,  8,  9,  4,  7,
    #      5,  1,  1,  6,  5,  7,  2,  9,  5,  5,  9,  5,  2,  0,
    #      3,  4,  5,  7,  0,  3,  4,  1,  6,  8,  1,  7,  1,  4,
    #      0,  3,  2,  7,  3,  4,  4,  9,  5,  7,  8,  2,  5,  8,
    #      9,  9,  3,  6,  6,  5,  8,  6,  5,  6,  2,  9,  5,  1,
    #      5,  6,  4,  4,  5,  4,  3,  4,  8,  2,  7,  6,  3,  9,
    #      5,  6,  5,  7,  9,  4,  9,  8,  4,  7,  8,  0,  2,  8,
    #      9,  5]
    # y = torch.Tensor(y)
    # eula_dist = euclidean_dist_all(fc_margin)
    # loss = pairwise_sigmoid_loss(eula_dist, y)
    # print(loss)
    num_class = config.N_CLASS
    print(num_class)


