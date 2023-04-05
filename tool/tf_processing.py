import tensorflow as tf


def gaussian_kernel2d(variable, mean, sigma, threshold):
    rank_assert = tf.Assert(tf.equal(tf.rank(variable), tf.rank(mean)),
                            [""])
    shape_assert = tf.Assert(tf.equal(tf.shape(variable)[-1], 2),
                             [""])
    with tf.control_dependencies([rank_assert, shape_assert]):
        distance = tf.reduce_sum((variable - mean)**2, axis=-1)
    scores = tf.exp(-distance / 2. / sigma**2)
    return tf.where(scores < threshold,
                    tf.zeros_like(scores),
                    scores)


def confidence_maps(height, width, all_keypoints, parts=18, sigma=8., model_stride=None, threshold=0.01):
    model_stride = model_stride or 1
    rank_assert = tf.Assert(
         tf.equal(tf.rank(all_keypoints), 2),
         [''])
    parts_assert = tf.Assert(tf.equal(tf.shape(all_keypoints)[0], parts),
                            [''])
    with tf.control_dependencies([rank_assert, parts_assert]):
        x = tf.linspace(0., tf.cast(width - 1, dtype=tf.float32), width)
        y = tf.linspace(0., tf.cast(height - 1, dtype=tf.float32), height)
        xs, ys = tf.meshgrid(x, y)
        grid = tf.concat([tf.expand_dims(xs, axis=-1), tf.expand_dims(ys, axis=-1)], axis=-1)
    heat_map = []
    for i in range(parts):
        maps = tf.where(
            tf.equal(all_keypoints[i, -1], 0.),
            tf.zeros(shape=(height, width)),
            gaussian_kernel2d(
                variable=(model_stride / 2 - 0.5) + grid * model_stride,
                mean=tf.tile(tf.reshape(all_keypoints[i, :2], (1, 1, -1)), [height, width, 1]),
                sigma=sigma,
                threshold=threshold)
        )
        heat_map.append(maps)

    heat_map = tf.stack(heat_map, axis=-1)
    # when S*{j}(p) < threshold will be set to background
    background = tf.clip_by_value(1. - tf.reduce_max(heat_map, axis=-1), 0.0, 1.0)
    return tf.concat([heat_map, tf.expand_dims(background, axis=-1)], axis=-1)


def _choose_valid_points(image_points, start_coord, end_coord, threshold):
    """
    Arg:
        image_points: A `tensor` shape of [?, ?, 2], represent coordinates of image point
    """

    inner_product = lambda v1, v2: tf.reduce_sum(v1 * v2, axis=-1)

    height, width = tf.shape(image_points)[0], tf.shape(image_points)[1]

    # `vector_length` is `l{c, k}` = ||x{j2, k} - x{j1, k}||{2}.
    vector_length = tf.sqrt(tf.reduce_sum((end_coord - start_coord) ** 2))
    # `unit_vector` is (x{j2, k} - x{j1, k}) / ||x{j2, k} - x{j1, k}||{2}.
    unit_vector = (end_coord - start_coord) / vector_length
    # `image_point` is `p` , start_coord is `x{j1, k}`.
    #  so `vector` is `p - x{j1, k}`.
    vectors = image_points - start_coord
    # Computed orthogonal projection
    scalar = tf.expand_dims(inner_product(vectors, unit_vector), axis=-1)
    #  l*{c, k}(p) = {v if p on limb c, k; 0 otherwise}
    mask = tf.where(
        # 0 ≤ v · (p - x{j1, k}) ≤ l{c, k}
        tf.logical_and(tf.less_equal(scalar, vector_length), tf.greater_equal(scalar, 0.)),
        tf.ones(shape=(height, width, 1)),
        tf.zeros(shape=(height, width, 1)))

    orthogonal_vectors = vectors - scalar * unit_vector
    # l*{c, k}(p) = {v if p on limb c,k
    #                0 otherwise }
    mask = tf.where(
        # |v⊥ · (p - xj1, k)| ≤ σl
        tf.less_equal(tf.expand_dims(tf.reduce_sum(orthogonal_vectors**2, axis=-1), axis=-1), threshold**2),
        mask,
        tf.zeros_like(mask))
    vector_fields = mask * unit_vector
    return vector_fields


def part_affinity_fields(height, width, all_keypoints, connections, parts=18, model_stride=None, threshold=8.):
    model_stride = model_stride or 1
    rank_assert = tf.Assert(
         tf.equal(tf.rank(all_keypoints), 2),
         [''])
    parts_assert = tf.Assert(tf.equal(tf.shape(all_keypoints)[0], parts),
                             [''])
    with tf.control_dependencies([rank_assert, parts_assert]):
        x = tf.linspace(0., tf.cast(width - 1, dtype=tf.float32), width)
        y = tf.linspace(0., tf.cast(height - 1, dtype=tf.float32), height)
        xs, ys = tf.meshgrid(x, y)

    image_points = tf.concat([tf.expand_dims(xs, axis=-1), tf.expand_dims(ys, axis=-1)], axis=-1)
    part_aff_vector = []
    for i, (j_index1, j_index2) in enumerate(connections):
        start_coord = all_keypoints[j_index1, :2]
        end_coord = all_keypoints[j_index2, :2]
        part_aff_vector.append(
            tf.where(
                tf.logical_or(
                    tf.reduce_all(tf.equal(start_coord, end_coord)),
                    tf.logical_or(tf.equal(all_keypoints[j_index1, -1], 0.),
                                  tf.equal(all_keypoints[j_index2, -1], 0.))),
                tf.zeros_like(image_points),
                _choose_valid_points(
                    (model_stride / 2 - 0.5) + image_points * model_stride,
                    start_coord, end_coord, threshold)))

    part_aff_vector = tf.concat(part_aff_vector, axis=-1)
    return part_aff_vector


def create_target(h, w, batch_joints, connections=None, model_stride=None):
    """
    Arg:
        model_stride: A python `integer` or `None`,
        when `model_stride` is `None` or `1`,  `h` and `w` should be the height and width of the input
        of the model,
        when `model_stride` is positive integer, 'h' and 'w' should be the height and width of the feature map
        of the model (such logits);  note: (h * model_stride) and (w * model_stride) must be equal height anf width of
        input of the model.

        joints: shape of (#batch, 18, 3), dtype is tf.float32
    return:
        batch_heatmap: shape of (#h, #w, 19), dtype is tf.float32
        batch_fieldmap: shape of (#h, #w, 38), dtype is tf.float32
    raise:
        when model_stride is negative integer
    """
    if model_stride is not None:
        if not isinstance(model_stride, int) or model_stride < 0:
            raise ValueError("Unexpected value of `model_stride`.")

    if connections is None:
        connections = [[1, 7], [1, 6], [7, 9], [9, 11], [6, 8], [8, 10], [1, 13], [13, 15], [15, 17],
                       [1, 12], [12, 14], [14, 16], [1, 0], [0, 3], [3, 5], [0, 2], [2, 4], [7, 5], [6, 4]]
    batch_heatmap = []
    batch_fieldmap = []
    for joints in tf.unstack(batch_joints, axis=0):
        heat_map = confidence_maps(h, w, joints, model_stride=model_stride)
        field_map = part_affinity_fields(h, w, joints, connections, model_stride=model_stride)
        batch_heatmap.append(heat_map)
        batch_fieldmap.append(field_map)
    return tf.stack(batch_heatmap), tf.stack(batch_fieldmap)


# class Test(tf.test.TestCase):
#
#     def test_part_affinity_fields(self):
#         connections = [[1, 2], [3, 4], [3, 5], [2, 3], [1, 4]]
#         all_keywords = tf.constant(
#              [[5., 5., 2.], [10., 5., 2.], [3., 3., 1.], [3., 8., 2.], [0., 0., 0.]])
#              #[[5., 5., 1.], [5., 10., 2.], [3., 3., 1.], [1., 3., 2.], [0., 0., 0.]])
#         fig, ax = plt.subplots()
#         w, h = 15, 15
#         with self.test_session() as sess:
#             x = tf.linspace(0., tf.cast(w - 1, dtype=tf.float32), w)
#             y = tf.linspace(0., tf.cast(h - 1, dtype=tf.float32), h)
#             xs, ys = tf.meshgrid(x, y)
#             connections = np.array(connections) - 1
#             image_points = tf.concat([tf.expand_dims(xs, axis=-1), tf.expand_dims(ys, axis=-1)], axis=-1)
#             local = part_affinity_fields(h, w, all_keywords, connections, 5)
#             pos, directs = sess.run([image_points, local])
#             assert directs.shape == (15, 15, 10)
#             x_pos, y_pos = pos[..., 0], pos[..., 1]
#             x_direct, y_direct = directs[..., 0], directs[..., 1]
#             ax.quiver(x_pos, y_pos, x_direct, y_direct, scale=18)
#             ax.axis([0., 15., 0., 15.])
#             plt.show()
