import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies
from tensorlib.research.CDCL.core.model import Model, model_arg_scope
import tensorlib as lib
from tensorlib.contrib.data_flow.image_ops import imwrite
from collections import OrderedDict
import cv2
import numpy as np
import os


def create_model(is_training=True, output_stride=8):
    if output_stride == 8:
        endpoints = ['bottle_neck_7/output',
                     'bottle_neck_15/expansion_output',
                     'convolution_1/output']
        extra_depths = (512, 256)
    else:
        endpoints = ['bottle_neck/output',
                     'bottle_neck_2/output',
                     'bottle_neck_5/output',
                     'bottle_neck_12/output',
                     'convolution_1/output']
        extra_depths = None
    backbone = lib.research.MobileNetV2(
        in_channels=3, endpoints=endpoints,
        base_only=True, output_stride=output_stride)
    with lib.arg_scope(model_arg_scope()):
        model = Model(backbone=backbone, num_stages=3,
                      extra_depths=extra_depths)
    if is_training:
        model.train()
    else:
        model.eval()
    return model


def _div_maybe_zero(total_loss, num_present):
    return tf.cast(num_present > 0, tf.float32) * tf.divide(
        total_loss,
        tf.maximum(1e-5, num_present))


def GMFAVG_weights(one_hot_labels, norm=False):
    if norm:
        weights = tf.constant([0.155, 1.188, 0.521,
                               1.628, 1.695, 0.768,
                               1.116, 0.733, 1.196])
    else:
        weights = tf.constant([0.139, 1.064, 0.467,
                               1.459, 1.518, 0.688,
                               1., 0.657, 1.071])
    weights = tf.reduce_max(one_hot_labels * weights, axis=-1)
    tf.summary.histogram('weight', weights)
    return weights


def segment_loss(logits,
                 labels,
                 num_classes,
                 batch_size,
                 ignore_labels=255,
                 weight_method='GMFAVG',
                 hard_example_mining_step=0,
                 top_k_percent_pixels=1.0):
    keep_mask = tf.cast(tf.not_equal(labels, ignore_labels), dtype=tf.float32)
    labels = tf.one_hot(labels, num_classes)
    if weight_method == 'GMFAVG':
        weights = GMFAVG_weights(one_hot_labels=labels)
    else:
        weights = 1.
    weights = tf.multiply(weights, keep_mask)
    default_loss_scope = 'softmax_all_pixel_loss' \
        if top_k_percent_pixels == 1.0 else 'softmax_hard_example_mining'
    with tf.name_scope(default_loss_scope, values=[logits, labels, weights]):
        pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels, name='train_labels_stop_gradient'),
            logits=logits,
            name='pixel_losses')
        weighted_pixel_losses = tf.multiply(pixel_losses, weights)
        if top_k_percent_pixels == 1.0:
            total_loss = tf.reduce_sum(weighted_pixel_losses)
            num_present = tf.reduce_sum(keep_mask)
            loss = _div_maybe_zero(total_loss, num_present)
            # loss = total_loss / batch_size
        else:
            num_pixels = tf.cast(tf.reduce_prod(tf.shape(logits)[1:-1]), tf.float32)
            if hard_example_mining_step == 0:
                top_k_pixels = tf.cast(top_k_percent_pixels * num_pixels, tf.int32)
            else:
                global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
                ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
                top_k_pixels = tf.cast((ratio * top_k_percent_pixels + (1. - ratio)) * num_pixels, tf.int32)
            weighted_pixel_losses = tf.reshape(weighted_pixel_losses, [int(batch_size), -1])
            top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                          k=top_k_pixels,
                                          sorted=True,
                                          name='top_k_percent_pixels')
            total_loss = tf.reduce_sum(top_k_losses)
            num_present = tf.reduce_sum(tf.cast(tf.not_equal(top_k_losses, 0.), tf.float32))
            loss = _div_maybe_zero(total_loss, num_present)
    return loss


class Losses:
    def __init__(self,
                 batch_size,
                 seg_classes,
                 seg_weight_method='EN',
                 hard_example_mining_step=0,
                 top_k_percent_pixels=1.):
        self.summary_tensors = {}
        self._euclid_loss = lambda x: tf.square(x) / 2.
        self.batch_size = float(batch_size)
        self.hard_example_mining_step = hard_example_mining_step
        self.top_k_percent_pixels = top_k_percent_pixels
        self.seg_classes = seg_classes
        self.seg_weight_method = seg_weight_method

    def _classification_loss(self, ground_truth, logits):
        loss = segment_loss(logits, ground_truth, self.seg_classes,
                            batch_size=self.batch_size,
                            ignore_labels=255,
                            weight_method=self.seg_weight_method,
                            hard_example_mining_step=self.hard_example_mining_step,
                            top_k_percent_pixels=self.top_k_percent_pixels)
        return loss

    def _affinity_losses(self, ground_truth, prediction, mask):
        """
        Ft{L} = 1/2 ∑ ∑ W(p) · ||Lt{c}(p) - L*{c}(p)||^2
        ground_truth: shape of (？, #height, #width, 38)
        """
        per_entry_aff_loss = self._euclid_loss((ground_truth - prediction) * mask)
        return tf.reduce_sum(per_entry_aff_loss) / self.batch_size

    def _part_losses(self, ground_truth, prediction, mask):
        """
        Ft{S} = 1/2 ∑ ∑ W(p) · ||St{j}(p) - S*{j}(p)||^2
        """
        per_entry_part_loss = self._euclid_loss((ground_truth - prediction) * mask)
        return tf.reduce_sum(per_entry_part_loss) / self.batch_size

    def _add_summary(self, summary_tensors):
        for log_name, log_value in summary_tensors.items():
            tf.summary.scalar(log_name, log_value)
            self.summary_tensors[log_name] = log_value

    def loss(self, ground_truth, prediction, mask):
        stage_loss_part = []
        stage_loss_field = []
        stage_loss_seg = []
        losses = []

        with tf.name_scope('loss_function'):
            truth_part_affinity, truth_part_maps, truth_segment = ground_truth
            mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=3)
            mask = tf.image.resize_nearest_neighbor(mask, prediction[0][1].shape[1:-1], align_corners=True)
            for t, (affinity_predict, part_predict, seg_logit) in enumerate(zip(*prediction)):
                parts_loss = self._part_losses(ground_truth=truth_part_maps,
                                               prediction=part_predict, mask=mask)
                fields_loss = self._affinity_losses(ground_truth=truth_part_affinity,
                                                    prediction=affinity_predict, mask=mask)
                classes_loss = self._classification_loss(ground_truth=truth_segment,
                                                         logits=seg_logit)
                losses.append(tf.reduce_mean([parts_loss, fields_loss, classes_loss]))
                stage_loss_part.append(parts_loss)
                stage_loss_field.append(fields_loss)
                stage_loss_seg.append(classes_loss)

            ####################################################
            total_loss = tf.reduce_sum(losses)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                total_loss = with_dependencies([updates], total_loss)

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss_sum')
            total_loss = tf.add(total_loss, regularization_loss, name="total_loss")
            self._add_summary({'regularization_loss': regularization_loss, 'total_loss': total_loss})

            for t, (p, f, s, l) in enumerate(zip(stage_loss_part,
                                                 stage_loss_field,
                                                 stage_loss_seg,
                                                 losses)):
                with tf.name_scope("stage_%d_losses" % t):
                    summary_tensors = {
                        'stage%d_parts_loss' % t: p,
                        'stage%d_fields_loss' % t: f,
                        'stage%d_classes_loss' % t: s,
                        "stage%d_losses" % t: l}
                    self._add_summary(summary_tensors)
        return total_loss


class Accuracies:
    def __init__(self,
                 class_names,
                 num_classes):
        if class_names is None:
            class_names = ["mean_iou", "Bkg", "Head",
                           "Torso", "U-arms", "L-arms",
                           "U-legs", "L-legs", "backpack", "shoulderBag"]
        self.class_names = class_names
        self.num_classes = num_classes
        self.metric_op = lib.training.MeanIou(num_classes=num_classes)

    def __call__(self, predicts, labels):
        for t, (_, part_logits, seg_logits) in enumerate(zip(*predicts)):
            with tf.name_scope('stage_%d_accuracies' % t):
                part_truth, seg_truth = labels
                weights = tf.not_equal(seg_truth, 255)
                seg_logits = tf.nn.softmax(seg_logits)
                seg_truth = tf.one_hot(seg_truth, depth=self.num_classes)
                mean_iou, classes_iou = self.metric_op(
                    labels=tf.argmax(seg_truth, -1),
                    predicts=tf.argmax(seg_logits, -1),
                    weights=weights)
                classes_iou = tf.unstack(classes_iou)
                for iou in classes_iou:
                    setattr(iou, '_metric_obj', self.metric_op)
                accuracies = [mean_iou] + classes_iou
                for name, acc in zip(self.class_names, accuracies):
                    tf.summary.scalar(name, acc)
        results = OrderedDict([(name, value) for name, value in zip(
            self.class_names, accuracies)])
        return results


class VisualizeHook(lib.training.SessRunHook):
    def __init__(self, vis_dir, logits, images, vis_freq=10000):
        self.vis_dir = vis_dir
        shape = lib.engine.int_shape(images)[1:-1]
        self.logits = [tf.argmax(tf.nn.softmax(tf.image.resize_bilinear(
            logit, shape, align_corners=True), axis=-1), axis=-1) for logit in logits]
        self.images = images
        self._step = 0
        self._vis_freq = vis_freq

    def before_run(self, run_context):
        self._step += 1
        return lib.training.SessRunArgs(self.logits + [self.images])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        if self._step % self._vis_freq == 0:
            results = run_values.results
            logits = results[:-1]
            images = results[-1]
            # print(images.shape, truths.shape)
            step_dir = os.path.join(self.vis_dir, 'step_' + str(self._step))
            if not os.path.exists(step_dir):
                os.mkdir(step_dir)
            # print("Write {:d} steps' {:d} results into dir {}".format(
            #     self._step, len(logits[0]) * len(images), step_dir))
            images = images[..., ::-1].astype(np.uint8)
            for b in range(len(images)):
                cv2.imwrite(os.path.join(step_dir, 'b{:d}.jpg'.format(b)), images[b])
                for t, logit in enumerate(logits):
                    imwrite(os.path.join(step_dir, 'b{:d}_t{:d}.png'.format(b, t)), logit[b])
