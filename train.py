import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.normpath(os.path.join(current_dir, "../../.."))
sys.path.append(rootPath)
from tensorlib.research.CDCL.core import model_helper as helper
from tensorlib.research.CDCL.data.data_generator import DataPipe, create_aug
from tensorlib.research.CDCL.tool.tf_processing import create_target
import tensorlib as lib
import tensorflow as tf


def create_model_fn(run_config):
    def model_fn(features, labels=None):
        images = features['image']
        network = helper.create_model(is_training=True, output_stride=None)
        outputs = network(images)
        if labels is None:
            return lib.training.ExecutorSpec(
                outputs=outputs, )
        joints = labels['joint']
        segments = labels['segment']
        mask = labels['mask']
        heat_maps, vector_maps = create_target(
            run_config.input_shapes[0][0] // run_config.model_stride,
            run_config.input_shapes[0][1] // run_config.model_stride,
            joints, model_stride=run_config.model_stride)
        pretrained = len(list(lib.utils.list_files(run_config.model_dir))) == 0
        if pretrained:
            network.load_weights(run_config.pretrained_path,
                                 allow_skip=True,
                                 prefixes=network.name + '/')
        # down sample label
        segments = tf.image.resize_nearest_neighbor(
            tf.expand_dims(segments, axis=3),
            size=run_config.resample_size, align_corners=True)
        segments = tf.squeeze(segments, axis=3)
        # vis hook
        hooks = [helper.VisualizeHook(run_config.test_dir, outputs[2], images, vis_freq=1)]
        loss_func = helper.Losses(run_config.batch_size,
                                  seg_weight_method=run_config.seg_weight_method,
                                  seg_classes=run_config.seg_classes)
        acc_func = helper.Accuracies(num_classes=run_config.seg_classes,
                                     class_names=run_config.class_names)
        ground_truth = (vector_maps, heat_maps, segments)
        accuracies = acc_func(predicts=outputs,
                              labels=(joints, segments))
        outputs = tuple(outputs)
        loss = loss_func.loss(ground_truth=ground_truth,
                              prediction=outputs,
                              mask=mask)
        return lib.training.ExecutorSpec(
            outputs=outputs, loss=loss,
            metrics=accuracies,
            params=list(network.trainable_weights),
            val_hooks=hooks)

    return model_fn


def prepare_configs():
    import math
    # data_dir = 'D:/GeekGank/workspace/Data/coco/json'
    data_dir = '../human_structure/source/data'
    data_config = lib.contrib.training.HParams(data_dir=data_dir,
                                               annotations=[
                                                   'person_keypoints_train2014.txt',
                                                   'aux_person_keypoints_train2014.txt'],
                                               input_shapes=((224, 224, 3), (224, 224), (18, 3), (224, 224)),
                                               num_class=9,
                                               batch_size=12,
                                               prefetch=10,
                                               buffer=50,
                                               repeats=5000,
                                               scheduler=create_aug(True),
                                               num_parallel_calls=2,
                                               shuffle=True,
                                               name='DataConfig')
    val_data_config = lib.contrib.training.HParams(data_dir=data_dir,
                                                   annotations=[
                                                       'person_keypoints_val2014.txt'],
                                                   input_shapes=((224, 224, 3), (224, 224), (18, 3), (224, 224)),
                                                   num_class=9,
                                                   batch_size=12,
                                                   prefetch=10,
                                                   buffer=50,
                                                   repeats=5000,
                                                   scheduler=lib.contrib.data_flow.KeepAspectResize(
                                                       target_size=(224, 224), seg_border_value=255),
                                                   num_parallel_calls=4,
                                                   shuffle=True,
                                                   name='ValDataConfig')
    run_config = lib.contrib.training.RunConfig(model_name='mb2_hs',
                                                root_dir='checkpoint',
                                                input_shapes=((224, 224, 3), (224, 224), (18, 3), (224, 224)),
                                                batch_size=12,
                                                save_summary_steps=1000,
                                                save_checkpoints_steps=1000,
                                                lr_multiplier={'seg': 25., 'paf': 10., 'joint': 10.},
                                                scheduler='exponential',
                                                staircase=True,
                                                decay_rate=0.333,
                                                decay_steps=100000,
                                                learning_rate=4e-5,
                                                steps=600000,
                                                boundaries=[80000, 180000, 300000, 420000, 540000],
                                                lr_values=[4e-5 * math.pow(0.333, i)
                                                           for i in range(6)],
                                                model_stride=8,
                                                resample_size=(28, 28),
                                                seg_weight_method='GMFAVG',
                                                class_names=["mean-iou", "Bkg",
                                                             "Head", "Torso", "U-arms",
                                                             "L-arms", "U-legs", "L-legs",
                                                             "backpack", "shoulderBag"],
                                                seg_classes=9,
                                                joint_classes=19,  ## 18 + 1 center point
                                                paf_classes=38)
    run_config.pretrained_path = '../../tensorlib/research/mobilenet/checkpoints/mobilenet_v2_1.0.ckpt'
    lib.training.RunMetaManager.register_meta(
        'checkpoint', 'mb2_hs', run_config, data_config, val_data_config)


def main():
    run_config, data_config, val_data_config = lib.training.RunMetaManager.get_meta(
        'mb2_hs', root_dir='checkpoint')
    lr = tf.train.exponential_decay(
        learning_rate=run_config.learning_rate,
        global_step=tf.train.get_or_create_global_step(),
        decay_steps=run_config.decay_steps,
        decay_rate=run_config.decay_rate,
        staircase=run_config.staircase)
    x, y = DataPipe(data_config).input_fn()
    val_x, val_y = DataPipe(val_data_config).input_fn()
    exe = lib.training.Executor(create_model_fn(run_config))
    exe.compile(optimizer=lib.training.MultiLROptimizer(
        tf.train.AdamOptimizer(learning_rate=lr),
        global_step=tf.train.get_or_create_global_step(),
        lr_multiplier=run_config.lr_multiplier),
        checkpoint_dir=run_config.model_dir,
        CUDA_VISIBLE_DEVICES='0',
        per_process_gpu_memory_fraction=0.8)
    exe.fit(x, y,
            val_x, val_y, epochs=100,
            steps_per_epoch=5184,
            validation_steps=73)
    # exe.evaluate(val_x, val_y,
    #              validation_steps=73)


def get_pb():
    run_config, _, _ = lib.training.RunMetaManager.get_meta(
        'mb2_hs', root_dir='checkpoint')
    from tensorlib.training import get_frozen_graph
    shape = (None, 224, 224, 3)
    graph = tf.Graph()
    with graph.as_default():
        lib.engine.set_learning_phase(0)
        network = helper.create_model(is_training=False, output_stride=None)
        input_tensor = tf.placeholder(dtype=tf.float32, shape=shape, name='input')
        outputs = network(input_tensor)
        outputs[0][-1] = tf.image.resize_bilinear(
            outputs[0][-1], (224, 224), align_corners=True, name='paf')
        outputs[1][-1] = tf.image.resize_bilinear(
            outputs[1][-1], (224, 224), align_corners=True, name='joint')
        outputs[2][-1] = tf.image.resize_bilinear(
            outputs[2][-1], (224, 224), align_corners=True)
        outputs[2][-1] = tf.nn.softmax(outputs[2][-1], axis=-1, name='seg')
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        tf.global_variables_initializer()
    get_frozen_graph(saver, graph, run_config.model_dir, 'paf,joint,seg', '0')


def run_pb():
    from tensorlib.training import load_graph
    data_dir = r'D:\GeekGank\data\pedestrian\0729_02\female'
    g = load_graph('checkpoint/static/frozen_model.pb')
    writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', g)
    writer.close()
    # for i in g.get_operations():
    #     print(i)
    # inputs = g.get_tensor_by_name('input:0')
    # seg = g.get_tensor_by_name('seg:0')
    # i = 0
    # with tf.Session(graph=g) as sess:
    #     for path in lib.utils.list_files(data_dir):
    #         if i > 100:
    #             break
    #         image = cv2.imread(path)
    #         if image.shape[0] <= 120 or image.shape[1] <= 60:
    #             continue
    #         i += 1
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         image = lib.data.keep_aspect_resize_padding(
    #             image, 224, 224, border_value=(255, 255, 255))
    #         output = sess.run(seg, feed_dict={inputs: [image]})[0]
    #         output = np.argmax(output, axis=-1)
    #         shutil.copy(path, os.path.join('./test', os.path.basename(path) + '.jpg'))
    #         lib.data.imwrite(os.path.join('./test', '{}.png'.format(os.path.basename(path))), output)


if __name__ == '__main__':
    # prepare_configs()
    # main()
    # get_pb()
    run_pb()
