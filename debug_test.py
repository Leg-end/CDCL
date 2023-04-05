import tensorflow as tf
from tensorlib.contrib import data_flow as aug
from tensorlib.research.CDCL.core import model_helper as helper
import tensorlib as lib


class ModelTest(tf.test.TestCase):

    def test_forward(self):
        net = helper.create_model(output_stride=None)
        inputs = lib.Input((224, 224, 3))
        outputs = net(inputs)
        for output in outputs:
            print('='*10)
            for o in output:
                print(o)
        # writer = tf.summary.FileWriter('D:/GeekGank/workspace/graph/model_graph', tf.get_default_graph())
        # writer.close()

    def test_serialize_augs(self):
        schedule = aug.Sequential([
            aug.RandomBrightness(prob=0.5, delta=32.),
            aug.RandomContrast(prob=0.5, lower=0.6, upper=1.2),
            aug.KeepAspectResize(target_size=(224, 168)),
            aug.OneOf([aug.Rotation(target_size=(224, 168),
                                    scale_range=(0.55, 1.1),
                                    seg_border_value=255),
                       aug.Affine(target_size=(224, 168),
                                  seg_border_value=255)]),
            aug.KeepAspectResize(target_size=(224, 224), random_offset=True, seg_border_value=255),
            aug.RandomFlip(prob=0.5)
        ])
        print("Finished")
        schedule.save_json('./augmenter.json')

    def test_load_weights(self):
        network = helper.create_model(is_training=True)
        inputs = lib.Input((224, 224, 3))
        outputs = network(inputs)
        network.load_weights(
            '../tensorlib/research/mobilenet/checkpoints/mobilenet_v2_1_0.h5',
            allow_skip=True)

    def test_trainable(self):
        network = helper.create_model(is_training=True)
        network.train()
        inputs = lib.Input((224, 224, 3))
        outputs = network(inputs)
        for layer in network.layers():
            if not layer.trainable:
                print(layer.name)

    def test_inception(self):
        from models.human_structure.core.inception.inception_v3 import inception_v3_base
        end_points = ('Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Mixed_5d', 'Mixed_6e', 'Mixed_7c')
        _, nets = inception_v3_base(
            tf.ones((1, 224, 224, 3)), final_endpoint='Mixed_7c')
        print('='*10)
        for k, v in nets.items():
            print(k, v)
        nets = [nets[name] for name in end_points]
        print(nets)
