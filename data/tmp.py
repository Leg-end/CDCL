import tensorflow as tf
import numpy as np
import os
import base64
import io
from PIL import Image


def _img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(Image.open(f))
    return img_arr


def _img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = _img_data_to_arr(img_data)
    return img_arr


def imread(img_path):
    with open(img_path, 'rb') as f:
        img_data = f.read()
        img_data = base64.b64encode(img_data).decode('utf-8')
    img = _img_b64_to_arr(img_data)
    return img


def process_fn(string):
    d = eval(string.decode("utf-8"))
    img_path = d['image']
    flag = '/' in img_path and img_path[:img_path.find('/')] == 'train'
    return img_path, d['segmentation'], np.array(d['keypoints'], dtype=np.int32), flag


def decode_image(path, suffix):
    if suffix == 'png':
        image = tf.py_func(imread, [path], tf.uint8)
    else:
        encoded_image = tf.read_file(path)
        image = tf.image.decode_jpeg(encoded_image)
    return image


def image_process(img, seg, joints, flag):
    mask = tf.where(tf.equal(seg, 255),
                    tf.zeros_like(seg, dtype=tf.int64),
                    tf.clip_by_value(tf.cast(seg, tf.int64), 0, 1))
    mask = tf.cast(mask, seg.dtype)
    seg = tf.cond(tf.equal(flag, True),
                  lambda: tf.ones_like(seg, dtype=seg.dtype) * 255,
                  lambda: seg)
    return img, seg, joints, mask


class DataPipe(object):
    def __init__(self,
                 config,
                 name=None):
        self.name = name or self.__class__.__name__
        self.config = config

    def init_data(self):
        annotations = [os.path.join(self.config.data_dir, 'annotations', ann)
                       for ann in self.config.annotations]
        if len(annotations) > 1:
            data = tf.data.Dataset.from_tensor_slices(annotations)
            data = data.interleave(lambda x: tf.data.TextLineDataset(x).repeat(),
                                   cycle_length=len(self.config.annotations),
                                   block_length=1,
                                   num_parallel_calls=self.config.num_parallel_calls)
        else:
            data = tf.data.TextLineDataset(annotations[0])
        data = data.map(lambda x: tf.py_func(
            process_fn, [x], [tf.string, tf.string, tf.int32, tf.bool]),
                        num_parallel_calls=self.config.num_parallel_calls)
        data = data.map(lambda x, y, z, f: (
            decode_image(tf.string_join([self.config.data_dir, '/', x]), 'jpg'),
            decode_image(tf.string_join([self.config.data_dir, '/', y]), 'png'),
            tf.expand_dims(z, axis=0), f),
                        num_parallel_calls=self.config.num_parallel_calls)
        data = data.map(image_process,
                        num_parallel_calls=self.config.num_parallel_calls)
        return data

    def input_fn(self):
        with tf.name_scope(self.name):
            data = self.init_data()
            if self.config.shuffle:
                data = data.shuffle(self.config.buffer)
            data = data.batch(self.config.batch_size, drop_remainder=True).prefetch(
                self.config.prefetch)
            iterator = data.make_one_shot_iterator()
            imgs, segs, joints, masks = iterator.get_next()
            imgs.set_shape([self.config.batch_size] + list(self.config.input_shapes[0]))
            segs.set_shape([self.config.batch_size] + list(self.config.input_shapes[1]))
            joints.set_shape([self.config.batch_size] + list(self.config.input_shapes[2]))
            masks.set_shape([self.config.batch_size] + list(self.config.input_shapes[3]))
        return dict(image=imgs), dict(segment=segs, joint=joints, mask=masks)


def demo():
    import cv2
    from tensorlib.contrib.training import HParams
    from tensorlib.contrib import visual_tool, vis_save
    data_dir = 'D:/GeekGank/workspace/Data/coco/json'
    data_config = HParams(data_dir=data_dir,
                          annotations=['person_keypoints_train2014.txt',
                                       'aux_person_keypoints_train2014.txt'],
                          input_shapes=((224, 224, 3), (224, 224), (18, 3), (224, 224)),
                          num_class=9,
                          batch_size=10,
                          buffer=50,
                          repeats=5000,
                          num_parallel_calls=3,
                          shuffle=True)
    # data_config.to_json_file('./data_config.json')
    data_pipe = DataPipe(data_config)
    _data = data_pipe.init_data()
    _iter = _data.make_one_shot_iterator()
    record = _iter.get_next()
    with tf.Session() as sess:
        for i in range(3):
            imgs, segs, joints, masks = sess.run(record)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            for point in joints:
                if point[2] == 0.:
                    continue
                imgs = cv2.circle(imgs, (point[0], point[1]), 1, (0, 0, 255), 4)
            vis_img = visual_tool.segment2rgb(segs, imgs)
            vis_save('./meta_sequential_%d.png' % i, vis_img)


if __name__ == '__main__':
    demo()
