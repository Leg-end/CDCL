import random
import tensorflow as tf


def split():
    path = 'D:/GeekGank/workspace/Data/coco/json/annotations/person_keypoints_train2014.txt'
    with open(path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    train = lines[:3000]
    val = lines[3000:]
    with open('person_keypoints_train2014.txt', 'w') as f:
        f.write(''.join(train))
    with open('person_keypoints_val2014.txt', 'w') as f:
        f.write(''.join(val))


if __name__ == '__main__':
    with tf.name_scope('a'):
        with tf.name_scope('b'):
            a = tf.ones((2, 2)) + tf.ones((2, 2))
    print(a.op.name)