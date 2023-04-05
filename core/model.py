import tensorlib as lib


def pre_process_zero_mean_unit_range(inputs):
    # inputs = inputs[..., ::-1]
    inputs = lib.engine.float32(inputs)
    inputs = (2.0 / 255.0) * inputs - 1.0
    return inputs


class StageBlock(lib.Sequential):

    def __init__(self,
                 depth,
                 kernel_size,
                 num_classes,
                 **kwargs):
        super(StageBlock, self).__init__(**kwargs)
        self.add_layer(lib.contrib.Conv2D(
            out_channels=depth // 4, kernel_size=kernel_size, name='conv1'))
        self.add_layer(lib.contrib.Conv2D(
            out_channels=depth // 4, kernel_size=[kernel_size, 1], name='conv2'))
        self.add_layer(lib.contrib.Conv2D(
            out_channels=depth // 4, kernel_size=[1, kernel_size], name='conv3'))
        self.add_layer(lib.contrib.Conv2D(
            out_channels=depth, kernel_size=1))
        self.add_layer(lib.contrib.Conv2D(
            out_channels=num_classes, kernel_size=1,
            activation=None, normalizer=None,
            normalizer_params=None, name='classifier'))


class TimeSeries(lib.LayerList):

    def __init__(self,
                 depth,
                 kernel_size,
                 num_stages,
                 num_classes,
                 **kwargs):
        super(TimeSeries, self).__init__(**kwargs)
        self.num_stages = num_stages
        for i in range(1, num_stages + 1):
            self.add_layer(StageBlock(depth=depth, num_classes=num_classes,
                                      kernel_size=kernel_size,
                                      name='stage_%d' % i))

    def forward(self, inputs):
        nets = []
        x = inputs
        net = inputs
        for i, layer in enumerate(self):
            net = layer(net)
            nets.append(net)
            if i < self.num_stages:
                net = lib.layers.concat([x, net], axis=-1, name='fuse')
        return nets


class ExtraFeatures(lib.LayerList):

    def __init__(self,
                 depths,
                 **kwargs):
        super(ExtraFeatures, self).__init__(**kwargs)
        for i, depth in enumerate(depths):
            self.add_layer(lib.Sequential(
                lib.contrib.Conv2D(out_channels=depth // 2, kernel_size=1,
                                   strides=1, name='conv1'),
                lib.contrib.SeparableConv2D(out_channels=None, kernel_size=3,
                                            strides=1, depth_multiplier=1,
                                            padding='SAME', name='conv2'),
                lib.contrib.Conv2D(out_channels=depth, kernel_size=1,
                                   padding='SAME', name='conv3')))

    def forward(self, inputs):
        nets = []
        net = inputs
        for layer in self:
            net = layer(net)
            nets.append(net)
        return nets


class Fusion(lib.Network):

    def __init__(self,
                 depth,
                 **kwargs):
        super(Fusion, self).__init__(**kwargs)
        convs = [lib.contrib.Conv2D(out_channels=depth, kernel_size=1,
                                    padding='SAME', name='conv%d' % i)
                 for i in range(1, 6)]
        self.convs = lib.LayerList(*convs)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == len(self.convs)
        nets = [conv(net) for conv, net in zip(self.convs, inputs)]
        outputs = lib.layers.concat(nets, axis=3, name='fuse')
        return outputs


class SPPFusion(lib.Network):

    def __init__(self,
                 depth,
                 **kwargs):
        super(SPPFusion, self).__init__(**kwargs)
        cr_convs = [lib.contrib.Conv2D(out_channels=depth, kernel_size=1,
                                       padding='SAME', name='cr_conv%d' % i)
                    for i in range(1, 6)]
        self.deconv5_1 = lib.layers.Conv2DTranspose(out_channels=depth,
                                                    kernel_size=4,
                                                    strides=2,
                                                    spatial_size=(14, 14),
                                                    name='deconv5_1')
        self.deconv5_2 = lib.layers.Conv2DTranspose(out_channels=depth,
                                                    kernel_size=4,
                                                    strides=2,
                                                    spatial_size=(28, 28),
                                                    name='deconv5_2')

        self.deconv4_1 = lib.layers.Conv2DTranspose(out_channels=depth,
                                                    kernel_size=4,
                                                    strides=2,
                                                    spatial_size=(28, 28),
                                                    name='deconv4_1')

        self.dnconv1_1 = lib.layers.Conv2D(out_channels=depth, kernel_size=1,
                                           strides=2, padding='SAME',
                                           name='dnconv1_1')
        self.dnconv1_2 = lib.layers.Conv2D(out_channels=depth, kernel_size=1,
                                           strides=2, padding='SAME',
                                           name='dnconv1_2')

        self.dnconv2_1 = lib.layers.Conv2D(out_channels=depth, kernel_size=1,
                                           strides=2, padding='SAME',
                                           name='dnconv2_1')
        rf_convs = [lib.contrib.Conv2D(out_channels=depth, kernel_size=3,
                                       strides=1, padding='SAME',
                                       name='rf_conv%d' % i)
                    for i in range(1, 6)]
        self.cr_convs = lib.LayerList(*cr_convs)
        self.rf_convs = lib.LayerList(*rf_convs)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == len(self.cr_convs)
        nets = [conv(net) for conv, net in zip(self.cr_convs, inputs)]
        nets[4] = self.deconv5_2(self.deconv5_1(nets[4]))
        nets[3] = self.deconv4_1(nets[3])
        nets[0] = self.dnconv1_2(self.dnconv1_1(nets[0]))
        nets[1] = self.dnconv2_1(nets[1])
        nets = [conv(net) for conv, net in zip(self.rf_convs, nets)]
        outputs = lib.layers.concat(nets, axis=3, name='fuse')
        return outputs


class Model(lib.Network):

    def __init__(self,
                 backbone,
                 num_stages,
                 depth=256,
                 paf_classes=38,
                 joint_classes=19,
                 seg_classes=9,
                 extra_depths=None,  # (512, 256)
                 **kwargs):
        super(Model, self).__init__(**kwargs)
        self.pre_process = lib.layers.Lambda(pre_process_zero_mean_unit_range, name='pre_process')
        self.backbone = backbone
        if extra_depths is not None:
            self.extra_feat = ExtraFeatures(depths=extra_depths, name='extra_feat')
            self.fuse = Fusion(depth=depth, name='fusion')
        else:
            self.fuse = SPPFusion(depth=depth, name='fusion')
        self.paf_series = TimeSeries(depth=512, kernel_size=3, num_stages=num_stages,
                                     num_classes=paf_classes, name='paf')
        self.joint_series = TimeSeries(depth=128, kernel_size=7, num_stages=num_stages,
                                       num_classes=joint_classes, name='joint')
        self.seg_series = TimeSeries(depth=512, kernel_size=3, num_stages=num_stages,
                                     num_classes=seg_classes, name='seg')

    def forward(self, inputs):
        net = self.pre_process(inputs)
        nets = self.backbone(net)
        if hasattr(self, 'extra_feat'):
            nets.extend(self.extra_feat(nets[-1]))
        net = self.fuse(nets)
        paf_nets = self.paf_series(net)
        joint_nets = self.joint_series(lib.layers.concat([net, paf_nets[-1]], axis=-1))
        seg_nets = self.seg_series(net)
        return paf_nets, joint_nets, seg_nets


def model_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001,
        activation='relu',
        use_batch_norm=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    with lib.engine.arg_scope(
            [lib.contrib.Conv2D, lib.contrib.SeparableConv2D],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer=lib.initializers.truncated_normal(mean=0., stddev=0.03),
            activation=activation,
            normalizer=lib.layers.BatchNorm if use_batch_norm else None):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params):
            with lib.engine.arg_scope([lib.layers.MaxPool2D], padding='SAME') as arg_sc:
                return arg_sc
