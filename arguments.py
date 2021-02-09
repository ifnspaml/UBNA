from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

DEFAULT_IMAGES_KITTI = 57874
DEFAULT_ADAPTATION_BATCH_SIZE = 6
DEFAULT_SEG_BATCH_SIZE = 6
DEFAULT_BATCHES_PER_EPOCH = DEFAULT_IMAGES_KITTI // DEFAULT_ADAPTATION_BATCH_SIZE


class ArgumentsBase(object):
    DESCRIPTION = 'UBNA Arguments'

    def __init__(self):
        self.ap = ArgumentParser(
            description=self.DESCRIPTION,
            formatter_class=ArgumentDefaultsHelpFormatter
        )

    def _harness_init_system(self):
        self.ap.add_argument(
            '--sys-cpu', default=False, action='store_true',
            help='Disable GPU acceleration'
        )

        self.ap.add_argument(
            '--sys-num-workers', type=int, default=3,
            help='Number of worker processes to spawn per DataLoader'
        )

        self.ap.add_argument(
            '--sys-best-effort-determinism', default=False, action='store_true',
            help='Try and make some parts of the training/validation deterministic'
        )

    def _harness_init_model(self):
        self.ap.add_argument(
            '--model-type', type=str, default='resnet', choices=('resnet', 'vgg'),
            help='Type of model, which determines the network architecture'
        )

        self.ap.add_argument(
            '--model-num-layers', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the adaptation and segmentation encoder'
        )

        self.ap.add_argument(
            '--model-num-layers-vgg', type=int, default=16, choices=(11, 13, 16, 19),
            help='Number of VGG Layers in the adaptation and segmentation encoder'
        )

        self.ap.add_argument(
            '--experiment-class', type=str, default='iccv_experiments',
            help='Folder containing the current experiment series'
        )

        self.ap.add_argument(
            '--model-name', type=str, default='ubna',
            help='A nickname for this model'
        )

        self.ap.add_argument(
            '--model-load', type=str, default=None,
            help='Load a model state from a state directory containing *.pth files'
        )

        self.ap.add_argument(
            '--model-disable-lr-loading', default=False, action='store_true',
            help='Do not load the training state but only the model weights'
        )

    def _harness_init_segmentation(self):
        self.ap.add_argument(
            '--segmentation-validation-resize-height', type=int, default=512,
            help='Segmentation images are resized to this height prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-validation-resize-width', type=int, default=1024,
            help='Segmentation images are resized to this width prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-validation-loaders', type=str, default='cityscapes_validation',
            help='Comma separated list of segmentation dataset loaders from loaders/segmentation/validation.py to '
                 'use for validation'
        )

        self.ap.add_argument(
            '--segmentation-validation-batch-size', type=int, default=1,
            help='Batch size for segmentation validation'
        )

        self.ap.add_argument(
            '--segmentation-eval-num-images', type=int, default=20,
            help='Number of generated images to store to disk during evaluation'
        )

        self.ap.add_argument(
            '--segmentation-eval-remaps', type=str, default='none',
            help='Segmentation label remap modes for reduced number of classes, can be "none" (19 classes), '
                 '"synthia_16" (16 classes) or "synthia_13" (13 classes)'
        )

    def _training_init_train(self):
        self.ap.add_argument(
            '--train-batches-per-epoch', type=int, default=DEFAULT_BATCHES_PER_EPOCH,
            help='Number of batches we consider in an epoch'
        )

        self.ap.add_argument(
            '--train-num-epochs', type=int, default=20,
            help='Number of epochs to train for'
        )

        self.ap.add_argument(
            '--train-checkpoint-frequency', type=int, default=5,
            help='Number of epochs between model checkpoint dumps'
        )

        self.ap.add_argument(
            '--train-tb-frequency', type=int, default=500,
            help='Number of steps between each info dump to tensorboard'
        )

        self.ap.add_argument(
            '--train-print-frequency', type=int, default=2500,
            help='Number of steps between each info dump to stdout'
        )

        self.ap.add_argument(
            '--train-learning-rate', type=float, default=1e-4,
            help='Initial learning rate to train with',
        )

        self.ap.add_argument(
            '--train-scheduler-step-size', type=int, default=15,
            help='Number of epochs between learning rate reductions',
        )

        self.ap.add_argument(
            '--train-weight-decay', type=float, default=0.0,
            help='Weight decay to train with',
        )

        self.ap.add_argument(
            '--train-weights-init', type=str, default='pretrained', choices=('pretrained', 'scratch'),
            help='Initialize the encoder networks with Imagenet pretrained weights or start from scratch'
        )

    def _training_init_adaptation(self):
        self.ap.add_argument(
            '--adaptation-training-loaders', type=str, default='kitti_kitti_train',
            help='Comma separated list of adaptation dataset loaders from loaders/adaptation/train.py to use '
                 'for training'
        )

        self.ap.add_argument(
            '--adaptation-training-batch-size', type=int, default=DEFAULT_ADAPTATION_BATCH_SIZE,
            help='Batch size for adaptation training'
        )

        self.ap.add_argument(
            '--adaptation-num-batches', type=int, default=50,
            help='Trains the model for only this number of batches and then stops the whole training.'
                 'Mind that default value of 0 means, that this option will not be used'
        )

        self.ap.add_argument(
            '--adaptation-xlsx-frequency', type=int, default=1,
            help='Number of steps between each info dump to xlsx file'
        )

        self.ap.add_argument(
            '--adaptation-resize-height', type=int, default=192,
            help='adaptation images are resized to this height'
        )

        self.ap.add_argument(
            '--adaptation-resize-width', type=int, default=640,
            help='Adaptation images are resized to this width'
        )

        self.ap.add_argument(
            '--adaptation-crop-height', type=int, default=192,
            help='Adaptation images are cropped to this height'
        )

        self.ap.add_argument(
            '--adaptation-crop-width', type=int, default=640,
            help='Adaptation images are cropped to this width'
        )

        self.ap.add_argument(
            '--adaptation-mode-sequential', type=str, default='none',
            choices=('none', 'batch_shrinking', 'layer_shrinking')
        )

        self.ap.add_argument(
            '--adaptation-alpha-batch', type=float, default=0.1,
            help='Determine how fast momentum is shrinking depending on the number of batches trained'
        )

        self.ap.add_argument(
            '--adaptation-alpha-layer', type=float, default=0.1,
            help='Determines how fast momentum is shrinking depending on the layer depth of the BN Layer within '
                 'the architecture'
        )

        self.ap.add_argument(
            '--adaptation-batchnorm-momentum', type=float, default=0.1,
            help='Momentum for the BatchNorm layer of the shared encoder'
        )

    def _training_init_segmentation(self):
        self.ap.add_argument(
            '--segmentation-training-loaders', type=str, default='cityscapes_train',
            help='Comma separated list of segmentation dataset loaders from loaders/segmentation/train.py to use '
                 'for training'
        )

        self.ap.add_argument(
            '--segmentation-training-batch-size', type=int, default=DEFAULT_SEG_BATCH_SIZE,
            help='Batch size for segmentation training'
        )

        self.ap.add_argument(
            '--segmentation-resize-height', type=int, default=512,
            help='Segmentation images are resized to this height prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-resize-width', type=int, default=1024,
            help='Segmentation images are resized to this width prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-crop-height', type=int, default=192,
            help='Segmentation images are cropped to this height'
        )

        self.ap.add_argument(
            '--segmentation-crop-width', type=int, default=640,
            help='Segmentation images are cropped to this width'
        )

    def _parse(self):
        return self.ap.parse_args()


class TrainingArguments(ArgumentsBase):
    DESCRIPTION = 'UBNA training arguments'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_segmentation()
        self._training_init_train()
        self._training_init_segmentation()

    def parse(self):
        opt = self._parse()

        return opt


class AdaptationArguments(ArgumentsBase):
    DESCRIPTION = 'UBNA training arguments'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_segmentation()
        self._training_init_train()
        self._training_init_adaptation()

    def parse(self):
        opt = self._parse()

        return opt


class SegmentationEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'UBNA Segmentation Evaluation'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_segmentation()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_weights_init = 'scratch'

        return opt
