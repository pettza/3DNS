import argparse
import configargparse


def create_parser():
    arg_parser = configargparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add('-c', '--config_filepath', required=False, is_config_file=True,
                   help='Path to config file.')
    return arg_parser


def create_edit_parser():
    arg_parser = create_parser()
    training_group = add_edit_training_options(arg_parser)
    pretrained_group = add_pretrained_model_options(arg_parser)
    dataset_group = add_edit_dataset_options(arg_parser)
    interaction_group = add_interaction_options(arg_parser)
    return arg_parser, training_group, pretrained_group, dataset_group, interaction_group


def add_training_options(arg_parser):
    group = arg_parser.add_argument_group('Training options')
    group.add_argument('--model_dir', type=str, required=True,
                       help='Name of directory where summaries and checkpoints will be saved.')
    group.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate.')
    group.add_argument('--num_epochs', type=int, default=1_000_000,
                       help='Number of epochs to train for.')
    group.add_argument('--epochs_til_ckpt', type=int, default=10_000,
                       help='Numbr of epochs between checkpoints.')
    group.add_argument('--pretrain_epochs', type=int, default=100,
                       help='Number of epochs for pretraining')
    group.add_argument('--regularization_samples', type=int, default=120_000,
                       help='Number of samples for regularization.')
    return group                            


def add_edit_training_options(arg_parser):
    group = arg_parser.add_argument_group('Training options')
    group.add_argument('--model_dir', type=str, required=True,
                       help='Name of directory where summaries and checkpoints will be saved.')
    group.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate.')
    group.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs to train for.')
    group.add_argument('--epochs_til_ckpt', type=int, default=10_000,
                       help='Numbr of epochs between checkpoints.')
    group.add_argument('--regularization_samples', type=int, default=120_000,
                       help='Number of samples for regularization.')
    group.add_argument('--no_empty_space', action='store_true',
                       help='If specified, Empty space loss is not used.')
    group.add_argument('--ewc', action='store_true',
                       help='If specified, ewc loss is added.')
    return group


def add_edit_dataset_options(arg_parser):
    group = arg_parser.add_argument_group('Dataset options')
    group.add_argument('--num_interaction_samples', type=int, default=5000,
                       help='Number of samples for interaction.')
    group.add_argument('--num_model_samples', type=int, default=120000,
                       help='Number of samples from pretrained model.')
    return group


def add_pretrained_model_options(arg_parser):
    group = arg_parser.add_argument_group('Model options')
    group.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model.')
    return group


def add_interaction_options(arg_parser):
    group = arg_parser.add_argument_group('Interaction options')
    group.add_argument('--ox', type=float, default=0.,
                       help='X component of origin of interaction ray')
    group.add_argument('--oy', type=float, default=0.,
                       help='Y component of origin of interaction ray')
    group.add_argument('--oz', type=float, default=0.9,
                       help='Z component of origin of interaction ray')

    group.add_argument('--dx', type=float, default=0.,
                       help='X component of direction of interaction ray')
    group.add_argument('--dy', type=float, default=0.,
                       help='Y component of direction of interaction ray')
    group.add_argument('--dz', type=float, default=-1.,
                       help='Z component of direction of interaction ray')

    group.add_argument('--brush_radius', type=float, default=0.08,
                       help='The radius of the brush')
    group.add_argument('--brush_intensity', type=float, default=0.03,
                       help='The intensity of the brush')
    group.add_argument('--brush_type', choices=['linear', 'cubic', 'quintic', 'exp'], default='quintic',
                       help='The type of the brush')
    return group


def add_model_options(arg_parser):
    group = arg_parser.add_argument_group('Model options')
    group.add_argument('--hidden_layers', type=int, default=3,
                       help='The number hidden layers.')
    group.add_argument('--hidden_features', type=int, default=256,
                       help='The number of features in the hidden layers.')
    group.add_argument('--weight_norm', action='store_true',
                       help='If specified, weight normalization is applied to the layers.')
    return group
