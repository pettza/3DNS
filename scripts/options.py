import argparse
import configargparse


def create_parser():
    arg_parser = configargparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    return arg_parser


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
    group.add_argument('--regularization_samples', type=int, default=120_000,
                       help='Number of samples for regularization.')
    group.add_argument('--no_empty_space', action='store_true',
                       help='If specified, Empty space loss is not used.')
    group.add_argument('--ewc', action='store_true',
                       help='If specified, ewc loss is added.')
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
