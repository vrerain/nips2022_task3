import argparse
import os


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        exer_n = 116
        user_n = 6468
        skill_n = 116
        self.add_argument('--exer_n', type=int, default=exer_n,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=skill_n,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=user_n,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=2,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=12,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.01,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')
        self.add_argument('--division_ratio', type=float,
                          default=0.8, help='the ratio for dividing data')
        self.add_argument('--batch_size', type=int,
                          default=64, help='batch_size')
        self.add_argument('--hidden_size', type=int,
                          default=80, help='hidden_size')
        self.add_argument('--num_layers', type=int,
                          default=1, help='num_layers')
