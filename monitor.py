import sys
import torch

from tensorboardX import SummaryWriter

class Logger(object):

    def __init__(self, args):
        super(Logger, self).__init__()
        self.enabled = args.enable_logging
        self.log_steps = args.log_steps
        self.args = args
        self.writer = None
        self.comment = args.comment if hasattr(args, 'comment') else args.approach
        self.iteration = 0
        self.create()

    def create(self):
        self.writer = SummaryWriter(comment=self.comment)

    def reset_iter(self):
        self.iteration = 0

    def log_hist(self, tag, value, i=None):
        if not self.enabled:
            return
        if i is None:
            i = self.iteration
        if self.iteration % self.log_steps == 0:
            self.writer.add_histogram(tag, value, i, bins='auto')

    def log_scalar(self, tag, value, i=None, alt=None):
        if self.enabled:
            if i is None:
                i = self.iteration
            self.writer.add_scalar(tag, value, i)
        elif not self.enabled and alt:
            alt(tag, value)

    def inc_iter(self):
        self.iteration += 1