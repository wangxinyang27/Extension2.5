import torch

from EDSR_master.src import utility
from EDSR_master.src import data
from EDSR_master.src.model import Model
from EDSR_master.src import loss
from EDSR_master.src.option import args
from EDSR_master.src.trainer import Trainer
from torchsummary import summary

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def edsr_main():
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    edsr_main()
