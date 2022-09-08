import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size', default=2, type=int)
    # train
    ## files
    parser.add_argument('--train', default='eval/1/train-10000-30-800-1.0-l_f',
                             help="training data")
    parser.add_argument('--eval', default='eval/1/test-10000-30-800-1.0-l_f',
                             help="evaluation data")

    # training scheme
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--eval_batch_size', default=20, type=int)

    parser.add_argument('--lr', default=0.0005, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=400, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=512, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=3, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=200, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=200, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")