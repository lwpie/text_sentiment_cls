import argparse
import os

from torch import default_generator, equal


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # record args in configuration
    with open(os.path.join(args.output_dir,'config.txt'),'w') as f:
        for elem in vars(args):
            f.write(f'{elem}---->{eval(f"args.{elem}")}\n')

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str,
                        default='/data/ovo/text_sentiment_data')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--exp_name')
    args = parser.parse_args()

    main(args)
