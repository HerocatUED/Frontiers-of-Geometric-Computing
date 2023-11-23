import argparse
import torch
from model import MLPnet
from trainer import Trainer
from utils import load_data, decode


def main(args):

    data = load_data(args.input_path)
    model = None

    if args.load: # load a model
        model = torch.load(args.model_path, map_location=torch.device("cuda"))
    else: # train a model
        model = MLPnet()
        trainer = Trainer(model=model, epoch_num=args.epochs)
        trainer.train(data)
        torch.save(model, args.model_path)

    decode(model, args.output_path, resolution=args.resolution, min=torch.min(data[:,:3]), max=torch.max(data[:,:3]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2400,
        help="Number of trainning epochs"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default='gargoyle.xyz',
        help="Path of input file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='output.obj',
        help="Path of output file"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of output file"
    )
    parser.add_argument(
        "--load",
        type=bool,
        default=True,
        help="True: Load a model(need cuda); False: Train a model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='model.pt',
        help="Path for load or save model"
    )

    args = parser.parse_args()
    main(args)