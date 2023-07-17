import torch
import argparse
from src.model import SAMS
from src.model import MoEDNN


def initialize_model(args: argparse.ArgumentParser) -> torch.nn.Module:

    if args.net == "sams":
        return SAMS(args)

    if args.net == "moe_dnn":
        return MoEDNN(args.nfield, args.nfeat, args.data_nemb, args.moe_num_layers, args.moe_hid_layer_len, args.dropout, args.K)

    if args.net == "dnn":
        return MoEDNN(args.nfield, args.nfeat, args.data_nemb, args.moe_num_layers, args.moe_hid_layer_len, args.dropout, 1)
