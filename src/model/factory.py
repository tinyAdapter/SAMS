import torch
import argparse
from src.model import SAMS

def initialize_model(args:argparse.ArgumentParser)->torch.nn.Module:
    if args.net == "sams":
        return SAMS(args)
    
    