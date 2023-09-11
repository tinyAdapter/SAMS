import torch
import argparse
from src.model import SAMS
from src.model import MoEDNN
from src.model.dnn import DNN
from src.model.sparsemax_verticalMoe import SparseMax_VerticalSAMS
from src.model.verticalMoE import VerticalSAMS
from src.model.verticalMoE_Plus import VerticalMoE_Predict_Sams


def initialize_model(args: argparse.Namespace):

    if args.net == "sams":
        return SAMS(args)

    if args.net == "moe_dnn":
        return MoEDNN(args.nfield, args.nfeat, args.data_nemb, args.moe_num_layers, args.moe_hid_layer_len, args.dropout, args.K)

    if args.net == "dnn":
        return DNN(args.nfield, args.nfeat, args.data_nemb, args.moe_hid_layer_len,args.dropout)

    if args.net == "vertical_sams":
        return VerticalSAMS(args)
    
    
    if args.net == "sparsemax_vertical_sams":
        return SparseMax_VerticalSAMS(args)
    
    
    # if args.net == "sparsemax_plus_predict_layer":
    #     return SparseMaxMoE_PlusPredict(args)
    
    if args.net == "vertical_predict_sams":
        return VerticalMoE_Predict_Sams(args)