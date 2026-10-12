{\rtf1\ansi\ansicpg936\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww5580\viewh7400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch.nn as nn\
\
\
def zero_initialize(model):\
    for layer in model.modules():\
        if isinstance(layer, nn.Linear):\
            nn.init.constant_(layer.weight, 0.0)\
            nn.init.constant_(layer.bias, 0.0)\
\
\
def xavier_initialize(model):\
    for layer in model.modules():\
        if isinstance(layer, nn.Linear):\
            nn.init.xavier_normal_(layer.weight)\
            nn.init.constant_(layer.bias, 0.0)\
\
\
def he_initialize(model):\
    for layer in model.modules():\
        if isinstance(layer, nn.Linear):\
            nn.init.kaiming_normal_(\
                layer.weight,\
                nonlinearity="relu"\
            )\
            nn.init.constant_(layer.bias, 0.0)\
\
\
def get_gradient_norms(model):\
    gradient_norms = \{\}\
\
    for name, parameter in model.named_parameters():\
        if parameter.grad is not None:\
            gradient_norms[name] = parameter.grad.norm().item()\
\
    return gradient_norms\
\
\
def print_gradient_norms(model):\
    gradient_norms = get_gradient_norms(model)\
\
    for name, grad_norm in gradient_norms.items():\
        print(f"\{name\}: \{grad_norm:.8f\}")}