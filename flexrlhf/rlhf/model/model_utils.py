# coding: utf-8

import os

import torch.distributed as dist
from alignment.app.util import logger
import torch


# class PolicyModel(torch.nn.Module):
class PolicyModel():
    def __init__(self, role, model_path):
        self.role = role
        self.model_path = model_path
        self.module = 'xxx'
        # super().__init__()
        
    def get_data_len(self, data):
        if isinstance(data, dict):
            return len(list(data.values())[0])
        return len(data)
        
    def generate(self, data):
        logger.info(f'model: {self.role} generate at rank: {dist.get_rank()}, data_len: {self.get_data_len(data)}')
        return data

    def compute_log_prob(self, data):
        logger.info(f'model: {self.role} compute_log_prob at rank: {dist.get_rank()}, data_len: {self.get_data_len(data)}')
        return data

    def update_actor(self, data):
        logger.info(f'model: {self.role} train at rank: {dist.get_rank()}, data_len: {self.get_data_len(data)}')
        return data

    def eval(self):
        pass

    def train(self):
        pass

# class ValueModel(torch.nn.Module):
class ValueModel():
    def __init__(self, role, model_path):
        self.role = role
        self.model_path = model_path
        self.module = 'xxx'

    def get_data_len(self, data):
        if isinstance(data, dict):
            return len(list(data.values())[0])
        return len(data)        

    def forward_value(self, data):
        logger.info(f'model: {self.role} forward_value at rank: {dist.get_rank()}, data_len: {self.get_data_len(data)}')
        return data

    def update_critic(self, data):
        logger.info(f'model: {self.role} train at rank: {dist.get_rank()}, data_len: {self.get_data_len(data)}')
        return data
        
    def eval(self):
        pass        

    def train(self):
        pass        
##### from deepspeed code start

def create_hf_model(role, model_path):
    if role in ['actor', 'ref']:
        return PolicyModel(role, model_path)
    elif role in ['critic', 'reward']:
        return ValueModel(role, model_path)
    else:
        raise ValueError(f'Unsupported role: {role}')