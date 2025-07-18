# config.py

import os
import torch
import argparse

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # 数据路径参数
        parser.add_argument('--input_images_path', type=str, default="./data/input/",
                            help='path of input images(underwater images)')
        parser.add_argument('--label_images_path', type=str, default="./data/label/",
                            help='path of label images(clear images)')
        parser.add_argument('--test_images_path', type=str, default="./data/input/",
                            help='path of input images for testing')
        parser.add_argument('--GTr_test_images_path', type=str, default="./data/input/",
                            help='path of ground truth test images')

        # 训练控制参数
        parser.add_argument('--test', default=True, action='store_true')
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--step_size', type=int, default=400, help='Period of learning rate decay')
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--train_batch_size', type=int, default=1)
        parser.add_argument('--test_batch_size', type=int, default=1)
        parser.add_argument('--resize', type=int, default=256)
        parser.add_argument('--cuda_id', type=int, default=0)
        parser.add_argument('--print_freq', type=int, default=1)
        parser.add_argument('--snapshot_freq', type=int, default=2)
        parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
        parser.add_argument('--output_images_path', type=str, default="./data/output/")
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--eval_steps', type=int, default=1)
        #默认test_model = true
        parser.add_argument('--test_mode', action='store_true', default=True, help="enable test mode")

        args = parser.parse_args()
        
        # === 添加属性别名，兼容训练逻辑 ===
        


        # 将 argparse 参数保存为类属性
        for k, v in vars(args).items():
            setattr(self, k, v)
       
        self.epochs = self.num_epochs # 兼容训练代码中使用的参数名
        self.eval_interval = self.eval_steps  # 兼容字段名
        self.cuda_id = str(self.cuda_id)
        self.train_raw_image_path = self.input_images_path
        self.train_clear_image_path = self.label_images_path
        self.test_raw_image_path = self.test_images_path
        self.test_clear_image_path = self.GTr_test_images_path
        # 推理设备设置
        self.device = f"cuda:{self.cuda_id}" if torch.cuda.is_available() else "cpu"

        # 快照路径和输出路径自动创建
        os.makedirs(self.snapshots_folder, exist_ok=True)
        os.makedirs(self.output_images_path, exist_ok=True)

        # 显式保存 snapshot_dir 用于训练时模型保存
        self.snapshot_dir = self.snapshots_folder
         
# 实例化后暴露一个全局 config 对象供导入
config = Config()
