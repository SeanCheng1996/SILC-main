# -*- coding: utf-8 -*-
"""
Created on Thu May 12 20:41:14 2022

@author: Admin
"""

import json
import glob
import cv2
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import random
import time
import csv
import traceback
from transformers import LlamaTokenizerFast
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle


def find_images_for_uid(uid,
                        pattern="D:/work/projects/data/IU_xray/images/images_normalized/{uid}_IM-*.dcm.png"):  # todo1
    return glob.glob(pattern.format(uid=uid))


def csv_to_list(report_dir):
    """
    transfer csv with columns: uid, findings
    to list: [
                ( [uid_img_path1, uid_img_path2], "findings" ),
                ( [uid_img_path1, uid_img_path2], "findings" ),
                ...
                ]
    """

    """
    1. read report csv file and preprocess the report:
        1.1 filter out none findings or finding shorter than 10
        1.2 replace word appeared less than 3 times to <UNK>.
        1.3 mapping uid to image paths:
                if image paths >= 3, random sample 2;
                if image paths<2, drop the row

    """
    # 1. 读取原始数据
    df = pd.read_csv(report_dir)
    raw_data = list(zip(df['uid'].astype(str), df['findings'].astype(str)))

    # 2. 初步过滤文本
    filtered_data = [(uid, text) for uid, text in raw_data
                     if text.strip() and len(text.strip()) >= 10]

    # 3. 统计词频（用于UNK替换）
    all_words = []
    for _, text in filtered_data:
        if text[-1] == ".":
            text = text[:-1]
        all_words.extend(text.lower().split())
    word_counts = Counter(all_words)

    # 4. 处理图片路径并最终过滤
    final_data = []
    for uid, text in tqdm(filtered_data, desc="Processing images"):
        # 4.1 查找该uid对应的所有图片
        matched_images = find_images_for_uid(uid)

        # 4.2 图片数量检查
        if len(matched_images) < 2:
            continue  # 跳过不足2张图片的条目

        # 4.3 随机选择2张图片
        selected_images = random.sample(matched_images, 2)

        # 4.4 处理文本中的低频词
        processed_words = []
        for word in text.split():
            suffix = ""
            if word[-1] == ".":
                suffix = word[-1]
                word = word[:-1]
            if word_counts[word.lower()] < 3:
                processed_words.append(f"<unk>{suffix}")
            else:
                processed_words.append(f"{word}{suffix}")
        processed_text = ' '.join(processed_words)

        # 4.5 添加到最终结果
        final_data.append((selected_images, processed_text))
    return final_data


def split_list(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=71):
    train_val, test = train_test_split(
        data,
        test_size=test_ratio,
        random_state=random_state
    )

    # 然后从train_val中分出val
    # 注意val_ratio现在是相对于train_val的比例
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (train_ratio + val_ratio),  # 计算新的比例
        random_state=random_state
    )

    return train, val, test


def save_list(data, dest_path):
    with open(dest_path, "wb") as f:
        pickle.dump(data, f)


def read_list(dest_path):
    with open(dest_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


if __name__ == '__main__':
    all_list = csv_to_list(report_dir="D:/work/projects/data/IU_xray/indiana_reports.csv")
    train, val, test = split_list(all_list)
    save_list(train, "./train.pkl")
    save_list(val, "./val.pkl")
    save_list(test, "./test.pkl")

    train = read_list("./train.pkl")
    print(train[0])
    print(train[1])
    print(train[2])
