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
import pickle

class CompatibleLlamaTokenizer(object):
    def __init__(self, model_name_or_path, max_l=512):
        super().__init__()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)

        self.max_l = max_l  # 保留 max_l 参数以兼容原有逻辑
        # 添加特殊标记的映射（如果原 Tokenizer 使用了 SOS/EOS/UKN）
        self.token2idx = self.tokenizer.get_vocab()  # 直接复用 Hugging Face 的词汇表
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode(self, sentence):
        """
        兼容原有 encode 方法：
        输入: str 或 list，输出: np.array([max_l], dtype=np.int64)
        """
        if isinstance(sentence, str):
            sentence = [sentence]  # 单句处理（Hugging Face 默认批处理）

        encoded = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_l - 1,  # no </s>
            add_special_tokens=True,  # 自动添加 <s> (SOS) 和 </s> (EOS)
            return_tensors="np"  # 返回 numpy 数组
        )['input_ids'][0]  # 取第一个（因为输入是单句）
        encoded = np.concatenate([encoded, [self.tokenizer.eos_token_id]])

        # pad to max_l
        padded = np.zeros(self.max_l, dtype=np.int64)
        length = min(len(encoded), self.max_l)
        padded[:length] = encoded[:length]
        return padded

    def decode(self, idx, ignore=["<s>", "</s>", "<pad>"]):
        """
        兼容原有 decode 方法：
        输入: np.array 或 list，输出: str（忽略特殊标记）
        """
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()
        ret = self.tokenizer.decode(idx, skip_special_tokens=True)
        return ret

    def decode_as_token(self, idx, ignore=["<s>", "</s>", "<pad>"]):
        """
        兼容原有 decode_as_token 方法：
        输入: np.array 或 list，输出: token 列表
        """
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()

        ret = []
        for cur_idx in idx:
            if cur_idx == self.tokenizer.eos_token_id:
                break
            cur_token = self.tokenizer.decode(cur_idx)
            if cur_token not in self.tokenizer.all_special_tokens:
                ret.append(cur_token)
        return ret


class Tokenizer():
    def __init__(self, token_dir, max_l, file_prefix='', english=False):
        with open(token_dir + '/%stoken2idx.json' % file_prefix, 'r') as fp:
            self.token2idx = json.load(fp)
        with open(token_dir + '/%sidx2token.json' % file_prefix, 'r') as fp:
            temp = json.load(fp)
            self.idx2token = {int(x): temp[x] for x in temp}  # json里int键保存变成str了
        self.max_l = max_l
        self.english = english

    def encode(self, sentence):
        """
        sentence: str or list
        """
        if self.english and isinstance(sentence, str):  # 英语str，先用空格分词
            sentence = sentence.split()
        ret = np.zeros(self.max_l, np.int64)  # 默认0为padding。np.int64会转换为torch.int64，在embedding中用
        ret[0] = self.token2idx['SOS']
        for i, x in enumerate(sentence[:self.max_l - 2]):
            if x not in self.token2idx:
                ret[i + 1] = self.token2idx['UKN']
            else:
                ret[i + 1] = self.token2idx[x]
        ret[min(self.max_l - 1, len(sentence) + 1)] = self.token2idx['EOS']
        return ret

    def decode(self, idx, ignore=['SOS', 'EOS', 'PAD']):
        """
        idx: array
        """
        ret = ''

        for i in range(len(idx)):
            if idx[i] not in self.idx2token:
                ret += str(idx[i]) + 'UKN' + ' '
                continue
            c = self.idx2token[idx[i]]
            if c == 'EOS' and c in ignore:
                break
            if c not in ignore:
                ret += c
                if self.english:
                    ret += ' '
        return ret

    def decode_as_token(self, idx, ignore=['SOS', 'EOS', 'PAD']):
        """
        idx: array
        """
        ret = []
        for i in range(len(idx)):
            if idx[i] not in self.idx2token:
                ret.append('UKN')
                continue
            c = self.idx2token[idx[i]]
            if c == 'EOS' and c in ignore:
                break
            if c not in ignore:
                ret.append(c)
        return ret


class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        # Used to deal with potential accidental disk read failure
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e:
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait * 2, 1000)


# def collate(data):
#    return [x[0] for x in data], [x[1] for x in data], [x[2] for x in data]

class CaptionDataset(BaseDataset):
    def decorate(self, report, delimiter, nes=False, sos_ix=1, eos_ix=2, number_indices=[29900]):
        # report: 1darray，为我的token。[0]为sos。把其中各句子前面都加上sos
        # 8.12.2022加入nes
        now = 1
        idx = 0
        new = True
        ret = np.zeros(report.shape, np.int64)
        seg = np.zeros(report.shape, np.int64)
        for i in range(1, len(report)):
            # 因为原来报告每个句子前面加一个sos，所以最后总长度可能超过原来长度上限。这时直接舍弃。后果是sentence decoder时舍弃掉的部分不会计算loss，可以接受
            if idx >= len(report) or (nes and report[i] == eos_ix):  # 2:eos
                break
                # 严格来说nes and report[i]==2时应该ret[idx]=2，然后seg不赋值还是0。否则最后一个句子最后会没有eos直接pad，word decoder解码时可能生成一个句号后面没有eos直接0
            if new:
                ret[idx] = sos_ix  # sos
                seg[idx] = now + 100
                idx += 1
                new = False
                if idx >= len(report):
                    break
            ret[idx] = report[i]
            seg[idx] = now
            if report[i] in delimiter and report[i + 1] not in number_indices:
                new = True
                now += 1
            idx += 1
        # print(nes, report, ret, seg)
        return ret, seg


class XRayDataset(CaptionDataset):
    def __init__(self, report_dir, split, tokenizer, n_views, test=False,
                 seg_caption=False, decorate_sentences=False, paths=None,
                 suffix='_cleaned2', nes=False, delimiter=None, augment=False,
                 image_size=224, image_pad='blank', sample_lv=False):
        # with open(report_dir + '/%s%s.txt' % (split, suffix), 'r') as fp:
        #     reader = csv.reader(fp, delimiter='\t')
        #     self.report = [row for row in reader]

        with open(f"{report_dir}/{split}.pkl", "rb") as f:
            self.report = pickle.load(f)

        self.tokenizer = tokenizer
        self.test = test
        self.seg_caption = seg_caption
        self.decorate_sentences = decorate_sentences
        if augment and not test:  # 11.11加的。IU数据集过拟合严重
            assert image_size == 224
            if not test:
                resize = [transforms.Resize(256), transforms.RandomCrop(224)]
                self.aug = RandomRotate(15)
            else:
                # resize = [transforms.Resize(256), transforms.CenterCrop(224)] #611上和下面的差不多
                resize = [transforms.Resize((224, 224))]
        else:
            resize = [transforms.Resize((image_size, image_size))]
        self.transform = transforms.Compose(resize +
                                            [transforms.ToTensor(),
                                             # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        self.nes = nes
        self.n_views = n_views
        self.image_size = image_size
        self.split = split
        self.sample_lv = sample_lv  # whether randomly sample fewer views for multi-view model on MIMIC
        assert image_pad in ['blank', 'repeat']
        self.image_pad = image_pad

    def __len__(self):
        return len(self.report)

    def find_images_for_uid(self, uid,
                            pattern="D:/work/projects/data/IU_xray/images/images_normalized/{uid}_IM-*.dcm.png"):  # todo1
        return glob.glob(pattern.format(uid=uid))

    def sample_s(self, text):
        # print(text)
        text = text.split()
        i = 0
        sentence = []
        for j, x in enumerate(text):
            if x == '.':
                sentence.append((i, j))  # [i,j]闭区间
                i = j + 1
        # print(sentence)
        l, r = random.sample(sentence, 1)[0]
        ret = ' '.join(text[l: r + 1])
        # print(ret)
        return ret

    def _try_getitem(self, idx):
        if self.test:
            random.seed(idx)
        # if Path(self.report[idx][0]).is_dir():  # Multi-view input
        #     # if self.report[idx][0][-3:]!='jpg': #report[idx][0]为文件夹，多图片输入
        #     paths = glob.glob(self.report[idx][0] + '/*')
        #     n_image = len(paths)
        #
        #     if len(paths) > self.n_views:
        #         paths = random.sample(paths, self.n_views)
        #     if self.sample_lv is True:
        #         paths = random.sample(paths, random.randint(1, len(paths)))
        #     elif self.sample_lv == 1:
        #         paths = random.sample(paths, 1)
        #     if self.image_pad == 'repeat' and len(paths) < self.n_views:
        #         temp = paths * (self.n_views // len(paths))
        #         paths = paths + random.sample(temp, self.n_views - len(paths))
        #
        # else:  # Single view input
        #     paths = [self.report[idx][0]]
        #     n_image = 1
        paths = self.report[idx][0]
        n_image = len(paths)

        images = []
        for path in paths:
            img = Image.open(path)
            images.append(img)
        for i in range(self.n_views - len(paths)):  # pad empty images
            images.append(Image.fromarray(np.zeros((self.image_size, self.image_size), dtype=np.uint8)))
        for i in range(len(images)):
            images[i] = images[i].convert('RGB')  # 转为三通道
        for i in range(len(images)):  # 多张图片
            images[i] = self.transform(images[i])
            if hasattr(self, 'aug'):
                images[i] = self.aug(images[i])
        if len(images) > 1:  # 多张图片
            images = torch.stack(images)
        else:
            images = images[0].unsqueeze(0)
        target = {'find': self.tokenizer.encode(self.report[idx][1])}

        if self.decorate_sentences:
            find_s, seg_s = self.decorate(report=target['find'],
                                          delimiter=[self.tokenizer.token2idx['.']],
                                          nes=self.nes,
                                          sos_ix=self.tokenizer.tokenizer.bos_token_id,
                                          eos_ix=self.tokenizer.tokenizer.eos_token_id,
                                          number_indices=[self.tokenizer.token2idx[str(cur_num)] for cur_num in range(
                                              10)]
                                          )  # if the token is "." and not followed by a number, it is an end of a sentence.
            target['find_s'] = find_s
            target['seg_s'] = seg_s
        target['len'] = len(target['find'])
        debug = {'paths': [self.report[idx][0]], 'n_views': n_image}  # 之前是paths。MIMIC 多视角paths不一样长会出问题
        return images, target, debug


class RandomRotate():
    def __init__(self, angle):
        assert angle >= 0
        self.angle = angle

    def __call__(self, img):
        """
        image:[3,h,w]
        """
        x = random.randint(-self.angle, self.angle)
        img = transforms.functional.rotate(img, x, interpolation=InterpolationMode.BILINEAR)
        return img


if __name__ == '__main__':
    # # test tokenizer
    # """
    # special_tokens:  ['<s>', '</s>', '<unk>', '<pad>', '<image>']
    # """
    # myTokenizer = CompatibleLlamaTokenizer("./tokenizer", max_l=512)
    # input_ids = myTokenizer.encode("Today is a good day.")
    # de_tokens = myTokenizer.decode_as_token(input_ids)
    # de_test = myTokenizer.decode(input_ids)

    """
    test dataset
    """
    myTokenizer = CompatibleLlamaTokenizer("./tokenizer", max_l=512)
    split="train"
    sample_lv=False
    dataset = XRayDataset(report_dir="D:/work/projects/SILC-main/preprocess/IU_xray",  # todo1
                          split=split, tokenizer=myTokenizer, test=(split == 'test'),
                          n_views=2, seg_caption=True,
                          decorate_sentences=True, paths=None, suffix="_R2Gen",
                          nes=True, delimiter="",
                          augment=(split == 'train' and False),
                          image_size=224, image_pad="blank",
                          sample_lv=(sample_lv if sample_lv else (False if split == 'train' else False)))
    images, target, debug=dataset[0]
    print(debug)
