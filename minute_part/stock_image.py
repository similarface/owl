# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 下午11:27
# @Author  : similarface
# @Site    :
# @File    : class StockImage.py
# @Software: PyCharm
import numpy as np
from numpy import ndarray
from PIL import Image
import time
import pandas as pd
import os
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
from statistics import mean

class StockImageParsing:

    def binarization(self, image_arr: ndarray, boundary=128):
        """
        二值化 将非有效信息区域归为0
        :param image_arr: 图像数组
        :param boundary:  颜色分界线
        :return:
        """
        for i in range(image_arr.shape[0]):
            for j in range(image_arr.shape[1]):
                if image_arr[i, j] > boundary:
                    image_arr[i, j] = 0
                else:
                    image_arr[i, j] = 1
        return image_arr


class EastmoneyStockImageParsing(StockImageParsing):
    """
    东方财富股票图片识别
    """

    def __init__(self):
        self.classifier = self.get_classifier()

    def horizontal_split(self, image_arr: ndarray):
        '''
        #水平切割 行切割
        '''
        horizontal_split = []
        flag = False
        for row_index in range(image_arr.shape[0]):
            if any(image_arr[row_index, :] == 1) and not flag:
                horizontal_split.append(row_index)
                flag = True
            if flag and all(image_arr[row_index, :] == 0):
                horizontal_split.append(row_index)
                flag = False
        return horizontal_split

    def load_rgb_img(self, fpath):
        """
        获取图片的RGB model
        :param fpath:
        :return:
        """
        return Image.open(fpath).convert("RGB")

    def crop_gray_image(self, img_obj, left, upper, right, down):
        """
        裁剪图片
        :param img_obj:
        :param left:
        :param upper:
        :param right:
        :param down:
        :return:
        """
        _box = (left, upper, right, down)
        return img_obj.crop(_box).convert("L")

    def get_stock_price_area_arr(self, stock_image_path):
        """
        获取股价均线图区域 -> array
        :param stock_image_path:
        :return: np.array
        """
        image_obj = self.load_rgb_img(stock_image_path)
        r, g, b = image_obj.split()
        # 股价所在区域
        stock_line_image = self.crop_gray_image(r, 59, 22, 661, 419)
        stock_line_arr = np.array(stock_line_image)
        return stock_line_arr

    def get_stock_minutes_price(self, stock_image_path):
        """
        获取股票 的分钟价格
        :param stock_id:
        :return:
        """
        try:
            k_line_arr = self.get_stock_price_area_arr(stock_image_path)
            day_str = os.path.basename(os.path.dirname(stock_image_path))
            x_lables = self.stock_k_line_x_lables(run_day=day_str, k_line_arr=k_line_arr)
            y_lables = self.get_stock_line_y_lables(stock_image_path, k_line_arr)
            stock_id = os.path.splitext(os.path.basename(stock_image_path))[0]
            results = []
            for col in range(k_line_arr.shape[1]):
                stock_timestamp = x_lables[col]
                stock_timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stock_timestamp))
                col_data = k_line_arr[:,col]
                row_indexs = np.where(col_data < 5)
                stock_prices = y_lables[row_indexs]
                if stock_prices.size > 0:
                    stock_price = stock_prices.mean()
                    #results.append([stock_id, stock_price, stock_timestamp, stock_timestr])
                    results.append([stock_id, stock_price, stock_timestamp])
            return pd.DataFrame(results, columns=['id', 'price', 'timestamp'])
        except Exception:
            return pd.DataFrame([], columns=['id', 'price', 'timestamp'])

    def stock_k_line_x_lables(self, run_day=None, k_line_arr=None):
        """
        x_axis
        :param run_day:
        :param k_line_arr:
        :return:
        """
        morning_begin = self.formate_str_to_time(f"{run_day} 9:30:00")
        morning_end = self.formate_str_to_time(f"{run_day} 11:30:00")
        afternoon_begin = self.formate_str_to_time(f"{run_day} 13:00:00")
        afternoon_end = self.formate_str_to_time(f"{run_day} 15:00:00")
        x_axis = np.append(
            np.arange(morning_begin, morning_end, (morning_end - morning_begin) / (k_line_arr.shape[1] / 2)),
            np.arange(afternoon_begin, afternoon_end, (afternoon_end - afternoon_begin) / (k_line_arr.shape[1] / 2)))
        return x_axis

    def formate_str_to_time(self, time_str):
        return time.mktime(time.strptime(time_str, "%Y%m%d %H:%M:%S"))

    def get_stock_line_y_lables(self, stock_image_path, stock_line_arr):
        y_prices = self.get_y_prices_axle(stock_image_path)
        y_prices_f = np.array(y_prices, dtype=float)

        y_prices_max = y_prices_f.max()
        y_prices_min = y_prices_f.min()
        y_axis = np.arange(y_prices_max, y_prices_min, -(y_prices_max - y_prices_min) / stock_line_arr.shape[0])
        return y_axis

    def get_classifier(self):
        # classifier = svm.SVC(gamma=0.001)
        # base_dir = os.path.dirname(__file__)
        # train_dir = os.path.join(base_dir,"trains")
        # trains_set = defaultdict(list)
        # digits_map = {"p": "."}
        # for filename in os.listdir(train_dir):
        #     if filename != '.DS_Store':
        #         fpath = os.path.join(train_dir, filename)
        #         target = digits_map.get(filename, filename)
        #         fpaths = [os.path.join(fpath, _) for _ in os.listdir(fpath) if _.endswith("png")]
        #         for fpath in fpaths:
        #             item_data = self.binarization(Image.open(fpath).convert("L"))
        #             trains_set[target].append(self.pad_matrix(item_data))
        # data = []
        # targets = []
        # for key in trains_set:
        #     _data = trains_set[key]
        #     data = data + _data
        #     targets = targets + [key] * len(_data)
        #
        # datas = np.array(data)
        # targets = np.array(targets)
        #
        # X_train, X_test, y_train, y_test = train_test_split(datas, targets, test_size=0.5, shuffle=True)
        # classifier.fit(X_train, y_train)
        #
        # return classifier
        _base_dir = os.path.dirname(__file__)
        _model_file = os.path.join(_base_dir, "models", "price_svm.pickle")
        _model = pickle.load(open(_model_file, 'rb'))
        return _model

    def get_y_prices_axle(self, stock_image_path):
        classifier = self.get_classifier()
        stock_prices = []
        for row_mt in self.crop_stock_price_letter_arr(stock_image_path):
            stock_prices.append(''.join(classifier.predict(np.array(row_mt)).tolist()))
        return stock_prices

    def crop_stock_price_letter_arr(self, stock_image_path):
        '''
        切分股票价格为字母数组
        '''
        all_letters = self.crop_stock_price_letter(stock_image_path)
        letters = []
        for rows in all_letters:
            letters.append([self.pad_matrix(self.binarization(_)) for _ in rows])
        return letters

    @staticmethod
    def pad_matrix(item_data):
        """
        # 填充矩阵 分割的图片大小不一 需要统一大小
        :param item_data:
        :return:
        """
        if item_data.shape == (9, 6):
            return item_data.reshape(54)
        else:
            # 主要是列
            item_data_cols_pad = 6 - item_data.shape[1]
            new_item_data = np.pad(item_data, ((0, 0), (0, item_data_cols_pad)), 'constant')
            return new_item_data.reshape(54)

    def row_split_index(self, im_arr):
        '''
        #水平切割 行切割 返回索引
        '''
        row_split = None
        flag = False
        for row_index in range(im_arr.shape[0]):
            if any(im_arr[row_index, :] == 1) and not flag:
                row_split.append(row_index)
                flag = True
            if flag and all(im_arr[row_index, :] == 0):
                row_split.append(row_index)
                flag = False
        return row_split

    def crop_stock_price_letter(self, stock_image_path):
        '''
        切分股票Yaxis 价格为字母
        :return:
        '''
        image_obj = self.load_rgb_img(stock_image_path)
        r, g, b = image_obj.split()
        # 股价所在区域
        _box = (0, 0, 58, 420)
        stock_y_price_image = self.crop_gray_image(b, 0, 0, 58, 420)
        stock_axis_price_binarray = self.binarization(stock_y_price_image)
        # 获取行切割位置
        rows_index = self.list_chunks_each(self.row_split(stock_axis_price_binarray), 2)
        all_letters = []
        for row_index in rows_index:
            rows_arr = stock_axis_price_binarray[row_index[0]:row_index[1], :]
            # 行切割后的区间进去列切割
            cols_index = self.list_chunks_each(self.col_split(rows_arr), 2)
            row_letters = []
            for col_index in cols_index:
                # 行切割 & 列切割 的交叉区间就是图像股票价格每个数字的区域
                letter = self.crop_gray_image(image_obj, col_index[0], row_index[0], col_index[1], row_index[1])
                row_letters.append(letter)
            all_letters.append(row_letters)
        return all_letters

    @staticmethod
    def crop_gray_image(img_obj, left, upper, right, down):
        _box = (left, upper, right, down)
        return img_obj.crop(_box).convert("L")

    @staticmethod
    def list_chunks_each(arr, n):
        """
        分割成一个中有n个
        :param arr:
        :param n:
        :return:
        """
        return [arr[i:i + n] for i in range(0, len(arr), n)]

    @staticmethod
    def col_split(im_arr):
        '''
        列切割 垂直切割
        '''
        col_split = []
        flag = False
        for col_index in range(im_arr.shape[1]):
            if any(im_arr[:, col_index] == 1) and not flag:
                col_split.append(col_index)
                flag = True
            if flag and all(im_arr[:, col_index] == 0):
                col_split.append(col_index)
                flag = False
        return col_split

    @staticmethod
    def row_split(im_arr):
        '''
        #水平切割 行切割
        '''
        row_split = []
        flag = False
        for row_index in range(im_arr.shape[0]):
            if any(im_arr[row_index, :] == 1) and not flag:
                row_split.append(row_index)
                flag = True
            if flag and all(im_arr[row_index, :] == 0):
                row_split.append(row_index)
                flag = False
        return row_split

    @staticmethod
    def binarization(img_obj, limit=143) -> np.array:
        '''
        # 二值化 将非有效信息区域归为0
        :return:
        '''
        im = np.array(img_obj)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im[i, j] > limit:
                    im[i, j] = 0
                else:
                    im[i, j] = 1
        return im

import time
if __name__ == '__main__':
    sobj = EastmoneyStockImageParsing()
    base_dir = "/mnt/stock_images_collect/"
    out_dir = "/home/ywb/tmp/"
    for f_dir in os.listdir(base_dir):
        results_data = []
        if len(f_dir)==8 and f_dir.startswith('20'):
            f_dir_path = os.path.join(base_dir, f_dir)
            fpaths = [os.path.join(f_dir_path,_) for _ in os.listdir(f_dir_path) if _.endswith('png')]
            for fpath in fpaths:
                f_data = sobj.get_stock_minutes_price(fpath)
                if not f_data.empty:
                    results_data.append(f_data)
        if len(results_data) > 0:
            df = pd.concat(results_data)
            df.to_csv(os.path.join(out_dir,f_dir),header=False)