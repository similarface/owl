# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 下午11:27
# @Author  : similarface
# @Site    :
# @File    : class StockImage.py
# @Software: PyCharm
import numpy as np
from numpy import ndarray
from PIL.Image import Image
import time

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
    def binarization(self, image_arr):
        return self.binarization(image_arr, boundary=143)


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

    def get_stock_line_area_arr(self, stock_image_path):
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
        k_line_arr = self.get_stock_price_area_arr(stock_image_path)
        x_lables = self.stock_k_line_x_lables(k_line_arr)


        y_lables = get_stock_line_y_lables(stock_id, stock_line_arr)



        with open(f"{stock_id}.price.txt", 'w') as oper:
            for col in range(stock_line_arr.shape[1]):
                for row in range(stock_line_arr.shape[0]):
                    if stock_line_arr[row, col] < 5:
                        stock_price = y_lables.get(row)
                        stock_timestamp = x_lables.get(col)
                        stock_timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stock_timestamp))
                        oper.write(f"{stock_id}\t{stock_price}\t{stock_timestr}\t{stock_timestamp}\n")
                        # print(f"{stock_id}\t{stock_price}\t{stock_timestr}\t{stock_timestamp}")

    def stock_k_line_x_lables(self, run_day=None, k_line_arr=None):
        morning_begin = self.formate_str_to_time(f"{run_day} 9:30:00")
        morning_end = self.formate_str_to_time(f"{run_day} 11:30:00")
        afternoon_begin = self.formate_str_to_time(f"{run_day} 13:00:00")
        afternoon_end = self.formate_str_to_time(f"{run_day} 15:00:00")
        x_axis = np.append(np.arange(morning_begin, morning_end, (morning_end - morning_begin) / (k_line_arr.shape[1] / 2)),
                 np.arange(afternoon_begin, afternoon_end, (afternoon_end - afternoon_begin) / (k_line_arr.shape[1] / 2)))
        return x_axis

    def formate_str_to_time(self, time_str):
        return time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))

    def get_stock_line_y_lables(self, stock_image_path, stock_line_arr):
        y_prices = get_y_prices_axle(stock_id)
        y_prices_f = [float(_) for _ in y_prices]
        y_prices_max = max(y_prices_f)
        y_prices_min = min(y_prices_f)
        y_axis = np.arange(y_prices_max, y_prices_min, -(y_prices_max - y_prices_min) / stock_line_arr.shape[0])
        y_lables = dict(zip([idx for idx in range(len(y_axis))], y_axis))
        return y_lables

    def get_y_prices_axle(stock_id):
        stock_prices = []
        for row_mt in crop_stock_price_letter_arr(stock_id):
            stock_prices.append(''.join(classifier.predict(np.array(row_mt)).tolist()))
        return stock_prices

    def crop_stock_price_letter_arr(stock_id):
        '''
        切分股票价格为字母数组
        '''
        all_letters = crop_stock_price_letter(stock_id)
        letters = []
        for rows in all_letters:
            letters.append([pad_matrix(binarization(_)) for _ in rows])
        return letters

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
        rows_index = list_chunks_each(row_split(stock_price_binarray), 2)
        all_letters = []
        for row_index in rows_index:
            rows_arr = stock_price_binarray[row_index[0]:row_index[1], :]
            # 行切割后的区间进去列切割
            cols_index = list_chunks_each(col_split(rows_arr), 2)
            row_letters = []
            for col_index in cols_index:
                # 行切割 & 列切割 的交叉区间就是图像股票价格每个数字的区域
                letter = crop_gray_image(image_obj, col_index[0], row_index[0], col_index[1], row_index[1])
                row_letters.append(letter)
            all_letters.append(row_letters)
        return all_letters


    @classmethod
    def binarization(img_obj, limit=143) -> np.array():
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

