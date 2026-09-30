
import numpy as np
import cv2
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
import glob
import os
from PIL import Image
import random
from plane_hu import motion_blur as mb
import math





class MOTION_BLUR():
    def __init__(self, target_resolution, txt_path, exposure_time):
        self.target_resolution = target_resolution
        self.dis_size = self.get_dis_size(txt_path,exposure_time=exposure_time)

    def img_padding(self, img, padding_range, pading_value):
        img = cv2.copyMakeBorder(img, padding_range, padding_range, padding_range, padding_range, \
                                 cv2.BORDER_CONSTANT, value=pading_value)  # 图像padding
        return img


    def blur_edge(self, img, cx, cy, shadow_mask, noise_mode=None):
        """
        对目标边缘及边缘内外区域进行多层高斯模糊（核心区域不模糊），并支持添加噪声
        :param img: 待处理的图像（BGR格式）
        :param cx: 目标中心x坐标
        :param cy: 目标中心y坐标
        :param shadow_mask: 阴影掩码(255为背景，非255为目标区域)
        :param noise_mode: 噪声模式，可选值：None(无噪声)、'gaussian'(高斯噪声)、'salt_pepper'(椒盐噪声)、'poisson'(泊松噪声)
        :return: 处理后的图像
        """
        # 统计目标区域总像素（非255的像素）
        target_pixel_count = np.sum(shadow_mask != 255)
        # print(f"target_pixel_count:{target_pixel_count}")
        if target_pixel_count == 0:
            return img  # 无目标区域时直接返回原图

        # 1. 根据目标像素数量确定内外扩展层数
        if target_pixel_count < 30:
            inner_layers = 1  # 仅边缘和外扩1层
            outer_layers = 2
        elif 30 <= target_pixel_count < 60:
            inner_layers = 2  # 内扩1层，外扩2层
            outer_layers = 3
        elif 60 <= target_pixel_count < 100:
            inner_layers = 3  # 内扩2层，外扩3层
            outer_layers = 4
        else:
            inner_layers = 4  # 内扩3层，外扩4-6层（随机）
            outer_layers = random.randint(5, 7)
        print(f"inner:{inner_layers}, outter:{outer_layers}")
        # 生成二值掩码：目标区域（非255）为1，背景（255）为0
        target_mask = (shadow_mask != 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)  # 形态学操作核

        # 2. 计算内层模糊区域（向目标内部扩展）
        inner_regions = []
        current_erode = target_mask.copy()
        inner_valid = True  # 标记内层是否足够

        for i in range(inner_layers):
            # 每次腐蚀得到更内层
            next_erode = cv2.erode(current_erode, kernel, iterations=1)
            # 当前内层 = 上一次腐蚀 - 当前腐蚀（即两层之间的区域）
            inner_region = current_erode - next_erode

            # 检查内层是否为空（不足）
            if np.sum(inner_region) <= 0:
                inner_valid = False
                break

            inner_regions.append(inner_region)
            current_erode = next_erode.copy()

        # 处理内层不足的情况：按target_pixel_count < 30处理
        if not inner_valid:
            inner_layers = 0
            outer_layers = 3
            inner_regions = []
            current_erode = target_mask.copy()
            # 重新计算内扩1层
            next_erode = cv2.erode(current_erode, kernel, iterations=1)
            inner_region = current_erode - next_erode
            inner_regions.append(inner_region)
            current_erode = next_erode.copy()

        # 3. 计算边缘区域（最内层与原始目标的边界）
        # edge_region = current_erode  # 经过inner_layers次腐蚀后的剩余区域与原始目标的差
        edge_region = target_mask - current_erode

        # 4. 计算外层模糊区域（向背景扩展）
        outer_regions = []
        current_dilate = target_mask.copy()

        for _ in range(outer_layers):
            # 每次膨胀得到更外层
            next_dilate = cv2.dilate(current_dilate, kernel, iterations=1)
            # 当前外层 = 当前膨胀 - 上一次膨胀（即两层之间的区域）
            outer_region = next_dilate - current_dilate
            outer_regions.append(outer_region)
            current_dilate = next_dilate.copy()

        # 合并所有需要模糊的区域（内层 + 边缘 + 外层）
        blur_regions = inner_regions + [edge_region] + outer_regions

        # 5. 对所有模糊区域应用高斯模糊（3x3核，sigma=1）
        result = img.copy()
        for region in blur_regions:
            # 获取当前区域的坐标
            coords = np.where(region == 1)
            if len(coords[0]) == 0:  # 跳过空区域
                continue

            # 计算区域边界（确保不超出图像范围）
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            min_y = max(0, min_y)
            max_y = min(img.shape[0], max_y + 1)
            min_x = max(0, min_x)
            max_x = min(img.shape[1], max_x + 1)

            # 提取ROI并模糊
            roi = result[min_y:max_y, min_x:max_x]
            blurred_roi = cv2.GaussianBlur(roi, (3, 3), sigmaX=0.35) # 整体模糊

            # 将模糊后的ROI替换回原区域
            region_roi = region[min_y:max_y, min_x:max_x]
            result[min_y:max_y, min_x:max_x][region_roi == 1] = blurred_roi[region_roi == 1]

        # 6. 根据噪声模式添加噪声（仅作用于模糊区域）
        if noise_mode is not None:
            # 合并所有模糊区域作为噪声区域
            noise_mask = np.zeros_like(target_mask, dtype=np.bool_)
            for region in blur_regions:
                noise_mask |= (region == 1)

            if not np.any(noise_mask):  # 无噪声区域时直接返回
                return result

            # 转换为float类型避免整数溢出
            img_float = result.astype(np.float32)

            # 根据噪声模式生成并应用噪声
            if noise_mode == 'gaussian':
                # 高斯噪声：均值0，方差5-20
                mean = 0
                var = random.uniform(2, 15) # 噪声强度
                sigma = np.sqrt(var)
                gauss_noise = np.random.normal(mean, sigma, result.shape) # 边缘区域
                img_float[noise_mask] += gauss_noise[noise_mask]

            elif noise_mode == 'salt_pepper':
                # 椒盐噪声：概率0.005-0.02
                prob = random.uniform(0.005, 0.02)
                # 在噪声区域内生成椒盐点掩码
                salt_mask = (np.random.choice([False, True], size=result.shape[:2], p=[1 - prob / 2, prob / 2])
                             & noise_mask)
                pepper_mask = (np.random.choice([False, True], size=result.shape[:2], p=[1 - prob / 2, prob / 2])
                               & noise_mask)
                # 应用椒盐噪声（三通道）
                for c in range(result.shape[2]):
                    img_float[salt_mask, c] = 255  # 盐噪声（白色）
                    img_float[pepper_mask, c] = 0  # 椒噪声（黑色）

            elif noise_mode == 'poisson':
                # 泊松噪声：基于图像灰度分布
                unique_vals = len(np.unique(result))
                vals = 2 ** np.ceil(np.log2(unique_vals))  # 转为2的幂次
                # 仅对噪声区域计算泊松噪声
                poisson_noise = np.random.poisson(img_float[noise_mask] * vals) / vals
                img_float[noise_mask] = poisson_noise

            else:
                raise ValueError(f"不支持的噪声模式: {noise_mode}，可选模式：'gaussian'、'salt_pepper'、'poisson'")

            # 裁剪值范围并转回uint8类型
            result = np.clip(img_float, 0, 255).astype(np.uint8)

        return result

    # def process_shadow(self, shadow_rotate, RS_img): # 目的处理遥感图像黑边
    #     """
    #     处理shadow_rotate：将有效区域（shadow_rotate!=255）中对应RS_img为黑边(0,0,0)的位置设为255
    #
    #     参数：
    #         shadow_rotate: 二维ndarray (h, w)，值为255表示无效区域
    #         RS_img: 三维ndarray (h, w, 3)，RGB图像，黑边为(0,0,0)
    #
    #     返回：
    #         更新后的shadow_rotate（原地修改，也可返回副本）
    #     """
    #     if np.min(RS_img)!=0:
    #         return shadow_rotate
    #     # 1. 生成有效区域掩码（shadow_rotate != 255的位置）
    #     valid_mask = shadow_rotate != 255  # 形状为(h, w)的布尔数组
    #
    #     # 2. 在有效区域中，筛选出RS_img为黑边的位置
    #     # 先获取有效区域的RS_img像素（形状为(m, 3)，m为有效像素数）
    #     valid_rs_pixels = RS_img[valid_mask]
    #     # 检查这些像素是否为黑边（三通道均为0），得到形状为(m,)的布尔数组
    #     is_black_edge = np.all(valid_rs_pixels == 0, axis=1)  # axis=1检查每个像素的3个通道
    #
    #     # 3. 找到需要更新为255的位置索引
    #     # 获取有效区域的坐标（rows和cols分别为有效像素的行、列索引）
    #     valid_rows, valid_cols = np.where(valid_mask)
    #     # 从有效区域中筛选出黑边对应的坐标
    #     black_rows = valid_rows[is_black_edge]
    #     black_cols = valid_cols[is_black_edge]
    #
    #     # 4. 更新shadow_rotate：将黑边对应的有效区域设为255
    #     shadow_rotate[black_rows, black_cols] = 255
    #
    #     return shadow_rotate  # 可选，因ndarray是原地修改
    def process_shadow(self, shadow_rotate, RS_img):  # 目的处理遥感图像黑边
        """
        处理shadow_rotate：将有效区域（shadow_rotate!=255）中对应RS_img为黑边(三通道均<10)的位置设为255

        参数：
            shadow_rotate: 二维ndarray (h, w)，值为255表示无效区域
            RS_img: 三维ndarray (h, w, 3)，RGB图像，黑边定义为三通道均小于10

        返回：
            更新后的shadow_rotate（原地修改，也可返回副本）
        """
        # 如果RS_img中最小像素值已大于等于10，说明无符合条件的黑边，直接返回
        if np.min(RS_img) >= 12:
            return shadow_rotate

        # 1. 生成有效区域掩码（shadow_rotate != 255的位置）
        valid_mask = shadow_rotate != 255  # 形状为(h, w)的布尔数组

        # 2. 在有效区域中，筛选出RS_img为黑边的位置（三通道均小于10）
        # 获取有效区域的RS_img像素（形状为(m, 3)，m为有效像素数）
        valid_rs_pixels = RS_img[valid_mask]
        # 检查这些像素是否为黑边（三通道均小于10），得到形状为(m,)的布尔数组
        is_black_edge = np.all(valid_rs_pixels < 12, axis=1)  # axis=1检查每个像素的3个通道

        # 3. 找到需要更新为255的位置索引
        # 获取有效区域的坐标（rows和cols分别为有效像素的行、列索引）
        valid_rows, valid_cols = np.where(valid_mask)
        # 从有效区域中筛选出黑边对应的坐标
        black_rows = valid_rows[is_black_edge]
        black_cols = valid_cols[is_black_edge]

        # 4. 更新shadow_rotate：将黑边对应的有效区域设为255
        shadow_rotate[black_rows, black_cols] = 255

        return shadow_rotate  # 可选，因ndarray是原地修改
    def mix_img(self, RS_img, plane_patch, plane_shadow, patch_cy, patch_cx, rotate_angle, cloud_mask=None, wj_index=None, wj_img=None, wj_angle=None,
                wwhh=None, RS_cxcy_list=None, diejia_flag = False, for_avg=False):
        # 将切片plane_patch边缘补255使得尺寸等于RS_img,得到plane_padding和shadow_padding
        # 这里的RS_img作为patch和飞机patch融合的时候，RS_img是分辨率和飞机统一后的大一点的patch，hw可能不相等
        # 当patch和原图融合的时候，RS_img就是原图512，plane_patch就是74
        RS_h, RS_w = RS_img.shape[:2]
        patch_h, patch_w = plane_patch.shape[:2]
        left = int(patch_cx - patch_w / 2)
        right = RS_w - patch_w - left
        top = int(patch_cy - patch_h / 2)
        bottom = RS_h - patch_h - top
        if left<0:
            # right -= left
            plane_patch = plane_patch[:, -left:, :]
            plane_shadow = plane_shadow[:, -left:]
            left = 0
        if right<0:
            # left -= right
            plane_patch = plane_patch[:, :patch_w+right, :]
            plane_shadow = plane_shadow[:, :patch_w+right]
            right = 0
        if top<0:
            # bottom -= top
            plane_patch = plane_patch[-top:, :, :]
            plane_shadow = plane_shadow[-top:, :]
            top = 0
        if bottom<0:
            # top -= bottom
            plane_patch = plane_patch[:patch_h+bottom, :, :]
            plane_shadow = plane_shadow[:patch_h+bottom, :]
            bottom = 0

        plane_padding = cv2.copyMakeBorder(plane_patch, top, bottom, left, right,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像padding
        shadow_padding = cv2.copyMakeBorder(plane_shadow, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像padding
        # 旋转，得到plane_rotate和shadow_rotate
        M = cv2.getRotationMatrix2D((RS_w / 2, RS_h / 2), rotate_angle, 1)  # rotate_angle 0-360, 向上为0,逆时针
        plane_rotate = cv2.warpAffine(plane_padding, M, (RS_w, RS_h))
        shadow_rotate = cv2.warpAffine(shadow_padding, M, (RS_w, RS_h), borderValue=255)
        shadow_rotate = np.ceil(shadow_rotate).astype('uint8')
        # shadow_rotate!=255的区域表示有效区域，对应坐标查找RS_img，如果RS_img对应区域为(0,0,0)表示黑边，此处的shadow_rotate=255,其中shadow_rotate为(h,w), RS_img为(h,w,3)
        shadow_rotate = self.process_shadow(shadow_rotate, RS_img)
        if for_avg: # 仅为计算平均灰度值，不做后续动作直接返回
            RS_img_gray = cv2.cvtColor(RS_img, cv2.COLOR_BGR2GRAY)
            plane_img_gray = cv2.cvtColor(plane_rotate, cv2.COLOR_BGR2GRAY)
            valid_count = np.sum(shadow_rotate != 255)
            if valid_count == 0:
                return 0, 0, 0
            RS_img_gray_avg = sum(RS_img_gray[shadow_rotate != 255]) / valid_count
            plane_img_gray_avg = sum(plane_img_gray[shadow_rotate != 255]) / valid_count
            # 先获取筛选后的数组
            mask = (shadow_rotate != 255) & (RS_img_gray != 0)
            filtered_rs_gray = RS_img_gray[mask]

            # 检查数组是否为空
            if filtered_rs_gray.size == 0:
                # 为空时设置默认值（根据实际需求调整，比如0）
                RS_patch_shadow_gray_min = 0
            else:
                # 不为空时计算最小值
                RS_patch_shadow_gray_min = np.min(filtered_rs_gray)

            # 返回结果
            return RS_img_gray_avg, plane_img_gray_avg, RS_patch_shadow_gray_min
        # 初始化结果图片
        result = RS_img.copy()

        plane_cloud_index = 2  # 表示无云
        if cloud_mask is not None:  # 如果有云层掩码
            plane_rand = random.random()  # 随机生成概率值
            # 判断飞机位置是否在云层区域（cloud_mask对应位置为True）
            if cloud_mask[int(patch_cx), int(patch_cy)]:  # 用飞机中心索引云层掩码
                if plane_rand < 0.2:  # 20%概率：飞机与云层混合（弱化飞机对比度）
                    # 赋值贴图
                    # 注意shadow_rotate因为运动模糊，产生了一些非0的阴影，比如183
                    result[shadow_rotate != 255] = result[shadow_rotate != 255] * 0.05 + plane_rotate[shadow_rotate != 255] * 0.95
                    # 有云的情况下模拟5%云层或背景
                    plane_cloud_index = 0  # 标识“云中”状态
                else:  # 80%概率：飞机覆盖云层（强对比度）
                    result[shadow_rotate != 255] = plane_rotate[shadow_rotate != 255]
                    plane_cloud_index = 1  # 标识“云上”状态
            else:  # 非云层区域：直接覆盖
                if diejia_flag == True:# 和背景叠加
                    result[shadow_rotate != 255] = plane_rotate[shadow_rotate != 255]*0.7 + result[shadow_rotate != 255]*0.3
                else:
                    result[shadow_rotate != 255] = plane_rotate[shadow_rotate != 255]
        else:  # 无云层：直接覆盖 # 实际仿真走这条路 # lhg
            if diejia_flag == True:  # 和背景叠加
                result[shadow_rotate != 255] = plane_rotate[shadow_rotate != 255] * 0.7 + result[shadow_rotate != 255] * 0.3
            else:
                result[shadow_rotate != 255] = plane_rotate[shadow_rotate != 255]

        if wj_img is not None:
            wj_img[wj_img[:, :, 3] == 0] = wj_img[wj_img[:, :, 3] == 0] * 0
            wj_rgb = cv2.merge([wj_img[:, :, 0], wj_img[:, :, 1], wj_img[:, :, 2]])
            wj_gray = cv2.cvtColor(wj_rgb, cv2.COLOR_RGB2GRAY)

            wj_resize = wj_gray.copy()
            if wj_index == 1:
                wj_resize = cv2.resize(wj_gray, (int(wwhh[0] * 2), int(wj_gray.shape[0] / wj_gray.shape[1] * 10 * 22)),
                                       interpolation=cv2.INTER_AREA)  # wwhh[1]
            if wj_index == 2:
                wj_resize = cv2.resize(wj_gray, (int(wwhh[0] * 4), int(wj_gray.shape[0] / wj_gray.shape[1] * 10 * 22)),
                                       interpolation=cv2.INTER_AREA)  # wwhh[1]
            if wj_index == 4:
                wj_resize = cv2.resize(wj_gray, (int(wwhh[0] * 4), int(wj_gray.shape[0] / wj_gray.shape[1] * 10 * 30)),
                                       interpolation=cv2.INTER_AREA)  # wwhh[1]

            wj_resize[wj_resize < 90] = 0
            wj_resize = wj_resize * 1.2
            wj_resize[wj_resize > 255] = 255

            wj_mask1 = np.ones_like(wj_resize)
            wj_mask2 = wj_mask1.copy()
            len_index = 10
            len_ = math.floor(wj_mask2.shape[0] / len_index)
            for ii in range(len_index):
                if ii < len_index - 1:
                    wj_mask2[ii * len_: (ii + 1) * len_, :] = wj_mask2[ii * len_: (ii + 1) * len_] * (1 - ii * 0.1)
                else:
                    wj_mask2[ii * len_:, :] = wj_mask2[ii * len_:, :] * (1 - ii * 0.1)

            wj_mask_resize = wj_mask1 * (wj_resize > 0) * wj_mask2
            H = 0.503 * (wwhh[1] + wj_resize.shape[0])

            wj_angle = wj_angle % 360
            wj_angle1 = wj_angle / 180 * math.pi
            wj_cx = patch_cy + H * math.sin(wj_angle1)
            wj_cy = patch_cx + H * math.cos(wj_angle1)

            wj_patch_h, wj_patch_w = wj_resize.shape[:2]
            left = int(wj_cx - wj_patch_w / 2)
            right = RS_w - wj_patch_w - left
            top = int(wj_cy - wj_patch_h / 2)
            bottom = RS_h - wj_patch_h - top
            crop_index_l, crop_index_r, crop_index_t, crop_index_b = 0, 0, 0, 0
            if left < 0:
                left1 = 0
                wj_cx = wj_cx - left
                crop_index_l = 1
            else:
                left1 = left
            if right < 0:
                right1 = 0
                crop_index_r = 1
            else:
                right1 = right
            if top < 0:
                wj_cy = wj_cy - top
                top1 = 0
                crop_index_t = 1
            else:
                top1 = top
            if bottom < 0:
                bottom1 = 0
                crop_index_b = 1
            else:
                bottom1 = bottom

            wj_padding = cv2.copyMakeBorder(wj_resize, top1, bottom1, left1, right1, \
                                            cv2.BORDER_CONSTANT, value=0)  # 图像padding
            wj_mask_padding = cv2.copyMakeBorder(wj_mask_resize, top1, bottom1, left1, right1, \
                                                 cv2.BORDER_CONSTANT, value=0)  # 图像padding

            # wj_mask_padding = wj_mask_padding*(wj_mask_padding>0.5)
            wj_M = cv2.getRotationMatrix2D((wj_cx, wj_cy), wj_angle, 1)  # rotate_angle 0-360, 向上为0,逆时针
            wj_rotate = cv2.warpAffine(wj_padding, wj_M, (wj_padding.shape[1], wj_padding.shape[0]))
            wj_mask_rotate = cv2.warpAffine(wj_mask_padding, wj_M, (wj_padding.shape[1], wj_padding.shape[0]))
            if crop_index_l == 1:
                wj_rotate = wj_rotate[:, -left:]
                wj_mask_rotate = wj_mask_rotate[:, -left:]
            if crop_index_r == 1:
                wj_rotate = wj_rotate[:, :right]
                wj_mask_rotate = wj_mask_rotate[:, :right]
            if crop_index_t == 1:
                wj_rotate = wj_rotate[-top:, :]
                wj_mask_rotate = wj_mask_rotate[-top:, :]
            if crop_index_b == 1:
                wj_rotate = wj_rotate[:bottom, :]
                wj_mask_rotate = wj_mask_rotate[:bottom, :]
            wj_rotate = wj_rotate[:1024, :1024]
            wj_mask_rotate = wj_mask_rotate[:1024, :1024]

            # 防止wj飞机重叠
            aa = np.nonzero(wj_mask_rotate)
            xxyy = np.concatenate((aa[1][:, None, None], aa[0][:, None, None]), -1)
            rect = cv2.minAreaRect(xxyy)
            box = cv2.boxPoints(rect).astype('int')
            wj_mask_rotate2 = wj_mask_rotate.copy().astype('uint8')
            wj_mask_rotate2 = cv2.fillPoly(wj_mask_rotate2, [box], 255)
            wj_tf = True
            for cxcy in RS_cxcy_list:
                fj_cx, fj_cy = int(cxcy[0]), int(cxcy[1])
                if wj_mask_rotate2[fj_cy, fj_cx] == 255:
                    wj_tf = False
                    wj_index = -1
                    break
            if wj_tf:
                for i in range(len_index):
                    wj_mask_rotate_ = (i * 0.1 < wj_mask_rotate) * (wj_mask_rotate <= (i + 1) * 0.1)
                    wj_thr = 0.1 * (i + 1) * 0.8
                    result[wj_mask_rotate_] = wj_rotate[wj_mask_rotate_] * wj_thr + result[wj_mask_rotate_] * (1 - wj_thr)  # 255
                # result[wj_mask_rotate>0.5]=wj_rotate[wj_mask_rotate>0.5]*0.45 + result[wj_mask_rotate>0.5]*0.55 # 255

        return  patch_cx, patch_cy, result, shadow_rotate, plane_cloud_index, wj_index

    def shape_size(self, img, img_shadow):
        locs = np.where(img_shadow != 255)  # 有目标的位置
        x0 = np.min(locs[1])
        x1 = np.max(locs[1])
        y0 = np.min(locs[0])
        y1 = np.max(locs[0])
        img = img[y0:y1, x0:x1]
        img_shadow = img_shadow[y0:y1, x0:x1]
        return img, img_shadow, x1 - x0, y1 - y0  # 返回飞机目标的最小包围patch，裁剪掉多余的边界

    def img_resize(self, img, ori_resolution, target_resolution):
        img_h, img_w = img.shape[:2]  # patch
        img_resize = cv2.resize(img, (int(img_w / target_resolution[0] * ori_resolution[0]), \
                                      int(img_h / target_resolution[1] * ori_resolution[1])), interpolation=cv2.INTER_AREA)
        return img_resize, [int(img_h / target_resolution[1] * ori_resolution[1]), int(img_w / target_resolution[0] * ori_resolution[0])]

    def get_ori_resolution(self, img_shadow, img_name, dis_size):
        # 找到阴影图像中像素值为0的位置（即飞机目标阴影区域）
        locs = np.where(img_shadow == 0)
        # 计算阴影区域在水平方向（x轴）的最小和最大坐标，确定宽度范围
        x0 = np.min(locs[1])  # 阴影区域最左侧x坐标
        x1 = np.max(locs[1])  # 阴影区域最右侧x坐标
        # 计算阴影区域在垂直方向（y轴）的最小和最大坐标，确定高度范围
        y0 = np.min(locs[0])  # 阴影区域最顶部y坐标
        y1 = np.max(locs[0])  # 阴影区域最底部y坐标
        # 计算阴影区域的像素宽度和高度
        w = x1 - x0  # 像素宽度（水平方向像素数）
        h = y1 - y0  # 像素高度（垂直方向像素数）
        # 从dis_size字典中获取该图像对应的实际物理宽度和高度（单位可能为米等）
        w_resolution = float(dis_size[img_name][1])  # 实际物理宽度
        h_resolution = float(dis_size[img_name][2])  # 实际物理高度
        # 计算每像素对应的物理宽度（水平分辨率）和物理高度（垂直分辨率）
        dpi_w = w_resolution / w  # 水平方向：1像素 = dpi_w 物理单位
        dpi_h = h_resolution / h  # 垂直方向：1像素 = dpi_h 物理单位
        # 返回计算得到的水平和垂直分辨率
        return dpi_w, dpi_h  # 注意此时dpi并不是目标在RS上的目标分辨率，而是根据实际meter和当前切片的像素数计算当前每像素表示的长和宽，单位m/pixel

    def get_dis_size(self, path, exposure_time=0.02):
        with open(path, 'r', encoding='utf-8') as filein:
            info = []
            dis_size = dict()
            for i, line in enumerate(filein):
                line_list = line.strip().split(' ')
                info.append(line_list)
            n = len(info)
            for i in range(1, n):
                name = info[i][0]
                mahe = float(info[i][1])
                km = float(info[i][2]) * (random.random() * 0.4 + 0.6) # 随机0.7-1倍 # km/h
                h = float(info[i][3]) # 修改
                w = float(info[i][4])
                wj = float(info[i][5])  # 尾迹数量
                meter = km / 3600 * 1000 * exposure_time  # 0.02成像时间，meter是模糊距离（米）
                dis_size[name] = [meter, w, h, wj, km]
        return dis_size

    def blur_limpid(self, img, img_shadow, img_name, severity, angle):
        degree = self.dis_size[img_name][0] / np.mean(self.target_resolution)
        # img_blur = self.motion_blur(img, degree, angle)
        img_blur = mb(img, severity, angle=angle)
        shadow_blur = mb(img_shadow, severity, angle)
        return img_blur, shadow_blur

    def get_patch(self, img, ph, pw):
        h, w = img.shape[:2]
        if h > ph + 4:
            ih = random.randrange(2, h - ph - 2)  # 确保切片完整且不在边缘
        else:
            ih = 0
        if w > pw + 4:
            iw = random.randrange(2, w - pw - 2)
        else:
            iw = 0
        patch = img[ih: ih + ph, iw: iw + pw]  # bbox
        patch_cy, patch_cx = ih + ph / 2, iw + pw / 2  # 中心,patch中心在原图的坐标
        return patch, patch_cx, patch_cy  # , mean_pixel
    '''def get_patch2(self, img, cx, cy, ph, pw):
        print(f"cx:{cx},cy:{cy},ph:{ph},pw:{pw}")
        lx = int(cx-pw/2)
        ly = int(cy-ph/2)
        rx = int(cx+pw/2)
        ry = int(cy+ph/2)
        patch = img[ly:ry, lx:rx,:]
        print(patch.shape)
        return patch
    '''


    def get_patch2(self, img, cx, cy, ph, pw):
        h, w = img.shape[0], img.shape[1]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        lx = int(round(cx - pw / 2))
        rx = lx + pw
        ly = int(round(cy - ph / 2))
        ry = ly + ph

        left_pad = max(0, -lx)
        right_pad = max(0, rx - w)
        top_pad = max(0, -ly)
        bottom_pad = max(0, ry - h)

        # 关键修正：clamp到[0, w]和[0, h]，确保valid尺寸 >= 0
        clamped_lx = max(0, min(lx, w))
        clamped_rx = max(0, min(rx, w))
        clamped_ly = max(0, min(ly, h))
        clamped_ry = max(0, min(ry, h))

        valid_h = clamped_ry - clamped_ly
        valid_w = clamped_rx - clamped_lx

        patch = np.zeros((ph, pw, channels), dtype=img.dtype)

        if valid_h > 0 and valid_w > 0:
            patch_ly = top_pad
            patch_ry = top_pad + valid_h
            patch_lx = left_pad
            patch_rx = left_pad + valid_w
            patch[patch_ly:patch_ry, patch_lx:patch_rx, :] = \
                img[clamped_ly:clamped_ry, clamped_lx:clamped_rx, :]

        return patch


    # # TODO 根据目标数量生成轨迹信息
    # def get_trace(self, trace_num, resolution=(3,3), RS_h=1024, RS_w=1024,):
    #     total_cx_cy_list = []
    #     totoal_angle_list = []
    #     for t in trace_num:
    #         one_cx_cy_list = []
    #         one_angle_list = []
    #         #TODO 初始化起始位置
    #
    #         #TODO 生成曲线
    #
    #         #TODO 取点
    #
    #         #TODO 记录
    #         total_cx_cy_list.append(one_cx_cy_list)
    #         totoal_angle_list.append(one_angle_list)
    #     return total_cx_cy_list, totoal_angle_list



def distance_count(a, b):
    distance = math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)  # 欧氏距离
    return distance

''' 单通道cx
def cloud_gen(cloud_list):
    cloud_select_list = random.sample(cloud_list, random.sample([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9], 1)[0])
    cloud_img_list = []
    for cloud_select in cloud_select_list:
        # (1024,1024,4)
        cloud_select_ = np.array(Image.open(cloud_select))
        cloud_select_[cloud_select_[:, :, -1] == 0] = 0  # 移除背景透明信息
        cloud_img_list.append(cloud_select_)

    cloud_name_list = [os.path.basename(cloud_list_).split('.')[0] for cloud_list_ in cloud_select_list]
    cloud_select_np = np.array(cloud_img_list)
    cloud_select_np = cloud_select_np.max(0)  # 取最大值实现叠加效果
    cloud_img = cloud_select_np[:, :, 0]
    cloud_img = cv2.medianBlur(cloud_img, 3)  # 中值滤波平滑
    cloud_mask = cloud_img != 0
    cloud_text = str(cloud_name_list)
    # 显示结果
    if cloud_mask is None or cloud_img is None:
        show_clouds(cloud_img, cloud_mask, cloud_text)
    return cloud_img, cloud_mask, cloud_text # (1024,1024) # (1024,1024)
'''
# 彩色图 lhg
def cloud_gen(cloud_list):
    cloud_select_list = random.sample(cloud_list, random.sample([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9], 1)[0])
    cloud_img_list = []
    for cloud_select in cloud_select_list:
        # 读取4通道RGBA图像
        cloud_select_ = np.array(Image.open(cloud_select))  # shape: (h, w, 4)
        # 保留RGB通道，透明区域(RGBA的A=0)的RGB值设为0
        cloud_select_[cloud_select_[:, :, -1] == 0] = [0, 0, 0, 0]  # 透明区域RGB置0
        cloud_img_list.append(cloud_select_)

    cloud_name_list = [os.path.basename(cloud_list_).split('.')[0] for cloud_list_ in cloud_select_list]
    cloud_select_np = np.array(cloud_img_list)  # shape: (n, h, w, 4)

    # 按RGB通道分别取最大值实现彩色叠加
    cloud_rgb = (cloud_select_np[:, :, :, 0]+cloud_select_np[:, :, :, 1]+cloud_select_np[:, :, :, 2]).max(axis=0)  # shape: (h, w, 3)

    # 中值滤波平滑（对RGB通道分别处理）
    cloud_img = cv2.medianBlur(cloud_rgb, 3)  # 直接对3通道图像进行中值滤波

    # 云掩码：任何一个RGB通道不为0则为1
    cloud_mask = (cloud_img != 0).any(axis=-1)  # 沿最后一个轴（通道轴）判断是否有非0值
    cloud_text = str(cloud_name_list)

    return cloud_img, cloud_mask, cloud_text

def show_clouds(cloud_img, cloud_mask, cloud_text):
    """显示云朵图像和掩码，并添加标题"""
    # 创建画布
    plt.figure(figsize=(12, 5))
    # 显示云朵图像
    plt.subplot(121)
    plt.imshow(cloud_img, cmap='gray')
    plt.title(cloud_text)
    plt.axis('off')  # 关闭坐标轴
    # 显示掩码
    plt.subplot(122)
    plt.imshow(cloud_mask, cmap='gray')
    plt.title('云朵掩码')
    plt.axis('off')  # 关闭坐标轴
    # 调整布局并显示
    plt.tight_layout()
    plt.show()


def show_one_img(img):
    # 显示ndarray图像
    h, w = img.shape[:2]
    plt.figure(figsize=(h / 100, w / 100))  # 设置显示窗口大小
    if len(img.shape) == 3:
        plt.imshow(img)  # 灰度图需指定cmap='gray'
        print("rgb")
    else:
        plt.imshow(img, cmap='gray')
        print('gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像


def wj_select(img_name_meta, dis_size, wj_img_1_list, wj_img_2_list, wj_img_4_list):
    wj_num = int(dis_size[img_name_meta][3])  # wj 数量
    wj_rand = random.random()
    wj_index = wj_num if wj_rand <= 0.3 else -1  # 0.3的概率产生尾迹

    # 1 2 4表示尾迹的数量
    if wj_index == 1:  # 3
        # wj_img = random.sample(wj_img_1_list, 1)[0]
        wj_img = wj_img_1_list[6]
    elif wj_index == 2:  # 0,3,5
        # wj_img = random.sample(wj_img_2_list, 1)[0]
        wj_img = wj_img_2_list[0]
    elif wj_index == 4:
        # wj_img = random.sample(wj_img_4_list, 1)[0]
        wj_img = wj_img_4_list[5]
    else:
        # if wj_index == -1 or wj_index == 0:  # 场景不出现尾迹或本身不产生尾迹
        wj_img = None
    return wj_img, wj_index


def shadow_mask(img, w, h):
    """

    :param img:
    :param w:
    :param h:
    :return:
    shadow_mask 函数的作用是对飞机目标的阴影区域进行随机的灰度调整，模拟不同光照条件下阴影的明暗分布差异

    """
    shadow_rand = random.random()
    if shadow_rand < 0.4:
        img = img
        shadow_index = 0
    # if 0.2<=shadow_rand <0.4:
    #     img_ones = np.ones_like(img).astype('float')
    #     img_ones[:int(0.33 * h), :] = 0.9
    #     img_ones[int(0.67 * h):, :] = 0.9  # 所有w 前后各33%的区域*0.9
    #     img = img * img_ones
    #     shadow_index = 4
    if 0.4 <= shadow_rand < 0.6:
        img_ones = np.ones_like(img).astype('float')
        img_ones[:, :int(0.33 * w)] = 0.9
        img_ones[:, int(0.67 * w):] = 0.9  # 所有h 左右各33%的区域*0.9
        img = img * img_ones
        shadow_index = 1
    if 0.6 <= shadow_rand < 0.8:
        img_ones = np.ones_like(img).astype('float')
        img_ones[:, :int(0.5 * w)] = 0.9  # 右侧50%的区域*0.9
        img = img * img_ones
        shadow_index = 2
    if 0.8 <= shadow_rand <= 1:
        img_ones = np.ones_like(img).astype('float')
        img_ones[:, int(0.5 * w):] = 0.9  # 左侧50%的区域*0.9
        img = img * img_ones
        shadow_index = 3
    return img, shadow_index


def calculate_shadow_avg_gray(background, shadow):
    """
    计算阴影区域对应的背景图平均灰度值

    参数:
        background: 3通道背景图，shape为(h, w, 3)，值范围0-255（ndarray）
        shadow: 单通道阴影图，shape为(h, w)，背景区域为255，阴影区域为0（ndarray）

    返回:
        float: 阴影区域对应的背景平均灰度值（若无阴影区域返回0）
    """
    # 1. 校验输入尺寸是否匹配
    if background.shape[:2] != shadow.shape[:2]:
        raise ValueError(f"背景图与阴影图尺寸不匹配！背景：{background.shape[:2]}，阴影：{shadow.shape[:2]}")

    # 2. 将3通道背景图转为单通道灰度图（使用RGB转灰度公式）
    # 灰度 = 0.299*R + 0.587*G + 0.114*B
    gray_background = np.dot(background[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # 3. 生成阴影区域的掩码（阴影图中值为0的区域）
    shadow_mask = (shadow == 0)  # shape为(h, w)，True表示阴影区域

    # 4. 检查是否存在阴影区域
    if not np.any(shadow_mask):
        print("警告：未检测到阴影区域（阴影图中无0值区域）")
        return 0.0

    # 5. 提取阴影区域对应的灰度值并计算平均值
    shadow_gray_values = gray_background[shadow_mask]  # 提取所有阴影区域的灰度值
    avg_gray = np.mean(shadow_gray_values).item()  # 计算平均值并转为Python浮点数

    return avg_gray
import cv2
import random

# 边缘处理
def random_transform_img_and_shadow(img, plane_shadow):
    """
    对飞机切片图像和阴影掩码进行随机变换：
    1. 阴影掩码内层多层区域随机置255（内层数由图像宽度决定），并通过形态学操作平滑边缘
    2. 飞机图像三通道分别随机调整亮度（像素值×0.9~1.1，截断0-255）

    :param img: 3通道飞机切片图像，ndarray， dtype=uint8，shape=(h, w, 3)
    :param plane_shadow: 单通道阴影掩码，ndarray，dtype=uint8，shape=(h, w)
                        （非255区域为飞机有效掩码，255为背景）
    :return: tuple，(transformed_img, transformed_plane_shadow)
             - transformed_img：变换后的3通道飞机图像，dtype=uint8
             - transformed_plane_shadow：变换后的单通道阴影掩码，dtype=uint8
    """
    # show_one_img(plane_shadow)
    # 1. 输入有效性校验
    assert img.shape[:2] == plane_shadow.shape, \
        f"img与plane_shadow尺寸不匹配！img.shape={img.shape}, plane_shadow.shape={plane_shadow.shape}"
    h, w = img.shape[:2]
    if w == 0:
        return img.copy(), plane_shadow.copy()  # 避免宽度为0的极端情况

    # 2. 处理阴影掩码：内层多层随机置255 + 形态学平滑
    # 2.1 生成二值掩码（非255=1表示飞机区域，255=0表示背景）
    plane_binary_mask = (plane_shadow != 255).astype(np.uint8)
    if np.sum(plane_binary_mask) == 0:
        return img.copy(), plane_shadow.copy()  # 无有效区域时直接返回

    # 2.2 计算内层扩充的宽度（像素数）：图像宽度的0~0.1倍随机值，至少1像素
    layer_ratio = random.uniform(0, 0.05)  # 随机比例（0到0.1） # v20将0.05改为0.05没变
    inner_layer_width = max(1, int(w * layer_ratio))  # 确保至少1像素宽
    print(inner_layer_width)
    print(f"内层扩充宽度: {inner_layer_width}像素 (宽度{w} × 比例{layer_ratio:.3f})")

    # 2.3 提取内层多层区域（从边缘向内扩充inner_layer_width层）
    kernel = np.ones((3, 3), np.uint8)  # 形态学操作核
    current_mask = plane_binary_mask.copy()  # 初始为完整飞机区域
    multi_layer_mask = np.zeros_like(plane_binary_mask)  # 存储多层区域的合并掩码

    for _ in range(inner_layer_width):
        # 每次腐蚀得到更内层区域
        eroded_mask = cv2.erode(current_mask, kernel, iterations=1)
        # 当前层 = 上一层 - 腐蚀后的层（即两层之间的环形区域）
        current_layer = current_mask - eroded_mask
        multi_layer_mask |= current_layer  # 合并到多层掩码中
        current_mask = eroded_mask.copy()
        # 若腐蚀后无剩余区域，提前退出（避免无效循环）
        if np.sum(current_mask) == 0:
            break

    # 2.4 对多层区域随机置255（每个像素50%概率变为背景）
    transformed_shadow = plane_shadow.copy()
    layer_coords = np.where(multi_layer_mask == 1)  # 获取多层区域所有像素坐标
    if len(layer_coords[0]) > 0:
        # 生成0~1随机概率，筛选出50%的像素置为255
        rand_probs = np.random.random(size=len(layer_coords[0]))
        change_indices = rand_probs < 0.35 # 50%概率变化 # v20将0.4改为0.35
        # 对选中的像素置255（转为背景）
        y_change = layer_coords[0][change_indices]
        x_change = layer_coords[1][change_indices]
        transformed_shadow[y_change, x_change] = 255
    # show_one_img(transformed_shadow)
    # 2.5 形态学操作：先闭运算（填充空洞）再开运算（平滑边缘），恢复边缘平滑度
    # 使用椭圆核更适合边缘平滑
    morph_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 黑前景白背景，开闭运算作用相反
    # 目的：先去点再补充
    # 闭运算：充物体内部小于核大小的空洞、缝隙，连接邻近的断裂部分，同样基本保持物体整体形态。
    transformed_shadow = cv2.morphologyEx(transformed_shadow, cv2.MORPH_CLOSE, morph_kernel1)
    # show_one_img(transformed_shadow)
    # 开运算：去除图像中小于核大小的孤立噪声点、细小结节，分离相邻的粘连物体，同时基本保持较大物体的形状和尺寸不变。
    transformed_shadow = cv2.morphologyEx(transformed_shadow, cv2.MORPH_OPEN, morph_kernel2)
    # show_one_img(transformed_shadow)

    # 3. 处理飞机图像：三通道分别随机调整亮度（保持原逻辑不变）
    transformed_img = img.astype(np.float32).copy()
    # 生成三通道独立的随机亮度因子（0.9~1.1）
    brightness_factors = np.random.uniform(low=0.85, high=1.15, size=(h, w, 3)) # v20将0.7-1.3改为0.8-1.2 v22改为0.9-1.1
    # print(brightness_factors)
    transformed_img *= brightness_factors
    # 截断并转回uint8
    transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
    # print('transform!')
    # 1. 二值化掩码：非255=1（有效区域），255=0（背景）
    mask_original = (plane_shadow != 255).astype(np.uint8)
    mask_transformed = (transformed_shadow != 255).astype(np.uint8)

    # 2. 逻辑与（AND）操作：仅保留两个掩码均为有效区域的部分
    intersection_binary = cv2.bitwise_and(mask_original, mask_transformed)

    # 3. 转换回原始掩码格式：
    # 交集区域保留原始掩码(plane_shadow)的非255值，背景置255
    transformed_shadow = np.where(intersection_binary == 1, plane_shadow, 255).astype(np.uint8)
    # show_one_img(transformed_shadow)
    return transformed_img, transformed_shadow



# 示例用法
if __name__ == "__main__":
    from Cv2ForChinese import *
    # 生成测试数据（实际使用时替换为你的图像数据）
    h, w = 100, 100  # 图像尺寸
    # 随机生成3通道背景图（0-255）
    background = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    # 生成阴影图（随机指定部分区域为阴影0，其余为背景255）
    shadow = np.ones((h, w), dtype=np.uint8) * 255
    shadow[20:50, 30:70] = 0  # 人为指定一块阴影区域

    # # 计算平均灰度
    # avg = calculate_shadow_avg_gray(background, shadow)
    # print(f"阴影区域对应的背景平均灰度值：{avg:.2f}")
    motion_bulr = MOTION_BLUR(target_resolution=5, txt_path='标称巡航速度查询文件4.txt', exposure_time=0.02)


