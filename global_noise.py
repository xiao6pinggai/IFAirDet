import cv2
import numpy as np
import random


class PhysicalBasedSimulator:
    def __init__(self):
        # PSF参数 - 模拟光学系统模糊
        self.psf_kernel_sizes = {
            '3m': 3,  # 3米分辨率使用更小的模糊
            '5m': 5  # 5米分辨率使用稍大的模糊
        }

        # 噪声参数 - 基于传感器模型
        self.noise_params = {
            'gaussian_sigma': 0.05,  # 读出噪声
            'poisson_scale': 0.05,  # 光子噪声强度
        }

    def create_psf_kernel(self, resolution_type='5m', kernel_size=None):
        """
        创建物理合理的点扩散函数(PSF)核
        使用圆盘或高斯核模拟光学衍射和像差
        """
        if kernel_size is None:
            kernel_size = self.psf_kernel_sizes.get(resolution_type, 3)

        # 方法1: 圆盘核 (更接近光学衍射)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        radius = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    kernel[i, j] = 1.0

        kernel /= np.sum(kernel)  # 归一化

        # 方法2: 高斯核 (备选)
        # kernel = cv2.getGaussianKernel(kernel_size, sigma=0.3 * ((kernel_size-1)*0.5-1) + 0.8)
        # kernel = kernel * kernel.T

        return kernel

    def apply_global_psf(self, image, resolution_type='5m'):
        """
        对整个图像应用PSF模糊 - 模拟光学系统特性
        """
        psf_kernel = self.create_psf_kernel(resolution_type)

        # 对每个通道分别应用卷积
        blurred_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            blurred_image[:, :, c] = cv2.filter2D(image[:, :, c].astype(np.float32), -1, psf_kernel)

        return np.clip(blurred_image, 0, 255).astype(np.uint8)

    def add_sensor_noise_global(self, image, noise_type='mixed',is_active_sensor=True, noise_qiangdu=[0.05,0.05]):
        if not is_active_sensor:
            return image
        """
        改进的全局传感器噪声模拟
        """
         # 噪声参数 - 基于传感器模型
        self.noise_params = {
            'gaussian_sigma': noise_qiangdu[0],  # 读出噪声
            'poisson_scale': noise_qiangdu[1],  # 光子噪声强度
        }

        noisy_image = image.astype(np.float32)
        h, w, c = image.shape
        # 创建有效区域掩码（非全0区域）
        # 方法1：检查所有通道是否都为0
        valid_mask = np.any(image != 0, axis=2)  # 二维掩码，True表示有效区域

        # 扩展为3通道掩码
        valid_mask_3d = np.stack([valid_mask] * c, axis=2)
        # 高斯噪声 (始终存在，强度固定)
        if noise_type in ['mixed', 'gaussian']:
            gaussian_noise = np.random.normal(0, self.noise_params['gaussian_sigma'], (h, w, c))
            # 只在有效区域添加高斯噪声
            gaussian_noise[~valid_mask_3d] = 0  # 将无效区域的噪声置零
            noisy_image += gaussian_noise

        # 泊松噪声 (信号相关噪声)
        if noise_type in ['mixed', 'poisson']:
            # 使用高斯分布近似泊松噪声 (计算更稳定)
            # 泊松噪声的标准差 = sqrt(信号强度)
            signal_dependent_std = np.sqrt(np.maximum(noisy_image, 0) * self.noise_params['poisson_scale'])
            poisson_noise = np.random.normal(0, signal_dependent_std)
            poisson_noise[~valid_mask_3d] = 0  # 将无效区域的噪声置零
            noisy_image += poisson_noise

        # 大气湍流效应 (可选)
        if hasattr(self, 'add_micro_distortion') and random.random() < 0.1:
            distorted_image = self.add_micro_distortion(noisy_image)

            # 只在有效区域应用畸变结果
            noisy_image = np.where(valid_mask_3d, distorted_image, noisy_image)
            
        result = np.clip(noisy_image, 0, 255).astype(np.uint8)
        result[~valid_mask_3d] = 0  # 强制黑边区域为0

        return result
        

    def add_micro_distortion(self, image, max_shift=0.3):
        """
        添加微小的几何畸变模拟大气湍流效应
        """
        h, w = image.shape[:2]

        # 创建随机位移场
        shift_x = np.random.normal(0, max_shift, (h, w))
        shift_y = np.random.normal(0, max_shift, (h, w))

        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # 应用位移
        map_x = x + shift_x
        map_y = y + shift_y

        # 重新映射图像
        distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return distorted

    def simulate_atmospheric_effects(self, image, transparency=None, random_atmospheric_light=1, is_active_atmospheric=True):
        # print(is_active_atmospheric)
        if not is_active_atmospheric:
            return image
        """
        模拟大气效应 - 雾霾和对比度降低
        """
        if transparency is None:
            transparency = random.uniform(0.85, 0.98)  # 更合理的范围

        # 修正的大气光估计 - 取图像中最亮的1%像素的平均值
        flat_image = image.reshape(-1, 3)
        bright_pixels = flat_image[np.argsort(np.mean(flat_image, axis=1))[-len(flat_image) // 100:]]
        atmospheric_light = np.mean(bright_pixels, axis=0) * random_atmospheric_light

        # 确保大气光是标量或与图像同维度
        atmospheric_light = atmospheric_light.reshape(1, 1, 3)

        # 大气散射模型
        hazy_image = image.astype(np.float32) * transparency + atmospheric_light * (1 - transparency)

        # 对比度调整 - 基于物理的大气透射率
        contrast_reduction = 0.7 + 0.3 * transparency  # 透射率越高，对比度保持越好
        hazy_image = 127 + (hazy_image - 127) * contrast_reduction

        return np.clip(hazy_image, 0, 255).astype(np.uint8)



# 使用示例
def complete_simulation_pipeline(background_img, resolution='5m',):
    """
    完整的物理仿真管线
    """
    simulator = PhysicalBasedSimulator()



    # 3. 全局PSF模糊 - 模拟光学系统
    # background_img = simulator.apply_global_psf(background_img, resolution)

    # 4. 大气效应
    background_img = simulator.simulate_atmospheric_effects(background_img)

    # 5. 全局传感器噪声
    background_img = simulator.add_sensor_noise_global(background_img, 'mixed')

    return background_img

if __name__ == '__main__':
    from Cv2ForChinese import *
    from motion_bulr_newV9 import show_one_img
    path = r'E:\NUDT-Master\Academic\20250919SimData\New3\AircraftDataset8\images\train\portA_2018_1_2_5_10\img1/000001.png'
    img = cv_imread(path)
    img2 = complete_simulation_pipeline(img, resolution='5m')
    show_one_img(img)
    show_one_img(img2)

