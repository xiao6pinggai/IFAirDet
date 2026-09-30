from motion_bulr_newV9 import show_one_img
import numpy as np
import cv2
def get_edge_regions(shadow_mask):
    """
    从shadow_mask中提取边缘区域和边缘内层区域
    shadow_mask: 背景为0，目标区域为非0
    """
    # 创建二值掩码：目标区域为1，背景为0
    target_mask = (shadow_mask > 0).astype(np.uint8)
    # show_one_img(target_mask)
    # 使用3x3椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 1. 获取边缘区域（单像素宽度的边缘线）
    # 方法：先膨胀再减去原图，得到外边缘(边缘线线)
    dilated = cv2.dilate(target_mask, kernel, iterations=1)
    # edge_region = dilated - target_mask  # 外边缘
    # dilated_twice = cv2.dilate(dilated, kernel, iterations=1)
    edge_region = dilated - target_mask  # 外边缘
    # 2. 获取边缘内层区域（边缘线向内一层的区域）
    # 方法：原图减去腐蚀结果，得到内边缘
    eroded = cv2.erode(target_mask, kernel, iterations=1)
    inner_edge_region = target_mask - eroded  # 内边缘

    # 3. 获取完整的边缘内层（边缘线+向内一层）
    # 方法：原图减去两次腐蚀的结果
    eroded_twice = cv2.erode(eroded, kernel, iterations=1)
    full_inner_region = dilated - eroded  # 外边缘+边缘
    # show_one_img(full_inner_region)
    return {
        'target_mask': target_mask,
        'edge_region': edge_region,  # 单像素外边缘
        'inner_edge_region': inner_edge_region,  # 单像素内边缘
        'full_inner_region': full_inner_region,  # 边缘+向内一层
        'dilated': dilated,
        'eroded': eroded,
        'eroded_twice': eroded_twice
    }


def blur_full_inner_region(img, shadow_mask, blur_strength=10.0):
    """
    对完整内层区域（边缘+向内一层）进行模糊
    """
    shadow_mask = (shadow_mask > 255/2).astype(np.uint8)
    regions = get_edge_regions(shadow_mask)
    full_inner_region = regions['full_inner_region']

    if np.sum(full_inner_region) == 0:
        return img

    # 计算模糊参数
    sigma = blur_strength

    # 应用模糊
    result = img.copy()
    blurred_img = cv2.GaussianBlur(result, (3, 3), sigmaX=sigma)

    # 创建3通道的完整内层掩码
    full_inner_mask_3ch = np.stack([full_inner_region] * 3, axis=2)

    # 只替换完整内层像素
    result = np.where(full_inner_mask_3ch != 0, blurred_img, result)

    return result.astype(np.uint8)


def visualize_edge_regions(shadow_mask):
    """
    可视化边缘区域和边缘内层区域
    """
    regions = get_edge_regions(shadow_mask>255/2)

    # 创建可视化图像
    vis = np.zeros((*shadow_mask.shape, 3), dtype=np.uint8)

    # 目标区域：绿色
    vis[regions['target_mask'] == 1] = [0, 255, 0]

    # 边缘区域：红色（外边缘）
    vis[regions['edge_region'] == 1] = [255, 0, 0]

    # 边缘内层区域：蓝色（内边缘）
    vis[regions['inner_edge_region'] == 1] = [0, 0, 255]

    # 完整内层区域：黄色（边缘+向内一层）
    yellow_mask = regions['full_inner_region'] == 1
    vis[yellow_mask] = [255, 255, 0]

    # 显示各个中间结果
    print("原始目标掩码:")
    # show_one_img(regions['target_mask'] * 255)
    print("膨胀结果:")
    # show_one_img(regions['dilated'] * 255)
    print("腐蚀结果:")
    # show_one_img(regions['eroded'] * 255)
    print("二次腐蚀结果:")
    # show_one_img(regions['eroded_twice'] * 255)
    print("边缘区域 (红色):")
    # show_one_img(regions['edge_region'] * 255)
    print("边缘内层区域 (蓝色):")
    # show_one_img(regions['inner_edge_region'] * 255)
    print("完整内层区域 (黄色):")
    # show_one_img(regions['full_inner_region'] * 255)
    print("最终可视化:")
    # show_one_img(vis)

    return vis, regions
if __name__ == "__main__":
    from Cv2ForChinese import *
    path_img = r'E:\NUDT-Master\Academic\20250919SimData\New3\AircraftDataset9\images\train\portA_2018_3_5_3_10\img1/000007.png'
    path_shadow = r'E:\NUDT-Master\Academic\20250919SimData\New3\AircraftDataset9\images\train\portA_2018_3_5_3_10\mask1/000007.jpg'
    shadow_mask = cv_imread(path_shadow)
    img = cv_imread(path_img)
    img2 = blur_full_inner_region(img, shadow_mask, blur_strength=0.8)
    # 可视化边缘线（用于调试）
    edge_visualization = visualize_edge_regions(shadow_mask)
    show_one_img(img)

    show_one_img(img2)
    show_one_img(shadow_mask)


