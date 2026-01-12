import h5py
import cv2
import numpy as np
import os
from tqdm import tqdm

# 创建输出目录
os.makedirs('output/rgb', exist_ok=True)
os.makedirs('output/depth_gray', exist_ok=True)
os.makedirs('output/depth_16bit', exist_ok=True)
os.makedirs('output/depth_color', exist_ok=True)  # 新增：深度图彩色可视化目录

# 打开HDF5文件
with h5py.File('/data/share/aloha_multiview/cube_plate_new/episode_20.hdf5', 'r') as f:
    # 查看文件结构
    print("文件中的数据集和组:")
    def print_attrs(name, obj):
        print(name)
        if hasattr(obj, 'attrs'):
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
    f.visititems(print_attrs)
    
    # 读取RGB数据
    rgb_data = None
    if 'observations/cam_high_rgb' in f:
        rgb_data = f['observations/cam_high_rgb'][:]
        print(f"RGB数据形状: {rgb_data.shape}")
    else:
        print("未找到RGB数据集")
    
    # 读取深度数据
    depth_data = None
    if 'observations/cam_high_depth' in f:
        depth_dataset = f['observations/cam_high_depth']
        depth_data = depth_dataset[:]  # 立即读取到内存
        print(f"深度数据形状: {depth_data.shape}")
        print(f"深度数据类型: {depth_data.dtype}")
        print(f"深度数据范围: [{np.min(depth_data):.3f}, {np.max(depth_data):.3f}]")
        
        # 检查数据集属性，可能包含重要的缩放信息
        if hasattr(depth_dataset, 'attrs'):
            print("深度数据属性:")
            for key, val in depth_dataset.attrs.items():
                print(f"  {key}: {val}")
    else:
        print("未找到深度数据集")
        # 尝试其他可能的路径
        possible_depth_paths = [
            'observations/depth/cam_high',
            'observations/depth_images',
            'depth/cam_high'
        ]
        for path in possible_depth_paths:
            if path in f:
                depth_data = f[path][:]
                print(f"在路径 '{path}' 找到深度数据: {depth_data.shape}")
                break

    # 保存RGB图像
    if rgb_data is not None:
        print("保存RGB图像...")
        for i in tqdm(range(rgb_data.shape[0]), desc="RGB图像"):
            img = rgb_data[i]
            # 确保图像数据格式正确
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            # 转换颜色空间并保存
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'output/rgb/image_rgb_{i:04d}.png', img_bgr)

    # 保存深度图像为灰度图和彩色可视化
    if depth_data is not None:
        print("处理深度图像...")
        
        # 分析深度数据特性
        depth_min, depth_max = np.min(depth_data), np.max(depth_data)
        print(f"深度值范围: {depth_min:.4f} 到 {depth_max:.4f}")
        
        # 保存深度范围信息
        np.savetxt('output/depth_range.txt', 
                   [depth_min, depth_max], 
                   header=f'Depth range for {depth_data.shape[0]} images')
        
        for i in tqdm(range(depth_data.shape[0]), desc="深度图像处理"):
            depth_frame = depth_data[i]
            
            # 方法1: 保存为8位灰度图（用于可视化）
            # 归一化到0-255范围
            depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            
            # 处理可能的NaN或Inf值
            if np.any(np.isnan(depth_normalized)) or np.any(np.isinf(depth_normalized)):
                depth_normalized = np.nan_to_num(depth_normalized, nan=0.0, posinf=255.0, neginf=0.0)
            
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            # 保存为灰度图
            cv2.imwrite(f'output/depth_gray/image_depth_gray_{i:04d}.png', depth_uint8)
            
            # 新增：深度图彩色可视化（使用Jet颜色映射）
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(f'output/depth_color/image_depth_color_{i:04d}.png', depth_color)
            
            # 方法2: 保存为16位灰度图（保留原始精度信息）
            if depth_frame.dtype in [np.float32, np.float64]:
                # 浮点型深度数据，缩放到16位整数范围
                depth_scaled = ((depth_frame - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
                cv2.imwrite(f'output/depth_16bit/image_depth_16bit_{i:04d}.png', depth_scaled)
            else:
                # 已经是整数类型，直接保存
                cv2.imwrite(f'output/depth_16bit/image_depth_16bit_{i:04d}.png', depth_frame)
    
    if 'observations/images/cam_high' in f:
        alt_rgb_data = f['observations/images/cam_high'][:]
        print(f"备用RGB数据形状: {alt_rgb_data.shape}")
        # 保存视频到指定文件夹
        print("保存备用RGB图像...")
        for i in tqdm(range(alt_rgb_data.shape[0]), desc="备用RGB图像"):
            img = alt_rgb_data[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'output/rgb/alt_image_rgb_{i:04d}.png', img_bgr)  
        print("视频保存在: output/rgb/")
        

print("处理完成！")
print(f"RGB图像保存在: output/rgb/")
print(f"8位灰度深度图保存在: output/depth_gray/")
print(f"16位精度深度图保存在: output/depth_16bit/")
print(f"深度图彩色可视化保存在: output/depth_color/")  # 新增：输出彩色深度图路径
print(f"深度范围信息保存在: output/depth_range.txt")