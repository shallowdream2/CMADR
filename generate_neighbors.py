"""
脚本用于生成卫星网络中每个时隙、每个节点的邻居关系
- 卫星之间：欧氏距离小于阈值且每个卫星最多4个邻居
- 地面站：连接距离小于等于阈值的所有卫星
"""

import json
import numpy as np
from typing import List, Tuple
import argparse
from scipy.spatial.distance import cdist


def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """
    计算两个坐标点之间的欧氏距离
    支持2D和3D坐标
    """
    if len(pos1) == 2 and len(pos2) == 2:
        # 2D coordinates
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    elif len(pos1) == 3 and len(pos2) == 3:
        # 3D coordinates
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    else:
        raise ValueError("Position coordinates must be either 2D or 3D, and both positions must have the same dimension")


# def haversine_distance_3d(pos1: List[float], pos2: List[float]) -> float:
#     """
#     使用haversine公式计算地球表面两点间的3D距离(km)
#     考虑高度差异
#     pos1, pos2: [longitude, latitude, altitude] in degrees and km
#     """
#     if len(pos1) != 3 or len(pos2) != 3:
#         raise ValueError("3D haversine distance requires 3D coordinates [lon, lat, alt]")
    
#     R = 6371  # 地球半径(km)
    
#     lon1, lat1, alt1 = np.radians(pos1[0]), np.radians(pos1[1]), pos1[2]
#     lon2, lat2, alt2 = np.radians(pos2[0]), np.radians(pos2[1]), pos2[2]
    
#     # 计算地表距离
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
    
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     surface_distance = R * c
    
#     # 计算3D距离（考虑高度差）
#     height_diff = alt2 - alt1
#     distance_3d = np.sqrt(surface_distance**2 + height_diff**2)
    
#     return distance_3d


def haversine_distance(pos1: List[float], pos2: List[float]) -> float:
    """
    使用haversine公式计算地球表面两点间的距离(km)
    pos1, pos2: [longitude, latitude] in degrees (2D) or [longitude, latitude, altitude] (3D)
    """
    R = 6371  # 地球半径(km)
    
    # 支持2D和3D坐标
    lon1, lat1 = np.radians(pos1[0]), np.radians(pos1[1])
    lon2, lat2 = np.radians(pos2[0]), np.radians(pos2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    surface_distance = R * c
    
    # 如果是3D坐标，考虑高度差
    if len(pos1) == 3 and len(pos2) == 3:
        height_diff = pos2[2] - pos1[2]
        return np.sqrt(surface_distance**2 + height_diff**2)
    
    return surface_distance


def find_satellite_neighbors(sat_positions: List[List[float]], 
                           distance_threshold: float, 
                           max_neighbors: int = 4,
                           use_haversine: bool = False) -> List[List[int]]:
    """
    为每个卫星找到邻居节点
    
    Args:
        sat_positions: 卫星位置列表 [[lon, lat], ...]
        distance_threshold: 距离阈值
        max_neighbors: 每个卫星的最大邻居数
        use_haversine: 是否使用haversine距离公式
        
    Returns:
        List[List[neighbor_ids]] - neighbors[i] 表示第i个卫星的邻居列表
    """
    num_sats = len(sat_positions)
    neighbors = [[] for _ in range(num_sats)]
    
    # 计算所有卫星间的距离矩阵
    if use_haversine:
        distances = np.zeros((num_sats, num_sats))
        for i in range(num_sats):
            for j in range(i+1, num_sats):
                dist = haversine_distance(sat_positions[i], sat_positions[j])
                distances[i][j] = dist
                distances[j][i] = dist
    else:
        # 使用scipy计算欧氏距离矩阵
        sat_array = np.array(sat_positions)
        distances = cdist(sat_array, sat_array)
    
    # 为每个卫星找到邻居
    for sat_id in range(num_sats):
        # 获取距离小于阈值的卫星
        valid_distances = []
        for other_sat in range(num_sats):
            if sat_id != other_sat and distances[sat_id][other_sat] <= distance_threshold:
                valid_distances.append((distances[sat_id][other_sat], other_sat))
        
        # 按距离排序，选择最近的max_neighbors个
        valid_distances.sort(key=lambda x: x[0])
        neighbor_ids = [sat_id for _, sat_id in valid_distances[:max_neighbors]]
        
        neighbors[sat_id] = neighbor_ids
    
    return neighbors


def find_ground_station_neighbors(gs_positions: List[List[float]], 
                                sat_positions: List[List[float]], 
                                distance_threshold: float,
                                use_haversine: bool = False) -> List[List[int]]:
    """
    为每个地面站找到邻居卫星节点
    
    Args:
        gs_positions: 地面站位置列表
        sat_positions: 卫星位置列表  
        distance_threshold: 距离阈值
        use_haversine: 是否使用haversine距离公式
        
    Returns:
        List[List[satellite_ids]] - neighbors[i] 表示第i个地面站的邻居卫星列表
    """
    num_gs = len(gs_positions)
    num_sats = len(sat_positions)
    neighbors = [[] for _ in range(num_gs)]
    
    for gs_id in range(num_gs):
        neighbor_sats = []
        gs_pos = gs_positions[gs_id]
        
        for sat_id in range(num_sats):
            sat_pos = sat_positions[sat_id]
            
            if use_haversine:
                dist = haversine_distance(gs_pos, sat_pos)
            else:
                dist = calculate_distance(gs_pos, sat_pos)
            
            if dist <= distance_threshold:
                neighbor_sats.append(sat_id)
        
        neighbors[gs_id] = neighbor_sats
    
    return neighbors


def add_neighbors_to_data(data_file: str, 
                        sat_distance_threshold: float = 10.0,
                        gs_distance_threshold: float = 15.0,
                        max_sat_neighbors: int = 4,
                        use_haversine: bool = False) -> None:
    """
    在源数据文件中添加邻居关系字段
    
    Args:
        data_file: 输入/输出数据文件路径
        sat_distance_threshold: 卫星间连接距离阈值
        gs_distance_threshold: 地面站连接卫星距离阈值
        max_sat_neighbors: 每个卫星最大邻居数
        use_haversine: 是否使用haversine距离公式
    """
    # 读取数据
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    num_satellites = data['num_satellites']
    num_ground = data['num_ground']
    sat_positions_per_slot = data['sat_positions_per_slot']
    gs_positions = data['gs_positions']
    
    # 计算训练和预测阶段的总时隙数
    total_slots = len(sat_positions_per_slot)
    
    print(f"处理 {num_satellites} 个卫星, {num_ground} 个地面站, {total_slots} 个时隙")
    print(f"卫星距离阈值: {sat_distance_threshold}, 地面站距离阈值: {gs_distance_threshold}")
    print(f"每个卫星最大邻居数: {max_sat_neighbors}")
    print(f"使用距离计算方法: {'Haversine' if use_haversine else 'Euclidean'}")
    
    # 初始化邻居关系列表
    neighbors_per_slot = []
    
    # 为每个时隙生成邻居关系
    for slot_id in range(total_slots):
        print(f"处理时隙 {slot_id + 1}/{total_slots}")
        
        sat_positions = sat_positions_per_slot[slot_id]
        
        # 生成卫星邻居关系
        sat_neighbors = find_satellite_neighbors(
            sat_positions, 
            sat_distance_threshold, 
            max_sat_neighbors,
            use_haversine
        )
        
        # 生成地面站邻居关系
        gs_neighbors = find_ground_station_neighbors(
            gs_positions, 
            sat_positions, 
            gs_distance_threshold,
            use_haversine
        )
        
        # 合并卫星和地面站邻居关系
        # 卫星节点ID: 0 到 num_satellites-1
        # 地面站节点ID: num_satellites 到 num_satellites+num_ground-1
        combined_neighbors = []
        
        # 添加卫星邻居关系 (节点ID 0 到 num_satellites-1)
        for sat_id in range(num_satellites):
            combined_neighbors.append(sat_neighbors[sat_id])
        
        # 添加地面站邻居关系 (节点ID num_satellites 到 num_satellites+num_ground-1)
        # 地面站的邻居是卫星，不需要调整ID
        for gs_id in range(num_ground):
            combined_neighbors.append(gs_neighbors[gs_id])
        
        # 创建时隙邻居关系数据结构
        slot_neighbors = {
            "slot_id": slot_id,
            "neighbors": combined_neighbors
        }
        
        neighbors_per_slot.append(slot_neighbors)
        
        # 统计连接信息
        total_sat_connections = sum(len(neighbors) for neighbors in sat_neighbors)
        total_gs_connections = sum(len(neighbors) for neighbors in gs_neighbors)
        
        # 打印统计信息
        if slot_id % 10 == 0 or slot_id == total_slots - 1:
            print(f"  时隙 {slot_id}: 卫星连接 {total_sat_connections}, "
                  f"地面站连接 {total_gs_connections}")
    
    # 将邻居关系添加到原始数据中
    data['neighbors_per_slot'] = neighbors_per_slot
    
    # 添加配置信息
    data['neighbor_config'] = {
        'sat_distance_threshold': sat_distance_threshold,
        'gs_distance_threshold': gs_distance_threshold,
        'max_sat_neighbors': max_sat_neighbors,
        'use_haversine': use_haversine
    }
    
    # 保存更新后的数据
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"邻居关系已添加到源数据文件: {data_file}")
    print(f"新增字段: neighbors_per_slot, neighbor_config")
    print(f"节点ID分配: 卫星 0-{num_satellites-1}, 地面站 {num_satellites}-{num_satellites+num_ground-1}")


def analyze_neighbors(data_file: str):
    """分析数据文件中邻居关系的统计信息"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if 'neighbors_per_slot' not in data:
        print("错误: 数据文件中没有找到邻居关系字段，请先运行邻居关系生成")
        return
    
    print("\n=== 邻居关系分析结果 ===")
    
    num_satellites = data['num_satellites']
    num_ground = data['num_ground']
    neighbors_per_slot = data['neighbors_per_slot']
    total_slots = len(neighbors_per_slot)
    
    # 统计所有时隙的连接情况
    sat_connections_per_slot = []
    gs_connections_per_slot = []
    
    for slot_data in neighbors_per_slot:
        slot_id = slot_data['slot_id']
        neighbors = slot_data['neighbors']
        
        # 统计卫星连接数 (节点ID 0 到 num_satellites-1)
        sat_connections = 0
        for sat_id in range(num_satellites):
            sat_connections += len(neighbors[sat_id])
        
        # 统计地面站连接数 (节点ID num_satellites 到 num_satellites+num_ground-1)
        gs_connections = 0
        for gs_id in range(num_satellites, num_satellites + num_ground):
            gs_connections += len(neighbors[gs_id])
        
        sat_connections_per_slot.append(sat_connections)
        gs_connections_per_slot.append(gs_connections)
    
    print(f"卫星间连接统计:")
    print(f"  平均连接数: {np.mean(sat_connections_per_slot):.2f}")
    print(f"  最大连接数: {max(sat_connections_per_slot)}")
    print(f"  最小连接数: {min(sat_connections_per_slot)}")
    
    print(f"地面站-卫星连接统计:")
    print(f"  平均连接数: {np.mean(gs_connections_per_slot):.2f}")
    print(f"  最大连接数: {max(gs_connections_per_slot)}")
    print(f"  最小连接数: {min(gs_connections_per_slot)}")
    
    print(f"\n节点ID分配:")
    print(f"  卫星: 0 到 {num_satellites-1}")
    print(f"  地面站: {num_satellites} 到 {num_satellites+num_ground-1}")
    
    if 'neighbor_config' in data:
        config = data['neighbor_config']
        print(f"\n配置信息:")
        print(f"  卫星距离阈值: {config['sat_distance_threshold']}")
        print(f"  地面站距离阈值: {config['gs_distance_threshold']}")
        print(f"  每个卫星最大邻居数: {config['max_sat_neighbors']}")
        print(f"  距离计算方法: {'Haversine' if config['use_haversine'] else 'Euclidean'}")


def main():
    parser = argparse.ArgumentParser(description='在源数据文件中添加卫星网络邻居关系')
    parser.add_argument('--data_file', type=str, default='data/LEO_02.json',
                       help='输入/输出数据文件路径')
    parser.add_argument('--sat_threshold', type=float, default=2000.0,
                       help='卫星间连接距离阈值')
    parser.add_argument('--gs_threshold', type=float, default=700.0,
                       help='地面站连接卫星距离阈值')
    parser.add_argument('--max_neighbors', type=int, default=4,
                       help='每个卫星最大邻居数')
    parser.add_argument('--use_haversine', type=int,default=1,
                       help='使用haversine距离公式而非欧氏距离')
    parser.add_argument('--analyze', action='store_true',
                       help='分析邻居关系统计信息')
    
    args = parser.parse_args()
    
    # 添加邻居关系到数据文件
    add_neighbors_to_data(
        data_file=args.data_file,
        sat_distance_threshold=args.sat_threshold,
        gs_distance_threshold=args.gs_threshold,
        max_sat_neighbors=args.max_neighbors,
        use_haversine=args.use_haversine
    )
    
    # 分析邻居关系
    if args.analyze:
        analyze_neighbors(args.data_file)


if __name__ == "__main__":
    main()
