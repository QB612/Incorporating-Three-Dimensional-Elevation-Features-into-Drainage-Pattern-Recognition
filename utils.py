import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle as pkl

import geopandas as gpd

from geopandas import GeoSeries
from shapely.geometry import LineString, MultiLineString
import scipy.sparse as sp
import pandas as pd
import os
from collections import namedtuple
from scipy.sparse.linalg import eigsh

seed = 300
np.random.seed(seed)


def distance(p1, p2):
    return ((p1.X - p2.X) ** 2 + (p1.Y - p2.Y) ** 2) ** 0.5


def angle(p1, p2, mode=1):
    dx = p1.X - p2.X
    dy = p1.Y - p2.Y
    if dx == 0.0:
        if mode == 2:
            return 90.0
        else:
            if dy > 0:
                return 90.0
            elif dy < 0:
                return -90.0
            else:
                return 0.0
    else:
        if mode == 1:
            return np.arctan(dy / dx) / np.pi * 180.0
        elif mode == 2:
            _angle = np.arctan(dy / dx) / np.pi * 180.0
            if _angle > 0:
                return _angle
            else:
                return _angle + 180.0
        else:
            _angle = np.arctan(dy / dx) / np.pi * 180.0
            if (dx > 0 and dy > 0) or (dx > 0 and dy < 0):
                return _angle
            elif dx < 0 and (dy < 0 or dy == 0.0):
                return _angle - 180.0
            elif dx < 0 and (dy > 0 or dy == 0.0):
                return _angle + 180.0


def angle3(p1, p2, p3):
    a = distance(p1, p2)
    b = distance(p1, p3)
    c = distance(p2, p3)
    if a * b == 0.0:
        return 0
    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))




def line_angle(line1, line2, mode=1):
    points = [line1[0], line1[-1], line2[0], line2[-1]]
    min_distance = float('inf')
    closest_points = None
    i_num = 0
    j_num = 0
    num_list=[0,1,2,3]
    num_list_copy = num_list.copy()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))

            if distance < min_distance:
                min_distance = distance
                closest_points = (points[i], points[j])
                i_num = i
                j_num = j
                
    p_shared, _ = closest_points   
    num_list_copy.remove(i_num)
    num_list_copy.remove(j_num)
    p1 = points[num_list_copy[0]]
    p2 = points[num_list_copy[1]]

    vector1 = np.array([p_shared[0] - p1[0], p_shared[1] - p1[1]])
    vector2 = np.array([p_shared[0] - p2[0], p_shared[1] - p2[1]])
    
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / magnitude_product)

    angle_deg = np.degrees(angle_rad)

    return angle_deg

class Point:
    def __init__(self, x, y):
        self.X = x
        self.Y = y

def get_linePointList(shapefilePath):
    lineShape = gpd.read_file(shapefilePath, encode="utf-8")
    lineList = lineShape["geometry"]
    
    start_elevation = lineShape["start_e"]
    end_elevation = lineShape["end_e"]
    
    lineLengthList = lineList.length
    lineCoorList = []
    firstLastPoints = []
    lineSegmentCenterPoint = []
    
    lineElevation = []

    for i in range(0, lineList.shape[0]):
        lineSegment = np.array(lineList.iat[i].xy).T
        firstPoint = Point(round(lineSegment[0][0], 7), round(lineSegment[0][1], 7))
        lastPoint = Point(round(lineSegment[-1][0], 7), round(lineSegment[-1][1], 7))
        
        lineElevation.append(end_elevation[i] - start_elevation[i])
        
        lineCoorList.append(lineSegment) 
        firstLastPoints.append((firstPoint, lastPoint))
        _centerXY = np.array(LineString(lineSegment).centroid.coords.xy)
        lineSegmentCenterPoint.append(Point(_centerXY[0], _centerXY[1]))

    centerXY = np.array(MultiLineString(lineCoorList).centroid.coords.xy)
    centerPoint = Point(centerXY[0], centerXY[1])
    return (
        lineCoorList,
        lineLengthList.values,
        firstLastPoints,
        centerPoint,
        lineSegmentCenterPoint,
        lineElevation
    )

def sinuousIndex(lineSegmentCoor):
    straightLength = LineString(
        [
            (lineSegmentCoor[0][0], lineSegmentCoor[0][1]),
            (lineSegmentCoor[-1][0], lineSegmentCoor[-1][1]),
        ]
    ).length
    if straightLength == 0:
        straightLength = 0.00001
    curveLength = LineString(lineSegmentCoor).length
    return curveLength, curveLength / straightLength

def calMeander(lineSegmentCoor):
    if lineSegmentCoor.shape[0] < 3:
        return 0
    rect = LineString(lineSegmentCoor).minimum_rotated_rectangle

    try:
        rectPoints = rect.exterior.coords.xy
    except:
        return 0

    length01 = LineString(
        [(rectPoints[0][0], rectPoints[1][0]),
         (rectPoints[0][1], rectPoints[1][1])]
    ).length
    length02 = LineString(
        [(rectPoints[0][0], rectPoints[1][0]),
         (rectPoints[0][3], rectPoints[1][3])]
    ).length
    if length01 > length02: 
        length = length01
        width = length02 + 0.00000001  
    else:
        length = length02
        width = length01 + 0.00000001

    radius = ((length / 2) ** 2 + width**2) / (2 * width)
    angleA = 2 * np.arcsin((length / 2) / radius)
    arcLength = angleA * radius
    return (LineString(lineSegmentCoor).length - arcLength) / arcLength


def calMinRectParam(lineSegmentCoor):
    if lineSegmentCoor.shape[0] < 3:
        rectDirection = angle(
            Point(lineSegmentCoor[0][0], lineSegmentCoor[0][1]),
            Point(lineSegmentCoor[1][0], lineSegmentCoor[1][1]),
        )
        lengthWidthRatio = 0
        return rectDirection, lengthWidthRatio
    rect = LineString(lineSegmentCoor).minimum_rotated_rectangle

    try:
        rectPoints = rect.exterior.coords.xy  # 最小旋转外接矩形的外部坐标，  可能没有
    except:
        rectDirection = angle(
            Point(lineSegmentCoor[0][0], lineSegmentCoor[0][1]),
            Point(lineSegmentCoor[-1][0], lineSegmentCoor[-1][1]),
        )
        lengthWidthRatio = 0
        return rectDirection, lengthWidthRatio
    length01 = LineString(
        [(rectPoints[0][0], rectPoints[1][0]),
         (rectPoints[0][1], rectPoints[1][1])]
    ).length
    length02 = LineString(
        [(rectPoints[0][0], rectPoints[1][0]),
         (rectPoints[0][3], rectPoints[1][3])]
    ).length
    if length01 > length02:
        rectDirection = angle(
            Point(rectPoints[0][0], rectPoints[1][0]),
            Point(rectPoints[0][1], rectPoints[1][1]),
        )
        lengthWidthRatio = length02 / length01
    else:
        rectDirection = angle(
            Point(rectPoints[0][0], rectPoints[1][0]),
            Point(rectPoints[0][2], rectPoints[1][2]),
        )
        lengthWidthRatio = length01 / length02
    return rectDirection, lengthWidthRatio

def calRiverTopology(shapefilePath):
    riverNetwork = gpd.read_file(shapefilePath)
    outletIndex = int(
        riverNetwork[riverNetwork["outlet"] == 1].index.values)  
    riverList = list(range(riverNetwork.shape[0]))

    point1 = np.array(riverNetwork.loc[outletIndex, "geometry"].coords)[0]
    point2 = np.array(riverNetwork.loc[outletIndex, "geometry"].coords)[-1]
    select_riverList = [outletIndex] 

    riverTopology = dict()  
    currentIndex = outletIndex 
    groupedIndexList = [currentIndex]
    for i in riverList:
        riverTopology[i] = [[], []]
    while len(select_riverList) > 0:
        riverList.remove(currentIndex)
        for i in riverList:
            if i in select_riverList or i in groupedIndexList:
                continue
            if((np.floor(np.around(point1))/ 10**8).tolist()
               in (np.floor(np.around(riverNetwork.loc[i,'geometry'].coords))/ 10**8).tolist())\
                    or ((np.floor(np.around(point2))/ 10**8).tolist()
                        in (np.floor(np.around(riverNetwork.loc[i,'geometry'].coords))/ 10**8).tolist()):
                riverTopology[currentIndex][0].append(i)
                riverTopology[i][1].append(currentIndex)
        select_riverList.remove(currentIndex)
        select_riverList += riverTopology[currentIndex][0]
        if len(select_riverList) > 0:
            currentIndex = select_riverList[0]
            groupedIndexList.append(currentIndex)
            point1 = np.array(
                riverNetwork.loc[currentIndex, "geometry"].coords)[0]
            point2 = np.array(
                riverNetwork.loc[currentIndex, "geometry"].coords)[-1]
    if not riverList:
        pass
    else:
        print("riverlist内有未遍历内容:",riverList)
    
    
    for i in riverTopology[outletIndex][0]:
        if (np.floor(np.around(np.array(riverNetwork.loc[outletIndex, 'geometry'].coords)[0]))/ 10**8).tolist() \
                in (np.floor(np.around(np.array(riverNetwork.loc[i, 'geometry'].coords)))/ 10**8).tolist():
            outletXY = np.array(riverNetwork.loc[outletIndex, 'geometry'].coords)[-1]
            break
        else:
            outletXY = np.array(riverNetwork.loc[outletIndex, 'geometry'].coords)[0]
            break
        
    outletPoint = Point(outletXY[0], outletXY[1])
    return riverTopology, riverNetwork, outletPoint


def calHortonCode(riverNetwork, riverTopology):
    hortonCode = np.zeros(riverNetwork.shape[0])
    unCodeList = list(range(riverNetwork.shape[0]))
    codeList = []
    downStreamList = []
    for i in range(riverNetwork.shape[0]):
        if riverTopology[i][0] == []:
            hortonCode[i] = 1
            unCodeList.remove(i)
            codeList.append(i)
            downStreamList += riverTopology[i][1]
    downStreamList = list(set(downStreamList))
    while len(downStreamList) > 0:
        downStreamListCopy = downStreamList.copy()
        for i in downStreamListCopy:
            upHortonCode = []
            if not set(riverTopology[i][0]) <= set(codeList):
                continue
            for j in riverTopology[i][0]:
                upHortonCode.append(hortonCode[j])
            maxHortonCode = max(upHortonCode)
            if len(upHortonCode) > 1 and sum(upHortonCode == maxHortonCode) == len(
                upHortonCode
            ):
                hortonCode[i] = maxHortonCode + 1
            else:
                hortonCode[i] = maxHortonCode
            codeList.append(i)
            downStreamList.remove(i)
            downStreamList += riverTopology[i][1]
            downStreamList = list(set(downStreamList))
    return hortonCode


def demSlope(lineSegmentCoor,lineSegmentElevation):
    straightLength = LineString(
        [
            (lineSegmentCoor[0][0], lineSegmentCoor[0][1]),
            (lineSegmentCoor[-1][0], lineSegmentCoor[-1][1]),
        ]
    ).length
    if straightLength == 0:
        straightLength = 0.00001
    curveLength = LineString(lineSegmentCoor).length
    
    dem_Slope = abs(math.degrees(math.atan(lineSegmentElevation/straightLength))) #坡度
    mean_Elevation = abs(lineSegmentElevation/curveLength)

    return dem_Slope , mean_Elevation




def calculate_river_angles(i, lineCoorList, riverTopology_cra):
    
    line_coords = lineCoorList[i]
    parent_rivers =riverTopology_cra[i][1]
    river_angles = 0.0 
    
    if parent_rivers:

        for parent_river in parent_rivers:
            sibling_rivers = riverTopology_cra[parent_river][0]
            
            sibling_rivers_copy = sibling_rivers.copy()
            sibling_rivers_copy.remove(i)
            if sibling_rivers_copy:
                if len(sibling_rivers_copy) == 1:
                    angle = line_angle(line_coords, lineCoorList[sibling_rivers_copy[0]])
                    river_angles = angle
                else:
                    angles = [(j, line_angle(line_coords, lineCoorList[j])) for j in sibling_rivers_copy]
                    min_angle_info = min(angles, key=lambda x: x[1])
                    river_angles = min_angle_info[1]

        
    return river_angles

def flaNumpy(a):
    return a.reshape(-1, 1).flatten()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        cords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return cords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)#将输入的邻接矩阵 adj 转换为 Coordinate Format (COO) 的稀疏矩阵，这种格式更适合进行元素级操作。
    rowsum = np.array(adj.sum(1), dtype="float64")#计算了每个节点（行）连接的边的权重之和，这是一个一维数组
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()#计算了 rowsum 中每个元素的倒数的平方根。————度矩阵开平方根后的结果
                                                #flatten() 用于将结果展平为一维数组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0#用于处理那些具有零度节点的情况，将其标记为零。
    #d_inv_sqrt = np.power(np.maximum(rowsum, 1e-10), -0.5).flatten()

    
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)#创建了一个对角线上元素为 d_inv_sqrt 的对角矩阵。
    a = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    #adj.dot(d_mat_inv_sqrt)：邻接矩阵 adj 与对角矩阵 d_mat_inv_sqrt 相乘
    #transpose()：这是对第一步中的结果进行转置操作
    return a


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized #
    largest_eigval, _ = eigsh(laplacian, 1, which="LM")#计算拉普拉斯矩阵的最大特征值 largest_eigval
    scaled_laplacian = (
        2.0 / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])#归一化

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        # s_lap = scaled_lap
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)