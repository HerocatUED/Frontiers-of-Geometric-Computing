import numpy as np
from Table import EdgeStateTable, EdgeOrdsTable


def MarchingCubes(SDF):
    positions = []  # 网格中顶点的坐标(x,y,z)
    faces = []  # 网格中三角形面的三个顶点索引(f1,f2,f3)
    normals = []  # 网格中顶点法向量(nx,ny,nz)
    h, w, d = np.shape(SDF)
    unit = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],dtype=int)  # 3个单位方向向量
    length = 0  # 记录mesh中已经有多少顶点
    # 遍历场中所有小立方体
    for i in range(h-1):
        for j in range(w-1):
            for k in range(d-1):
                index = int(0)  # 当前立方体在Cubes列表中的索引
                # 以固定的顺序遍历立方体8个顶点
                for v in range(8):
                    # 计算当前顶点的位置
                    x = i+(v & 1)
                    y = j+(v >> 1 & 1)
                    z = k+(v >> 2 & 1)
                    if SDF[x, y, z] < 0:
                        index |= 1 << v
                # 在表中查询当前立方体的状态
                edges = EdgeStateTable[index]
                triangles = EdgeOrdsTable[index]
                record = np.array([-1]*12)  # 记录三角形顶点的索引
                # 以固定顺序遍历立方体的12条边
                for e in range(12):  
                    if edges & (1 << e):
                        # 计算这条边的两个顶点v1,v2的位置
                        base = np.array([i, j, k], dtype=int)
                        direction = (e & 1) * unit[((e >> 2) + 1) %
                                                   3] + (e >> 1 & 1) * unit[((e >> 2) + 2) % 3]
                        v1 = base+direction
                        v2 = v1+unit[e >> 2]
                        sdf1 = SDF[v1[0], v1[1], v1[2]]
                        sdf2 = SDF[v2[0], v2[1], v2[2]]
                        # 线性插值确认顶点位置,限制比率防止出现重合
                        rate = sdf1 / (sdf1-sdf2)
                        rate = min(max(rate, 1e-5), 1-(1e-5))
                        # 计算SDF梯度作为顶点法向
                        grad0 = [SDF[v1[0]+1, v1[1], v1[2]]-sdf1, SDF[v1[0],
                                                                      v1[1]+1, v1[2]]-sdf1, SDF[v1[0], v1[1], v1[2]+1]-sdf1]
                        grad1 = [SDF[v2[0]+1, v2[1], v2[2]]-sdf2, SDF[v2[0],
                                                                      v2[1]+1, v2[2]]-sdf2, SDF[v2[0], v2[1], v2[2]+1]-sdf2]
                        grad0 = np.array(grad0, dtype=float)
                        grad1 = np.array(grad1, dtype=float)
                        grad = grad0 + (grad1 - grad0) * rate
                        # 计算顶点位置
                        v1=v1.astype(float)
                        pos = v1 + unit[e >> 2]*rate
                        pos = tuple(np.array(pos, dtype=float))
                        # 查找此点是否出现过，判重、除去重复
                        if pos in positions:
                            record[e] = positions.index(pos)
                        else:
                            positions.append(pos)
                            normals.append(tuple(grad))
                            record[e] = length
                            length += 1
                # 遍历立方体的三角形面并存入列表
                t = 0
                while triangles[t] != -1 and t < 16:
                    faces.append(int(record[triangles[t]]+1))
                    t += 1
    # 重组成一个面表示成(f1,f2,f3)的形式，便于写入
    faces_ = [(faces[3*i], faces[3*i+1], faces[3*i+2])
              for i in range(int(np.shape(faces)[0]/3))]
    return positions, faces_, normals
