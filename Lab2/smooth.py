import openmesh as om
import numpy as np
import time


def main(num_iterations: int = 20, lr: float = 0.1):
    mesh = om.read_trimesh('smoothing.obj')
    print(f"Processing smoothing.obj, num_iterations = {num_iterations}, lambda = {lr}")
    mesh = smooth(mesh, num_iterations, lr)
    om.write_mesh('smoothed.obj', mesh)


def smooth(mesh, num_iterations: int, lr: float):
    """ 拉普拉斯平滑 """
    for loop in range(num_iterations):  # 最大迭代次数num_iterations
        print(f"Iteration {loop+1}")
        t1 = time.time()
        # 保存初始的点的坐标
        pre_positions = mesh.points()
        # 遍历所有顶点,计算并更新顶点位置
        for vertex in mesh.vertices():
            neighbors = mesh.vv(vertex)
            faces = mesh.vf(vertex)
            sum_position = np.array([0, 0, 0], dtype=float)
            sum_weight = 0
            # 遍历所有邻居点求得更新权重与更新坐标
            for neighbor in neighbors:
                # 查找邻域内与边（vertex,neighbor）相对的顶点
                oppo_vertex_index = []
                for face in faces:
                    iter = om.FaceVertexIter(mesh, face)
                    face_vertices = []
                    for _ in range(3):
                        face_vertices.append(next(iter).idx())
                    if neighbor.idx() in face_vertices:
                        face_vertices.remove(neighbor.idx())
                        face_vertices.remove(vertex.idx())
                        oppo_vertex_index.append(face_vertices[0])
                # 求与边（vertex,neighbor）相对的两个夹角alpha、beta
                # alpha
                e0 = pre_positions[oppo_vertex_index[0]] - \
                    pre_positions[vertex.idx()]
                e1 = pre_positions[oppo_vertex_index[0]] - \
                    pre_positions[neighbor.idx()]
                e0 = e0/(np.dot(e0, e0)**0.5)
                e1 = e1/(np.dot(e1, e1)**0.5)
                alpha_cos = np.dot(e0, e1)
                alpha_cot = alpha_cos/(1-alpha_cos*alpha_cos)**0.5
                # beta
                e2 = pre_positions[oppo_vertex_index[1]] - \
                    pre_positions[vertex.idx()]
                e3 = pre_positions[oppo_vertex_index[1]] - \
                    pre_positions[neighbor.idx()]
                e2 = e2/(np.dot(e2, e2)**0.5)
                e3 = e3/(np.dot(e3, e3)**0.5)
                beta_cos = np.dot(e2, e3)
                beta_cot = beta_cos/(1-beta_cos*beta_cos)**0.5
                # 加权
                weight = alpha_cot+beta_cot
                weight = abs(weight)
                sum_weight += weight
                sum_position += weight*pre_positions[neighbor.idx()]
            # 更新顶点
            sum_position /= sum_weight
            new_position = (1-lr)*pre_positions[vertex.idx()]+lr*sum_position
            mesh.set_point(vertex, new_position)
        t2 = time.time()
        print(f"Time cost: {t2-t1}")
    return mesh


if __name__ == '__main__':
    main()
