import torch
from torch.autograd import grad
import numpy as np
from skimage import measure
from tqdm import tqdm

def load_data(file_path):
    f = open(file_path)
    lines = f.readlines()
    data_str = ''
    for line in lines:
        line = line.strip('\n') + ' '
        data_str = data_str + line
    data_str = data_str.split()
    data = np.array(data_str, dtype=float).reshape((-1, 6))
    data = torch.from_numpy(data).float()
    return data


def decode(model, output_path, resolution, min, max):
    print("Decoding")
    with torch.no_grad():
        model.eval()
        # create grid
        x = np.linspace(min, max, resolution)
        y = x
        z = x
        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
        grid_points = grid_points.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # pridict SDF
        SDF = []
        for _, pnts in enumerate(tqdm(torch.split(grid_points,200000,dim=0))):
            SDF.append(model(pnts).detach().cpu().numpy())
        SDF = np.concatenate(SDF,axis=0).astype(float)
        SDF = SDF.reshape((resolution, resolution, resolution))
        # marching cube and save to file
        marching_cube(SDF, output_path)
        

def marching_cube(SDF, output_path):
    # MarchingCube
    positions, faces, normals, _ = measure.marching_cubes(SDF, 0.0)
    faces = faces + 1 # MeshLab 从1开始计数
    # 保存成obj文件
    print(f"Save to {output_path}")
    meshfile = open(f'{output_path}', 'w')
    for item in positions:
        meshfile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
    for item in normals:
        meshfile.write("vn {0} {1} {2}\n".format(
            item[0], item[1], item[2]))
    for item in faces:
        meshfile.write(
            "f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))
        if item[0]==item[1] or item[0]== item[2] or item[1]==item[2]:
            print(faces.index(item))
    meshfile.close()


def gradient(inputs, outputs):
    delta_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(outputs, inputs, delta_points, create_graph=True, retain_graph=True, only_inputs=True)[0][:, -3:]
    return points_grad