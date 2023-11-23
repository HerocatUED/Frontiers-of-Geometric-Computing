import numpy as np
from MarchingCube import MarchingCubes


def main():
    filenames = ["01.sdf", "02.sdf"]
    for filename in filenames:
        print(f"Processing {filename}")
        # 读取SDF并reshape成(128*128*128)
        SDF = np.fromfile(filename, dtype=np.float32)
        SDF = SDF.reshape((128, 128, -1))
        print("Shape:",np.shape(SDF))
        # MarchingCube
        positions, faces, normals = MarchingCubes(SDF)
        # 保存成obj文件
        print(f"Save to {filename[:-4]}.obj")
        meshfile = open(f'{filename[:-4]}.obj', 'w')
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


if __name__ == '__main__':
    main()
