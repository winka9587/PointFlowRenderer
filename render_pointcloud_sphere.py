from hashlib import new
import open3d as o3d
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
import os
import sys

import open3d as o3d
import numpy as np
# arg_scale 控制整体点云的大小, 默认为1.0
def standardize_bbox(pcl, points_per_object, shuffle=True, arg_scale=1.0):
    if shuffle:
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt_indices)
        pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins) / arg_scale
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

def configure_xml(radius=0.025):
    xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    # 0.025
    xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="{}"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """
    return xml_head, xml_ball_segment, xml_tail

def colormap(xyz):
    vec = xyz
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def rotate_point_cloud(point_cloud, angle_degrees, center=[0, 0, 0], axis='z'):
    """
    旋转点云。

    参数:
    - point_cloud: 要旋转的点云，一个 open3d.geometry.PointCloud 对象。
    - angle_degrees: 旋转角度（以度为单位）。
    - center: 旋转中心点的坐标，格式为 [x, y, z]。
    - axis: 旋转轴，可以是 'x', 'y' 或 'z'。
    """
    # 将角度从度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 根据选定的轴创建旋转矩阵
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_radians), -np.sin(angle_radians)],
                      [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                      [0, 1, 0],
                      [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                      [np.sin(angle_radians), np.cos(angle_radians), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    return R


# generate_xml: 是否检查已经生成xml文件
def render_image(ply_file_path, scene_path, image_path, use_existed_xml=False):
    radius = 0.02 # 0.025
    # 检查xml文件是否存在
    filename = ply_file_path.split('/')[-1].split('.')[0]
    xml_file_path = os.path.join(scene_path, "{}.xml".format(filename))
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    R = rotate_point_cloud(point_cloud, -115, center=[0, 0, 0], axis='x')
    point_cloud.rotate(R, center=(0, 0, 0))
    R = rotate_point_cloud(point_cloud, 90, center=[0, 0, 0], axis='z')
    point_cloud.rotate(R, center=(0, 0, 0))

    if os.path.exists(xml_file_path) and use_existed_xml:
        print("xml file {} already exists!".format(xml_file_path))
    else:
        print("xml file {} does not exist, creating...".format(xml_file_path))  
        # Configure xml
        point_radius=0.01
        xml_head, xml_ball_segment, xml_tail = configure_xml()
        # Load a PLY point cloud
        xyz = np.asarray(point_cloud.points)
        xyz = standardize_bbox(xyz, 2048, shuffle=False, arg_scale=1.0)
        color = np.asarray(point_cloud.colors)

        print("load point cloud with {} points, {} colors".format(xyz.shape[0], color.shape[0]))

        if color.shape[0] == 0:
            color = colormap(xyz[:,0], xyz[:,1], xyz[:,2])
            
        xml_segments = [xml_head.format(filename)]
        for i in range(xyz.shape[0]):
            xml_segments.append(xml_ball_segment.format(radius, xyz[i,0],xyz[i,1],xyz[i,2], color[i,0], color[i,1], color[i,2]))
            # xml_segments.append(xml_ball_segment.format(radius, xyz[i,0],xyz[i,1],xyz[i,2], ) # 自定义rgb颜色

        xml_segments.append(xml_tail)
        
        xml_content = str.join('', xml_segments)
        
        
        with open(xml_file_path, 'w') as f:
            f.write(xml_content)
        
    # 读取xml文件
    mi.set_variant("scalar_rgb")
    scene = mi.load_file(xml_file_path)
    # params = mi.traverse(scene)
    # # params['PLYMesh.vertex_color'] /= 255.0
    # # params.update()
    
    print("load xml file from {}".format(xml_file_path))
    image = mi.render(scene, spp=256)
    
    from mitsuba import ScalarTransform4f as T
    plt.axis("off")
    plt.imshow((image ** (1.0 / 2.2))/255); # approximate sRGB tonemapping
    image_save_to = os.path.join(image_path, "{}.png".format(filename))
    mi.util.write_bitmap(image_save_to, image)
    print("save image {}".format(image_save_to))

def run(ply_file_path):
    # ply_file_path = 'vr_pcd/mug_feat2.ply'
    # ply_file_path = os.path.join('vr_pcd', )
    # ply_file_path = 'vr_pcd/mug1.ply'
    scene_path = './results/scenes/'
    image_path = './results/images/'
    render_image(ply_file_path, scene_path, image_path, use_existed_xml=False)

if __name__ == '__main__':
    run(sys.argv[1])
