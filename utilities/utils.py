"""
    Brief: Utility functions of apolloscape tool kit
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions representation.
    Input:
        angle: n x 3 matrix, each row is [roll, pitch, yaw]
    Output:
        q: n x 4 matrix, each row is corresponding quaternion.
    """

    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = angle[:, 0], angle[:, 1], angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    return q


def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 1 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                print(q)
                q[3] = 0

    return q



def quaternion_to_euler_angle(q):

    """Convert quaternion to euler angel.
    Input:
        q: 1 * 4 vector,
    Output:
        angle: 1 x 3 vector, each row is [roll, pitch, yaw]
    """
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def intrinsic_vec_to_mat(intrinsic, shape=None):
    """Convert a 4 dim intrinsic vector to a 3x3 intrinsic
       matrix
    """
    if shape is None:
        shape = [1, 1]

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = intrinsic[0] * shape[1]
    K[1, 1] = intrinsic[1] * shape[0]
    K[0, 2] = intrinsic[2] * shape[1]
    K[1, 2] = intrinsic[3] * shape[0]
    K[2, 2] = 1.0

    return K


def round_prop_to(num, base=4.):
    """round a number to integer while being propotion to
       a given base number
    """
    return np.ceil(num / base) * base


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    yaw, pitch, roll = angle[0], angle[1], angle[2]

    yawMatrix = np.matrix([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]])

    pitchMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]])

    rollMatrix = np.matrix([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]])

    R = np.dot(yawMatrix, np.dot(pitchMatrix, rollMatrix))
    R = np.array(R).T

    if is_dir:
        R = R[:, 2]

    return R


def rotation_matrix_to_euler_angles(R, check=True):
    """Convert rotation matrix to euler angles
    Input:
        R: 3 x 3 rotation matrix
        check: whether Check if a matrix is a valid
            rotation matrix.
    Output:
        euler angle [x/roll, y/pitch, z/yaw]
    """

    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        #return n < 3 *(1e-6)
        # Di Wu relax the condition for TLESS dataset
        return n < 1e-5

    if check:
        assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def convert_pose_mat_to_6dof(pose_file_in, pose_file_out):
    """Convert a pose file with 4x4 pose mat to 6 dof [xyz, rot]
    representation.
    Input:
        pose_file_in: a pose file with each line a 4x4 pose mat
        pose_file_out: output file save the converted results
    """

    poses = [line for line in open(pose_file_in)]
    output_motion = np.zeros((len(poses), 6))
    f = open(pose_file_out, 'w')
    for i, line in enumerate(poses):
        nums = line.split(' ')
        mat = [np.float32(num.strip()) for num in nums[:-1]]
        image_name = nums[-1].strip()
        mat = np.array(mat).reshape((4, 4))

        xyz = mat[:3, 3]
        rpy = rotation_matrix_to_euler_angles(mat[:3, :3])
        output_motion = np.hstack((xyz, rpy)).flatten()
        out_str = '%s %s\n' % (image_name, np.array2string(output_motion,
            separator=',',
            formatter={'float_kind':lambda x: "%.7f" % x})[1:-1])
        f.write(out_str)
    f.close()

    return output_motion


def trans_vec_to_mat(rot, trans, dim=4):
    """ project vetices based on extrinsic parameters
    """
    mat = euler_angles_to_rotation_matrix(rot)
    mat = np.hstack([mat, trans.reshape((3, 1))])
    if dim == 4:
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat


def project(pose, scale, vertices):
    """ transform the vertices of a 3D car model based on labelled pose
    Input:
        pose: 0-3 rotation, 4-6 translation
        scale: the scale at each axis of the car
        vertices: the vertices position
    """

    if np.ndim(pose) == 1:
        mat = trans_vec_to_mat(pose[:3], pose[3:])
    elif np.ndim(pose) == 2:
        mat = pose

    vertices = vertices * scale
    p_num = vertices.shape[0]

    points = vertices.copy()
    points = np.hstack([points, np.ones((p_num, 1))])
    points = np.matmul(points, mat.transpose())

    return points[:, :3]


def plot_images(images,
                layout=[2, 2],
                fig_size=10,
                save_fig=False,
                fig_name=None):
    """Plot a dictionary of images:
    Input:
        images: dictionary {'image', image}
        layout: the subplot layout of output
        fig_size: size of figure
        save_fig: bool, whether save the plot images
        fig_name: if save_fig, then provide a name to save
    """

    plt.figure(figsize=(10, 5))
    pylab.rcParams['figure.figsize'] = fig_size, fig_size / 2
    Keys = images.keys()
    for iimg, name in enumerate(Keys):
        assert len(images[name].shape) >= 2

    for iimg, name in enumerate(Keys):
        s = plt.subplot(layout[0], layout[1], iimg + 1)
        plt.imshow(images[name])

        s.set_xticklabels([])
        s.set_yticklabels([])
        s.set_title(name)
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')

    plt.tight_layout()
    if save_fig:
        pylab.savefig(fig_name)
    else:
        plt.show()


def extract_intrinsic(dataset):
    intrinsic_mat = dataset.Car3D.get_intrinsic_mat()
    fx = intrinsic_mat[0][0]
    fy = intrinsic_mat[1][1]
    cx = intrinsic_mat[0][2]
    cy = intrinsic_mat[1][2]
    return fx, fy, cx, cy


def im_car_trans_geometric(dataset, boxes, euler_angle, car_cls, im_scale=1.0):
    ###
    fx, fy, cx, cy = extract_intrinsic(dataset)

    car_cls_max = np.argmax(car_cls, axis=1)
    car_names = [dataset.Car3D.car_id2name[x].name for x in dataset.Car3D.unique_car_models[car_cls_max]]

    if im_scale != 1:
        raise Exception("not implemented, check it")
    boxes = boxes / im_scale

    car_trans_pred = []
    for car_idx in range(boxes.shape[0]):
        box = boxes[car_idx]
        xc = ((box[0] + box[2]) / 2 - cx) / fx
        yc = ((box[1] + box[3]) / 2 - cy) / fy
        ymax = (box[3] - cy) / fy

        # project 3D points to 2d image plane
        euler_angle_i = euler_angle[car_idx]
        rmat = euler_angles_to_rotation_matrix(euler_angle_i)

        car = dataset.Car3D.car_models[car_names[car_idx]]
        x_y_z_R = np.matmul(rmat, np.transpose(np.float32(car['vertices'])))
        Rymax = x_y_z_R[1, :].max()
        Rxc = x_y_z_R[0, :].mean()
        Ryc = x_y_z_R[1, :].mean()
        Rzc = x_y_z_R[2, :].mean()
        zc = (Ryc - Rymax) / (yc - ymax)

        xt = zc * xc - Rxc
        yt = zc * yc - Ryc
        zt = zc - Rzc
        pred_pose = np.array([xt, yt, zt])
        car_trans_pred.append(pred_pose)

    return np.array(car_trans_pred)


def im_car_trans_geometric_ssd6d(dataset, boxes, euler_angle, car_cls, im_scale=1.0):
    ###
    fx, fy, cx, cy = extract_intrinsic(dataset)

    car_cls_max = np.argmax(car_cls, axis=1)
    car_names = [dataset.Car3D.car_id2name[x].name for x in dataset.Car3D.unique_car_models[car_cls_max]]

    if im_scale != 1:
        raise Exception("not implemented, check it")
    boxes = boxes / im_scale

    car_trans_pred = []
    # canonical centroid zr = 10.0
    zr = 10.0
    trans_vect = np.zeros((3, 1))
    trans_vect[2] = zr
    for car_idx in range(boxes.shape[0]):
        box = boxes[car_idx]

        # lr denotes diagonal length of the precomputed bounding box and ls denotes the diagonal length
        # of the predicted bounding box on the image plane
        ls = np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1])**2)
        # project 3D points to 2d image plane
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        euler_angle_i = euler_angle[car_idx]
        rmat = euler_angles_to_rotation_matrix(euler_angle_i)
        car = dataset.Car3D.car_models[car_names[car_idx]]
        x_y_z_R = np.matmul(rmat, np.transpose(np.float32(car['vertices'])))
        x_y_z_R_T = x_y_z_R + trans_vect
        x_y_z_R_T_hat = x_y_z_R_T / x_y_z_R_T[2, :]

        u = fx * x_y_z_R_T_hat[0, :] + cx
        v = fy * x_y_z_R_T_hat[1, :] + cy
        lr = np.sqrt((u.max() - u.min())**2 + (v.max() - v.min())**2)

        zs = lr * zr / ls

        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2
        xc_syn = (u.max() + u.min())/2
        yc_syn = (v.max() + v.min())/2

        xt = zs * (xc - xc_syn) / fx
        yt = zs * (yc - yc_syn) / fy

        pred_pose = np.array([xt, yt, zs])
        car_trans_pred.append(pred_pose)

    return np.array(car_trans_pred)


if __name__ == '__main__':
    pass
