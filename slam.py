import cv2
import numpy as np
from display import Display
from frame import Frame, denormalize, match_frames
import g2o
import open3d as o3d


# Intrinsic Parameters Of The Camera
W = 1920//2
H = 1080//2
F = 200
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))

# Main classes
display = Display(W, H)


class Map(object):

    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.pcd1 = o3d.geometry.PointCloud()
        self.pcd = o3d.geometry.PointCloud()

    def convert_to_pcd(self, spts, ppts):
        # turn state into points
        self.pcd1.points = o3d.utility.Vector3dVector(spts)
        self.pcd.points = o3d.utility.Vector3dVector(ppts)

    def custom_draw_geometry(self):
        # The following code achieves the same effect as:
        # o3d.visualization.draw_geometries([pcd])
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', width=1920//2, height=1080//2, left=50, top=50, visible=True)
        pcd = o3d.geometry.PointCloud(self.pcd)
        #pcd1 = o3d.geometry.PointCloud(self.pcd1)

        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.update_renderer()
        vis.poll_events()

    def display_map(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.xyz)

        poses = np.asarray(poses)
        pts = np.asarray(pts)

        self.state = poses, pts

        # From state to points
        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array([d[:3] for d in self.state[1]])  # Dropping the last column shape (n, 3) np.delete(self.state[1], 3, 1)

        self.convert_to_pcd(spts, ppts)
        self.custom_draw_geometry()
        #o3d.visualization.draw_geometries_with_custom_animation(self.pcd)


mapp = Map()


class Point(object):
    # Point in the world
    # Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        self.xyz = loc
        self.frames = []
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pt1, pt2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pt1.T, pt2.T).T


def process_frame(img):
    img = cv2.resize(img, (H, W))
    frame = Frame(mapp, img, K)

    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    # Homogenous 3D points
    pts4d /= pts4d[:, 3:]

    # Reject points without enough parallox
    # Reject the points behind the camera
    good_4dpts = np.abs(pts4d[:, 3] > 0.005) & (pts4d[:, 2] > 0)
    pts4d = pts4d[good_4dpts]
    # print(sum(good_4dpts), len(good_4dpts))

    for i,p in enumerate(pts4d):
        if not good_4dpts[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    display.draw(img)

    # Desplaying a stupid map

    mapp.display_map()


if __name__ == "__main__":
    capture = cv2.VideoCapture('carvid.mp4')
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    while capture.isOpened():
        # Capture frame by frame
        ret, frame = capture.read()
        if ret:
            # Rotating and Flipping
            frame = np.fliplr(frame)
            frame = np.rot90(frame)
            # From BRG to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            process_frame(frame)

        else:
            break
