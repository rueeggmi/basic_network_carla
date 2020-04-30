import os.path
import numpy as np
import event_representations as er
import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class CarlaDataset:
    def __init__(self, root, transform=None, normalize=False):
        self.root = root
        self.transform = transform
        self.normalize = normalize
        self.image_files = self.get_files(root)
        self.measurements = self.get_measurements(root)
        assert len(self.image_files) > 0
        self.sensor_resolution = cv2.imread(self.image_files[0]).shape[:2]

    def get_files(self, root):
        # load all paths that point to images to a list
        IMG_EXTENSIONS = ["png", "jpg"]
        files = []
        assert os.path.isdir(root), '%s is not a valid directory' % root

        for subroot, _, fnames in sorted(os.walk(root)):
            data = {}
            for fname in fnames:
                path = os.path.join(subroot, fname)
                if fname.split(".")[1] in IMG_EXTENSIONS and subroot.split("/")[-1] == "rgb":
                    files += [path]

        return files

    def get_measurements(self, root):
        # load all paths that point to images to a list
        MEASUREMENTS_EXTENSIONS = ["npz"]
        measurements = []
        assert os.path.isdir(root), '%s is not a valid directory' % root

        for subroot, _, fnames in sorted(os.walk(root)):
            data = {}
            for fname in fnames:
                path = os.path.join(subroot, fname)
                if fname.split(".")[1] in MEASUREMENTS_EXTENSIONS and subroot.split("/")[-1] == "measurements":
                    measurements += [path]

        return measurements

    def __getitem__(self, index):
        path = self.image_files[index]
        path_meas = self.measurements[index]

        img = cv2.imread(path)
        meas = np.load(path_meas, allow_pickle=True)
        measurements = meas['measurements']

        #if len(img.shape) == 3:  # if image is RGB, convert to grayscale
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalize to range 0-1 to facilitate learning
        img = img.astype("float32") / 255.0
        # crop image to 160x346
        img = img[80:240, :]

        speed = np.asarray(measurements[0][0], dtype=float)
        steer = np.asarray(measurements[5], dtype=float)
        throttle = np.asarray(measurements[4], dtype=float)
        brake = np.asarray(measurements[6], dtype=float)

        if self.transform:
            img = self.transform(img)
            # speed = self.transform(speed)
            # steer = self.transform(steer)
            # throttle = self.transform(throttle)
            # brake = self.transform(brake)

        if self.normalize:
            # mean & std can be calculated by setting Trainer.calculate_mean_std = True
            normalization = transforms.Normalize((0.2013, 0.2283, 0.2659), (0.1581, 0.1688, 0.2086))
            img = normalization(img)

        return img, speed, steer, throttle, brake
        # return [[img, speed], [steer, throttle, brake]]
        # return img
        # return [img, meas]  # output can be list, dict or tuple for other outputs

    def __len__(self):
        return len(self.image_files)







class EventCameraMemmapEventsDataset:
    def __init__(self, root, sensor_resolution):
        self.root = root
        self.file_handles = self.get_file_handles(root)
        self.sensor_resolution = sensor_resolution

    def get_file_handles(self, root):
        handles = []

        assert os.path.isdir(root), '%s is not a valid directory' % root

        for subroot, _, fnames in sorted(os.walk(root)):
            data = {}
            for fname in fnames:
                path = os.path.join(subroot, fname)

                if fname.endswith(".dat.t"):  # timestamps
                    num_events = int(os.path.getsize(path) / (8.0))  # 8 bytes per entry, 1 entry per event
                    thandle = np.memmap(path, dtype="float64", mode="r", shape=(num_events, 1))
                    data["t"] = thandle

                elif fname.endswith(".dat.xy"): # coordinates
                    num_events = int(os.path.getsize(path) / (2.0 * 2.0))  # 2 bytes per entry, 2 entry per event
                    xyhandle = np.memmap(path, dtype="int16", mode="r", shape=(num_events, 2))
                    data["xy"] = xyhandle

                elif fname.endswith(".dat.p"): # polarity
                    num_events = int(os.path.getsize(path) / (1.0 * 1.0))  # 1 bytes per entry, 1 entry per event
                    phandle = np.memmap(path, dtype="uint8", mode="r", shape=(num_events, 1))
                    data["p"] = phandle

                elif fname.endswith(".npy"):  # index mapping image index to event idx
                    indices = np.load(path)
                    indices = indices[indices>=0].astype("int64")  # ignore event indices which are -1 (before first image)
                    data["index"] = indices

            if len(data) > 0:
                handles.append(data)

        return handles

    def get_relative_index(self, index):
        rel_index = index
        for fh in self.file_handles:
            indices = fh["index"]
            if rel_index < indices.shape[0] - 1:
                selected_handle = fh
                break
            rel_index -= indices.shape[0] - 1
        else:
            raise ValueError("Index %s is too large for dataset with len %s" % (index, len(self)))

        return rel_index, selected_handle

    def select_events(self, fh, idx0, idx1):
        events = np.concatenate(
            [
                fh["xy"][idx0:idx1].astype("float32"),
                fh["t"][idx0:idx1]-fh["t"][idx0],            # timestamps start at 0 within event slice
                2*fh["p"][idx0:idx1].astype("float32")-1,    # maps polarity from [0,1] to [-1,1]
            ],
            axis=-1
        )
        return events

    def __getitem__(self, index):
        rel_index, filehandle = self.get_relative_index(index)

        # select events with indices between current frame at rel_index and next frame at rel_index+1
        indices = filehandle["index"]
        event_index_current_frame = indices[rel_index]
        event_index_next_frame = indices[rel_index+1]
        events = self.select_events(filehandle, event_index_current_frame, event_index_next_frame)

        # generate voxel grid which has size C x H x W
        interpolation = True
        normalize = True
        H, W = self.sensor_resolution
        channels = 5
        voxel_grid = er.voxel_grid(events, channels, H, W, interpolation, normalize)
        voxel_grid = voxel_grid.astype("float32")

        return voxel_grid

    def __len__(self):
        len = 0
        for fh in self.file_handles.values():
            len += fh["index"].shape[0]-1
        return len

if __name__ == '__main__':
    dataset = EventCameraMemmapEventsDataset(root="/data/scratch/daniel/event_sim2real/event_camera_dataset")
    dataset[10]
