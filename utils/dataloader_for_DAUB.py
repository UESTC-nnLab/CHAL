import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess(image):

    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

def enhanced_augmentation(images, boxes, h, w, augment_params):
\
\
\
\
\

    if augment_params.get('flip_horizontal', False) and rand() < augment_params.get('flip_prob', 0.5):
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    if augment_params.get('flip_vertical', False) and rand() < augment_params.get('flip_prob', 0.5):
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        for i in range(len(boxes)):
            boxes[i][[1,3]] = h - boxes[i][[3,1]]

    return np.array(images, dtype=np.float32), np.array(boxes, dtype=np.float32)

def augmentation(images, boxes,h, w, hue=.1, sat=0.7, val=0.4):

    filp = rand()<.5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    return np.array(images,dtype=np.float32), np.array(boxes,dtype=np.float32)

class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5 ,type='train', augment_params=None):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.txt_path = dataset_path
            self.aug = True

            if augment_params is not None:
                self.augment_params = augment_params
            else:
                self.aug = False
                self.augment_params = None
        else:
            self.txt_path = dataset_path
            self.aug = False
            self.augment_params = None
        with open(self.txt_path) as f:
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line_idx, line in enumerate(data_lines):
                line = line.strip('\n').split()
                self.img_idx.append(line[0])

                annotations = []
                for box in line[1:]:
                    box_data = list(map(int, box.split(',')))
                    if len(box_data) >= 5:
                        class_id = box_data[4]

                        if class_id != 0:
                            print(f"Warning: Invalid class_id {class_id} found in line {line_idx}, setting to 0")
                            box_data[4] = 0
                    annotations.append(np.array(box_data))

                self.anno_idx.append(np.array(annotations))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images),(3, 0, 1, 2))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        return images, box

    def get_data(self, index):
        image_data = []

        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        label_data = self.anno_idx[index]
        for id in range(0, self.num_frame):
            img_path_bmp = image_path + '%d.bmp' % max(image_id - id, 0)
            img_path_png = image_path + '%d.png' % max(image_id - id, 0)
            img_path_jpg = image_path + '%d.jpg' % max(image_id - id, 0)
            if os.path.exists(img_path_jpg):
                img = Image.open(img_path_jpg)
            elif os.path.exists(img_path_bmp):
                img = Image.open(img_path_bmp)
            elif os.path.exists(img_path_png):
                img = Image.open(img_path_png)
            else:
                raise FileNotFoundError(f"Image not found: {img_path_bmp} or {img_path_png}")
            img = cvtColor(img)
            iw, ih = img.size

            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            img = img.resize((nw, nh), Image.Resampling.BICUBIC)
            new_img = Image.new('RGB', (w,h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            image_data.append(np.array(new_img, np.float32))

            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]]*nw/iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]]*nh/ih + dy

                label_data[:, 0:2][label_data[:, 0:2]<0] = 0
                label_data[:, 2][label_data[:, 2]>w] = w
                label_data[:, 3][label_data[:, 3]>h] = h

                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w>1, box_h>1)]

        image_data = np.array(image_data[::-1])
        label_data = np.array(label_data, dtype=np.float32)
        if self.aug is True and self.augment_params is not None:

            augmented_images, augmented_boxes = enhanced_augmentation(
                image_data, label_data[:,:4], h, w, self.augment_params)
            image_data = augmented_images

            if len(augmented_boxes) > 0 and len(label_data) > 0:

                label_data[:,:4] = augmented_boxes
            elif len(augmented_boxes) == 0:

                pass
        return image_data, label_data

class RsCarDataset(Dataset):
\
\
\

    def __init__(self, dataset_path, image_size, num_frame=5, type='train', augment_params=None):
        super(RsCarDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame

        if type == 'train':
            self.txt_path = dataset_path
            self.aug = True

            if augment_params is not None:
                self.augment_params = augment_params
            else:
                self.aug = False
                self.augment_params = None
        else:
            self.txt_path = dataset_path
            self.aug = False
            self.augment_params = None

        with open(self.txt_path, 'r') as f:
            data_lines = f.readlines()
            self.length = len(data_lines)

            for line_idx, line in enumerate(data_lines):
                line = line.strip('\n').split()
                if len(line) < 1:
                    continue

                image_path = line[0]
                self.img_idx.append(image_path)

                annotations = []
                for box_str in line[1:]:
                    try:
                        box_data = list(map(int, box_str.split(',')))
                        if len(box_data) >= 4:

                            x1, y1, x2, y2 = box_data[:4]

                            if x1 >= x2 or y1 >= y2:
                                print(f"Warning: Invalid box coordinates in line {line_idx}: {box_data}")
                                continue

                            if len(box_data) >= 5:
                                class_id = box_data[4]
                            else:
                                class_id = 0

                            annotations.append(np.array([x1, y1, x2, y2, class_id]))
                        else:
                            print(f"Warning: Insufficient box data in line {line_idx}: {box_str} (need at least 4 values)")
                            continue
                    except ValueError as e:
                        print(f"Warning: Failed to parse box data in line {line_idx}: {box_str}, error: {e}")
                        continue

                self.anno_idx.append(np.array(annotations) if annotations else np.array([]).reshape(0, 5))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images), (3, 0, 1, 2))
        if len(box) != 0:

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box

    def get_data(self, index):
        image_data = []

        h, w = self.image_size, self.image_size
        file_path = self.img_idx[index]

        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        try:
            image_id = int(os.path.splitext(file_name)[0])
        except ValueError:
            print(f"Warning: Cannot extract image ID from {file_name}, using 0")
            image_id = 0

        label_data = self.anno_idx[index].copy() if len(self.anno_idx[index]) > 0 else np.array([]).reshape(0, 5)

        for id in range(self.num_frame):

            current_id = max(image_id - id, 1)

            base_name = f"{current_id:06d}"
            possible_extensions = ['.jpg', '.bmp', '.png']

            img = None
            for ext in possible_extensions:
                img_path = os.path.join(dir_path, base_name + ext)
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        break
                    except Exception as e:
                        print(f"Warning: Failed to open {img_path}: {e}")
                        continue

            if img is None:
                try:
                    img = Image.open(file_path)
                    if id > 0:
                        print(f"Warning: Frame {current_id} not found, using current frame {image_id}")
                except Exception as e:
                    raise FileNotFoundError(f"Cannot load image: {file_path}, error: {e}")

            img = cvtColor(img)
            iw, ih = img.size

            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            img = img.resize((nw, nh), Image.Resampling.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            image_data.append(np.array(new_img, np.float32))

            if len(label_data) > 0 and id == 0:

                np.random.shuffle(label_data)

                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy

                label_data[:, 0:2] = np.maximum(label_data[:, 0:2], 0)
                label_data[:, 2] = np.minimum(label_data[:, 2], w)
                label_data[:, 3] = np.minimum(label_data[:, 3], h)

                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                valid_mask = np.logical_and(box_w > 1, box_h > 1)
                label_data = label_data[valid_mask]

        image_data = np.array(image_data[::-1])
        label_data = np.array(label_data, dtype=np.float32)

        if self.aug and self.augment_params is not None:
            if len(label_data) > 0:
                augmented_images, augmented_boxes = enhanced_augmentation(
                    image_data, label_data[:, :4], h, w, self.augment_params)
                image_data = augmented_images

                if len(augmented_boxes) > 0:
                    label_data[:, :4] = augmented_boxes
            else:

                augmented_images, _ = enhanced_augmentation(
                    image_data, np.array([]).reshape(0, 4), h, w, self.augment_params)
                image_data = augmented_images

        return image_data, label_data

def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

if __name__ == "__main__":

    print("Testing RsCarDataset class...")

    try:

        augment_params = {
            'flip_horizontal': True,
            'flip_vertical': False,
            'flip_prob': 0.5
        }

        for num_frames in [1, 3, 5]:
            print(f"\n=== Testing with {num_frames} frames ===")

            rscar_dataset = RsCarDataset(
                os.environ.get("CHAL_RSCAR_TRAIN_ANNOTATION", "datasets/RsCarData/coco_train_RsCarData.txt"),
                256,
                num_frames,
                'train',
                augment_params
            )
            print(f"Dataset length: {len(rscar_dataset)}")

            rscar_dataloader = DataLoader(
                rscar_dataset,
                shuffle=True,
                batch_size=2,
                collate_fn=dataset_collate
            )

            print("Testing data loading...")
            t = time.time()
            for index, batch in enumerate(rscar_dataloader):
                images, targets = batch[0], batch[1]
                print(f"  Batch {index}:")
                print(f"    Images shape: {images.shape}")
                print(f"    Number of samples: {len(targets)}")

                for i, target in enumerate(targets):
                    if len(target) > 0:
                        print(f"    Sample {i} targets: {len(target)} objects")
                        print(f"    Sample target shape: {target.shape}")
                        if len(target) > 0:
                            print(f"    First target: {target[0]}")
                    else:
                        print(f"    Sample {i}: No targets")

                if index >= 2:
                    break

            print(f"Time for {index+1} batches: {time.time()-t:.2f}s")

        print(f"\n=== Testing validation dataset ===")
        val_dataset = RsCarDataset(
            os.environ.get("CHAL_RSCAR_VAL_ANNOTATION", "datasets/RsCarData/coco_val_RsCarData.txt"),
            256,
            5,
            'val'
        )
        print(f"Validation dataset length: {len(val_dataset)}")

        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=1,
            collate_fn=dataset_collate
        )

        for index, batch in enumerate(val_dataloader):
            images, targets = batch[0], batch[1]
            print(f"Val batch {index}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Targets: {len(targets[0]) if len(targets) > 0 and len(targets[0]) > 0 else 0} objects")
            break

        print("RsCarDataset test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
