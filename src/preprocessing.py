import logging
import os
import pickle
import cv2
import torch.cuda
import torch.utils.data as data
from deepface import DeepFace

train_path = "D:/AK/CelebA_Spoof/dataset/train"
test_path = "D:/AK/CelebA_Spoof/dataset/test"
prep_train_path = "D:/AK/CelebA_Spoof/transformed_dataset/train"
prep_test_path = "D:/AK/CelebA_Spoof/transformed_dataset/test"
device = torch.device("cuda:0")
print(torch.cuda.is_available())
print(torch.version.cuda)


def get_face_numpy(path, face_img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    try:
        faces = DeepFace.extract_faces(img, enforce_detection=True, detector_backend='opencv', align=True)
    except ValueError:
        return None
    else:
        if len(faces) != 0:
            x, y, w, h = faces[0]['facial_area']['x'], faces[0]['facial_area']['y'], faces[0]['facial_area']['w'], \
                faces[0]['facial_area']['h']
            face_img = img[y:y + h, x:x + w]
            resized_image = cv2.resize(face_img, face_img_size)
            return resized_image
        return None


def read_image(image_path, size):
    img = cv2.imread(image_path)
    real_h, real_w, c = img.shape
    assert os.path.exists(image_path[:-4] + '_BB.txt'), 'path not exists' + ' ' + image_path

    with open(image_path[:-4] + '_BB.txt', 'r') as f:
        material = f.readline()
        try:
            x, y, w, h, score = material.strip().split(' ')
        except:
            logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')

        try:
            w = int(float(w))
            h = int(float(h))
            x = int(float(x))
            y = int(float(y))
            w = int(w * (real_w / 224))
            h = int(h * (real_h / 224))
            x = int(x * (real_w / 224))
            y = int(y * (real_h / 224))

            # Crop face based on its bounding box
            y1 = 0 if y < 0 else y
            x1 = 0 if x < 0 else x
            y2 = real_h if y1 + h > real_h else y + h
            x2 = real_w if x1 + w > real_w else x + w
            img = img[y1:y2, x1:x2, :]

        except:
            logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def split_dataset(dataset, subset_size=3000):
    num_subsets = int(len(dataset) / subset_size) + 1
    subsets = []
    for i in range(num_subsets):
        start = i * subset_size
        end = min((i+1) * subset_size, len(dataset))
        subset = data.Subset(dataset, range(start, end))
        subsets.append(subset)
    return subsets


def get_data_in_pack(path, face_img_size):
    X_dataset = []
    y_dataset = []
    for i, person_index in enumerate(os.listdir(path)):
        person_path = os.path.join(path, person_index)
        for directory in os.listdir(person_path):
            directory_path = os.path.join(person_path, directory)
            for file_name in os.listdir(directory_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    img_path = os.path.join(directory_path, file_name)
                    face_img = read_image(img_path, face_img_size)
                    if face_img is not None:
                        X_dataset.append(face_img)
                        if os.path.basename(directory_path) == 'live':
                            y_dataset.append([1, 0])
                        if os.path.basename(directory_path) == 'spoof':
                            y_dataset.append([0, 1])
    return X_dataset, y_dataset


def get_data_by_packs_and_save(load_path, save_path, name, img_size):
    i = 0
    for package in os.listdir(load_path):
        package_path = os.path.join(load_path, package)
        print(f"getting {package}...")
        images, labels = get_data_in_pack(package_path, img_size)
        save_file = os.path.join(save_path, f"{name}{i}.pickle")
        with open(save_file, 'wb') as f:
            print(f"saving {name}{i}...")
            pickle.dump([images, labels], f)
        i += 1
        print(package)

if __name__ == '__main__':
    # with open(prep_train_path, 'wb') as f:
    #     pickle.dump("test", f)
    get_data_by_packs_and_save(train_path, prep_train_path, "train", [224, 224])
    get_data_by_packs_and_save(test_path, prep_test_path, "test", [224, 224])
