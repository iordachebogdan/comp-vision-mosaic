import pickle
import numpy as np


class CifarBusiness:
    def __init__(self, dir: str, grayscale: bool):
        self.dict_types = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }
        self.images = []
        self.type_classes = []
        for i in range(1, 6):
            file = dir + "data_batch_" + str(i)
            with open(file, "rb") as fo:
                dict = pickle.load(fo, encoding="latin1")
                images = dict["data"]
                self.images.extend(
                    images.reshape(10000, 3, 32, 32)
                    .transpose(0, 2, 3, 1)
                    .astype(np.uint8)
                )
                self.type_classes.extend(dict["labels"])
        # convert RGB to BGR
        self.images = np.array(self.images)
        self.images = self.images[:, :, :, [2, 1, 0]]
        if grayscale:
            # convert to grayscale
            self.images = np.mean(self.images, axis=3)
        self.type_classes = np.array(self.type_classes)

    def read_images(self, type: str):
        tp = self.dict_types[type]
        return self.images[self.type_classes == tp]
