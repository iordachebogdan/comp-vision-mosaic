import numpy as np
from config.parameters import Parameters
import sys
import cv2 as cv


class Progress:
    def __init__(self, checkpoint: float, parameters: Parameters):
        self.add_checkpoint = checkpoint
        self.next_checkpoint = checkpoint
        self.parameters = parameters

    def update(self, percentage: float, image: np.ndarray):
        sys.stdout.write("\r")
        sys.stdout.write("Progress %d%%" % (int(percentage * 100)))
        sys.stdout.flush()
        if percentage == 1:
            sys.stdout.write("\nWriting solution ...\n")
            cv.imwrite(
                self.parameters.results_dir
                + "result."
                + self.parameters.small_images_type,
                image,
            )
            sys.stdout.write("Done!\n")
        elif percentage >= self.next_checkpoint:
            self.next_checkpoint = min(self.next_checkpoint + self.add_checkpoint, 1)
            cv.imwrite(
                self.parameters.results_dir
                + str(int(100 * percentage))
                + "."
                + self.parameters.small_images_type,
                image,
            )
