import os
from PIL import Image
import numpy as np
import pickle
# cat: 0; dog: 1
# import the data

class ImageProcess:
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def import_train(self, path):
        X = []
        y = []
        for filename in os.listdir(path):
            if filename[:3] == "cat":
                with Image.open(os.path.join(path, filename)) as image:
                    X.append(np.array(image.resize((self.W, self.H))) / 255)
                    y.append(0)
            elif filename[:3] == "dog":
                with Image.open(os.path.join(path, filename)) as image:
                    X.append(np.array(image.resize((self.W, self.H))) / 255)
                    y.append(1)
        
        with open("train.pkl", "wb") as file:
            pickle.dump((np.array(X), np.array(y)), file)


    def import_test(self, path):
        X = []
        numbers = []
        for filename in os.listdir(path):
            # the filename is 1.jpg, 2.jpg, 3.jpg etc, we take the number
            numbers.append(int(filename[:-4]))
            with Image.open(os.path.join(path, filename)) as image:
                X.append(np.array(image.resize((self.W, self.H))))

        X_and_numbers = sorted(list(zip(X, numbers)), key= lambda x: x[1])
        X = [combined[0] for combined in X_and_numbers]

        with open("test.pkl", "wb") as file:
            pickle.dump(np.array(X), file)

if __name__ == "__main__":
    process = ImageProcess(64, 64)
    process.import_train("./train")
    process.import_test("./test")