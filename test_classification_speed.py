from roboflow import Roboflow
import time
import numpy as np
import glob
import os

## DEFINITIONS
# glob params
image_dir = os.path.join("Images")
file_extension_type = ".jpg"

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("12072022_classifier")
model = project.version(1).model

image_glob = glob.glob('Images' + '/*' + file_extension_type)

# print(image_glob)

time_array = []

# perform upload
for image in image_glob:

    t1 = time.time()

    # infer on a local image
    print(model.predict(image).json())

    t2 = time.time()

    speed = t2-t1
    # print(speed)

    time_array.append(speed)

# print(time_array)

total_speed = sum(time_array)
average_speed = np.mean(time_array)

print()
print("TOTAL SPEED IN SECONDS: " + str(total_speed))
print("AVERAGE SPEED IN SECONDS: " + str(average_speed))




