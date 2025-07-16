from zipfile import ZipFile
zip_file = 'digit-recognizer.zip'
extracted_zip_file = ZipFile(zip_file,'r')
extracted_zip_file.extractall()
extracted_zip_file.close()
# if you want to extract all and save it in an another file
from zipfile import ZipFile
zip_file = 'digit-recognizer.zip'
extract_to_folder = 'extracted_data'  # <-- your custom folder
with ZipFile(zip_file, 'r') as extracted_zip_file:
    extracted_zip_file.extractall(path=extract_to_folder)
print(f"ZIP file extracted to: {extract_to_folder}")
import numpy as np
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
X_train = train_data.drop("label",axis=1).values# axis=1 → Tells pandas to drop a column (not a row).
y_train = train_data["label"].values# .values → This converts the pandas Series into a NumPy array.
# Square brackets [] are used for:
# Accessing columns in a DataFrame (like a dictionary).
# Accessing rows in a list/array by index.

#Parentheses () are used for:
# Calling a function or method with arguments.
X_test = test_data.values
X_train = X_train/255.0
X_test = X_test/255.0
# we know in MNIST 0 means black and 255 means white after normalization,
# 0 means black and 1 means white
# neural networks learn better when input values are in a smaller, more consistent range — typically [0, 1].
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
# Convert your flat image data (probably from MNIST or a similar dataset)
# into 4D format suitable for deep learning models.
# Meaning of Each Dimension:
# Dimension	 Value	Meaning
#    -1	     60000	Number of images (inferred automatically)
#    28	      28	Height of the image (pixels)
#    28	      28	Width of the image (pixels)
#    1	      1	    Channels — 1 means grayscale
# Why Use -1?
# -1 tells NumPy to automatically figure out the number of images based on the total number of pixels.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
model = Sequential([
    Conv2D(32,(3,3), activation = 'relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation = 'relu'),
     Dropout(0.5), 
    Dense(10,activation = 'softmax')
    # Softmax is used for multi-class classification, not regression.
])
model.summary()
model.compile( optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'] )
model.fit(X_train,y_train,epochs=5,batch_size=64)
predictions = model.predict(X_test)
# This runs a forward pass of your trained model on the test data X_test.
# probabilities for digits : [0,1,2,3,4,5,6,7,8,9]
# [
# [0.01, 0.03, 0.92, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],  # image 0
# [0.10, 0.70, 0.05, 0.05, 0.02, 0.03, 0.01, 0.01, 0.01, 0.02],  # image 1
# [0.05, 0.05, 0.05, 0.75, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],  # image 2
#           .
#           .
#           .
# ]
predicted_labels = np.argmax(predictions,axis=1)
# in the first case for digit 2 probability is 0.92 so the image is 2
# so upper 3 prediction : predicted_labels = [2, 1, 3,...]
num_images = 5
# Create a figure with subplots (1 row, num_images columns)

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_labels)+1),
    "Label": predicted_labels
})
submission.to_csv("submission.csv", index=False)