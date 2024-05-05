import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = to_categorical(y_train, num_classes=10)
Y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1),
          padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=128, epochs=2, validation_data=(X_test, Y_test))

test_scores = model.evaluate(X_test, Y_test, verbose=2)
loss_cnn = test_scores[0] * 100
accuracy_cnn = test_scores[1] * 100
print('Test loss:', loss_cnn)
print('Test accuracy:', accuracy_cnn)

def preprocess_image(image):
  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    resized_image = cv2.resize(gray_image, (28, 28))
    
    normalized_image = resized_image / 255.0
    
    preprocessed_image = normalized_image.reshape(1, 28, 28, 1)
    return preprocessed_image
def perform_segmentation(image):
    
    return [(0, 0, image.shape[1], image.shape[0])]
image = cv2.imread('Capture.PNG')
segmented_characters = perform_segmentation(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, len(segmented_characters) + 1, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

for i, bbox in enumerate(segmented_characters):
    x, y, w, h = bbox
    char_image = image[y:y + h, x:x + w]
  
    preprocessed_char = preprocess_image(char_image)
    predicted_label = np.argmax(model.predict(preprocessed_char))
    plt.subplot(1, len(segmented_characters) + 1, i + 2)
    plt.imshow(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title(f'Char {i + 1}: Predicted {predicted_label}')
    plt.axis('off')

plt.show()
