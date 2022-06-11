# нейронка распознает цифры датасет мнист со структурой v2

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory


# загрузка датасета mnist и стандартизация входных данных
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train_cat, x_test, y_test_cat, y_test


# загрузка кастомного датасета на доработке
def load_custom_dataset():
    train = image_dataset_from_directory(directory='dataset/train', labels='inferred')
    x_train = list(train.as_numpy_iterator())[0][0]
    y_train_cat = keras.utils.to_categorical(train.class_names, 10)

    # стандартизация входных данных
    x_train = x_train / 255

    # x_test = x_test / 255
    # print(x_train)
    # print('---------------------')
    # print(y_train_cat)
    # for i in x_train:
    #     print(i)
    # for i in list(x_train.as_numpy_iterator())[0][0]:
    #     print(i)


# создание модели
def make_model():
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    # print(model.summary())      # вывод структуры НС в консоль

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# тренируем модель
def train_model(model, x_train, y_train_cat):
    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
    # model.evaluate(x_test, y_test_cat)
    model.save('cnn_digits_28x28.h5')


# Распознавание тестовой выборки
def test_predicate(model, x_test, y_test):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print('результаты по тестовой выборке')
    print(pred[:10])
    print(y_test[:10])


# Распознавание цифр
def digits_predict(model, pics):
    print('результаты по загруженным изображениям')
    for pic in pics:
        image_size = 28
        img = keras.preprocessing.image.load_img(pic,
                                                 target_size=(image_size, image_size), color_mode='grayscale')
        img_arr = np.expand_dims(img, axis=0)
        img_arr = 1 - img_arr / 255.0
        img_arr = img_arr.reshape((1, 28, 28, 1))

        pred = model.predict(img_arr)
        pred = np.argmax(pred, axis=1)
        print(f'вероятно цифра на картинке: {pred[0]}')


if __name__ == "__main__":
    x_train, y_train_cat, x_test, y_test_cat, y_test = load_data()
    # создание модели
    model = make_model()
    # обучение модели
    # train_model(model, x_train, y_train_cat)
    # загрузка обученной модели если имеется
    model = keras.models.load_model('cnn_digits_28x28.h5')
    # предсказание тестовой выборки
    test_predicate(model, x_test, y_test)
    # предсказание изображений из директории
    pics = ['test/0.png', 'test/1.png', 'test/5.png', 'test/8.png', 'test/9.png']
    digits_predict(model, pics)