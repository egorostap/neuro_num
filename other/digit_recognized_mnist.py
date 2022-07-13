# нейронка распознает цифры датасет мнист со структурой v2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory


# # загрузка датасета mnist и стандартизация входных данных
# def load_data():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = x_train / 255
#     x_test = x_test / 255
#
#     y_train_cat = keras.utils.to_categorical(y_train, 10)
#     y_test_cat = keras.utils.to_categorical(y_test, 10)
#
#     return x_train, y_train_cat, x_test, y_test_cat, y_test


# # тренирует модель
# def train_model(model, x_train, y_train_cat):
#     model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
#     # model.evaluate(x_test, y_test_cat)
#     model.save('cnn_digits_28x28.h5')

# # Распознавание тестовой выборки
# def test_predicate(model, x_test, y_test):
#     pred = model.predict(x_test)
#     pred = np.argmax(pred, axis=1)
#     print('результаты по тестовой выборке')
#     print(pred[:10])
#     print(y_test[:10])

# # Распознавание цифр
# def digits_predict(model, pics):
#     print('результаты по загруженным изображениям')
#     for pic in pics:
#         image_size = 28
#         img = keras.preprocessing.image.load_img(pic, target_size=(image_size, image_size), color_mode='grayscale')
#         img_arr = np.expand_dims(img, axis=0)
#         img_arr = 1 - img_arr / 255.0
#         img_arr = img_arr.reshape((1, 28, 28, 1))
#
#         pred = model.predict(img_arr)
#         pred = np.argmax(pred, axis=1)
#         print(f'вероятно на картинке цифра: {pred[0]}')


# загрузка кастомного датасета на доработке
def load_custom_dataset():
    ds_train = image_dataset_from_directory(directory='dataset/train', labels='inferred', label_mode='categorical', image_size=(28, 28), color_mode='grayscale')
    ds_val = image_dataset_from_directory(directory='dataset/train', shuffle=False, labels='inferred', label_mode='categorical', image_size=(28, 28), color_mode='grayscale')
    # for i in ds_train:
    #     print(i)

    return ds_train, ds_val


# создание модели
def make_model():
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),  # здесь у нас входной размер на этот слой, ты изначально подавал (256, 256, 3)
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # тут количество классов, я тестила на 3, но для твоей задачи их 10
    ])
    # print(model.summary())      # вывод структуры НС в консоль
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# тренирует модель кастомного датасета
def train_model_custom(model, train_ds, val_ds):
    # Здесь подаем трейн и валидацию для обучения
    model.fit(train_ds, batch_size=32, epochs=50, validation_split=0.2, validation_data=(val_ds))
    model.save('cnn_digits_28x28.h5')


# Распознавание тестовой выборки кастомного датасета
def test_predicate_custom(model, ds_val):
    pred = model.predict(ds_val)
    pred = np.argmax(pred, axis=1)
    print('результаты по тестовой выборке')
    print(pred)


# Распознавание цифр для кастомного датасета
def digits_predict_custom(model, pics):
    print('результаты по загруженным изображениям')
    for pic in pics:
        image_size = 28
        img = keras.preprocessing.image.load_img(pic, target_size=(image_size, image_size), color_mode='grayscale')
        img_arr = np.expand_dims(img, axis=0)
        img_arr = 1 - img_arr / 255.0
        img_arr = img_arr.reshape((1, 28, 28, 1))

        pred = model.predict(img_arr)
        pred = np.argmax(pred, axis=1)
        print(f'вероятно на картинке цифра: {pred[0]}')


if __name__ == "__main__":
    # pass

    # для кстомного датасета
    ds_train, ds_val = load_custom_dataset()
    print(ds_train.class_names)
    print(ds_val.class_names)

    # создание модели
    model = make_model()
    # обучение модели
    # train_model_custom(model, ds_train, ds_val)
    # загрузка обученной модели если имеются данные
    # model = keras.models.load_model('cnn_digits_28x28_custom.h5')
    # # предсказание тестовой выборки
    # test_predicate_custom(model, ds_val)
    # предсказание изображений из директории
    # pics = ['test_nums/0.png', 'test_nums/1.png', 'test_nums/2.png', 'test_nums/3.png', 'test_nums/4.png']
    # digits_predict_custom(model, pics)

    # для датасета мнист
    # x_train, y_train_cat, x_test, y_test_cat, y_test = load_data()
    # # создание модели
    # model = make_model()
    # # обучение модели
    # # train_model(model, x_train, y_train_cat)
    # # загрузка обученной модели если имеются данные # model = keras.models.load_model('cnn_digits_28x28.h5')
    # # предсказание тестовой выборки
    # test_predicate(model, x_test, y_test)
    # # предсказание изображений из директории
    # pics = ['test_nums/0.png', 'test_nums/1.png', 'test_nums/5.png', 'test_nums/8.png', 'test_nums/9.png']
    # digits_predict(model, pics)

