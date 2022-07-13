# нейросеть распознает цифры 0-9 кастомный датасет

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory


# загрузка кастомного датасета на доработке
def load_custom_dataset():
    ds_train = image_dataset_from_directory(directory='dataset/train', labels='inferred', label_mode='categorical', image_size=(28, 28), color_mode='grayscale')
    ds_val = image_dataset_from_directory(directory='dataset/val', labels='inferred', label_mode='categorical', image_size=(28, 28), color_mode='grayscale')
    ds_test = image_dataset_from_directory(directory='dataset/test', shuffle=False, labels='inferred', label_mode='categorical', image_size=(28, 28), color_mode='grayscale')

    return ds_train, ds_val, ds_test


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
    model.fit(train_ds, batch_size=32, epochs=55, validation_split=0.2, validation_data=(val_ds))
    model.save('cnn_digits_28x28_custom.h5')


# Распознавание тестовой выборки кастомного датасета
def test_predicate_custom(model, test_ds):
    pred = model.predict(test_ds)
    pred = np.argmax(pred, axis=1)
    print('результаты по тестовой выборке')
    print(pred)


if __name__ == "__main__":
    # pass

    # для кстомного датасета
    ds_train, ds_val, ds_test = load_custom_dataset()

    # создание модели
    model = make_model()

    # обучение модели
    # train_model_custom(model, ds_train, ds_val)

    # загрузка обученной модели если имеются данные
    model = keras.models.load_model('cnn_digits_28x28_custom.h5')

    # точность модели
    test_loss, test_acc = model.evaluate(ds_val)
    print('точность модели: ', test_acc)

    # предсказание тестовой выборки кастомного датасета
    predictions = model.predict(ds_test)
    for index in range(10):
        print(f'число тестовой выборки по индексу {index} определяется как: ', np.argmax(predictions[index]))




