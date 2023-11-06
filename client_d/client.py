import grpc
from from_proto.my_pb2 import FileRequest, BatchRequest
from from_proto.my_pb2_grpc import FileTransferServiceStub
from dataloader import Dataloader
from os.path import join
import time
import threading
import copy

# Создаем массив для хранения изображений он сразу идет объектом сообщения из файла proto
batch_request = BatchRequest()
# Создаем блокировку для безопасного доступа к массиву изображений
image_lock = threading.Lock()


# Функция для считывания изображений и добавления их в массив с задержкой
def read_images(images, path_l, path_r):
    global batch_request  # Объявляем, что мы используем глобальную переменную
    while True:
        for i in range(len(images.filenames1)):
            print("Изображение добавленное в массив - ", images.filenames1[i])
            request = zapros(images, path_l, path_r, i)

            with image_lock:  # Блокируем доступ к массиву
                batch_request.images.extend([request])  # Добавляем изображение к batch_request

            print("количество изображений в массиве скопилось - ", len(batch_request.images))
            time.sleep(0.3)  # Задержка в 1 секунду


# функция по которой открываю нужные изображения и создаю запрос
def zapros(images, path_l, path_r, i):
    with open(join(path_l, images.filenames1[i]), 'rb') as f1, open(join(path_r, images.filenames2[i]), 'rb') as f2:
        image1_bytes = f1.read()
        image2_bytes = f2.read()

    request = FileRequest(
        image_1=image1_bytes,
        image_2=image2_bytes,
        filename1=str(images.filenames1[i]),
        filename2=str(images.filenames1[i]))

    return request


# унарная передача
def run_client_case5(stub):
    while True:
        # time.sleep(0.7)
        # print("tut")
        global batch_request
        if batch_request.images:
            data_to_send = copy.copy(batch_request)
            print(len(data_to_send.images))
            # новый пустой объект BatchRequest, который заменит глобальный объект
            batch_request = BatchRequest()
            result = stub.Case5(data_to_send)
            print("Результат - ", result)


def run():
    # Устанавливаем максимальный размер сообщения на клиенте в 10 МБ
    max_message_length = 2000 * 1024 * 1024  # 10 МБ в байтах
    channel = grpc.insecure_channel('localhost:50053', options=(('grpc.max_send_message_length', max_message_length),))
    stub = FileTransferServiceStub(channel)
    # path_l = "/Users/aroslavsapoval/myProjects/data/images_grpc_512/left"
    # path_r = "/Users/aroslavsapoval/myProjects/data/images_grpc_512/right"
    path_l = "data/images_grpc_512/left"
    path_r = "data/images_grpc_512/right"
    images = Dataloader(path_l, path_r)
    while True:
        all_time = []
        print("\n\n\n")
        print("1. Выход")
        print("2 - унарная передача массива")
        # otvet = input("Выберете от 1 до 2:")
        otvet = "2"
        if otvet == "1":
            break
        elif otvet == "2":
            read_thread = threading.Thread(target=read_images, args=(images, path_l, path_r))
            sent_thread = threading.Thread(target=run_client_case5, args=(stub,))
            read_thread.start()
            sent_thread.start()
            # Ожидаем завершения потоков
            read_thread.join()
            sent_thread.join()


if __name__ == '__main__':
    run()
