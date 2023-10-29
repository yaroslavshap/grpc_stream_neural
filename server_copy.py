import time
import grpc
from concurrent import futures
from PIL import Image
from io import BytesIO
import my_pb2
import my_pb2_grpc
import os
import torch
from model import CommonNet
import numpy as np


class FileTransferService(my_pb2_grpc.FileTransferServiceServicer):
    def __init__(self):
        self.image_name1 = None
        self.image_name2 = None
        self.merged_image_name = None

    def work_with_img(self, request, context, case_nom):
        device = torch.device('cpu')
        width_1, height_1 = 512, 256
        model = CommonNet()
        model.to(device)
        model.load_state_dict(torch.load(r'/Users/aroslavsapoval/PycharmProjects/grpc_stream_neural/ves.pth'))
        image1 = Image.open(BytesIO(request.image_1))
        image2 = Image.open(BytesIO(request.image_2))
        left_test = np.array(image1.convert('RGB').resize((width_1, height_1), Image.BICUBIC))
        right_test = np.array(image2.convert('RGB').resize((width_1, height_1), Image.BICUBIC))
        model.eval()
        with torch.no_grad():
            left_test = left_test.transpose(2, 0, 1).astype('float32') / 255.0
            right_test = right_test.transpose(2, 0, 1).astype('float32') / 255.0
            left_test = torch.from_numpy(left_test).unsqueeze(0).to(device)
            right_test = torch.from_numpy(right_test).unsqueeze(0).to(device)
            pred = model(left_test, right_test).squeeze(0)
            pred = pred.squeeze(0)
            pred = pred[..., 100:, :].detach().cpu().numpy()

        pred_img = Image.fromarray(255 * pred.astype(np.uint8))
        self.image_name1 = request.filename1
        self.image_name2 = request.filename2
        self.merged_image_name = self.image_name1 + "_" + self.image_name2
        output_folder = f"merged_image_case_{case_nom}"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, self.merged_image_name)
        pred_img.save(output_path, format="PNG")

    def Case5(self, request, context):
        otv = []
        for r in request.images:
            self.work_with_img(r, context, case_nom=5)
            otv.append(self.merged_image_name)

        response = my_pb2.FileResponse(message=f"Uspeshno - {otv}")
        return response


def run_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_receive_message_length', 1000 * 1024 * 1024)])
    my_pb2_grpc.add_FileTransferServiceServicer_to_server(FileTransferService(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    print("Сервер запущен на порту 50053...")
    server.wait_for_termination()


if __name__ == '__main__':
    run_server()
