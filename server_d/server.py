import grpc
from concurrent import futures
from PIL import Image
from io import BytesIO
from from_proto import my_pb2_grpc
from from_proto import my_pb2
import os
import torch
from neural.model import CommonNet
import numpy as np
import matplotlib.pyplot as plt


class FileTransferService(my_pb2_grpc.FileTransferServiceServicer):
    def __init__(self):
        self.image_name1 = None
        self.image_name2 = None
        self.merged_image_name = None
        self.device = torch.device('cpu')
        self.model = CommonNet()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('/Users/aroslavsapoval/myProjects/Practic3/GRPC_stream_neural/server_d/ves.pth'))
        self.model.eval()

    def work_with_img(self, request, context, case_nom):
        otv = []
        # width_1, height_1 = 512, 256
        for r in request.images:
            image1 = Image.open(BytesIO(r.image_1))
            image2 = Image.open(BytesIO(r.image_2))
            left_test = np.array(image1.convert('RGB'))  # .resize((width_1, height_1), Image.BICUBIC)
            right_test = np.array(image2.convert('RGB'))  # .resize((width_1, height_1), Image.BICUBIC)

            with torch.no_grad():
                left_test = left_test.transpose(2, 0, 1).astype('float32') / 255.0
                right_test = right_test.transpose(2, 0, 1).astype('float32') / 255.0
                left_test = torch.from_numpy(left_test).unsqueeze(0).to(self.device)
                right_test = torch.from_numpy(right_test).unsqueeze(0).to(self.device)
                pred = self.model(left_test, right_test).squeeze(0)
                pred = pred.squeeze(0)
                pred = pred[..., 100:, :].detach().cpu().numpy()


            self.image_name1 = r.filename1
            self.image_name2 = r.filename2
            self.merged_image_name = self.image_name1 + "_" + self.image_name2
            output_folder = f"merged_image_case_{case_nom}"
            output_folder2 = f"pred_image_case_{case_nom}"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, self.merged_image_name)
            pred_jet_path = os.path.join(output_folder2, self.merged_image_name + '_pred_jet.png')
            # Объедините изображения
            combined_image = Image.blend(Image.fromarray(left_test.astype(np.uint8)),
                                         Image.fromarray(plt.imread(pred_jet_path)), alpha=0.5)

            combined_image.save(output_path, format='PNG')

            otv.append(r.filename1)
        return otv

    def Case5(self, request, context):
        otv = self.work_with_img(request, context, case_nom=5)
        response = my_pb2.FileResponse(message=f"Uspeshno - {otv}")
        return response


def run_server():
    # port = '0.0.0.0:50053'
    port = 'localhost:50054'
    # port = '192.168.3.102:50053'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_receive_message_length', 2000 * 1024 * 1024)])
    my_pb2_grpc.add_FileTransferServiceServicer_to_server(FileTransferService(), server)
    server.add_insecure_port(port)
    server.start()
    print(f"Сервер запущен на порту {port}", flush=True)
    server.wait_for_termination()


if __name__ == '__main__':
    run_server()
