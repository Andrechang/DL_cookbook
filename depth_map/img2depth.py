
import cv2
import torch
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq
import PIL
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def img2depth(image, outname = 'example.parquet'):
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

 
    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu()
    records = [{"data": output.view(-1).numpy(), 
                "shape": list(output.shape)}]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, outname)

    return output

if __name__ == "__main__":
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outp = img2depth(img)





