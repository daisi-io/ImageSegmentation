from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.predict import visualize_segmentation
import cv2
import io
import base64

model = pspnet_50_ADE_20K()


def compute(image_path):
    out = model.predict_segmentation(
        inp=image_path
    )
    
    # visualize segmentation
    input = cv2.imread(image_path, 1)
    seg_img = visualize_segmentation(out, input, n_classes=model.n_classes)
    is_success, buffer = cv2.imencode(".jpg", seg_img)
    io_buf = io.BytesIO(buffer)
    s = base64.b64encode(io_buf.getvalue()).decode("utf-8").replace("\n", "")
    
    return [
        {"type": "image", "label": "segmented_image", "data":  {"alt": "Image Segmentation", "src": "data:image/png;base64, " + s}}
    ]

if __name__ == "__main__":
    input = "/Users/zhenshanjin/Documents/Belmont/sandy/UtilityDaisies/ImageSegmentation/1_input.jpg"
    s = compute(input)