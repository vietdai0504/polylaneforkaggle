import os
import cv2
import numpy as np
import torch
import time

from lib.config import Config

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 255, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def normalize_image_for_test(img, normalize=True):
    # Chuyển ảnh từ BGR sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Đưa ảnh về phạm vi [0, 1]
    img = img / 255.0

    # 2. Chuẩn hóa (nếu cần) theo ImageNet
    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # 3. Chuyển đổi ảnh từ NumPy (H, W, C) sang Tensor (C, H, W)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # chuyển sang (C, H, W)

    return img

def preprocess_image(image, input_size=(360, 640)):
    # Chuyển đổi ảnh sang RGB nếu cần
    if image.shape[2] == 4:  # Ảnh có 4 kênh (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:  # Ảnh có 1 kênh (Grayscale)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize ảnh
    image = cv2.resize(image, input_size)

    # Chuẩn hóa ảnh (scale pixel về [0, 1])
    image = image.astype(np.float32) / 255.0

    # Đổi thứ tự kênh từ (H, W, C) sang (C, H, W)
    image = np.transpose(image, (2, 0, 1))

    # Chuyển đổi sang tensor
    image = torch.from_numpy(image).float()

    return image

y_threshold = None

def draw_annotation(pred=None, img=None, cls_pred=None):
    global y_threshold

    img_h, img_w, _ = img.shape

    if pred is None:
        return img

    # Draw predictions
    pred = pred[pred[:, 0] != 0]  # filter invalid lanes
    overlay = img.copy()
    for i, lane in enumerate(pred):
        color = PRED_MISS_COLOR
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        ys = np.linspace(lower, upper, num=100)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]
        if y_threshold is None:
            y_threshold = points[0][1]
        points = points[points[:, 1] >= y_threshold]
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)

    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)

    return img

def predict_on_image(model, image, test_parameters, save=False, device=None):
    # image = cv2.resize(image, (640, 360))
    input_image = normalize_image_for_test(image).unsqueeze(0).to(device)

    # print(input_image.shape)
    with torch.no_grad():
        outputs = model(input_image)

        outputs = model.decode(outputs, None, **test_parameters)
        outputs, extra_outputs = outputs
        preds = draw_annotation(pred=outputs[0].cpu().numpy(), img=image)
        if save:
            cv2.imwrite('prediction.png', preds)
    return preds


if __name__ == "__main__":
    cfg = Config('cfgs/tusimple.yaml')

    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath('experiments')))
    os.makedirs(exp_root, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cfg.get_model().to(device)
    test_parameters = cfg.get_test_parameters()
    epoch = 10
    if epoch > 0:
        model_path = os.path.join(exp_root, "models", f"model_{epoch:03d}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.to(device)

    # Đọc ảnh thay vì video
    img = cv2.imread('0000.png')  # Đọc ảnh từ file

    # Resize ảnh nếu cần
    frame_resize = cv2.resize(img, (640, 360))

    # Dự đoán trên ảnh
    image = predict_on_image(model, frame_resize, test_parameters, save=True, device=device)

    # Lưu kết quả ảnh dự đoán
    cv2.imwrite('prediction_image.png', image)  # Lưu ảnh kết quả dự đoán
