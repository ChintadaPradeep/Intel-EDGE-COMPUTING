//Edge-Based Real-Time YOLOv8 Object Detection with Multi-View Image Processing//
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import time

def draw_detections(img, boxes, scores, class_ids, class_names=None, color=(0,255,0)):
    for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cid] if class_names else cid}:{conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1]-6), (x1 + t_size[0]+6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_binary(gray, thresh=127):
    _, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return b

def adaptive_binary(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def to_edges(gray):
    return cv2.Canny(gray, 100, 200)

def sobel_edges(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad = cv2.convertScaleAbs(grad_x + grad_y)
    return abs_grad

def apply_colormap(gray, cmap=cv2.COLORMAP_JET):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cmap)

def heatmap_from_magnitude(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

def night_vision(img):
    img_f = img.astype(np.float32)/255.0
    b, g, r = cv2.split(img_f)
    g = np.clip(g*1.8, 0,1.0)
    merged = cv2.merge([b,g,r])
    merged = np.clip((merged-0.5)*1.2+0.5,0,1.0)
    out = (merged*255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3,3),0)
    tint = np.zeros_like(out)
    tint[...,1]=10
    return cv2.add(out,tint)

def sepia(img):
    kernel = np.array([[0.272,0.534,0.131],
                       [0.349,0.686,0.168],
                       [0.393,0.769,0.189]])
    sep = cv2.transform(img, kernel)
    return np.clip(sep,0,255).astype(np.uint8)

def blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def rgb_channels(img):
    b,g,r = cv2.split(img)
    return cv2.merge([b,b,b]), cv2.merge([g,g,g]), cv2.merge([r,r,r])

def ensure_bgr(img):
    if len(img.shape)==2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def timestamp_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def label_tile(img, name):
    img = ensure_bgr(img)
    cv2.putText(img, name, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return img

def make_dynamic_grid(tiles, screen_w=1920, screen_h=1080, max_cols=4, bg_color=(30,30,30)):
    if len(tiles) == 0:
        return np.zeros((screen_h, screen_w,3), dtype=np.uint8)
    n = len(tiles)
    rows = int(np.ceil(n / max_cols))
    grid_rows = []
    for r in range(rows):
        start_idx = r * max_cols
        end_idx = min((r+1)*max_cols, n)
        row_tiles = tiles[start_idx:end_idx]
        cols = len(row_tiles)
        tile_w = screen_w // cols
        tile_h = screen_h // rows
        resized_row = [cv2.resize(ensure_bgr(t), (tile_w, tile_h)) for t in row_tiles]
        grid_rows.append(cv2.hconcat(resized_row))
    return cv2.vconcat(grid_rows)

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    detect=True
    fps, last_time = 0.0, time.time()
    record=False
    out=None
    binary_thresh=127
    print("Controls: SPACE=toggle YOLO, v=record, +/-=binary threshold, q=quit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        tiles=[]
        gray = to_grayscale(frame)
        if detect:
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            tiles.append(label_tile(annotated, "YOLO"))
        tiles.append(label_tile(frame, "Original"))
        tiles.append(label_tile(gray, "Grayscale"))
        tiles.append(label_tile(to_binary(gray,binary_thresh), "Binary"))
        tiles.append(label_tile(adaptive_binary(gray), "Adaptive Binary"))
        tiles.append(label_tile(to_edges(gray), "Edges"))
        tiles.append(label_tile(sobel_edges(gray), "Sobel"))
        tiles.append(label_tile(apply_colormap(gray), "Colormap"))
        tiles.append(label_tile(heatmap_from_magnitude(gray), "Heatmap"))
        tiles.append(label_tile(night_vision(frame), "Night Vision"))
        tiles.append(label_tile(sepia(frame), "Sepia"))
        tiles.append(label_tile(blur(frame), "Blur"))
        b,g,r = rgb_channels(frame)
        tiles.extend([label_tile(b,"Blue Channel"), label_tile(g,"Green Channel"), label_tile(r,"Red Channel")])
        combined = make_dynamic_grid(tiles, screen_w=1920, screen_h=1080)
        cur_time=time.time()
        dt = cur_time - last_time
        fps = 0.9*fps + 0.1*(1.0/dt if dt>0 else 0)
        last_time = cur_time
        cv2.putText(combined,f"FPS: {fps:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(combined,f"Time: {timestamp_text()}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow("Ultimate YOLO Multi-Processing", combined)
        if record:
            if out is None:
                h_out,w_out=combined.shape[:2]
                out = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"XVID"),20,(w_out,h_out))
            out.write(combined)
        key = cv2.waitKey(1) & 0xFF
        if key==ord("q"): break
        elif key==32: detect=not detect
        elif key==ord("v"): record=not record; print("Recording:",record)
        elif key==ord("+"): binary_thresh=min(255,binary_thresh+5)
        elif key==ord("-"): binary_thresh=max(0,binary_thresh-5)
    cap.release()
    cv2.destroyAllWindows()
    if out is not None: out.release()

if __name__=="__main__":
    main()
