from src.yolo import pr_score
import torch
import time

model = model = torch.hub.load('./yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

start = time.time()
pr_score(model)
exec_time = time.time()-start
print(f"Execution time: {exec_time}")
