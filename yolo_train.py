from ultralytics import YOLO

# model = YOLO("yolov8n.yaml") 
model = YOLO("yolov8s.yaml")
# model = YOLO("yolov8m.yaml")

# model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)

model.train(data="aero.yaml", epochs=20)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
