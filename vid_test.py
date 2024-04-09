from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    VID_NAME = 'vids/IMG_1855.MP4'
    SKIP_SEC_BEGINNING = 0

    vid = cv.VideoCapture(VID_NAME)
    vid_fps = vid.get(cv.CAP_PROP_FPS)
    vid.set(1, vid_fps * SKIP_SEC_BEGINNING)
    ret, frame1 = vid.read()
    ret, frame2 = vid.read()
    fr_counter = 2

    model = YOLO("aero2.pt")

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_name = "IMG_1855_PREDICTED_AERO2.MP4"
    out = cv.VideoWriter(out_name, fourcc, 30.0, (1280, 720))
    while ret and vid.isOpened():
        # res = model(cv.cvtColor(frame1, cv.COLOR_BGR2RGB), size=640)
        results = model.predict(source=frame1)
        for res in results:
            boxes = res.boxes.xyxy
            for box in boxes:
                cv.rectangle(
                    frame1, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    (255),
                    thickness=1
                    )
        
        out.write(frame1)
            
        cv.imshow("", frame1)
        waitkey = cv.waitKey(1)
        if waitkey == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = vid.read()
        fr_counter += 1
        sec = fr_counter / int(vid_fps)

    out.release()
    vid.release()
    cv.destroyAllWindows()
