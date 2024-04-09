import cv2
import numpy as np


def augment_image(image, angle):
    height, width = image.shape[:2]
    center = (width/2, height/2)

    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, rotate_matrix, (width, height))
    
    # translation_matrix = np.float32([[1, 0, 100], [0, 1, 600]])
    # return cv2.warpAffine(rotated, translation_matrix, (width, height))


def write_aug(vid_name: str, aug_angle: int):
    vid = cv2.VideoCapture(vid_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = f"{aug_angle}_{vid_name}"
    out = cv2.VideoWriter(out_name, fourcc, 30.0, (1280, 720))

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        # frame = cv2.bitwise_not(frame)
        frame = augment_image(frame, aug_angle)
        # cv2.imshow("", frame)
        out.write(frame)

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q'):
            break
    
    vid.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for i in ['IMG_1855.MP4', "IMG_1870.MP4"]:
        for j in [45, 135, 180, 225, 315]:
            write_aug(i, j)

