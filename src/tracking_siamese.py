from siamese_model import SiameseNet, EmbeddingNet
import torch
from memory import Memory
import os
import pickle
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import random


def random_color():
    return tuple(np.random.choice(range(256), size=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        default='/home/jiaqi/PycharmProjects/epic-kitchens-100-hand-object-bboxes/handobj_detections/handobj_detections')
    args = parser.parse_args()
    data_base_path = args.root_dir

    embedding_net = EmbeddingNet()
    discriminator = SiameseNet(embedding_net).cuda()
    discriminator.load_state_dict(torch.load("model_v2.pth"))
    discriminator.eval()
    memory = Memory(discriminator)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
            0.229, 0.224, 0.225])
    ])

    font1 = ImageFont.truetype("news serif bolditalic.ttf", 25)
    font2 = ImageFont.truetype("news serif bolditalic.ttf", 20)

    person_colors = [random_color() for i in range(20)]
    hand_colors = [random_color() for i in range(40)]

    for video in os.listdir(data_base_path):
        memory.reset()
        frame_base = os.path.join(data_base_path, video, "frames")
        det_base = os.path.join(data_base_path, video, "frames_det_meta_openpose")
        frames = os.listdir(frame_base)
        frames.sort()

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(video+".mp4", fourcc, 5, (1280, 720))

        frames = frames[50:300]

        for frame_idx, frame in enumerate(frames):
            print(frame_idx)
            img_path = os.path.join(frame_base, frame)
            image = Image.open(img_path)
            detection_file = frames[frame_idx].replace(".jpg", ".pkl")
            det_path = os.path.join(det_base, detection_file)

            with open(det_path, 'rb') as handle:
                data = pickle.load(handle)
                body_boxes = data['body_boxes']
                hand_boxes = data['hand_boxes']
                for i, body_box in enumerate(body_boxes):
                    x1, y1, x2, y2 = body_box
                    image_patch = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    image_patch = transform(image_patch).unsqueeze(0).cuda()
                    person_idx = memory.predict(image_patch)

                    draw = ImageDraw.Draw(image)
                    draw.rectangle(body_box, width=4, outline=person_colors[person_idx])

                    draw.text((x1+5, y1+5), "Person" + str(person_idx), font=font1, fill=person_colors[person_idx])
                    for hand_info in hand_boxes[i]:
                        if hand_info[0] is None:
                            continue
                        hand_bbox = hand_info[:4]
                        is_left = hand_info[4]
                        if is_left:
                            draw.text((hand_bbox[0] + 5, hand_bbox[1] + 5), "P" + str(person_idx) + "L",
                                      font=font2, fill=hand_colors[person_idx*2])
                            draw.rectangle(hand_bbox, width=4, outline=hand_colors[person_idx * 2])
                        else:
                            draw.text((hand_bbox[0] + 5, hand_bbox[1] + 5), "P" + str(person_idx) + "R",
                                      font=font2, fill=hand_colors[person_idx*2+1])
                            draw.rectangle(hand_bbox, width=4, outline=hand_colors[person_idx*2+1])

            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(img)

        cv2.destroyAllWindows()
        out.release()


if __name__ == '__main__':
    main()
