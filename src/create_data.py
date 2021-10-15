import os
from random import choice


f = open("people.txt", "w")
people_list = os.listdir("People")
for i, person_idx in enumerate(people_list):
    pos_image_names = os.listdir(os.path.join("People", person_idx))
    for img_name in pos_image_names:
        img_path = os.path.join("People", person_idx, img_name)

        neg_person_idx = people_list[choice([j for j in range(0, 22) if i != j])]
        neg_image_names = os.listdir(os.path.join("People", neg_person_idx))
        neg_img_name = choice(neg_image_names)
        neg_img_path = os.path.join("People", neg_person_idx, neg_img_name)

        pos_img_name = choice(pos_image_names)
        pos_img_path = os.path.join("People", person_idx, pos_img_name)

        f.write(img_path + " " + neg_img_path + " " + pos_img_path + '\n')

f.close()
