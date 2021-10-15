import collections
import torch
from statistics import mean


class Memory:
    def __init__(self, discriminator, cache_size=50, reference_size=10, confidence_thres=0.7) -> None:
        self.dic = collections.defaultdict(list)

        self.discriminator = discriminator

        self.cache_size = cache_size
        self.reference_size = reference_size
        self.confidence_thres = confidence_thres

        self.next_id = 0

    def reset(self):
        self.dic.clear()
        self.next_id = 0

    def predict(self, hand_patch):
        confidence_map = collections.defaultdict(int)
        for id, hand_refs in self.dic.items():
            selected_hand_refs = hand_refs[::-1]
            if len(selected_hand_refs) > self.reference_size:
                selected_hand_refs = selected_hand_refs[::(len(selected_hand_refs) // self.reference_size)]

            # confs = []
            for i, hand_ref in enumerate(selected_hand_refs):
                _, _, confidence = self.discriminator(hand_patch, hand_ref)
                confidence = torch.sigmoid(confidence)[0, 1].item()
                # confs.append(confidence)
                confidence_map[id] = max(confidence, confidence_map[id])

            # confidence_map[id] = mean(confs)

        id_with_max_confidence = None
        max_confidence = 0
        for id, confidence in confidence_map.items():
            if confidence > self.confidence_thres and confidence > max_confidence:
                id_with_max_confidence = id
                max_confidence = confidence

        if id_with_max_confidence is None:
            id_with_max_confidence = self.next_id
            self.next_id += 1

        self.dic[id_with_max_confidence].append(hand_patch)
        if len(self.dic[id_with_max_confidence]) > self.cache_size:
            self.dic[id_with_max_confidence].pop(0)

        return id_with_max_confidence
