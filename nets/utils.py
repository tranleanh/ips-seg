import cv2


def get_bboxes(mask, min_area=50):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append([x, y, x + w, y + h])
    return bboxes