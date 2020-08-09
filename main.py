import cv2 as cv
import numpy as np
import imutils
from centroidtracker import CentroidTracker
from collections import defaultdict

cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.2
nmsThreshold = 0.2

# Coco Names
classesFile = "model files/yolov3-320/coco.names"
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# Model Files
modelConfiguration = "model files/yolov3-320/yolov3_320.cfg"
modelWeights = "model files/yolov3-320/yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Tracker
tracker = CentroidTracker(maxDisappeared=15, maxDistance=90)

# Global variables
show_nerds_info = False
people_inside = 0

centroid_dict = defaultdict(dict)
object_id_list = []

global left_side_counter, right_side_counter
left_side_counter = 0
right_side_counter = 0



# NOn Max Suppression
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    rects = []
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        if classNames[classIds[i]] == "person":
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # cX = x+(w//2)
            # cY = y +(h//2)
            #
            # print(cX, cY)
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            rects.append(box)
            # cv.circle(img, (cX, cY), 4,(0,255,0), -1)

            if show_nerds_info:
                cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                           (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    old_objects = tracker.objects.keys()

    try:
        for i in centroid_dict.keys():
            if  type(i)== int and not i in old_objects :
                del centroid_dict[i]
    except:
        pass
    objects = tracker.update(rects)
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cX = (x1 + x2) // 2
        cY = (y1 + y2) // 2
        try:
            centroid_dict[objectId]['pts'].append((cX, cY))
        except:
            centroid_dict[objectId]['pts'] = list()
            centroid_dict[objectId]['marked'] = 0    # 0 => Not Marked ; 1 => marked
            centroid_dict[objectId]['dx'] = 0
            centroid_dict[objectId]['dy'] = 0

        if centroid_dict[objectId]['pts'].__len__() > 10:
            centroid_dict[objectId]['pts'] = centroid_dict[objectId]['pts'][1:]


        if objectId not in object_id_list:
            # centroid_dict[objectId]['marked'] = 0  # 0 for n
            object_id_list.append(objectId)
            start_pt = (cX, cY)
            end_pt = start_pt
            cv.line(img, start_pt, end_pt, (0, 255, 0), 2)

        else:
            l = len(centroid_dict[objectId]['pts'])
            for pt in range(len(centroid_dict[objectId]['pts'])):
                if not pt + 1 == l:
                    start_pt = (centroid_dict[objectId]['pts'][pt][0], centroid_dict[objectId]['pts'][pt][1])
                    end_pt = (centroid_dict[objectId]['pts'][pt + 1][0], centroid_dict[objectId]['pts'][pt + 1][1])


                    cv.line(img, start_pt, end_pt, (0, 255, 0), 2)


        # window size of action for counting

        global left_side_counter, right_side_counter
        try:

            centroid_dict[objectId]['dx'] = centroid_dict[objectId]['pts'][0][0] - centroid_dict[objectId]['pts'][-1][0]
            centroid_dict[objectId]['dy'] = centroid_dict[objectId]['pts'][0][1] - centroid_dict[objectId]['pts'][-1][1]
            if  centroid_dict[objectId]['pts'][-1][0] > 600:
                if centroid_dict[objectId]['marked'] == 0 and centroid_dict[objectId]['dx'] <5  :
                    left_side_counter += 1
                    centroid_dict[objectId]['marked'] = 1
            if centroid_dict[objectId]['pts'][-1][0] < 150 :
                if centroid_dict[objectId]['marked'] == 0 and centroid_dict[objectId]['dx'] > 5:
                    right_side_counter += 1
                    centroid_dict[objectId]['marked'] = 1


            if   centroid_dict[objectId]['marked'] == 0 and centroid_dict[objectId]['pts'][-1][-1] < 700 :
                if centroid_dict[objectId]['pts'][-1][0] > 600 and centroid_dict[objectId]['dx'] >5  :
                    left_side_counter -= 1
                    centroid_dict[objectId]['marked'] = 1
            if  centroid_dict[objectId]['marked'] == 0 and centroid_dict[objectId]['pts'][-1][-1] > 150 :
                if centroid_dict[objectId]['pts'][-1][0] < 100 and centroid_dict[objectId]['dx'] < 5:
                    right_side_counter -= 1
                    centroid_dict[objectId]['marked'] = 1


        except:
            pass
        if show_nerds_info:
            cv.circle(img, (cX, cY), 4, (0, 255, 0), -1)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = 'ID: {}'.format(objectId)
            cv.putText(img, text, (x1, y1 - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv.putText(img, str('Left Counter : '+ str(left_side_counter)+ ' Right Counter : ' + str(right_side_counter) ),
               (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # cv.line(img, (100,0), (100,700), (0,255,255), 2)


frame_count = 0
while True:
    success, img = cap.read()
    img = imutils.resize(img, width=800)
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    # Skipping the frames
    # frame_count += 5
    # cap.set(1, frame_count)

    cv.imshow('Image', img)

    k = cv.waitKey(1)

    if k == 27:
        break

    if k == ord('i'):
        show_nerds_info = not show_nerds_info

cv.destroyAllWindows()
