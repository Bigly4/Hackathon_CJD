import cv2
import imutils
import numpy as np
import pytesseract
import argparse

# Funktion erstellt durch Adrian Rosebrock @pyimagesearch
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

# Funktion erstellt durch Adrian Rosebrock @pyimagesearch
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def getWarped(imgPath):
    image = cv2.imread(imgPath)
    ratio = image.shape[0] / image.shape[1]
    orig = image.copy()

    # Blur + Canny-Kantenerkennung
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(image, 75, 200)

    # Alle Konturen finden
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Nur die größten 5 extrahieren
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    success = False

    for c in cnts:
        # Kontur vereinfachen
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Viereckiige Kontur finden
        if len(approx) == 4:
            screenCnt = approx
            success = True
            break

    if not success:
        print("Ging nich... :(")
        return orig

    # Funktion verwenden, um die Perspektive zu ändern
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # Wenn das erstellte Bild hell ist, ist es mit höchster Wahrscheinlichkeit Papier
    if getDominantBrightness(warped) > 100:
        print("Eine Seite wurde gefunden")
        return warped
    else:
        # Ansonsten wahrscheinlich ein Bild
        print("Kein ganzes, weißes Blatt gefunden")
        return orig


def getDominantBrightness(img):
    # Für numpy convertieren
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)
    # In uint8 array konvertieren -> Farbe
    dominant = np.uint8([[centers[0]]])

    dominant = cv2.cvtColor(dominant, cv2.COLOR_BGR2HSV)

    return dominant[0][0][2]


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
args = vars(ap.parse_args())

img = getWarped(args["image"])
cv2.imwrite("output_warped.png", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.medianBlur(img, 3)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
config = "-l deu --oem 1 --psm 7"
text = pytesseract.image_to_string(img, config=config)

outputFile = open("recognisedText", "w")
words = text.split()
for word in words:
    if len(word) > 4:
        outputFile.write(word)

outputFile.close()
cv2.imwrite("output_scan.png", img)
cv2.waitKey(0)