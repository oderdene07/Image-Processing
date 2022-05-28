import cv2
import numpy as np

def detect_coins():
    coins = cv2.imread('./coins.jpg', 1)

    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 45, 255, cv2.THRESH_TOZERO)
    cv2.imwrite("./coins_threshold.jpg", threshold)
    img = cv2.medianBlur(threshold, 9)
    
    circles = cv2.HoughCircles(
        img,  # source image
        cv2.HOUGH_GRADIENT,  # type of detection
        1,
        50,
        param1=100,
        param2=50,
        minRadius=10,  # minimal radius
        maxRadius=380,  # max radius
    )

    coins_copy = coins.copy()

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(
            coins_copy,
            (int(x_coor), int(y_coor)),
            int(detected_radius),
            (0, 255, 0),
            4,
        )

    cv2.imwrite("./coins_test.jpg", coins_detected)

    return circles

def calculate_amount():
    amount = {
        "1": {
            "value": 1,
            "radius": 20,
            "ratio": 1,
            "count": 0,
        },
        "2": {
            "value": 2,
            "radius": 21.5,
            "ratio": 1.075,
            "count": 0,
        },
        "50": {
            "value": 50,
            "radius": 24.5,
            "ratio": 1.225,
            "count": 0,
        },
        "10": {
            "value": 10,
            "radius": 26,
            "ratio": 1.3,
            "count": 0,
        },
        "100": {
            "value": 100,
            "radius": 27.3,
            "ratio": 1.365,
            "count": 0,
        },
        "500": {
            "value": 500,
            "radius": 30.5,
            "ratio": 1.525,
            "count": 0,
        },
    }

    circles = detect_coins()
    radius = []
    coordinates = []

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)
        coordinates.append([x_coor, y_coor])

    smallest = min(radius)
    tolerance = 0.0375
    total_amount = 0

    coins_circled = cv2.imread('./coins_test.jpg', 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for coin in circles[0]:
        ratio_to_check = coin[2] / smallest
        coor_x = coin[0]
        coor_y = coin[1]
        for i in amount:
            value = amount[i]['value']
            if abs(ratio_to_check - amount[i]['ratio']) <= tolerance:
                amount[i]['count'] += 1
                total_amount += amount[i]['value']
                cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                            (0, 0, 0), 4)

    print(f"Niit: {total_amount}")
    for i in amount:
        pieces = amount[i]['count']
        print(f"{i} = {pieces} shirheg")


    cv2.imwrite("./amount.jpg", coins_circled)



if __name__ == "__main__":
    calculate_amount()