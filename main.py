import asyncio
import json
import os
import pickle

###################
import threading
import time
from typing import Literal

import cv2
import imutils
import numpy as np
import websockets
from numpy.linalg import norm

ip_address = "192.168.4.1"  # "192.168.1.100"  #
ws_url = f"ws://{ip_address}/d/ws/issue"

OFFLINE_MODE = False
CAMERA_INDEX = 0

current_frame = None

# br_history = []
start_averaging = False
START_AVERAGING_KEY = "y"
start_event_loop = False
START_EVENT_LOOP_KEY = "u"
start_tuning = False
START_TUNING_KEY = "i"
LOAD_CONTOURS_KEY = "g"
# Dimensions to corp phone image to get keypad part

left = 140
width = 290
top = 160
height = 320

# Allow accumulation of edged images to ease changes
start_accumulate = False

# Stick the contour rectangles
take_the_last_10_cont_as_numbers = False

# Global contours object
cnts = None

cnt_save_path = "/home/mostafa/Desktop/cnt_save.var"
# Countour mapping dictionary path
cm = "/home/mostafa/Desktop/countour_mapping.dict"
pickle_path = "/home/mostafa/Desktop/list_of_possibilities.pickle"
# Countour brightness stats dictionary path
c_stats = "/home/mostafa/Desktop/countour_stats.dict"
stats_average__stats_average_min_max__path = (
    "/home/mostafa/Desktop/stats_average_data.pickle"
)


IMAGES_PATH_AFTER_TEST = "/home/mostafa/Desktop/after_test/"


# Countour brightness stats dictionary
c_stats_dict = {
    "cnt0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt4": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt5": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt6": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt7": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt8": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt10": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cnt11": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
stats_average_min_max = {}
key_state = {}
# Mapping for numbers to esp
mapping_dict = {
    "1": "30",
    "2": "31",
    "3": "32",
    "4": "33",
    "5": "34",
    "6": "35",
    "7": "36",
    "8": "37",
    "9": "38",
    "0": "39",
}

#### --------------------------------------
#### --------------------------------------
#### --------------------------------------


def calc_stats_average():
    global stats_average_min_max
    global key_state
    with open(cm, "r") as file:
        mapper = eval(file.read())
    # print(mapper)
    stats_average = {}
    # print("c_stats_dict:", c_stats_dict)
    for key, val_list in c_stats_dict.items():
        avg_brightness = stats_average[mapper[key]] = int(sum(val_list) / len(val_list))
        if mapper[key] in stats_average_min_max:
            mn = min(avg_brightness, stats_average_min_max[mapper[key]][0])
            mx = max(avg_brightness, stats_average_min_max[mapper[key]][2])
        else:
            mx = 0
            mn = 999
        thresh = (mx + mn) / 2
        stats_average_min_max[mapper[key]] = (
            mn,
            thresh,
            mx,
        )
        key_state[mapper[key]] = avg_brightness > thresh
    # br_history.append(stats_average["1"])
    return stats_average, stats_average_min_max, key_state


def which_button_is_on():
    res = []
    for key, value in key_state.items():
        if value:
            res.append(key)
    return res


def get_current_frame():
    global current_frame
    if current_frame is not None:
        return current_frame
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera")
        raise ValueError("Frame is not available")

    ret, frame = cap.read()
    return frame


def handle_input():
    global left
    global width
    global top
    global height
    global start_accumulate
    global take_the_last_10_cont_as_numbers
    global cnts
    global start_averaging
    global start_event_loop
    global start_tuning
    global stats_average_min_max

    # Check for key press, if 'q' is pressed, exit the loop
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        return False
    # top
    elif k == ord("q"):
        top = min(480, top + 10)
        print(f"l pressed top = {top}")
    elif k == ord("e"):
        top = max(0, top - 10)
        print(f"top = {top}")
    # left
    elif k == ord("z"):
        left = min(640, left + 10)
        print(f"l pressed left = {left}")
    elif k == ord("c"):
        left = max(0, left - 10)
        print(f"left = {left}")
    # height
    elif k == ord("w"):
        height = min(480, height + 10)
        print(f"l pressed height = {height}")
    elif k == ord("s"):
        height = max(0, height - 10)
        print(f"height = {height}")
    # width
    elif k == ord("a"):
        width = min(480, width + 10)
        print(f"l pressed width = {width}")
    elif k == ord("d"):
        width = max(0, width - 10)
        print(f"width = {width}")
    elif k == ord("r"):
        start_accumulate = True
    # save is disabled
    # elif k == ord("f"):
    #     print("Contourse will stick and will be pickled")
    #     with open(cnt_save_path, "wb") as file:
    #         pickle.dump(cnts, file)
    #     take_the_last_10_cont_as_numbers = True
    elif k == ord(LOAD_CONTOURS_KEY):
        with open(cnt_save_path, "rb") as file:
            cnts = pickle.load(file)
        print("Contourse will stick on the loaded file path")
        take_the_last_10_cont_as_numbers = True
    elif k == ord(START_AVERAGING_KEY):
        start_averaging = True
    elif k == ord(START_EVENT_LOOP_KEY):
        start_event_loop = True
    elif k == ord(START_TUNING_KEY):
        start_tuning = True
    elif k == ord("t"):
        print("Stas average", calc_stats_average())
        # print("History = ", br_history)
    elif k == ord("h"):
        with open(stats_average__stats_average_min_max__path, "wb") as file:
            pickle.dump(stats_average_min_max, file)
        print("stats_average_min_max saved, : ", stats_average_min_max)
    elif k == ord("j"):
        with open(stats_average__stats_average_min_max__path, "rb") as file:
            stats_average_min_max = pickle.load(file)
        print("stats_average_min_max loaded: ", stats_average_min_max)
    elif k == ord(" "):
        print(
            f"""
                    left = {left}
                    width = {width}
                    top = {top}
                    height = {height}
                """
        )
    return True


def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        res = np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        res = np.average(img)
    if np.isnan(res):
        return 0
    return res


def monitor(*args):
    print("monitor thread started ...")
    # Open the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera")
        exit()

    global left
    global width
    global top
    global height
    global cnts

    last_images = []
    # Loop to continuously capture frames
    while True:
        global current_frame
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is captured successfully
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break

        original_img = frame
        img = original_img.copy()
        hh, ww = img.shape[:2]
        # print(hh, ww)
        max_hh_ww = max(hh, ww)

        # illumination normalize
        # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # separate channels
        # y, cr, cb = cv2.split(ycrcb)

        crop_img = img[left : left + width, top : top + height]
        # pre-process the image by resizing it, converting it to
        # graycale, blurring it, and computing an edge map
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/
        rotated = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # rotated = crop_img

        cv2.imshow("0-crop_img", crop_img)
        current_frame = rotated
        cv2.imshow("1-rotated", rotated)

        blurred = cv2.GaussianBlur(rotated, (5, 5), 0)
        # blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
        # blurred = cv2.GaussianBlur(rotated, (1, 1), 0)
        cv2.imshow("2- blurred", blurred)

        # https://stackoverflow.com/questions/19103933/depth-error-in-2d-image-with-opencv-python
        edged = cv2.Canny(blurred, 50, 200, 255)
        ###
        if start_accumulate:
            gray = edged.copy().astype("float")

            if len(last_images) < 10:
                last_images.append(gray)

            if len(last_images) == 10:
                cv2.accumulate(gray, gray)
                cv2.accumulate(last_images[0] * 0.001, gray)
                cv2.accumulate(last_images[1] * 0.002, gray)
                cv2.accumulate(last_images[2] * 0.005, gray)
                cv2.accumulate(last_images[3] * 0.01, gray)
                cv2.accumulate(last_images[4] * 0.02, gray)
                cv2.accumulate(last_images[5] * 0.05, gray)
                cv2.accumulate(last_images[6] * 0.1, gray)
                cv2.accumulate(last_images[7] * 0.2, gray)
                cv2.accumulate(last_images[8] * 0.5, gray)
                cv2.accumulate(last_images[9] * 1.0, gray)
                gray = gray / 2.888

                for i in range(1, len(last_images)):
                    last_images[i - 1] = last_images[i]
                last_images[-1] = gray

            edged = np.uint8(gray)
        ###
        cv2.imshow("3- edged", edged)

        kernel = np.ones((5, 5), np.uint8)
        # https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
        edged_dilate = cv2.dilate(edged, kernel, iterations=2)
        cv2.imshow("4- edged_dilate", edged_dilate)

        output = rotated.copy()
        output_thresh = rotated.copy()
        output_thresh[edged != 0] = [255, 255, 255]
        # find contours in the thresholded image, then initialize the
        # digit contours lists

        if not take_the_last_10_cont_as_numbers:
            # https://stackoverflow.com/questions/57258173/opencv-join-contours-when-rectangle-overlaps-another-rect
            cnts = cv2.findContours(
                edged_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )
            cnts = imutils.grab_contours(cnts)

        digitCnts = []
        # loop over the digit area candidates
        for id, c in enumerate(cnts):
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            ###cnt_image = output[x - 10 : x + w + 10, y - 5 : y + h + 5]
            if id < 10:
                real_cnt_image = blurred[y - 5 : y + h + 5, x - 10 : x + w + 10]
            else:
                real_cnt_image = blurred[y : y + h, x : x + w]

            br = brightness(real_cnt_image)
            if f"cnt{id}" in c_stats_dict:
                c_stats_dict[f"cnt{id}"] = c_stats_dict[f"cnt{id}"][1:] + [br]
            if take_the_last_10_cont_as_numbers:
                hhh, www = real_cnt_image.shape[:2]
                if hhh > 0 and www > 0:
                    cv2.imshow(f"cnt{id}", real_cnt_image)
            # --
            # hh, ww = cnt_image.shape[:2]
            # if id < 10 and hh > 0 and ww > 0:
            #     cv2.imshow(f"cnt{id}", cnt_image)
            if id < 10:
                cv2.rectangle(
                    output,
                    (x - 10, y - 5),
                    (x + w + 10, y + h + 5),
                    (255, 255, 20 * id),
                    1,
                )
            else:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.rectangle(output_thresh, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (x, y)

            # fontScale
            fontScale = 0.2

            # Blue color in BGR
            color = (0, 0, 0)

            # Line thickness of 2 px
            thickness = 1

            # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
            output = cv2.putText(
                output,
                str(id) + ":" + str(int(br or 0)),
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        with open(c_stats, "w") as file:
            file.write(json.dumps(c_stats_dict))
            # if the contour is sufficiently large, it must be a digit
            # if w >= 15 and (h >= 30 and h <= 40):
            #     digitCnts.append(c)
        cv2.imshow("output", output)
        cv2.imshow("output_thresh", output_thresh)

        if start_averaging:
            calc_stats_average()

        if not handle_input():
            break
        continue

    # Release the capture
    cap.release()


#### --------------------------------------
#### --------------------------------------
#### --------------------------------------


async def __send(websocket, id, ck):
    if id == -1:
        data_to_send = "wx"
    else:
        data_to_send = f"CK{id}	{ck}"  # Your data to send

    await websocket.send(data_to_send)

    # try:
    #     message = await websocket.recv()
    #     print(f"Received message: {message}")
    # except websockets.exceptions.ConnectionClosed:
    #     print("Connection to the WebSocket server closed.")

    print(f"->{data_to_send}", end="")

    time.sleep(0.1)


async def press_backspace(websocket):
    print(".bs>", end="")
    await __send(websocket, 0, 42)
    await __send(websocket, 0, 0)


async def press_number(
    websocket, number_as_str: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
):
    if number_as_str not in mapping_dict:
        raise ValueError(f"Number as str is bad {number_as_str}")

    ck = mapping_dict[number_as_str]
    print(f".num[{number_as_str}]>", end="")
    await __send(websocket, 0, ck)
    await __send(websocket, 0, 0)


async def press_shifttab(websocket):
    print(f".st>", end="")
    await __send(websocket, 2, 225)
    await __send(websocket, 2, 43)
    await __send(websocket, 0, 0)
    await __send(websocket, 0, 0)


async def goto_editbar(websocket):
    """
    This function is called after the goto_1
    """
    print("\nGoing to editbar")
    if OFFLINE_MODE:
        return False
    # TODO: goto_editbar Implementation
    counter = 0
    while not key_state["edit"] and counter < 5:
        await press_shifttab(websocket)
        for i in range(4):
            time.sleep(1 / 4)
            await press_backspace(websocket)
        if not key_state["edit"]:
            print(f"\nFailed to go to editbar, will try again, counter = {counter}/4")
        counter += 1

    if counter == 5:
        raise Exception("Failed to reach edit bar")
    print(f"\nDone, Should be now on edit bar: {which_button_is_on()}")


async def goto_number_1(websocket):
    print("\ngoto_number_1: Going to number 1")
    if OFFLINE_MODE:
        return False
    counter = 0
    while not key_state["1"] and counter < 40:
        await press_shifttab(websocket)
        for i in range(4):
            time.sleep(1 / 4)
            await press_backspace(websocket)
        print("\ngoto_number_1: Which_button_is_on : ", which_button_is_on())
        counter += 1
    if counter == 40:
        raise Exception(
            "goto_number_1: Failed to reach number 1, counter exceeded threshold 40"
        )
    print("goto_number_1: Done, Should be now on number 1")


async def enter_number(websocket, num: str, start_from: int):
    assert len(num) == 4
    print(f"\nenter_number: Trying to send number -----> {num}")
    if OFFLINE_MODE:
        return False
    for index, c in enumerate(num):
        assert c in "0123456789"
        await press_number(websocket, c)
        for i in range(4):
            time.sleep(1 / 4)
            await __send(websocket, -1, "")
        print("enter_number: Which_button_is_on : ", which_button_is_on())
        if "edit" in key_state:
            print(f'enter_number: key_state["edit"] = {key_state["edit"]}')
            if index < 3 and key_state["edit"] is False:
                print(
                    "WARNING ------- SOMETHING BAD HAPPENED number index < 3 and key_state['edit'] is False !"
                )
                cv2.imwrite(
                    f"/home/mostafa/Desktop/errors/test_error_{str(start_from).zfill(6)}_{num}_{index}.png",
                    get_current_frame(),
                )
        cv2.imwrite(
            f"/home/mostafa/Desktop/after_press_{index}/test_{str(start_from).zfill(6)}_{num}.png",
            get_current_frame(),
        )


async def keep_looping():
    print("function keep_looping started ...")

    print(f"Trying to connect to websocket: {ws_url}")

    async with websockets.connect(ws_url) as websocket:
        print("await press_shifttab")
        while not start_tuning:
            print(
                f"Start tuning is false, load contours '{LOAD_CONTOURS_KEY}' then\
START_AVERAGING_KEY: {START_AVERAGING_KEY}, then {START_TUNING_KEY}"
            )
            await press_shifttab(websocket)
            time.sleep(0.2)


async def tune_thresholds():
    print("function tune_thresholds started ...")
    print(f"Trying to connect to websocket: {ws_url}")

    async with websockets.connect(ws_url) as websocket:
        print("await press_shifttab")
        for i in range(30):
            print(f"press_shifttab {i}/30")
            await press_shifttab(websocket)
            time.sleep(0.3)


start_from = len(os.listdir(IMAGES_PATH_AFTER_TEST))


async def try_all_pos(possibilities):
    global start_from
    print("function try_all_pos started ...")

    print(f"Trying to connect to websocket: {ws_url}")

    async with websockets.connect(ws_url) as websocket:

        print(f"Looping throgh possibilities starting from {start_from}")

        for pos in possibilities[start_from:]:

            start_time = time.time()

            await goto_number_1(websocket)

            end_time = time.time()
            elasped_time = end_time - start_time
            if elasped_time < 30:
                print(f"\nElapsed time since start going to number 1 is {elasped_time}")
                print(f"Will wait {31 - elasped_time} seconds")
                sleep_time = 31 - elasped_time
                for i in range(int(sleep_time * 2)):
                    await press_backspace(websocket)
                    time.sleep(0.5)

            cv2.imwrite(
                f"/home/mostafa/Desktop/on_number_1/test_{str(start_from).zfill(6)}_{pos}.png",
                get_current_frame(),
            )

            await goto_editbar(websocket)

            cv2.imwrite(
                f"/home/mostafa/Desktop/on_edit_bar/test_{str(start_from).zfill(6)}_{pos}.png",
                get_current_frame(),
            )

            for i in range(5):
                time.sleep(0.2)
                await press_backspace(websocket)

            await enter_number(websocket, pos, start_from)

            time.sleep(0.5)
            await press_backspace(websocket)

            if key_state["edit"]:
                raise Exception("After entering number, seems to be still on edit bar!")

            # Capture the current state
            cv2.imwrite(
                f"/home/mostafa/Desktop/after_test/test_{str(start_from).zfill(6)}_{pos}.png",
                get_current_frame(),
            )

            # Test done
            start_from += 1


#### --------------------------------------
#### --------------------------------------
#### --------------------------------------


def make_event_loop_ready():
    # https://stackoverflow.com/questions/46727787/runtimeerror-there-is-no-current-event-loop-in-thread-in-async-apscheduler
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith("There is no current event loop in thread"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise


def main(*args):
    global start_event_loop

    print("main thread started ...")
    list_of_possibilities = get_possibilities()

    make_event_loop_ready()

    print("Will call tune_thresholds")
    asyncio.get_event_loop().run_until_complete(keep_looping())

    while not start_tuning:
        time.sleep(1)

    print("tune_thresholds will started")
    asyncio.get_event_loop().run_until_complete(tune_thresholds())

    print("Tune done")

    while not start_event_loop:
        print(
            f"Waiting start_event_loop to become True: start_event_loop, press {START_EVENT_LOOP_KEY}"
        )
        time.sleep(1)

    counter = 0
    while True:

        for _ in range(9):
            print("\a")
            time.sleep(0.1)

        try:
            print("Start event loop run until complete")
            asyncio.get_event_loop().run_until_complete(
                try_all_pos(list_of_possibilities)
            )
            print("Normal exit, bye bye")
            break
        except Exception as e:
            print("Exception : ", e)
            print(f"Connection will restart (counter = {counter})")
            for _ in range(9):
                print("\a")
                time.sleep(0.1)
        counter += 1


def get_possibilities():
    print(f"pickle_path = {pickle_path}")
    list_of_possibilities = []
    with open(pickle_path, "rb") as f:
        list_of_possibilities = pickle.load(f)
    print(
        len(list_of_possibilities),
        "Possibilities retrieved from pickle",
    )
    return list_of_possibilities


if __name__ == "__main__":
    # https://www.geeksforgeeks.org/multithreading-python-set-1/
    t1 = threading.Thread(target=monitor, args=(10,))
    t2 = threading.Thread(target=main, args=(10,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")
