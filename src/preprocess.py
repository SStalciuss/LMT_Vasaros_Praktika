import os
import cv2
from cv2.typing import MatLike
import numpy as np
from matplotlib import pyplot as plt
from typing import List

circles_root_path = "data/CIRCLES/"
path_to_phase = "cine/PNG/"
extension = ".png"

prepared_root_path = "data/prepared/"


def get_content(
    content: MatLike, mask: MatLike, inner: bool, include_border: bool
) -> MatLike:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_mask = np.zeros_like(mask)
    cv2.drawContours(circle_mask, contours, -1, 255, thickness=cv2.FILLED)  # type: ignore

    content = cv2.bitwise_and(
        content, content, mask=circle_mask if inner else cv2.bitwise_not(circle_mask)
    )
    if include_border:
        return content

    return cv2.bitwise_and(content, content, mask=cv2.bitwise_not(mask))


def remove_color(content: MatLike, mask: MatLike) -> MatLike:
    return cv2.bitwise_and(content, content, mask=cv2.bitwise_not(mask))


def process_image(image_path: str, output_path: str) -> MatLike:
    if os.path.exists(output_path):
        return cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(image_path)

    b, g, r = cv2.split(image)

    gray_mask = np.uint8((g == b) & (b == r)) * 255

    green_mask = np.uint8(np.less(b, g) & np.less(r, g)) * 255

    # red and pink masks get convoluted and causes issues, so some outlier have to be deleted by ranges
    lower_red = np.array([0, 0, 110])  # Blue green red
    upper_red = np.array([100, 80, 255])
    red_range = cv2.bitwise_and(
        cv2.bitwise_not(gray_mask), cv2.inRange(image, lower_red, upper_red)
    )
    red_mask_base = np.uint8((g == b) & np.less(g, r)) * 255
    red_mask = cv2.bitwise_or(red_mask_base, red_range)

    lower_pink = np.array([80, 0, 100])
    upper_pink = np.array([255, 60, 255])
    pink_range = cv2.bitwise_and(
        cv2.bitwise_not(gray_mask), cv2.inRange(image, lower_pink, upper_pink)
    )
    pink_base_mask = np.uint8((r == b) & np.greater(r, g)) * 255
    pink_mask = cv2.bitwise_or(pink_base_mask, pink_range)

    inside_green_content = get_content(
        image, green_mask, inner=True, include_border=False
    )
    between_green_and_red_content = get_content(
        inside_green_content, red_mask, inner=False, include_border=True
    )
    final_image = remove_color(between_green_and_red_content, pink_mask)

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(output_path, final_image)

    # plt.imshow(between_green_and_red_content)
    # plt.title('Content')
    # plt.axis('off')
    # plt.show()

    return cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)


def process_image_by_patient_and_phase(patient: str, phase: int) -> MatLike:
    patient_dir = os.path.join(circles_root_path, patient, path_to_phase)
    phase_files = [
        f for f in os.listdir(patient_dir) if f.endswith(f"_{phase}{extension}")
    ]

    if not phase_files:
        raise FileNotFoundError(
            f"No matching file found for patient {patient} and phase {phase}"
        )

    image_path = os.path.join(patient_dir, phase_files[0])
    prepared_image_path = os.path.join(
        prepared_root_path, patient, f"P{phase}{extension}"
    )
    return process_image(image_path, prepared_image_path)


def process_images_by_patient(
    patient: str, start_phase: int, end_phase: int
) -> list[MatLike]:
    """
    Parameters:
    start_phase (int): value between 1 and 25, must be lower than end_phase.
    end_phase (int): value between 2 and 26, must be higher than start_phase.
    """
    images = []
    for i in range(start_phase, end_phase):
        image = process_image_by_patient_and_phase(patient, i)
        images.append(image)

    return images


def process_all_images_by_patient(patient: str) -> list[MatLike]:
    """
    Parameters:
    start_phase (int): value between 1 and 25, must be lower than end_phase.
    end_phase (int): value between 2 and 26, must be higher than start_phase.
    """
    images = []
    try:
        for i in range(1, 100):
            image = process_image_by_patient_and_phase(patient, i)
            images.append(image)
    except FileNotFoundError:
        pass

    return images


def get_patient_list(root_path: str) -> List[str]:
    return [
        f
        for f in os.listdir(root_path)
        if not f.startswith(".") and os.path.isdir(os.path.join(root_path, f))
    ]
