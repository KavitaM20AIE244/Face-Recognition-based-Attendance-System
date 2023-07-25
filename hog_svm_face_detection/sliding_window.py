def sliding_window(image, window_size, step_size):
    """
    This function generates patches of an input image using a sliding window approach.

    Args:
        image (numpy.ndarray): The input image.
        window_size (tuple): The size of the sliding window in the format (width, height).
        step_size (tuple): The step size of the sliding window in the format (x_step_size, y_step_size).

    Yields:
        tuple: A tuple (x, y, im_window), where:
            - x is the x-coordinate of the top-left corner of the window.
            - y is the y-coordinate of the top-left corner of the window.
            - im_window is the image patch corresponding to the sliding window.

    The function generates sliding windows of size `window_size` on the input `image`, starting from the top-left corner of the image and incrementing the window position by `step_size` in both the x and y directions.
    The function yields a tuple (x, y, im_window) for each sliding window, where:
        - x is the x-coordinate of the top-left corner of the window.
        - y is the y-coordinate of the top-left corner of the window.
        - im_window is the image patch corresponding to the sliding window.
    """
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
