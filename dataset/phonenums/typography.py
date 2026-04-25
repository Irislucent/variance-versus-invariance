import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def random_gradual_blur(image, strips=10):
    # Get the height and width of the image
    h, w, _ = image.shape

    # Prepare a new image canvas
    blurred_image = np.zeros_like(image)

    # Calculate the width of each strip
    strip_width = w // strips

    # Generate random blur levels for each strip, ensuring kernel sizes are odd
    blur_levels = [
        2 * (np.random.randint(1, np.random.randint(2, 5)) * 2) + 1
        for _ in range(strips)
    ]

    for i in range(strips):
        # Define the start and end of the current strip
        start = i * strip_width
        if i == strips - 1:
            end = w  # Ensure the last strip goes to the edge of the image
        else:
            end = start + strip_width

        # Interpolate blur level between this strip and the next
        if i < strips - 1:
            kernel_size = np.linspace(blur_levels[i], blur_levels[i + 1], strip_width)
        else:
            kernel_size = np.full(strip_width, blur_levels[i])

        # Apply variable Gaussian blur to the current strip
        for j in range(strip_width):
            col = start + j
            k_size = int(kernel_size[j])  # Ensure the kernel size is an integer
            if k_size % 2 == 0:
                k_size += 1  # Kernel size must be odd
            blurred_image[:, col] = cv2.GaussianBlur(
                image[:, col : col + 1], (k_size, k_size), 0
            ).reshape(h, 3)

    return blurred_image


class Typography:
    def __init__(
        self,
        pagesize="20x1",
        patchsize="32x48",
        verbose=False,
    ):
        self.page_width, self.page_height = map(int, pagesize.split("x"))
        self.patch_width, self.patch_height = map(int, patchsize.split("x"))
        self.image_width = self.page_width * self.patch_width
        self.image_height = self.page_height * self.patch_height
        self.verbose = verbose

        if self.verbose:
            print(f"Page size: {self.page_width}x{self.page_height}")
            print(f"Patch size: {self.patch_width}x{self.patch_height}")
            print(f"Image size: {self.image_width}x{self.image_height}")

    def char_to_matrix(
        self,
        char,
        font="consolab.ttf",
        fg=(0, 0, 0),
        bg=(255, 255, 255),
        translate=False,
        jitter=False,
    ):
        """
        convert a character to a maybe-binary matrix using the size information and the given font
        the character should be centered and large enough to fill the patch

        ImageDraw.textsize was deprecated, the correct attribute is textlength which gives you the width of the text. for the height use the fontsize * how many rows of text you wrote.
        """
        if self.verbose:
            print(f"Converting '{char}' to matrix using font '{font}'")
        w, h = self.patch_width, self.patch_height

        # create an image
        patch = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(patch)
        # load a font
        fnt = ImageFont.truetype(font, h)  # some fonts are too tall
        # get the size of the text
        text_w = draw.textlength(char, fnt)
        text_h = h
        if self.verbose:
            print(f"Text size: {text_w}x{text_h}")
        # maybe jitter the text color
        if jitter:
            fg = np.array(fg) + np.random.randint(-2, 3, 3)
            fg = tuple(np.clip(fg, 0, 255))
        # draw the text
        if translate:
            draw.text(
                (
                    (w - text_w) // 2 + np.random.random(),
                    (h - text_h) // 2 + np.random.random() - 8,  # a little offset
                ),
                char,
                font=fnt,
                fill=fg,
            )
        else:
            draw.text(((w - text_w) // 2, (h - text_h) // 2), char, font=fnt, fill=fg)
        mtx = np.array(patch)

        return mtx

    def printer(
        self,
        text,
        output_path=None,
        font="consolab.ttf",
        fg=(10, 10, 10),
        bg=(245, 245, 245),
        distortion="color, gaussian, salt, blur, translate",
        grey_scale=False,
    ):
        # create an image
        canvas = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 255
        # if word wrap is not allowed, fill the char list with spaces
        chars = list(text)

        # Prune or pad the text to fit the page
        if len(chars) > self.page_width * self.page_height:
            chars = chars[: self.page_width * self.page_height]
        if len(chars) < self.page_width * self.page_height:
            chars += [" "] * (self.page_width * self.page_height - len(chars))

        final_text = "".join(chars)

        # maybe change the color a little bit
        fg = np.array(fg) + np.random.randint(-2, 3, 3)
        bg = np.array(bg) + np.random.randint(-2, 3, 3)
        fg = tuple(np.clip(fg, 0, 255))
        bg = tuple(np.clip(bg, 0, 255))

        # The original version
        if self.verbose:
            print(f"Characters: {chars}")
        # fill individual patches to form the image
        for i, char in enumerate(chars):
            x = (i % self.page_width) * self.patch_width
            y = (i // self.page_width) * self.patch_height
            if self.verbose:
                print(f"Printing '{char}' at ({x}, {y})")
            mtx = self.char_to_matrix(
                char,
                font,
                fg,
                bg,
                translate="translate" in distortion,
                jitter="color" in distortion,
            )
            canvas[y : y + self.patch_height, x : x + self.patch_width] = mtx

            # overflow
            if (
                y + self.patch_height >= self.image_height
                and x + self.patch_width >= self.image_width
            ):
                break
        # add distortion
        if "blur" in distortion:  # the blur in every image is changing
            canvas = random_gradual_blur(canvas)
        # Gaussian noise
        if "gaussian" in distortion:
            noise = np.random.normal(0, np.random.random(), canvas.shape)
            # noise = np.clip(noise, -5, 5)
            canvas += np.uint8(noise)
            canvas = np.clip(canvas, 0, 255)

        if output_path is not None:
            out = Image.fromarray(canvas, mode="RGB")
            if grey_scale:
                out = out.convert("L")
            out.save(output_path)
        else:
            return canvas, final_text
