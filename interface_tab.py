import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

class ImageSegmentation:
    def __init__(self):
        self.image = None
        self.bounding_boxes = []
        # self.task_interface = task_interface

    def image_segmentation_ui(self):
        """Creates the UI for image segmentation."""
        
        # if self.task_interface is None:
        #     raise ValueError("task_interface is not initialized. Ensure get_task_interface_page() is called first.")

        # with self.task_interface:  # Now it's guaranteed to be initialized
        with gr.Row():
            input_image = gr.Image(label="Input Image", type="numpy")
        with gr.Row():
            clear_button = gr.Button("Clear")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Row():
            output_image = gr.Image(label="Output Image")
            
        submit_button.click(fn=self.process_image, inputs=[input_image], outputs=[output_image])
        clear_button.click(fn=self.reset, inputs=[], outputs=[input_image, output_image])
        
        return None  # No need to return anything
    def process_image(self, image):
        # """Dummy segmentation logic: Convert image to grayscale."""
        # if image is None:
        #     return None
        # pil_image = Image.fromarray(image).convert("L")  # Convert to grayscale
        # return np.array(pil_image)
        return image

    def reset(self):
        """Clears both input and output images."""
        return None, None


class InterfaceTask:
    def __init__(self):
        # self.task_interface=task_interface
        pass

    def image_sharpness_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Image Sharpness")
        return block

    def image_segmentation_ui(self):
        # img_segmentation = ImageSegmentation(self.task_interface)
        img_segmentation = ImageSegmentation()
        return img_segmentation.image_segmentation_ui()

    def image_mask_generation_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Image Mask Generation")
        return block

    def face_detection_recognition_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Face Detection & Recognition")
        return block

    def image_editing_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Image Editing")
        return block

    def lyric_generation_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Lyric Generation")
        return block

    def music_generation_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Music Generation")
        return block

    def unknown_task_ui(self):
        with gr.Blocks() as block:
            gr.Markdown("## Unknown Task Selected")
        return block

    def get_ui_for_task(self, task_name):
        task_ui_map = {
            "Image Sharpness": self.image_sharpness_ui,
            "Image Segmentation": self.image_segmentation_ui,
            "Image Mask Generation": self.image_mask_generation_ui,
            "Face Detection & Recognition": self.face_detection_recognition_ui,
            "Image Editing": self.image_editing_ui,
            "Lyric Generation": self.lyric_generation_ui,
            "Music Generation": self.music_generation_ui,
        }
        return task_ui_map.get(task_name, self.unknown_task_ui)()
