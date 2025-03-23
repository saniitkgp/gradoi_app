import gradio as gr
import yaml
import logging
from typing import Dict, Any
from model import ImageSegmentation,Imagesharpeness
# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class ImageSegmentation:
    """Handles image segmentation logic."""
    
    def process_image(self, image):
        """Dummy segmentation logic: Returns the input image."""
        return image

    def reset(self):
        """Clears both input and output images."""
        return None, None

class GradioApp:
    """A scalable Gradio application with dynamic UI updates based on task selection."""

    def __init__(self):
        self.config = self.load_config("config.yaml")
        logging.info(f"Loaded config: {self.config}")
        self.task_menu = self.config["task"]["task_menu"]
        self.segmentation = ImageSegmentation()  # Keep one instance
        self.sharpness = Imagesharpeness()
        self.setup_ui()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads the YAML config file."""
        logging.info(f"Loading config from {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    
    def image_segmentation_ui(self):
        """Creates UI for Image Segmentation."""
        with gr.Column():
            input_image = input_image = gr.Image(label="Input Image", type="numpy", image_mode="RGB", interactive=True)
        with gr.Row():
            reset_button = gr.Button("Reset")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Output Image", type="numpy", image_mode="RGB", interactive=False, show_download_button=True)
        with gr.Row():
            clear_button = gr.Button("Clear")
        return input_image, output_image, clear_button, submit_button,reset_button


    def image_sharpness_ui(self):
        """Creates UI for Image Sharpness."""
        with gr.Column():
            input_image = input_image = gr.Image(label="Input Image", type="numpy", image_mode="RGB", interactive=True)
        with gr.Row():
            reset_button = gr.Button("Reset")
            submit_button = gr.Button("Sharp Image", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Output Image", type="numpy", image_mode="RGB", interactive=False, show_download_button=True)
        with gr.Row():
            clear_button = gr.Button("Clear")
        return input_image, output_image, clear_button, submit_button,reset_button

    def Face_Detection_Recognition_ui():
        """Creates UI for Face Detection & Recognition."""
        with gr.Column():
            input_image = input_image = gr.Image(label="Input Image", type="numpy", image_mode="RGB", interactive=True)
        with gr.Row():
            reset_button = gr.Button("Reset")
            submit_button = gr.Button("Detect Faces", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Output Image", type="numpy", image_mode="RGB", interactive=False, show_download_button=True)
        with gr.Row():
            clear_button = gr.Button("Clear")
        return input_image, output_image, clear_button, submit_button,reset_button

    def get_task_interface_page(self):
        """Generates the task interface page."""
        with gr.Blocks() as task_block:
            gr.Markdown("## Select a Task")
            task_dropdown = gr.Dropdown(choices=self.task_menu, label="Choose a Task", interactive=True)
            
            with gr.Column(visible=False) as segmentation_ui:
                seg_input_image,seg_output_image, seg_clear_button, seg_submit_button,seg_reset_button = self.image_segmentation_ui()
                seg_submit_button.click(fn=self.segmentation.process_image, inputs=[seg_input_image], outputs=[seg_output_image])
                seg_clear_button.click(fn=self.segmentation.reset, inputs=[], outputs=[seg_input_image, seg_output_image])
                seg_reset_button.click(fn=self.segmentation.reset, inputs=[], outputs=[seg_input_image, seg_output_image])
                
            with gr.Column(visible=False) as sharpness_ui:
                seg_input_image,seg_output_image, seg_clear_button, seg_submit_button,seg_reset_button = self.image_sharpness_ui()
                seg_submit_button.click(fn=self.sharpness.process_image, inputs=[seg_input_image], outputs=[seg_output_image])
                seg_clear_button.click(fn=self.sharpness.reset, inputs=[], outputs=[seg_input_image, seg_output_image])
                seg_reset_button.click(fn=self.sharpness.reset, inputs=[], outputs=[seg_input_image, seg_output_image])
                
            def toggle_ui(task_name):
                return (
                    gr.update(visible=task_name == "Image Segmentation"),
                    gr.update(visible=task_name == "Image Sharpness")
                )
            
            task_dropdown.change(fn=toggle_ui, inputs=[task_dropdown], outputs=[segmentation_ui, sharpness_ui])
        
        return task_block

    def setup_ui(self):
        """Initializes the Gradio UI."""
        self.menu_page = gr.Blocks()
        with self.menu_page:
            gr.Markdown("## Welcome to the Menu")
        
        self.task_page = self.get_task_interface_page()
        
        self.model_page = gr.Blocks()
        with self.model_page:
            gr.Markdown("## Model Configuration")
    
    def launch(self):
        """Launches the Gradio app."""
        with gr.Blocks() as demo:
            with gr.Tabs():
                with gr.Tab("Menu"):
                    self.menu_page.render()
                with gr.Tab("Task"):
                    self.task_page.render()
                with gr.Tab("Model"):
                    self.model_page.render()
        demo.launch()

# Run the app
if __name__ == "__main__":
    app = GradioApp()
    app.launch()
