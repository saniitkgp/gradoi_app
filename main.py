import gradio as gr
import yaml
import logging
from typing import Dict, Any
from model import ImageSegmentation,Imagesharpeness,ImageGeneration
# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from gradio_image_annotation import image_annotator


example_annotation = {
    "image": "https://gradio-builds.s3.amazonaws.com/demo-files/base.png",
    "boxes": [
        {
            "xmin": 636,
            "ymin": 575,
            "xmax": 801,
            "ymax": 697,
            "label": "Vehicle",
            "color": (255, 0, 0)
        },
        {
            "xmin": 360,
            "ymin": 615,
            "xmax": 386,
            "ymax": 702,
            "label": "Person",
            "color": (0, 255, 0)
        }
    ]
}


class GradioApp:
    """A scalable Gradio application with dynamic UI updates based on task selection."""

    def __init__(self):
        self.config = self.load_config("config.yaml")
        logging.info(f"Loaded config: {self.config}")
        self.task_menu = self.config["task"]["task_menu"]
        self.segmentation = ImageSegmentation()  # Keep one instance
        self.sharpness = Imagesharpeness()
        # self.image_generation = ImageGeneration()
        self.setup_ui()
        
    def get_boxes_json(self,annotations):
        return annotations["boxes"]


    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads the YAML config file."""
        logging.info(f"Loading config from {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
        
    def image_annotation_ui(self):
        annotator = image_annotator(
            example_annotation,
            label_list=["Person", "Vehicle"],
            label_colors=[(0, 255, 0), (255, 0, 0)],
        )
        button_get = gr.Button("Get bounding boxes")
        json_boxes = gr.JSON()
       
        return annotator, button_get, json_boxes
    
    
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

    def Image_Generation_ui(self):
        """Creates UI for Image Generation."""
        gr.Markdown("# Text-to-Image Generation with Stable Diffusion")
        with gr.Column():
            prompt = gr.Textbox(label="Enter your prompt")
            
        with gr.Row():
            width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
            height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
        with gr.Column():
             with gr.Row():
                guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.5, label="Guidance Scale")
                step_scale = gr.Slider(minimum=1, maximum=100, value=10, step=1,label="Number of inference steps")
        
        with gr.Row():
            generate_btn = gr.Button("Generate Image")
        
        output_image = gr.Image(label="Generated Image")
        
        
        return prompt, guidance_scale, width, height,step_scale,generate_btn,output_image
    
    
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
            
            with gr.Column(visible=False) as annotation_ui:
                annotator, button_get, json_boxes = self.image_annotation_ui()
                button_get.click(self.get_boxes_json, annotator, json_boxes)
                
            with gr.Column(visible=False) as image_generation_ui:
                prompt, guidance_scale, width, height,step_scale,generate_btn,output_image = self.Image_Generation_ui()
                generate_btn.click(ImageGeneration().generate_image, inputs=[prompt, guidance_scale, width, height,step_scale], outputs=output_image)
                
            def toggle_ui(task_name):
                return (
                    gr.update(visible=task_name == "Image Segmentation"),
                    gr.update(visible=task_name == "Image Sharpness"),
                    gr.update(visible=task_name == "Image Annotation"),
                    gr.update(visible=task_name == "Image Generation")
                )
            
            task_dropdown.change(fn=toggle_ui, inputs=[task_dropdown], 
                                 outputs=[segmentation_ui,sharpness_ui,annotation_ui,
                                          image_generation_ui])
        
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
