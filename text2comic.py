import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import re
import torch
from langchain_community.llms import HuggingFaceEndpoint
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Initialize HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",
                          huggingfacehub_api_token='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',
                          temperature=0.3)

# Load Stable Diffusion Pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator("cuda").manual_seed(1)
pipe.enable_vae_tiling()
# Define font path
font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')

prompt_comic = '''User: Explain the poem Baa Baa black sheep in 4 image prompts and for each prompt give its explanation.
Answer:
Image 1: The Black Sheep
Drawing Steps: Draw a black sheep standing on a green field. Use simple shapes for the sheep's body and head, and add a few woolly tufts for the wool.
Explanation: This image represents the black sheep that the poem is about.
Image 2: The Three Bags of Wool
Drawing Steps: Draw three bags of wool, each labeled with the names of the recipients: the master, the dame, and the little boy. Place them next to the sheep.
Explanation: This image shows the three bags of wool that the black sheep has, symbolizing the division of the sheep's wool among the master, the dame, and the little boy.
Image 3: The Master and the Dame
Drawing Steps: Draw two people standing next to the bags of wool. One is dressed in a fancy outfit, representing the master, and the other is in a simpler dress, representing the dame.
Explanation: This image illustrates the master and the dame receiving their bags of wool, highlighting the social hierarchy and the distribution of wealth.
Image 4: The Little Boy
Drawing Steps: Draw a small boy standing alone, holding a small bag of wool. He is looking at the sheep and the bags of wool with a sad expression.
Explanation: This image represents the little boy who lives down the lane, receiving only a small portion of the wool, symbolizing the unfairness of the wool distribution.

Based on above template, Explain the query in 6 images exact and give easy drawing steps for each image, and some explnation for all images.
Query: '''

# Function to generate grid image
def generate_grid_image(input_text):
    # Generate output from HuggingFaceEndpoint
    input_text=prompt_comic+input_text
    output = llm(input_text)

    # Extract drawing steps, explanation steps, and title steps
    pattern = r"Drawing Steps: (.*?)\nExplanation:"
    drawing_steps = re.findall(pattern, output)[:6]
    pattern = r"Explanation: (.*?)\."
    explanation_steps = re.findall(pattern, output)[:6]
    pattern = r"Image (.*?)\n"
    title_steps = re.findall(pattern, output)[:6]
    # st.write(output)
    # return
    # Generate images using Stable Diffusion Pipeline
    with torch.no_grad():
      res=pipe(drawing_steps, negative_prompt=['low quality, worst quality, text, watermark, signature']*6, generator=generator, num_inference_steps=30)# st.write('0')
    # Create a new image for the grid
    total_width = max(img.width for img in res.images) * 2
    total_height = max(img.height for img in res.images) * 3
    grid_image = Image.new('RGB', (total_width, total_height))

    # Create ImageDraw object
    draw = ImageDraw.Draw(grid_image)

    # Load font
    title_font = ImageFont.truetype(font_path, size=40)
    caption_font = ImageFont.truetype(font_path, size=20)

    # Paste images and add titles and captions
    for i, img in enumerate(res.images):
        row = i // 2
        col = i % 2

        row *= img.height
        x = col * img.width
        y = row

        grid_image.paste(img, (x, y))

        title = title_steps[i]
        draw.text((x, y), title, fill="yellow", font=title_font)
        caption = explanation_steps[i]
        draw.text((x, y + img.height - 50), caption, fill="yellow", font=caption_font)

    return grid_image

# Streamlit UI
st.title("Generate Grid Image from Text")

# Text input
input_text = st.text_area("Enter text:", height=200)

# Generate and display grid image
if input_text:
    grid_image = generate_grid_image(input_text)
    st.image(grid_image, caption="Generated Grid Image", use_column_width=True)
