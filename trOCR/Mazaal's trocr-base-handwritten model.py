from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# Load image from the IAM database
file_path = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\preprocessing\test\Step3_Paragraph\mnsit_paragraph2.jpeg'
image = Image.open(file_path).convert("RGB")

# Load the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Process the image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate text from the image
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)
