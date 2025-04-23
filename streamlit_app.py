import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from vit_mnist import ViT
from streamlit_drawable_canvas import st_canvas

class DigitRecognizer:
    def __init__(self):
        self.model = ViT(
            image_size=28,
            patch_size=4,  # Smaller patches
            num_classes=10,
            dim=256,       # Increased dimension
            depth=8,       # Increased depth
            heads=8,
            mlp_dim=512,   # Increased MLP dimension
            channels=1,
            dim_head=32,
            dropout=0.1,
            emb_dropout=0.1
        )
        # Load the best model instead of the old one
        checkpoint = torch.load('vit_mnist_best.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def preprocess_digit(self, image_array):
        # Convert to PIL Image
        image = Image.fromarray(image_array.astype('uint8'))
        image = image.convert('L')
        
        # Simple thresholding to make it binary
        image = image.point(lambda x: 255 if x > 128 else 0)
        
        # Find bounding box
        bbox = ImageOps.invert(image).getbbox()
        if bbox:
            image = image.crop(bbox)
            
            # Add small padding
            padding = int(min(image.size) * 0.1)
            image = ImageOps.expand(image, border=padding, fill=0)
            
            # Resize while maintaining aspect ratio
            target_size = 20
            ratio = min(target_size / image.size[0], target_size / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Center in 28x28 image
            final_image = Image.new('L', (28, 28), 0)
            paste_x = (28 - image.size[0]) // 2
            paste_y = (28 - image.size[1]) // 2
            final_image.paste(image, (paste_x, paste_y))
            
            return final_image
        return Image.new('L', (28, 28), 0)

    def predict_digit(self, image_array):
        processed_img = self.preprocess_digit(image_array)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(processed_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > 0.1:  # Very low threshold for testing
                return predicted.item(), confidence.item()
            return None, 0.0

def draw_guidelines(image_array, num_digits):
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    segment_width = width // num_digits
    
    # Draw simple, thin lines
    for i in range(1, num_digits):
        x = i * segment_width
        draw.line([(x, 0), (x, height)], 
                 fill=(100, 100, 100, 128),  # Lighter, semi-transparent line
                 width=1)  # Thinner line
    
    return np.array(image)

def main():
    st.set_page_config(layout="wide")
    
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = DigitRecognizer()
    if 'start_new' not in st.session_state:
        st.session_state.start_new = True
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    st.title("Digit Recognition")

    # Updated CSS with more prominent output styling
    st.markdown("""
        <style>
        .stCanvas {
            display: flex;
            justify-content: center;
        }
        .prediction-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            border: 2px solid #e0e0e0;
        }
        .prediction-text {
            font-size: 32px;
            font-weight: bold;
            color: #0066cc;
        }
        .warning-text {
            color: #ff4b4b;
            font-size: 18px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        num_digits = st.slider("Number of digits", 1, 5, 3)
        
        # Canvas dimensions
        canvas_height = 200
        digit_width = 100
        total_width = digit_width * num_digits

        # Canvas with guidelines
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=canvas_height,
            width=total_width,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.start_new}"
        )

        if canvas_result.image_data is not None:
            image_with_guidelines = draw_guidelines(canvas_result.image_data, num_digits)
            st.image(image_with_guidelines, use_container_width=True)

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            predict_button = st.button("Predict", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

        # Output display container
        output_container = st.empty()

        if clear_button:
            st.session_state.start_new = not st.session_state.start_new
            st.session_state.last_prediction = None
            output_container.empty()
            st.rerun()

        if predict_button and canvas_result.image_data is not None:
            predictions = []
            confidences = []
            img_array = canvas_result.image_data
            
            for i in range(num_digits):
                start_x = i * digit_width
                end_x = (i + 1) * digit_width
                start_x = max(0, start_x - 5)
                end_x = min(total_width, end_x + 5)
                digit_section = img_array[:, start_x:end_x]
                
                digit, conf = st.session_state.recognizer.predict_digit(digit_section)
                if digit is not None:
                    predictions.append(str(digit))
                    confidences.append(conf)
                else:
                    predictions.append("?")
                    confidences.append(0.0)
            
            if predictions:
                result = ''.join(predictions)
                if "?" not in predictions:
                    st.session_state.last_prediction = result
                    output_container.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-text">
                                Predicted Number: {result}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidences
                    st.write("Confidence scores:")
                    for idx, conf in enumerate(confidences):
                        st.write(f"Digit {idx+1}: {conf:.2f}")
                else:
                    output_container.markdown(f"""
                        <div class="prediction-box">
                            <div class="warning-text">
                                Partial Recognition: {' '.join(predictions)}
                            </div>
                            <div>Please write digits more clearly</div>
                        </div>
                    """, unsafe_allow_html=True)

        elif st.session_state.last_prediction:
            output_container.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-text">
                        Last Prediction: {st.session_state.last_prediction}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Tips section
        with st.expander("Tips for better recognition"):
            st.markdown("""
            - Write each digit clearly in its section
            - Use medium-thick strokes
            - Center the digits
            - Keep digits separate
            - Write numbers in a standard format
            """)

if __name__ == "__main__":
    main()