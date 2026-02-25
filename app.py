import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

from model import build_resnet_model, train_model, save_model, load_trained_model
from utils import preprocess_image, predict_digit, load_image_from_upload, preprocess_for_display

# Set page config
st.set_page_config(
    page_title="ResNet OCR System",
    page_icon="🔤",
    layout="wide"
)

# Model path
MODEL_PATH = "ocr_model.h5"


@st.cache_resource
def get_model():
    """Load or train the model"""
    if os.path.exists(MODEL_PATH):
        try:
            return load_trained_model(MODEL_PATH)
        except:
            pass
    
    # Train new model if not found
    with st.spinner('Training model on MNIST data... This may take a few minutes...'):
        model = build_resnet_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model, history = train_model(model, epochs=5)
        save_model(model, MODEL_PATH)
        st.success("Model trained and saved!")
    
    return model


def main():
    # Title and description
    st.title("🔤 ResNet-Driven OCR Text Recognition System")
    st.markdown("""
    This application uses a **ResNet-based CNN** model trained on the MNIST dataset 
    to recognize handwritten digits from uploaded images.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    
    if model_exists:
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.info("⏳ Model will be trained on first use")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🖌️ Draw Digit", "ℹ️ About"])
    
    with tab1:
        st.header("Upload an Image")
        st.write("Upload an image containing a handwritten digit (0-9)")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Upload an image of a handwritten digit"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = load_image_from_upload(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(preprocess_for_display(image), caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("Prediction Result")
                
                with st.spinner('Processing image...'):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    model = get_model()
                    predicted_digit, confidence, all_probs = predict_digit(model, processed_image)
                
                # Display result
                st.markdown(f"## 🎯 Predicted Digit: **{predicted_digit}**")
                st.markdown(f"### Confidence: {confidence * 100:.2f}%")
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                probabilities = {str(i): prob for i, prob in enumerate(all_probs)}
                st.bar_chart(probabilities)
    
    with tab2:
        st.header("Draw a Digit")
        st.write("Use the canvas to draw a digit and get predictions")
        
        # Drawing canvas using streamlit-drawable-canvas
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Canvas settings
            stroke_width = st.sidebar.slider("Stroke width", 1, 25, 15)
            stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
            bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")
            
            # Create canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )
            
            if canvas_result.image_data is not None and st.button("Predict from Drawing"):
                # Convert canvas to image
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('L')  # Convert to grayscale
                
                # Resize to 280x280 (canvas size) then predict
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.expand_dims(img_array, axis=-1)
                img_array = tf.image.resize(img_array, [32, 32])
                
                # Predict
                model = get_model()
                predictions = model.predict(img_array, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                st.success(f"🎯 Predicted Digit: **{predicted_digit}** (Confidence: {confidence*100:.2f}%)")
                
        except ImportError:
            st.warning("📦 Install 'streamlit-drawable-canvas' for drawing functionality:")
            st.code("pip install streamlit-drawable-canvas", language="bash")
    
    with tab3:
        st.header("About This Project")
        st.markdown("""
        ### 🔍 Project Overview
        This is an end-to-end deep learning pipeline that automates text recognition 
        from image inputs using the MNIST handwritten digits dataset.
        
        ### 🧠 Model Architecture
        - **ResNet-enhanced CNN** for deep feature extraction
        - Residual blocks with skip connections
        - Global Average Pooling
        - Softmax output layer for 10-class classification
        
        ### 📊 Dataset
        - MNIST handwritten digits (0-9)
        - 60,000 training samples
        - 10,000 test samples
        
        ### 🛠️ Tech Stack
        | Component | Technology |
        |-----------|------------|
        | Frontend | Streamlit |
        | Backend | TensorFlow/Keras |
        | Model | ResNet-based CNN |
        
        ### 📝 How to Run
        
```
bash
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the app
        streamlit run app.py
        
```
        """)


if __name__ == "__main__":
    main()
