import numpy as np
from PIL import Image


def preprocess_image(image, target_size=(32, 32)):
    """
    Preprocess an image for prediction
    
    Args:
        image: PIL Image or file path
        target_size: Target size for the model (default: 32x32)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Expand dimensions to match model input (1, 32, 32, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array


def predict_digit(model, image):
    """
    Predict digit from preprocessed image
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
    
    Returns:
        Predicted digit and confidence scores
    """
    predictions = model.predict(image, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_digit, confidence, predictions[0]


def load_image_from_upload(uploaded_file):
    """
    Load an image from Streamlit file uploader
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        PIL Image
    """
    image = Image.open(uploaded_file)
    return image


def preprocess_for_display(image):
    """
    Convert image to displayable format for Streamlit
    
    Args:
        image: PIL Image
    
    Returns:
        Image suitable for st.image()
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
