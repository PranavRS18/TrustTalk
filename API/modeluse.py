# Create custom objects scope for KerasLayer

import tensorflow_hub as hub
from keras.models import load_model
import tensorflow as tf 

custom_objects = {'KerasLayer': hub.KerasLayer}

def load_model_with_custom_layer(model_path):
    """
    Load a TensorFlow model with custom layer support
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        loaded_model: TensorFlow model instance
    """
    try:
        # Load the model with custom objects scope
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(model_path)
            print("Model loaded successfully!")
            return model
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
#print(load_model_with_custom_layer("mymodel.h5").predict(["Hello sir, you have got an lottery give me you otp to win 1 million dollar"]))
    
