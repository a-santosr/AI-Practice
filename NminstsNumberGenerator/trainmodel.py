import streamlit as st
import numpy as np
import tensorflow as tf

# Load the trained decoder model
decoder = tf.keras.models.load_model('cvae_decoder.h5')

st.title("CVAE Image Generator")

# Input: text field (for example, a digit label)
label_text = st.text_input("Enter a digit (0-9):", "0")

if st.button("Generate Image"):
    try:
        # Convert input to integer class (adapt as needed for your model)
        label = int(label_text)
        if not (0 <= label <= 9):
            st.error("Please enter a digit between 0 and 9.")
        else:
            # Create one-hot encoding for the label
            y = np.zeros((1, 10))
            y[0, label] = 1

            # Sample a random latent vector (adapt size as needed)
            z = np.random.normal(size=(1, 2))  # Change 2 to your latent dim

            # Generate image using the decoder
            generated_img = decoder.predict([z, y])

            # Reshape and display the image
            img = generated_img[0].reshape(28, 28)  # Change shape as needed
            st.image(img, caption=f"Generated image for label {label}", width=200, clamp=True)
    except ValueError:
        st.error("Please enter a valid integer.")