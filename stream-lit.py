import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from skimage.measure import regionprops
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import center_of_mass
import pickle
from cellpose import models
import io
import pandas as pd
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] { height: 4rem; }
        .stImage { margin-top: 1rem; margin-bottom: 1rem; }
        .filter-button { margin: 5px; }
        </style>
    """, unsafe_allow_html=True)

st.title("🏥 Medical Analysis Dashboard")

stop_words = set(['a', 'an', 'the', 'and', 'is', 'in', 'it', 'of', 'to', 'for', 'on', 'with'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    cleaned_words = [re.sub(r'(ed|ing|ly|es|s)$', '', word) for word in words if word]
    return ' '.join(cleaned_words)

def create_sequences(texts, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(sequences, maxlen=max_len)
    return padded_seqs

def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)

def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (512, 512))
    processed_image = np.expand_dims(resized_image, axis=-1)
    uploaded_file.seek(0)
    return processed_image, original_image

def get_cell_info(input_image, nucleus_model, cytoplasm_model):
    predict = nucleus_model.predict(np.expand_dims(np.expand_dims(input_image, axis=-1), axis=0))
    nucleus_segmented_image = predict[0]
    
    if nucleus_segmented_image.ndim == 3 and nucleus_segmented_image.shape[2] == 1:
        nucleus_segmented_image = np.squeeze(nucleus_segmented_image)
    
    binary_image = nucleus_segmented_image > 0.5
    labeled_image, _ = ndimage_label(binary_image)
    regions = regionprops(labeled_image)
    nucleus_info = {}
    for region in regions:
        mask = np.zeros_like(binary_image, dtype=bool)
        for coord in region.coords:
            mask[coord[0], coord[1]] = True
        nucleus_info[region.label] = {
            'mask': mask,
            'centroid': (int(region.centroid[0]), int(region.centroid[1]))
        }
    mask, _, _ = cytoplasm_model.eval(input_image, diameter=150, channels=[0,0])
    unique_labels = np.unique(mask)
    cytoplasm_info = {}
    for label_id in unique_labels:
        if label_id == 0:
            continue
        binary_mask = (mask == label_id).astype(np.uint8)
        centroid = center_of_mass(binary_mask)
        cytoplasm_info[label_id] = {
            'mask': binary_mask,
            'centroid': centroid
        }
    
    return match_nucleus_cytoplasm(nucleus_info, cytoplasm_info)

def match_nucleus_cytoplasm(nucleus_info, cytoplasm_info):
    def euclidean_distance(centroid1, centroid2):
        return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    
    cell_info = {}
    temp_cytoplasm_info = cytoplasm_info.copy()
    
    for nucleus_label, nucleus_data in list(nucleus_info.items()):
        nucleus_centroid = nucleus_data['centroid']
        closest_cytoplasm_label = None
        min_distance = float('inf')
        
        for cytoplasm_label, cytoplasm_data in temp_cytoplasm_info.items():
            distance = euclidean_distance(nucleus_centroid, cytoplasm_data['centroid'])
            if distance < min_distance:
                min_distance = distance
                closest_cytoplasm_label = cytoplasm_label
        
        if closest_cytoplasm_label is not None:
            cell_info[nucleus_label] = {
                'nucleus_mask': nucleus_data['mask'],
                'nucleus_centroid': nucleus_centroid,
                'cytoplasm_label': closest_cytoplasm_label,
                'cytoplasm_mask': temp_cytoplasm_info[closest_cytoplasm_label]['mask'],
                'cytoplasm_centroid': temp_cytoplasm_info[closest_cytoplasm_label]['centroid']
            }
            del temp_cytoplasm_info[closest_cytoplasm_label]
    
    return cell_info

def visualize_combined_image(input_image, cell_info):
    if input_image.ndim > 2:
        if input_image.shape[-1] == 1:
            input_image = np.squeeze(input_image)
        else:
            input_image = input_image[:, :, 0] if input_image.ndim == 3 else input_image[:, :, 0, 0]
    
    original_image = input_image.reshape(512, 512)
    overview_image = np.stack([original_image] * 3, axis=-1)
    
    colors = [(255, 0, 0), (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]
    
    for idx, (label, info) in enumerate(cell_info.items()):
        nucleus_mask = info['nucleus_mask'].reshape(512, 512)
        cytoplasm_mask = info['cytoplasm_mask'].reshape(512, 512).astype(bool)
        
        color = colors[idx % len(colors)]
        color_overlay = np.zeros_like(overview_image)
        color_overlay[cytoplasm_mask] = color
        
        opacity = 0.3
        overview_image = np.where(
            cytoplasm_mask[:, :, np.newaxis],
            overview_image.astype(float) * (1 - opacity) + color_overlay * opacity,
            overview_image
        )
        
        nucleus_opacity = 0.5
        nucleus_overlay = np.zeros_like(overview_image)
        nucleus_overlay[nucleus_mask] = color
        overview_image = np.where(
            nucleus_mask[:, :, np.newaxis],
            overview_image.astype(float) * (1 - nucleus_opacity) + nucleus_overlay * nucleus_opacity,
            overview_image
        )
    
    return np.clip(overview_image, 0, 255).astype(np.uint8)

def get_individual_cell_images(input_image, cell_info):
    if input_image.ndim > 2:
        if input_image.shape[-1] == 1:
            input_image = np.squeeze(input_image)
        else:
            input_image = input_image[:, :, 0] if input_image.ndim == 3 else input_image[:, :, 0, 0]
    
    original_image = input_image.reshape(512, 512)
    individual_cells = []
    
    for label, info in cell_info.items():
        nucleus_mask = info['nucleus_mask'].reshape(512, 512)
        cytoplasm_mask = info['cytoplasm_mask'].reshape(512, 512).astype(bool)
        
        combined_mask = nucleus_mask | cytoplasm_mask
        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            continue
        
        padding = 20
        ymin = max(row_indices[0] - padding, 0)
        ymax = min(row_indices[-1] + padding + 1, 512)
        xmin = max(col_indices[0] - padding, 0)
        xmax = min(col_indices[-1] + padding + 1, 512)
        
        cropped_image = original_image[ymin:ymax, xmin:xmax]
        cropped_nucleus = nucleus_mask[ymin:ymax, xmin:xmax]
        cropped_cytoplasm = cytoplasm_mask[ymin:ymax, xmin:xmax]
        
        single_cell = np.full_like(cropped_image, 255)
        single_cell[cropped_cytoplasm] = cropped_image[cropped_cytoplasm]
        
        cell_rgb = np.stack([single_cell] * 3, axis=-1)
        
        nucleus_opacity = 0.5
        red_overlay = np.zeros_like(cell_rgb)
        red_overlay[cropped_nucleus] = [255, 0, 0]
        
        cell_rgb = cell_rgb.astype(float)
        cell_rgb = np.where(
            cropped_nucleus[:, :, np.newaxis],
            cell_rgb * (1 - nucleus_opacity) + red_overlay * nucleus_opacity,
            cell_rgb
        )
        
        individual_cells.append((label, np.clip(cell_rgb, 0, 255).astype(np.uint8)))
    
    return individual_cells
@st.cache_resource
def load_all_models():
    with st.spinner("Loading models..."):
        nucleus_model = load_model('nucleus_segmentation_model.h5', 
                                 custom_objects={'iou_metric': iou_metric})
        with open('cytoplasm_segmentation_model.pkl', 'rb') as f:
            cytoplasm_model = pickle.load(f)
        cancer_model = load_model('cancer_model_final.h5')
        data = pd.read_csv('cancer-sentiment.csv')
        label_encoder = LabelEncoder()
        data['encoded_labels'] = label_encoder.fit_transform(data['cancer_type'])
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(data['text'].apply(clean_text))
        
    return nucleus_model, cytoplasm_model, cancer_model, tokenizer, label_encoder
def process_image_with_filters(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Apply different filters
    median_filtered = cv2.medianBlur(input_image, 5)
    bilateral_filtered = cv2.bilateralFilter(input_image, 9, 75, 75)
    gaussian_filtered = cv2.GaussianBlur(input_image, (5, 5), 0)
    denoised_image = cv2.fastNlMeansDenoisingColored(input_image, None, 10, 10, 7, 21)
    
    # Convert all images to RGB for display
    median_filtered = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB)
    bilateral_filtered = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2RGB)
    gaussian_filtered = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB)
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    
    uploaded_file.seek(0)
    return original_image, median_filtered, bilateral_filtered, gaussian_filtered, denoised_image

@st.cache_resource
def load_denoising_model():
    return load_model('best_denoising_model.keras')

def apply_dl_denoising(image, denoising_model):
    # Preprocess image for the denoising model
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized_image = gray_image / 255.0
    input_image = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=-1)
    
    # Get denoised prediction
    denoised = denoising_model.predict(input_image)[0, :, :, 0]
    denoised = (denoised * 255).astype(np.uint8)
    
    # Convert back to RGB
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return denoised_rgb

def main():
    try:
        with st.spinner("Loading models..."):
            nucleus_model, cytoplasm_model, cancer_model, tokenizer, label_encoder = load_all_models()
            denoising_model = load_denoising_model()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Initialize session state variables
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'dl_denoised_image' not in st.session_state:
        st.session_state.dl_denoised_image = None
    
    tab1, tab2 = st.tabs(["📸 Cell Analysis", "📝 Cancer Type Classification"])
    
    with tab1:
        st.header("Cell Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload cell image",
            type=['png', 'jpg', 'jpeg'],
            key="cell_image"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing image..."):
                    original, median, bilateral, gaussian, nlm_denoised = process_image_with_filters(uploaded_file)
                    
                    st.subheader("Select Processing Method")
                    filter_tabs = st.tabs(["Basic Filters", "Advanced Processing"])
                    
                    with filter_tabs[0]:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.image(original, caption="Original", use_container_width=True)
                            if st.button("Use Original", key="orig"):
                                st.session_state.selected_image = original
                                st.session_state.processing_complete = True
                        
                        with col2:
                            st.image(median, caption="Median Filter", use_container_width=True)
                            if st.button("Use Median", key="med"):
                                st.session_state.selected_image = median
                                st.session_state.processing_complete = True
                        
                        with col3:
                            st.image(bilateral, caption="Bilateral Filter", use_container_width=True)
                            if st.button("Use Bilateral", key="bil"):
                                st.session_state.selected_image = bilateral
                                st.session_state.processing_complete = True
                        
                        with col4:
                            st.image(gaussian, caption="Gaussian Filter", use_container_width=True)
                            if st.button("Use Gaussian", key="gauss"):
                                st.session_state.selected_image = gaussian
                                st.session_state.processing_complete = True
                        
                        with col5:
                            st.image(nlm_denoised, caption="NLM Denoised", use_container_width=True)
                            if st.button("Use NLM", key="nlm"):
                                st.session_state.selected_image = nlm_denoised
                                st.session_state.processing_complete = True
                    
                    with filter_tabs[1]:
                        st.write("Deep Learning Based Denoising")
                        if st.button("Apply Deep Learning Denoising", key="dl_denoise"):
                            with st.spinner("Applying deep learning denoising..."):
                                dl_denoised = apply_dl_denoising(original, denoising_model)
                                st.session_state.dl_denoised_image = dl_denoised
                                st.image(dl_denoised, caption="DL Denoised", use_container_width=True)
                        
                        # Only show the "Use DL Denoised Result" button if we have a denoised image
                        if st.session_state.dl_denoised_image is not None:
                            if st.button("Use DL Denoised Result", key="use_dl"):
                                st.session_state.selected_image = st.session_state.dl_denoised_image
                                st.session_state.processing_complete = True
                    
                    # Process the selected image
                    if st.session_state.processing_complete and st.session_state.selected_image is not None:
                        st.subheader("Cell Analysis Results")
                        
                        # Process the selected image
                        processed_image = cv2.resize(
                            cv2.cvtColor(st.session_state.selected_image, cv2.COLOR_RGB2GRAY),
                            (512, 512)
                        )
                        
                        # Get cell information
                        with st.spinner("Analyzing cells..."):
                            cell_info = get_cell_info(processed_image, nucleus_model, cytoplasm_model)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(
                                st.session_state.selected_image,
                                caption="Selected Image",
                                use_container_width=True
                            )
                        
                        with col2:
                            combined_image = visualize_combined_image(processed_image, cell_info)
                            st.image(
                                combined_image,
                                caption="Detected Cells",
                                use_container_width=True
                            )
                        
                        # Display individual cells
                        st.subheader("Individual Cells")
                        individual_cells = get_individual_cell_images(processed_image, cell_info)
                        
                        if individual_cells:
                            cell_cols = st.columns(4)
                            for idx, (label, cell_image) in enumerate(individual_cells):
                                with cell_cols[idx % 4]:
                                    st.image(
                                        cell_image,
                                        caption=f"Cell {label}",
                                        use_container_width=True
                                    )
                        
                        # Cell analysis details
                        with st.expander("Cell Analysis Details"):
                            for label, info in cell_info.items():
                                st.write(f"Cell {label}:")
                                st.write(f"- Nucleus position: ({info['nucleus_centroid'][0]:.1f}, {info['nucleus_centroid'][1]:.1f})")
                                st.write(f"- Cytoplasm label: {info['cytoplasm_label']}")
                        
                        # Add download button for processed image
                        if 'combined_image' in locals():
                            is_success, buffer = cv2.imencode(
                                ".png",
                                cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                            )
                            if is_success:
                                st.download_button(
                                    label="Download Processed Image",
                                    data=buffer.tobytes(),
                                    file_name="processed_cells.png",
                                    mime="image/png"
                                )
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    
    # Cancer Type Classification Tab
    with tab2:
        st.header("Cancer Type Classification")
        text_input = st.text_area(
            "Enter medical text for cancer type classification:",
            height=150
        )
        
        if st.button("Analyze Text"):
            if text_input:
                try:
                    with st.spinner("Analyzing text..."):
                        # Clean and process text
                        cleaned_text = clean_text(text_input)
                        sequences = create_sequences([cleaned_text], tokenizer)
                        
                        # Get prediction
                        prediction = cancer_model.predict(sequences)
                        predicted_class = np.argmax(prediction, axis=1)[0]
                        predicted_cancer_type = label_encoder.inverse_transform([predicted_class])[0]
                        confidence = float(prediction[0][predicted_class])
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Cancer Type", predicted_cancer_type)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Show processed text
                        with st.expander("View Processed Text"):
                            st.write(cleaned_text)
                        
                        # Show prediction details
                        with st.expander("View Prediction Details"):
                            st.write("Top 3 predictions:")
                            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                            for idx in top_3_indices:
                                cancer_type = label_encoder.inverse_transform([idx])[0]
                                prob = prediction[0][idx]
                                st.write(f"{cancer_type}: {prob:.2%}")
                
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This dashboard combines cell image analysis and cancer type classification capabilities:
    
    - **Cell Analysis**: Upload microscopy images to detect and analyze individual cells
    - **Cancer Classification**: Input medical text for cancer type prediction
    
    Built with Streamlit, TensorFlow, and OpenCV
    """)

if __name__ == "__main__":
    main()
