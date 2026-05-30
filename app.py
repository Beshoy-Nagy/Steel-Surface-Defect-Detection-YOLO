import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------
# Load Model
# ---------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------------------
# Sidebar
# ---------------------------

with st.sidebar:
    st.header("📊 Model Information")

    st.write("**Model:** YOLO")
    st.write("**Classes:** 6")

    st.write("### Defect Classes")

    st.write("• Crazing")
    st.write("• Inclusion")
    st.write("• Patches")
    st.write("• Pitted Surface")
    st.write("• Rolled-in Scale")
    st.write("• Scratches")

    show_conf = st.checkbox(
        "Show Confidence Scores",
        value=True
    )

# ---------------------------
# Main Page
# ---------------------------

st.title("🔍 Steel Surface Defect Detection")

st.markdown(
    """
    Upload a steel surface image and the model will
    detect and classify surface defects automatically.
    """
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction
# ---------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        results = model(tmp.name)

    annotated = results[0].plot()

    boxes = results[0].boxes

    st.metric(
        "Detected Defects",
        len(boxes)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Detection Result")
        st.image(
            annotated,
            use_container_width=True
        )

    if len(boxes) > 0:

        st.subheader("📋 Detection Details")

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            defect_name = model.names[cls_id]

            if show_conf:

                st.success(
                    f"{defect_name} — {conf:.2%}"
                )

            else:

                st.success(
                    defect_name
                )

    else:

        st.success(
            "✅ No defects detected."
        )

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")

st.caption(
    "Developed using YOLO and Streamlit"
)