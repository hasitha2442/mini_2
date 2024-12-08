import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Define SEBlock
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale

# Define the model architecture
class EnsembleEfficientNet_TwoBranch(nn.Module):
    def __init__(self, embed_dim=24):
        super(EnsembleEfficientNet_TwoBranch, self).__init__()

        # Load EfficientNet as the backbone
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])

        # Backbone processing layers with SEBlock
        self.backbone_conv = nn.Conv2d(1280, embed_dim, kernel_size=1)
        self.backbone_bn = nn.BatchNorm2d(embed_dim)
        self.backbone_se = SEBlock(embed_dim)

        # Two branches from the input with SEBlocks
        self.branch1_conv = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        self.branch1_bn = nn.BatchNorm2d(embed_dim)
        self.branch1_se = SEBlock(embed_dim)

        self.branch2_conv = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        self.branch2_bn = nn.BatchNorm2d(embed_dim)
        self.branch2_se = SEBlock(embed_dim)

        # Dense and spectral blocks for branches
        self.dense_block1 = self._dense_block(embed_dim, embed_dim, 4)
        self.dense_spectral_block = self._dense_block(embed_dim, embed_dim, 4)

        # Glaucoma syndrome mechanism (1D convolution) for concatenated output
        self.glaucoma_conv = nn.Conv1d(72, embed_dim, kernel_size=1)

        # Final fully connected layer for classification
        self.fc = nn.Linear(embed_dim, 3)

        # Adaptive pooling and dropout
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

    def _dense_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        return nn.Sequential(*layers)

    def forward_branch(self, x, conv, bn, se_block):
        x = torch.relu(bn(conv(x)))
        x = se_block(x)
        x1 = self.dense_block1(x)
        x = x + x1
        x2 = self.dense_spectral_block(x)
        x = x + x2
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        # Forward through EfficientNet backbone
        backbone_out = self.backbone(x)
        backbone_out = torch.relu(self.backbone_bn(self.backbone_conv(backbone_out)))
        backbone_out = self.backbone_se(backbone_out)
        backbone_out = self.adaptive_pool(backbone_out).view(backbone_out.size(0), -1)
        backbone_out = self.dropout(backbone_out)

        # Forward through Branch 1
        branch1_out = self.forward_branch(x, self.branch1_conv, self.branch1_bn, self.branch1_se)

        # Forward through Branch 2
        branch2_out = self.forward_branch(x, self.branch2_conv, self.branch2_bn, self.branch2_se)

        # Concatenate outputs from backbone, branch1, and branch2
        combined_out = torch.cat((backbone_out, branch1_out, branch2_out), dim=1)

        # Glaucoma syndrome mechanism
        combined_out = self.glaucoma_conv(combined_out.unsqueeze(-1)).squeeze(-1)

        # Final fully connected layer
        out = self.fc(combined_out)

        return out

# Cache the model loading process
@st.cache_resource
def load_model():
    model_path = '/content/model_epoch_58.pth'
    model = EnsembleEfficientNet_TwoBranch(embed_dim=24)  # Ensure to match saved model's embed_dim
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Prediction function
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities[0]

# Load the model
model = load_model()

# Streamlit app
st.title("Multi-Class Classification with Two-Branch Model")
st.write("Upload an image to classify it into one of three classes.")

# File uploader
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Main logic
if file is None:
    st.text("Please upload an image file.")
else:
    try:
        image = Image.open(file.stream if hasattr(file, 'stream') else file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        preprocessed_image = preprocess_image(image)
        predicted_class, probabilities = predict(preprocessed_image, model)

        class_names = ["Advanced", "Early", "Normal"]  # Replace with your class names
        st.success(f"This image most likely belongs to: **{class_names[predicted_class]}**")
        st.write("### Prediction Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {probabilities[i]:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
