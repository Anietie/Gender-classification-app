import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torch import nn

class GenderClassificationModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=8*32*32, out_features=output_shape)
        )

    def forward(self, x):
        x = self.convblock(x)
        x = self.classifier(x)
        return x


model = torch.load("model.pth")
device = 'cpu'
model.to(device)
model.eval()

transformation = transforms.Compose([
transforms.Resize((64, 64)),
transforms.ToTensor()
])



st.title("Ben's Gender Classification DL Model App")
st.write("Upload an image of a person to predict the gender of the person.")
st.write("Performs worse on Female images.")

st.warning("**Note:** The model does not perform with high accuracy, it's just for educational purpose.")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_tensor = transformation(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.round(torch.sigmoid(output))
        # prediction_list = ["Female", "Male"]
        # predicted_gender = prediction_list[int(prediction)]
        prediction = int(prediction)

    # Display the result
    if prediction == 0:
        st.write("Prediction: Female")
    else:
        st.write("Prediction: Male")

