> Update 2023-05-26: A lot has changed over the years since I created this. I think there are some interesting opportunities for creating a better system with multimodal models and I hope to have them out soon.

# This wine does not exist
The goal for this project is to create an ensemble of various machine learning models that work together to generate synthetic bottles of wine. At a high level this includes generating fake wine names, then generating the attributes and descriptions based on the name, and finally using a GAN to create a label for the bottle based on the synthetic attributes.

# Link: www.ThisWineDoesNotExist.com

### Models at Work
- **Name**: (*TensorFlow*) Character embedding layer, LSTM x 2, Dense ASCII output
- **Description**: (*PyTorch*) GPT2-XL 1.5b parameters, trained on 130,000 samples
- **Label**: (*TensorFlow*) NVIDIA StyleGAN2 and CV2

### Data
The original release of this project used about 15,000 pages of wines scraped from www.wine.com and then cleaned up for use in the models. v1.1.0 has been updated with over 130,000 wines now, and the data has now been used to re-train the GPT2 model, now with 1.5 billion parameters.

### Sample Page

<img src="https://raw.githubusercontent.com/cipher982/this-wine-does-not-exist/master/images/page_sample.png" alt="sample-wine-page" width="700"/>


