# This wine does not exist
The goal for this project is to create an ensemble of various machine learning models that work together to generate synthetic bottles of wine. At a high level this includes generating fake wine names, then generating the attributes and descriptions based on the name, and finally using a GAN to create a label for the bottle based on the synthetic attributes.


### Models at Work
- *LSTM* - Generate the fake wine names
- *GPT-2* - Generate the wine attributes and natural language description
- *Stylegan2* - Generate the label images conditioned on the wine attributes

### Data
The original version of this project used about 15,000 pages of wines scraped from www.wine.com and then cleaned up for use in the models. Version 2 has been updated with over 130,000 wines now, and the models are all being re-trained with this new data as of 2020-03-18.

**LINK**: www.ThisWineDoesNotExist.com (still using all v1 models)

### Sample Page:
----
![Sample Page](https://raw.githubusercontent.com/cipher982/this-wine-does-not-exist/master/images/sample_page.png)
----

