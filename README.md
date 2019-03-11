# This wine does not exist
Training neural networks to generate fake wine names, descriptions, labels, etc . . .

I originally Scraped ~15,000 wines and their data from the www.Wine.com and then trained various TensorFlow/Keras models to generate new wine names and descriptions, price, and labels.

### Models at Work
This project includes various different models trained for specific purposes. There is one for generating names that operates completely at random I use as a starting point. I have then trained another for the descriptions that uses the names as a seed. The (real) names and descriptions (as opposed to the fake ones I just created) were passed through a quick model of embedding, convolution, and a dense layer to linearly predict the numeric price value. This is then run on the recently made fake data. 
