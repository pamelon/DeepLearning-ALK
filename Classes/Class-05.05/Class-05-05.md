# Good evening again ^2!

So lets start with:
1. Any questions, concerns or thoughts from our last lecture? 
2. So one from me, what did we talk about last time? 
3. And, what problems do we need to address when building a model for language understanding?

Some quick terms to repeat:
1. Bag of words
2. Information bottleneck
3. Transformer
4. BERT

Two words about baselines  
![](img/baseline.png)  

# Let's now look at vision

One thing to remember is that Depp Learning rose to the recent promimance in 2010s because of problems related to vision, image recognition and image tagging. What we have with DL now, is because of a very famous algorithmic competition called ImageNet Large Scale Visual Recognition Challenge (ILSVRC), in short just ImageNet Challenge. In the challenge you had access to the data of over a milion images for training, that are put in a 1000 categories from "Strawberry", "Dog" to "Animal", overlapping each other in some cases.  
Competition: https://www.image-net.org/challenges/LSVRC/   

The whole ImageNet dataset has been manually tagged and consists of 14 milion images belonging to 20 000 classes. 

![](img/imagenet-pictures.png) 

Do you see any problems with that?

The problems:
1. Manual annotation done by humans
2. Need for large dataset of examples
3. Is that the way we, humans think?

[Finally the brain!](../Class-04.20/Class-04.20.md)  

Getting back to the ImageNet:

In 2012 Alex Krivshevsky with his team designed a network that declassified the competition. The network achieved an error of 15.3%, around 11 percentage points lower than that next best one. This is crazy for ML standards, as you are probably aware. 

<img src="img/alexnet.jpg" alt="AlexNet" style="width:800px;"/>

Deep dive into AlexNet: https://www.youtube.com/watch?v=7LQSdPjWjdA  
The AlexNet code example: https://github.com/krishnaik06/Advanced-CNN-Architectures/blob/master/Transfer%20Learning%20Alexnet.ipynb  
The actual paper (worth looking into for those of you who are interested): https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf   

# So what is this convolution?

<img src="img/parts-of-cat.png" alt="Parts of a cat" style="width:700px;"/>

![](img/convolution.gif)

![](img/filter-conv.png)
Source: https://www.manning.com/books/deep-learning-with-python 

More on convolution: https://www.youtube.com/watch?v=YRhxdVk_sIs  
And deeper (you get the joke): https://www.youtube.com/watch?v=m8pOnJxOcqY  

And padding?  

# And now into MaxPooling

In short - downsampling  
For:
1. Efficiency of processing
2. Efficiency of memory
3. Keeping the most important parts of each section

![](img/maxpool.png)

![](img/maxpooling-example.jpg)

MaxPooling: https://www.youtube.com/watch?v=ZjM_XQa5s6s  

# Now back to AlexNet!

What did AlexNet do that made it so revolutionary? 
1. Data augmentation

![](img/augmentation.png)

2. Dropout

![](img/dropout.png)

3. ReLU
Performance with precision 
![](../Class-04.20/img/relu.png)

4. ...
