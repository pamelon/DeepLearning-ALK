# Good evening again ^3!

So lets start with:
1. Any questions, concerns or thoughts from our last lecture? 
2. So one from me, what did we talk about last time? 
3. And, what problems do we encounter when training models for vision?

Some quick terms to repeat:
1. Convolutional Layer
2. MaxPooling Layer
3. ZlexNet
4. Baseline Model

# Processing Units

What is compute power?  
What is a CPU?  

![](img/or-gate.gif)  

And then...  

![](img/gate-circuit.jpg)  

But for us, users - this is a transparent layer. We just need an object to do something for us.  

<img src="animation.jpg" alt="You as a graphic designer" style="width:700px;"/>

So what is this CPU?

![](img/cpu.png)  

CPU is a single pair of hands:
 
<img src="cpu-hands.jpg" alt="CPU as a hand" style="width:900px;"/>

But what if we need multiple pairs of hands?

<img src="gpu-hands.jpg" alt="GPU as hands" style="width:900px;"/>

![](img/gpu.png)  

Parralelism is the core of GPU. We have a lot of hands but these are smaller and less complex hands. Actually GPU cannot do anything that CPU cannot do. CPU just does it (usually) slower.

What GPU is good for (and why?):
1. Image rendering (in real time)
2. Deep Learning
3. Data mining
4. Cryptocurrency mining
5. Gaming

# CUDA 

![](img/nvidia.jpg)  

Example is in /code folder for you to [have a peek](code/cuda-example.cu).  
And if you have a GPU - give it a spin!

So how big should my rented apartment be to fit in all the cores?  

<img src="img/data-center.png" alt="My house as a Data Center" style="width:800px;"/>

That's why:  

![](img/cloud-computing.jpg)  
<img src="img/cloud-computing.jpg" alt="Cloud Computing sign" style="width:900px;"/>