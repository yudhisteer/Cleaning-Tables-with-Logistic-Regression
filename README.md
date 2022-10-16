# Cleaning Tables with Logistic Regression

## Abstract

## Plan of Action

1. Linear Classification
    - Equation of a linear classifier
    - The Neuron
    - Output of logistic regression

2. Optimal Weights
    - Bayes' Solution
    - Error Function
    - Maximum Likelihood: Bernoulli
    - Closed-Form Solution
    - Gradient Descent

3. Regularization
    - Interpretation of the weights
    - L2 Regularization
    - L1 Regularization
    - ElasticNet

4. Special Cases
    - Donut Problem
    - XOR Problem

5. Clean/Dirty Tables Classification

----------
## 1. Linear Classification

### 1.1 Equation of a linear classifier

Logistic regression is a statistical **binary classification** model. Since it is a **supervised learning** model, data should consist of ```features``` and a ```label```. An example of data for such a model is shown below. Note that x1 and x2 can take any numerical **continuous** value whereas the label is **discrete** such as ```0``` representing a **class** and ```1``` for another class.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196005774-6ec28549-4b05-4dbe-904e-cf25918357aa.png" width="200" height="200"/>
</p>


For the following example, we will assume a  "linear classifier":

- The equation of the classifier: ![CodeCogsEqn (6)](https://user-images.githubusercontent.com/59663734/196002313-f226cd57-b27f-4791-ba88-b70e154ba4a2.png)
- Since our data is in the form of <img src="https://latex.codecogs.com/png.image?\dpi{110}(x_1,x_2)" title="https://latex.codecogs.com/png.image?\dpi{110}(x_1,x_2)" /> with label ```y```then rewriting the above equation: ![CodeCogsEqn (7)](https://user-images.githubusercontent.com/59663734/196002344-c70756b2-e033-4e6b-9116-359d34e37809.png)
- Taking everything on the LHS: <img src="https://latex.codecogs.com/png.image?\dpi{110}x_2&space;-&space;x_1&space;=&space;0" title="https://latex.codecogs.com/png.image?\dpi{110}x_2 - x_1 = 0" />
- Puting it in vector form:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196003242-2d607b17-7f01-4770-83ec-718632518e70.png"/>
</p>

- ![CodeCogsEqn (10)](https://user-images.githubusercontent.com/59663734/196003354-7df6285b-3802-414f-9b60-76964223e10f.png) are the **features** and ![CodeCogsEqn (11)](https://user-images.githubusercontent.com/59663734/196003369-503d69cd-98cc-42a8-a5c5-d890c38269ec.png) are called the **weights**.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/195506122-b274ac02-7936-4b4d-bae0-4c619f5df52b.png" width="700" height="250"/>
</p>


For the above example, we are mapping the label column as color: ```1``` = red, ```0``` = blue. That is, we are representing 3-dimensional data as 2-dimensional. In order to classify a new data (yellow triangle) visually, we need to check on which side of the classifier it is on. Mathematically, we just need to check the ```sign``` of our linear classifier function.

In the first graph, suppose our yellow triangle has coordinates ```(3, 4)```. When plotting that coordinates in our linear classifier, the output is a positive number. Hence, we associate ```positive``` to the class "blue". 

In the second graph, the yellow triangle has coordinates ```(4, 3)```. Doing the exact same steps above, we obtain a negative result. Hence, we conclude ```negative``` means "red" class.

What happens if the triangle is exactly on the linear classifier with coordinates ```(3,3)```? We obtain a ```0``` which we will see later  means that our triangle can come from either the "**red**" class or the "**blue**" class.


In general, the equation of our linear classifier is as follows:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196007462-7595a194-9833-49a2-a85f-c927d902cbb2.png"/>
</p>

Expressing the above equation in vector form:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196007350-c0fac2d9-8dc0-4dca-8715-285ddb2dc51b.png"/>
</p>

Absorbing <img src="https://latex.codecogs.com/png.image?\dpi{110}w_0" title="https://latex.codecogs.com/png.image?\dpi{110}w_0" />:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196007399-2d83636d-c572-45cd-9a75-53a2180c80a3.png"/>
</p>

Finally: 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196007475-a01414c3-825a-4aa7-833a-3d62b5fffd66.png"/>
</p>
where:

- ```x```: feature vector
- ```w```: weights vector
- ```D```: number of features
- <img src="https://latex.codecogs.com/png.image?\dpi{110}w_0" title="https://latex.codecogs.com/png.image?\dpi{110}w_0" />: bias

In order to classify a new point, we need to check for the **sign** of ```h(w;x)```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196007721-3bb07ebc-e9b0-4767-a082-45f0cde46a2f.png"/>
</p>

We do not really care about the magnitude of the output for now but only the **sign**. Hence, we can describe our linear classifier as a **function** which outputs ```{-1, 0, +1}``` if the the **sign** of ```h(w;x)``` is ```negative```, ```0``` and ```positive``` respectively. 


Some clarifications: 

        - Why do we need to check the sign and not visually see on which side of the classifier our new data falls into?
          Becasue in our example we had only 2 features which could be represented in a 2-d plane. With more features, it will be impossible to visualize since we live in a 3-d world.
          
        - Where was the "bias" in the first example?
          The equation of a line can be expressed as: y = mx + c where c is the bias. m is the gradient which allows tilting of the line while c allows translation in the plane. Since the equation of our linear classifier was y = x, the bias term was 0.
          
        - How do we know our linear classifier is y = x?
          We assumed. Later on we will describe the steps that will allow us to optimize the weights and get us the most efficient classifier.
          
          
### 1.2 The Neuron
The logistic regression model can be described as closest to how a neuron works. I will not dive deep in the function or inner workings of a neuron but just describe some analogies of the neuron and our logistic regression model. Below is an image of the structure of a neuron:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196008465-dd9ffb64-76a6-4469-96f1-9ce0ef56ef7d.jpg"/>
</p>

Let's describe some analogies:

1. It has **many inputs** (dendrites) and **one output** (axon) - just like our logistic model who can have many features (x) and one label column (y).
2. The output of the neuron is either a spike or no spike: 0/1 output. That is, we have some **excitability threshold** which is the minimal value above which a spike is elicited. Hence, a **binary classifier** just as our logistic model.
3. The connections among neurons are realized in the synapses (not shown in picture). The **strengths** of these synapses are analogous to the **linear weights** in our logistic regression model.
4. The output of a neuron is one of the inputs of another neuron which is exactly how we represent a **Neural Network**.


### 1.3 Output of logistic regression
So far we have talked only about the **sign** of the output of our logistic regression model and not about the **magnitude** of that value. When plugging in values for a new data point in h(w;x), the output can be any value from ```-inf``` to ```+inf```. What we want to know now is if a new data point is classified as positive then how positive is it? We want our output to have a clear defined ```range``` and a ```threshold```. In order to achieve the later, we need to pass the output of h(w;x) through a ```non-linear``` function called the **sigmoid** function. The two main reasons for using a non-linear sigmoid function are:

1. A non-linear function allows learning of **complex** features.
2. The output of the sigmoid function which is between ```0``` an ```1``` can be interpreted as **probability**.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196009063-4937e268-533d-4ec9-b612-89668f2c9468.png" width="470" height="180"/>
</p>

The sigmoid function is defined as follows:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196010942-63333281-7bb2-434b-a58d-e8d4f85ff156.png"/>
</p>

where:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196010968-97778ab7-dfd7-423b-ab80-dde06ca9981f.png"/>
</p>

Below we have the graph of the sigmoid function. We notice a few things:

- The output is always between ```0``` and ```1```.
- The graph never reaches ```0``` or ```1``` but is acually asymptotic.
- We can set the midpoint, ```0.5```, as a theshold dividing the two classes.



<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196011261-c1ecb8e4-c5a3-4abf-9145-f475a837463d.png" width="500" height="330"/>
</p>

The outputs can be defined as:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196011934-5a9f4e5a-96a5-4dba-9889-2e52cc1aef65.png" width="310" height="120"/>
</p>

The output of the logistic regression is a number between ```0``` and ```1``` and we can actually interpret it as a **probability**. That is, we can now quantify how positive is a positive point and how negative is a negative point.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196011203-10a6fa86-150b-4759-804c-0af4e7d75a2e.png" width="350" height="100"/>
</p>

In summary:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/196011554-4c70462e-26e9-4fb5-baf8-3081392bca3b.png" width="270" height="100"/>
</p>

We can write a pseudo code for our algorithm:

```python
if P(y=1|x) > P(y=0|x):
    predict class 1
else:
    predict class 0
```

We can also use the threshold as a deciding factor:

```python
if P(y=1|x) > 0.5:
    predict class 1
else:
    predict class 0
```

Ultimately we want the predicted label (<img src="https://latex.codecogs.com/png.image?\dpi{110}\hat{y}" title="https://latex.codecogs.com/png.image?\dpi{110}\hat{y}" />) to be either ```0``` or ```1``` hence, we can use the ```round``` function as follows:

```python
round(0.2) = 0
round(0.5) = 1
round(0.99) = 1 
```

**Does the output of the logistic function really makes sense?**

1. When a point gets futher and further away from the linear classifier, <img src="https://latex.codecogs.com/svg.image?|w^Tx|" title="https://latex.codecogs.com/svg.image?|w^Tx|" /> gets larger.
2. As <img src="https://latex.codecogs.com/svg.image?|w^Tx|\to&space;\infty&space;" title="https://latex.codecogs.com/svg.image?|w^Tx|\to \infty " />, <img src="https://latex.codecogs.com/svg.image?\mathbb{P}(y=1|x)\to&space;" title="https://latex.codecogs.com/svg.image?\mathbb{P}(y=1|x)\to " /> ```0``` or ```1```.
3. Further away, we become more **confident** that a point should be either positive or negative as it is far from the oppositite class.
4. For <img src="https://latex.codecogs.com/svg.image?\mathbb{P}(y=1|x)&space;=&space;\mathbb{P}(y=0|x)&space;=&space;0.5" title="https://latex.codecogs.com/svg.image?\mathbb{P}(y=1|x) = \mathbb{P}(y=0|x) = 0.5" />, we cannot really be sure if the label should be positive or negative.

## 2. Optimal Weights
One very important characteristic of the logistic regression is that the model makes the assumption that our data can be **linearly separated** by a **line** or **plane**. That is, it is a linear model with input features ```x``` weighted by weights ```w```. The sigmoid function squashes the output to the range ```(0,1)```. Now, we want to find the weights ```w```.

### 2.1 Bayes' solution

## References
1. https://scipython.com/blog/logistic-regression-for-image-classification/
