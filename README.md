# Cleaning Tables with Logistic Regression

## Abstract

## Plan of Action

1. Linear Classification
    - Equation of a linear classifier
    - The Neuron
    - Output of logistic regression

2. Optimal Weights
    - Bayes Solution
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

5. Clean Tables Recognition

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



        - Why do we need to check the sign and not visually see on which side of the classifier our new data falls into?
          Becasue in our example we had only 2 features which could be represented in a 2-d plane. With more features, it will be impossible to visualize since we live in a 3-d world.
          
        - Where was the "bias" in the first example?
          




## References
1. https://scipython.com/blog/logistic-regression-for-image-classification/
