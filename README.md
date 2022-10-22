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

```python
def sigmoid(z):
    return 1/(1 + np.exp(-z))
```

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

### 2.1 Bayes' Solution
The Baye's solution is a special case of the logistic regression where we can find a **closed-form solution** to the problem. For that we need to assume that our data is **Gaussian distributed** with **equal variance**.

Assumptions:

1. Data is from ```2``` classes and they are both ```Gaussian dsitributed```.
2. ```Same covariance``` but ```different mean```.


We will derive our closed form solution using the two equations below:

- **Multivariate Gaussian PDF**

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197076851-b9885a67-f31d-4342-b77a-269ef18d4790.png"/>
</p>

- **Baye's Rule**

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197077224-575c2876-fea1-489d-818c-c8ec99eba764.png"/>
</p>


1. Manipulating Baye's Rule:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197077989-8a8f6d0c-027c-4514-8d48-9b4d15d95b58.png"/>
</p>


2. Dividing by ```p(x|y=1).p(y=1)```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197078423-adb91650-7243-4a4c-8c56-fbb774f28fb6.png"/>
</p>

3. Recall the sigmoid function:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197078643-bfea563c-7a85-4070-975b-71281688ba97.png"/>
</p>

4. By comparing the last two equations:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197079545-d9391333-3cdd-4a68-a427-1d39b94a2e29.png"/>
</p>

5. We manipulate only the LHS and let ```p(y=1) = 1-alpha``` and ```p(y=0) = alpha```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197079783-6c549547-55ed-4ade-badc-6799b207605c.png"/>
</p>

6. We now replace equation (1) in the above equation:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197080522-33434c99-f4ac-4c29-a144-708fb31437fb.png"/>
</p>


7. After some manipulation and using the fact that the **covariance** is ```symmetric``` and ```positive semdefinite```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197080802-4293797f-c81b-480c-bd64-4b7b6dda6f17.png"/>
</p>

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197081159-4bed9c43-869c-45cd-b24f-23fb179f46c2.png"/>
</p>

8. Recall from earlier:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197079545-d9391333-3cdd-4a68-a427-1d39b94a2e29.png"/>
</p>

9. Therefore, our **weights** and **bias**:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197081892-da6466d4-21aa-415a-a0ce-0c973b2a4120.png"/>
</p>

> Note: We can only use this closed-form solution if our data is Gaussian distributed with equal variance which is mostly never in real-life scenarios.

Let us walk through an example. In order to use Bayes' rule we need data to have ediffereent mean but same covariance:

```python
mean1 = [0, 4]
cov1 = [[2.5, 0], [0, 2.5]]  # diagonal covariance

mean2 = [6, 0]
cov2 = [[2.5, 0], [0, 2.5]]  # diagonal covariance
```

Note that it is difficult to have data which is Gaussian distributed with same covariance in real life we need to fabricate this data as shown below:

```python
# X_mat is the design matrix
# first 1000 observations are from one distribution, the second 1000 observations are from another distribution

x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T

x = np.array([x1,x2]).flatten()
y = np.array([y1,y2]).flatten()

X_mat = np.stack([np.ones(2000),x,y],axis=1) #shape = (2000, 3)
```
We now write Bayes' closed-form solution:

```python
weights = np.dot(np.array(mean2).T - np.array(mean1).T,np.linalg.inv(cov1)) # cov1 = cov2
```
From the weights we now calculate the predictions:

```python
# weights from solution
w = [0, w1, w2]

# predicted probabilities from the sigmoid function
z = X_mat.dot(w)

predictions = 1/(1 + np.exp(-z))

# actual classes (0 for Negative, 1 for Positive)
actuals = [0]*1000+[1]*1000
```

Our dataframe is as follows:

```python

       bias	x1	          x2	        prediction	actual
0	1.0	2.553723	1.281897	0.267096	0
1	1.0	1.546036	4.878440	0.999902	0
2	1.0	-1.950017	3.617210	0.999993	0
3	1.0	-2.540598	5.422829	1.000000	0
4	1.0	0.511690	2.190639	0.988328	0
```
We then plot the decision boundary with the scatter plot:

```python
ax = sns.scatterplot('x1','x2',data = results_df,hue = 'prediction',palette='RdBu') # data points

#equation of line: w1x1 + w2x2 + c = 0
# x2 = -w1x1/x2 where c = 0
plt.plot(np.linspace(-5,10,10),(-w1*np.linspace(-5,10,10)/w2),linestyle='--',color='k',alpha=0.25) # decision boundary
```

Note that we have also mapped the probability of our predictions to color. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197355986-11cb2bad-198a-4cb6-ad74-44bc578b4934.png"/>
</p>

This shows that data points that are far from the decision boundary has higher probabilites of coming from a certain class, thus we are more certain they are correctly predicted. Data points that are close to the decision boundary have less probabilites hence, have higher chances of wrong predictions.

### 2.2 Error Function

#### 2.2.1 Sum of Squares Error(SSE)
So we cannot use the Bayes' solution except under certain assumptions hence, our next best approach will be to find a closed-form solution from an Error function similar to the Linear Regression. Recall our cost function with linear regression were as follows:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197217301-82e46e1d-1e60-4cfe-b8e6-9bdd9add669f.png"/>
</p>

We can solve this to find a closed-form solution:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197217520-b63739b1-41fa-4eba-870c-c7f5f5bdb5d7.png"/>
</p>

Note:

- We used the Sum of Squares Error(SSE) because we assumed our residuals are Gaussian dsitributed.
- However, the logistic error CANNOT be Gaussian dsitributed as the output and target can only be 0 or 1.

We cannot use SSE for the Logistic Regression so we need another error function. In summary, we want our error function to be:

1. ```0``` when we have no error
2. Get larger when the more incorrect we are.

#### 2.2.1 Cross Entropy Error Function

The Cross-Entropy function is defined as follows:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197266812-c014b041-eb16-48d9-b7fa-1716478b5318.png"/>
</p>

Note that the target can either be ```0``` or ```1``` and not both at the same time:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197267180-ed8d28e2-968d-44ca-a8ed-74d6ee5e6ab7.png"/>
</p>


Let's see if we satisfy the two points above for our new error function:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197221465-60db9386-726c-426d-b566-a6a6291c7135.png" width="400" height="200"/>
</p>

The equation above was for finding the error for one datapoint but we want to find the errors for all the datapoints in our dataset, hence:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197266713-c52e707e-a0bd-4249-9586-8af089961cc8.png"/>
</p>


### 2.3 Maximum Likelihood: Bernoulli
We can also arrive at the same equation using the Maximum Likelihood Estimation for a Bernoulli Distribution. Note that we use Bernoulli because our targets and label are either 0 or 1 only.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197224540-2550007a-ec8b-41f9-95f9-05a2531b437f.png"/>
</p>

The likelihood function is defined as:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197225224-c797740b-3be3-4c8b-85e8-eb6b47cc7953.png"/>
</p>

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197225134-696f12e0-1e64-428b-8fea-9dddf2e4b31b.png"/>
</p>

In our case we want to find how likely we are to observe a certain set of input ```X``` (distribution) and targets ```y``` when given parameter ```w```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197267528-2a80accc-45d6-4468-b4e5-1e3092a421a9.png"/>
</p>

We now want the Log Likelihood:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197266659-78cadfdf-741b-4eb7-9ede-35f43fb5fdea.png"/>
</p>

To conclude: 

> Maximixing Log-Likelihood = Minimizing Cross-Entropy Error



### 2.4 Closed-Form Solution
In order to find the closed-form solution we need to find the derivation of the cross-entropy function and set it to ```0```. We start by finding the derivative:

Recall from earlier the formula for the sigmoid function:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197268174-57c1c765-7096-4ed3-b2c5-f885ea52857d.png"/>
</p>

We replace ```y_hat``` in the cross-entropy function and transform our cost function in vector form:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197268520-7ab05526-3dbb-4116-8b3c-ff2cee51ef85.png"/>
</p>

We need to maximize the equation above. Now we differentiate w.r.t w:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197268964-016c484b-2aee-4d7d-8793-c8f5b10916e7.png"/>
</p>

We **CANNOT** set the equation above to ```0``` and solve for ```w```! Hence, we do **NOT** have a closed-form function for Logistic Regression using the Cross-Entropy function.


### 2.5 Gradient Descent
If we cannot solve using a closed-form solution then we need another optimization algorithm: **Gradient Descent**. We start with some initial values for the weights then we want to reach the global minima of our error function curve where the error is minimum and the weights are optimal. 

```python
w --> some ranfom initial values
while not converged:
    w ---> w - η.dJ/dW // η = learning rate
```

An image showing the same below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197270401-daf4ca6e-79b3-415f-b9d1-fbefcd04661a.png" width="500" height="380"/>
</p>

We start by iterating our objective function:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353229-c4b98395-6324-412e-b0d8-5fda8628aa78.png"/>
</p>

We need to use chain rule to find ```dJ/dw```:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353380-94aca4e2-4919-4702-b506-37d0920dc934.png"/>
</p>

1. First component:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353504-55beb0df-65e6-4923-a78e-0b33a357b904.png"/>
</p>

2. Second component:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353694-b9c0ac3b-4487-426b-89b8-63f6cc3a9ac5.png"/>
</p>

3. Third component:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353768-262251f9-8575-4aab-832f-bcd6b2e88021.png"/>
</p>

Putting all together and formatting in vector form:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353867-13fc9790-3a00-4650-8d86-7f4dc9aa011f.png"/>
</p>

Hence the equation for the gradient descent becomes:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/197353961-23c4103b-2317-4622-b478-aad591387f25.png"/>
</p>

In summary:

1. If we assume our data is Gaussian distributed with equal covariance then we can use the **Bayes' closed-form solution**.
2. However, Gaussian assumtion may not be true so we **maximize** our **log-likelihhod** OR **minimize** the **Cross-Entropy** function. But there is NO closed-form solution.
3. We thus use a more general optimization method: **Gradient Descent**.


## 3. Regularization







## References
1. https://scipython.com/blog/logistic-regression-for-image-classification/
