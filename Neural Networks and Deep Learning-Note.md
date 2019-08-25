# Neural Networks and Deep Learning

- Neural networks is a beautiful biologically inspired programming paradigm which enables to learn from observational data.
- Deep learning is a powerful set of techniques for learning in neural networks.

## Chapter 1 Using Neural nets to recognize handwritten digits

### Percetrons

Perceptrons were developed in the 1950s and 1960s by the scientist Frank Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts. Today, it's more common to use other models of artificial neurons, and the main neuron model used is one called the sigmoid neuron.

So how do perceptrons work? A perceptron takes several binary inputs, x1,x2,…, and produces a single binary output:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz0.png)

The neuron's output, 0 or 1, is determined by whether the weighted sum $\sum_j \omega_j x_j$ is less than or greater than some threshold value.

$$
output = \left\{
\begin{array}{ll}
0 \qquad if \sum_j \omega_j x_j \le threshold \\
1 \qquad if \sum_j \omega_j x_j > threshold
\end{array}
\right.
$$

A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence. Using the bias instead of the threshold, the perceptron rule can be rewritten:

$$
output = \left\{
\begin{array}{ll}
0 \qquad if \omega_j \cdot x_j + b \le 0 \\
1 \qquad if \omega_j \cdot x_j + b > 0
\end{array}
\right.
$$

I've described perceptrons as a method for weighing evidence to make decisions. Another way perceptrons can be used is to compute the elementary logical functions we usually think of as underlying computation, functions such as AND, OR, and NAND. For example, suppose we have a perceptron with two inputs, each with weight −2, and an overall bias of 3. Here's our perceptron:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz2.png)

Then we see that input 00 produces output 1, since $(−2)∗0+(−2)∗0+3=3$ is positive. Here, I've introduced the ∗ symbol to make the multiplications explicit. Similar calculations show that the inputs 01 and 10 produce output 1. But the input 11 produces output 0, since $(−2)∗1+(−2)∗1+3=−1$ is negative. And so our perceptron implements a NAND gate!

In fact, we can use networks of perceptrons to compute any logical function at all.

### Sigmoid Neurons

$$
\sigma(z) \equiv \frac{1}{1+e^{-z}}
$$

where $z \equiv \omega \cdot x + b$.

### The Architecture of Neural Networks

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz10.png)

The leftmost layer in this network is called the input layer, and the neurons within the layer are called input neurons. The rightmost or output layer contains the output neurons, or, as in this case, a single output neuron. The middle layer is called a hidden layer, since the neurons in this layer are neither inputs nor outputs.

The network above has just a single hidden layer, but some networks have multiple hidden layers. For example, the following four-layer network has two hidden layers:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

Up to now, we've been discussing neural networks where the output from one layer is used as input to the next layer. Such networks are called **feedforward neural networks**. This means there are no loops in the network - information is always fed forward, never fed back.

However, there are other models of artificial neural networks in which feedback loops are possible. These models are called [recurrent neural networks](http://en.wikipedia.org/wiki/Recurrent_neural_network). The idea in these models is to have neurons which fire for some limited duration of time, before becoming quiescent. That firing can stimulate other neurons, which may fire a little while later, also for a limited duration. That causes still more neurons to fire, and so over time we get a cascade of neurons firing. Loops don't cause problems in such a model, since a neuron's output only affects its input at some later time, not instantaneously.

### A Simple Network to Classify Handwritten Digits

First, we'd like a way of breaking an image containing many digits into a sequence of separate images, each containing a single digit. Once the image has been segmented, the program then needs to classify each individual digit.

We'll focus on writing a program to solve the second problem, that is, classifying individual digits. We do this because it turns out that the segmentation problem is not so difficult to solve, once you have a good way of classifying individual digits. There are many approaches to solving the segmentation problem. One approach is to trial many different ways of segmenting the image, using the individual digit classifier to score each trial segmentation. A trial segmentation gets a high score if the individual digit classifier is confident of its classification in all segments, and a low score if the classifier is having a lot of trouble in one or more segments. The idea is that if the classifier is having trouble somewhere, then it's probably having trouble because the segmentation has been chosen incorrectly. This idea and other variations can be used to solve the segmentation problem quite well. So instead of worrying about segmentation we'll concentrate on developing a neural network which can solve the more interesting and difficult problem, namely, recognizing individual handwritten digits.

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz12.png)

Learning With Gradient Descent

We'll use the [MNIST](http://yann.lecun.com/exdb/mnist/) data set, which contains tens of thousands of scanned images of handwritten digits, together with their correct classifications. MNIST's name comes from the fact that it is a modified subset of two data sets collected by [NIST](http://en.wikipedia.org/wiki/National_Institute_of_Standards_and_Technology), the United States' National Institute of Standards and Technology. Here's a few images from MNIST:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/digits_separate.png)

The MNIST data comes in two parts. The first part contains 60,000 images to be used as training data. The images are greyscale and 28 by 28 pixels in size. The second part of the MNIST data set is 10,000 images to be used as test data.

To quantify how well we're achieving this goal we define a cost function:

$$
C(\omega, b) \equiv \frac{1}{2n}\sum_x\|y(x) - a\|^2
$$

Here, w denotes the collection of all weights in the network, b all the biases, n is the total number of training inputs, a is the vector of outputs from the network when x is input, and the sum is over all training inputs, x.

Why introduce the quadratic cost? After all, aren't we primarily interested in the number of images correctly classified by the network? Why not try to maximize that number directly, rather than minimizing a proxy measure like the quadratic cost? The problem with that is that **the number of images correctly classified is not a smooth function of the weights and biases in the network**. That makes it difficult to figure out how to change the weights and biases to get improved performance.  If we instead use a smooth cost function like the quadratic cost it turns out to be easy to figure out how to make small changes in the weights and biases so as to get an improvement in the cost.

Writing out the gradient descent update rule in terms of components, we have:

$$
\begin{array}{ll}
\omega_k \rightarrow\omega^{\prime}_k &=& \omega_k - \eta\frac{\partial C}{\partial\omega_k} \\
b_l \rightarrow b^{\prime}_l &=& b_k - \eta\frac{\partial C}{\partial b_l}
\end{array}
$$

Notice that this cost function has the form $C=\frac{1}{n}\sum_x C_x$, that is, it's an average over costs $C_x \equiv \frac{\|y(x)−a\|^2}{2}$ for individual training examples. In practice, to compute the gradient $\nabla C$ we need to compute the gradients $\nabla C_x$ separately for each training input, x, and then average them, $\nabla C = \frac{1}{n}\sum_x \nabla C_x$. Unfortunately, when the number of training inputs is very large this can take a long time, and learning thus occurs slowly.

An idea called **stochastic gradient descent** can be used to speed up learning. The idea is to estimate the gradient $\nabla C$ by computing $\nabla C_x$ for a small sample of randomly chosen training inputs.

To connect this explicitly to learning in neural networks, suppose wk and bl denote the weights and biases in our neural network. Then stochastic gradient descent works by picking out a randomly chosen mini-batch of training inputs, and training with those,

$$
\begin{array}{ll}
\omega_k \rightarrow\omega^{\prime}_k &=& \omega_k - \frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial\omega_k} \\
b_l \rightarrow b^{\prime}_l &=& b_k - \frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial b_l}
\end{array}
$$

where the sums are over all the training examples $X_j$ in the current mini-batch. Then we pick out another randomly chosen mini-batch and train with those. And so on, until we've exhausted the training inputs, which is said to complete an epoch of training. At that point we start over with a new training epoch.

### Inplementing Our Neural Network to Classify Digits

```bash
git clone https://github.com/mnielsen/neural-networks-and-deep-learning.git
```

We execute the following commands in a Python shell,

```ipython
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
>>> import network
>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

Let's try using one of the best known algorithms, the **support vector machine** or **SVM**. We'll use a Python library called [scikit-learn](http://scikit-learn.org/stable/), which provides a simple Python interface to a fast C-based library for SVMs known as [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).

If we run scikit-learn's SVM classifier using the default settings, then it gets 9,435 of 10,000 test images correct. (The code is available [here](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_svm.py).) That's a big improvement over our naive approach of classifying an image based on how dark it is. Indeed, it means that the SVM is performing roughly as well as our neural networks, just a little worse. In later chapters we'll introduce new techniques that enable us to improve our neural networks so that they perform much better than the SVM.

That's not the end of the story, however. The 9,435 of 10,000 result is for scikit-learn's default settings for SVMs. SVMs have a number of tunable parameters, and it's possible to search for parameters which improve this out-of-the-box performance.  [This blog post](http://peekaboo-vision.blogspot.de/2010/09/mnist-for-ever.html) by [Andreas Mueller](http://peekaboo-vision.blogspot.ca/) shows that with some work optimizing the SVM's parameters it's possible to get the performance up above 98.5 percent accuracy. In other words, a well-tuned SVM only makes an error on about one digit in 70. That's pretty good! Can neural networks do better?

In fact, they can. At present, well-designed neural networks outperform every other technique for solving MNIST, including SVMs. The current (2013) record is classifying 9,979 of 10,000 images correctly. This was done by [Li Wan](http://www.cs.nyu.edu/~wanli/), [Matthew Zeiler](http://www.matthewzeiler.com/), Sixin Zhang, [Yann LeCun](http://yann.lecun.com/), and [Rob Fergus](http://cs.nyu.edu/~fergus/pmwiki/pmwiki.php). At that level the performance is close to human-equivalent, and is arguably better, since quite a few of the MNIST images are difficult even for humans to recognize with confidence, for example:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/mnist_really_bad_images.png)

In some sense, the moral of both our results and those in more sophisticated papers, is that for some problems:

> sophisticated algorithm ≤ simple learning algorithm + good training data.

## Chapter 2. How The Backpropagation Algorithms Works

The backpropagation algorithm was originally introduced in the 1970s, but its importance wasn't fully appreciated until a [famous 1986 paper](http://www.nature.com/nature/journal/v323/n6088/pdf/323533a0.pdf) by [David Rumelhart](http://en.wikipedia.org/wiki/David_Rumelhart), [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/), and [Ronald Williams](http://en.wikipedia.org/wiki/Ronald_J._Williams). That paper describes several neural networks where backpropagation works far faster than earlier approaches to learning, making it possible to use neural nets to solve problems which had previously been insoluble. Today, the backpropagation algorithm is the workhorse of learning in neural networks.

### Matrix-based Approach to Computing the Output from A Neural Network

Let's begin with a notation which lets us refer to weights in the network in an unambiguous way. We'll use $\omega^l_{jk}$ to denote the weight for the connection from the $k^{th}$ neuron in the $(l−1)^{th}$ layer to the $j^{th}$ neuron in the lth layer. So, for example, the diagram below shows the weight on a connection from the fourth neuron in the second layer to the second neuron in the third layer of a network:

![IamgeNA](http://neuralnetworksanddeeplearning.com/images/tikz16.png)

$$
a^l = \sigma(\omega^l a^{l-1} + b^l)
$$

or we may right it as:

$$
a^l = \sigma((\omega^l)^T a^{l-1} + b^l)
$$

if we want $\omega$'s column to be the multiplyer from level (l-1) to one nuron in level l.

We compute the intermediate quantity $z^l \equiv w^l a^{l−1} + b^l$ along the way. This quantity turns out to be useful enough to be worth naming: we call $z^l$ the weighted input to the neurons in layer l.

### Two Assumptions We Need About The Cost Function

For backpropagation to work we need to make two main assumptions about the form of the cost function.

The first assumption we need is that the cost function can be written as an average $C = \frac{1}{n}\sum_x C_x$ over cost functions $C_x$ for individual training examples, x.

The second assumption we make about the cost is that it can be written as a function of the outputs from the neural network:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz18.png)

For example, the quadratic cost function satisfies this requirement, since the quadratic cost for a single training example x may be written as:

$$
C = \frac{1}{2}\|y-a^L\|^2=\frac{1}{2}\sum_j(y_j - a_j^L)^2
$$

and thus is a function of the output activations.

### The Hadamard product, $s\odot t$

Suppose s and t are two vectors of the same dimension. Then we use $s\odot t$ to denote the elementwise product of the two vectors. This kind of elementwise multiplication is sometimes called the **Hadamard product** or **Schur product**.

### Four Fundamental Equations Behind Backpropogation

First introduce an intermediate quantity, $\delta^l_j$, which we call the error in the $j^{th}$ neuron in the $l^{th}$ layer. Backpropagation will give us a procedure to compute the error $\delta^l_j$, and then will relate $\delta^l_j$ to $\partial C / \partial\omega^l_{jk}$ and $\partial C / \partial b^l_j$.

Define the error $\delta^l_j$ of neuron j in layer l by:

$$
\delta^l_j \equiv \frac{\partial C}{\partial z^l_j}
$$

As per our usual conventions, we use $\delta^l$ to denote the vector of errors associated with layer l. Backpropagation will give us a way of computing δl for every layer, and then relating those errors to the quantities of real interest, $\partial C / \partial\omega^l_{jk}$ and $\partial C / \partial b^l_j$.

![IamgeNA](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

### Proof of The Four Fundamental Equations

### Backpropagation Algorithm

1. **Input x**: Set the corresponding activation a^1 for the input layer.
2. **Feedforward**: For each l=2,3,…,L compute $z^l=w^la^{l−1}+b^l$ and $a^l=\sigma(z^l)$
3. **Output error $\delta^L$**: Compute the vector $\delta^L=\nabla_a C\odot \sigma^{\prime}(z^L)$
4. **Backpropagate the error**: For each l=L−1,L−2,…,2 compute $\delta^l=((w^{l+1})^T\delta^{l+1})⊙\delta^{\prime}(z^l)$
5. **Output**: The gradient of the cost function is given by $\frac{\partial C}{\partial\omega^l_{jk}}=a^{l−1}_k\delta^l_j$ and $\frac{\partial C}{\partial b^l_j}=\delta^l_j$.

In practice, it's common to combine backpropagation with a learning algorithm such as stochastic gradient descent, in which we compute the gradient for many training examples. In particular, given a mini-batch of m training examples, the following algorithm applies a gradient descent learning step based on that mini-batch:

1. **Input a set of training examples**.
2. **For each training example x**: Set the corresponding input activation $a^{x,1}$, and perform the following steps:
    - **Feedforward**: For each l=2,3,…,L compute $z^{x,l}=w^la^{x,l−1}+b^l$ and $a^{x,l}=\sigma\left(z^{x,l}\right)$.
    - **Output error $\delta^{x,L}$**: Compute the vector $\delta^{x,L}=\nabla_a C_x\odot \sigma^{\prime}\left(z^{x,L}\right)$
    - **Backpropagate the error**: For each l=L−1,L−2,…,2 compute $\delta^{x,l}=((w^{l+1})^T\delta^{x,l+1})\odot\sigma^{\prime}(z^{x,l})$
3. **Gradient descent**: For each l=L,L−1,…,2 update the weights according to the rule $\omega^l \rightarrow \omega^l−\frac{\eta}{m}\sum_x \delta^{x,l}(a^{x,l−1})^T$, and the biases according to the rule $b^l\rightarrow b^l − \frac{\eta}{m}\sum_x\delta^{x,l}$.

Of course, to implement stochastic gradient descent in practice you also need an outer loop generating mini-batches of training examples, and an outer loop stepping through multiple epochs of training. I've omitted those for simplicity.

### Problem

**Fully matrix-based approach to backpropagation over a mini-batch**. Our implementation of stochastic gradient descent loops over training examples in a mini-batch. It's possible to modify the backpropagation algorithm so that it computes the gradients for all training examples in a mini-batch simultaneously. The idea is that instead of beginning with a single input vector, x, we can begin with a matrix $X=[x_1, x_2,\dots, x_m]$ whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms, and applying the sigmoid function everywhere. We backpropagate along similar lines. Explicitly write out pseudocode for this approach to the backpropagation algorithm. Modify network.py so that it uses this fully matrix-based approach. The advantage of this approach is that it takes full advantage of modern libraries for linear algebra. As a result it can be quite a bit faster than looping over the mini-batch.

### In what sense is backpropagation a fast algorithm

Compare backpropagation with computing $\partial C / \partial \omega_j$ directly:

$$
\frac{\partial C}{\partial \omega_j} \approx \frac{C(\omega+\varepsilon e_j) - C(\omega)}{\varepsilon}
$$

This approach looks very promising. It's simple conceptually, and extremely easy to implement, using just a few lines of code. Certainly, it looks much more promising than the idea of using the chain rule to compute the gradient!

What's clever about backpropagation is that it enables us to simultaneously compute all the partial derivatives ∂C/∂wj using just one forward pass through the network, followed by one backward pass through the network. Roughly speaking, the computational cost of the backward pass is about the same as the forward pass. And so the total cost of backpropagation is roughly the same as making just two forward passes through the network.

## Chapter 3. Improving the Way Neural Networks Learn

The techniques we'll develop in this chapter include: a better choice of cost function, known as the **cross-entropy** cost function; four so-called **"regularization" methods** (L1 and L2 regularization, dropout, and artificial expansion of the training data), which make our networks better at generalizing beyond the training data; a better method for **initializing the weights** in the network; and **a set of heuristics to help choose good hyper-parameters** for the network. I'll also overview **several other techniques** in less depth. The discussions are largely independent of one another, and so you may jump ahead if you wish. We'll also implement many of the techniques in running code, and use them to improve the results obtained on the handwriting classification problem.

Of course, we're only covering a few of the many, many techniques which have been developed for use in neural nets. The philosophy is that the best entree to the plethora of available techniques is in-depth study of a few of the most important. Mastering those important techniques is not just useful in its own right, but will also deepen your understanding of what problems can arise when you use neural networks. That will leave you well prepared to quickly pick up other techniques, as you need them.

### The cross-entropy cost function

**Think**: Why quadratic function will learn slowly?

If $C = \frac{(y - a)^2}{2}$, recall that $a = \sigma(z)$, then we will have:

$$
\begin{aligned}
\frac{\partial C}{\partial \omega} &= (a - y)\sigma^{\prime}(z)x \\
\frac{\partial C}{\partial b} &= (a - y)\sigma^{\prime}(z)
\end{aligned}
$$

How can we address the learning slowdown? It turns out that we can solve the problem by replacing the quadratic cost with a different cost function, known as the cross-entropy.

$$
C = -\frac{1}{n}\sum_x[y\ln a + (1-y)\ln(1-a)]
$$

The cross-entropy is positive, and tends toward zero as the neuron gets better at computing the desired output, y, for all training inputs, x. These are both properties we'd intuitively expect for a cost function. Indeed, both properties are also satisfied by the quadratic cost. So that's good news for the cross-entropy. But the cross-entropy cost function has the benefit that, unlike the quadratic cost, **it avoids the problem of learning slowing down**.

$$
\begin{aligned}
\frac{\partial C}{\partial\omega_j} &= - \frac{1}{n}\sum_x\left(\frac{y}{\sigma(z)} - \frac{1-y}{1-\sigma(z)}\right)\frac{\partial \sigma}{\partial \omega_j} \\
&= - \frac{1}{n}\sum_x\left(\frac{y}{\sigma(z)} - \frac{1-y}{1-\sigma(z)}\right)\sigma^{\prime}(z)x_j \\
&= \frac{1}{n}\sum_x\frac{\sigma^{\prime}(z)x_j}{\sigma(z)(1-\sigma(z))}(\sigma(z) - y)
\end{aligned}
$$

and also we can have that

$$
\sigma^{\prime}(z) = \sigma(z)(1-\sigma(z))
$$

the above two equations will siplify to become:

$$
\frac{\partial C}{\partial \omega_j} = \frac{1}{n}\sum_x x_j(\sigma(z) - y)
$$

In a similar way, we can compute the partial derivative for the bias. I won't go through all the details again, but you can easily verify that:

$$
\frac{\partial C}{\partial b} = \frac{1}{n}\sum_x (\sigma(z) - y)
$$

This avoids the learning slowdown caused by the $\sigma^{\prime}(z)$ term in the analogous equation for the quadratic cost function.

It's easy to generalize the cross-entropy to many-neuron multi-layer networks. In particular, suppose y=y1,y2,… are the desired values at the output neurons, i.e., the neurons in the final layer, while aL1,aL2,… are the actual output values. Then we define the cross-entropy by

$$
C = -\frac{1}{n}\sum_x\sum_y\left[y_j\ln a^L_j + (1-y_j)\ln(1-a^L_j)\right].
$$

This is the same as our earlier expression, except now we've got the $\sum_j$ summing over all the output neurons.

The term "cross-entropy" is common to define the cross-entropy for two probability distributions, $p_j$ and $q_j$, as $\sum_j p_j\ln q_j$. However, when we have many sigmoid neurons in the final layer, the vector $a^L_j$ of activations don't usually form a probability distribution. As a result, a definition like $\sum_j p_j\ln q_j$ doesn't even make sense, since we're not working with probability distributions. Instead, you can think of it as a summed set of per-neuron cross-entropies, with the activation of each neuron being interpreted as part of a two-element probability distribution. In this sense, it is a generalization of the cross-entropy for probability distributions.

When should we use the cross-entropy instead of the quadratic cost? In fact, the cross-entropy is nearly always the better choice, provided the output neurons are sigmoid neurons. Think why.

#### Problems

- **Many-layer multi-neuron networks** Show that for the quadratic cost the partial derivative with respect to weights in the output layer is

$$
\frac{\partial C}{\partial\omega_{jk}^L} = \frac{1}{n}\sum_xa^{L-1}_k(a_j^L - y_j)\sigma^{\prime}(z_j^L)
$$

The term $\sigma^{\prime}(z^L_j)$ causes a learning slowdown whenever an output neuron saturates on the wrong value. Show that for the cross-entropy cost the output error $\delta^L$ for a single training example x is given by

$$
\delta^L = a^L− y
$$

Use this expression to show that the partial derivative with respect to the weights in the output layer is given by

$$
\frac{\partial C}{\partial \omega_{jk}^L} = \frac{1}{n}\sum_x a^{L−1}_k(a^L_j−y_j)
$$

The $\sigma^{\prime}(z^L_j)$ term has vanished, and so the cross-entropy avoids the problem of learning slowdown, not just when used with a single neuron, but also in many-layer multi-neuron networks. A simple variation on this analysis holds also for the biases.

- **Using the quadratic cost when we have linear neurons in the output layer** Suppose that we have a many-layer multi-neuron network and all the neurons in the final layer are linear neurons, meaning that the sigmoid activation function is not applied, and the outputs are simply $a^L_j=z^L_j$. Show that if we use the quadratic cost function then the output error $\delta^L$ for a single training example x is given by

$$
\delta^L = a^L−y
$$

Similarly to the previous problem, use this expression to show that the partial derivatives with respect to the weights and biases in the output layer are given by

$$
\begin{aligned}
\frac{\partial C}{\partial \omega_{jk}^L} &= \frac{1}{n}\sum_x a^{L-1}_k(a_j^L - y_j) \\
\frac{\partial C}{\partial b_j^L} &= \frac{1}{n} \sum_x(a_j^L - y_j)
\end{aligned}
$$

This shows that if the output neurons are linear neurons then the quadratic cost will not give rise to any problems with a learning slowdown. In this case the quadratic cost is, in fact, an appropriate cost function to use.

So why so much focus on cross-entropy? Part of the reason is that the cross-entropy is a widely-used cost function, and so is worth understanding well. But the more important reason is that neuron saturation is an important problem in neural nets, a problem we'll return to repeatedly. Disscussing the cross-entropy at length is a good laboratory to begin understanding neuron saturation and how it may be addressed.

#### What does the cross-entropy mean? Where does it come from

We have discoverd that the learning slowdown is caused by the $\sigma^{\prime}(z)$ term in $\partial C / \partial \omega$ and $\partial C / \partial b$. We wonder if it's possible to choose a cost function so that the $\sigma^{\prime}(z)$ term disappeared. In that case, the cost $C = C_x$ for a single training example x would satisfy

$$
\begin{aligned}
\frac{\partial C}{\partial \omega_j} &= x_j(a-y) \\
\frac{\partial C}{\partial b} &= (a-y)
\end{aligned}
$$

If we could choose the cost function to make these equations true, then they would capture in a simple way the intuition that the greater the initial error, the faster the neuron learns. They'd also eliminate the problem of a learning slowdown.

Note that from the chain rule we have:

$$
\frac{\partial C}{\partial b} = \frac{\partial C}{\partial a}\sigma^{\prime}(z)
$$

Substitute $\sigma^{\prime}(z) = \sigma(z)(1−\sigma(z))=a(1−a)$, integrate with respect to a, we have:

$$
C = -[y\ln a + (1-y) \ln (1-a)] + \mathbb{constant}
$$

for some constant of integration. This is the contribution to the cost from a single training example, x. To get the full cost function we must average over training examples, obtaining

$$
C = -\frac{1}{n}\sum_x[y\ln a + (1-y) \ln (1-a)] + \mathbb{constant}
$$

where the constant here is the average of the individual constants for each training example.

#### Softmax

The idea of softmax is to define a new type of output layer for our neural networks. It begins in the same way as with a sigmoid layer, by forming the weighted inputs $z^L_j = \sum_k \omega^L_{jk}a^{L−1}_k + b^L_j$. However, we don't apply the sigmoid function to get the output. Instead, in a softmax layer we apply the so-called **softmax function** to the $z^L_j$. According to this function, the activation $a^L_j$ of the jth output neuron is

$$
a^L_j = \frac{e^{z^L_j}}{\sum_k e^{z^L_k}}
$$

where in the denominator we sum over all the output neurons.

$$
\sum_j a^L_j = \frac{\sum_j e^{z^L_j}}{\sum_k e^{z^L_k}} = 1
$$

we see that the output from the softmax layer is a set of positive numbers which sum up to 1. In other words, the output from the softmax layer can be thought of as a probability distribution.

The fact that a softmax layer outputs a probability distribution is rather pleasing. In many problems it's convenient to be able to interpret the output activation $a^L_j$ as the network's estimate of the probability that the correct output is j. So, for instance, in the MNIST classification problem, we can interpret $a^L_j$ as the network's estimated probability that the correct digit classification is j.

By contrast, if the output layer was a sigmoid layer, then we certainly couldn't assume that the activations formed a probability distribution.

**We will see how a softmax layer lets us address the learning slowdown problem**. Let's define the log-likelihood cost function. We'll use x to denote a training input to the network, and y to denote the corresponding desired output. Then the log-likelihood cost associated to this training input is

$$
C \equiv -\ln a^L_y
$$

What about the learning slowdown problem? To analyze that, recall that the key to the learning slowdown is the behaviour of the quantities $\partial C/\partial \omega^L_{jk}$ and $\partial C/\partial b^L_j$.

$$
\begin{aligned}
\frac{\partial C}{\partial b^L_j} &= a^l_j - y_j \\
\frac{\partial C}{\partial \omega^L_{jk}} &= a^{L-1}_k(a^l_j - y_j)
\end{aligned}
$$

These equations are the same as the analogous expressions obtained in our earlier analysis of the cross-entropy. And, just as in the earlier analysis, these expressions ensure that we will not encounter a learning slowdown. In fact, it's useful to think of a softmax output layer with log-likelihood cost as being quite similar to a sigmoid output layer with cross-entropy cost.

Given this similarity, should you use a sigmoid output layer and cross-entropy, or a softmax output layer and log-likelihood? In fact, in many situations both approaches work well. As a more general point of principle, **softmax plus log-likelihood is worth using whenever you want to interpret the output activations as probabilities**.

#### Problems about softmax

- **Backpropagation with softmax and the log-likelihood cost** To apply the backpropagation algorithm to a network with a softmax layer we need to figure out an expression for the error $\partial^L_j \equiv \partial C/\partial z^L_j$ in the final layer. Show that a suitable expression is:

$$
\delta^L_j = a^L_j − y_j
$$

Using this expression we can apply the backpropagation algorithm to a network using a softmax output layer and the log-likelihood cost.

### Overfitting and regularization

Overfitting is a major problem in neural networks. This is especially true in modern networks, which often have very large numbers of weights and biases. To train effectively, we need a way of detecting when overfitting is going on, so we don't overtrain. And we'd like to have techniques for reducing the effects of overfitting.

The obvious way to detect overfitting is keeping track of accuracy on the test data as our network trains. If we see that the accuracy on the test data is no longer improving, then we should stop training. Of course, strictly speaking, this is not necessarily a sign of overfitting. It might be that accuracy on the test data and the training data both stop improving at the same time. Still, adopting this strategy will prevent overfitting.

We choose *test_data* to train the model, *validation_data* to figure out the hyper-parameters, and do a final evaluation of accuracy using the *test_data*. That gives us confidence that our results on the *test_data* are a true measure of how well our neural network generalizes.  To put it another way, you can think of the *validation_data* as a type of training data that helps us learn good hyper-parameters. This approach to finding good hyper-parameters is sometimes known as the **hold out method**.

#### Regularization

The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term. Here's the regularized cross-entropy:

$$
C=−\frac{1}{n}\sum_{x_j}\left[y_j\ln a^L_j + (1−y_j)\ln(1−a^L_j)\right] + \frac{\lambda}{2n}\sum_{\omega}\omega^2
$$

The first term is just the usual expression for the cross-entropy. This is scaled by a factor $\lambda/2n$, where $\lambda > 0$ is known as the regularization parameter, and n is, as usual, the size of our training set.

Of course, it's possible to regularize other cost functions, such as the quadratic cost. This can be done in a similar way:

$$
C=\frac{1}{2n}\sum_x\|y−a^L\|^2 + \frac{\lambda}{2n}\sum_{\omega}\omega^2
$$

In both cases we can write the regularized cost function as

$$
C = C_0 + \frac{\lambda}{2n}\sum_{\omega}\omega^2
$$

where $C_0$ is the original, unregularized cost function.

The learning rule for the weights becomes:

$$
\begin{aligned}
\omega &\rightarrow \omega - \eta\frac{\partial C_0}{\partial \omega} - \frac{\eta\lambda}{n}\omega \\
&=\left(1 - \frac{\eta\lambda}{n}\right)\omega - \eta\frac{\partial C_0}{\partial \omega}
\end{aligned}
$$

Well, just as in unregularized stochastic gradient descent, we can estimate $\partial C_0/\partial \omega$ by averaging over a mini-batch of m training examples. Thus the regularized learning rule for stochastic gradient descent becomes

$$
\omega \rightarrow \left(1 - \frac{\eta\lambda}{n}\right)\omega - \frac{\eta}{m}\sum_x \frac{\partial C_x}{\partial \omega}
$$

where the sum is over training examples x in the mini-batch, and $C_x$ is the (unregularized) cost for each training example. Finally, the regularized learning rule for the biases is exactly the same as in the unregularized.

$$
b \rightarrow b - \frac{\eta}{m}\sum_x \frac{\partial C_x}{\partial b}
$$

where the sum is over training examples x in the mini-batch.

Regularization can be viewed as something of a pludge. While it often helps, we don't have an entirely satisfactory systematic understading of what's going on, merely incomplete heuristics and rules of thumb.

#### Other techniques for regularization

There are many regularization techniques other than L2 regularization. In this section I briefly describe three other approaches to reducing overfitting: L1 regularization, dropout, and artificially increasing the training set size.

**L1 regularization:** In this approach we modify the unregularized cost function by adding the sum of the absolute values of the weights:

$$
C = C_0 + \frac{\lambda}{n}\sum_{\omega}|w|
$$

Using this expression, we can easily modify backpropagation to do stochastic gradient descent using L1 regularization. The resulting update rule for an L1 regularized network is

$$
\omega^{\prime} \rightarrow = \omega\left(1 − \frac{\eta\lambda}{n}\right)\mathbb{sgn}(\omega) − \eta\frac{\partial C_0}{\partial\omega}
$$

where, as per usual, we can estimate $\partial C_0/\partial \omega$ using a mini-batch average, if we wish.

In both L1 and L2 regularizations is to shrink the weights. This accords with our intuition that both kinds of regularization penalize large weights. But the way the weights shrink is different. In L1 regularization, the weights shrink by a constant amount toward 0. In L2 regularization, the weights shrink by an amount which is proportional to $\omega$. And so when a particular weight has a large magnitude, $|\omega|$, L1 regularization shrinks the weight much less than L2 regularization does. By contrast, when $|\omega|$ is small, L1 regularization shrinks the weight much more than L2 regularization. The net result is that L1 regularization tends to concentrate the weights of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero.

**Dropout:** Dropout is a radically different technique for regularization. Unlike L1 and L2 regularization, dropout doesn't rely on modifying the cost function. Instead, in dropout we modify the network itself. Let me describe the basic mechanics of how dropout works, before getting into why it works, and what the results are.

Suppose we're trying to train a network:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz30.png)

With dropout, this process is modified. We start by randomly (and temporarily) deleting half the hidden neurons in the network, while leaving the input and output neurons untouched. After doing this, we'll end up with a network along the following lines. Note that the dropout neurons, i.e., the neurons which have been temporarily deleted, are still ghosted in:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz31.png)

We forward-propagate the input x through the modified network, and then backpropagate the result, also through the modified network. After doing this over a mini-batch of examples, we update the appropriate weights and biases. We then repeat the process, first restoring the dropout neurons, then choosing a new random subset of hidden neurons to delete, estimating the gradient for a different mini-batch, and updating the weights and biases in the network.

By repeating this process over and over, our network will learn a set of weights and biases. Of course, those weights and biases will have been learnt under conditions in which half the hidden neurons were dropped out. When we actually run the full network that means that twice as many hidden neurons will be active. To compensate for that, we halve the weights outgoing from the hidden neurons.

Heuristically, when we dropout different sets of neurons, it's rather like we're training different neural networks. And so the dropout procedure is like averaging the effects of a very large number of different networks. The different networks will overfit in different ways, and so, hopefully, the net effect of dropout will be to reduce overfitting.

A related heuristic explanation for dropout is given in [one of the earliest papers](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) to use the technique: "This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons." In other words, if we think of our network as a model which is making predictions, then we can think of dropout as a way of making sure that the model is robust to the loss of any individual piece of evidence. In this, it's somewhat similar to L1 and L2 regularization, which tend to reduce weights, and thus make the network more robust to losing any individual connection in the network.

Of course, the true measure of dropout is that it has been very successful in improving the performance of neural networks. The [original paper](http://arxiv.org/pdf/1207.0580.pdf) introducing the technique applied it to many different tasks.

**Artificially expanding the training data:** This idea is very powerful and has been widely used. Let's look at some of the results from [a paper](http://dx.doi.org/10.1109/ICDAR.2003.1227801), which applied several variations of the idea to MNIST. One of the neural network architectures they considered was along similar lines to what we've been using, a feedforward network with 800 hidden neurons and using the cross-entropy cost function. Running the network with the standard MNIST training data they achieved a classification accuracy of 98.4 percent on their test set. But then they expanded the training data, using not just rotations, as I described above, but also translating and skewing the images. By training on the expanded data set they increased their network's accuracy to 98.9 percent. They also experimented with what they called "elastic distortions", a special type of image distortion intended to emulate the random oscillations found in hand muscles. By using the elastic distortions to expand the data they achieved an even higher accuracy, 99.3 percent. Effectively, they were broadening the experience of their network by exposing it to the sort of variations that are found in real handwriting.

**An aside on big data and what it means to compare classification accuracies:** Many papers focus on finding new tricks to wring out improved performance on standard benchmark data sets. "Our whiz-bang technique gave us an improvement of X percent on standard benchmark Y" is a canonical form of research claim.  The message to take away, especially in practical applications, is that what we want is both better algorithms and better training data. It's fine to look for better algorithms, but make sure you're not focusing on better algorithms to the exclusion of easy wins getting more or better training data.

### Weight initialization

It turns out that we can do quite a bit better than initializing with normalized Gaussians. To see why, suppose we're working with a network with a large number - say 1,000 - of input neurons. And let's suppose we've used normalized Gaussians to initialize the weights connecting to the first hidden layer. For now I'm going to concentrate specifically on the weights connecting the input neurons to the first neuron in the hidden layer, and ignore the rest of the network:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz32.png)

Let's consider the weighted sum $z = \sum_j\omega_j x_j + b$ of inputs to our hidden neuron. 500 terms in this sum vanish, because the corresponding input xj is zero. And so z is a sum over a total of 501 normalized Gaussian random variables, accounting for the 500 weight terms and the 1 extra bias term. Thus z is itself distributed as a Gaussian with mean zero and standard deviation $\sqrt{501} \approx 22.4$. That is, z has a very broad Gaussian distribution, not sharply peaked at all. Then the output $\sigma(z)$ from the hidden neuron will be very close to either 1 or 0. That means our hidden neuron will have saturated. And when that happens, as we know, making small changes in the weights will make only absolutely miniscule changes in the activation of our hidden neuron. That miniscule change in the activation of the hidden neuron will, in turn, barely affect the rest of the neurons in the network at all, and we'll see a correspondingly miniscule change in the cost function. As a result, those weights will only learn very slowly when we use the gradient descent algorithm. It's similar to the problem we discussed earlier in this chapter, in which output neurons which saturated on the wrong value caused learning to slow down. We addressed that earlier problem with a clever choice of cost function. Unfortunately, while that helped with saturated output neurons, it does nothing at all for the problem with saturated hidden neurons.

Suppose we have a neuron with $n_{in}$ input weights. Then we shall initialize those weights as Gaussian random variables with mean 0 and standard deviation $1/\sqrt{n_{in}}.$ That is, we'll squash the Gaussians down, making it less likely that our neuron will saturate, and correspondingly much less likely to have problems with a learning slowdown.

Let's compare the results for both our old and new approaches to weight initialization, using the MNIST digit classification task.

The final classification accuracy is almost exactly the same. But the new initialization technique brings us there much, much faster. So it looks as though the improved weight initialization only speeds up learning, it doesn't change the final performance of our networks. However, in Chapter 4 we'll see examples of neural networks where the long-run behaviour is significantly better with the $1/\sqrt{n_{in}}$ weight initialization. Thus it's not only the speed of learning which is improved, it's sometimes also the final performance.

The $1/\sqrt{n_{in}}$ approach to weight initialization helps improve the way our neural nets learn. Other techniques for weight initialization have also been proposed, many building on this basic idea. I won't review the other approaches here, since $1/\sqrt{n_{in}}$ works well enough for our purposes. If you're interested in looking further, I recommend looking at the discussion on pages 14 and 15 of [a 2012 paper](http://arxiv.org/pdf/1206.5533v2.pdf) by Yoshua Bengio, as well as the references therein.

### How to choose a neural network's hyper-parameters

In practice, when you're using neural nets to attack a problem, it can be difficult to find good hyper-parameters.

**Broad strategy:** When using neural networks to attack a new problem the first challenge is to get any *non-trivial* learning, i.e., for the network to achieve results better than chance. This can be surprisingly difficult, especially when confronting a new class of problem. The main idea of this strategy is:

- First, simplify the neural net model, for example, using one layer of hidden neuron.
- Tuning the hyper-parameter one by one until you get a improving, non-trivial learning result.
- Enchance the neural net, tuning again.

In the early stage make sure you get quick feedback from experiments. Intuitively, it may seem as though simplifying the problem and the architecture will merely slow you down. In fact, it speeds things up, since you much more quickly find a network with a meaningful signal. Once you've got such a signal, you can often get rapid improvements by tweaking the hyper-parameters. **As with many things in life, getting started can be the hardest thing to do.**

Let's now look at some specific recommendations for setting hyper-parameters. I will focus on the learning rate, $\eta$, the L2 regularization parameter, $\lambda$, and the mini-batch size. However, many of the remarks apply also to other hyper-parameters, including those associated to network architecture, other forms of regularization, and some hyper-parameters we'll meet later in the book, such as the momentum co-efficient.

**Learning rate initialization:**  Gradient descent uses a first-order approximation to the cost function as a guide to how to decrease the cost. For large $\eta$, higher-order terms in the cost function become more important, and may dominate the behaviour, causing gradient descent to break down. This is especially likely as we approach minima and quasi-minima of the cost function, since near such points the gradient becomes small, making it easier for higher-order terms to dominate behaviour.

We can set η as follows. First, we estimate the threshold value for $\eta$ at which the cost on the training data immediately begins decreasing, instead of oscillating or increasing. This estimate doesn't need to be too accurate. You may optionally refine your estimate, to pick out the largest value of $\eta$ at which the cost decreases during the first few epochs. This gives us an estimate for the threshold value of $\eta$.

Obviously, the actual value of $\eta$ that you use should be no larger than the threshold value. In fact, if the value of $\eta$ is to remain usable over many epochs then you likely want to use a value for $\eta$ that is smaller, say, a factor of two below the threshold. Such a choice will typically allow you to train for many epochs, without causing too much of a slowdown in learning.

**Use early stopping to determine the number of training epochs:** As we discussed earlier in the chapter, early stopping means that at the end of each epoch we should compute the classification accuracy on the validation data. When that stops improving, terminate. This makes setting the number of epochs very simple. In particular, it means that we don't need to worry about explicitly figuring out how the number of epochs depends on the other hyper-parameters. Furthermore, early stopping also automatically prevents us from overfitting. This is, of course, a good thing, although in the early stages of experimentation it can be helpful to turn off early stopping, so you can see any signs of overfitting, and use it to inform your approach to regularization.

To implement early stopping we need to say more precisely what it means that the classification accuracy has stopped improving. If we stop the first time the accuracy decreases then we'll almost certainly stop when there are more improvements to be had. A better rule is to terminate if the best classification accuracy doesn't improve for quite some time.

Using the no-improvement-in-ten rule for initial experimentation, and gradually adopting more lenient rules, as you better understand the way your network trains: no-improvement-in-twenty, no-improvement-in-fifty, and so on.

**Learning rate schedule:** It's best to use a large learning rate that causes the weights to change quickly, and then reduce the learning rate to make more fine-tuned adjustments to the weights.

Many approaches are possible. One natural approach is to use the same basic idea as early stopping. The idea is to hold the learning rate constant until the validation accuracy starts to get worse. Then decrease the learning rate by some amount, say a factor of two or ten. We repeat this many times, until, say, the learning rate is a factor of 1,024 (or 1,000) times lower than the initial value. Then we terminate. A variable learning schedule can improve performance, but it also opens up a world of possible choices for the learning schedule. Those choices can be a headache - you can spend forever trying to optimize your learning schedule. For first experiments my suggestion is to use a single, constant value for the learning rate. That'll get you a good first approximation. Later, if you want to obtain the best performance from your network, it's worth experimenting with a [learning schedule](http://arxiv.org/abs/1003.0358).

**The regularization parameter, $\lambda$:** I suggest starting initially with no regularization ($\lambdaλ=0.0$), and determining a value for $\eta$, as above.Start by trialling $\lambda=1.0$, and then increase or decrease by factors of 10, as needed to improve performance on the validation data. Once you've found a good order of magnitude, you can fine tune your value of $\lambda$. That done, you should return and re-optimize $\eta$ again.

**Mini-batch size:** Choosing the best mini-batch size is a compromise. Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. Too large and you're simply not updating your weights often enough. What you need is to choose a compromise value which maximizes the speed of learning. Fortunately, the choice of mini-batch size at which the speed is maximized is relatively independent of the other hyper-parameters (apart from the overall architecture), so you don't need to have optimized those hyper-parameters in order to find a good mini-batch size. The way to go is therefore to use some acceptable (but not necessarily optimal) values for the other hyper-parameters, and then trial a number of different mini-batch sizes, scaling η as above. Plot the validation accuracy versus time (as in, real elapsed time, not epoch!), and choose whichever mini-batch size gives you the most rapid improvement in performance. With the mini-batch size chosen you can then proceed to optimize the other hyper-parameters.

**Automated techniques:** A great deal of work can be done on automating the process. A common technique is *grid search*, which systematically searches through a grid in hyper-parameter space. A review of both the achievements and the limitations of grid search (with suggestions for easily-implemented alternatives) may be found in a [2012 paper](http://dl.acm.org/citation.cfm?id=2188395) by James Bergstra and Yoshua Bengio. Many more sophisticated approaches have also been proposed. A particularly [promising 2012 paper](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf) uses a Bayesian approach to automatically optimize hyper-parameters. The code from the paper is [publicly available](https://github.com/jaberg/hyperopt), and has been used with some success by other researchers.

**Summing up:** In practice, there are relationships between the hyper-parameters and it helps to bounce backward and forward, gradually closing in good values. You should be on the lookout for signs that things aren't working, and be willing to experiment. In particular, this means carefully monitoring your network's behaviour, especially the validation accuracy.

One thing that becomes clear as you read articles and, especially, as you engage in your own experiments, is that hyper-parameter optimization is not a problem that is ever completely solved. There's always another trick you can try to improve performance. The space of hyper-parameters is so large that one never really finishes optimizing, one only abandons the network to posterity. So your goal should be to develop a workflow that enables you to quickly do a pretty good job on the optimization, while leaving you the flexibility to try more detailed optimizations, if that's important.

The challenge of setting hyper-parameters has led some people to complain that neural networks require a lot of work when compared with other machine learning techniques. Of course, from a practical point of view it's good to have easy-to-apply techniques. This is particularly true when you're just getting started on a problem, and it may not be obvious whether machine learning can help solve the problem at all. On the other hand, if getting optimal performance is important, then you may need to try approaches that require more specialist knowledge. While it would be nice if machine learning were always easy, there is no a priori reason it should be trivially simple.

### Other techniques

#### Variations on stochastic gradient descent

Stochastic gradient descent by backpropagation has served us well in attacking the MNIST digit classification problem. However, there are many other approaches to optimizing the cost function, and sometimes those other approaches offer performance superior to mini-batch stochastic gradient descent. In this section I sketch two such approaches, the Hessian and momentum techniques.

**Hessian technique(Newton Method):** By Taylor's theorem, the cost function can be approximated near a point $\omega$ by:

$$
C(\omega + \Delta\omega) = C(\omega) + \nabla C\cdot \Delta\omega + \frac{1}{2}\Delta\omega^T H\Delta\omega + \dots
$$

Suppose we approximate C by discarding the higher-order terms,

$$
C(\omega + \Delta\omega) \approx C(\omega) + \nabla C\cdot \Delta\omega + \frac{1}{2}\Delta\omega^T H\Delta\omega
$$

Using calculus we can show that the expression on the right-hand side can be minimized(Strictly speaking, for this to be a minimum, and not merely an extremum, we need to assume that the Hessian matrix is positive definite. Intuitively, this means that the function C looks like a valley locally, not a mountain or a saddle.) by choosing

$$
\Delta\omega = -H^{-1}\nabla C
$$

This is a good approximate expression for the cost function, then we'd expect that moving from the point $\omega$ to $\omega + \Delta\omega = \omega − H^{−1}\nabla C$ should significantly decrease the cost function. That suggests a possible algorithm for minimizing the cost:

- Choose a starting point, $\omega$.
- Update $\omega$ to a new point $\omega^{\prime} = \omega - H^{-1} \nabla C$, where the Hessian $H$ and $\nabla C$ are computedt at $\omega$.
- Update $\omega^{\prime}$ to a new point $\omega^{\prime\prime} = \omega^{\prime} - H^{\prime-1}\nabla^{\prime}C$, where the Hessian $H^{\prime}$ and $\nabla^{\prime}C$ are computed at $\omega^{prime}$
- ...

We do this by repeatedly changing $\omega$ by an amount $\Delta\omega = −\eta H^{-1}\nabla C$, where $\eta$ is known as the learning rate.

This approach to minimizing a cost function is known as the *Hessian technique* or *Hessian optimization* (We used to call it the *Newton Method*).  There are theoretical and empirical results showing that Hessian methods converge on a minimum in fewer steps than standard gradient descent. In particular, by incorporating information about second-order changes in the cost function it's possible for the Hessian approach to avoid many pathologies that can occur in gradient descent. Furthermore, there are versions of the backpropagation algorithm which can be used to compute the Hessian.

If Hessian optimization is so great, why aren't we using it in our neural networks? Unfortunately, while it has many desirable properties, it has one very undesirable property: it's very difficult to apply in practice. However, that doesn't mean that it's not useful to understand. In fact, there are many variations on gradient descent which are inspired by Hessian optimization, but which avoid the problem with overly-large matrices. Let's take a look at one such technique, momentum-based gradient descent.

**Momentum-based gradient descent:** Intuitively, the advantage Hessian optimization has is that it incorporates not just information about the gradient, but also information about how the gradient is changing.  Momentum-based gradient descent is based on a similar intuition, but avoids large matrices of second derivatives.

The momentum technique modifies gradient descent in two ways that make it more similar to the physical picture of a ball rolling down into a valley. First, it introduces a notion of "velocity" for the parameters we're trying to optimize. The gradient acts to change the velocity, not (directly) the "position", in much the same way as physical forces change the velocity, and only indirectly affect position. Second, the momentum method introduces a kind of friction term, which tends to gradually reduce the velocity.

Let's give a more precise mathematical description. We introduce velocity variables $v = v_1, v_2,\dots$ one for each corresponding $\omega_j$ variable

$$
\begin{aligned}
v&\rightarrow v^{\prime} = \mu v - \eta\nabla C \\
\omega &\rightarrow \omega^{\prime} = \omega + v^{\prime}
\end{aligned}
$$

In these equations, $\mu$ is a hyper-parameter which controls the amount of damping or friction in the system. The "force" $\nabla C$ is now modifying the velocity, v, and the velocity is controlling the rate of change of $\omega$.  Intuitively, we build up the velocity by repeatedly adding gradient terms to it. That means that if the gradient is in (roughly) the same direction through several rounds of learning, we can build up quite a bit of steam moving in that direction.

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz34.png)

With each step the velocity gets larger down the slope, so we move more and more quickly to the bottom of the valley. This can enable the momentum technique to work much faster than standard gradient descent. Of course, a problem is that once we reach the bottom of the valley we will overshoot. Or, if the gradient should change rapidly, then we could find ourselves moving in the wrong direction. That's the reason for the $\mu$ hyper-parameter: to control the amount of friction. When $\mu = 1$, as we've seen, there is no friction, and the velocity is completely driven by the gradient $\nabla C$. By contrast, when $\mu = 0$ there's a lot of friction, the velocity can't build up, it reduces to the usual equation for gradient descent, $\omega \rightarrow \omega^{\prime} = \omega − \eta \nabla C$. In practice, using a value of μ intermediate between 0 and 1 can give us much of the benefit of being able to build up speed, but without causing overshooting. We can choose such a value for μ using the held-out validation data, in much the same way as we select $\eta$ and $\lambda$. The standard name for $\mu$ is *momentum coefficient* which is badly chosen.

A nice thing about the momentum technique is that it takes almost no work to modify an implementation of gradient descent to incorporate momentum. We can still use backpropagation to compute the gradients, just as before, and use ideas such as sampling stochastically chosen mini-batches. In this way, we can get some of the advantages of the Hessian technique, using information about how the gradient is changing. But it's done without the disadvantages, and with only minor modifications to our code. In practice, the momentum technique is commonly used, and often speeds up learning.

**Other approaches to mininizing the cost function:** Many other approaches to minimizing the cost function have been developed, and there isn't universal agreement on which is the best approach. As you go deeper into neural networks it's worth digging into the other techniques, understanding how they work, their strengths and weaknesses, and how to apply them in practice.  [A paper mentioned earlier](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) introduces and compares several of these techniques, including conjugate gradient descent and the BFGS method (see also the closely related limited-memory BFGS method, known as [L-BFGS](http://en.wikipedia.org/wiki/Limited-memory_BFGS)). Another technique which has recently shown [promising results](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf) is Nesterov's accelerated gradient technique, which improves on the momentum technique. However, for many problems, plain stochastic gradient descent works well, especially if momentum is used, and so we'll stick to stochastic gradient descent through the remainder of this book.

#### Other models of artificial neuron

In principle, a network built from sigmoid neurons can compute any function. In practice, however, networks built using other model neurons sometimes outperform sigmoid networks. Depending on the application, networks based on such alternate models may learn faster, generalize better to test data, or perhaps do both.

Perhaps the simplest variation is the $\tanh$ neuron, which replaces the sigmoid function by the hyperbolic tangent function.

$$
\tanh(\omega \cdot x + b)
$$

With a little algebra it can easily be verified that

$$
\sigma(z) = \frac{1 + \tanh(z/2)}{2}
$$

One difference between tanh neurons and sigmoid neurons is that the output from $\tanh$ neurons ranges from -1 to 1, not 0 to 1. This means that if you're going to build a network based on $\tanh$ neurons you may need to normalize your outputs (and, depending on the details of the application, possibly your inputs) a little differently than in sigmoid networks.

Similar to sigmoid neurons, a network of $\tanh$ neurons can, in principle, compute any function mapping inputs to the range -1 to 1. Furthermore, ideas such as backpropagation and stochastic gradient descent are as easily applied to a network of $\tanh$ neurons as to a network of sigmoid neurons.

Which type of neuron should you use in your networks, the tanh or sigmoid? A priori the answer is not obvious, to put it mildly! However, there are theoretical arguments and some empirical evidence to suggest that the $\tanh$ sometimes performs better (or example, [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), by Yann LeCun, Léon Bottou, Genevieve Orr and Klaus-Robert Müller (1998), and [Understanding the difficulty of training deep feedforward networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf), by Xavier Glorot and Yoshua Bengio (2010).).

Let me briefly give you the flavor of one of the theoretical arguments for $\tanh$ neurons. Suppose we're using sigmoid neurons, so all activations in our network are positive. Let's consider the weights $w^{l+1}_{jk}$ input to the jth neuron in the l+1th layer. The rules for backpropagation tell us that the associated gradient will be $a^l_k\delta^{l+1}_j$. Because the activations are positive the sign of this gradient will be the same as the sign of $\delta^{l+1}_j$. In other words, all weights to the same neuron must either increase together or decrease together. That's a problem, since some of the weights may need to increase while others need to decrease. That can only happen if some of the input activations have different signs. That suggests replacing the sigmoid by an activation function, such as $\tanh$, which allows both positive and negative activations. Indeed, because $\tanh$ is symmetric about zero, $\tanh(−z) = −\tanh(z)$, we might even expect that, roughly speaking, the activations in hidden layers would be equally balanced between positive and negative. That would help ensure that there is no systematic bias for the weight updates to be one way or the other.

How seriously should we take this argument? While the argument is suggestive, it's a heuristic, not a rigorous proof that $\tanh$ neurons outperform sigmoid neurons. Perhaps there are other properties of the sigmoid neuron which compensate for this problem? Indeed, for many tasks the $\tanh$ is found empirically to provide only a small or no improvement in performance over sigmoid neurons. Unfortunately, we don't yet have hard-and-fast rules to know which neuron types will learn fastest, or give the best generalization performance, for any particular application.

Another variation on the sigmoid neuron is the *rectified linear neuron* or *rectified linear unit*.

$$
\max(0, \omega \cdot x + b)
$$

Obviously such neurons are quite different from both sigmoid and tanh neurons. However, like the sigmoid and tanh neurons, rectified linear units can be used to compute any function, and they can be trained using ideas such as backpropagation and stochastic gradient descent.

#### Other stories in neural networks

This part is of great interests that the author explain the situation of why most result of deep learning are unrigorious. Worth reading!!

## Chapter 4. A Visual Proof That Neural Nets Can Compute Any Function

One of the most striking facts about neural networks is that they can compute any function at all.

No matter what the function, there is guaranteed to be a neural network so that for every possible input, x, the value f(x) (or some close approximation) is output from the network.

This result tells us that neural networks have a kind of universality. No matter what function we want to compute, we know that there is a neural network which can do the job.

What's more, this universality theorem holds even if we restrict our networks to have just a single layer intermediate between the input and the output neurons - a so-called single hidden layer. So even very simple network architectures can be extremely powerful.

The universality theorem is well known by people who use neural networks. But why it's true is not so widely understood. Most of the explanations available are quite technical. For instance, one of the original papers proving the result ([Approximation by superpositions of a sigmoidal function](http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf), by George Cybenko (1989). The result was very much in the air at the time, and several groups proved closely related results. Cybenko's paper contains a useful discussion of much of that work. Another important early paper is [Multilayer feedforward networks are universal approximators](http://www.sciencedirect.com/science/article/pii/0893608089900208), by Kurt Hornik, Maxwell Stinchcombe, and Halbert White (1989). This paper uses the Stone-Weierstrass theorem to arrive at similar results.) did so using the Hahn-Banach theorem, the Riesz Representation theorem, and some Fourier analysis.

In this chapter I give a simple and mostly visual explanation of the universality theorem. We'll go step by step through the underlying ideas. You'll understand why it's true that neural networks can compute any function. You'll understand some of the limitations of the result. And you'll understand how the result relates to deep neural networks.

### Two caveats

Before explaining why the universality theorem is true, I want to mention two caveats to the informal statement "a neural network can compute any function".

**First**, this doesn't mean that a network can be used to exactly compute any function. Rather, we can get an approximation that is as good as we want. By increasing the number of hidden neurons we can improve the approximation.

And we can do still better by further increasing the number of hidden neurons.

To make this statement more precise, suppose we're given a function $f(x)$ which we'd like to compute to within some desired accuracy $\varepsilon > 0$. The guarantee is that by using enough hidden neurons we can always find a neural network whose output $g(x)$ satisfies $|g(x)−f(x)| < \varepsilon$, for all inputs $x$. In other words, the approximation will be good to within the desired accuracy for every possible input.

**The second caveat** is that the class of functions which can be approximated in the way described are the continuous functions. However, even if the function we'd really like to compute is discontinuous, it's often the case that a continuous approximation is good enough. If that's so, then we can use a neural network. In practice, this is not usually an important limitation.

### Universality with one input and one output

To understand why the universality theorem is true, let's start by understanding how to construct a neural network which approximates a function with just one input and one output:

![ImageNA](data:image/png;base64,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)

It turns out that this is the core of the problem of universality. Once we've understood this special case it's actually pretty easy to extend to functions with many inputs and many outputs.

## Chapter 5. Why are deep neural networks hard to train

### The vanishing gradient problem

We have here an important observation: in at least some deep neural networks, the gradient tends to get smaller as we move backward through the hidden layers. This means that neurons in the earlier layers learn much more slowly than neurons in later layers. The phenomenon is known as the ***vanishing gradient problem*** (See [Gradient flow in recurrent nets: the difficulty of learning long-term dependencies](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.7321), by Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber (2001). This paper studied recurrent neural nets, but the essential phenomenon is the same as in the feedforward networks we are studying. See also Sepp Hochreiter's earlier Diploma Thesis, [Untersuchungen zu dynamischen neuronalen Netzen](http://www.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)).

Why does the vanishing gradient problem occur? Are there ways we can avoid it? And how should we deal with it in training deep neural networks? In fact, we'll learn shortly that it's not inevitable, although the alternative is not very attractive, either: sometimes the gradient gets much larger in earlier layers! This is the ***exploding gradient problem***, and it's not much better news than the vanishing gradient problem. More generally, it turns out that the gradient in deep neural networks is unstable, tending to either explode or vanish in earlier layers. This instability is a fundamental problem for gradient-based learning in deep neural networks. It's something we need to understand, and, if possible, take steps to address.

#### What's causing the vanishing gradient problem? Unstable gradients in deep neural nets

To get insight into why the vanishing gradient problem occurs, let's consider the simplest deep neural network: one with just a single neuron in each layer. Here's a network with three hidden layers:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz37.png)

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz38.png)

$$
\frac{\partial C}{\partial b_1} = \sigma^{\prime}(z_1)\omega_2\sigma^{\prime}(z_2)\omega_3\sigma^{\prime}(z_3)\omega_4\sigma^{\prime}(z_4)\frac{\partial C}{\partial a_4}
$$

Excepting the very last term, this expression is a product of terms of the form $\omega_j\sigma^{\prime}(z_j)$. The derivatice $\sigma^{\prime}(z)$ reaches its maximum at $\sigma^{\prime}(0) = 1/4$. Now, if we use our standard approach to initializing the weights in the network, then we'll choose the weights using a Gaussian with mean 0 and standard deviation 1, the terms $\omega_j\sigma^{\prime}(z_j)$ will usually satisfy $|\omega_j\sigma^{\prime}(z_j)|<1/4$. And when we take a product of many such terms, the product will tend to exponentially decrease: the more terms, the smaller the product will be. This is starting to smell like a possible explanation for the vanishing gradient problem.

**The unstable gradient problem:** The fundamental problem here isn't so much the vanishing gradient problem or the exploding gradient problem. It's that the gradient in early layers is the product of terms from all the later layers. When there are many layers, that's an intrinsically unstable situation. The only way all layers can learn at close to the same speed is if all those products of terms come close to balancing out. Without some mechanism or underlying reason for that balancing to occur, it's highly unlikely to happen simply by chance. In short, the real problem here is that neural networks suffer from an *unstable gradient problem*. As a result, if we use standard gradient-based learning techniques, different layers in the network will tend to learn at wildly different speeds.

### Unstable gradients in more complex networks

What about more complex deep networks, with many neurons in each hidden layer?

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz40.png)

In fact, much the same behaviour occurs in such networks. In the earlier chapter on backpropagation we saw that the gradient in the lth layer of an $L$ layer network is given by:

$$
\delta^l=\Sigma^{\prime}(z^l)(\omega^{l+1})^T\Sigma^{\prime}(z^{l+1})(\omega^{l+2})^T\dots\Sigma^{\prime}(z^L)\nabla_a C
$$

Here, $\Sigma^{\prime}(z^l)$ is a diagonal matrix whose entries are the $\sigma^{\prime}(z)$ values for the weighted inputs to the lth layer. The $\omega^l$ are the weight matrices for the different layers. And $\nabla_a C$ is the vector of partial derivatives of $C$ with respect to the output activations.

This is a much more complicated expression than in the single-neuron case. Still the essential form is very similar, with lots of pairs of the form $(\omega^j)^T\Sigma^{\prime}(z^j)$. What's more, the matrices $\Sigma^{\prime}(z^j)$ have small entries on the diagonal, none larger than 1/4. Provided the weight matrices $\omega^j$ aren't too large, each additional term $(\omega^j)^T\Sigma^{\prime}(z^j)$ tends to make the gradient vector smaller, leading to a vanishing gradient. More generally, the large number of terms in the product tends to lead to an unstable gradient, just as in our earlier example. In practice, empirically it is typically found in sigmoid networks that gradients vanish exponentially quickly in earlier layers. As a result, learning slows down in those layers. This slowdown isn't merely an accident or an inconvenience: it's a fundamental consequence of the approach we're taking to learning.

### Other obstacles to deep learning

In this chapter we've focused on vanishing gradients - and, more generally, unstable gradients - as an obstacle to deep learning.  In fact, unstable gradients are just one obstacle to deep learning, albeit an important fundamental obstacle.

As a first example, in 2010 Glorot and Bengio([Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf), by Xavier Glorot and Yoshua Bengio (2010). See also the earlier discussion of the use of sigmoids in [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), by Yann LeCun, Léon Bottou, Genevieve Orr and Klaus-Robert Müller (1998).) found evidence suggesting that the use of sigmoid activation functions can cause problems training deep networks. In particular, they found evidence that the use of sigmoids will cause the activations in the final hidden layer to saturate near 0 early in training, substantially slowing down learning. They suggested some alternative activation functions, which appear not to suffer as much from this saturation problem.

As a second example, in 2013 Sutskever, Martens, Dahl and Hinton([On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf), by Ilya Sutskever, James Martens, George Dahl and Geoffrey Hinton (2013).) studied the impact on deep learning of both the random weight initialization and the momentum schedule in momentum-based stochastic gradient descent. In both cases, making good choices made a substantial difference in the ability to train deep networks.

In this chapter, we've focused on the instabilities associated to gradient-based learning in deep networks. The results in the last two paragraphs suggest that there is also a role played by the choice of activation function, the way weights are initialized, and even details of how learning by gradient descent is implemented. And, of course, choice of network architecture and other hyper-parameters is also important. Thus, many factors can play a role in making deep networks hard to train, and understanding all those factors is still a subject of ongoing research. This all seems rather downbeat and pessimism-inducing. But the good news is that in the next chapter we'll turn that around, and develop several approaches to deep learning that to some extent manage to overcome or route around all these challenges.

## Chapter 6. Deep learning

In this chapter, we'll develop techniques which can be used to train deep networks, and apply them in practice. We'll also look at the broader picture, briefly reviewing recent progress on using deep nets for image recognition, speech recognition, and other applications. And we'll take a brief, speculative look at what the future may hold for neural nets, and for artificial intelligence.

The main part of the chapter is an introduction to one of the most widely used types of deep network: deep convolutional networks. We'll work through a detailed example - code and all - of using convolutional nets to solve the problem of classifying handwritten digits from the MNIST data set:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/digits.png)

We'll start our account of convolutional networks with the shallow networks used to attack this problem earlier in the book. Through many iterations we'll build up more and more powerful networks. As we go we'll explore many powerful techniques: convolutions, pooling, the use of GPUs to do far more training than we did with our shallow networks, the algorithmic expansion of our training data (to reduce overfitting), the use of the dropout technique (also to reduce overfitting), the use of ensembles of networks, and others.

The remainder of the chapter discusses deep learning from a broader and less detailed perspective. We'll briefly survey other models of neural networks, such as recurrent neural nets and long short-term memory units, and how such models can be applied to problems in speech recognition, natural language processing, and other areas. And we'll speculate about the future of neural networks and deep learning, ranging from ideas like intention-driven user interfaces, to the role of deep learning in artificial intelligence.

The chapter builds on the earlier chapters in the book, making use of and integrating ideas such as backpropagation, regularization, the softmax function, and so on.

The chapter is not a tutorial on the latest and greatest neural networks libraries. Nor are we going to be training deep networks with dozens of layers to solve problems at the very leading edge. Rather, the focus is on understanding some of the core principles behind deep neural networks, and applying them in the simple, easy-to-understand context of the MNIST problem. Put another way: the chapter is not going to bring you right up to the frontier. Rather, the intent of this and earlier chapters is to focus on fundamentals, and so to prepare you to understand a wide range of current work.

### Introducing convolutional networks

We did recognize images of handwritten digits using networks in which adjacent network layers are fully connected to one another. But upon reflection, it's strange to use networks with fully-connected layers to classify images. The reason is that such a network architecture does not take into account the spatial structure of the images. For instance, it treats input pixels which are far apart and close together on exactly the same footing. Such concepts of spatial structure must instead be inferred from the training data. But what if, instead of starting with a network architecture which is tabula rasa, we used an architecture which tries to take advantage of the spatial structure?

In this section I describe convolutional neural networks. These networks use a special architecture which is particularly well-adapted to classify images. Using this architecture makes convolutional networks fast to train. This, in turn, helps us train deep, many-layer networks, which are very good at classifying images. Today, deep convolutional networks or some close variant are used in most neural networks for image recognition.

Convolutional neural networks use three basic ideas: *local receptive fields*, *shared weights*, and *pooling*.

**Local receptive fields:** In the fully-connected layers shown earlier, the inputs were depicted as a vertical line of neurons.

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz42.png)

As per usual, we'll connect the input pixels to a layer of hidden neurons. But we won't connect every input pixel to every hidden neuron. Instead, we only make connections in small, localized regions of the input image.

To be more precise, each neuron in the first hidden layer will be connected to a small region of the input neurons, say, for example, a $5\times 5$ region, corresponding to 25 input pixels. So, for a particular hidden neuron, we might have connections that look like this:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz43.png)

That region in the input image is called the *local receptive field* for the hidden neuron. Each connection learns a weight. And the hidden neuron learns an overall bias as well. You can think of that particular hidden neuron as learning to analyze its particular local receptive field.

We then slide the local receptive field across the entire input image. For each local receptive field, there is a different hidden neuron in the first hidden layer. To illustrate this concretely, let's start with a local receptive field in the top-left corner:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz44.png)

Then we slide the local receptive field over by one pixel to the right (i.e., by one neuron), to connect to a second hidden neuron:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz45.png)

And so on, building up the first hidden layer. I've shown the local receptive field being moved by one pixel at a time. In fact, sometimes a different stride length is used. For instance, we might move the local receptive field 2 pixels to the right (or down), in which case we'd say a stride length of 2 is used. In this chapter we'll mostly stick with stride length 1, but it's worth knowing that people sometimes experiment with different stride lengths.

**Shared weights and biases:** It is shown previously that each hidden neuron has a bias and $5\times 5$ weights connected to its local receptive field. Instead of using a lot of different weights and biases, we're going to use the same weights and bias for the hidden neurons. In other words, for the j,kth hidden neuron, the output is:

$$
\sigma\left(b + \Sigma_{l = 0}^4\Sigma_{m = 0}^4\omega_{l,m}a_{j+l,k+m}\right)
$$

Here, $\sigma$ is the neural activation function. $b$ is the shared value for the bias. $\omega_{l,m}$ is a 5×5 array of shared weights. And, finally, we use $a_{x,y}$ to denote the input activation at position $x,y$.

This means that all the neurons in the first hidden layer detect exactly the same feature(Informally, think of the feature detected by a hidden neuron as the kind of input pattern that will cause the neuron to activate: it might be an edge in the image, for instance, or maybe some other type of shape.), just at different locations in the input image. To see why this makes sense, suppose the weights and bias are such that the hidden neuron can pick out, say, a vertical edge in a particular local receptive field. That ability is also likely to be useful at other places in the image. And so it is useful to apply the same feature detector everywhere in the image. To put it in slightly more abstract terms, convolutional networks are well adapted to the *translation invariance* of images: move a picture of a cat (say) a little ways, and it's still an image of a cat.

For this reason, we sometimes call the map from the input layer to the hidden layer a *feature map*. We call the weights defining the feature map the *shared weights*. And we call the bias defining the feature map in this way the *shared bias*. The shared weights and bias are often said to define a *kernel or filter*.

The network structure I've described so far can detect just a single kind of localized feature. To do image recognition we'll need more than one feature map. And so a complete convolutional layer consists of several different feature maps:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz46.png)

In the example shown, there are 3 feature maps. Each feature map is defined by a set of 5×5 shared weights, and a single shared bias. The result is that the network can detect 3 different kinds of features, with each feature being detectable across the entire image.

A big advantage of sharing weights and biases is that it greatly reduces the number of parameters involved in a convolutional network. For each feature map we need 25=5×5 shared weights, plus a single shared bias. If we have 20 feature maps that's a total of 20×26=520 parameters defining the convolutional layer. By comparison, suppose we had a fully connected first layer, with 784=28×28 input neurons, and a relatively modest 30 hidden neurons, as we used in many of the examples earlier in the book. That's a total of 784×30 weights, plus an extra 30 biases, for a total of 23,550 parameters.

Of course, we can't really do a direct comparison between the number of parameters, since the two models are different in essential ways. But, intuitively, it seems likely that the use of translation invariance by the convolutional layer will reduce the number of parameters it needs to get the same performance as the fully-connected model. That, in turn, will result in faster training for the convolutional model, and, ultimately, will help us build deep networks using convolutional layers.

Incidentally, the name convolutional comes from the fact that the opration in equation

$$
a^1 = \sigma(b+\omega * a^0)
$$

is sometimes known as a *convolution*.

**Pooling layers:** In addition to the convolutional layers just described, convolutional neural networks also contain pooling layers. Pooling layers are usually used immediately after convolutional layers. What the pooling layers do is to simplify the information in the output from the convolutional layer.

In detail, a pooling layer takes each feature map(activation of the hidden neurons output from the layer) output from the convolutional layer and prepares a condensed feature map. For instance, each unit in the pooling layer may summarize a region of (say) 2×2 neurons in the previous layer. As a concrete example, one common procedure for pooling is known as max-pooling. In max-pooling, a pooling unit simply outputs the maximum activation in the 2×2 input region, as illustrated in the following diagram:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz47.png)

Note that since we have 24×24 neurons output from the convolutional layer, after pooling we have 12×12 neurons.

As mentioned above, the convolutional layer usually involves more than a single feature map. We apply max-pooling to each feature map separately. So if there were three feature maps, the combined convolutional and max-pooling layers would look like:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz48.png)

We can think of max-pooling as a way for the network to ask whether a given feature is found anywhere in a region of the image. It then throws away the exact positional information. The intuition is that once a feature has been found, its exact location isn't as important as its rough location relative to other features. A big benefit is that there are many fewer pooled features, and so this helps reduce the number of parameters needed in later layers.

Max-pooling isn't the only technique used for pooling. Another common approach is known as *L2 pooling*.

**Putting it all together:** We can now put all these ideas together to form a complete convolutional neural network. It's similar to the architecture we were just looking at, but has the addition of a layer of 10 output neurons, corresponding to the 10 possible values for MNIST digits ('0', '1', '2', etc):

![ImageNA](http://neuralnetworksanddeeplearning.com/images/tikz49.png)

The network begins with 28×28 input neurons, which are used to encode the pixel intensities for the MNIST image. This is then followed by a convolutional layer using a 5×5 local receptive field and 3 feature maps. The result is a layer of 3×24×24 hidden feature neurons. The next step is a max-pooling layer, applied to 2×2 regions, across each of the 3 feature maps. The result is a layer of 3×12×12 hidden feature neurons.

The final layer of connections in the network is a fully-connected layer. That is, this layer connects every neuron from the max-pooled layer to every one of the 10 output neurons.

We will train our network using stochastic gradient descent and backpropagation. This mostly proceeds in exactly the same way as in earlier chapters. However, we do need to make a few modifications to the backpropagation procedure.

### Convolutional neural networks in practice

Let's look at how convolutional neural net works in practice, by implementing some convolutional networks, and applying them to the MNIST digit classification problem.  In this section, we'll use `network3.py` as a library to build convolutional networks. For `network3.py` we're going to use a machine learning library known as [**Theano**](http://deeplearning.net/software/theano/). Using Theano makes it easy to implement backpropagation for convolutional neural networks, since it automatically computes all the mappings involved. In particular, one great feature of Theano is that it can run code on either a CPU or, if available, a GPU.

If you don't have a GPU available locally, then you may wish to look into Amazon Web Services [EC2 G2](http://aws.amazon.com/ec2/instance-types/) spot instances.

To get a baseline, we'll start with a shallow architecture using just a single hidden layer, containing 100 hidden neurons. We'll train for 60 epochs, using a learning rate of η=0.1, a mini-batch size of 10, and no regularization(Code for the experiments in this section may be found in [this script](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/conv.py). Note that the code in the script simply duplicates and parallels the discussion in this section.).

A best classification accuracy of 97.80 percent is obtained. This is the classification accuracy on the `test_data`, evaluated at the training epoch where we get the best classification accuracy on the `validation_data`. Using the validation data to decide when to evaluate the test accuracy helps avoid overfitting to the test data.

This 97.80 percent accuracy is close to the 98.04 percent accuracy obtained back, using a similar network architecture and learning hyper-parameters. There were, however, two differences in the earlier network. First, we regularized the earlier network, to help reduce the effects of overfitting. Regularizing the current network does improve the accuracies, but the gain is only small, and so we'll hold off worrying about regularization until later. Second, while the final layer in the earlier network used sigmoid activations and the cross-entropy cost function, the current network uses a softmax final layer, and the log-likelihood cost function.

Let's begin by inserting a convolutional layer, right at the beginning of the network. We'll use 5 by 5 local receptive fields, a stride length of 1, and 20 feature maps. We'll also insert a max-pooling layer, which combines the features using 2 by 2 pooling windows. So the overall network architecture looks much like the architecture discussed in the last section, but with an extra fully-connected layer:

![ImageNA](http://neuralnetworksanddeeplearning.com/images/simple_conv.png)

In this architecture, we can think of the convolutional and pooling layers as learning about local spatial structure in the input training image, while the later, fully-connected layer learns at a more abstract level, integrating global information from across the entire image. This is a common pattern in convolutional neural networks.

That gets us to 98.78 percent accuracy, which is a considerable improvement over any of our previous results.

In specifying the network structure, the convolutional and pooling layers are treated as a single layer. Whether they're regarded as separate layers or as a single layer is to some extent a matter of taste.

Let's try inserting a second convolutional-pooling layer. We'll make the insertion between the existing convolutional-pooling layer and the fully-connected hidden layer. Again, we'll use a 5×5 local receptive field, and pool over 2×2 regions.

```python
>>> net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
>>> net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
```

Once again, we get an improvement: we're now at 99.06 percent classification accuracy!

There's two natural questions to ask at this point.

1. What does it even mean to apply a second convolutional-pooling layer?
2. The output from the previous layer involves multiple separate feature maps, how should neurons in the second convolutional-pooling layer respond to these multiple input images?

The feature detectors in the second convolutional-pooling layer have access to all the features from the previous layer, but only within their particular local receptive field. This issue would have arisen in the first layer if the input images were in color. In that case we'd have 3 input features for each pixel, corresponding to red, green and blue channels in the input image. So we'd allow the feature detectors to have access to all color information, but only within a given local receptive field.

**Using rectified linear units:** The network we've developed at this point is actually a variant of one of the networks used in the [seminal 1998 pape](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) introducing the MNIST problem, a network known as *LeNet-5*. It's a good foundation for further experimentation, and for building up understanding and intuition.

As a beginning, let's change our neurons so that instead of using a sigmoid activation function, we use rectified linear units. That is, we'll use the activation function $f(z) \equiv \max(0,z)$. We'll train for 60 epochs, with a learning rate of $\eta = 0.03$. I also found that it helps a little to use some l2 regularization, with regularization parameter $\lambda = 0.1$:

```python
>>> from network3 import ReLU
>>> net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
>>> net.SGD(training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
```

I obtained a classification accuracy of 99.23 percent. It's a modest improvement over the sigmoid results (99.06). However, across all my experiments I found that networks based on rectified linear units consistently outperformed networks based on sigmoid activation functions. There appears to be a real gain in moving to rectified linear units for this problem.

What makes the rectified linear activation function better than the sigmoid or tanh functions? At present, we have a poor understanding of the answer to this question. Indeed, rectified linear units have only begun to be widely used in the past few years. The reason for that recent adoption is empirical: a few people tried rectified linear units, often on the basis of hunches or heuristic arguments. A common justification is that $\max(0,z)$ doesn't saturate in the limit of large z, unlike sigmoid neurons, and this helps rectified linear units continue learning. The argument is fine, as far it goes, but it's hardly a detailed justification, more of a just-so story.

In an ideal world we'd have a theory telling us which activation function to pick for which application. But at present we're a long way from such a world. I should not be at all surprised if further major improvements can be obtained by an even better choice of activation function. And I also expect that in coming decades a powerful theory of activation functions will be developed. Today, we still have to rely on poorly understood rules of thumb and experience.

**Expanding the training data:** nother way we may hope to improve our results is by *algorithmically expanding the training data*. A simple way of expanding the training data is to displace each training image by a single pixel, either up one pixel, down one pixel, left one pixel, or right one pixel. We can do this by running the program [expand_mnist.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/expand_mnist.py) from the shell prompt:

```python
python expand_mnist.py
```

Running this program takes the 50,000 MNIST training images, and prepares an expanded training set, with 250,000 training images. We'll use the same network as above, with rectified linear units. The number of training epochs will be reduced since we're traning with 5 times as much data. But, in fact, expanding the data turned out to considerably reduce the effect of overfitting. And so, after some experimentation, I eventually went back to training for 60 epochs. In any case, let's train:

```python
expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
>>> net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
>>> net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
```

Using the expanded training data I obtained a 99.37 percent training accuracy. So this almost trivial change gives a substantial improvement in classification accuracy. This idea of algorithmically expanding the data can be taken further. In 2003 Simard, [Steinkraus and Platt](http://dx.doi.org/10.1109/ICDAR.2003.1227801) improved their MNIST performance to 99.6 percent using a neural network otherwise very similar to ours, using two convolutional-pooling layers, followed by a hidden fully-connected layer with 100 neurons. There were a few differences of detail in their architecture - they didn't have the advantage of using rectified linear units, for instance - but the key to their improved performance was expanding the training data. They did this by rotating, translating, and skewing the MNIST training images. They also developed a process of "elastic distortion", a way of emulating the random oscillations hand muscles undergo when a person is writing. By combining all these processes they substantially increased the effective size of their training data, and that's how they achieved 99.6 percent accuracy.

**Inserting an extra fully-connected layer:** Expanding the fully-connected layer to 300 and 1000 neurons will obtain results of 99.46 and 99.43 precent, respectively. That's interesting, but not really a convincing win over the earlier result (99.37 percent). What about adding an extra fully-connected layer? Let's try inserting an extra fully-connected layer, so that we have two 100-hidden neuron fully-connected layers:

```python
>>> net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
>>> net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
```

Doing this, I obtained a test accuracy of 99.43 percent. Again, the expanded net isn't helping so much. Running similar experiments with fully-connected layers containing 300 and 1,000 neurons yields results of 99.48 and 99.47 percent. That's encouraging, but still falls short of a really decisive win.

Is it that the expanded or extra fully-connected layers really don't help with MNIST? Or might it be that our network has the capacity to do better, but we're going about learning the wrong way? For instance, maybe we could use stronger regularization techniques to reduce the tendency to overfit. One possibility is the dropout technique. Recall that the basic idea of dropout is to remove individual activations at random while training the network. This makes the model more robust to the loss of individual pieces of evidence, and thus less likely to rely on particular idiosyncracies of the training data. Let's try applying dropout to the final fully-connected layers:

```python
>>> net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
>>> net.SGD(expanded_training_data, 40, mini_batch_size, 0.03,
            validation_data, test_data)
```

Using this, we obtain an accuracy of 99.60 percent, which is a substantial improvement over our earlier results, especially our main benchmark, the network with 100 hidden neurons, where we achieved 99.37 percent.

There are two changes worth noting.

- First, I reduced the number of training epochs to 40: dropout reduced overfitting, and so we learned faster.
- Second, the fully-connected hidden layers have 1,000 neurons, not the 100 used earlier. Of course, dropout effectively omits many of the neurons while training, so some expansion is to be expected.

**Using an ensemble of networks:** An easy way to improve performance still further is to create several neural networks, and then get them to vote to determine the best classification. Even though the networks would all have similar accuracies, they might well make different errors, due to the different random initializations. It's plausible that taking a vote amongst our networks might yield a classification better than any individual network.

This sounds too good to be true, but this kind of ensembling is a common trick with both neural networks and other machine learning techniques. And it does in fact yield further improvements: we end up with 99.67 percent accuracy.

The remaining errors in the test set are shown below.

![ImageNA](http://neuralnetworksanddeeplearning.com/images/ensemble_errors.png)

**Why only applied dropout to the fully-connected layers:** In principle we could apply a similar procedure to the convolutional layers. But, in fact, there's no need: the convolutional layers have considerable inbuilt resistance to overfitting. The reason is that the shared weights mean that convolutional filters are forced to learn from across the entire image. This makes them less likely to pick up on local idiosyncracies in the training data. And so there is less need to apply other regularizers, such as dropout.

**Going further:** It's possible to improve performance on MNIST still further. Rodrigo Benenson has compiled an [informative summary page](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html), showing progress over the years, with links to papers. Many of these papers use deep convolutional networks along lines similar to the networks we've been using. If you dig through the papers you'll find many interesting techniques, and you may enjoy implementing some of them. If you do so it's wise to start implementation with a simple network that can be trained quickly, which will help you more rapidly understand what is going on.

**Why are we able to train:** There are fundamental obstructions to training in deep, many-layer neural networks. In particular, we saw that the gradient tends to be quite unstable. Since the gradient is the signal we use to train, this causes problems.

How have we avoided those results? Of course, the answer is that we haven't avoided these results.

Instead, we've done a few things that help us proceed anyway. In particular:

- (1) Using convolutional layers greatly reduces the number of parameters in those layers, making the learning problem much easier;
- (2) Using more powerful regularization techniques (notably dropout and convolutional layers) to reduce overfitting, which is otherwise more of a problem in more complex networks;
- (3) Using rectified linear units instead of sigmoid neurons, to speed up training - empirically, often by a factor of 3-5;
- (4) Using GPUs and being willing to train for a long period of time.

In particular, in our final experiments we trained for 40 epochs using a data set 5 times larger than the raw MNIST training data.

Of course, we've used other ideas, too:

- making use of sufficiently large data sets (to help avoid overfitting);
- using the right cost function (to avoid a learning slowdown);
- using good weight initializations (also to avoid a learning slowdown, due to neuron saturation);
- algorithmically expanding the training data.

We discussed these and other ideas early, and have for the most part been able to reuse these ideas with little comment in this chapter.

**A word on procedure:** If you start experimenting, I can guarantee things won't always be so smooth. The reason is that I've presented a cleaned-up narrative, omitting many experiments - including many failed experiments. This cleaned-up narrative will hopefully help you get clear on the basic ideas. But it also runs the risk of conveying an incomplete impression. Getting a good, working network can involve a lot of trial and error, and occasional frustration. In practice, you should expect to engage in quite a bit of experimentation. To speed that process up you may find it helpful to revisit [how to choose a neural network's hyper-parameters](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters), and perhaps also to look at some of the further reading suggested in that section.

### The code for our convolutional networks

### Recent progrss in image recognition

In 1998, the year MNIST was introduced, it took weeks to train a state-of-the-art workstation to achieve accuracies substantially worse than those we can achieve using a GPU and less than an hour of training. Thus, MNIST is no longer a problem that pushes the limits of available technique; rather, the speed of training means that it is a problem good for teaching and learning purposes. Meanwhile, the focus of research has moved on, and modern work involves much more challenging image recognition problems. In this section, I briefly describe some recent work on image recognition using neural networks.

We've focused on ideas likely to be of lasting interest - ideas such as backpropagation, regularization, and convolutional networks. Fashionable but whose long-term value is unknown are avoided.

**The 2012 LRMD paper:** Let me start with a [2012 paper](http://research.google.com/pubs/pub38115.html)(Note that the detailed architecture of the network used in the paper differed in many details from the deep convolutional networks we've been studying. Broadly speaking, however, LRMD is based on many similar ideas.) from a group of researchers from Stanford and Google. I'll refer to this paper as LRMD, after the last names of the first four authors. LRMD used a neural network to classify images from [ImageNet](http://www.image-net.org/), a very challenging image recognition problem. The 2011 ImageNet data that they used included 16 million full color images, in 20 thousand categories. The images were crawled from the open net, and classified by workers from Amazon's Mechanical Turk service. These is a 2014 dataset, which is somewhat changed from 2011. Qualitatively, however, the dataset is extremely similar. Details about ImageNet are available in the original ImageNet paper, [ImageNet: a large-scale hierarchical image database](http://www.image-net.org/papers/imagenet_cvpr09.pdf), by Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei (2009).

These are, respectively, in the categories for beading plane, brown root rot fungus, scalded milk, and the common roundworm. If you're looking for a challenge, I encourage you to visit ImageNet's list of [hand tools](http://www.image-net.org/synset?wnid=n03489162), which distinguishes between beading planes, block planes, chamfer planes, and about a dozen other types of plane, amongst other categories. This is obviously a much more challenging image recognition task than MNIST! LRMD's network obtained a respectable 15.8 percent accuracy for correctly classifying ImageNet images. That may not sound impressive, but it was a huge improvement over the previous best result of 9.3 percent accuracy. That jump suggested that neural networks might offer a powerful approach to very challenging image recognition tasks, such as ImageNet.

**The 2012 KSH paper:** The work of LRMD was followed by a [2012 paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) of Krizhevsky, Sutskever and Hinton (KSH). KSH trained and tested a deep convolutional neural network using a restricted subset of the ImageNet data. The subset they used came from a popular machine learning competition - the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC). Using a competition dataset gave them a good way of comparing their approach to other leading techniques. The ILSVRC-2012 training set contained about 1.2 million ImageNet images, drawn from 1,000 categories. The validation and test sets contained 50,000 and 150,000 images, respectively, drawn from the same 1,000 categories.

![KSH](http://neuralnetworksanddeeplearning.com/images/KSH.jpg)

That's an overview of many of the core ideas in the KSH paper. I've omitted some details, for which you should look at the paper. You can also look at Alex Krizhevsky's [cuda-convnet](https://code.google.com/p/cuda-convnet/) (and successors), which contains code implementing many of the ideas. A Theano-based implementation has also been developed([Theano-based large-scale visual recognition with multiple GPUs](http://arxiv.org/abs/1412.2302), by Weiguang Ding, Ruoyan Wang, Fei Mao, and Graham Taylor (2014).), with the code available [here](https://github.com/uoguelph-mlrg/theano_alexnet). The code is recognizably along similar lines to that developed in this chapter, although the use of multiple GPUs complicates things somewhat. The Caffe neural nets framework also includes a version of the KSH network, see their [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) for details.

**The 2014 ILSVRC competition:** Since 2012, rapid progress continues to be made. Consider the 2014 ILSVRC competition. As in 2012, it involved a training set of 1.2 million images, in 1,000 categories, and the figure of merit was whether the top 5 predictions included the correct category. The winning team, based primarily at Google - [Going deeper with convolutions](http://arxiv.org/abs/1409.4842), used a deep convolutional network with 22 layers of neurons. They called their network GoogLeNet, as a homage to LeNet-5. GoogLeNet achieved a top-5 accuracy of 93.33 percent, a giant improvement over the 2013 winner ([Clarifai](http://www.clarifai.com/), with 88.3 percent), and the 2012 winner (KSH, with 84.7 percent).

Since this work, several teams have reported systems whose top-5 error rate is actually better than 5.1%. This has sometimes been reported in the media as the systems having better-than-human vision. While the results are genuinely exciting, there are many caveats that make it misleading to think of the systems as having better-than-human vision. The ILSVRC challenge is in many ways a rather limited problem - a crawl of the open web is not necessarily representative of images found in applications! And, of course, the top-5 criterion is quite artificial. We are still a long way from solving the problem of image recognition or, more broadly, computer vision. Still, it's extremely encouraging to see so much progress made on such a challenging problem, over just a few years.

**Other activity:** I've focused on ImageNet, but there's a considerable amount of other activity using neural nets to do image recognition. Let me briefly describe a few interesting recent results, just to give the flavour of some current work.

One encouraging practical set of results comes from a team at Google, who applied deep convolutional networks to the problem of recognizing street numbers in Google's Street View imagery([Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/abs/1312.6082), by Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013).). In their paper, they report detecting and automatically transcribing nearly 100 million street numbers at an accuracy similar to that of a human operator.

I've perhaps given the impression that it's all a parade of encouraging results. Of course, some of the most interesting work reports on fundamental things we don't yet understand.  For instance, a 2013 paper([Intriguing properties of neural networks](http://arxiv.org/abs/1312.6199), by Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus (2013)) showed that deep networks may suffer from what are effectively blind spots. Consider the lines of images below. On the left is an ImageNet image classified correctly by their network. On the right is a slightly perturbed image (the perturbation is in the middle) which is classified incorrectly by the network. The authors found that there are such "adversarial" images for every sample image, not just a few special ones.

![Adversarial](http://neuralnetworksanddeeplearning.com/images/adversarial.jpg)

This is a disturbing result. The paper used a network based on the same code as KSH's network - that is, just the type of network that is being increasingly widely used. While such neural networks compute functions which are, in principle, continuous, results like this suggest that in practice they're likely to compute functions which are very nearly discontinuous. Worse, they'll be discontinuous in ways that violate our intuition about what is reasonable behavior. That's concerning. Furthermore, it's not yet well understood what's causing the discontinuity: is it something about the loss function? The activation functions used? The architecture of the network? Something else? We don't yet know.

Now, these results are not quite as bad as they sound. Although such adversarial images are common, they're also unlikely in practice. As the paper notes:

> The existence of the adversarial negatives appears to be in contradiction with the network’s ability to achieve high generalization performance. Indeed, if the network can generalize well, how can it be confused by these adversarial negatives, which are indistinguishable from the regular examples? The explanation is that the set of adversarial negatives is of extremely low probability, and thus is never (or rarely) observed in the test set, yet it is dense (much like the rational numbers), and so it is found near virtually every test case.

Nonetheless, it is distressing that we understand neural nets so poorly that this kind of result should be a recent discovery. Of course, a major benefit of the results is that they have stimulated much followup work. For example, one [recent paper](http://arxiv.org/abs/1412.1897) shows that given a trained network it's possible to generate images which look to a human like white noise, but which the network classifies as being in a known category with a very high degree of confidence. This is another demonstration that we have a long way to go in understanding neural networks and their use in image recognition.

Despite results like this, the overall picture is encouraging. We're seeing rapid progress on extremely difficult benchmarks, like ImageNet. We're also seeing rapid progress in the solution of real-world problems, like recognizing street numbers in StreetView. But while this is encouraging it's not enough just to see improvements on benchmarks, or even real-world applications. There are fundamental phenomena which we still understand poorly, such as the existence of adversarial images. When such fundamental problems are still being discovered (never mind solved), it is premature to say that we're near solving the problem of image recognition. At the same time such problems are an exciting stimulus to further work.

### Other approaches to deep neural nets

To classify the MNIST digits, We have usderstand many powerful ideas: stochastic gradient descent, backpropagation, convolutional nets, regularization, and more. But it's also a narrow problem. If you read the neural networks literature, you'll run into many ideas we haven't discussed: recurrent neural networks, Boltzmann machines, generative models, transfer learning, reinforcement learning, and so on, on and on $\dots$ and on! Neural networks is a vast field. However, many important ideas are variations on ideas we've already discussed, and can be understood with a little effort. In this section a glimpse of these is provided as yet unseen vistas.

**Recurrent neural networks(RNNs):** In the feedforward nets we've been using there is a single input which completely determines the activations of all the neurons through the remaining layers. It's a very static picture: everything in the network is fixed, with a frozen, crystalline quality to it. But suppose we allow the elements in the network to keep changing in a dynamic way. For instance, the behaviour of hidden neurons might not just be determined by the activations in previous hidden layers, but also by the activations at earlier times. That's certainly not what happens in a feedforward network. Or perhaps the activations of hidden and output neurons won't be determined just by the current input to the network, but also by earlier inputs.

Neural networks with this kind of time-varying behaviour are known as *recurrent neural networks* or *RNNs*. There are many different ways of mathematically formalizing the informal description of recurrent nets given in the last paragraph. You can get the flavour of some of these mathematical models by glancing at the [Wikipedia article on RNNs](http://en.wikipedia.org/wiki/Recurrent_neural_network). But mathematical details aside, the broad idea is that RNNs are neural networks in which there is some notion of dynamic change over time. And, not surprisingly, they're particularly useful in analysing data or processes that change over time. Such data and processes arise naturally in problems such as speech or natural language, for example.

One way RNNs are currently being used is to connect neural networks more closely to traditional ways of thinking about algorithms, ways of thinking based on concepts such as Turing machines and (conventional) programming languages. [A 2014 paper](http://arxiv.org/abs/1410.4615) developed an RNN which could take as input a character-by-character description of a (very, very simple!) Python program, and use that description to predict the output. Informally, the network is learning to "understand" certain Python programs. [A second paper, also from 2014](http://arxiv.org/abs/1410.5401), used RNNs as a starting point to develop what they called a neural Turing machine (NTM). This is a universal computer whose entire structure can be trained using gradient descent. They trained their NTM to infer algorithms for several simple problems, such as sorting and copying.

As it stands, these are extremely simple toy models. Learning to execute the Python program `print(398345+42598)` doesn't make a network into a full-fledged Python interpreter! It's not clear how much further it will be possible to push the ideas. Still, the results are intriguing. Historically, neural networks have done well at pattern recognition problems where conventional algorithmic approaches have trouble. Vice versa, conventional algorithmic approaches are good at solving problems that neural nets aren't so good at. No-one today implements a web server or a database program using a neural network! It'd be great to develop unified models that integrate the strengths of both neural networks and more traditional approaches to algorithms. RNNs and ideas inspired by RNNs may help us do that.

RNNs have also been used in recent years to attack many other problems. They've been particularly useful in speech recognition. Approaches based on RNNs have, for example, [set records for the accuracy of phoneme recognition](http://arxiv.org/abs/1303.5778). They've also been used to develop [improved models of the language people use while speaking](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf). Better language models help disambiguate utterances that otherwise sound alike. A good language model will, for example, tell us that "to infinity and beyond" is much more likely than "two infinity and beyond", despite the fact that the phrases sound identical.

This work is, incidentally, part of a broader use of deep neural nets of all types, not just RNNs, in speech recognition. For example, an approach based on deep nets has achieved [outstanding results on large vocabulary continuous speech recognition](http://arxiv.org/abs/1309.1501). And another system based on deep nets has been deployed in [Google's Android operating system](http://www.wired.com/2013/02/android-neural-network/) (for related technical work, see [Vincent Vanhoucke's 2012-2015 papers](http://research.google.com/pubs/VincentVanhoucke.html)).

It perhaps won't surprise you to learn that many of the ideas used in feedforward networks can also be used in RNNs. In particular, we can train RNNs using straightforward modifications to gradient descent and backpropagation. Many other ideas used in feedforward nets, ranging from regularization techniques to convolutions to the activation and cost functions used, are also useful in recurrent nets. And so many of the techniques we've developed can be adapted for use with RNNs.

**Long short-term memory units(LSTMs):** One challenge affecting RNNs is that early models turned out to be very difficult to train, harder even than deep feedforward networks. The reason is the unstable gradient problem discussed before. Recall that the usual manifestation of this problem is that the gradient gets smaller and smaller as it is propagated back through layers. This makes learning in early layers extremely slow. The problem actually gets worse in RNNs, since gradients aren't just propagated backward through layers, they're propagated backward through time. If the network runs for a long time that can make the gradient extremely unstable and hard to learn from. Fortunately, it's possible to incorporate an idea known as long short-term memory units (LSTMs) into RNNs. The units were introduced by [Hochreiter and Schmidhuber in 1997](http://dx.doi.org/10.1162/neco.1997.9.8.1735) with the explicit purpose of helping address the unstable gradient problem. LSTMs make it much easier to get good results when training RNNs, and many recent papers (including many that I linked above) make use of LSTMs or related ideas.

**Deep belief nets, generative models, and Boltzmann machines:** Modern interest in deep learning began in 2006, with papers explaining how to train a type of neural network known as a *deep belief network* (DBN)(See [A fast learning algorithm for deep belief nets](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf), by Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh (2006), as well as the related work in [Reducing the dimensionality of data with neural networks](http://www.sciencemag.org/content/313/5786/504.short), by Geoffrey Hinton and Ruslan Salakhutdinov (2006)). DBNs were influential for several years, but have since lessened in popularity, while models such as feedforward networks and recurrent neural nets have become fashionable. Despite this, DBNs have several properties that make them interesting.

One reason DBNs are interesting is that they're an example of what's called a generative model. In a feedforward network, we specify the input activations, and they determine the activations of the feature neurons later in the network. A generative model like a DBN can be used in a similar way, but it's also possible to specify the values of some of the feature neurons and then "run the network backward", generating values for the input activations. More concretely, a DBN trained on images of handwritten digits can (potentially, and with some care) also be used to generate images that look like handwritten digits. In other words, the DBN would in some sense be learning to write. In this, a generative model is much like the human brain: not only can it read digits, it can also write them. In Geoffrey Hinton's memorable phrase, [to recognize shapes, first learn to generate images](http://www.sciencedirect.com/science/article/pii/S0079612306650346).

A second reason DBNs are interesting is that they can do unsupervised and semi-supervised learning. For instance, when trained with image data, DBNs can learn useful features for understanding other images, even if the training images are unlabelled. And the ability to do unsupervised learning is extremely interesting both for fundamental scientific reasons, and - if it can be made to work well enough - for practical applications.

Given these attractive features, why have DBNs lessened in popularity as models for deep learning? Part of the reason is that models such as feedforward and recurrent nets have achieved many spectacular results, such as their breakthroughs on image and speech recognition benchmarks. It's not surprising and quite right that there's now lots of attention being paid to these models. There's an unfortunate corollary, however. The marketplace of ideas often functions in a winner-take-all fashion, with nearly all attention going to the current fashion-of-the-moment in any given area. It can become extremely difficult for people to work on momentarily unfashionable ideas, even when those ideas are obviously of real long-term interest. My personal opinion is that DBNs and other generative models likely deserve more attention than they are currently receiving. And I won't be surprised if DBNs or a related model one day surpass the currently fashionable models. For an introduction to DBNs, see [this overview](http://www.scholarpedia.org/article/Deep_belief_networks). I've also found [this article](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) helpful. It isn't primarily about deep belief nets, per se, but does contain much useful information about restricted Boltzmann machines, which are a key component of DBNs.

**Other ideas:** What else is going on in neural networks and deep learning? Well, there's a huge amount of other fascinating work. Active areas of research include using neural networks to do [natural language processing](http://machinelearning.org/archive/icml2008/papers/391.pdf) (see [also this informative review paper](http://arxiv.org/abs/1103.0398)), [machine translation](http://neuralnetworksanddeeplearning.com/assets/MachineTranslation.pdf), as well as perhaps more surprising applications such as [music informatics](http://yann.lecun.com/exdb/publis/pdf/humphrey-jiis-13.pdf). There are, of course, many other areas too.

Let me finish this section by mentioning a particularly fun paper. It combines deep convolutional networks with a technique known as reinforcement learning in order to learn to [play video games well](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (see also [this followup](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)). The idea is to use the convolutional network to simplify the pixel data from the game screen, turning it into a simpler set of features, which can be used to decide which action to take: "go left", "go down", "fire", and so on. What is particularly interesting is that a single network learned to play seven different classic video games pretty well, outperforming human experts on three of the games. Now, this all sounds like a stunt, and there's no doubt the paper was well marketed, with the title "Playing Atari with reinforcement learning". But looking past the surface gloss, consider that this system is taking raw pixel data - it doesn't even know the game rules! - and from that data learning to do high-quality decision-making in several very different and very adversarial environments, each with its own complex set of rules. That's pretty neat.

### [On the future of neural networks](http://neuralnetworksanddeeplearning.com/chap6.html#on_the_future_of_neural_networks)
