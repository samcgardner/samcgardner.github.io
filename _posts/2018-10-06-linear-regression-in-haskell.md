---
layout: post
title: "An Introduction to Linear Regression Using Haskell"
---

Having glossed over my Machine Learning classes during my BSc due to an ill-thought-out disinterest in the subject and a well-thought-out dislike for the professor teaching it, I've felt a little embarrassed as Machine Learning has become increasingly prominent in the popular and professional discourse while I've remained ignorant of it. I recently found myself with some time on my hands, so I thought it might be an idea to bring myself up-to-date on the subject using [Andrew Ng's excellent course on Coursera](https://www.coursera.org/learn/machine-learning).

However, I hit a bit of a snag - the course is based entirely on MATLAB (or, for those of who prefer our software free-as-in-everything, Octave). While MATLAB is a powerful tool that enables a lot of people we wouldn't classically consider programmers to be productive in this field, I find it quite frustrating as someone who comes from a more traditional Computer Science background. MATLAB is something of an idioglossia, obsessed with imperative operations on nested matrices, and I don't find my existing knowledge of programming very applicable. It's also pretty limited in scope - while it's very powerful for performing linear algebra, it's woefully ill-equipped for parsing input, making RPC calls, or interacting with databases; all of which are potential dealbreakers in a modern environment.

So I decided to stray a little off the beaten track and implement the programming exercises from Andrew's course in Haskell instead. Haskell feels well-suited for this problem: it has powerful vector libraries that should be roughly competitive with the optimised linear algebra libraries available in languages with larger Data Science ecosystems, and it's extremely expressive, so hopefully our implementations should be pretty readable. Let's find out!

# What Is Linear Regression?

In its simplest form, Linear Regression tries to model the relationship between continuous values (i.e. numbers) by assuming that one is a linear function of the other (i.e. they have the relationship $$ y = \theta_{0} + \theta_{1}x $$). We can then find values for $$ \theta_{0} $$ and $$ \theta_{1} $$ that model what we observe in our training data well. We will do this using gradient descent. Later, we will see how gradient descent has some nice properties that let us generalise our implementation to functions of more than one variable or which are not purely linear in nature.   

# OK, So What's Gradient Descent?

Before we can use Linear Regression to model data, we need a reliable way to work out what  $$ \theta_{0} $$ and $$ \theta_{1} $$ actually are. Gradient descent depends on two simple observations:
  1. Functions (or at least, continuous differentiable functions, which are enough to be getting on with) have minima at points where all the partial derivatives are zero. For the simple case of $$ y = mx + c$$, that means we have $$ \frac{dy}{dx} = 0 $$. For more complex functions of the form $$ f(x, y) $$, we require that $$ \frac{d}{dx} = \frac{d}{dy} = 0 $$, and so on.
  2. If we go in the direction of negative gradients, we will eventually reach a minimum, assuming one exists (for us not to do so, the function would need to be non-continuous or of unbounded domain, which we will explain away by placing safely out of scope).

Gradient descent works very simply by calculating the gradient of our function for some value of $$x$$ and then moving in whichever direction has negative gradient until we reach a minimum (which we can easily identify, because we will no longer be moving at all - the gradient will be zero)

Thus, if we can construct some continuous differentiable function of $$ \theta_{0} $$ and $$ \theta_{1} $$ which is at a minimum when $$ \theta_{0} $$ and $$ \theta_{1} $$ best describe our data, we can apply gradient descent to find out what those values actually are. Let's do that. (This function will end up being called the 'Cost Function')

I'm going to gloss the actual derivation, because I'm not a strong enough mathematician to add value. If you're interested, I'd encourage you to read [Chris McCormick's Derivation](http://mccormickml.com/2014/03/04/gradient-descent-derivation/), but if you're not, you can just take my word for the formulae and read the implementations.

However you choose to get there, you'll arrive at the following:

Our function for estimating $$y$$ at a given point is now given as $$ h_\theta $$

The cost function we seek to minimise is

$$ J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 $$ 

Or, in plain English, the average of the squared difference between our prediction and the actual value. Using the formula for gradient descent, we will end up with the following update formulae for $$ \theta_{0} $$ and $$ \theta_{1} $$:

$$ \theta_{0} := \theta_{0} - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) $$


$$ \theta_{1} := \theta_{1} - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

Which we can then iterate until convergence. 

# OK, Enough Maths. Show Me Some Code!
First, let's define some type aliases for readability.

```haskell
module Types where

{- A type wrapper for the coefficients for our estimation function
i.e. theta0 and theta1 -}
newtype Coefficients =
  Coefficients (Float, Float)
  deriving (Show)

{- A type wrapper for a data point in the training set, 
with the first member being the x value and the second the y value -} 
newtype Example =
  Example (Float, Float)

-- A type wrapper for a training set, which is just a list of examples
newtype TrainingSet =
  TrainingSet [Example]
```

Now we have some types, let's write the meat of our function

```haskell
module Lib where

import Data.List
import Types

-- Finds coefficients for linear regression using gradient descent
linearRegression :: Coefficients -> Float -> TrainingSet -> Int -> Coefficients
linearRegression coefficients alpha dataset iterations
  | iterations == 0 = coefficients
  | otherwise =
    let thetas = newThetas coefficients alpha dataset
     in linearRegression thetas alpha dataset (iterations - 1)

-- Calculate new values for t0 and t1
newThetas :: Coefficients -> Float -> TrainingSet -> Coefficients
newThetas thetas alpha dataset =
  let deltas = map (calculateDelta thetas) examples
      adjustedDeltas = adjustDeltas deltas examples
      newt0 = t0 - 0.5 * alpha * avg deltas
      newt1 = t1 - 0.5 * alpha * avg adjustedDeltas
   in Coefficients (newt0, newt1)
  where
    Coefficients (t0, t1) = thetas
    TrainingSet examples = dataset

-- Calculates the difference between h(x) and y
calculateDelta :: Coefficients -> Example -> Float
calculateDelta thetas example = t0 + t1 * x - y
  where
    Coefficients (t0, t1) = thetas
    Example (x, y) = example

-- For the case where d/dx != 1 and we need to multiply through, do so 
adjustDeltas :: [Float] -> [Example] -> [Float]
adjustDeltas deltas examples =
  let xs = map (\(Example (x, _)) -> x) examples
      zipped = zip deltas xs
   in map (uncurry (*)) zipped

avg xs = realToFrac (sum xs) / genericLength xs
```

Here we provide an implementation for the very simple case of a linear relationship. To get a basic example working, this is all we need. We'll look at some extensions to this later.

# The Prestige
Let's write a quick `main` function that calls what we've written with some dummy data to check it's doing what we expect.

```haskell
module Main where

import Types

main :: IO ()
main = do
  let thetas = Coefficients (0, 0)
  let alpha = 1
  let trainingset =
        TrainingSet [Example (0, 50), Example (1, 60), Example (2, 70)]
  let iterations = 500
  print $ show $ linearRegression thetas alpha trainingset iterations
```

```haskell
*Main> main
"Coefficients (49.999992,10.000006)"
```

Which, I hope you'll agree, is pretty damn close to our training data.

# Final Thoughts
There's a lot more I'd like to cover here, but this is pretty long and pretty dense. I'll cover the various improvements we'd need to make to this to cover other common use-cases for Linear Regression in another post. If you'd like to see the full project for this example, it's available on [my GitHub Profile](https://github.com/samcgardner/linear-regression).
