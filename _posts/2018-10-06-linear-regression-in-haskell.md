---
layout: post
title: "An Introduction to Linear Regression Using Haskell"
---

Having glossed over my Machine Learning classes during my BSc due to an ill-thought-out disinterest in the subject and a well-thought-out dislike for the professor teaching it, I've felt a little embarrassed as Machine Learning has become increasingly prominent in the popular and professional discourse while I've remained ignorant of it. I recently found myself with some time on my hands, so I thought it might be an idea to bring myself up-to-date on the subject using [Andrew Ng's excellent course on Coursera](https://www.coursera.org/learn/machine-learning).

However, I hit a bit of a snag - the course is based entirely on MATLAB (or, for those of who prefer our software free-as-in-everything, Octave). While MATLAB is a powerful tool that enables a lot of people we wouldn't classically consider programmers to be productive in this field, I find it quite frustrating as someone who comes from a more traditional Computer Science background. MATLAB is something of an idioglossia, obsessed with imperative operations on nested matrices, and I don't find my existing knowledge of programming very applicable. It's also pretty limited in scope - while it's very powerful for performing linear algebra, it's woefully ill-equipped for parsing input, making RPC calls, or interacting with databases; all of which are potential dealbreakers in a modern environment.

So I decided to stray a little off the beaten track and implement the programming exercises from Andrew's course in Haskell instead. Haskell feels well-suited for this problem: it has powerful vector libraries that should be roughly competitive with the optimised linear algebra libraries available in languages with larger Data Science ecosystems, and it's extremely expressive, so hopefully our implementations should be pretty readable. Let's find out!

# What Is Linear Regression?

In its simplest form, Linear Regression tries to model the relationship between continuous values (i.e. numbers) by assuming that one is a linear function of the other (i.e. they have the relationship `y = mx + c`). We can then find values for `m` and `c` that model what we observe our training data well. We will do this using gradient descent. Later, we will see how gradient descent has some nice properties that let us generalise our implementation to functions of more than one variable or which are not purely linear in nature.   

# OK, So What's Gradient Descent?

Before we can use Linear Regression to model data, we need a reliable way to work out what `m` and `c` actually are. Gradient descent depends on two simple observations:
  1. Functions (or at least, continuous differentiable functions, which are enough to be getting on with) have minima at points where all the partial derivatives are zero. For the simple case of `y = mx + c`, that means we have $$ \frac{dy}{dx} = 0 $$. For more complex functions of the form $$ f(x, y) $$, we require that $$ \frac{d}{dx} = \frac{d}{dy} = 0 $$, and so on.
  2. If we go in the direction of negative gradients, we will eventually reach a minimum, assuming one exists (for us not to do so, the function would need to be non-continuous or of unbounded domain, which we will explain away by placing safely out of scope).

Gradient descent works very simply by calculating the gradient of our function for some value of `x` and then moving in whichever direction has negative gradient until we reach a minimum (which we can easily identify, because we will no longer be moving at all - the gradient will be zero)

Thus, if we can construct some continuous differentiable function of `m` and `c` which is at a minimum when `m` and `c` best describe our data, we can apply gradient descent to find out what those values actually are. Let's do that. (This function will end up being called the 'Cost Function')

I'm going to gloss the actual derivation, because I'm not a strong enough mathematiciation to add value. If you're interested, I'd encourage you to read [Chris McCormick's Derivation](http://mccormickml.com/2014/03/04/gradient-descent-derivation/), but if you're not, you can just take my word for the formulae and read the implementations.

However you choose to get there, you'll arrive at the following:

Our function for estimating `y` at a given point is now given as $$ h_\theta $$

The cost function we seek to minimise is $$ J(\theta) = \frac{1}{m} \sum_{i=1}^m (i) $$ J

