---
layout: post
title: "An Introduction to Linear Regression Using Haskell"
---

Having glossed over my Machine Learning classes during my BSc due to an ill-thought-out disinterest in the subject and a well-thought-out dislike for the professor teaching it, I've felt a little embarrassed as Machine Learning has become increasingly prominent in the popular and professional discourse while I've remained ignorant of it. I recently found myself with some time on my hands, so I thought it might be an idea to bring myself up-to-date on the subject using [Andrew Ng's excellent course on Coursera](https://www.coursera.org/learn/machine-learning).

However, I hit a bit of a snag - the course is based entirely on MATLAB (or, for those of who prefer our software free-as-in-everything, Octave). While MATLAB is a powerful tool that enables a lot of people we wouldn't classically consider programmers to be productive in this field, I find it quite frustrating as someone who comes from a more traditional Computer Science background. MATLAB is something of an idioglossia, obsessed with imperative operations on nested matrices, and I don't find my existing knowledge of programming very applicable. It's also pretty limited in scope - while it's very powerful for performing linear algebra, it's woefully ill-equipped for parsing input, making RPC calls, or interacting with databases; all of which are potential dealbreakers in a modern environment.

So I decided to stray a little off the beaten track and implement the programming exercises from Andrew's course in Haskell instead. Haskell feels well-suited for this problem: it has powerful vector libraries that should be roughly competitive with the optimised linear algebra libraries available in languages with larger Data Science ecosystems, and it's extremely expressive, so hopefully our implementations should be pretty readable. Let's find out!

# What Is Linear Regression?

  

