---
title: Markov Chains
category:
indexing: true
comments: true
excerpt: Markov Chains are powerful estimation tools that give a probability distribution for a next state only based on the current state. They are especially useful for understanding multivariate systems where an analytical solution cannot be reached.
author: Arthur York
---

Markov chains are useful probability tools that work by defining real world phenomenon as a series of states and the transitions between them. Given the current state only, a Markov chain has enough information to create a probaility distribution of the states that can immediately follow. To better understand them, we will work with an imaginary town that experiences four types of weather: sunny, cloudy, rainy, or snowy. Weird as it may be, the most accurate forecast for any day is based on the previous day's weather. The graph shown in figure 1 is the model for the weather in this town.

## The Markovian Property: Building Blocks of Simulation

In order for a model to accurately depict a Markov chain, that model must follow the Markovian property. This property states that any given state holds all of the information from previous states in order to make a prediction. This is because a Markov chain is memoryless and only "remembers" the current state. As soon as it transitions it forgets where it has been. This means that a Markov chain must be designed in such a way that every state holds all of the information it needs to predict the next state.

The model shown in figure 1 for our weather example has the Markovian property because each state holds enough information to predict the next state. The only relevant information for determing the next day's weather is the current day's weather, so that is all we need to track in our model. If the weather in this town is actually dependent on the previous week of weather, the above model would no longer hold the Markovian property. Our model only keeps information from the previous day and we need the previous seven, therefore the current state does not contain the necessary information to predict the next state and fails to satisfy the Markovian property. We can modify our system to track all of these transitioning states, but that would quickly become unwieldy.

## Implementing a Markov Chain

We have established that our Markov chain model in figure 1 holds the Markovian property, so we can move ahead and implement it. 

Our model is a weighted directed graph so I will be using an adjacency matrix to implement it in julia. The following code snippet is a 4x4 adjacency matrix that represents the graph from figure 1. Each column corresponds to a different current state: 1 is sunny, 2 is cloudy, 3 is rainy, and 4 is snowy. The sum across each column is equal to 1 because at any given state it MUST transition to another state (this can just be back to itself).

I chose to make the columns sum to one because julia is a column major language. While this is a small simulation, it makes more sense for the related probabilities to be stored close in memory.

{% highlight julia %}
const markov_model = [ 0.5 0.3 0.2 0.0
                       0.4 0.2 0.3 0.2
                       0.1 0.4 0.3 0.7
                       0.0 0.1 0.2 0.1 ]
{% endhighlight %}

We can now use this matrix to determine to calculate probability distributions. Let's say that today (the current state) is cloudy (column 3), and we want to create a forecast for tomorrow's weather. We can look down the third column in the matrix to find the probabilities of seeing each type of weather tomorrow. The model shows there is a 20% chance of it being sunny, 30% of it staying cloudy, 30% chance it will rain, and a 20% chance it will snow. Figure 2 shows the possible transitions with weights that this Markov chain model can make from the "cloudy" state.

## Markov Chain Monte Carlo Algorithms

Suppose now that our fictional town is competing for the title of rainiest town in the United States, but since they only remember the previous day's weather they haven't kept any records! They want to find the average number of days each year that their town experiences rain but their model doesn't show that. 
