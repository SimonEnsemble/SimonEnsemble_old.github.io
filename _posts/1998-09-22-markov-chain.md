---
title: Markov Chains
category:
indexing: true
comments: true
excerpt: Markov Chains are powerful estimation tools that give a probability distribution for a next state only based on the current state. They are especially useful for understanding multivariate systems where an analytical solution cannot be reached.
author: Arthur York
---

## Markov Chains

Markov chains are useful probability tools that work by defining real world phenomenon as a series of states and the transitions between them. Given the current state only, a Markov chain has enough information to create a probaility distribution of the states that can immediately follow. To better understand them, we will work with an imaginary town that experiences four types of weather: sunny, cloudy, rainy, or snowy. Weird as it may be, the most accurate forecast for any day is based on the previous day's weather. The graph shown in figure 1 is the model for the weather in this town.

### The Markovian Property: Building Blocks of Simulation

In order for a model to accurately depict a Markov chain, that model must follow the Markovian property. This property states that any given state holds all of the information from previous states in order to make a prediction. This is because a Markov chain is memoryless and only "remembers" the current state. As soon as it transitions it forgets where it has been. This means that a Markov chain must be designed in such a way that every state holds all of the information it needs to predict the next state.

The model shown in figure 1 for our weather example has the Markovian property because each state holds enough information to predict the next state. The only relevant information for determing the next day's weather is the current day's weather, so that is all we need to track in our model. If the weather in this town is actually dependent on the previous week of weather, the above model would no longer hold the Markovian property. Our model only keeps information from the previous day and we need the previous seven, therefore the current state does not contain the necessary information to predict the next state and fails to satisfy the Markovian property. We can modify our system to track all of these transitioning states, but that would quickly become unwieldy.

### Implementing a Markov Chain

We have established that our Markov chain model in figure 1 holds the Markovian property, so we can move ahead and implement it. 

The defining aspect of a Markov Chain is the Markovian property: it only needs the information from the current state to show all states that can immediately follow. The Markovian property is built on the idea that the given state summarizes everything about the past that is relevant to the future. Our model has the markovian property because a single day (or state) holds all the information we need to predict the next day (or provide a probability distribution of the states that follow).


