---
title: On toy models
category: Research directions
indexing: true
comments: true
excerpt: On the utility of toy models; kidney donations and porous materials with rotating ligands.
author: Cory Simon
---

There are currently ~95,000 people waiting for a kidney transplant in the US. In 2017, 30% of the 18,000 kidney donations were from living donors. [1]

Imagine person W needs a kidney transplant. Humans are normally born with two kidneys and typically can live a healthy life with only one kidney. Consequently, the husband H of person W is willing to donate his kidney to his wife W to save her life.

However, several biological conditions, e.g. blood type compatibility, must be met to ensure that a donated kidney will not be rejected when transplanted. The probability of an exact match between two unrelated people is a daunting 1 in 100,000 [2]. So, it is very unlikely that husband H is able, despite willing, to donate his kidney to his wife W.

While many people would be willing to donate his/her kidney to a loved one, as husband H is willing, it is much less likely that a given person will volunteer to donate his/her kidney to a random other person in need of a kidney. The wait list for kidneys is long.

For good reason, it is illegal for kidneys to be sold on the market, so buying a kidney for wife W is not an option.

Alvin Roth helped devise a market to increase the rate of kidney donations. 
Along with Lloyd Shapley, he was awarded the Nobel Prize in Economic Science in 2012 for "for the theory of stable allocations and the practice of market design".

The key idea is as follows. Many other couples are in the same situation as husband H and wife W. For example, there is a couple (H', W'), where the wife W' needs a kidney transplant, her husband H' is willing to donate his kidney, and he is not a match to W'. However, if H' is a match to W and H is a match to W', two simultaneous kidney transactions can occur, whereby H' donates his compatible kidney to W and H donates his compatible kidney to W'. Both H and H' donated their kidney in order to, albeit indirectly, save their loved one. This was their incentive to donate their kidney to an unknown person.

Ruthanne Leishman remarked on Alvin Roth's matching algorithm on the Freakonomics podcast [3]:

> It’s saving a lot of lives. We have about 600 kidney-paired donation transplants a year right now in the United States. In 2000 we had 2. We would have stayed doing 2 or 4 or 6 a year without the algorithm.

Interestingly, Alvin Roth didn't start off studying kidney transplants. Instead, he was curious about the theoretical question of how to exchange indivisible goods when money cannot be used. It was, in the words of Roth, "entirely theoretical" [3]. To present his work, he used the example of houses and hypothetically assumed that money could not be used:

> We were talking about how to trade houses, and of course, no one trades houses without money. I can tell you, I’ve just bought a house in California and money played a role.

On the Freakonomics podcast [3], Roth states:

> Mathematical economists learn about things is a little bit [like] the way children learn about things. You find toys to play with and then by playing with the toys you gain experiences that might help you with other things. This is a toy. This toy model that allows you to think about the question of how to trade goods when you can’t use money and when you can’t divide the good. You can’t say, “You have a big house and I have a little house, so just give me half of your house for my house.” You say, “Houses are indivisible, we have to trade.”

I am sharing this story for two reasons. First, this underscores how curiosity-driven research can lead to transformative ideas-- step changes in the incumbent. Second, Roth's fondness for toy models resonates with me.

I find toy models incredibly helpful to bring clarity to a problem. In 2015, we [me, Braun, Carraro, and Smit] developed a [toy model](http://dx.doi.org/10.1073/pnas.1613874114) [4] for a porous material that possesses rotating ligands. Our goal was to intimately understand the physical implications of rotating ligands on how gas adsorbs into such a material, as a consequence of the rotating ligands.

{:.center}
<figure>
    <img src="/images/rotatingligandsmodel.png" alt="image" style="width: 70%;">
    <figcaption>The squares represent cages of a porous material where gas molecules (green balls) can sit. The purple bars are rotating ligands that can adopt one of two rotational conformations: rotated into the pore or parallel to the cage walls. The affinity of a gas molecule for a cage depends on the states of its two ligands.</figcaption>
</figure>

As the figure above illustrates, there is very little detail about the porous material built into the model. The porous material has cages where a gas molecule can adsorb. The energy of adsorption depends on how many ligands are rotated into the pore (zero, one, or two). There is an energetic penalty for the ligand to rotate into the pore. That's the entire model.

This deceptively simple model for a porous material with rotating ligands, however, gives rise to rich behavior [4]; remarkably, it captures much of the physics exhibited by materials with rotating ligands. The model gives rise to inflections and steps in the adsorption isotherm, alignment of the ligands as gas adsorbs, and an offset to the heat of adsorption from the energetic penalty for the ligand to change its rotational conformation. 

Playing with the toy model-- changing the parameters and observing how the adsorption isotherm, ligand alignments, and heat of adsorption changed-- helped me intimately understand the consequence of rotating ligands on gas adsorption; by publishing our article, we hope it will help the porous materials community harness rotating ligands for more effective adsorption-based engineering processes.

The model particularly sheds light on how to optimize the chemistry of materials with rotating ligands to maximize their working capacity for gas storage (for example, utilizing porous materials to store natural gas or hydrogen onboard vehicles for fuel). Real materials consist of molecular building blocks; while our model lacks any concrete notion of which molecular building blocks can give rise to an optimal material, it does reveal qualitatively e.g. to choose more or less attractive functional groups or make it easier for the ligand to rotate.

Finally, while we developed the model with the aim of understanding the implications of rotating ligands for gas storage in porous materials, the model gave rise to interesting thermodynamic behavior that I doubt I would have predicted without the experience of playing with the model. Depending on how much gas the porous material is holding, heating a material with rotating ligands can have opposite effects on the conformations of the ligands. If the material is full of gas, heat can induce more ligands to align with the cage walls; if the material has little gas adsorbed, heat can induce more ligands to rotate into the pores. I don't know if this property could be exploited for an engineering process, but it is a nice example of how a simple model can reveal interesting, unintuitive, and perhaps useful qualitative insights. For our model, the insight surrounds the implication of rotating ligands on the adsorption behavior of porous materials.

Because I find toy models so illuminating and helpful, I plan to continue developing toy models in my research.

## References

1. https://optn.transplant.hrsa.gov/
2. http://www.ucdmc.ucdavis.edu/transplant/learnabout/learn_hla_type_match.html
3. http://freakonomics.com/2015/06/17/110307
4. ❝Statistical mechanical model of gas adsorption in porous crystals with dynamic moieties❞ C. Simon, E. Braun, C. Carraro, B. Smit. Proceedings of the National Academy of Sciences. (2017)
