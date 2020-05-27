## *Big-Five Personality Categorization via Autoencoders*

### Terry Wang

**Abstract**



**Background**

Personality testing, or the search for orthogonal dimensions that explain the more nebulous concept of "personality", is a fascinating field with a lot of competing and compelling theories. The most well-known implimentation of such theories would be the Myers-Briggs Type Indicator (MBTI), which, while influential, is beset with criticism of lack of scientific vigor in its development. Many academic psychology researchers have proposed the Big-Five personality traits of openness, conscientiousness, extroversion, agreeableness, and neuroticism. Subsequently, a questionnaire was developed as a standardized way to capture the Big Five factors via adjectives that supposedly reflect these factors to varying degrees. (https://openpsychometrics.org/tests/IPIP-BFFM/) This questionnaire is now an open project online and free to the public, and the data from this project is available at Kaggle (https://www.kaggle.com/tunguz/big-five-personality-test).

While there is no doubt that the Big-Five theory is developed with a considerable amount of scientific vigor, it is still a categorization that arises from human intuition that may or may not reflect the actual mechanisms of how personalities inform behavior via self-description. Compared to a more "natural" clustering approach, which I will elaborate further in this post, the Big-Five theory pre-assigns the five orthogonal dimensions of personality, and then try to capture those dimensions through the use of purposeful adjectives that reflect those dimensions. For example, to capture the Neuroticism dimension, the survey would use words like "irritated", "stressed", and "worry", which according to research more or less capture these dimensions specifically (See Goldberg 1992, https://psycnet.apa.org/doiLanding?doi=10.1037%2F1040-3590.4.1.26). A potential issue is that this approach assumes that these **particular** five dimensions would explain the answers obtained from the surveys, meaning that if the theory is not correct, i.e. some of these big-five traits are not orthogonal to each other, the model would prove much less meaningful.

The idea that inspired this post is the following: what if we reverse this thought process, and design a survey that captures a wide range of adjectives, then use machine learning algorithms (or Bayesian methods) to come up with a much smaller number of orthogonal dimensions of personalities? Similar to the Big-Five approach, this uses a mental model where a few latent dimensions would beget the answers in the survey. But to the contrary of that approach, we would not pre-suppose what those dimensions are, in human concepts, but attempt to interpret those dimensions **after** we have obtained them. Moreover, we are not limited by the number of dimensions, but we can try to fit the model with many different number of latent dimensions to see which ones lead to accurate and more interpretable results. I believe this is a viable approach given the advances in computing power since the days Big-Five was developed, and it might lead to some surprise findings in human psychology. 

**Data**

The data is a collection of just over 1 million survey answers to a standardized set of 50 questions designed to measure 5 different personality traits mentioned in the previous section. This data can be found [here](https://www.kaggle.com/tunguz/big-five-personality-test). There are 110 columns in total, 50 of which are answers to the 50 survey questions on the scale of 1-5 (Likert scale), another 50 of which are how much time was spent on each question, and the rest are metadata such as country, screen size, etc.

There are about 2000 rows where all columns are missing data and were deleted. Additionally, about 13% of the rows include answers that are 0 instead of 1-5. This is most likely due to the respondent failing to provide any answer to that question (i.e. missing response), but due to computational constraints, I have decided to not include any rows with such answers. I also decided to only keep the 50 columns recording the answers in Likert scale, as the information from the other columns are out of scope of this investigation, so finally I am left with a dataset with 877434 rows and 50 columns.

**Method**

Let's set up the following mental model that represents how the theory of personality types work:

![Personality Type Mental Model](graphs/Model.png)

This graph basically says there are a small number of personality types or traits (in this case there are five) that are orthogonal or independent of each other. These personality traits could each be a random variable or just a number, depending on how you wish to model them. According to this model, each person would have a different set of (5) numbers to represent their personality measurements. Then, these traits will go through a transformation process, which can involve adding them, weighing them, and multiplying them, etc., that ultimately produce a vector representing the answers a person would give on the personality test, i.e. the answers to the 50 questions on the scale of 1-5.  

If this model is an accurate representation of the natural process through which humans answer personality survey questions, then theoretically we will be able to backtrack from the answers to their personality types that resulted in these particular answers in the first place. 

Ostensibly this is exactly the underlying model 



**Data**



**Model**



**Latent Dimension Interpretation**



**Conclusions and Next Steps**

















https://openpsychometrics.org/tests/IPIP-BFFM/

https://ipip.ori.org/new_ipip-50-item-scale.htm

https://www.kaggle.com/tunguz/big-five-personality-test