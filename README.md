# Nonlinear State Estimation via Machine Learning
The present repository has been used in relation with a Master's Thesis at the Technical University of Denmark, DTU.

This project was prepared in the Department of Civil and Mechanical Engineering at the Technical University of Denmark as part of the fulfillment of the requirements
for acquiring a Master's degree in Civil Engineering.

The project accounts for 30 ECTS from each of the two authors, and have been carried out in the period between August 29th 2022 and January 29th 2023.

**Title:** Nonlinear State Estimation via Machine Learning <br />
**Authors:** Gabriele Mauro s202277 & Lars K. Fogh s163761 <br />
**University:** Technical University of Denmark <br />

The full report of the thesis is avalable from: [Nonlinear State Estimation via Machine Learning](MSc_Thesis_Nonlinear_State_Estimation_Intro.pdf)


## Abstract
The following thesis focuses on the topic of state estimation via machine learning. Here, state estimation refers to the prediction of acceleration responses from a reinforced concrete frame structure which has been excited with strong ground motions. The main objective has been to predict responses in one part of the structure given input responses from a different location.

An initial number of 301 ground motions have been selected and scaled such that a total number of 903 ground motions have been used to excite the structure. The structure has been modelled using OpenSees, from where the responses have been recorded in all free nodes. The materials defined in the model follow nonlinear constitutive law, therefore, the structure might respond in the non-linear regime. This implies that the generated responses can be associated with a structure operating either in the linear or non-linear domain, according to the intensity of the exciting ground motion.

Two machine learning models, namely a Gaussian Process (GP) and a Neural Network (NN), have then been trained and tested with the obtained structural responses in order for the models to predict accelerational time series responses. From the predictions of the two models, it has been seen that it is possible to predict the responses given an input response from a different location. Especially when the inputs are located on the same floor as the predictions, they then are most accurate. When the input and predictions are to be on different floors, predictions are still possible but less accurate. The NN performs, in general, better than the GP, both in terms of accuracy of the predictions as well as computational time. 

Even when noise is added to the input responses, the NN is able to make predictions with only slight implications on the accuracy. The NN further scales well in terms of training data. However, the NN does not give a deep insight into any physical relationship between the input responses and predictions. On the contrary, the GP does not scale well in terms of training data, and is therefore limited to smaller sets that must be selected based on the matching characteristics of the predicted responses. However, when using the GP, the predictions are associated with uncertainty measures.

**Keywords:** Gaussian Process, Neural Network, State estimation, Response prediction, Non-linear responses, OpenSees.
 
 
## Citation

    @Misc{NSEML2023,
      author =   {G. Mauro and L. K. Fogh},
      title =    {Nonlinear State Estimation via Machine Learning. Technical University of Denmark},
      howpublished = {\url{https://github.com/s163761/Nonlinear_State_Estimation_ML}},
      year = {2023}
    }
