## MMadness2021

Simple kaggle submission using `tensorflow-probability`. Creates a model that works for both the predict-the-winner competition and the predict-the-spread competition.

The model is a variant of the "epistemic and aleatoric uncertainty" example from tfp's walkthrough. It learns a variational model that captures the uncertainty in both the model weights and the underlying data distribution.

The result scored 11th in the spreads competition, though a middle-of-the-road 300th or so in the winner competition. This could be because it did not directly optimize `log_loss` at all.

The model, with both its simplicity and its uncertainty baked in, was extremely robust to overfitting, to the point where almost all runs converged to a reasonable `0.50 to 0.57` output depending on the holdout season.
