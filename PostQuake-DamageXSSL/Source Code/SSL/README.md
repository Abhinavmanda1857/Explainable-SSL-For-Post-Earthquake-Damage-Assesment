## SSL Pipeline Procedure (EMA-Teacher)

1. **Data Split:** Partition the xBD dataset into 10% Labeled and 90% Unlabeled subsets.
2. **Model Init:** Deploy a Twin-ViT setup with an active **Student** model and a gradient-frozen **Teacher** model.
3. **Supervised Loss:** Calculate Cross-Entropy loss by passing the 10% labeled data through the Student.
4. **Consistency Loss:** The Teacher generates pseudo-labels on the 90% unlabeled data; the Student predicts the same data to calculate Mean Squared Error (MSE).
5. **Optimization:** Combine both losses and backpropagate to update the Student's weights.
6. **EMA Update:** Synchronize the Teacher's weights smoothly using an Exponential Moving Average (EMA) of the Student.
