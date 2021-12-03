# tensorflow_latent_rep_map2_UMAP

What is needed?

Project dataB onto the UMAP of dataA.


Attempted solution:

dataA and dataB have been combined using the package scAEspy (https://gitlab.com/cvejic-group/scaespy)

To visualise dataB on the UMAP of dataA, a simple fully connected deep learning model was trained to try and map the latent vectors back to the UMAP of dataA.

Conclusion:

There might not be enough training data to properly train the models.

