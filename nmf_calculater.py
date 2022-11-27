import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans, DBSCAN


def get_nmf_matrix(dataframe, n_components = 20) -> pd.DataFrame:
    nmf_model = NMF(n_components=n_components, max_iter=300)
    nmf_model.fit(dataframe)
    Q_matrix = pd.DataFrame(nmf_model.components_, columns=list(dataframe.columns), 
                    index=[f"cluster_{i+1}" for i in range(n_components)])
    P_matrix = pd.DataFrame(nmf_model.transform(dataframe),
                index=dataframe.index,
                columns = [f"cluster_{i+1}" for i in range(n_components)])
    R_hat = pd.DataFrame(np.dot(P_matrix, Q_matrix), 
                index=dataframe.index,
                columns=list(dataframe.columns))
    
    pickle.dump(nmf_model,open(f'./models/nmf_{n_components}.sav', "wb"))
    print("Model saved in models/ folder")
    
    print(f"Reconstruction Error : {nmf_model.reconstruction_err_}")

    return Q_matrix