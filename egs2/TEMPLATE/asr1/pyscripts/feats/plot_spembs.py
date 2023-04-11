from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from kaldiio import ReadHelper, WriteHelper


with ReadHelper("scp:dump/raw/org/train/xvector.scp") as reader:
  vecs = []
  spk2sid={}
  spkrs=[]
  
  for utt_id, spemb in reader:
    spkr=utt_id.split("-")[0]

    if spkr not in spk2sid:
      spk2sid[spkr] = len(spk2sid)
    
    spkrs += [spk2sid[spkr]]
    vecs += [np.expand_dims(spemb,0)]
  print(vecs[-1].shape)
  vecs = np.concatenate(vecs, 0).squeeze()

  pca=PCA(n_components=12)
  vecs = pca.fit_transform(vecs)

  print(vecs.shape)
    
  transformed = TSNE().fit_transform(vecs)
  scatter = plt.scatter(transformed[:,0], transformed[:,1], c=spkrs)
  handles, _ = scatter.legend_elements(prop='colors')
  plt.savefig(f"myplot.png")

  plt.legend(handles, spkrs)