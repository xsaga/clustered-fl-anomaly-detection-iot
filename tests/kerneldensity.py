from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

bw entre 0 y percentil 99 y luego dividir entre 10 ? ...

In [30]: kde = KernelDensity(kernel="gaussian", bandwidth=0.00001).fit(resultsn_df["rec_err"].values.reshape(-1,1))
t = np.linspace(0, 0.002, 1000)
In [31]: log_den_t = kde.score_samples(t.reshape(-1,1))

In [32]: plt.plot(t, log_den_t); plt.show()