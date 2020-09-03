import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#meanは平均μ,covは分散共分散行列∑
mean1 = np.array([10,5])
cov1 = np.array([[1,2],[2,5]])

#N(μ,Σ)に従う正規乱数を100個生成 https://docs.scipy.org/doc//numpy-1.15.0/reference/generated/numpy.random.multivariate_normal.html
N = 100
x1 = np.random.multivariate_normal(mean1, cov1, N)

#生成したデータの標準化
data = (x1-np.mean(x1, axis=0)) / np.std(x1, axis=0)
#分散共分散行列の生成(標準化後)
X = np.cov(data, rowvar=False)

#特異値分解SVD(スクラッチ)------------------------------------------------------------------------

#対角行列Ds,左特異行列Us,右特異行列Vsと置き,順に計算をする.

#特異値はXX.Tの固有値の平方根なのでまずXX.Tの固有値,固有ベクトルを求める
XXT = np.dot(X,X.T)
XXTvalues , XXTvectors = np.linalg.eig(XXT) #XXTvalues(固有値),XXTvectors(固有値ベクトル)

#特異値の計算
values = np.sqrt(XXTvalues)

#特異値を大きい順に並べ替える(もとの特徴を表す大きさ順)
index = np.argsort(values)[::-1]

#対角化
Ds = np.diag(values[index])

#右特異行列
Vs = XXTvectors[:,index]

#左特異行列
Us = np.array([np.dot(X, Vs[:,i]) / Ds.diagonal()[i] for i in range(len(Ds.diagonal()))])

print("Numpy:SVD(スクラッチ)")
print(X)#分散共分散行列
print(Us)#固有ベクトル
print(Vs.T)#固有ベクトル
print(Ds)#固有値(特異値)

#特異値分解SVD(固有値分解) -----------------------------------------------------------------------
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
U, D, V = np.linalg.svd(X)

print("Numpy:SVD")
print(X)#分散共分散行列
print(U)#固有ベクトル
print(V.T)#固有ベクトル(対称行列のためUと同値)
print(D)#固有値(特異値)


#sklearnによるPCA分析-----------------------------------------------------------------------------
pca = PCA()
pca.fit(data)

print("Sklearn:PCA")
print(pca.get_covariance()) # 分散共分散行列
print(pca.components_) # 固有ベクトル
print(pca.explained_variance_) # 固有値

