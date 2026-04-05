```shell
 pip install matplotlib  
 pip install scipy
 pip install numpy    
```

https://kuboweb.github.io/-kubo/ce/IwanamiBook.html
のglmmフォルダのdata.csvを見る

### 7.4 一般化線形混合モデルの最尤推定
#### 二項分布 と N(0,1)の混合
Binomial GLMM（N=8, logitリンク, ランダム効果 𝑟 ∼ 𝑁(0 , 3 ^2))）

線形予測子は

\eta = r

確率変換は

q = \frac{1}{1+e^{-r}}

条件付き分布は

Y\mid r \sim \mathrm{Binomial}(8,q)

ランダム効果は

r \sim N(0,3^2)

[
P(Y=y)=\int \binom{8}{y}q(r)^y(1-q(r))^{8-y}\phi(r),dr
]
をモンテカルロ近似して求まる。
![p157.png](p157.png)

#### Poisson と N(0,1)の混合
![p178.png](p158.png)

![glmm_vs_true.png](glmm_vs_true.png)
```markdown
推定値  β̂₀ = -3.4046  (真値 -4.0)
推定値  β̂₁ = 0.8268  (真値 1.0)
切片の誤差: +0.5954
傾きの誤差: -0.1732
```
#### 
pymer4はRのlme4をPythonから呼ぶラッパー を使わないときれいに書けないかも。

![glmm_mixed_binom_x4.png](glmm_mixed_binom_x4.png)

### Rのインストール
```shell
brew install cmake
```
```shell
brew install r
R -e "install.packages('nloptr', repos='https://cloud.r-project.org', type='source')"
```
上が通ったら lme4 → lmerTest の順に
```shell
R --version
pip install pymer4
R -e "install.packages('lme4', repos='https://cloud.r-project.org')"
R -e "install.packages('lmerTest', repos='https://cloud.r-project.org')"
```

```shell
pip install pymer4
pip install polars
pip install rpy2
pip install joblib
```

```shell
sudo mkdir -p /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib
```

```shell
sudo ln -s /opt/homebrew/Cellar/r/4.5.3/lib/R/lib/libR.dylib /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libR.dylib
sudo ln -s /opt/homebrew/Cellar/r/4.5.3/lib/R/lib/libRlapack.dylib /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib
```
```shell
sudo ln -s /opt/homebrew/Cellar/r/4.5.3/lib/R/lib/libRlapack.dylib /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRblas.dylib
```
```markdown
# 旧バージョンの書き方
# from pymer4.models import Lmer
```
```shell
python -c "from pymer4.models import lmer; print(dir(lmer))"
```
でクラス名確認する

Rでやると、
β₁ = -4.1893
β₂ = 1.0047
s  = 2.4080

````shell
R -e "install.packages('glmmML', repos='https://cloud.r-project.org')"
````

---> https://kuboweb.github.io/-kubo/ce/IwanamiBook.html#ch07
>大坪さん (創価大) から p.159 などに掲載されている推定結果についてのご指摘をいただきました (2012-12-25):
掲載されている glmmML() の推定結果は method="ghq" を指定した場合に相当
これは対数尤度の積分に Gauss-Helmite 法を使っている
何も指定しないで glmmML() を使用した場合には，推定値などが少し異なる結果が得られる
default では glmmML() は Laplace 近似法 ({#code method="Laplace"}) を使うため

から`method="ghq"`をつけることでうまく教科書と一致する。
β₁ = -4.1296
β₂ = 0.9903
s  = 2.4939


```shell
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOSのヒラギノフォントを使う
plt.rcParams['font.family'] = 'Hiragino Sans'
```