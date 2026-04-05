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