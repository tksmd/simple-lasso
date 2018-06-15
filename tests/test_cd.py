import unittest

import numpy as np

from lasso.cd import coordinate_descent

# 特徴量の次元
n_features = 1000
# サンプル数
n_samples = 100
# 非ゼロの特徴量の数
n_nonzero_coefs = 20
# イテレーションの回数
n_iter = 1000
# 正則化パラメータ
alpha = 0.6


class TestCoordinateDescent(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_coordinate_descent(self):
        # 真の重みの生成
        idx = np.random.randint(0, n_features, n_nonzero_coefs)
        w = np.zeros(n_features)
        w[idx] = np.random.normal(0.0, 1.0, n_nonzero_coefs)

        # 入力データと観測情報
        X = np.random.normal(0.0, 1.0, (n_samples, n_features))
        y = X.dot(w) + np.random.normal(0.0, 1.0, n_samples)

        # 座標効果法の実行
        w_pred = coordinate_descent(X, y, alpha, n_iter)

        # 非ゼロ要素の数
        print(f'Number of nonzero coefficients (true) : {np.sum(w != 0)}')
        print(f'Number of nonzero coefficients (predicted) : {np.sum(w_pred != 0)}')

        # 推定値と真の値のユークリッド距離
        print(f'Euclidean distance between coefficients : {np.linalg.norm(w - w_pred):#.2f}')
        print(f'Euclidean distance between estimated output : {np.linalg.norm(y - X.dot(w_pred)):#.2f}')


if __name__ == '__main__':
    unittest.main()
