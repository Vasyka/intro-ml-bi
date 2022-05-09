{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hTHQzSMuDG-5"
      },
      "source": [
        "## Введение в машинное обучение\n",
        "\n",
        "## НИУ ВШЭ\n",
        "\n",
        "### Домашнее задание №2\n",
        "\n",
        "### О задании\n",
        "\n",
        "В этом домашнем задании вы реализуете алгоритм kNN и линейную регрессию, попрактикуетесь в решении задачи регрессии, а также решите теоретические задачи.\n",
        "\n",
        "### Оценивание и штрафы\n",
        "\n",
        "Оценка за ДЗ вычисляется по следующей формуле:\n",
        "\n",
        "$$\n",
        "\\text{points} \\times 10 / 18,\n",
        "$$\n",
        "\n",
        "где points — количество баллов за обязательную часть, которое вы набрали.\n",
        "\n",
        "__Внимание!__ Домашнее задание выполняется самостоятельно. «Похожие» решения считаются плагиатом и все задействованные студенты (в том числе те, у кого списали) не могут получить за него больше 0 баллов."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JohzNA6oDG-_"
      },
      "source": [
        "# kNN своими руками (5)\n",
        "\n",
        "В этом задании вам предстоит реализовать взвешенный алгоритм kNN для регрессии. Пусть необходимо вычислить значение $y$ для некоторого $x$ при известных данных $\\left(x_1, y_1\\right), \\ldots, \\left(x_\\ell, y_\\ell\\right)$. Предсказанием вашего регрессора будет являться\n",
        "\n",
        "$$\n",
        "\\hat{y} = \\frac{\\sum\\limits_{i=1}^kw_iy_{(i)}}{\\sum\\limits_{i=1}^kw_i},\n",
        "$$\n",
        "где $\\left(x_{(1)}, y_{(1)}\\right), \\ldots, \\left(x_{(k)}, y_{(k)}\\right)$ - ближайшие $k$ объектов к $x$ по некоторой метрике $d(\\cdot, \\cdot)$. Ваш алгоритм должен уметь работать с двумя метриками:\n",
        "\n",
        "$$\n",
        "d\\left(x_{(i)}, x\\right) = \\|x_{(i)} - x\\|_2 = \\sqrt{\\sum\\limits_{j=1}^n\\left(x_{(i)}^j - x^j\\right)^2}\\qquad\\text{(евклидова)}\n",
        "$$\n",
        "$$\n",
        "d\\left(x_{(i)}, x\\right) = \\|x_{(i)} - x\\|_1 = \\sum\\limits_{j=1}^n\\left|x_{(i)}^j - x^j\\right|\\qquad\\text{(манхэттена)}\n",
        "$$"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4Ia3TRLJDG-_"
      },
      "source": [
        "### Реализуйте две функции расстояния (1 балл)\n",
        "- евклидова метрика **(0.5 балла)**\n",
        "- метрика Манхэттена **(0.5 балла)**\n",
        "\n",
        "Обе функции должны на вход получать матрицу `np.array of shape(n, m)` и вектор `np.array of shape(m,)`, а возвращать вектор расстояний от каждой строчки матрицы до вектора `np.array of shape(n,)`\n",
        "\n",
        "**В данном пункте запрещено использование циклов for, while. Пользуйтесь возможностями numpy.** "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7UnE-lOKDG_A"
      },
      "outputs": [],
      "source": [
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FWmjupGvDG_B"
      },
      "outputs": [],
      "source": [
        "def euclidian_metric(X, x):\n",
        "    distances = # your code here\n",
        "    return distances\n",
        "\n",
        "def manhattan_metric(X, x):\n",
        "    distances = # your code here\n",
        "    return distances"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hgzA0cENDG_C"
      },
      "outputs": [],
      "source": [
        "# проверка\n",
        "X = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])\n",
        "y = np.ones(3)\n",
        "\n",
        "assert np.allclose(euclidian_metric(X, y), np.array([ 2.23606798,  8.77496439, 13.92838828]))\n",
        "assert np.allclose(manhattan_metric(X, y), np.array([ 3., 15., 24.]))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IX3mu-DKDG_D"
      },
      "source": [
        "### Реализуйте алгоритм kNN для регрессии (4 балла)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hUEOfsaxDG_D"
      },
      "source": [
        "- реализуйте класс kNN для равномерных весов (то есть $w_1 = \\ldots = w_k$) **(3 балла)**\n",
        "- добавьте возможность передать данному классу параметр `weights='distance'` для вызова взвешенной версии алгоритма kNN (то есть $w_i = \\frac{1}{d\\left(x, x_{(i)}\\right)}$ **(1 балл)**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sTDJG1roDG_E"
      },
      "outputs": [],
      "source": [
        "class KNN:\n",
        "    def __init__(self, metric='euclid', k=5, weights='uniform'):\n",
        "        \"\"\"\n",
        "        PARAMETERS:\n",
        "        metric ('euclid' or 'manhattan')\n",
        "        k - number of nearest neighbors\n",
        "        \"\"\"\n",
        "\n",
        "        self.metric = metric\n",
        "        self.k = k\n",
        "        self.weights = weights\n",
        "        \n",
        "        self.X_train = None\n",
        "        self.y_train = None\n",
        "        \n",
        "    def fit(self, X_train, y_train):\n",
        "        \"\"\"\n",
        "        INPUT:\n",
        "        X_train - np.array of shape (n, d)\n",
        "        y_train - np.array of shape (n,)\n",
        "        \"\"\"\n",
        "\n",
        "        # your code here\n",
        "        \n",
        "    def predict(self, X_test):\n",
        "        \"\"\"\n",
        "        INPUT:\n",
        "        X_test - np.array of shape (m, d)\n",
        "        \n",
        "        OUTPUT:\n",
        "        y_pred - np.array of shape (m,)\n",
        "        \"\"\"\n",
        "        \n",
        "        # your code here\n",
        "\n",
        "        return y_pred"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UX3o2DEyDG_E"
      },
      "source": [
        "Сверьте для нескольких комбинаций различных гиперпараметров свой результат на искусственной выборке с результатом соответствующего алгоритма из `sklearn`. **Не забудьте про гиперпараметр `weights`.**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a7qD29LUDG_F"
      },
      "outputs": [],
      "source": [
        "np.random.seed(13)\n",
        "X_train = np.random.randn(1000, 50)\n",
        "y_train = np.random.randn(1000,)\n",
        "X_test = np.random.randn(500, 50)\n",
        "y_test = np.random.randn(500,)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v68bKe7mDG_F"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c-69pp1lDG_G"
      },
      "source": [
        "# Линейная регрессия своими руками (5)\n",
        "\n",
        "Реализуйте линейную регрессию с градиентным спуском для [функции потерь Хьюбера](https://en.wikipedia.org/wiki/Huber_loss):\n",
        "\n",
        "$$\n",
        "L_\\delta\\left(y, \\hat{y}\\right) =\n",
        "\\begin{cases}\n",
        "\\frac{1}{2}\\left(y - \\hat{y}\\right)^2, \\qquad &|y - \\hat{y}| \\leq \\delta\\\\\n",
        "\\delta\\left|y - \\hat{y}\\right| - \\frac{1}{2}\\delta^2,\\qquad & \\text{otherwise}\n",
        "\\end{cases}\n",
        "$$\n",
        "\n",
        "В таком случае общее значение функции потерь на всем датасете $(x_1, y_1), \\ldots, (x_\\ell, y_\\ell)$ будет равно\n",
        "\n",
        "$$\n",
        "L = \\frac{1}{\\ell}\\sum\\limits_{i=1}^\\ell L_\\delta\\left(y_i, \\hat{y}_i\\right)\n",
        "$$\n",
        "\n",
        "*Чему будет равен градиент этой функции по $w$? Вспомните, что за вектор $\\hat{y}$ и как он зависит от $X$ и $w$.*\n",
        "\n",
        "Эти ссылки могут показаться вам полезными:\n",
        "- https://github.com/esokolov/ml-course-hse/blob/master/2019-fall/lecture-notes/lecture02-linregr.pdf\n",
        "- https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931\n",
        "- https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7cs0Q-PLDG_G"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qQrSYtfzDG_H"
      },
      "source": [
        "### Реализуйте функцию потерь Хьюбера для одного примера и ее градиент по весам (1.5 балла)\n",
        "\n",
        "- функция потерь **(0.5 балла)**\n",
        "- градиент **(1 балл)**\n",
        "\n",
        "**В данном пункте запрещено использование циклов for, while. Пользуйтесь возможностями numpy. Однако можно использовать оператор if.**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z7EXTgyzDG_H"
      },
      "outputs": [],
      "source": [
        "def huber_loss(x, y, w, delta):\n",
        "    \"\"\"\n",
        "    x - np.array shape=(d,)\n",
        "    y - scalar\n",
        "    w - np.array shape=(d,)\n",
        "    delta - scalar\n",
        "    \n",
        "    OUTPUT:\n",
        "    loss - scalar\n",
        "    \"\"\"\n",
        "\n",
        "    pass\n",
        "\n",
        "def huber_grad(x, y, w, delta):\n",
        "    \"\"\"\n",
        "    INPUT:\n",
        "    x - np.array shape=(d,)\n",
        "    y - scalar\n",
        "    w - np.array shape=(d,)\n",
        "    delta - scalar\n",
        "    \n",
        "    OUTPUT:\n",
        "    grad - np.array shape=(d,)\n",
        "    \"\"\"\n",
        "\n",
        "    pass"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "adB2WWTNDG_H"
      },
      "outputs": [],
      "source": [
        "# проверка\n",
        "\n",
        "x = np.array([1, 2, 3])\n",
        "w = np.array([3, 5, 12])\n",
        "y = 19\n",
        "delta = 1\n",
        "\n",
        "assert huber_loss(x, y, w, delta) == 29.5\n",
        "assert np.allclose(huber_grad(x, y, w, delta), np.array([1, 2, 3]))\n",
        "\n",
        "y = 49.2\n",
        "\n",
        "assert np.allclose(huber_loss(x, y, w, delta), 0.02000000000000057)\n",
        "assert np.allclose(huber_grad(x, y, w, delta), np.array([-0.2, -0.4, -0.6]))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OttG0yODDG_I"
      },
      "source": [
        "### Реализуйте линейную регрессию (3.5 балла)\n",
        "\n",
        "*Вы можете опустить единичный признак в модели и не добавлять его в данные. Для данной искусственной выборки это не актуально, потому что целевая переменная в этом случае является случайной величиной из стандартного нормального распределения со средним 0.*\n",
        "\n",
        "*Вектор весов в градиентном спуске можете инициализировать нулями.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3MI7pcufDG_I"
      },
      "outputs": [],
      "source": [
        "class LinearRegressionHuber:\n",
        "    def __init__(self, delta=1.0, max_iter=1000, tol=1e-6, eta=1e-2):\n",
        "        \"\"\"\n",
        "        PARAMETERS:\n",
        "        delta - scalar in Huber loss\n",
        "        max_iter - maximum possible number of iterations in Gradient Descent\n",
        "        tol - precision for stopping criterion in Gradient Descent\n",
        "        eta - step size in Gradient Descent (learning rate)\n",
        "        \"\"\"\n",
        "\n",
        "        self.delta = delta\n",
        "        self.max_iter = max_iter\n",
        "        self.tol = tol\n",
        "        self.eta = eta\n",
        "        \n",
        "        self.w = None\n",
        "        self.loss_history = None\n",
        "        \n",
        "    def fit(self, X, y):\n",
        "        \"\"\"\n",
        "        INPUT:\n",
        "        X_train - np.array of shape (n, d)\n",
        "        y_train - np.array of shape (n,)\n",
        "        \n",
        "        В этой функции вы должны инициализировать веса (можно нулями), а также \n",
        "        итерационно обновлять веса с помощью \n",
        "        градиентного спуска (считать и запоминать лосс (значение функции потерь) будет хорошим решением)\n",
        "        \"\"\"\n",
        "\n",
        "        self.w = # your code here\n",
        "        self.loss_history = # your code here\n",
        "        \n",
        "        # your code here\n",
        "        \n",
        "        return self.loss_history\n",
        "        \n",
        "    def predict(self, X):\n",
        "        \"\"\"\n",
        "        INPUT:\n",
        "        X_test - np.array of shape (m, d)\n",
        "        \n",
        "        OUTPUT:\n",
        "        y_pred - np.array of shape (m,)\n",
        "        \n",
        "        Предскажите ответы с помощью обученных весов\n",
        "        \"\"\"\n",
        "        \n",
        "        y_pred = # your code here\n",
        "\n",
        "        return y_pred\n",
        "    \n",
        "    def calc_gradient(self, X, y):\n",
        "        \"\"\"\n",
        "        Calculates the gradient of Huber loss by weights.\n",
        "        \n",
        "        INPUT:\n",
        "        X - np.array of shape (l, d)\n",
        "        y - np.array of shape (l,)\n",
        "        \n",
        "        OUTPUT:\n",
        "        grad - np.array of shape (d,)\n",
        "        \n",
        "        Посчитайте градиент как среднее от градиентов для каждого примера\n",
        "        \"\"\"\n",
        "\n",
        "        grad = np.zeros_like(self.w)\n",
        "\n",
        "        # your code here\n",
        "        \n",
        "        return grad \n",
        "    \n",
        "    def calc_loss(self, X, y):\n",
        "        \"\"\"\n",
        "        Calculates the Huber loss.\n",
        "        \n",
        "        INPUT:\n",
        "        X - np.array of shape (l, d)\n",
        "        y - np.array of shape (l,)\n",
        "        \n",
        "        OUTPUT:\n",
        "        loss - float\n",
        "        \n",
        "        Посчитайте loss по выборке как среднее loss'ов для каждого \n",
        "        примера\n",
        "        \"\"\"\n",
        "\n",
        "        loss = 0\n",
        "        \n",
        "        # your code here\n",
        "        \n",
        "        return loss"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wz14GnyJDG_I"
      },
      "source": [
        "Проверьте работу вашего метода: выведите результаты его работы на той же искусственной выборке, что и в задаче выше (в качестве метрик качества используйте MSE и Huber loss). Постройте график зависимости значения функции потерь от итерации градиентного спуска."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FFm5AauVDG_J"
      },
      "outputs": [],
      "source": [
        "lrh = LinearRegressionHuber()\n",
        "loss_history = lrh.fit(X_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-tbLjCI-DG_J"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6lPqyS7NDG_J",
        "outputId": "6a96c7d2-f8c8-465e-c2b4-9ce113091790"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x432 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          },
          "output_type": "display_data"
        }
      ],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8KXSg6A-DG_K"
      },
      "source": [
        "# Практика (8)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WJ6DkjasDG_K"
      },
      "source": [
        "Пожалуйста, при использовании различных функций из библиотек импортируйте все, что вам понадобилось в данной части, в следующем блоке:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5lHb5koUDG_K"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression\n",
        "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures\n",
        "\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BB8RuXCXDG_L"
      },
      "source": [
        "В этой части вы поработаете с уже знакомыми вам данными из другого соревнования на Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques. Задача - предсказание цены дома."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "refgpqUYDG_L"
      },
      "outputs": [],
      "source": [
        "data = pd.read_csv('train.csv', header=0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LtSjKGIyDG_L"
      },
      "outputs": [],
      "source": [
        "data.columns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5mZi3_P0DG_L"
      },
      "outputs": [],
      "source": [
        "data.head()"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 0 (0.5)"
      ],
      "metadata": {
        "id": "DNNFsvpYFq9c"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Если в данных бессмысленные столбцы? Если да, избавьтесь от них и обоснуйте свое решение."
      ],
      "metadata": {
        "id": "OKB4M1JsFzAb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# your code here"
      ],
      "metadata": {
        "id": "mPuBOhR0FrMg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mMxQRnSCDG_L"
      },
      "source": [
        "## 1 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yfDYSKnNDG_M"
      },
      "source": [
        "Есть ли в данных пропуски? Если да, то для каждого столбца, в котором они имеются, посчитайте их количество и их долю от общего числа значений. Что вы наблюдаете?"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0iS3omMQDG_M"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4_hG_S69DG_M"
      },
      "source": [
        "## 2 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EU8kJsYSDG_M"
      },
      "source": [
        "Избавьтесь от пропусков. Для каждого из примененных методов обоснуйте свое решение. **Проверьте, что вы действительно избавились от пропусков.**\n",
        "\n",
        "*Напоминание. В зависимости от типа столбца, можно заполнить пропуски, например, средним арифметическим, медианой, модой, можно какими-то отдельными значениями. А можно такие столбцы вообще удалить.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VmhsxFsfDG_M"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rbcGD6nsDG_M"
      },
      "source": [
        "## 3 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YXxOCPuMDG_N"
      },
      "source": [
        "Обработайте категориальные признаки. В их обнаружении вам может помочь синтаксис `pandas` (например, можно обратить внимание на типы столбцов), а также описание датасета и его исследование. Объясните выбор метода (one-hot-encoding, label encoding, ...)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "98ZcVdWcDG_N"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-pyxvSd-DG_N"
      },
      "source": [
        "## 4 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mQXVPONNDG_N"
      },
      "source": [
        "Вычислите и визуализируйте попарную корреляцию Пирсона между всеми признаками. Какие выводы можно сделать?\n",
        "\n",
        "*Для визуализации можно использовать `seaborn.heatmap()`.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yunoRI6MDG_N"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sFa5z7P4DG_N"
      },
      "source": [
        "## 5 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XMMRm6UaDG_N"
      },
      "source": [
        "Найдите признаки с максимальным и минимальным **абсолютным** значением коэффициента корреляции Пирсона с предсказываемым значением. Изобразите на графиках зависимость найденных признаков от предсказываемого значения.\n",
        "\n",
        "*Не забудьте указать название графика и обозначить, что изображено по каждой из осей.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GOvHhKs5DG_O"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4kU1cZBtDG_O"
      },
      "source": [
        "## 6 (1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yiOziXEADG_O"
      },
      "source": [
        "Постройте гистограмму распределения предсказываемого значения. Для избавления от разницы в масштабах, а также \"смещения\" распределения переменной в сторону нормального (что бывает полезно при статистическом анализе), можно прологарифмировать ее (это обратимое преобразование, поэтому целевую переменную легко восстановить). В данном случае воспользуйтесь `numpy.log1p`, чтобы сделать преобразование $y \\to \\ln\\left(1 + y\\right)$. Постройте гистограмму распределения от нового предсказываемого значения. Опишите наблюдения.\n",
        "\n",
        "*В дальнейшем используйте в качестве предсказываемого значения вектор, который получился после логарифмирования.*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TJJNkRYCDG_O"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UsSmDjaaDG_O"
      },
      "source": [
        "Перейдем непосредственно к построению моделей. Разобьем выборку на обучение и контроль. Причем 75% объектов оставьте на обучение и 25% - на тестовую выборку.\n",
        "\n",
        "Зафиксируйте при разбиении значение `random_state` = 13. Это потребуется для выполнения последующих заданий"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vIto5eLKDG_P"
      },
      "outputs": [],
      "source": [
        "X_train, X_val, y_train, y_val = # your code here"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LW3G1AvXDG_P"
      },
      "outputs": [],
      "source": [
        "X_train.shape, X_val.shape, y_train.shape, y_val.shape"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NfqyUynFDG_P"
      },
      "source": [
        "## 7 (1.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xu02a0hdDG_P"
      },
      "source": [
        "Примените к данным следующие алгоритмы:\n",
        "\n",
        "- kNN\n",
        "- линейная регрессия\n",
        "- Lasso\n",
        "- Ridge\n",
        "\n",
        "Для каждого из методов подберите гиперпараметры с помощью кросс-валидации. Обучите алгоритмы с лучшими гиперпараметрами на обучающей выборке и оцените качество по метрикам \n",
        "- RMSE \n",
        "- MAE\n",
        "- $R^2$\n",
        "\n",
        "Интерпретируйте полученные результаты."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6_DrcSLlDG_P"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K6ZErd7zDG_P"
      },
      "source": [
        "## 8 (1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NELg3QQvDG_P"
      },
      "source": [
        "Постройте гистограммы значений весов для линейной регрессии, Lasso и Ridge. Опишите наблюдения. В чем различия между полученными наборами весов и почему?"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PdgE9vMBDG_Q"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j2N2ZQHfDG_Q"
      },
      "source": [
        "## 9 (0.5)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7Jk2JdR_DG_Q"
      },
      "source": [
        "Добейтесь того, чтобы в заданиях выше ваш лучший алгоритм давал качество не больше 0.212 на валидации по метрике RMSE (если вы дошли до этого задания, а качество выше уже удовлетворяет этому условию, вы автоматически получите за него полный балл)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nDnfQmvtDG_Q"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Et-ova3KDG_Q"
      },
      "source": [
        "## 10* (1)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cOQ_ZWnWDG_Q"
      },
      "source": [
        "Добейтесь того, чтобы в заданиях выше ваш лучший алгоритм давал качество не больше 0.210 на валидации по метрике RMSE. Для этого вы можете использовать самые разные методы, какие захотите - отбор признаков, генерация новых, разные способы предобработки данных. Единственное ограничение - не использовать никакие алгоритмы регрессии, кроме kNN, линейной регрессии, Lasso и Ridge."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f-2Mird3DG_Q"
      },
      "outputs": [],
      "source": [
        "# your code here"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.6.3"
    },
    "colab": {
      "name": "hw2.ipynb",
      "provenance": [],
      "collapsed_sections": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}