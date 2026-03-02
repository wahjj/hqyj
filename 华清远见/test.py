# test_simple.py
print("程序开始")

try:
    import numpy as np

    print("numpy 正常")

    from sklearn.datasets import load_iris

    print("sklearn 正常")

    X, y = load_iris(return_X_y=True)
    print("数据加载正常")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("数据划分正常")

    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("模型训练正常")

    score = model.score(X_test, y_test)
    print(f"准确率: {score}")

    print("所有测试通过!")

except Exception as e:
    print(f"错误: {e}")