import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns


class CarData:

    def data_istatistic(self):
        df = pd.read_csv("cars_dataset.csv")

        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        sns.set_theme(style="darkgrid")
        ax = sns.countplot(x="Make", data=df, ax=axs[0, 0],
                           order=df['Make'].value_counts().index)
        ax = sns.countplot(x="Make", data=df, hue='fuelType', ax=axs[0, 1],
                           order=df['Make'].value_counts().index)
        ax = sns.countplot(x="transmission", data=df, hue='fuelType', ax=axs[1, 0],
                           order=df['transmission'].value_counts().index)
        ax = sns.countplot(x="transmission", data=df, hue='Make', ax=axs[1, 1])
        ax2 = sns.countplot(x="year", data=df[df['year'] >= 2010], hue='Make', ax=axs[2, 0])
        ax3 = sns.countplot(x="year", data=df[df['year'] >= 2010], ax=axs[2, 1])
        axlab = ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
        axlab2 = ax3.set_xticklabels(ax3.get_xticklabels(), rotation=40, ha="right")

        numeric_vars = ['price', 'mileage', 'tax', 'mpg', 'engineSize']
        fig, axs = plt.subplots(5, figsize=(10, 20))
        sns.set_theme(style="darkgrid")
        for index, cols in enumerate(numeric_vars):
            sns.kdeplot(data=df, x=cols, ax=axs[index], fill=True)

        plt.show()

    def set_data(self):
        df = pd.read_csv("cars_dataset.csv")

        final_dataset = df[['model', 'year', 'price', 'transmission', 'mileage', 'mpg', 'fuelType',
                            'engineSize', 'Make']]

        final_dataset["currentYear"] = 2021
        final_dataset["age"] = final_dataset["currentYear"] - final_dataset["year"]

        final_dataset.drop(["year"], axis=1, inplace=True)
        final_dataset.drop(["currentYear"], axis=1, inplace=True)
        # final_dataset = pd.get_dummies(final_dataset, drop_first=True)

        return final_dataset

    def price_prediction(self, final_dataset):
        final_dataset = pd.get_dummies(final_dataset, drop_first=True)
        x = final_dataset.drop(['price'], axis=1)
        y = final_dataset.iloc[:, 0].values

        return x, y

    def linear_regression(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        print("Tahmin Basari Orani: ", r2_score(y_test, predictions))
        return y_test, predictions

    def model_prediction(self, final_dataset, make):
        print("\nMarka: ", make)

        final_dataset.drop(final_dataset.index[(final_dataset["Make"] != make)], axis=0, inplace=True)
        final_dataset.drop(["Make"], axis=1, inplace=True)

        label_e = LabelEncoder()
        final_dataset["model"] = label_e.fit_transform(final_dataset["model"])
        final_dataset["transmission"] = label_e.fit_transform(final_dataset["transmission"])
        final_dataset["fuelType"] = label_e.fit_transform(final_dataset["fuelType"])

        x = final_dataset.drop(['model'], axis=1)
        y = final_dataset.iloc[:, 0].values

        return x, y

    def logistic_regression(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model = LogisticRegression(solver="liblinear")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        print("Tahmin Basari Orani: ", r2_score(y_test, predictions))
        return y_test, predictions

    def PCA_LinearRegression(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("Tahmin Basari Orani: ", r2_score(y_test, predictions))
        return y_test, predictions

    def PCA_LogisticRegression(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        model = LogisticRegression(solver="liblinear")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("Tahmin Basari Orani: ", r2_score(y_test, predictions))
        return y_test, predictions

    def plot_result(self, y_test, predictions):
        sns.regplot(x=y_test, y=predictions, ci=None, scatter_kws={"color": "r", "s": 9})
        plt.show()


if __name__ == '__main__':
    pred = CarData()

    # veri istatistikleri gormek icin calistiriniz
    pred.data_istatistic()

    # verinin düzenlenmesi icin calistiriniz
    final_dataset = pred.set_data()

    # Linear regression
    x, y = pred.price_prediction(final_dataset)
    print("Linear Regression:")
    y_test, predictions = pred.linear_regression(x, y)
    pred.plot_result(y_test, predictions)

    # PCA uygulanmis veride Linear regression
    print("\nPCA uygulanmis veride Linear regression:")
    y_test, predictions = pred.PCA_LinearRegression(x, y)
    pred.plot_result(y_test, predictions)

    # Logistic regression
    for i in final_dataset["Make"].unique():
        final_dataset = pred.set_data()
        x, y = pred.model_prediction(final_dataset, i)
        print("Logistic Regression:")
        y_test, predictions = pred.logistic_regression(x, y)
        pred.plot_result(y_test, predictions)
    # PCA uygulanmis veride Linear regression
        print("\nPCA uygulanmis veride Logistic regression:")
        y_test, predictions = pred.PCA_LogisticRegression(x, y)
        pred.plot_result(y_test, predictions)
