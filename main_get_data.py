import lifelines.datasets
import yfinance as yf
import sklearn.datasets
import pandas as pd

class Data():

    def get_kidney_transplant(self, file: str="kidney_transplant.csv"):
        df =  lifelines.datasets.load_kidney_transplant()
        df.to_csv(
            file,
            header=True,
            sep=",",
            index=False
        )
    
    def get_stock_prices(self, file: str="stock_prices.csv"):
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="max")
        df = df.reset_index(drop=False)
        df.to_csv(
            file,
            header=True,
            sep=",",
            index=False
        )
    
    def get_digits(self, file: str="digits.csv"):
        X, y = sklearn.datasets.load_digits(return_X_y=True, as_frame=True)
        df = pd.concat([y, X], axis=1)
        df = df.reset_index(drop=False)
        df.to_csv(
            file,
            header=True,
            sep=",",
            index=False
        )

if __name__ == "__main__":
    data = Data()
    data.get_kidney_transplant()
    data.get_stock_prices()
    data.get_digits()